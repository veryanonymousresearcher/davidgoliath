import torch.nn as nn
import torch

from transformers import AutoModel
from peft import get_peft_model, LoraConfig

from ppm.models.common import InLayer, OutLayer, TimePositionalEncoding
from ppm.models.config import FreezeConfig

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import Module

from contextlib import nullcontext
import torch.utils.checkpoint as cp
from typing import Optional

import inspect


import os

HF_TOKEN = os.getenv("HF_TOKEN")


class NextEventPredictor(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        categorical_cols: list[str],
        categorical_sizes: dict[str, int],
        numerical_cols: list[str],
        categorical_targets: list[str],
        numerical_targets: list[str],
        padding_idx: int,
        strategy: str,
        backbone_name: str,
        backbone_pretrained: bool,
        backbone_finetuning: LoraConfig | FreezeConfig | None,
        backbone_type: str,
        backbone_hidden_size: int,
        backbone_n_layers: int,
        #max_seq_len: int, 
        time_positional_encoding: str,
        device: str,
        student_model: Module,
        use_weight_tying: bool = True,
        use_checkpoint: bool = False
    ):
        super(NextEventPredictor, self).__init__()

        self.categorical_cols = categorical_cols
        self.categorical_sizes = categorical_sizes
        print("categorical_sizes:", self.categorical_sizes)
        self.numerical_cols = numerical_cols
        self.categorical_targets = categorical_targets
        self.numerical_targets = numerical_targets

        self.embedding_size = embedding_size
        self.strategy = strategy

        self.backbone_name = backbone_name
        self.backbone_pretrained = backbone_pretrained
        self.backbone_finetuning = backbone_finetuning
        self.backbone_type = backbone_type
        self.backbone_hidden_size = backbone_hidden_size
        self.backbone_n_layers = backbone_n_layers

        self.padding_idx = padding_idx
        #self.max_seq_len = max_seq_len
        self.time_positional_encoding = time_positional_encoding
        self.device = device
        
        self.use_weight_tying = use_weight_tying
        self.use_checkpoint = use_checkpoint
        


        # define input layer
        self.in_layer = InLayer(
            # output size
            embedding_size=embedding_size,
            # input sizes
            categorical_cols=categorical_cols,
            categorical_sizes=categorical_sizes,
            numerical_cols=numerical_cols,
            # other params
            padding_idx=padding_idx,
            strategy=strategy,
        )

        # define time_positional embeddings for transformers
        if self.time_positional_encoding == "additive":
            self.positional_embeddings = TimePositionalEncoding(
                d_model=embedding_size)


        # define backbone
        if backbone_pretrained:
            # AutoModel: no built-in classification/LM heads
            self.backbone = AutoModel.from_pretrained(backbone_name, token=HF_TOKEN)
            if isinstance(backbone_finetuning, LoraConfig):
                self.backbone = get_peft_model(self.backbone, backbone_finetuning)
            elif isinstance(backbone_finetuning, FreezeConfig):
                # self._freeze_params(backbone_finetuning)
                backbone_finetuning.apply(self.backbone)
            elif backbone_finetuning is None:
                # No adapter/wrapping and no freezing: train the full backbone.
                pass
            else:
                raise NotImplementedError("Fine-tuning not implemented yet.")
        else:
            if backbone_name == "rnn":
                if backbone_type == "lstm":
                    self.backbone = nn.LSTM
                elif backbone_type == "gru":
                    self.backbone = nn.GRU
                elif backbone_type == "rnn":
                    self.backbone = nn.RNN
                else:
                    raise ValueError("Invalid RNN type.")
                self.backbone = self.backbone(
                    input_size=embedding_size,
                    hidden_size=backbone_hidden_size,
                    num_layers=backbone_n_layers,
                    batch_first=True,
                )
            elif backbone_name.endswith("gpt2"):
                self.backbone = AutoModel.from_pretrained(
                    "openai-community/gpt2",
                )
                self.backbone.apply(self.backbone._init_weights)
                
            elif backbone_name == "student_model":
                self.backbone = student_model
                self.backbone_hidden_size = student_model.config.n_embd

            
        if hasattr(self.backbone, "gradient_checkpointing_enable"): # HF models may have own memory behavior, impeding our checkpointing (to reduce memory needs by not storing intermediate activations for untrainable layers)
            if self.use_checkpoint:
                self.backbone.gradient_checkpointing_enable()
            else:
                # make sure it's OFF if the model supports disabling
                if hasattr(self.backbone, "gradient_checkpointing_disable"):
                    self.backbone.gradient_checkpointing_disable()
            
        if hasattr(self.backbone, "config"):  # some HF models needs this to avoid conflict with checkpointing
            self.backbone.config.use_cache = False


        # custom time positional encoding
        if backbone_name.endswith("gpt2") and self.time_positional_encoding == "additive":
            # switch off GPT2 positional embeddings
            self.backbone.wpe.weight.data.zero_()
            self.backbone.wpe.weight.requires_grad = False

        # define output layer(s)
        self.out_layers = nn.ModuleDict()
        for target in categorical_targets:
            self.out_layers[target] = OutLayer(
                input_size=backbone_hidden_size,
                output_size=categorical_sizes[target.replace("next_", "")],
            )
        for target in numerical_targets:
            self.out_layers[target] = OutLayer(
                input_size=backbone_hidden_size,
                output_size=1,
            )

        self._apply_weight_tying()

    def forward(self, x_cat, x_num=None, attention_mask=None, h=None):
        x = self.in_layer(x_cat, x_num)
        #position_ids = None
        
        if self.time_positional_encoding == "additive":
            t_idx = self.numerical_cols.index("accumulated_time")
            t = x_num[..., t_idx].to(x.dtype)
            x = self.positional_embeddings(x, t)
            #print("we are using custom additive time positional encoding")
            #seq_len = x.size(1)
            #position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0)

        if self.backbone_name == "rnn":
            # Guard against fully-masked samples (length 0), which are invalid for packing.
            lengths = attention_mask.sum(dim=-1).clamp_min(1).long().tolist()
            if not isinstance(lengths, list):
                lengths = [lengths]
            x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            x = self.backbone(x, h)
        else:
            # Using inputs_embeds to skip backbone's own embedding layer.
            # Run backbone with activation checkpointing for memory savings.
            x = self._run_backbone(x, attention_mask)

            
        if self.backbone_name == "rnn":
            # RNN path (unchanged)
            if isinstance(x, tuple):
                x, h = x
                h = tuple([h_.detach() for h_ in h])
            x, _ = pad_packed_sequence(x, batch_first=True, total_length=x_cat.size(1))
        else:
            # Non-RNN path: _run_backbone already returned hidden states tensor (B, T, H)
            pass


        out = {}
        for target in self.out_layers:
            out[target] = self.out_layers[target](x)
        return out, h

    def _apply_weight_tying(self) -> None:
        if not self.use_weight_tying:
            return

        for target in self.categorical_targets:
            source_col = target.replace("next_", "")
            if source_col not in self.in_layer.embedding_layers:
                continue
            if target not in self.out_layers:
                continue

            in_embed = self.in_layer.embedding_layers[source_col]
            out_head = self.out_layers[target]

            if out_head.linear.in_features != in_embed.embedding_dim:
                continue
            if out_head.linear.out_features != in_embed.num_embeddings:
                continue

            out_head.linear.weight = in_embed.weight
    
    def _backbone_last_hidden(self, inputs_embeds: torch.Tensor, attention_mask: Optional[torch.Tensor]):
        """
        Runs the backbone and returns a *Tensor* of hidden states (B, T, H).
        This is checkpoint-friendly and backbone-agnostic.
        """
        out = self.backbone(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        # nanoGPT-style student: backbone returns (logits, hidden_states)
        if self.backbone_name == "student_model":
            _, hidden = out
            return hidden

        # HF ModelOutput (common case)
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state

        # Some models may return tuples
        if isinstance(out, tuple) and len(out) > 0:
            return out[0]

        raise ValueError("Invalid output from backbone (cannot extract hidden states).")



    def _checkpoint(self, fn, *args, **kwargs):
        # Only pass kwargs supported by this torch version
        sig = inspect.signature(cp.checkpoint)
        safe_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return cp.checkpoint(fn, *args, **safe_kwargs)

    def _run_backbone(self, x, attention_mask):
        if not self.use_checkpoint:
            return self._backbone_last_hidden(x, attention_mask)

        # HF models that support their own gradient checkpointing:
        # rely on HF's mechanism (enabled in __init__) and DON'T wrap with cp.checkpoint
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            return self._backbone_last_hidden(x, attention_mask)

        # Non-HF backbones: use outer checkpoint wrapper
        return self._checkpoint(
            self._backbone_last_hidden,
            x,
            attention_mask,
            use_reentrant=False,
            preserve_rng_state=False,
        )
