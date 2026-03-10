from dataclasses import dataclass
from typing import Optional, Union, List
import torch.nn as nn
from peft import LoraConfig, TaskType
from ppm.datasets.event_logs import EventLog


PRETRAINED_CONFIGS = {
    "gpt-oss-20b": {
        "name": "openai/gpt-oss-20b",
        "embedding_size": 2880,
        "hidden_size": 2880,
        "pretrained": True,
        "fine_tuning_module_path": "h",
    },
    "gpt2-xl": {
        "name": "openai-community/gpt2-xl",
        "embedding_size": 1600,
        "hidden_size": 1600,
        "pretrained": True,
        "fine_tuning_module_path": "h",
    },
    "gpt2-large": {
        "name": "openai-community/gpt2-large",
        "embedding_size": 1280,
        "hidden_size": 1280,
        "pretrained": True,
        "fine_tuning_module_path": "h",
    },
    "gpt2-medium": {
        "name": "openai-community/gpt2-medium",
        "embedding_size": 1024,
        "hidden_size": 1024,
        "pretrained": True,
        "fine_tuning_module_path": "h",
    },
    "gpt2-small": {
        "name": "openai-community/gpt2",
        "embedding_size": 768,
        "hidden_size": 768,
        "pretrained": True,
        "fine_tuning_module_path": "h",
    },
    "distilgpt2": {
        "name": "distilbert/distilgpt2",
        "embedding_size": 768,
        "hidden_size": 768,
        "pretrained": True,
        "fine_tuning_module_path": "h",
    },
    "gpt2-mini": {
        "name": "erwanf/gpt2-mini",
        "embedding_size": 512,
        "hidden_size": 512,
        "pretrained": True,
        "fine_tuning_module_path": "h",
    },
    "gpt2-tiny": {
        "name": "sshleifer/tiny-gpt2",
        "embedding_size": 2,
        "hidden_size": 2,
        "pretrained": True,
        "fine_tuning_module_path": "h",
    },
    "llama32-1b": {
        "name": "meta-llama/Llama-3.2-1B",
        "embedding_size": 2048,
        "hidden_size": 2048,
        "pretrained": True,
        "fine_tuning_module_path": "layers",
    },
    "qwen3-0.6b": {
        "name": "Qwen/Qwen3-0.6B",
        "embedding_size": 1024,
        "hidden_size": 1024,
        "pretrained": True,
        "fine_tuning_module_path": "layers",
    },
    "qwen3-1.7b": {
        "name": "Qwen/Qwen3-1.7B-Base",
        "embedding_size": 2048,
        "hidden_size": 2048,
        "pretrained": True,
        "fine_tuning_module_path": "layers",
    },
    "qwen3-4b": {
        "name": "Qwen/Qwen3-4B-Base",
        "embedding_size": 2560,
        "hidden_size": 2560,
        "pretrained": True,
        "fine_tuning_module_path": "layers",
    },
    "qwen3-8b": {
        "name": "Qwen/Qwen3-8B-Base",
        "embedding_size": 4096,
        "hidden_size": 4096,
        "pretrained": True,
        "fine_tuning_module_path": "layers",
    },
    "qwen3-14b": {
        "name": "Qwen/Qwen3-14B-Base",
        "embedding_size": 5120,
        "hidden_size": 5120,
        "pretrained": True,
        "fine_tuning_module_path": "layers",
    },
    #"student_model": {
    #    "name": "student_model",
    #    "embedding_size": 896,  #?
    #    "hidden_size": 896,  #? check
    #    "pretrained": False,
    #    "fine_tuning_module_path": "layers",
    #},
}


def get_fine_tuning(fine_tuning, **kwargs):
    if fine_tuning == "lora":
        target_modules = (
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "up_proj",
                "down_proj",
                "o_proj",
                "gate_proj",
            ]
            if "gpt2" not in kwargs["model"]
            else None
        )
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=kwargs["r"],
            lora_alpha=kwargs["lora_alpha"],
            target_modules=target_modules,
            use_rslora=True,
        )
    elif fine_tuning == "freeze":
        return FreezeConfig(
            ix_layers=kwargs["freeze_layers"],
            module_path=kwargs["fine_tuning_module_path"],
        )
    elif fine_tuning is None:
        return
    else:
        raise ValueError("Invalid fine-tuning strategy")


def get_model_config(train_log: EventLog, training_config: dict): #, max_seq_len: int):
    pretrained_config = PRETRAINED_CONFIGS.get(training_config["backbone"], {})
    if pretrained_config:
        fine_tuning = get_fine_tuning(
            fine_tuning=training_config["fine_tuning"],
            r=training_config["r"],
            lora_alpha=training_config["lora_alpha"],
            freeze_layers=training_config["freeze_layers"],
            fine_tuning_module_path=pretrained_config["fine_tuning_module_path"],
            model=training_config["backbone"],
        )
        pretrained_config["fine_tuning"] = fine_tuning
    if training_config["backbone"] == "student_model":
        backbone_hf_name = "student_model"
    elif training_config["backbone"] != "rnn":
        backbone_hf_name = pretrained_config["name"]
    else:
        backbone_hf_name = "rnn"
    
    if pretrained_config:
        embedding_size = pretrained_config["embedding_size"]
        hidden_size = pretrained_config["hidden_size"]
    elif training_config["backbone"] == "student_model":
        hidden_size = training_config["hidden_size"]
        embedding_size = hidden_size
    else:
        hidden_size = training_config.get("hidden_size", 768)
        embedding_size = hidden_size
    
    return {
        "embedding_size": embedding_size,
        "categorical_cols": train_log.features.categorical,
        "categorical_sizes": train_log.categorical_sizes,
        "numerical_cols": train_log.features.numerical,
        "categorical_targets": train_log.targets.categorical,
        "numerical_targets": train_log.targets.numerical,
        "padding_idx": train_log.special_tokens["<PAD>"],
        "strategy": training_config["strategy"],
        "backbone_name": backbone_hf_name,
        "backbone_pretrained": True if pretrained_config else False,
        "backbone_finetuning": pretrained_config.get("fine_tuning", None),
        "backbone_type": training_config.get("rnn_type", None),
        "backbone_hidden_size": hidden_size,
        "backbone_n_layers": training_config.get("n_layers", None),
        "device": training_config["device"],
        #"max_seq_len": max_seq_len,
        "time_positional_encoding": training_config["time_positional_encoding"],
        "use_weight_tying": training_config.get("weight_tying", True),
    }
    

def get_model_params(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
            
    return all_param, trainable_params
            

@dataclass   #Python autogeneraties __init__
class FreezeConfig:
    """
    Configures which layers/blocks to freeze in a PyTorch model.
    Freezes all(except normalisation layers), then unfreezes specific layers

    Args:
        layers (Optional[Union[str, int, List[Union[str, int]]]):
            - Names (str) or indices (int) of layers to freeze
            - If `None`, freezes all parameters
        module_path (Optional[str]):
            - Dot-separated path to a `ModuleList` in the model (e.g., "bert.encoder.layer")
            - Required when using integer indices to locate layers
    """

    ix_layers: Optional[Union[int, List[int]]] = None
    module_path: Optional[str] = None

    def apply(self, model: nn.Module) -> None:
        """Freezes specified parameters in the model"""
        self._freeze_all(model)

        if self.ix_layers is None:
            return

        ix_layers = (
            [self.ix_layers] if not isinstance(self.ix_layers, list) else self.ix_layers
        )
        for ix in ix_layers:
            if isinstance(ix, int):
                self._unfreeze_by_index(model, ix)
            else:
                raise TypeError(f"Invalid layer type: {type(ix)}")

    def _freeze_all(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if "norm" not in name:
                param.requires_grad = False

    def _unfreeze_by_index(self, model: nn.Module, index: int) -> None:
        if not self.module_path:
            raise ValueError("module_path required for index-based freezing")

        module = model
        for part in self.module_path.split("."):
            module = getattr(module, part)

        if not isinstance(module, nn.ModuleList):
            raise TypeError(f"{self.module_path} must be a ModuleList")

        for param in module[index].parameters():
            param.requires_grad = True
