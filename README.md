# David vs. Goliath in Next Activity Prediction: Argmax vs. LSTM, Transformer, and LLM

*Paper under review.*


## Environment and Setup

This project requires a CUDA-enabled PyTorch environment for training and evaluation. To ensure reproducibility, a preconfigured containerized environment is provided. A minimal manual setup is also described for reference.

---

## Recommended Setup (Containerized Environment)

The repository includes a fully specified container configuration that reproduces the development and training environment.

### Requirements

- Docker  
- NVIDIA GPU with compatible drivers  
- NVIDIA Container Toolkit  

### Setup Procedure

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Open the repository in an editor or workflow that supports container-based development environments.

3. Reopen the project inside the container when prompted.

During initialization, the environment will automatically:

- install Python dependencies from `requirements.txt`  
- configure the Python environment  
- verify CUDA availability in PyTorch  

The project workspace is mounted at:

```bash
/app
```

### Persistent Data and Caching

To avoid repeated downloads and preserve intermediate data:

- Hugging Face cache is stored in `./.hf-cache`  
- runtime data is stored in a persistent Docker volume   

---

## Manual Setup (Alternative)

A manual installation is possible but requires careful alignment of CUDA, PyTorch, and system dependencies.

### Requirements

- Python 3.x  
- NVIDIA GPU with CUDA support (recommended)  
- PyTorch compatible with the installed CUDA version  

### Setup Procedure

```bash
git clone <repository-url>
cd <repository-name>

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
HF_TOKEN="your huggingface token"
WANDB_API_KEY="your wandb api key"
```

In addition, the project depends on the `nanoGPT` repository, which should be available on the Python path:

```bash
git clone https://github.com/karpathy/nanoGPT.git
export PYTHONPATH="$(pwd)/nanoGPT:${PYTHONPATH}"
```

## Reproducibility

The container configuration provided in this repository defines the reference environment used for all experiments. It specifies:

- base system (CUDA, PyTorch)  
- Python dependencies  
- runtime configuration  

Using this setup is recommended to reproduce results with minimal variation. The manual setup is provided for transparency and flexibility but may require additional adjustments depending on the system.

## Scripts and Structure

```
.
├── data/                           # Event logs (automatically downloaded)
├── scripts/                        # Experiment scripts and configs
│   ├── *.sh                        
│   ├── *.txt                                         
├── ppm/                            # Source code
├── results/                        # Results hyperparameter search and experiments
├── main_baseline.py                # Script to calculate baseline
├── main_distill.py                 # Main training script for distillation
├── main_nep.py                     # Main training script for finetuning LLMs and training Transformers and LSTMs from scratch
├── main_prefixes.py                # Calculates prefix statistics (Table 2)
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```


## Usage

### Single experiments

**Baseline**

```bash
python main_baseline.py \
  --dataset BPI20PrepaidTravelCosts \
  --lifecycle\
  --wandb
```

**LLM fine-tuning**

```bash
python main_nep.py \
  --project_name BPI20_003 \
  --dataset BPI20TravelPermitData  \
  --backbone qwen3-0.6b \
  --lr 0.0005  \
  --val_size .1 \
  --val_split prefix \
  --patience 10 \
  --freeze_layers 1 -1 \
  --categorical_features activity resource \
  --continuous_features accumulated_time  \
  --lifecycle \
  --wandb \
  --compile \
  --append_run_info
```

**Training from scratch**

```bash
python main_nep.py \
  --project_name BPI12_002 \
  --dataset BPI12 \
  --backbone nanogpt \
  --fine_tuning none \
  --n_layers 4 \
  --n_heads 8 \
  --hidden_size 512 \
  --lr 0.005 \
  --val_size .1 \
  --val_split prefix \
  --patience 10 \
  --freeze_layers 1 -1 \
  --categorical_features activity resource \
  --continuous_features accumulated_time amount \
  --lifecycle \
  --wandb \
  --compile \
  --append_run_info
```

**Training from scratch**
```bash
python main_distill.py \
  --project_name Distill_BPI17_001 \ 
  --dataset BPI17 \
  --t_path /app/persisted_models/best/ \
  --t_model_name BPI17_qwen3-1.7b_run_f3a4184u.pth \
  --hidden_size 768 \
  --n_layers 12 \
  --n_heads 12 \
  --lr 0.005  \
  --val_size .1 \
  --val_split prefix \
  --patience 10 \
  --categorical_features activity resource \
  --continuous_features accumulated_time amount  \
  --lifecycle \
  --wandb \
  --strategy sum
```


Use the argument `--wandb` to enable wandb.

### Hyperparameter search

We used Slurm on our HPC clusters. Check `scripts/*.params.txt` to see how to reproduce our jobs or run other configurations locally.

### Results
The results of our hyperparameter search and experiments can be found under `results/`.

## Contact

For questions or feedback, reach me at [xxx](mailto:xxx) or open an issue here.

## Credits

This code is based on Rafael Oyamada's: https://github.com/raseidi/llm-peft-ppm
