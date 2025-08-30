# Verl Codebase

## Global Settings

- `$CACHE`: Global Settings of the cache file, must be the absolute path without `~` or soft links
- `$CACHE/hf_models/{hf_id}/{hf_name}`: Default Path of model
- `$CACHE/verl-data/{dataset_name}/train.parquet(test.parquet)`: Default Path of Data

## Installation

```bash
conda create -n verl python==3.10
conda activate verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
cd peft
pip install -v -e .
```

## Train

```bash
python misc/download_model.py
python examples/data_preprocess/standard.py
bash scripts_math/example_qwen7b.sh
```