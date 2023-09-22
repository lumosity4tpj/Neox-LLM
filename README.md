# Enhancements and Differences from GPTNeox
This README outlines the key changes and additions made in this repository compared to the `GPTNeox`. Our aim is to maintain transparency about the updates and improvements made to the original codebase.

- Add data processing of sft (support `alpaca (single-round) && sharegpt (multi-round)`), dataset splicing mode and fix some bugs.

- Add `reset_mask` and `reset_id` to see if you can see the front when splicing, and reset the position id (`reset_mask` can currently only use `global attention` and `flash attention triton`).

- Add `flash_attention v1 && v2`, depending on the version installed, `reset_mask` is currently not supported; while `flash_attention_triton` only supports `v1` and can support `reset_mask`, which is difficult to test and loss will be a little different, but the trend is the same.

- `Llama2` and `Llama1` have been merged, the main difference is that when `qkv weight` is splicing, if `GQA/MQA` is used, `torch.cat(QKV)` is used, otherwise `torch.stack(QKV)` is used. It is reflected in `./tools/convert_neox_llama_weights_to_hf.py` and `./tools/convert_raw_llama_weights_to_neox.py`.

# To-Do List

- Address the reset mask functionality in flash attention.
- Add NTK.
- some issue: bf16 + zero stage 1 + cpu offload.


# Quick Start

## prepare sft data

It is currently only tested on `alpaca, sharegpt` data，`alpaca` is a single-round dialogue, `sharegpt` is a multi-round dialogue, which can be referred to.

```bash
python prepare_data_sft.py -d ./data/sft/alpaca_gpt4 -i ./data/raw_data/alpaca_gpt4 -t SPMTokenizer -v ./vocab_file/tokenizer.model alpaca_gpt4

python prepare_data_sft.py -d ./data/sft/sharegpt -i ./data/raw_data/sharegpt -t SPMTokenizer -v ./vocab_file/tokenizer.model sharegpt

```
Then generate the corresponding `.bin` file and `.idx` file for `text` and `label`, respectively.


## convert raw weight to neox

```bash
python ./tools/convert_raw_llama_weights_to_neox.py --input_dir {raw_model_path} --model_size 70B --output_dir ./model/pretrain/llama2/70B --num_output_shards 8 --pipeline_parallel
```

## run_sft
Modify your slurm configuration and config files, refer to the `./custom_config` file.

```bash
sbatch run_sft.slurm # please modify the slurm in your env or config.
```

## convert neox weight to hf

```bash
python ./tools/convert_neox_llama_weights_to_hf.py --input_dir ./model/pretrain/llama2/70B/global_step0/ --model_size 70B --output_dir ./model/pretrain/llama2/70B_hf
```