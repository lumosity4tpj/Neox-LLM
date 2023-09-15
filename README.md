There are problems with the code for now.

# Diff from GPTNeox

- Add data processing of sft, dataset splicing mode and fix some bugs.

- Add `reset_mask` and `reset_id` to see if you can see the front when splicing, and reset the position id.

- Currently, `flash_attention` supports both v1 and v2, depending on the version installed, only global is supported, while `flash_attention_triton` only supports v1 and can support `reset_mask`, which is difficult to test and loss will be a little different, but the trend is the same.

!TODO:

- The unity of llama2 and llama;
- the reset mask of flash attention.


## prepare sft data

It is currently only tested on `alpaca` data.

```bash
python prepare_data_sft.py -d ./data/sft/alpaca_gpt4 -i ./data/raw_data/alpaca_gpt4 -t SPMTokenizer -v ./vocab_file/tokenizer.model alpaca_gpt4
```
Then generate the corresponding `.bin` file and `.idx` file for `text` and `label`, respectively.

## run_sft
Modify your slurm configuration and config files, refer to the `./custom_config` file.

```bash
sbatch run_sft.slurm
```