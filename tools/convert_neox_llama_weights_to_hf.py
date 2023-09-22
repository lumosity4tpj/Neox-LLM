# Copyright 2022 EleutherAI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import os
import shutil
import warnings
import gc
import torch
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer


try:
    from transformers import LlamaTokenizerFast
except ImportError as e:
    warnings.warn(e)
    warnings.warn(
        "The converted tokenizer will be the `slow` tokenizer. To use the fast, update your `tokenizers` library and re-run the tokenizer conversion"
    )
    LlamaTokenizerFast = None


"""
Sample usage:

    ```
    python src/transformers/models/llama/convert_llama_weights_to_hf.py \
        --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
    ```

Thereafter, models can be loaded via:

    ```
    tokenizer = transformers.LLaMATokenizer.from_pretrained("/output/path/tokenizer/")

    model = transformers.LLaMAForCausalLM.from_pretrained("/output/path/llama-7b/")
    ```
"""

NUM_SHARDS = {
    "7B": 1,
    "7Bf": 1,
    "13B": 2,
    "13Bf": 2,
    "30B": 4,
    "65B": 8,
    "70B": 8,
    "70Bf": 8,
}


def compute_intermediate_size(n, ffn_dim_multiplier=1, multiple_of=256):
    return multiple_of * ((int(ffn_dim_multiplier * int(8 * n / 3)) + multiple_of - 1) // multiple_of)


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)


def write_model(model_path, input_base_path, model_size):
    assert model_size in NUM_SHARDS
    os.makedirs(model_path, exist_ok=True)
    tmp_model_path = os.path.join(model_path, "tmp")
    os.makedirs(tmp_model_path, exist_ok=True)

    params = read_json(os.path.join(input_base_path, "params.json"))
    num_shards = NUM_SHARDS[model_size]
    n_layers = params["n_layers"]
    n_heads = params["n_heads"]
    num_heads_per_input_shard = n_heads // num_shards
    dim = params["dim"]
    dims_per_head = dim // n_heads
    base = params.get("rope_theta", 10000.0)
    inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))
    if base > 10000.0:
        max_position_embeddings = 16384
    else:
        max_position_embeddings = 2048

    tokenizer_class = LlamaTokenizer if LlamaTokenizerFast is None else LlamaTokenizerFast
    tokenizer_path = os.path.join(input_base_path, "tokenizer.model")
    if tokenizer_path is not None:
        tokenizer = tokenizer_class(tokenizer_path)
        tokenizer.save_pretrained(model_path)
    vocab_size = tokenizer.vocab_size if tokenizer_path is not None else 32000

    if "n_kv_heads" in params:
        num_key_value_heads = params["n_kv_heads"]  # for GQA / MQA
        num_key_value_heads_per_input_shard = num_key_value_heads // num_shards
    else:  # compatibility with other checkpoints
        num_key_value_heads = n_heads
        num_key_value_heads_per_input_shard = num_heads_per_input_shard

    param_count = 0
    index_dict = {"weight_map": {}}
    for layer_i in range(n_layers):
        loaded = [torch.load(os.path.join(input_base_path, f"layer_{layer_i+2:02}-model_{i:02}-model_states.pt"), map_location="cpu") for i in range(num_shards)]
        filename = "pytorch_model-{:05d}-of-{:05d}.bin".format(
            layer_i + 1,
            n_layers + 1,
        )
        if num_key_value_heads != n_heads:
            query_key_value_weight = [loaded[i]["attention.query_key_value.weight"].view(-1, dims_per_head, dim) for i in range(num_shards)]
            query_weight = torch.cat([query_key_value_weight[i][:num_heads_per_input_shard, :, :].reshape(-1, dim) for i in range(num_shards)], dim=0)
            key_weight = torch.cat([query_key_value_weight[i][num_heads_per_input_shard:num_heads_per_input_shard + num_key_value_heads_per_input_shard, :, :].reshape(-1, dim) for i in range(num_shards)], dim=0)
            value_weight = torch.cat([query_key_value_weight[i][num_heads_per_input_shard + num_key_value_heads_per_input_shard:, :, :].reshape(-1, dim) for i in range(num_shards)], dim=0)
        else:
            query_key_value_weight = [loaded[i]["attention.query_key_value.weight"].view(-1, 3, dims_per_head, dim) for i in range(num_shards)]
            query_weight = torch.cat([query_key_value_weight[i][:, 0, :, :].reshape(-1, dim) for i in range(num_shards)], dim=0)
            key_weight = torch.cat([query_key_value_weight[i][:, 1, :, :].reshape(-1, dim) for i in range(num_shards)], dim=0)
            value_weight = torch.cat([query_key_value_weight[i][:, 2, :, :].reshape(-1, dim) for i in range(num_shards)], dim=0)

        # Unsharded
        state_dict = {
            f"model.layers.{layer_i}.self_attn.q_proj.weight": query_weight,
            f"model.layers.{layer_i}.self_attn.k_proj.weight": key_weight,
            f"model.layers.{layer_i}.self_attn.v_proj.weight": value_weight,
            f"model.layers.{layer_i}.self_attn.o_proj.weight": torch.cat([loaded[i]["attention.dense.weight"] for i in range(num_shards)], dim=1),
            f"model.layers.{layer_i}.mlp.gate_proj.weight": torch.cat([loaded[i]["mlp.w1.weight"] for i in range(num_shards)], dim=0),
            f"model.layers.{layer_i}.mlp.down_proj.weight": torch.cat([loaded[i]["mlp.w2.weight"] for i in range(num_shards)], dim=1),
            f"model.layers.{layer_i}.mlp.up_proj.weight": torch.cat([loaded[i]["mlp.w3.weight"] for i in range(num_shards)], dim=0),
            f"model.layers.{layer_i}.input_layernorm.weight": loaded[0]["input_layernorm.scale"],
            f"model.layers.{layer_i}.post_attention_layernorm.weight": loaded[0]["post_attention_layernorm.scale"],
            f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq": inv_freq
        }

        # state_dict[f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = loaded["attention.rotary_emb.inv_freq"]
        for k, v in state_dict.items():
            index_dict["weight_map"][k] = filename
            param_count += v.numel()
        torch.save(state_dict, os.path.join(tmp_model_path, filename))


    loaded = [torch.load(os.path.join(input_base_path, f"layer_00-model_{i:02}-model_states.pt"), map_location="cpu") for i in range(num_shards)]
    word_embeddings_weight = torch.cat([loaded[i]["word_embeddings.weight"] for i in range(num_shards)], dim=0)
    loaded = [torch.load(os.path.join(input_base_path, f"layer_{n_layers+3:02}-model_{i:02}-model_states.pt"), map_location="cpu") for i in range(num_shards)]
    norm_scale = loaded[0]["norm.scale"]
    loaded = [torch.load(os.path.join(input_base_path, f"layer_{n_layers+4:02}-model_{i:02}-model_states.pt"), map_location="cpu") for i in range(num_shards)]
    final_linear_weight = torch.cat([loaded[i]["final_linear.weight"] for i in range(num_shards)], dim=0)
    filename = "pytorch_model-{:05d}-of-{:05d}.bin".format(
        n_layers + 1,
        n_layers + 1,
    )
    # Unsharded
    state_dict = {
        "model.embed_tokens.weight": word_embeddings_weight,
        "model.norm.weight": norm_scale,
        "lm_head.weight": final_linear_weight,
    }

    for k, v in state_dict.items():
        index_dict["weight_map"][k] = filename
        param_count += v.numel()
    torch.save(state_dict, os.path.join(tmp_model_path, filename))

    # Write configs
    index_dict["metadata"] = {"total_size": param_count * 2}
    write_json(index_dict, os.path.join(tmp_model_path, "pytorch_model.bin.index.json"))
    ffn_dim_multiplier = params["ffn_dim_multiplier"] if "ffn_dim_multiplier" in params else 1
    multiple_of = params["multiple_of"] if "multiple_of" in params else 256
    config = LlamaConfig(
        hidden_size=dim,
        intermediate_size=compute_intermediate_size(dim, ffn_dim_multiplier, multiple_of),
        num_attention_heads=params["n_heads"],
        num_hidden_layers=params["n_layers"],
        rms_norm_eps=params["norm_eps"],
        num_key_value_heads=num_key_value_heads,
        pad_token_id=0,
        vocab_size=vocab_size,
        rope_theta=base,
        max_position_embeddings=max_position_embeddings,
    )
    config.save_pretrained(tmp_model_path)


    # Make space so we can load the model properly now.
    del state_dict
    del loaded
    gc.collect()

    print("Loading the checkpoint in a Llama model.")
    model = LlamaForCausalLM.from_pretrained(tmp_model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    # Avoid saving this as part of the config.
    del model.config._name_or_path
    print("Saving in the Transformers format.")
    model.save_pretrained(model_path, safe_serialization=True)
    tokenier = LlamaTokenizer.from_pretrained(tokenizer_path)
    tokenier.save_pretrained(model_path)
    shutil.rmtree(tmp_model_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Location of LLaMA weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "--model_size",
        choices=["7B", "13B", "30B", "65B", "70B"],
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write HF model and tokenizer",
    )
    args = parser.parse_args()
    write_model(
        model_path=os.path.join(args.output_dir, "llama-{}".format(args.model_size).lower()),
        input_base_path=args.input_dir,
        model_size=args.model_size,
    )


if __name__ == "__main__":
    main()
