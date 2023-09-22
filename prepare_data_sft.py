import os
import json
import argparse
from abc import abstractmethod
from tqdm import tqdm


TOKENIZER_CHOICES = [
    "HFGPT2Tokenizer",
    "HFTokenizer",
    "GPT2BPETokenizer",
    "CharLevelTokenizer",
    "TiktokenTokenizer",
    "SPMTokenizer",
]

DATASET_CHOICES = ["alpaca_gpt4", "sharegpt"]
def get_args():
    parser = argparse.ArgumentParser(description="prepare finetune data")
    parser.add_argument(
        "dataset",
        nargs="?",
        default="alpaca_gpt4",
        help="name of dataset.",
        choices=DATASET_CHOICES,
    )
    parser.add_argument(
        "-t",
        "--tokenizer",
        default="GPT2BPETokenizer",
        choices=TOKENIZER_CHOICES,
        help=f'Type of tokenizer to use - choose from {", ".join(TOKENIZER_CHOICES)}',
    )
    parser.add_argument(
        "-i",
        "--input-data-dir",
        default=None,
        help=f"Directory of input datasets"
        f"files - defaults to ./data",
    )
    parser.add_argument(
        "-d",
        "--output-data-dir",
        default=None,
        help=f"Directory to which to download datasets / tokenizer "
        f"files - defaults to ./data",
    )
    parser.add_argument(
        "-v", "--vocab-file", default=None, help=f"Tokenizer vocab file (if required)"
    )
    parser.add_argument(
        "-n", "--num-workers", default=1, help=f"num_workers"
    )
    return parser.parse_args()


class DataBase:
    def __init__(
        self,
        tokenizer_type=None,
        vocab_file=None,
        input_data_dir=None,
        output_data_dir=None,
        num_workers=1,
    ):
        self.vocab_file = vocab_file
        self.tokenizer_type = tokenizer_type
        self.input_data_dir = input_data_dir
        self.output_data_dir = output_data_dir
        self.num_workers = num_workers
        self.ftfy = False

        self.tmp_dir = f"./data/tmp/{self.name}"
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.tmp_file = self.tmp_dir + "/tmp.jsonl"

    @property
    @abstractmethod
    def name(self):
        """name of dataset"""
        pass

    @property
    @abstractmethod
    def prompt_dict(self):
        """prompt dict"""
        pass

    @property
    @abstractmethod
    def mask_before_text(self):
        """apply loss masks before certain text(s)"""
        pass

    @property
    @abstractmethod
    def mask_split_text(self):
        """a separator used for multiple conversation"""
        pass

    @property
    @abstractmethod
    def include_pivot(self):
        """whether mask include pivot"""
        pass

    def convert_data(self):
        """convert data format"""
        pass

    def tokenize(self):
        """tokenizes dataset"""

        cmd = f"python tools/preprocess_data_with_mask.py \
            --input {self.tmp_file} \
            --output-prefix {self.output_data_dir}/{self.name} \
            --vocab {self.vocab_file} \
            --dataset-impl mmap \
            --tokenizer-type {self.tokenizer_type} \
            --workers {self.num_workers} \
            --mask-before-text '{self.mask_before_text}' "
        
        if self.mask_split_text:
            cmd += f"--mask-split-text '{self.mask_split_text}' "

        if self.include_pivot:
            cmd += f"--include-pivot "
        
        if self.ftfy:
            cmd += f"--ftfy "

        os.system(cmd)

    def prepare(self):
        self.convert_data()
        self.tokenize()


class Alpaca(DataBase):
    name = "alpaca"
    prompt_dict = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\n{output}"
        ),
    }
    mask_before_text = "Response:\n"
    mask_split_text = None
    include_pivot = True

    def convert_data(self):
        prompt_input, prompt_no_input = self.prompt_dict["prompt_input"], self.prompt_dict["prompt_no_input"]
        with open(self.tmp_file, "wt") as f:
            for input_path in os.listdir(self.input_data_dir):
                ext = os.path.basename(input_path).split(".")[-1]
                if ext in ["txt", "jsonl"]:
                    for line in tqdm(open(self.input_data_dir + '/' + input_path, "rt")):
                        info = json.loads(line.strip())
                        if "instruction" not in info and "output" not in info:
                            continue
                        text = prompt_input.format_map(info) if info.get("input", "") != "" else prompt_no_input.format_map(info)
                        data = {
                            "text": text,
                        }
                        f.write(f"{json.dumps(data)}\n")
                        f.flush()


class ShareGPT(DataBase):
    name = "sharegpt"
    prompt_dict = {
        "system": (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions."
        ),
    }
    mask_before_text = "ASSISTANT: "
    mask_split_text = "</s>"
    include_pivot = True

    def convert_data(self):
        prompt = self.prompt_dict["system"]
        with open(self.tmp_file, "wt") as f:
            for input_path in os.listdir(self.input_data_dir):
                ext = os.path.basename(input_path).split(".")[-1]
                assert ext == "json"
                input_json = json.load(open(self.input_data_dir + '/' + input_path, "rt"))
                for info in tqdm(input_json):
                    info = info["conversations"]
                    text = prompt + " "
                    for (chat) in info:
                        role, message = chat["from"], chat["value"]
                        if role == "human":
                            text += "USER: " + message + " "
                        elif role == "gpt":
                            text += "ASSISTANT: " + message + self.mask_split_text
                        else:
                            break
                    else:
                        data = {
                            "text": text
                        }
                        f.write(f"{json.dumps(data)}\n")
                        f.flush()


DATA_DATALOADERS = {
    "alpaca_gpt4": Alpaca,
    "sharegpt": ShareGPT,
}

def prepare_dataset(
    dataset_name: str,
    tokenizer_type: str = None,
    input_data_dir: str = None,
    output_data_dir: str = None,
    vocab_file: str = None,
    num_workers: int = None,
):
    os.makedirs(output_data_dir, exist_ok=True)
    DataloaderClass = DATA_DATALOADERS.get(dataset_name.lower(), None)
    if DataloaderClass is None:
        raise NotImplementedError(
            f'Dataset "{dataset_name}" not recognized - please choose from {list(DATA_DATALOADERS.keys())}'
        )
    else:
        d = DataloaderClass(
            tokenizer_type=tokenizer_type,
            vocab_file=vocab_file,
            input_data_dir=input_data_dir,
            output_data_dir=output_data_dir,
            num_workers=num_workers,
        )
        d.prepare()



if __name__ == "__main__":
    args = get_args()
    prepare_dataset(
        dataset_name=args.dataset,
        tokenizer_type=args.tokenizer,
        input_data_dir=args.input_data_dir,
        vocab_file=args.vocab_file,
        output_data_dir=args.output_data_dir,
        num_workers=args.num_workers,
    )
