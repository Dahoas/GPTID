from datasets import load_dataset
import json


def jsonl_to_dict(samples: list[dict]):
    return {k: [sample[k] for sample in samples] for k in samples[0]}


def load_data(path: str, mode: str):
    if mode == "json":
        with open(path, "r") as f:
            data = jsonl_to_dict(json.load(f))
            data = {"train": data}
    elif mode == "jsonl":
        with open(path, "r") as f:
            data = f.readlines()
            data = jsonl_to_dict([json.loads(sample) for sample in data])
        data = {"train": data}
    elif mode == "hf":
        data = load_dataset(path)
    return data