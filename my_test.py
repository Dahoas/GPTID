from typing import List
import numpy as np
import pandas as pd
import argparse
import json
import pathlib
import datetime
import os

import torch
import torch.nn as nn

from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaModel

from scipy.spatial.distance import cdist
from skdim.id import MLE

from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset

from util import load_data


MAX_LENGTH = 512
batch_size = 1
log_size = 10_000
device = "cuda" if torch.cuda.is_available() else "cpu"
LOG_EMBEDDINGS = False

def get_embeddings(texts: List[str], 
                   tokenizer, 
                   model,):
    embedding_list = []
    log_list = []
    num_batches = (len(texts) + batch_size - 1) // batch_size
    batches = [texts[i*batch_size:(i+1)*batch_size] for i in range(num_batches)]
    for i, batch in tqdm(enumerate(batches)):
        inputs = tokenizer(batch, truncation=True, max_length=MAX_LENGTH, return_tensors="pt", padding="max_length")
        inputs["input_ids"] = inputs["input_ids"].to(device)
        inputs["attention_mask"] = inputs["attention_mask"].to(device)
        with torch.no_grad():
            embeddings = model(**inputs)[0].cpu()
            log_list.append(embeddings)
            # Remove EOS tokens
            mask = inputs["input_ids"] != tokenizer.eos_token_id
            for m, embedding in zip(mask, embeddings):
                embedding = embedding[:m.sum()]
                embedding_list.append(embedding)
            if len(log_list) >= log_size and LOG_EMBEDDINGS:
                log_embeddings = torch.cat(log_list, dim=0)
                torch.save(log_embeddings, os.path.join(log_folder, f"embeddings_{i}.pt"))
                log_list = []
    if len(log_list) > 0 and LOG_EMBEDDINGS:
        log_embeddings = torch.cat(log_list, dim=0)
        torch.save(log_embeddings, os.path.join(log_folder, f"embeddings_{i+1}.pt"))
    return embedding_list

def get_stats(nums: List[float]):
    return dict(mean=np.mean(nums), 
            std=np.std(nums),
            min=np.min(nums),
            max=np.max(nums),
            len=len(nums,))

def get_mle(samples: List[torch.tensor], K=5, mode="mean",):
    #TODO(dahoas): not entirely sure this is the right way of evaluating intrinsic dim
    mles = []
    solver = MLE()
    print("Dnoise: ", solver.dnoise)
    print("Neighborhood base: ", solver.neighborhood_based)
    extrinsic_dim = samples[0].shape[-1]
    if mode == "mean":
        for sample in tqdm(samples):
            int_dim = solver.fit_transform(sample.numpy(), n_neighbors=K,)
            mles.append(int_dim)
        print(sample.shape)
        int_dim = np.mean(mles)
        stats = get_stats(mles)
        stats["extrinsic_dim"] = extrinsic_dim
        return int_dim, mles, stats
    elif mode == "all":
        samples = torch.cat(samples, dim=0).numpy()
        print(samples.shape)
        int_dim = solver.fit_transform(samples, n_neighbors=K,)
        return int_dim, [], dict(int_dim=int_dim, extrinsic_dim=extrinsic_dim)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="gpt2")
    parser.add_argument("--dataset_path", default="roneneldan/TinyStories")
    parser.add_argument("--dataset_mode", default="hf")
    parser.add_argument("--split", default="train")
    parser.add_argument("--dataset_lower", default=0, type=int)
    parser.add_argument("--dataset_upper", default=100, type=int)
    parser.add_argument("--log_folder", default="/storage/home/hcoda1/6/ahavrilla3/p-wliao60-0/alex/repos/transformer_manifold_learning/GPTID/logs")
    parser.add_argument("--mle_mode", default="mean")
    parser.add_argument("--K", default=20, type=int)
    args = parser.parse_args()

    # Make logging folder
    log_folder = pathlib.Path(os.path.join(args.log_folder, datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")))
    log_folder.mkdir(exist_ok=True, parents=True)
    print("Logging in: ", log_folder)
    with open(os.path.join(log_folder, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    model = AutoModel.from_pretrained(args.model_path).to(device).half()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    texts = load_data(args.dataset_path, mode=args.dataset_mode)[args.split]["text"][args.dataset_lower:args.dataset_upper]
    
    embeddings = get_embeddings(texts, tokenizer, model)
    int_dim, mles, stats = get_mle(embeddings, mode=args.mle_mode, K=args.K)
    print("Intrinsic dim: ", int_dim)
    print(json.dumps(stats, indent=2))
    with open(os.path.join(log_folder, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
