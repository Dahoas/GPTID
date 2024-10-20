import subprocess


def pythia():
    pythia_models = [
        "EleutherAI/pythia-70m",
        "EleutherAI/pythia-160m",
        "EleutherAI/pythia-410m",
        "EleutherAI/pythia-1.4b",
        "EleutherAI/pythia-2.8b",
        "EleutherAI/pythia-6.9b",
        "EleutherAI/pythia-12b-deduped",
    ]
    

def id_across_time():
    folder_path = "~/repos/transformer_manifold_learning/nanoGPT/logs/tml_train/owt_1_1_embd_768_save_model/"
    ckpts = [10000 * i for i in range(13)]
    for ckpt in ckpts:
        command = f"""\
    python embeddings.py \
    --model_path {folder_path}model_{ckpt}.pt \
    --model_mode oai \
    --tokenizer_type oai \
    --dataset_path Dahoas/openwebtext_val \
    --dataset_mode hf \
    --split validation \
    --context_len 1024 \
    --dataset_upper 4007 \
    --num_dataset_subsample 1024 \
    --shuffle_embeddings \
    --max_embeddings_per_sample 32 \
    --shuffle_embeddings_per_sample"""
        subprocess.run(command,
                        shell=True,
                        stderr=subprocess.STDOUT,)


if __name__ == "__main__":
    id_across_time()