import os
import sys

from huggingface_hub import login, snapshot_download


MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "sentence-transformers/all-MiniLM-L6-v2",
]


def main():
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
    else:
        print("[warn] HF_TOKEN not set — gated repos (Meta-Llama) may fail.")

    cache_dir = os.environ.get(
        "HF_HOME",
        f"/scratch-shared/{os.environ.get('USER', 'user')}/hf_cache",
    )
    print(f"Cache directory: {cache_dir}")
    os.makedirs(cache_dir, exist_ok=True)

    failures = []
    for model_id in MODELS:
        print(f"\n=== {model_id} ===")
        try:
            snapshot_download(
                repo_id=model_id,
                cache_dir=cache_dir,
                resume_download=True,
            )
            print(f"OK  {model_id}")
        except Exception as e:
            print(f"FAIL  {model_id}: {e}")
            failures.append(model_id)

    if failures:
        print(f"\n{len(failures)} failed: {failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()
