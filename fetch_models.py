Attempting to perform the InitializeDefaultDrives operation on the 'FileSystem' provider failed.
"""Cache just the checkpoints used in the listening test."""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


MODELS = {
    "chatterbox": (
        "ResembleAI/chatterbox",
        ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors",
         "tokenizer.json", "conds.pt"],
    ),
    "qwen": ("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", None),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=MODELS)
    args = parser.parse_args()
    destination = Path(__file__).resolve().parent / ".runtime" / "models" / args.engine
    repo, files = MODELS[args.engine]
    revision_file = destination / "revision.txt"
    revision = (
        revision_file.read_text(encoding="utf-8").strip()
        if revision_file.exists()
        else HfApi().model_info(repo).sha
    )
    destination.mkdir(parents=True, exist_ok=True)
    revision_file.write_text(revision + "\n", encoding="utf-8")
    snapshot_download(repo, revision=revision, allow_patterns=files, local_dir=destination)


if __name__ == "__main__":
    main()

