Attempting to perform the InitializeDefaultDrives operation on the 'FileSystem' provider failed.
"""Generate and measure one narrator configuration in a dedicated process."""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import random
import re
import subprocess
import threading
import time

import numpy as np
import soundfile as sf
import torch


ROOT = Path(__file__).resolve().parent
ENGINES = ("chatterbox", "qwen")
MIB = 1024 ** 2
GUTENBERG_SOURCE = "https://www.gutenberg.org/files/1661/1661-0.txt"


def passage():
    book = (ROOT / ".runtime" / "gutenberg-1661.txt").read_text(encoding="utf-8-sig")
    chapter = book.split("I had called upon my friend, Mr. Sherlock Holmes,", 1)
    start = "I had called upon my friend, Mr. Sherlock Holmes," + chapter[1]
    opening = start.split("The portly client puffed out his chest", 1)[0]
    paragraphs = [" ".join(block.split()) for block in re.split(r"\n\s*\n", opening.strip())]
    # Use the final sentences of the next speech as a separate stress clip.
    _, last = paragraphs[-1].split(" Perhaps, Mr. Wilson", 1)
    return paragraphs[:-1], "Perhaps, Mr. Wilson" + last


def seed_rng(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_engine(engine, model_dir):
    if engine == "chatterbox":
        from chatterbox.tts import ChatterboxTTS

        model = ChatterboxTTS.from_local(model_dir, "cuda")

        def generate(text):
            audio = model.generate(text, exaggeration=0.5, cfg_weight=0.5)
            return audio.detach().cpu().numpy().reshape(-1), model.sr

        return generate

    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        str(model_dir), device_map="cuda:0", dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True,
    )

    def generate(text):
        # Allow over 50% more frames than slow narration needs and bound missed EOS.
        frame_limit = max(96, 10 * len(text.split()) + 48)
        audio, sample_rate = model.generate_custom_voice(
            text=text, language="English", speaker="Ryan",
            max_new_tokens=frame_limit,
        )
        if len(audio[0]) / sample_rate >= (frame_limit - 1) / 12.5:
            print(f"Qwen may have reached its {frame_limit}-frame generation limit.", flush=True)
        return np.asarray(audio[0]).reshape(-1), sample_rate

    return generate


def gpu_used():
    try:
        sample = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=3,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        return int(sample.stdout.splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


@contextmanager
def sample_gpu():
    samples = []
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            used = gpu_used()
            if used is not None:
                samples.append(used)
            stop.wait(0.15)

    monitor = threading.Thread(target=poll, daemon=True)
    monitor.start()
    try:
        yield samples
    finally:
        stop.set()
        monitor.join()


def synthesize(generate, text, seed, target):
    print(f"Generating {target.name} ({len(text.split())} words)", flush=True)
    seed_rng(seed)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        audio, sample_rate = generate(text)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if not audio.size or not np.isfinite(audio).all() or not np.any(audio):
        raise ValueError(f"Invalid waveform for {target.name}")
    sf.write(target, audio, sample_rate, subtype="FLOAT")
    result = {
        "file": target.name,
        "words": len(text.split()),
        "seconds": round(elapsed, 3),
        "audio_seconds": round(audio.size / sample_rate, 3),
        "peak_absolute_sample": float(np.max(np.abs(audio))),
        "seed": seed,
    }
    print(result, flush=True)
    return result


def make_preview(raw_dir, count, output, replace_index=None):
    clips = []
    for index in range(count):
        name = "regenerated.wav" if index == replace_index else f"segment-{index + 1:02d}.wav"
        audio, rate = sf.read(raw_dir / name, dtype="float64")
        if index:
            clips.append(np.zeros(round(0.35 * rate)))
        clips.append(audio)
    joined = np.concatenate(clips)
    active = joined[np.abs(joined) >= 0.01]
    rms = np.sqrt(np.mean(active ** 2)) if active.size else 1.0
    gain = min(10 ** (-22 / 20) / rms, 10 ** (-1 / 20) / np.max(np.abs(joined)))
    sf.write(output, joined * gain, rate, subtype="PCM_24")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=ENGINES)
    args = parser.parse_args()
    engine = args.engine
    model_dir = ROOT / ".runtime" / "models" / engine
    output = ROOT / "comparison"
    raw = output / engine
    raw.mkdir(parents=True, exist_ok=True)
    segments, long_segment = passage()
    (output / "passage.txt").write_text(
        "\n\n".join(segments) + "\n",
        encoding="utf-8",
    )
    (output / "long-paragraph.txt").write_text(long_segment + "\n", encoding="utf-8")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch did not detect a compatible GPU")
    baseline = gpu_used()
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()
    with sample_gpu() as samples:
        torch.cuda.synchronize()
        start = time.perf_counter()
        generate = load_engine(engine, model_dir)
        torch.cuda.synchronize()
        load_time = time.perf_counter() - start
        warmup = synthesize(generate, "Holmes closed the door.", 200, raw / "warmup.wav")
        results = [
            synthesize(generate, text, 300 + index, raw / f"segment-{index + 1:02d}.wav")
            for index, text in enumerate(segments)
        ]
        replacement_index = 5
        replacement = synthesize(
            generate, segments[replacement_index], 400, raw / "regenerated.wav"
        )
        stress = synthesize(generate, long_segment, 500, raw / "long-paragraph.wav")

    for suffix, index in (("", None), ("-regenerated", replacement_index)):
        make_preview(raw, len(segments), output / f"{engine}{suffix}.wav", index)

    metrics = {
        "engine": engine,
        "voice": "built-in" if engine == "chatterbox" else "Ryan",
        "model_revision": (model_dir / "revision.txt").read_text(encoding="utf-8").strip(),
        "source": GUTENBERG_SOURCE,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "cold_load_seconds": round(load_time, 3),
        "warmup": warmup,
        "warmed_synthesis_seconds": round(sum(item["seconds"] for item in results), 3),
        "raw_audio_seconds": round(sum(item["audio_seconds"] for item in results), 3),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / MIB, 1),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / MIB, 1),
        "device_baseline_mib": baseline,
        "device_sampled_peak_mib": max(samples, default=None),
        "device_sample_count": len(samples),
        "segments": results,
        "replacement_segment": replacement_index + 1,
        "replacement": replacement,
        "long_paragraph": stress,
    }
    (output / f"{engine}.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

