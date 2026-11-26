Attempting to perform the InitializeDefaultDrives operation on the 'FileSystem' provider failed.
"""Generate and checkpoint one Qwen segment at a time."""

import gc
import logging
import os
from pathlib import Path
from secrets import randbits
import sys
import threading
import time

import numpy as np
import soundfile as sf


SAMPLE_RATE = 24000
VOICE = "Ryan"
ATTEMPTS = 3
log = logging.getLogger(__name__)


class QwenNarrator:
    def __init__(self, model_dir: Path, graph_package: Path | None = None):
        self.model_dir = model_dir
        self.graph_package = graph_package
        self.model = None

    def load_model(self, torch):
        if self.graph_package:
            package_path = str(self.graph_package)
            if package_path not in sys.path:
                sys.path.insert(0, package_path)
            from faster_qwen3_tts import FasterQwen3TTS

            model = FasterQwen3TTS.from_pretrained(
                str(self.model_dir), dtype=torch.bfloat16, max_seq_len=1536, local_files_only=True,
            )
            model.warmup()
            return model

        from qwen_tts import Qwen3TTSModel

        return Qwen3TTSModel.from_pretrained(
            str(self.model_dir), device_map="cuda:0", dtype=torch.bfloat16,
            attn_implementation="sdpa", local_files_only=True,
        )

    def synthesize(self, text: str, seed: int):
        import torch

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("The local PyTorch runtime needs a CUDA GPU with BF16 support.")
        frame_limit = max(96, 10 * len(text.split()) + 48)
        try:
            if self.model is None:
                mode = "CUDA Graphs" if self.graph_package else "standard CUDA"
                log.info("Loading Qwen on %s with %s from %s",
                         torch.cuda.get_device_name(), mode, self.model_dir)
                self.model = self.load_model(torch)
            torch.manual_seed(seed)
            with torch.inference_mode():
                clips, rate = self.model.generate_custom_voice(
                    text=text, language="English", speaker=VOICE, max_new_tokens=frame_limit,
                )
        except Exception:
            if self.graph_package:
                log.warning("CUDA Graphs failed. The next attempt will use standard CUDA.", exc_info=True)
                self.model = None
                self.graph_package = None
            raise
        clip = np.asarray(clips[0], dtype=np.float32).reshape(-1)
        if rate != SAMPLE_RATE or not clip.size or not np.isfinite(clip).all() or not np.any(clip):
            raise ValueError("Qwen returned invalid audio.")
        if clip.size >= (frame_limit - 1) * 1920:
            raise ValueError("Qwen reached the segment length limit before finishing.")
        return clip

    def recover(self, error):
        import torch

        if isinstance(error, torch.OutOfMemoryError):
            self.model = None
            return "The GPU ran out of memory. Close other GPU-heavy apps and resume."
        return str(error)

    def release_unused_memory(self):
        import torch

        if self.model is None:
            gc.collect()
            torch.cuda.empty_cache()


class NarrationWorker:
    def __init__(self, library, narrator):
        self.library = library
        self.narrator = narrator
        self.wake = threading.Event()
        self.closed = threading.Event()
        self.thread = threading.Thread(target=self.run, name="narration", daemon=True)
        self.thread.start()

    def close(self):
        self.closed.set()
        self.wake.set()
        self.thread.join(timeout=1)

    def run(self):
        while not self.closed.is_set():
            self.wake.clear()
            active = self.library.active()
            if active is None:
                self.wake.wait(1)
                continue
            book_id = active["id"]
            state = active["state"]
            if state != "running":
                stopped = "paused" if state == "pausing" else "cancelled"
                self.library.finish(book_id, state, stopped)
                continue
            segment = self.library.pending(book_id)
            if segment is None:
                self.library.finish(book_id, state, "complete")
                continue
            try:
                self.generate(segment)
            except Exception as error:
                log.exception("Book %s stopped at segment %s", book_id, segment["ordinal"])
                self.library.report_error(book_id, f"Segment {segment['ordinal']}: {error}")

    def generate(self, segment):
        book_id, ordinal = segment["book_id"], segment["ordinal"]
        path = self.library.wav_path(book_id, ordinal)
        if path.is_file():
            # A previous process may have stopped after replacing the WAV but before SQLite committed.
            try:
                info = sf.info(path)
            except sf.LibsndfileError:
                info = None
            if info and info.samplerate == SAMPLE_RATE and info.frames > 0 and info.channels == 1:
                self.library.save_segment(book_id, ordinal, info.duration, None)
                return
        path.parent.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        for attempt in range(ATTEMPTS):
            seed = randbits(32)
            try:
                clip = self.narrator.synthesize(segment["text"], seed)
            except Exception as error:
                message = self.narrator.recover(error)
                log.warning("Book %s segment %s attempt %s (seed %s): %s",
                            book_id, ordinal, attempt + 1, seed, message)
            else:
                break
            self.narrator.release_unused_memory()
            if self.closed.is_set() or self.library.active()["state"] != "running":
                return
        else:
            raise RuntimeError(message)
        temporary = path.with_suffix(".partial")
        with temporary.open("w+b") as stream:
            sf.write(stream, clip, SAMPLE_RATE, format="WAV", subtype="FLOAT")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        self.library.save_segment(book_id, ordinal, len(clip) / SAMPLE_RATE, time.perf_counter() - start)

