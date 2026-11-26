Attempting to perform the InitializeDefaultDrives operation on the 'FileSystem' provider failed.
"""Stream pitch-preserving narration and pauses into a tagged MP3."""

from itertools import pairwise
import logging
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from uuid import uuid4

import numpy as np
import soundfile as sf

from studio.narration import SAMPLE_RATE


PARAGRAPH_PAUSE = 0.35
CHAPTER_PAUSE = 1.0
SPLIT_PAUSE = 0.12
DEFAULT_WPM = 145
BITRATES = (192, 256, 320)
log = logging.getLogger(__name__)


def find_ffmpeg(runtime):
    portable = next((runtime / "tempo-tools" / "imageio_ffmpeg" / "binaries").glob("ffmpeg*.exe"), None)
    return str(portable) if portable else shutil.which("ffmpeg")


def gap(previous, following):
    if previous["chapter"] != following["chapter"]:
        return CHAPTER_PAUSE
    return PARAGRAPH_PAUSE if previous["paragraph_end"] else SPLIT_PAUSE


def preview_audio(library, book_id):
    rows = library.latest_audio(book_id)
    if not rows:
        raise ValueError("Generate a segment for this book before loading its preview.")
    clips = []
    for index, row in enumerate(rows):
        clip, rate = sf.read(library.wav_path(book_id, row["ordinal"]), dtype="float32")
        if rate != SAMPLE_RATE or clip.ndim != 1 or not clip.size or not np.isfinite(clip).all():
            raise ValueError(f"Saved audio for segment {row['ordinal']} is invalid.")
        if index:
            clips.append(np.zeros(round(gap(rows[index - 1], row) * rate), dtype=np.float32))
        clips.append(clip)
    samples = np.concatenate(clips)
    # Preserve relative levels and reserve headroom when converting float WAV to PCM.
    gain = 0.9 / max(1, float(np.max(np.abs(samples))))
    return SAMPLE_RATE, np.rint(samples * (gain * 32767)).astype(np.int16)


def export_mp3(library, export_root, ffmpeg, book_id, chapter, pace, bitrate, *, on_progress=None):
    if not ffmpeg:
        raise ValueError("FFmpeg is required for MP3 export. See the README.")
    pace, bitrate = int(pace), int(bitrate)
    if not 110 <= pace <= 200 or bitrate not in BITRATES:
        raise ValueError("Choose a pace from 110 to 200 WPM and a listed MP3 bitrate.")
    book = library.book(book_id)
    if not book:
        raise ValueError("Select a saved book.")
    rows = library.export_segments(book_id, chapter)
    if not rows or any(row["audio_seconds"] is None for row in rows):
        raise ValueError("Finish generating the selected audiobook or chapter first.")
    if on_progress:
        on_progress(0, len(rows))

    title = book["title"]
    chapter_data = library.chapter(book_id, chapter) if chapter else None
    if chapter_data:
        title += f" - {chapter_data['title']}"
    basename = re.sub(r'[^\w .-]+', '_', title).strip(" .")[:100].rstrip(" .")
    destination = export_root / uuid4().hex[:12]
    destination.mkdir(parents=True)
    output = destination / f"book-{book_id}-{basename or 'narration'}.mp3"
    temporary = output.with_suffix(".partial")
    gaps = [gap(previous, following) for previous, following in pairwise(rows)]
    words = sum(row["words"] for row in rows)
    speech_seconds = sum(row["audio_seconds"] for row in rows)
    target_speech_seconds = max(words * 60 / pace - sum(gaps), speech_seconds / 2)
    factor = max(0.5, speech_seconds / target_speech_seconds)

    command = [
        ffmpeg, "-nostdin", "-hide_banner", "-v", "error", "-f", "f32le", "-ar", str(SAMPLE_RATE),
        "-ac", "1", "-i", "pipe:0", "-c:a", "libmp3lame", "-ar", "48000", "-b:a", f"{bitrate}k",
        "-metadata", f"title={title}", "-metadata", f"artist={book['author']}",
        "-metadata", f"album={book['title']}", "-f", "mp3", str(temporary),
    ]
    try:
        with tempfile.TemporaryFile() as errors:
            with subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors) as encoder:
                try:
                    for index, row in enumerate(rows):
                        path = library.wav_path(book_id, row["ordinal"])
                        result = subprocess.run([
                            ffmpeg, "-nostdin", "-v", "error", "-i", str(path),
                            "-af", f"atempo={factor:.8f}", "-f", "f32le", "pipe:1",
                        ], capture_output=True, check=False)
                        if result.returncode or not result.stdout:
                            detail = result.stderr.decode(errors="replace").strip()
                            raise RuntimeError(f"Could not read saved segment {row['ordinal']}. {detail}")
                        samples = np.frombuffer(result.stdout, dtype="<f4")
                        peak = float(np.max(np.abs(samples)))
                        if not np.isfinite(peak):
                            raise RuntimeError(f"Segment {row['ordinal']} contains invalid samples.")
                        # Retain the segment waveform while reserving headroom for MP3 encoding.
                        gain = min(0.8, 0.8 / max(peak, 0.8))
                        encoder.stdin.write((samples * gain).astype("<f4", copy=False).tobytes())
                        if index < len(gaps):
                            encoder.stdin.write(bytes(4 * round(gaps[index] * SAMPLE_RATE)))
                        if on_progress:
                            on_progress(index + 1, len(rows))
                    encoder.stdin.close()
                except BrokenPipeError:
                    pass
                if encoder.wait() != 0:
                    errors.seek(0)
                    raise RuntimeError(errors.read().decode(errors="replace").strip() or "MP3 encoding failed.")
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        if not any(destination.iterdir()):
            destination.rmdir()
        log.exception("MP3 export failed for book %s", book_id)
        raise
    return output

