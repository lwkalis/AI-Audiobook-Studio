Attempting to perform the InitializeDefaultDrives operation on the 'FileSystem' provider failed.
import tempfile
from pathlib import Path
import threading
import time
import unittest

import numpy as np
import soundfile as sf

from studio.export import export_mp3, find_ffmpeg, preview_audio
from studio.library import Library
from studio.narration import NarrationWorker, SAMPLE_RATE
from studio.text import prepare_book, split_paragraph


ROOT = Path(__file__).resolve().parents[1]
TEXT = "Chapter I\n\nThe rain had stopped.\n\nChapter II\n\nThe sky began to clear."


class FakeNarrator:
    def __init__(self):
        self.calls = 0
        self.entered = threading.Event()
        self.unblock = threading.Event()
        self.fail_next = False

    def synthesize(self, text, seed):
        self.calls += 1
        self.entered.set()
        if not self.unblock.wait(4):
            raise TimeoutError("Test did not release the narrator.")
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("Transient synthesis failure")
        time_axis = np.arange(SAMPLE_RATE // 2) / SAMPLE_RATE
        return (0.2 * np.sin(2 * np.pi * 220 * time_axis)).astype(np.float32)

    def recover(self, error):
        return str(error)

    def release_unused_memory(self):
        pass


class StudioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / ".runtime")
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name)
        self.library = Library(self.data / "projects")
        title, author, chapters, _ = prepare_book(TEXT, "The Rain")
        self.book_id = self.library.add(title, author, chapters)

    def wait_for_state(self, state):
        for _ in range(250):
            if self.library.book(self.book_id)["state"] == state:
                return
            time.sleep(0.02)
        self.fail(f"Expected state {state}, got {dict(self.library.book(self.book_id))}")

    def test_import_wrapped_header_and_numbered_titles(self):
        text = (
            "Title: Collected Tales\nAuthor: An Author\n"
            "*** START OF THE PROJECT GUTENBERG EBOOK COLLECTED\nTALES ***\n\n"
            "Collected Tales\n\nContents\n   I.     The Rain\n   II.    The Sky\n\n"
            "I. THE RAIN\n\n“It's raining,” she said.\n\n"
            "II. THE SKY\n\nDr. Gray saw the sky.\n\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK COLLECTED TALES ***\nlicense"
        )
        title, author, chapters, trimmed = prepare_book(text, "Fallback")
        self.assertEqual((title, author, trimmed), ("Collected Tales", "An Author", True))
        self.assertEqual([c.title for c in chapters], ["Opening material", "I. THE RAIN", "II. THE SKY"])
        self.assertEqual(chapters[0].body, "Collected Tales")
        self.assertEqual(chapters[1].body, "“It's raining,” she said.")
        self.assertEqual(chapters[2].body, "Dr. Gray saw the sky.")
        long_text = "Dr. Gray saw the sky. " * 300
        chunks = split_paragraph(long_text)
        self.assertEqual(" ".join(chunks), long_text.strip())
        self.assertTrue(all(len(chunk.split()) <= 60 for chunk in chunks))

    def test_contents_keeps_real_opening_letters_and_ambiguous_lists(self):
        text = ("The Book\n\nCONTENTS\nLetter 1\nLetter 2\nChapter 1\n\n"
                "Letter 1\n\nThe story begins.\n\nLetter 2\n\nThe story continues.\n\n"
                "Chapter 1\n\nThe first chapter.")
        _, _, chapters, _ = prepare_book(text, "Test")
        self.assertEqual(chapters[0].body,
                         "The Book\n\nLetter 1\n\nThe story begins.\n\nLetter 2\n\nThe story continues.")
        self.assertEqual(chapters[1].body, "The first chapter.")
        ambiguous = "Contents\n\nChapter 1\n\nChapter 2\n\nThese entries do not repeat."
        _, _, preserved, _ = prepare_book(ambiguous, "Test")
        self.assertEqual(preserved[0].body, "Contents")

    def test_pause_and_resume_with_real_mp3_export(self):
        narrator = FakeNarrator()
        worker = NarrationWorker(self.library, narrator)
        self.addCleanup(worker.close)
        self.addCleanup(narrator.unblock.set)
        self.library.start(self.book_id)
        worker.wake.set()
        self.assertTrue(narrator.entered.wait(3))
        self.library.stop(self.book_id)
        narrator.unblock.set()
        self.wait_for_state("paused")
        self.assertEqual(self.library.book(self.book_id)["completed"], 1)
        rate, audio = preview_audio(self.library, self.book_id)
        self.assertEqual((rate, len(audio)), (SAMPLE_RATE, SAMPLE_RATE // 2))
        _, _, chapters, _ = prepare_book("A different book.", "Other book")
        other_book = self.library.add("Other book", "", chapters)
        with self.assertRaisesRegex(ValueError, "this book"):
            preview_audio(self.library, other_book)
        saved = self.library.wav_path(self.book_id, 1)
        first_timestamp = saved.stat().st_mtime_ns
        self.library.start(self.book_id)
        worker.wake.set()
        self.wait_for_state("complete")
        self.assertEqual(narrator.calls, 4)
        self.assertEqual(saved.stat().st_mtime_ns, first_timestamp)

        ffmpeg = find_ffmpeg(ROOT / ".runtime")
        if ffmpeg:
            for chapter in (None, 2):
                updates = []
                result = export_mp3(self.library, self.data / "exports", ffmpeg,
                                    self.book_id, chapter, 145, 256,
                                    on_progress=lambda done, total: updates.append((done, total)))
                self.assertEqual(sf.info(result).samplerate, 48000)
                self.assertGreater(sf.info(result).duration, 1)
                count = 2 if chapter else 4
                self.assertEqual(updates, [(done, count) for done in range(count + 1)])

    def test_narration_stats_and_prose_based_eta(self):
        text = "Chapter 1\n\n" + "\n\n".join(["A " * 24 + "line."] * 5)
        title, author, chapters, _ = prepare_book(text, "Timing test")
        book_id = self.library.add(title, author, chapters)
        self.library.save_segment(book_id, 1, 2, 120)
        for ordinal in range(2, 5):
            self.library.save_segment(book_id, ordinal, 10, 5)
        self.assertIsNone(self.library.estimated_remaining_seconds(self.library.book(book_id)))
        self.library.save_segment(book_id, 5, 10, 5)
        book = self.library.book(book_id)
        self.assertEqual((book["audio_seconds"], book["synthesis_seconds"]), (42, 140))
        self.assertEqual((book["completed_words"], book["unfinished_chapters"]), (102, 1))
        self.assertEqual(self.library.estimated_remaining_seconds(book), 5)

    def test_restart_and_orphan_checkpoint(self):
        self.library.start(self.book_id)
        checkpoint = self.library.wav_path(self.book_id, 1)
        checkpoint.parent.mkdir(parents=True)
        sf.write(checkpoint, np.ones(SAMPLE_RATE // 2) * 0.1, SAMPLE_RATE, subtype="FLOAT")
        self.library = Library(self.library.directory)
        self.assertEqual(self.library.book(self.book_id)["state"], "paused")
        narrator = FakeNarrator()
        narrator.unblock.set()
        worker = NarrationWorker(self.library, narrator)
        self.addCleanup(worker.close)
        self.library.start(self.book_id)
        worker.wake.set()
        self.wait_for_state("complete")
        self.assertEqual(narrator.calls, 3)

    def test_cancel_and_retry_on_resume(self):
        narrator = FakeNarrator()
        worker = NarrationWorker(self.library, narrator)
        self.addCleanup(worker.close)
        self.addCleanup(narrator.unblock.set)
        self.library.start(self.book_id)
        worker.wake.set()
        self.assertTrue(narrator.entered.wait(3))
        self.library.stop(self.book_id, cancel=True)
        narrator.unblock.set()
        self.wait_for_state("cancelled")
        self.assertEqual(self.library.book(self.book_id)["completed"], 1)

        narrator.fail_next = True
        self.library.start(self.book_id)
        worker.wake.set()
        self.wait_for_state("complete")
        self.assertEqual(narrator.calls, 5)


if __name__ == "__main__":
    unittest.main()

