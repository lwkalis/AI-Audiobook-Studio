Attempting to perform the InitializeDefaultDrives operation on the 'FileSystem' provider failed.
"""Persist book text and completed WAV checkpoints in SQLite."""

from contextlib import contextmanager
from pathlib import Path
import sqlite3
from statistics import median


ACTIVE = ("running", "pausing", "cancelling")


class Library:
    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.database = directory / "library.sqlite3"
        with self.connect() as db:
            db.execute("PRAGMA journal_mode = WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS books (
                    id INTEGER PRIMARY KEY,
                    title TEXT NOT NULL,
                    author TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'ready',
                    message TEXT NOT NULL DEFAULT '',
                    total_segments INTEGER NOT NULL,
                    total_words INTEGER NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS one_active_book ON books ((1))
                    WHERE state IN ('running', 'pausing', 'cancelling');
                CREATE TABLE IF NOT EXISTS chapters (
                    book_id INTEGER NOT NULL REFERENCES books(id),
                    number INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    text TEXT NOT NULL,
                    PRIMARY KEY (book_id, number)
                );
                CREATE TABLE IF NOT EXISTS segments (
                    book_id INTEGER NOT NULL,
                    ordinal INTEGER NOT NULL,
                    chapter INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    words INTEGER NOT NULL,
                    paragraph_end INTEGER NOT NULL,
                    audio_seconds REAL,
                    synthesis_seconds REAL,
                    PRIMARY KEY (book_id, ordinal),
                    FOREIGN KEY (book_id, chapter) REFERENCES chapters(book_id, number)
                );
                CREATE INDEX IF NOT EXISTS pending_segments ON segments (book_id, ordinal)
                    WHERE audio_seconds IS NULL;
            """)
            db.execute("""
                UPDATE books SET state = CASE state WHEN 'cancelling' THEN 'cancelled' ELSE 'paused' END,
                    message = 'Session ended. Saved segments are available.'
                WHERE state IN ('running', 'pausing', 'cancelling')
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.database, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def add(self, title, author, chapters):
        with self.connect() as db:
            book_id = db.execute(
                "INSERT INTO books (title, author, total_segments, total_words) VALUES (?, ?, 0, 0)",
                (title, author),
            ).lastrowid
            ordinal = total_words = 0
            for chapter_number, chapter in enumerate(chapters, 1):
                chapter_text = "\n\n".join(filter(None, (chapter.heading, chapter.body)))
                db.execute("INSERT INTO chapters VALUES (?, ?, ?, ?)",
                           (book_id, chapter_number, chapter.title, chapter_text))
                for text, paragraph_end in chapter.segments():
                    ordinal += 1
                    words = len(text.split())
                    total_words += words
                    db.execute("""
                        INSERT INTO segments (book_id, ordinal, chapter, text, words, paragraph_end)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (book_id, ordinal, chapter_number, text, words, paragraph_end))
            if not ordinal:
                raise ValueError("The imported text contains no narration.")
            db.execute("UPDATE books SET total_segments = ?, total_words = ? WHERE id = ?",
                       (ordinal, total_words, book_id))
            return book_id

    def books(self):
        with self.connect() as db:
            return db.execute("SELECT id, title, author, state FROM books ORDER BY id DESC").fetchall()

    def book(self, book_id):
        with self.connect() as db:
            return db.execute("""
                SELECT books.*, COUNT(segments.audio_seconds) AS completed,
                    COALESCE(SUM(CASE WHEN audio_seconds IS NOT NULL THEN words ELSE 0 END), 0)
                        AS completed_words,
                    COALESCE(SUM(synthesis_seconds), 0) AS synthesis_seconds,
                    COALESCE(SUM(audio_seconds), 0) AS audio_seconds,
                    COUNT(DISTINCT CASE WHEN segments.audio_seconds IS NULL THEN segments.chapter END)
                        AS unfinished_chapters,
                    (SELECT COUNT(*) FROM chapters WHERE book_id = books.id) AS chapter_count
                FROM books LEFT JOIN segments ON segments.book_id = books.id
                WHERE books.id = ? GROUP BY books.id
            """, (book_id,)).fetchone()

    def estimated_remaining_seconds(self, book):
        remaining = book["total_words"] - book["completed_words"]
        if not remaining:
            return 0
        with self.connect() as db:
            recent = db.execute("""
                SELECT words, synthesis_seconds FROM segments
                WHERE book_id = ? AND words >= 20 AND audio_seconds IS NOT NULL
                    AND synthesis_seconds > 0
                ORDER BY ordinal DESC LIMIT 20
            """, (book["id"],)).fetchall()
        if len(recent) < 4 or sum(row["words"] for row in recent) < 100:
            return None
        seconds_per_word = median(row["synthesis_seconds"] / row["words"] for row in recent)
        return remaining * seconds_per_word

    def chapters(self, book_id):
        with self.connect() as db:
            return db.execute("SELECT number, title FROM chapters WHERE book_id = ? ORDER BY number",
                              (book_id,)).fetchall()

    def chapter(self, book_id, number):
        with self.connect() as db:
            return db.execute("SELECT * FROM chapters WHERE book_id = ? AND number = ?",
                              (book_id, number)).fetchone()

    def start(self, book_id):
        try:
            with self.connect() as db:
                result = db.execute("""
                    UPDATE books SET state = 'running', message = ''
                    WHERE id = ? AND state IN ('ready', 'paused', 'cancelled', 'error')
                """, (book_id,))
                if not result.rowcount:
                    raise ValueError("Select a book that is ready, paused, cancelled, or has an error.")
        except sqlite3.IntegrityError as error:
            raise ValueError("Pause the active narration before starting another book.") from error

    def stop(self, book_id, cancel=False):
        state = "cancelling" if cancel else "pausing"
        with self.connect() as db:
            if not db.execute("UPDATE books SET state = ? WHERE id = ? AND state = 'running'",
                              (state, book_id)).rowcount:
                raise ValueError("Select the book currently generating audio.")

    def active(self):
        with self.connect() as db:
            return db.execute("""
                SELECT id, state FROM books WHERE state IN ('running', 'pausing', 'cancelling')
            """).fetchone()

    def pending(self, book_id):
        with self.connect() as db:
            return db.execute("""
                SELECT * FROM segments WHERE book_id = ? AND audio_seconds IS NULL
                ORDER BY ordinal LIMIT 1
            """, (book_id,)).fetchone()

    def save_segment(self, book_id, ordinal, audio_seconds, synthesis_seconds):
        with self.connect() as db:
            db.execute("""
                UPDATE segments SET audio_seconds = ?, synthesis_seconds = ?
                WHERE book_id = ? AND ordinal = ?
            """, (audio_seconds, synthesis_seconds, book_id, ordinal))

    def finish(self, book_id, expected, state, message=""):
        with self.connect() as db:
            db.execute("UPDATE books SET state = ?, message = ? WHERE id = ? AND state = ?",
                       (state, message, book_id, expected))

    def report_error(self, book_id, message):
        with self.connect() as db:
            db.execute("UPDATE books SET state = 'error', message = ? WHERE id = ?",
                       (message, book_id))

    def wav_path(self, book_id, ordinal):
        return self.directory / str(book_id) / "segments" / f"{ordinal:07d}.wav"

    def latest_audio(self, book_id, limit=4):
        with self.connect() as db:
            rows = db.execute("""
                SELECT ordinal, chapter, paragraph_end FROM segments
                WHERE book_id = ? AND audio_seconds IS NOT NULL
                ORDER BY ordinal DESC LIMIT ?
            """, (book_id, limit)).fetchall()
        return rows[::-1]

    def export_segments(self, book_id, chapter=None):
        with self.connect() as db:
            if chapter is None:
                return db.execute("""
                    SELECT ordinal, chapter, words, paragraph_end, audio_seconds
                    FROM segments WHERE book_id = ? ORDER BY ordinal
                """, (book_id,)).fetchall()
            return db.execute("""
                SELECT ordinal, chapter, words, paragraph_end, audio_seconds
                FROM segments WHERE book_id = ? AND chapter = ? ORDER BY ordinal
            """, (book_id, chapter)).fetchall()

