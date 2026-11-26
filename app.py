Attempting to perform the InitializeDefaultDrives operation on the 'FileSystem' provider failed.
"""Launch the local audiobook studio."""

import argparse
from html import escape
import logging
import msvcrt
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / ".runtime"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ.setdefault("HF_HOME", str(RUNTIME / "hf"))
os.environ.setdefault("GRADIO_TEMP_DIR", str(RUNTIME / "gradio"))

import gradio as gr

from studio.export import BITRATES, DEFAULT_WPM, export_mp3, find_ffmpeg, preview_audio
from studio.library import Library
from studio.narration import NarrationWorker, QwenNarrator
from studio.text import prepare_book, read_txt


log = logging.getLogger(__name__)
STATUS = {
    "ready": "Ready", "running": "Generating", "pausing": "Pausing after this segment",
    "paused": "Paused", "cancelling": "Cancelling after this segment",
    "cancelled": "Cancelled", "complete": "Complete", "error": "Needs attention",
}
CSS = """
.studio-header {padding: 1.15rem 1.35rem; border-left: 4px solid #c39242; margin-bottom: .5rem}
.studio-header h1 {margin: 0; letter-spacing: -.035em}
.studio-header p {margin: .25rem 0 0; opacity: .78}
.book-status {padding: .9rem 1rem; border: 1px solid var(--border-color-primary); border-radius: 12px}
.book-status progress {width: 100%; accent-color: #b38236; margin: .45rem 0}
.book-status p {margin: .35rem 0 0; overflow-wrap: anywhere}
.narration-stats {display: grid; grid-template-columns: repeat(auto-fit, minmax(12rem, 1fr)); gap: .6rem}
.narration-stats div {padding: .75rem; border: 1px solid var(--border-color-primary); border-radius: 10px}
.narration-stats dt {font-size: .85em; opacity: .78}
.narration-stats dd {margin: .25rem 0 0; font-weight: 600; font-variant-numeric: tabular-nums}
"""


def format_duration(seconds):
    hours, rest = divmod(round(seconds), 3600)
    minutes, seconds = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes}m {seconds:02d}s" if minutes else f"{seconds}s"


def create_ui(library, worker, export_root, ffmpeg):
    def choices():
        return [(row["title"] + (f" — {row['author']}" if row["author"] else ""), row["id"])
                for row in library.books()]

    def status(book):
        if not book:
            return '<div class="book-status">Import a TXT file or paste book text to begin.</div>'
        completed = book["completed"]
        total = book["total_segments"]
        author = f" · {escape(book['author'])}" if book["author"] else ""
        message = f"<p>{escape(book['message'])}</p>" if book["message"] else ""
        return (
            f'<div class="book-status"><strong>{escape(book["title"])}</strong>{author}'
            f' · {book["chapter_count"]:,} chapter(s) · {book["total_words"]:,} words'
            f'<p>{STATUS[book["state"]]} · {completed:,}/{total:,} segments</p>'
            f'<progress value="{completed}" max="{total}" aria-label="Narration progress"></progress>'
            f'{message}</div>'
        )

    def narration_stats(book):
        if not book:
            return ""
        elapsed = book["synthesis_seconds"]
        audio = book["audio_seconds"]
        remaining = library.estimated_remaining_seconds(book)
        total_words = book["total_words"]
        chapter_count = book["chapter_count"]
        metrics = (
            ("Generation time recorded", format_duration(elapsed)),
            ("Saved audio before export", format_duration(audio)),
            ("Average generation speed", f"{audio / elapsed:.2f}× real time" if elapsed else "—"),
            ("Estimated generation remaining",
             format_duration(remaining) if remaining is not None else "Waiting for prose samples"),
            ("Words narrated", f"{book['completed_words']:,} / {total_words:,}"),
            ("Chapters complete", f"{chapter_count - book['unfinished_chapters']:,} / {chapter_count:,}"),
            (f"Estimated audiobook at {DEFAULT_WPM} WPM", format_duration(total_words * 60 / DEFAULT_WPM)),
        )
        cards = "".join(f"<div><dt>{label}</dt><dd>{value}</dd></div>" for label, value in metrics)
        return f'<dl class="narration-stats" aria-label="Narration statistics">{cards}</dl>'

    def refresh(book_id):
        book = library.book(book_id)
        return status(book), narration_stats(book)

    def select_book(book_id):
        chapters = [(row["title"], row["number"]) for row in library.chapters(book_id)]
        first = chapters[0][1] if chapters else None
        return (*refresh(book_id), gr.update(choices=chapters, value=first), preview(book_id, first),
                gr.update(choices=[("Full audiobook", 0), *chapters], value=0), None, None)

    def preview(book_id, number):
        chapter = library.chapter(book_id, number)
        return chapter["text"] if chapter else ""

    def import_book(upload, pasted, title, author):
        try:
            pasted = pasted or ""
            if upload and pasted.strip():
                raise ValueError("Choose a TXT file or paste text. Clear one of the inputs.")
            if upload and Path(upload).suffix.lower() != ".txt":
                raise ValueError("Choose a local TXT file.")
            text = read_txt(upload) if upload else pasted
            fallback = Path(upload).stem if upload else "Untitled book"
            detected_title, detected_author, chapters, trimmed = prepare_book(text, fallback)
            book_id = library.add(
                (title or "").strip() or detected_title, (author or "").strip() or detected_author, chapters
            )
        except (ValueError, OSError) as error:
            raise gr.Error(str(error)) from error
        message = "Book imported."
        if trimmed:
            message += " Project Gutenberg wrapper removed at its START and END markers."
        gr.Info(message)
        return gr.update(choices=choices(), value=book_id)

    def control(book_id, action):
        try:
            if action == "start":
                library.start(book_id)
            else:
                library.stop(book_id, cancel=action == "cancel")
            worker.wake.set()
            return status(library.book(book_id))
        except ValueError as error:
            raise gr.Error(str(error)) from error

    def export(book_id, selection, pace, bitrate, progress=gr.Progress()):
        def report(done, total):
            description = "Finalizing MP3" if done == total else f"Encoding MP3 · {done:,} / {total:,} segments"
            progress((done, total + 1), desc=description, unit="steps")

        progress(0, desc="Preparing MP3 export")
        try:
            output = export_mp3(
                library, export_root, ffmpeg, book_id, selection or None, pace, bitrate, on_progress=report,
            )
        except (ValueError, RuntimeError, OSError) as error:
            raise gr.Error(str(error)) from error
        return str(output)

    def listen(book_id):
        try:
            return preview_audio(library, book_id)
        except (ValueError, RuntimeError, OSError) as error:
            raise gr.Error(str(error)) from error

    saved = choices()
    initial = saved[0][1] if saved else None
    with gr.Blocks(title="AI Audiobook Studio") as ui:
        gr.HTML('<header class="studio-header"><h1>AI Audiobook Studio</h1>'
                '<p>Local audiobook production with Qwen3-TTS</p></header>')
        project = gr.Dropdown(choices=saved, value=initial, label="Saved book")
        initial_overview, initial_stats = refresh(initial)
        overview = gr.HTML(initial_overview)
        with gr.Tab("Library"):
            with gr.Row():
                with gr.Column():
                    upload = gr.File(label="Local TXT edition", file_types=[".txt"], type="filepath")
                    pasted = gr.Textbox(label="Or paste book text", lines=5)
                    with gr.Row():
                        title = gr.Textbox(label="Title", placeholder="Optional override")
                        author = gr.Textbox(label="Author", placeholder="Optional override")
                    add = gr.Button("Import book", variant="primary")
                with gr.Column():
                    chapter = gr.Dropdown(label="Chapter")
                    excerpt = gr.Textbox(label="Extracted text", lines=13, interactive=False)
        with gr.Tab("Narration"):
            stats = gr.HTML(initial_stats)
            gr.Markdown("Pause and Cancel finish the current segment. Start or resume continues from saved audio.")
            with gr.Row():
                start = gr.Button("Start or resume", variant="primary")
                pause = gr.Button("Pause")
                cancel = gr.Button("Cancel")
            load_audio = gr.Button("Load latest book audio")
            book_audio = gr.Audio(label="Saved book narration before export pacing", interactive=False)
        with gr.Tab("Export"):
            gr.Markdown("Verify that the book is public domain in your jurisdiction or that you have permission before publishing.")
            scope = gr.Dropdown(choices=[("Full audiobook", 0)], value=0, label="Export scope")
            with gr.Row():
                pace = gr.Slider(110, 200, value=DEFAULT_WPM, step=1, label="Target pace (words per minute)")
                bitrate = gr.Dropdown(choices=list(BITRATES), value=256, label="MP3 bitrate (kbps)")
            render = gr.Button("Export MP3", variant="primary", interactive=bool(ffmpeg))
            if not ffmpeg:
                gr.Markdown("FFmpeg is required. See the README for setup.")
            download = gr.File(label="Download MP3", interactive=False)

        outputs = [overview, stats, chapter, excerpt, scope, download, book_audio]
        ui.load(select_book, project, outputs, queue=False)
        project.change(select_book, project, outputs, queue=False)
        chapter.change(preview, [project, chapter], excerpt, queue=False)
        add.click(import_book, [upload, pasted, title, author], project, api_name="import_book")
        start.click(lambda book_id: control(book_id, "start"), project, overview, queue=False,
                    api_name="start_narration")
        pause.click(lambda book_id: control(book_id, "pause"), project, overview, queue=False,
                    api_name="pause_narration")
        cancel.click(lambda book_id: control(book_id, "cancel"), project, overview, queue=False,
                     api_name="cancel_narration")
        load_audio.click(listen, project, book_audio, api_name="preview_narration")
        render.click(export, [project, scope, pace, bitrate], download, concurrency_id="mp3",
                     show_progress="full", show_progress_on=download, api_name="export_mp3")
        gr.Timer(3).tick(refresh, project, [overview, stats], queue=False, show_progress="hidden")
    return ui


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "library")
    parser.add_argument("--standard-gpu", action="store_true", help="Use standard Qwen CUDA inference.")
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / "studio.lock").open("a+b") as lock:
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise SystemExit("AI Audiobook Studio is already using this library.")
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s",
            handlers=[logging.FileHandler(data_dir / "studio.log", encoding="utf-8"), logging.StreamHandler()],
        )
        library = Library(data_dir / "projects")
        export_root = data_dir / "exports"
        export_root.mkdir(exist_ok=True)
        graph_package = RUNTIME / "qwen-graphs"
        use_graphs = not args.standard_gpu and (graph_package / "faster_qwen3_tts" / "__init__.py").is_file()
        worker = NarrationWorker(
            library, QwenNarrator(RUNTIME / "models" / "qwen", graph_package if use_graphs else None),
        )
        try:
            ui = create_ui(library, worker, export_root, find_ffmpeg(RUNTIME))
            ui.launch(
                server_name="127.0.0.1", server_port=args.port, share=False,
                inbrowser=not args.no_browser, show_error=True, css=CSS,
                allowed_paths=[str(export_root)],
                blocked_paths=[str(library.directory), str(RUNTIME / "models")],
            )
        finally:
            worker.close()


if __name__ == "__main__":
    main()

