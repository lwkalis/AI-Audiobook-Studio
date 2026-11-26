Attempting to perform the InitializeDefaultDrives operation on the 'FileSystem' provider failed.
"""Read a local text edition and preserve its wording in narration segments."""

from dataclasses import dataclass
from pathlib import Path
import re


HEADING = re.compile(
    r"^(?:(?:chapter|part|book)\s+(?:[ivxlcdm]+|\d+|"
    r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)(?:\b.*)?|"
    r"(?:prologue|epilogue)(?:\s*[:.—–-].*)?)$",
    re.IGNORECASE,
)
NUMBERED_TITLE = re.compile(r"^(?:[IVXLCDM]+|\d+)\.\s+.+$")
LETTER = re.compile(r"^letter\s+(?:\d+|[ivxlcdm]+)$", re.IGNORECASE)
SENTENCE = re.compile(r"""[.!?][”’"')\]]*\s+""")
ABBREVIATION = re.compile(r"\b(?:mr|mrs|ms|dr|st|prof|rev|etc|[a-z])\.$", re.I)
MARKER = r"(?im)^\*\*\*[ \t]*{} OF (?:THE|THIS) PROJECT GUTENBERG EBOOK(?:[^\n]*\n){{0,2}}?[^\n]*?\*\*\*[ \t]*$"
HEADER = re.compile(MARKER.format("START"))
FOOTER = re.compile(MARKER.format("END"))
MAX_WORDS = 60
MAX_CHARS = 420


@dataclass
class Chapter:
    title: str
    heading: str
    body: str

    def segments(self):
        if self.heading:
            yield self.heading, True
        for paragraph in re.split(r"\n[ \t]*\n+", self.body.strip()):
            chunks = split_paragraph(paragraph)
            for index, chunk in enumerate(chunks):
                yield chunk, index == len(chunks) - 1


def read_txt(path: str) -> str:
    raw = Path(path).read_bytes()
    if b"\0" in raw and not raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        raise ValueError("Choose a plain-text TXT file encoded as UTF-8, UTF-16 with a BOM, or Windows-1252.")
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252")


def prepare_book(text: str, fallback_title: str):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    start = HEADER.search(text)
    end = FOOTER.search(text, start.end()) if start else None
    title = author = ""
    if start and end:
        front = text[:start.start()]
        title_match = re.search(r"(?im)^Title:[ \t]*(.+)$", front)
        author_match = re.search(r"(?im)^Author:[ \t]*(.+)$", front)
        title = title_match.group(1).strip() if title_match else ""
        author = author_match.group(1).strip() if author_match else ""
        text = text[start.end():end.start()]
    text = strip_repeated_contents(text)
    if not text.strip():
        raise ValueError("Enter or upload some book text.")

    chapters = []
    heading = ""
    lines = []
    source_lines = text.strip().splitlines()
    for index, line in enumerate(source_lines):
        candidate = line.strip()
        isolated_title = (
            candidate.isupper() and NUMBERED_TITLE.fullmatch(candidate)
            and (index == 0 or not source_lines[index - 1].strip())
            and (index == len(source_lines) - 1 or not source_lines[index + 1].strip())
        )
        if len(candidate) <= 100 and (HEADING.fullmatch(candidate) or isolated_title):
            body = "\n".join(lines).strip()
            if heading or body:
                chapters.append(Chapter(heading or "Opening material", heading, body))
            heading = candidate
            lines = []
        else:
            lines.append(line)
    body = "\n".join(lines).strip()
    chapters.append(Chapter(heading or ("Full text" if not chapters else "Untitled chapter"), heading, body))
    return title or fallback_title, author, chapters, bool(start and end)


def strip_repeated_contents(text: str) -> str:
    """Drop a front-matter contents list when its headings repeat in the body."""
    lines = text.split("\n")
    labels = [" ".join(line.split()).casefold() for line in lines]
    start = next((i for i, label in enumerate(labels[:120])
                  if label in ("contents", "table of contents")), None)
    if start is None:
        return text
    entries = set()
    for end in range(start + 1, len(lines)):
        label = labels[end]
        if not label:
            continue
        line = lines[end].strip()
        if label in entries or not (
            HEADING.fullmatch(line) or LETTER.fullmatch(line) or NUMBERED_TITLE.fullmatch(line)
        ):
            break
        entries.add(label)
    else:
        return text
    if len(entries) >= 2 and entries.issubset(labels[end:]):
        return "\n".join(lines[:start] + lines[end:])
    return text


def split_paragraph(paragraph: str) -> list[str]:
    text = " ".join(paragraph.split())
    if not text:
        return []
    ends = {match.end() for match in SENTENCE.finditer(text)
            if not ABBREVIATION.search(text[max(0, match.start() - 12):match.start() + 1])}
    words = list(re.finditer(r"\S+\s*", text))
    chunks = []
    start = 0
    while start < len(words):
        limit = min(start + MAX_WORDS, len(words))
        while limit > start + 1 and words[limit - 1].end() - words[start].start() > MAX_CHARS:
            limit -= 1
        if limit < len(words):
            preferred = next((i for i in range(limit, start, -1) if words[i - 1].end() in ends), limit)
            if preferred > start + 5:
                limit = preferred
        chunks.append(text[words[start].start():words[limit - 1].end()].strip())
        start = limit
    return chunks

