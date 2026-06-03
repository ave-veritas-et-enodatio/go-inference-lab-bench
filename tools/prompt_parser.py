#!/usr/bin/env python3
"""@FILENAME substitution for test_inference prompts.

INVARIANT: stdlib only. No third-party dependencies.

Parses prompt strings, replacing `@FILENAME` references with content parts
(image_url with a base64 data URI for image extensions; inline text for
everything else). Quote-aware: `@` inside ', ", `, or ``` quotes is left
literal. The `\\@` escape produces a literal `@` (the backslash is
consumed). Two reference shapes:

    @path/no/spaces.png         # terminates at whitespace
    @"path with spaces.png"     # quoted filename (single or double quotes)
    @'path with spaces.png'

Returns either:
  * the original string unchanged (no references found), or
  * a list of typed content parts ready to use as OpenAI-multimodal
    `messages[].content`.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

# Image extensions trigger the image_url branch; everything else is
# inlined as text. Keeping this small + explicit avoids inlining a 2GB
# binary file because someone forgot to quote a path.
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

# Per-file size cap. Matches apiserver/content.go's data-URI limit so
# requests we build here can't be rejected later for size reasons.
MAX_FILE_BYTES = 32 * 1024 * 1024  # 32 MiB

# Trailing-punctuation strip set for unquoted @FILENAME refs. These are
# common end-of-clause / end-of-list / end-of-parenthetical markers that
# users naturally type adjacent to a path — `summarize @README.md, please`
# should resolve `README.md`, not `README.md,`. Use `@"..."` to opt out.
# `.` is intentionally NOT in this set: dots are load-bearing in filenames
# (extensions, multi-suffix archives) and a trailing `.` is more likely
# to be a path component than sentence punctuation in this codebase.
_TRAILING_PUNCT = frozenset(",;:!?)]}")


class PromptParseError(ValueError):
    """Raised when the prompt or a referenced file can't be processed."""


class _FileRef:
    __slots__ = ("path", "offset")

    def __init__(self, path: str, offset: int):
        self.path = path
        self.offset = offset  # position in original prompt — useful for error messages


def parse_prompt_content(prompt: str, base_dir: Path | None = None) -> str | list[dict]:
    """Parse `prompt`, returning either the unchanged string (no @refs) or
    a typed-parts list with text + image content.

    `base_dir` is the directory relative paths resolve against. None means
    use the process cwd at call time.
    """
    if base_dir is None:
        base_dir = Path.cwd()

    parts = _split_prompt(prompt)
    if not any(isinstance(p, _FileRef) for p in parts):
        # All-text: join the (possibly escape-substituted) fragments.
        # If there were no escapes either, this yields the original
        # prompt unchanged.
        return "".join(parts) if parts else ""

    out: list[dict] = []
    text_buf: list[str] = []

    def flush_text() -> None:
        if text_buf:
            joined = "".join(text_buf)
            if joined:  # drop pure-whitespace text parts between adjacent refs? no — preserve
                out.append({"type": "text", "text": joined})
            text_buf.clear()

    for part in parts:
        if isinstance(part, str):
            text_buf.append(part)
        else:
            flush_text()
            out.append(_resolve_file_ref(part, base_dir))
    flush_text()
    return out


def _split_prompt(prompt: str) -> list:
    """State-machine scan over `prompt`. Yields alternating str fragments
    and _FileRef objects.

    States:
      NORMAL — outside any quote, @refs and quote-openers are honored.
      SQUOTE / DQUOTE / BACKTICK — single-char quote, closes on same char.
      TRIPLE_BACKTICK — closes on the next ``` (literal triple).

    Inside any quote state, characters are passed through unchanged
    (including @ and \\@). Backslash escapes are NOT processed inside
    quotes — quoted content is meant to be transmitted verbatim.
    """
    out: list = []
    buf: list[str] = []

    i = 0
    n = len(prompt)
    quote_state: str | None = None  # one of: None, "'", '"', "`", "```"

    def flush_text() -> None:
        if buf:
            out.append("".join(buf))
            buf.clear()

    while i < n:
        c = prompt[i]

        # Inside a quoted region: only the matching close terminates.
        if quote_state is not None:
            if quote_state == "```":
                if prompt[i : i + 3] == "```":
                    buf.append("```")
                    i += 3
                    quote_state = None
                    continue
            elif c == quote_state:
                buf.append(c)
                i += 1
                quote_state = None
                continue
            buf.append(c)
            i += 1
            continue

        # NORMAL state. Order matters: backslash-escape, quote openers,
        # @ references, default copy.

        # \@ → literal @, consume both characters.
        if c == "\\" and i + 1 < n and prompt[i + 1] == "@":
            buf.append("@")
            i += 2
            continue

        # Quote openers (longest first: ``` before `).
        if prompt[i : i + 3] == "```":
            buf.append("```")
            i += 3
            quote_state = "```"
            continue
        if c == "'" or c == '"' or c == "`":
            buf.append(c)
            i += 1
            quote_state = c
            continue

        # @FILENAME reference — only at start-of-string or after whitespace.
        if c == "@" and (i == 0 or prompt[i - 1].isspace()):
            # @"..." / @'...' — quoted filename (delimits spaces).
            if i + 1 < n and prompt[i + 1] in ("'", '"'):
                qchar = prompt[i + 1]
                end = prompt.find(qchar, i + 2)
                if end < 0:
                    raise PromptParseError(
                        f"unterminated quoted filename at offset {i}"
                    )
                fname = prompt[i + 2 : end]
                if not fname:
                    raise PromptParseError(
                        f"empty quoted filename at offset {i}"
                    )
                flush_text()
                out.append(_FileRef(fname, i))
                i = end + 1
                continue

            # @FILENAME — terminates at whitespace or end-of-string,
            # with trailing punctuation walked back into the following
            # text fragment.
            end = i + 1
            while end < n and not prompt[end].isspace():
                end += 1
            while end > i + 1 and prompt[end - 1] in _TRAILING_PUNCT:
                end -= 1
            fname = prompt[i + 1 : end]
            if not fname:
                # Bare @ with nothing after (or @ followed only by
                # stripped punct) — treat as literal text.
                buf.append("@")
                i += 1
                continue
            flush_text()
            out.append(_FileRef(fname, i))
            i = end
            continue

        # Default: copy char.
        buf.append(c)
        i += 1

    if quote_state is not None:
        # Unclosed quote — render the whole thing as text rather than fail.
        # The user might be legitimately mid-quote, and a hard error here
        # would be worse than a slightly weird-looking prompt.
        pass

    flush_text()
    return out


def _resolve_file_ref(ref: _FileRef, base_dir: Path) -> dict:
    """Read the referenced file and produce a content part dict.

    Image extension → image_url part with base64 data: URI.
    Other extension → text part (UTF-8 file contents inlined).
    """
    path_str = ref.path
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = (base_dir / p).resolve()

    if not p.exists():
        raise PromptParseError(f"@{path_str}: file not found")
    if not p.is_file():
        raise PromptParseError(f"@{path_str}: not a regular file")

    size = p.stat().st_size
    if size > MAX_FILE_BYTES:
        raise PromptParseError(
            f"@{path_str}: file size {size} exceeds {MAX_FILE_BYTES}-byte limit"
        )

    ext = p.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        mime, _ = mimetypes.guess_type(str(p))
        # mimetypes.guess_type is deterministic for these extensions on
        # macOS / Linux; pin the result rather than trust it for the
        # data: URI's MIME field.
        ext_to_mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime = ext_to_mime[ext]
        data = p.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
        }

    if not ext:
        raise PromptParseError(
            f"@{path_str}: file has no extension; expected an image "
            f"({'/'.join(sorted(IMAGE_EXTENSIONS))}) or a UTF-8 text file"
        )

    # Non-image: inline as text. Read UTF-8; non-decodable means binary,
    # which is almost certainly user error (forgot to quote a path).
    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise PromptParseError(
            f"@{path_str}: file is not valid UTF-8 (extension {ext} not "
            f"recognized as image): {e}"
        ) from e
    return {"type": "text", "text": text}
