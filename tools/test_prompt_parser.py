#!/usr/bin/env python3
"""Unit tests for tools/prompt_parser.py — stdlib unittest only.

Run via:
    python3 tools/test_prompt_parser.py
or:
    python3 -m unittest tools.test_prompt_parser
"""

from __future__ import annotations

import base64
import sys
import tempfile
import unittest
from pathlib import Path

# Allow running both `python3 tools/test_prompt_parser.py` and
# `python3 -m unittest tools.test_prompt_parser`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prompt_parser import (  # noqa: E402
    parse_prompt_content,
    PromptParseError,
)


def _write_file(dir: Path, name: str, content: bytes | str) -> Path:
    p = dir / name
    if isinstance(content, str):
        p.write_text(content, encoding="utf-8")
    else:
        p.write_bytes(content)
    return p


# A minimal 1×1 PNG. Smallest valid PNG; enough to exercise the
# image-MIME branch without needing PIL or a fixture file.
_MIN_PNG = bytes.fromhex(
    "89504e470d0a1a0a"          # PNG signature
    "0000000d49484452"          # IHDR chunk header (13 bytes)
    "00000001000000010802000000" # 1×1 RGB, no interlace
    "907753de"                  # IHDR CRC
    "0000000c4944415408d763f8cfc0c00000030001fbfbbecf00000000"  # IDAT
    "0000000049454e44ae426082"  # IEND
)


class TextOnlyTests(unittest.TestCase):
    """No @ refs → return the original string unchanged."""

    def test_plain_text(self):
        self.assertEqual(parse_prompt_content("hello world"), "hello world")

    def test_empty(self):
        self.assertEqual(parse_prompt_content(""), "")

    def test_at_in_email_not_a_ref(self):
        # @ preceded by a non-whitespace character → not a reference.
        self.assertEqual(
            parse_prompt_content("send to alice@example.com please"),
            "send to alice@example.com please",
        )

    def test_bare_at_with_no_filename(self):
        # `@` followed by whitespace → literal @, not a ref.
        self.assertEqual(parse_prompt_content("look @ this"), "look @ this")

    def test_escaped_at(self):
        # \@ → literal @, backslash consumed.
        self.assertEqual(parse_prompt_content("price is \\@$5"), "price is @$5")

    def test_at_inside_single_quotes(self):
        s = "a 'string with @notafile.png inside' rest"
        self.assertEqual(parse_prompt_content(s), s)

    def test_at_inside_double_quotes(self):
        s = 'a "string with @notafile.png inside" rest'
        self.assertEqual(parse_prompt_content(s), s)

    def test_at_inside_backticks(self):
        s = "code `@notafile.png` rest"
        self.assertEqual(parse_prompt_content(s), s)

    def test_at_inside_triple_backticks(self):
        s = "```\n@notafile.png\n``` text"
        self.assertEqual(parse_prompt_content(s), s)

    def test_nested_quotes_passthrough(self):
        # The first " enters DQUOTE, the inner ' is literal (not a state
        # toggle), the closing " exits DQUOTE. @notafile.png stays in the
        # DQUOTE region throughout, so it's literal.
        s = '"he said \'@notafile.png\' loudly"'
        self.assertEqual(parse_prompt_content(s), s)


class ImageRefTests(unittest.TestCase):
    """@FILENAME refs that resolve to image content parts."""

    def test_single_image_ref(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            _write_file(base, "pic.png", _MIN_PNG)
            result = parse_prompt_content("describe @pic.png briefly", base_dir=base)
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 3, msg=f"parts: {result}")
            self.assertEqual(result[0], {"type": "text", "text": "describe "})
            self.assertEqual(result[1]["type"], "image_url")
            self.assertTrue(
                result[1]["image_url"]["url"].startswith("data:image/png;base64,")
            )
            # The base64 payload should round-trip to the original PNG bytes.
            b64 = result[1]["image_url"]["url"].split(",", 1)[1]
            self.assertEqual(base64.b64decode(b64), _MIN_PNG)
            self.assertEqual(result[2], {"type": "text", "text": " briefly"})

    def test_image_ref_at_end_of_prompt(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            _write_file(base, "pic.jpg", b"\xff\xd8\xff\xd9")  # minimal JPEG-ish
            result = parse_prompt_content("describe @pic.jpg", base_dir=base)
            self.assertIsInstance(result, list)
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["text"], "describe ")
            self.assertTrue(result[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_image_ref_at_start_of_prompt(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            _write_file(base, "pic.png", _MIN_PNG)
            result = parse_prompt_content("@pic.png describe", base_dir=base)
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["type"], "image_url")
            self.assertEqual(result[1], {"type": "text", "text": " describe"})

    def test_multiple_image_refs(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            _write_file(base, "a.png", _MIN_PNG)
            _write_file(base, "b.png", _MIN_PNG)
            result = parse_prompt_content("compare @a.png to @b.png", base_dir=base)
            self.assertIsInstance(result, list)
            image_parts = [p for p in result if p["type"] == "image_url"]
            self.assertEqual(len(image_parts), 2)

    def test_quoted_filename_with_spaces(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            _write_file(base, "with space.png", _MIN_PNG)
            result = parse_prompt_content('look at @"with space.png"', base_dir=base)
            self.assertIsInstance(result, list)
            self.assertEqual(result[0]["text"], "look at ")
            self.assertEqual(result[1]["type"], "image_url")

    def test_quoted_filename_single_quotes(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            _write_file(base, "with space.png", _MIN_PNG)
            result = parse_prompt_content("look at @'with space.png'", base_dir=base)
            self.assertIsInstance(result, list)
            self.assertEqual(result[1]["type"], "image_url")


class TextFileRefTests(unittest.TestCase):
    """@FILENAME refs that resolve to text content parts (non-image extensions)."""

    def test_text_file_inlined(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            _write_file(base, "notes.md", "# title\nbody line\n")
            result = parse_prompt_content("read this: @notes.md", base_dir=base)
            self.assertIsInstance(result, list)
            text_parts = [p["text"] for p in result if p["type"] == "text"]
            self.assertIn("read this: ", text_parts)
            self.assertIn("# title\nbody line\n", text_parts)

    def test_non_utf8_text_file_errors(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            _write_file(base, "bad.txt", b"\xff\xfe\x00\xc3\x28")
            with self.assertRaises(PromptParseError) as ctx:
                parse_prompt_content("@bad.txt", base_dir=base)
            self.assertIn("not valid UTF-8", str(ctx.exception))


class ErrorTests(unittest.TestCase):
    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            with self.assertRaises(PromptParseError) as ctx:
                parse_prompt_content("@nope.png", base_dir=base)
            self.assertIn("file not found", str(ctx.exception))

    def test_no_extension(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            _write_file(base, "README", "some text")
            with self.assertRaises(PromptParseError) as ctx:
                parse_prompt_content("@README", base_dir=base)
            self.assertIn("no extension", str(ctx.exception))

    def test_unterminated_quoted_filename(self):
        with self.assertRaises(PromptParseError) as ctx:
            parse_prompt_content('look at @"unclosed', base_dir=Path("/tmp"))
        self.assertIn("unterminated quoted filename", str(ctx.exception))

    def test_empty_quoted_filename(self):
        with self.assertRaises(PromptParseError) as ctx:
            parse_prompt_content('look at @""', base_dir=Path("/tmp"))
        self.assertIn("empty quoted filename", str(ctx.exception))

    def test_oversized_file(self):
        # Synthesize a file larger than the 32 MiB cap. Use a sparse-write
        # trick? Plain write is fine — 32 MiB is ~5s to write.
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            big = base / "big.png"
            with big.open("wb") as f:
                f.write(b"\x00" * (33 * 1024 * 1024))
            with self.assertRaises(PromptParseError) as ctx:
                parse_prompt_content("@big.png", base_dir=base)
            self.assertIn("exceeds", str(ctx.exception))


class TrailingPunctTests(unittest.TestCase):
    """Trailing `,;:!?)]}` after an unquoted @FILENAME gets pushed back
    into the following text fragment — `summarize @README.md, please`
    should resolve `README.md`, not `README.md,`."""

    def test_trailing_comma(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            _write_file(base, "notes.md", "body")
            result = parse_prompt_content("summarize @notes.md, please", base_dir=base)
            self.assertIsInstance(result, list)
            self.assertEqual(result[0], {"type": "text", "text": "summarize "})
            self.assertEqual(result[1]["type"], "text")
            self.assertEqual(result[1]["text"], "body")
            self.assertEqual(result[2], {"type": "text", "text": ", please"})

    def test_trailing_close_paren(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            _write_file(base, "pic.png", _MIN_PNG)
            result = parse_prompt_content("(see @pic.png) for ref", base_dir=base)
            self.assertIsInstance(result, list)
            types = [p["type"] for p in result]
            self.assertEqual(types, ["text", "image_url", "text"])
            self.assertEqual(result[0]["text"], "(see ")
            self.assertEqual(result[2]["text"], ") for ref")

    def test_multiple_trailing_puncts(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            _write_file(base, "notes.md", "body")
            result = parse_prompt_content("read @notes.md)!", base_dir=base)
            self.assertIsInstance(result, list)
            self.assertEqual(result[-1], {"type": "text", "text": ")!"})

    def test_trailing_period_kept(self):
        # `.` is NOT stripped — it's load-bearing in filenames.
        # `summarize @notes.md.` tries to resolve `notes.md.` which fails.
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            _write_file(base, "notes.md", "body")
            with self.assertRaises(PromptParseError):
                parse_prompt_content("summarize @notes.md.", base_dir=base)

    def test_only_punct_after_at_is_literal(self):
        # `@,` would strip down to empty fname → treated as literal `@`.
        self.assertEqual(parse_prompt_content("look @, ok?"), "look @, ok?")

    def test_quoted_filename_skips_punct_strip(self):
        # Inside @"..." the entire content is the filename, including any
        # punctuation. Opt-out for users who really do have such a file.
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            _write_file(base, "weird,name.md", "body")
            result = parse_prompt_content('read @"weird,name.md"', base_dir=base)
            self.assertIsInstance(result, list)
            text_parts = [p for p in result if p["type"] == "text"]
            self.assertTrue(any(p["text"] == "body" for p in text_parts))


class QuoteInteractionTests(unittest.TestCase):
    """Confirm @refs are honored OUTSIDE quotes but ignored INSIDE."""

    def test_at_after_close_quote(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            _write_file(base, "pic.png", _MIN_PNG)
            # `"quoted" @pic.png` — the ref is after the closing quote.
            result = parse_prompt_content('"some quoted text" @pic.png', base_dir=base)
            self.assertIsInstance(result, list)
            self.assertEqual(
                [p["type"] for p in result],
                ["text", "image_url"],
            )

    def test_escape_inside_quotes_passes_through(self):
        # Inside quotes, \@ is NOT processed — it remains as literal `\@`.
        # The expectation: the entire quoted span is verbatim text.
        s = 'in quotes: "\\@still literal"'
        self.assertEqual(parse_prompt_content(s), s)

    def test_consecutive_refs_with_only_whitespace_between(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            _write_file(base, "a.png", _MIN_PNG)
            _write_file(base, "b.png", _MIN_PNG)
            result = parse_prompt_content("@a.png @b.png", base_dir=base)
            self.assertIsInstance(result, list)
            # text "" → image → text " " → image
            self.assertEqual(
                [p["type"] for p in result],
                ["image_url", "text", "image_url"],
            )
            self.assertEqual(result[1]["text"], " ")


if __name__ == "__main__":
    unittest.main()
