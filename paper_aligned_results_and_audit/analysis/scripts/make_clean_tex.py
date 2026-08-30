"""Create a clean LaTeX copy by removing revision-colour wrappers.

Only ``\\textcolor{red}{...}`` wrappers are removed.  Their contents, including
nested TeX groups, are preserved byte-for-byte apart from the wrapper itself.
The script fails closed on an unbalanced group and never edits the input file.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


TOKEN = r"\textcolor{red}{"
DISPLAY_MATH_RE = re.compile(
    r"\\begin\{(equation\*?|align\*?|gather\*?|multline\*?)\}"
    r".*?\\end\{\1\}",
    re.DOTALL,
)


def _is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        backslashes += 1
        index -= 1
    return bool(backslashes % 2)


def _matching_brace(text: str, opening_index: int) -> int:
    depth = 0
    in_comment = False
    for index in range(opening_index, len(text)):
        char = text[index]
        if in_comment:
            if char in "\r\n":
                in_comment = False
            continue
        if char == "%" and not _is_escaped(text, index):
            in_comment = True
            continue
        if _is_escaped(text, index):
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
            if depth < 0:
                break
    raise ValueError(f"unbalanced TeX group beginning at byte/character {opening_index}")


def strip_red_wrappers(text: str) -> tuple[str, int]:
    """Return clean TeX and the number of removed wrappers."""

    pieces: list[str] = []
    cursor = 0
    removed = 0
    while True:
        start = text.find(TOKEN, cursor)
        if start < 0:
            pieces.append(text[cursor:])
            break
        pieces.append(text[cursor:start])
        opening = start + len(TOKEN) - 1
        closing = _matching_brace(text, opening)
        inner = text[opening + 1 : closing]
        clean_inner, nested = strip_red_wrappers(inner)
        pieces.append(clean_inner)
        removed += 1 + nested
        cursor = closing + 1
    return "".join(pieces), removed


def _neutralize_display_math_blank_lines(text: str) -> str:
    """Keep wrapper-only lines from becoming paragraph breaks in math mode.

    A wrapper may occupy its own source line around an equation.  Removing the
    wrapper then leaves an empty line inside the display environment, which TeX
    treats as an invalid paragraph break.  Replacing only those blank lines
    with comment lines preserves source-line correspondence and valid math.
    """

    def replace_block(match: re.Match[str]) -> str:
        return re.sub(r"(?m)^[ \t]*$", "%", match.group(0))

    return DISPLAY_MATH_RE.sub(replace_block, text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source = args.input.read_text(encoding="utf-8")
    clean, removed = strip_red_wrappers(source)
    clean = _neutralize_display_math_blank_lines(clean)
    begin_document_re = re.compile(r"(?m)^\\begin\{document\}[ \t]*$")
    if not begin_document_re.search(clean):
        raise RuntimeError("missing \\begin{document}; refusing clean transformation")
    clean = begin_document_re.sub(
        lambda match: (
            "\\let\\linenumbers\\relax\n"
            + match.group(0)
            + "\n\\nolinenumbers"
        ),
        clean,
        count=1,
    )
    if TOKEN in clean:
        raise RuntimeError("revision-colour wrapper remains after cleaning")
    if removed == 0:
        raise RuntimeError("no revision-colour wrappers found; refusing empty transformation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(clean, encoding="utf-8", newline="\n")
    print(f"removed {removed} red wrapper(s): {args.output}")


if __name__ == "__main__":
    main()
