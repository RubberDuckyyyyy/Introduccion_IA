"""Lowercase word tokenizer (keeps Unicode letters and digits, drops punctuation)."""

from __future__ import annotations

import re

_WORD = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())
