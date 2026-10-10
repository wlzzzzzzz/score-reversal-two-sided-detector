"""Text normalization and structural quality rules for the generated corpus."""
from __future__ import annotations

import re
import unicodedata


LEXICAL_PATTERN = re.compile(
    r"[A-Za-z0-9]+(?:['\N{RIGHT SINGLE QUOTATION MARK}-][A-Za-z0-9]+)*|[\u4e00-\u9fff]"
)


def normalize_text(value: object) -> str:
    return " ".join(unicodedata.normalize("NFC", str(value)).split()).strip()


def lexical_count(value: object) -> int:
    return len(LEXICAL_PATTERN.findall(str(value)))


def sanitize_text(text: str) -> tuple[str, int]:
    kept, removed = [], 0
    for char in text:
        category = unicodedata.category(char)
        if char in {"\ufffd", "\x00"} or category == "Cs" or (
            category == "Cc" and char not in "\t\n\r"
        ):
            removed += 1
        else:
            kept.append(char)
    return "".join(kept), removed


def text_metrics(text: str) -> tuple[int, float, float, float, float]:
    lexical = [token.lower() for token in LEXICAL_PATTERN.findall(text)]
    ngrams = list(zip(*(lexical[offset:] for offset in range(4)))) if len(lexical) >= 4 else []
    unique = len(set(lexical)) / max(1, len(lexical))
    repeated = 1.0 - len(set(ngrams)) / max(1, len(ngrams))
    cjk = sum("\u4e00" <= char <= "\u9fff" for char in text) / max(1, len(text))
    controls = sum(unicodedata.category(char) in {"Cc", "Cs"} and char not in "\n\r\t" for char in text)
    return len(lexical), unique, repeated, cjk, float(controls)


def strict_text_quality_reasons(text: str, language: str = "en") -> list[str]:
    """Apply the registered language and repetition thresholds."""
    _, unique, repeated, cjk, _ = text_metrics(text)
    reasons = []
    if language == "zh":
        if cjk < 0.20:
            reasons.append("CJK_RATIO_BELOW_0.20")
    elif cjk > 0.05:
        reasons.append("CJK_RATIO_ABOVE_0.05")
    if unique < 0.15:
        reasons.append("UNIQUE_LEXICAL_RATIO_BELOW_0.15")
    if repeated > 0.35:
        reasons.append("REPEATED_4GRAM_RATIO_ABOVE_0.35")
    return reasons
