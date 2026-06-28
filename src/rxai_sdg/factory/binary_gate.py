"""Deterministic gate for encoded binary artifacts presented as output.

The LLM judge is structurally blind to this: a Base64/hex blob shown as "the
output of the script" can embed metadata (e.g. a PNG's IHDR width/height) that
CONTRADICTS what the conversation asserts about it. Confirmed defect: a doc that
asserts a 64x128 image 7+ times while the pasted Base64 decodes to a PNG whose
IHDR says 64x64. Decoding and cross-checking is a one-second programmatic check.
"""
from __future__ import annotations

import base64
import re
from typing import Optional

_PNG_SIG = b"\x89PNG\r\n\x1a\n"
# a Base64 run long enough to carry a real header (>= ~44 chars covers PNG IHDR),
# optionally introduced by a data URI prefix; trailing "..." truncation is fine -
# the header lives at the start.
_B64_RE = re.compile(r"(?:base64,)?([A-Za-z0-9+/\n\r ]{44,}={0,2})")
_DIM_RES = [
    re.compile(r"\b(\d{2,5})\s*(?:×|x|✕|\*|by)\s*(\d{2,5})\b", re.I),
    re.compile(r"\(\s*(\d{2,5})\s*,\s*(\d{2,5})\s*\)"),
]


def _claimed_dims(text: str) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for rx in _DIM_RES:
        for m in rx.finditer(text or ""):
            w, h = int(m.group(1)), int(m.group(2))
            # plausible image dimensions only (avoid version numbers / dates)
            if 1 <= w <= 20000 and 1 <= h <= 20000:
                out.add((w, h))
    return out


def _decode_prefix(b64: str) -> Optional[bytes]:
    s = re.sub(r"\s+", "", b64)
    # decode the largest 4-char-aligned prefix (truncated blobs still yield the header)
    s = s[: len(s) - (len(s) % 4)]
    if len(s) < 24:
        return None
    try:
        return base64.b64decode(s, validate=True)
    except Exception:  # noqa: BLE001
        try:
            return base64.b64decode(s, validate=False)
        except Exception:  # noqa: BLE001
            return None


def png_dimensions(raw: bytes) -> Optional[tuple[int, int]]:
    """(width, height) from a PNG byte string's IHDR, or None if not a PNG."""
    if not raw.startswith(_PNG_SIG) or len(raw) < 24:
        return None
    # IHDR: 8-byte sig, 4-byte length, 4-byte 'IHDR', then 4-byte W, 4-byte H.
    if raw[12:16] != b"IHDR":
        return None
    w = int.from_bytes(raw[16:20], "big")
    h = int.from_bytes(raw[20:24], "big")
    return (w, h)


def check_encoded_artifact_metadata(text: str) -> list[str]:
    """Return reasons if a Base64 PNG in ``text`` decodes to dimensions that
    contradict the dimensions the surrounding text claims for it."""
    claimed = _claimed_dims(text)
    if not claimed:
        return []
    reasons: list[str] = []
    for m in _B64_RE.finditer(text or ""):
        raw = _decode_prefix(m.group(1))
        if not raw:
            continue
        dims = png_dimensions(raw)
        if dims is None:
            continue
        # the decoded image must match SOME claimed dimension pair; if none of the
        # claimed pairs equals the decoded one, the artifact contradicts the prose.
        if dims not in claimed:
            reasons.append(
                f"Base64 PNG decodes to {dims[0]}x{dims[1]} but the text claims "
                f"{sorted(claimed)}")
            break
    return reasons
