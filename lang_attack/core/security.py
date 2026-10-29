"""Security utility functions for path traversal defense and filename sanitization."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

# OS reserved filenames (Windows device names, case-insensitive, with or without extensions)
OS_RESERVED_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    }
)

# Illegal characters in filenames across Linux/Windows/macOS
ILLEGAL_FILENAME_CHARS = re.compile(r'[\x00-\x1f<>:"/\\|?*]')


def sanitize_filename(
    name: str,
    default_name: str = "untitled",
    max_length: int = 128,
) -> str:
    """Sanitize a filename to guarantee Path Traversal and OS-level safety.

    - Rejects or strips null bytes (\\x00) and control characters.
    - Removes path traversal components (/, \\\\, ..).
    - Removes illegal filesystem characters (< > : " / \\\\ | ? *).
    - Resolves Windows reserved device names (CON, PRN, AUX, NUL, COM1-9, LPT1-9).
    - Disallows filenames consisting purely of dots (. or .. or ...).
    - Clamps filename length to max_length preserving extension.
    """
    if not isinstance(name, str):
        name = str(name) if name is not None else ""

    # Remove null bytes and control characters
    clean = ILLEGAL_FILENAME_CHARS.sub("_", name.strip())

    # Collapse multiple underscores and spaces
    clean = re.sub(r"[\s_]+", "_", clean).strip(" ._")

    if not clean:
        clean = default_name

    # Check for reserved device names
    base_stem = clean.split(".")[0].upper()
    if base_stem in OS_RESERVED_NAMES or clean in (".", ".."):
        clean = f"safe_{clean}"

    # Enforce max length while preserving extension
    if len(clean) > max_length:
        parts = clean.rsplit(".", 1)
        if len(parts) == 2 and 1 <= len(parts[1]) <= 10:
            stem, ext = parts[0], parts[1]
            clean = f"{stem[: max_length - len(ext) - 1]}.{ext}"
        else:
            clean = clean[:max_length]

    return clean or default_name


def resolve_safe_path(
    base_dir: Path | str,
    relative_target: Path | str,
    allow_root: bool = False,
) -> Path:
    """Resolve target path ensuring it strictly resides within base_dir.

    Raises:
        ValueError: If relative_target escapes base_dir or points to base_dir when allow_root=False.
    """
    if "\x00" in str(relative_target):
        raise ValueError("Embedded null byte in target path")

    base = Path(base_dir).resolve()
    rel_path = Path(relative_target)
    target = rel_path.resolve() if rel_path.is_absolute() else (base / rel_path).resolve()

    # Verify target is descendant of base
    try:
        target.relative_to(base)
    except ValueError as err:
        raise ValueError(
            f"Path traversal detected: '{relative_target}' escapes base directory '{base}'"
        ) from err

    if not allow_root and target == base:
        raise ValueError("Target path cannot resolve to the root directory itself")

    return target


def secure_temp_audio_file(prefix: str = "audio_", suffix: str = ".wav") -> Path:
    """Create a secure unique temporary audio file with restricted permissions (0600)."""
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    os.close(fd)
    return Path(path)
