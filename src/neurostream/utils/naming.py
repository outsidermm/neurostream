"""Identifier-to-filename sanitisation helpers."""


def safe_filename(s: str) -> str:
    """Sanitise an arbitrary identifier into a safe filename component.

    Non-alphanumeric characters become ``-``; leading/trailing dashes are
    stripped. An empty result falls back to ``"x"``.
    """
    out = "".join(c if c.isalnum() else "-" for c in s)
    return out.strip("-") or "x"
