"""Shared token counting utilities."""

import tiktoken


def get_encoder(model: str = "gpt-4o"):
    """Return a tiktoken encoder for *model*, falling back to cl100k_base."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_messages(messages: list[dict], model: str = "gpt-4o") -> int:
    """Count total tokens consumed by *messages* (including per-message overhead)."""
    enc = get_encoder(model)
    total = 0
    for msg in messages:
        total += 4
        for v in msg.values():
            if isinstance(v, str):
                total += len(enc.encode(v))
            elif isinstance(v, list):
                total += len(enc.encode(str(v)))
    return total


def count_text(text: str, model: str = "gpt-4o") -> int:
    """Count tokens for a plain-text string."""
    enc = get_encoder(model)
    return len(enc.encode(text))
