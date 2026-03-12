import bisect


def char_to_byte(text: str, char_idx: int) -> int:
    """Convert a character index to the corresponding UTF-8 byte offset."""
    return len(text[:char_idx].encode("utf-8"))


def byte_to_char(text: str, byte_idx: int) -> int:
    """Convert a UTF-8 byte offset back to a character index."""
    encoded = text.encode("utf-8")
    return len(encoded[:byte_idx].decode("utf-8", errors="replace"))


def token_index_at_byte(offsets: list[tuple[int, int]], byte_pos: int) -> int:
    """Return the index of the token whose byte span contains `byte_pos`.

    Uses a simple bisect on the start positions. If byte_pos falls before
    the first token, returns 0.
    """
    starts = [s for s, _ in offsets]
    idx = bisect.bisect_right(starts, byte_pos) - 1
    return max(idx, 0)

