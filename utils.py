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

    Bisects directly on the offset tuples — tuples compare element-by-
    element, so ``(byte_pos, inf)`` lands right after the last tuple whose
    start ≤ byte_pos.  O(log N) with no temporary list creation.
    """
    idx = bisect.bisect_right(offsets, (byte_pos, float("inf"))) - 1
    return max(idx, 0)

