from tokdelta.tokenizer.tokenizer_registry import TokenizerRegistry
from tokdelta.utils import char_to_byte, token_index_at_byte

# How many extra tokens on each side of an edit to include when doing
# incremental retokenization. The window expands adaptively if merges
# propagate further (see _retokenize_region).
_DEFAULT_WINDOW = 5


class PromptState:
    """Holds the byte buffer, token ids, and per-token byte offsets for a prompt.

    All edits go through the byte buffer so we stay aligned with how
    tokenizers actually see the text. External callers pass character
    positions; we convert to byte offsets internally.
    """

    def __init__(self, prompt: str, tokenizer_name: str, model_name: str):
        self.tokenizer_name = tokenizer_name
        self.model_name = model_name
        self.tokenizer = TokenizerRegistry.get_tokenizer(tokenizer_name, model_name)

        self.byte_buffer = bytearray(prompt.encode("utf-8"))
        self.token_ids: list[int] = []
        self.token_offsets: list[tuple[int, int]] = []  # (byte_start, byte_end)

        # initial full tokenization
        self._full_tokenize()

 

    @property
    def prompt(self) -> str:
        return self.byte_buffer.decode("utf-8")

    def append(self, text: str) -> None:
        """Append text to the end of the prompt."""
        new_bytes = text.encode("utf-8")
        edit_start = len(self.byte_buffer)
        self.byte_buffer.extend(new_bytes)
        self._retokenize_region(edit_start, edit_start + len(new_bytes))

    def insert(self, char_pos: int, text: str) -> None:
        """Insert text at a character position."""
        prompt_str = self.prompt
        if char_pos < 0 or char_pos > len(prompt_str):
            raise ValueError(
                f"char_pos {char_pos} out of range [0, {len(prompt_str)}]"
            )
        byte_pos = char_to_byte(prompt_str, char_pos)
        new_bytes = text.encode("utf-8")
        nbytes = len(new_bytes)
        self.byte_buffer[byte_pos:byte_pos] = new_bytes
        # shift offsets: tokens after the insertion move right,
        # tokens straddling the insertion point get their end extended
        self.token_offsets = [
            (s + nbytes, e + nbytes) if s >= byte_pos
            else (s, e + nbytes) if e > byte_pos
            else (s, e)
            for s, e in self.token_offsets
        ]
        self._retokenize_region(byte_pos, byte_pos + nbytes)

    def delete(self, start_char: int, end_char: int) -> None:
        """Delete the character range [start_char, end_char)."""
        prompt_str = self.prompt
        if start_char < 0 or end_char > len(prompt_str) or start_char > end_char:
            raise ValueError(
                f"Invalid range [{start_char}, {end_char}) "
                f"for prompt of length {len(prompt_str)}"
            )
        byte_start = char_to_byte(prompt_str, start_char)
        byte_end = char_to_byte(prompt_str, end_char)
        removed = byte_end - byte_start
        del self.byte_buffer[byte_start:byte_end]
        # shift offsets: tokens after the deletion move left,
        # tokens overlapping the deletion get shrunk to their surviving bytes
        shifted = []
        for s, e in self.token_offsets:
            if e <= byte_start:
                shifted.append((s, e))
            elif s >= byte_end:
                shifted.append((s - removed, e - removed))
            else:
                new_s = min(s, byte_start)
                kept_before = max(0, byte_start - s)
                kept_after = max(0, e - byte_end)
                shifted.append((new_s, new_s + kept_before + kept_after))

        # tokens fully inside the deleted range are now zero-width ghosts.
        # drop them so the retokenize window works with real byte spans.
        self.token_ids = [
            tid for tid, (s, e) in zip(self.token_ids, shifted) if s < e
        ]
        self.token_offsets = [(s, e) for s, e in shifted if s < e]
        self._retokenize_region(byte_start, byte_start)

    def get_tokens(self) -> dict:
        """Return current token ids and their byte offset spans."""
        return {
            "token_ids": list(self.token_ids),
            "offsets": list(self.token_offsets),
        }

    def _full_tokenize(self) -> None:
        text = self.byte_buffer.decode("utf-8")
        self.token_ids = self.tokenizer.encode(text)
        self.token_offsets = self._build_offsets(self.token_ids)

    def _build_offsets(self, token_ids: list[int]) -> list[tuple[int, int]]:
        """Walk the token list and compute (byte_start, byte_end) for each."""
        offsets = []
        pos = 0
        for tid in token_ids:
            length = len(self.tokenizer.token_bytes(tid))
            offsets.append((pos, pos + length))
            pos += length
        return offsets

    # incremental retokenization 
    def _retokenize_region(self, edit_byte_start: int, edit_byte_end: int) -> None:
        """Re-tokenize only the region around the edit,expanding until stable.
        The approach is as fllows:
          1. Find which tokens overlap the edited byte range.
          2. Expand by _DEFAULT_WINDOW tokens on each side.
          3. Re-tokenize that byte slice.
          4. If the new boundary tokens don't match the old neighbours,
             expand the window and retry.
          5. Splice the new tokens/offsets back in.
        """
        if not self.token_ids:
            self._full_tokenize()
            return

        n = len(self.token_ids)
        window = _DEFAULT_WINDOW

        # keep expanding until stable or we cover the whole prompt
        while True:
            left_tok = token_index_at_byte(self.token_offsets, edit_byte_start)
            right_tok = token_index_at_byte(self.token_offsets, edit_byte_end)

            left_tok = max(left_tok - window, 0)
            right_tok = min(right_tok + window, n - 1)

            region_byte_start = min(self.token_offsets[left_tok][0], edit_byte_start)
            # right boundary: furthest of the rightmost token end or the
            # actual edit end (an append may go past all existing tokens),
            # clamped to the current buffer length.
            token_end = self.token_offsets[right_tok][1] if right_tok < n else len(self.byte_buffer)
            region_byte_end = min(
                max(token_end, edit_byte_end),
                len(self.byte_buffer),
            )

            region_text = self.byte_buffer[region_byte_start:region_byte_end].decode(
                "utf-8", errors="replace"
            )
            new_token_ids = self.tokenizer.encode(region_text)
            new_offsets = []
            pos = region_byte_start
            for tid in new_token_ids:
                length = len(self.tokenizer.token_bytes(tid))
                new_offsets.append((pos, pos + length))
                pos += length

            # check if boundaries are stable
            stable_left = left_tok == 0 or (
                new_token_ids and new_token_ids[0] == self.token_ids[left_tok]
            )
            stable_right = right_tok >= n - 1 or (
                new_token_ids and new_token_ids[-1] == self.token_ids[right_tok]
            )

            if (stable_left and stable_right) or (left_tok == 0 and right_tok >= n - 1):
                # splice into the global lists
                self.token_ids[left_tok : right_tok + 1] = new_token_ids
                self.token_offsets[left_tok : right_tok + 1] = new_offsets
                return

            window *= 2  # double and retry