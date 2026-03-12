"""
Detailed cost breakdown: identify every O(N) bottleneck in
PromptState.insert() to understand where a rope / Fenwick tree
would actually help.
"""

import sys, os, time, statistics

from tokdelta import PromptState, TokenizerRegistry
from tokdelta.utils import char_to_byte, token_index_at_byte

tok = TokenizerRegistry.get_tokenizer("tiktoken", "gpt-4")
SNIPPET = "\nUser: What is the capital of France?\n"


def _make_prompt(n_tokens: int) -> str:
    base = "The quick brown fox jumps over the lazy dog. " * (n_tokens // 8 + 1)
    ids = tok.encode(base)
    return tok.decode(ids[:n_tokens])


def _time(fn, repeats=50):
    times = []
    for _ in range(3):
        fn()  # warmup
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(times)


SIZES = [1_000, 4_000, 16_000, 32_000, 64_000]


def main():
    print("=" * 100)
    print("Detailed bottleneck breakdown for PromptState.insert() (median µs)")
    print("=" * 100)
    print()

    header = (
        f"{'Tokens':>8} | "
        f"{'prompt→str':>11} | "
        f"{'char→byte':>11} | "
        f"{'bytearray':>11} | "
        f"{'ofs shift':>11} | "
        f"{'tok_idx_at':>11} | "
        f"{'tok.encode':>11} | "
        f"{'splice':>11} | "
        f"{'TOTAL ins':>11} | "
        f"{'full enc':>11} | "
        f"{'speedup':>8}"
    )
    print(header)
    print("-" * len(header))

    for size in SIZES:
        prompt = _make_prompt(size)
        actual = len(tok.encode(prompt))
        prompt_bytes = prompt.encode("utf-8")
        mid_char = len(prompt) // 2
        mid_byte = len(prompt[:mid_char].encode("utf-8"))
        snippet_bytes = SNIPPET.encode("utf-8")
        nbytes = len(snippet_bytes)

        state = PromptState(prompt, "tiktoken", "gpt-4")

        # 1. self.prompt (bytearray decode)
        t_prompt = _time(lambda: state.byte_buffer.decode("utf-8"))

        # 2. char_to_byte
        t_c2b = _time(lambda: char_to_byte(prompt, mid_char))

        # 3. bytearray insert + undo
        buf = bytearray(prompt_bytes)
        def _ba():
            buf[mid_byte:mid_byte] = snippet_bytes
            del buf[mid_byte:mid_byte + nbytes]
        t_ba = _time(_ba)

        # 4. offset shift (list comprehension)
        offsets = list(state.token_offsets)
        def _shift():
            return [
                (s + nbytes, e + nbytes) if s >= mid_byte
                else (s, e + nbytes) if e > mid_byte
                else (s, e)
                for s, e in offsets
            ]
        t_shift = _time(_shift)

        # 5. token_index_at_byte (called 2× inside _retokenize_region)
        t_tib = _time(lambda: token_index_at_byte(offsets, mid_byte)) * 2

        # 6. tokenizer.encode on window (~20 tokens)
        window_start = max(mid_byte - 200, 0)
        window_end = min(mid_byte + 200, len(prompt_bytes))
        window_text = prompt_bytes[window_start:window_end].decode("utf-8", errors="replace")
        t_enc = _time(lambda: tok.encode(window_text))

        # 7. list splice
        ids = list(state.token_ids)
        w = 10
        m = len(ids) // 2
        chunk = ids[m-w:m+w]
        t_splice = _time(lambda: ids.__setitem__(slice(m-w, m+w), chunk))

        # total insert (actual)
        state2 = PromptState(prompt, "tiktoken", "gpt-4")
        def _ins():
            state2.insert(mid_char, SNIPPET)
            n = len(state2.prompt)
            state2.delete(n - len(SNIPPET), n)
        t_total = _time(_ins)

        # full encode
        t_full = _time(lambda: tok.encode(prompt))

        speedup = t_full / t_total if t_total > 0 else float("inf")

        print(
            f"{actual:>8,} | "
            f"{t_prompt:>8,.0f} µs | "
            f"{t_c2b:>8,.0f} µs | "
            f"{t_ba:>8,.0f} µs | "
            f"{t_shift:>8,.0f} µs | "
            f"{t_tib:>8,.0f} µs | "
            f"{t_enc:>8,.0f} µs | "
            f"{t_splice:>8,.0f} µs | "
            f"{t_total:>8,.0f} µs | "
            f"{t_full:>8,.0f} µs | "
            f"{speedup:>7.1f}×"
        )

    print()
    print("Key:")
    print("  prompt→str = self.byte_buffer.decode('utf-8')     — needed by insert() to validate char_pos")
    print("  char→byte  = len(text[:char_pos].encode('utf-8')) — char-to-byte conversion")
    print("  bytearray  = mid-buffer insert (C-level memmove)")
    print("  ofs shift  = O(N) list comprehension shifting all token offsets")
    print("  tok_idx_at = token_index_at_byte × 2 (builds starts list + bisect)")
    print("  tok.encode = tokenizer.encode() on the ~20-token window")
    print("  splice     = list slice assignment for token_ids")
    print()

    # Also show: what would speedup be if we eliminated all O(N) overhead?
    print("=" * 100)
    print("Theoretical speedup if ALL O(N) overhead were O(log N):")
    print("=" * 100)
    for size in SIZES:
        prompt = _make_prompt(size)
        actual = len(tok.encode(prompt))
        t_full = _time(lambda: tok.encode(prompt))

        prompt_bytes = prompt.encode("utf-8")
        window_start = max(len(prompt_bytes)//2 - 200, 0)
        window_end = min(len(prompt_bytes)//2 + 200, len(prompt_bytes))
        window_text = prompt_bytes[window_start:window_end].decode("utf-8", errors="replace")
        t_window = _time(lambda: tok.encode(window_text))

        print(f"  {actual:>8,} tokens: full={t_full:,.0f} µs, "
              f"window_encode={t_window:,.0f} µs, "
              f"theoretical_speedup={t_full/t_window:,.0f}×")


if __name__ == "__main__":
    main()
