"""
Benchmark: incremental retokenization (TokDelta) vs full retokenization
at various prompt sizes.

Measures wall-clock time for:
  1. Full retokenize  — tokenizer.encode(entire_prompt)
  2. Incremental ops  — PromptState.append / .insert / .delete

Also breaks down where time is spent (offset shift vs tokenizer call
vs bytearray splice) to evaluate whether a rope / piece-table would help.
"""

import sys, os, time, statistics

from tokdelta import PromptState, TokenizerRegistry

# ── helpers ────────────────────────────────────────────────────────────

def _make_prompt(n_tokens: int) -> str:
    """Generate a prompt that tokenizes to approximately n_tokens tokens."""
    # "word " is typically 1-2 tokens in tiktoken gpt-4.
    # We over-generate then trim to the exact token count.
    tok = TokenizerRegistry.get_tokenizer("tiktoken", "gpt-4")
    base = "The quick brown fox jumps over the lazy dog. " * (n_tokens // 8 + 1)
    ids = tok.encode(base)
    # decode exactly n_tokens
    trimmed_ids = ids[:n_tokens]
    return tok.decode(trimmed_ids)


def _time_fn(fn, *, warmup: int = 3, repeats: int = 30) -> dict:
    """Time a callable, return stats in microseconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)  # µs
    return {
        "median_us": statistics.median(times),
        "mean_us": statistics.mean(times),
        "p95_us": sorted(times)[int(0.95 * len(times))],
        "min_us": min(times),
    }

# ── benchmarks ─────────────────────────────────────────────────────────

SIZES = [256, 1_000, 4_000, 16_000, 32_000, 64_000]
SNIPPET = "\nUser: What is the capital of France?\n"
REPLACE_SNIPPET = "[REPLACED_TOOL_OUTPUT]"

tok = TokenizerRegistry.get_tokenizer("tiktoken", "gpt-4")


def bench_full_retokenize(prompt: str) -> dict:
    """Time a full tokenizer.encode(prompt)."""
    return _time_fn(lambda: tok.encode(prompt))


def bench_append(prompt: str) -> dict:
    """Time PromptState.append() on a pre-built state."""
    state = PromptState(prompt, "tiktoken", "gpt-4")
    # we want to measure just the append, so we append + undo each iteration
    def fn():
        state.append(SNIPPET)
        # roll back so next iteration starts the same
        n = len(state.prompt)
        state.delete(n - len(SNIPPET), n)
    return _time_fn(fn)


def bench_insert_middle(prompt: str) -> dict:
    """Time PromptState.insert() at the midpoint."""
    state = PromptState(prompt, "tiktoken", "gpt-4")
    mid = len(prompt) // 2
    def fn():
        state.insert(mid, SNIPPET)
        n = len(state.prompt)
        state.delete(mid, mid + len(SNIPPET))
    return _time_fn(fn)


def bench_delete_middle(prompt: str) -> dict:
    """Time PromptState.delete() at the midpoint."""
    # insert a known chunk in the middle, then time deleting it
    state = PromptState(prompt, "tiktoken", "gpt-4")
    mid = len(prompt) // 2
    state.insert(mid, SNIPPET)
    def fn():
        state.delete(mid, mid + len(SNIPPET))
        state.insert(mid, SNIPPET)  # restore
    return _time_fn(fn)


def bench_offset_shift_only(prompt: str) -> dict:
    """Isolate offset-shift cost: build state then manually shift offsets."""
    state = PromptState(prompt, "tiktoken", "gpt-4")
    byte_pos = len(prompt.encode("utf-8")) // 2
    nbytes = len(SNIPPET.encode("utf-8"))

    def fn():
        # simulate insert offset shift (no actual buffer/retokenize work)
        _ = [
            (s + nbytes, e + nbytes) if s >= byte_pos
            else (s, e + nbytes) if e > byte_pos
            else (s, e)
            for s, e in state.token_offsets
        ]
    return _time_fn(fn)


def bench_bytearray_insert(prompt: str) -> dict:
    """Isolate bytearray insert cost (the memmove)."""
    buf = bytearray(prompt.encode("utf-8"))
    mid = len(buf) // 2
    snippet_bytes = SNIPPET.encode("utf-8")
    nbytes = len(snippet_bytes)

    def fn():
        buf[mid:mid] = snippet_bytes
        del buf[mid:mid + nbytes]  # undo
    return _time_fn(fn)


def bench_list_splice(prompt: str) -> dict:
    """Isolate Python list splice cost (replacing a slice of token_ids)."""
    state = PromptState(prompt, "tiktoken", "gpt-4")
    n = len(state.token_ids)
    mid = n // 2
    window = 10
    # slice to splice in (simulates retokenize result)
    replacement = state.token_ids[mid - window : mid + window]

    def fn():
        state.token_ids[mid - window : mid + window] = replacement
    return _time_fn(fn)


# ── main ───────────────────────────────────────────────────────────────

def main():
    print("=" * 90)
    print("TokDelta Benchmark: Incremental vs Full Retokenization")
    print("=" * 90)
    print(f"Backend: tiktoken / gpt-4")
    print(f"Snippet: {len(SNIPPET)} chars, ~{len(tok.encode(SNIPPET))} tokens")
    print()

    # ── Part 1: End-to-end comparison ──────────────────────────────────

    header = f"{'Tokens':>8} | {'Full encode':>14} | {'Append':>14} | {'Insert-mid':>14} | {'Delete-mid':>14} | {'Speedup(ins)':>13}"
    print("Part 1: End-to-end times (median µs)")
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for size in SIZES:
        prompt = _make_prompt(size)
        actual_tokens = len(tok.encode(prompt))

        full = bench_full_retokenize(prompt)
        app = bench_append(prompt)
        ins = bench_insert_middle(prompt)
        dlt = bench_delete_middle(prompt)
        speedup = full["median_us"] / ins["median_us"] if ins["median_us"] > 0 else float("inf")

        print(
            f"{actual_tokens:>8,} | "
            f"{full['median_us']:>11,.0f} µs | "
            f"{app['median_us']:>11,.0f} µs | "
            f"{ins['median_us']:>11,.0f} µs | "
            f"{dlt['median_us']:>11,.0f} µs | "
            f"{speedup:>12.1f}×"
        )

    print()

    # ── Part 2: Cost breakdown ─────────────────────────────────────────

    header2 = f"{'Tokens':>8} | {'Offset shift':>14} | {'bytearray ins':>14} | {'list splice':>14} | {'Total incr.':>14} | {'% offset':>9}"
    print("Part 2: Cost breakdown — where does time go? (median µs)")
    print("-" * len(header2))
    print(header2)
    print("-" * len(header2))

    for size in SIZES:
        prompt = _make_prompt(size)
        actual_tokens = len(tok.encode(prompt))

        ofs = bench_offset_shift_only(prompt)
        bai = bench_bytearray_insert(prompt)
        lsp = bench_list_splice(prompt)
        ins = bench_insert_middle(prompt)
        pct = (ofs["median_us"] / ins["median_us"] * 100) if ins["median_us"] > 0 else 0

        print(
            f"{actual_tokens:>8,} | "
            f"{ofs['median_us']:>11,.0f} µs | "
            f"{bai['median_us']:>11,.0f} µs | "
            f"{lsp['median_us']:>11,.0f} µs | "
            f"{ins['median_us']:>11,.0f} µs | "
            f"{pct:>8.1f}%"
        )

    print()
    print("=" * 90)
    print("Interpretation:")
    print("  - 'Full encode' = tokenizer.encode(entire_prompt) from scratch")
    print("  - 'Append/Insert/Delete' = PromptState incremental operations")
    print("  - 'Offset shift' = the O(N) list comprehension shifting all offsets")
    print("  - 'bytearray ins' = the O(N) memmove for mid-buffer insert")
    print("  - 'list splice' = Python list slice assignment for token_ids")
    print("  - '% offset' = what fraction of total insert time is offset shifting")
    print()
    print("A rope/piece-table would eliminate the bytearray memmove and could")
    print("reduce offset shifting to O(log N) with lazy propagation, but only")
    print("matters when '% offset' + 'bytearray ins' dominate the total time.")
    print("=" * 90)


if __name__ == "__main__":
    main()
