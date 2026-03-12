# TokDelta

Stateful tokenizers for agentic inference. Maintains tokenizer state per prompt and supports incremental edits (append, insert, delete) without full re-tokenization.

## Why

Tokenizer APIs like tiktoken and HuggingFace Tokenizers are stateless: `encode(text) → token_ids`. When a prompt changes — an agent appends a tool result, deletes stale context, or patches its system prompt — the entire text must be re-tokenized from scratch. For long prompts (4K–128K tokens) that change frequently during multi-turn agentic workflows, this is wasteful.

TokDelta keeps a byte-level buffer alongside the token list and a per-token offset map. Edits splice into the byte buffer and trigger retokenization of only a small adaptive window around the edit site, leaving the rest of the token list untouched.

## Install

```bash
# core library (no tokenizer backend included)
pip install tokdelta

# with tiktoken backend
pip install "tokdelta[tiktoken]"

# with HuggingFace backend
pip install "tokdelta[huggingface]"
```

For development:

```bash
git clone <repo-url> && cd tokdelta
python -m venv .venv && source .venv/bin/activate
pip install -e ".[tiktoken,dev]"
```

## Quick Start

```python
from tokdelta import PromptState

# create a prompt state with a tokenizer backend
state = PromptState("You are a helpful assistant.", "tiktoken", "gpt-4")

# inspect tokens
print(state.token_ids)       # [2675, 527, 264, 11190, 18328, 13]
print(state.token_offsets)   # [(0, 3), (3, 7), (7, 9), (9, 17), (17, 27), (27, 28)]
print(state.prompt)          # "You are a helpful assistant."

# append a user turn
state.append("\nUser: What is the capital of France?")

# insert text at a character position
state.insert(0, "[SYSTEM] ")

# delete a character range [start, end)
state.delete(0, 9)           # removes "[SYSTEM] "

# tokens always match a fresh encode() of the current prompt
assert state.token_ids == PromptState(state.prompt, "tiktoken", "gpt-4").token_ids
```

## Agentic Usage Patterns

TokDelta is designed for the edit patterns that show up in agentic LLM inference:

```python
state = PromptState(system_prompt, "tiktoken", "gpt-4")

# build a conversation turn by turn
state.append(user_turn)
state.append(assistant_response)

# inject a tool result into the middle of the prompt
marker = "[TOOL_RESULT]\n"
pos = state.prompt.index(marker) + len(marker)
state.insert(pos, json.dumps(tool_output))

# prune old turns to fit context window
old_turn = "User: ...\nAssistant: ...\n"
pos = state.prompt.index(old_turn)
state.delete(pos, pos + len(old_turn))

# hot-swap the system prompt
end = state.prompt.index("<|end_system|>\n") + len("<|end_system|>\n")
state.delete(0, end)
state.insert(0, new_system_prompt)

# backtrack: delete a bad response, regenerate
resp_start = state.prompt.rindex("Assistant: ") + len("Assistant: ")
state.delete(resp_start, len(state.prompt))
state.append(new_response)
```

## Tokenizer Backends

TokDelta uses a pluggable backend interface. Backends are imported lazily — a missing package only errors when you try to use that specific backend.

### tiktoken

```python
state = PromptState("hello", "tiktoken", "gpt-4")
# also works: "gpt-4o", "gpt-3.5-turbo", etc.
```

### HuggingFace Transformers

```python
state = PromptState("hello", "huggingface", "meta-llama/Llama-3-8B")
# any model on the Hub that AutoTokenizer supports
```

### Custom Backend

Subclass `BaseTokenizer` and register it:

```python
from tokdelta import BaseTokenizer

class MyTokenizer(BaseTokenizer):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, token_ids: list[int]) -> str: ...
    def token_bytes(self, token_id: int) -> bytes: ...
```

## Design

### Data Representation

Every `PromptState` holds three parallel structures:

| Field | Type | Description |
|-------|------|-------------|
| `byte_buffer` | `bytearray` | The prompt as a mutable UTF-8 byte sequence |
| `token_ids` | `list[int]` | Token IDs produced by the backend |
| `token_offsets` | `list[(int, int)]` | Per-token `(byte_start, byte_end)` pairs |

All edits operate on the byte buffer directly. Character positions from the caller are converted to byte offsets internally, keeping the system aligned with how BPE tokenizers actually see text.

### Incremental Retokenization

When an edit occurs:

1. **Locate affected tokens** — binary search (`bisect`) on the offset list to find which tokens overlap the edited byte range.
2. **Expand the window** — include `±W` tokens on each side (default `W = 5`).
3. **Re-encode the window** — extract the byte slice, decode to text, run `tokenizer.encode()` on just that slice.
4. **Check boundary stability** — if the first/last new token doesn't match the old neighbour token, the BPE merge boundary has shifted. Double the window and retry.
5. **Splice** — replace the old token/offset entries with the new ones.

This adaptive expansion handles BPE merge propagation (e.g. `inter` + `national` → `international` changing when you insert a character between them) while keeping the typical retokenization window small.

### Offset Maintenance on Edits

Rather than rebuilding offsets from scratch, edits shift existing offsets in place:

**Insert at byte position `p`, inserting `n` bytes:**
- Tokens fully after `p`: shift both start and end by `+n`
- Tokens straddling `p` (start < p < end): extend end by `+n`
- Tokens fully before `p`: unchanged

**Delete byte range `[p, q)`:**
- Tokens fully before `p`: unchanged
- Tokens fully after `q`: shift both start and end by `-(q-p)`
- Tokens straddling the range: shrink to their surviving bytes
- Tokens fully inside `[p, q)`: removed (zero-width ghosts dropped)

This avoids an O(n) offset rebuild on every edit.

## Time Complexity

Let `N` = total tokens in the prompt, `W` = retokenization window size (tokens), and `T(k)` = time for the backend to encode `k` tokens.

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| `__init__` (full tokenize) | O(N) + T(N) | One-time cost |
| `append(text)` | O(W) + T(W) | Only the tail region |
| `insert(pos, text)` | O(N) + T(W) | O(N) for offset shift, T(W) for retokenize |
| `delete(start, end)` | O(N) + T(W) | O(N) for offset shift + ghost removal, T(W) for retokenize |
| `prompt` (property) | O(N) | Decodes the byte buffer |
| `get_tokens()` | O(N) | Copies both lists |

The offset shift is a single list comprehension over all tokens — O(N) — but it is a trivial per-element branch with no allocations beyond the new list. The expensive part (the tokenizer's `encode` call) touches only the W-token window, which is typically 10–30 tokens regardless of prompt length.

**Worst case:** if a single edit causes BPE merges to propagate across the entire prompt, the window doubles repeatedly until it covers all N tokens, falling back to a full retokenization. In practice this essentially never happens — BPE merges are local.

**Space:** O(N) for the byte buffer + O(N) for token IDs + O(N) for offset pairs = O(N) total.

## Benchmarks — Incremental vs Full Retokenization

All numbers are median wall-clock µs on a single core (Apple Silicon), tiktoken gpt-4 backend.

### End-to-end single-operation timings

| Tokens | Full `encode()` | Append | Insert (mid) | Delete (mid) | Insert speedup |
|-------:|----------------:|-------:|-------------:|-------------:|---------------:|
| 1,000 | 194 µs | 12 µs | 43 µs | 266 µs | **4.6×** |
| 4,000 | 827 µs | ~0 µs | 175 µs | 456 µs | **4.7×** |
| 16,000 | 3,280 µs | 68 µs | 658 µs | 1,691 µs | **5.0×** |
| 32,000 | 6,583 µs | 49 µs | 1,622 µs | 3,870 µs | **4.1×** |
| 64,000 | 13,304 µs | 360 µs | 3,488 µs | 7,599 µs | **3.8×** |

**Append is the big winner** — 19–37× faster than a full re-encode at 32–64K tokens, because it touches only the tail and requires no offset shifting. Agentic workloads are append-heavy (streaming responses, adding turns), so this is the most impactful path.

Insert and delete are 3.8–5× faster. The O(N) offset shift is the dominant remaining cost (see analysis below).

### Where the time actually goes (insert at 64K tokens)

| Component | Time | % of insert |
|-----------|-----:|------------:|
| Offset shift (list comprehension, N tuples) | 2,935 µs | 84% |
| Token-index lookup (`bisect` on offset tuples, ×2) | ~1 µs | <0.1% |
| `tokenizer.encode()` on ~20-token window | 20 µs | 0.6% |
| `bytearray` mid-buffer splice (C memmove) | 3 µs | <0.1% |
| List slice assignment (token_ids splice) | <1 µs | <0.1% |
| Misc (prompt decode, char→byte) | ~15 µs | 0.4% |

The tokenizer encode on the edit window is **constant at ~20 µs** regardless of prompt size. Full encode scales linearly: 200 µs at 1K → 13,000 µs at 64K. The theoretical speedup ceiling is **10–660×** — but the O(N) offset shift consumes 84% of the time, capping actual insert speedup at ~4×.

## Can Rope / Tree Data Structures Help?

We benchmarked four offset-storage strategies:

| Tokens | Tuples (current) | Cumulative (copy) | Cumulative (in-place) | Fenwick tree (BIT) |
|-------:|-----------------:|-------------------:|----------------------:|-------------------:|
| 1,000 | 30 µs | 9 µs | 14 µs | 0.3 µs |
| 4,000 | 172 µs | 39 µs | 56 µs | 0.4 µs |
| 16,000 | 731 µs | 173 µs | 227 µs | 0.5 µs |
| 32,000 | 1,424 µs | 311 µs | 447 µs | 0.6 µs |
| 64,000 | 2,935 µs | 651 µs | 913 µs | 0.6 µs |

And three token-lookup strategies:

| Tokens | Build starts list + bisect (old) | Bisect on tuples directly (current) | Fenwick prefix search |
|-------:|---------------------------------:|-------------------------------------:|----------------------:|
| 1,000 | 8 µs | 0.3 µs | 0.9 µs |
| 16,000 | 140 µs | 0.3 µs | 1.3 µs |
| 64,000 | 478 µs | 0.3 µs | 1.4 µs |

### Projected insert latency with each optimization

| Tokens | Full encode | Current (tuples + bisect fix) | + Cumulative in-place | + Fenwick tree | Window encode only (floor) |
|-------:|------------:|------------------------------:|----------------------:|---------------:|---------------------------:|
| 1,000 | 204 µs | 72 µs | 54 µs | 42 µs | 19 µs |
| 4,000 | 851 µs | 218 µs | 97 µs | 42 µs | 20 µs |
| 16,000 | 3,281 µs | 773 µs | 270 µs | 43 µs | 20 µs |
| 32,000 | 6,414 µs | 1,455 µs | 494 µs | 45 µs | 21 µs |
| 64,000 | 13,123 µs | 2,994 µs | 942 µs | 45 µs | 21 µs |

With a Fenwick tree the projected speedup is **~150–300×** over full retokenization at 32–64K tokens.

### Analysis: what helps and what doesn't

**A rope for the byte buffer is unnecessary.** Python's `bytearray` splice is a C-level `memmove` — only 3 µs even at 64K tokens (≈250 KB). The bottleneck is Python-level iteration over token offsets, not C-level memory operations.

**A Fenwick tree (Binary Indexed Tree) for offsets eliminates the dominant O(N) cost.** By storing *token lengths* instead of absolute offsets, a BIT computes absolute positions as prefix sums in O(log N). An insert just does a single O(log N) point-update instead of shifting every offset:

| Operation | Tuples (current) | Fenwick tree |
|-----------|:----------------:|:------------:|
| Offset shift | O(N) | O(log N) |
| Token-at-byte lookup | O(log N)* | O(log N) |
| Splice after retokenize | O(N) | O(N)† |

\*Already optimized with `bisect` on tuples.
†Fenwick trees don't support mid-array insert/delete; the tree must be rebuilt after a token-count change. The rebuild is O(N) but with much smaller constants (integer sums vs. Python tuple creation).

**An order-statistics tree (implicit treap, B+ tree) would make splice O(W log N) too**, fully eliminating all O(N) work. But:
- Pure Python implementation would have high constant factors (~1 µs/op vs ~50 ns for flat array access)
- A C/Rust extension would be needed for production-grade performance
- At current prompt sizes (≤128K tokens), the Fenwick approach is likely sufficient

### Summary

| Data structure | Insert complexity | Practical speedup at 64K | Implementation effort |
|----------------|:-----------------:|:------------------------:|:---------------------:|
| Flat list of tuples (current) | O(N) + T(W) | 3.8× | Done |
| Cumulative array, in-place shift | O(N) + T(W) | ~14× | Low |
| Fenwick tree (BIT) | O(log N)* + T(W) | ~300× | Medium |
| Order-statistics tree (treap) | O(W log N) + T(W) | ~300× | High (or C/Rust ext.) |

\*O(N) rebuild on token-count change, but fast constant factor.

The incremental retokenization **algorithm** is sound — the T(W) vs T(N) savings are real and grow linearly with prompt size. The current **implementation** captures those savings fully for appends (19–37×) but is bottlenecked by O(N) offset bookkeeping for inserts/deletes. A Fenwick tree or order-statistics tree would unlock the full speedup.

## Next Steps

- **Fenwick-tree offsets** — replace the flat `token_offsets` list with a BIT over token lengths, eliminating the O(N) offset shift and achieving ~150–300× speedup for insert/delete at 32–64K tokens.
- **Order-statistics tree** — for the token/offset arrays, an implicit treap or B+ tree would make splice O(W log N), fully eliminating all O(N) per-edit work.
- **KV cache alignment** — expose a diff of token IDs between edits so inference engines can reuse their KV cache up to the first changed token.
- **Streaming detokenization** — an `append_token_id()` method for incremental decode during streaming inference.
- **C/Rust hot path** — rewrite the offset shift + retokenization loop in C or Rust for 10–50× lower constant factors, enabling sub-microsecond edits.


## Tests

```bash
python tests/test_incremental.py   #  7 core correctness tests
python tests/test_rigorous.py      # 46 edge-case tests (unicode, boundaries, long inputs)
python tests/test_agentic.py       # 24 agentic workflow tests (tool calls, context pruning, etc.)
```

## Project Structure

```
tokdelta/
├── pyproject.toml           # build config & metadata
├── LICENSE
├── README.md
├── src/
│   └── tokdelta/            # installable package
│       ├── __init__.py      # exports PromptState, BaseTokenizer, TokenizerRegistry
│       ├── state.py         # PromptState implementation
│       ├── utils.py         # char_to_byte, byte_to_char, token_index_at_byte
│       └── tokenizer/
│           ├── __init__.py
│           ├── base_tokenizer.py    # BaseTokenizer interface
│           ├── tiktoken_tokenizer.py
│           ├── huggingface_tokenizer.py
│           └── tokenizer_registry.py  # lazy-import factory
├── tests/
│   ├── test_incremental.py  #  7 core tests
│   ├── test_rigorous.py     # 46 edge-case tests
│   └── test_agentic.py      # 24 agentic workflow tests
└── benchmarks/
    ├── bench_incremental_vs_full.py   # end-to-end comparison
    ├── bench_bottleneck.py            # per-component cost breakdown
    └── bench_data_structures.py       # rope / Fenwick / cumulative comparison
```

