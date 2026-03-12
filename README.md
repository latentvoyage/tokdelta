# TokDelta

Stateful tokenizers for agentic inference. Maintains tokenizer state per prompt and supports incremental edits (append, insert, delete) without full re-tokenization.

## Why

Tokenizer APIs like tiktoken and HuggingFace Tokenizers are stateless: `encode(text) → token_ids`. When a prompt changes — an agent appends a tool result, deletes stale context, or patches its system prompt — the entire text must be re-tokenized from scratch. For long prompts (4K–128K tokens) that change frequently during multi-turn agentic workflows, this is wasteful.

TokDelta keeps a byte-level buffer alongside the token list and a per-token offset map. Edits splice into the byte buffer and trigger retokenization of only a small adaptive window around the edit site, leaving the rest of the token list untouched.

## Install

```bash
# requires Python 3.10+
pip install tiktoken          # for tiktoken backend
# or
pip install transformers      # for HuggingFace backend
```

Clone and use directly (no packaging step required):

```bash
git clone <repo-url> && cd tokdelta
python -m venv .venv && source .venv/bin/activate
pip install tiktoken
```

## Quick Start

```python
from src.state import PromptState

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

## API

### `PromptState(prompt, tokenizer_name, model_name)`

Create a new prompt state. Runs a full tokenization on init.

| Parameter | Type | Description |
|-----------|------|-------------|
| `prompt` | `str` | Initial prompt text |
| `tokenizer_name` | `str` | `"tiktoken"` or `"huggingface"` |
| `model_name` | `str` | Model name passed to the backend (e.g. `"gpt-4"`, `"meta-llama/Llama-3-8B"`) |

### `.append(text)`

Append text to the end of the prompt. Only the tail region is retokenized.

### `.insert(char_pos, text)`

Insert text at character position `char_pos`. Raises `ValueError` if out of range. Retokenizes the region around the insertion point.

### `.delete(start_char, end_char)`

Delete the character range `[start_char, end_char)`. Raises `ValueError` on invalid range. Retokenizes the region around the deletion.

### `.prompt` → `str`

Current prompt text (decoded from the byte buffer).

### `.token_ids` → `list[int]`

Current token ID list.

### `.token_offsets` → `list[tuple[int, int]]`

Per-token byte offset pairs `(byte_start, byte_end)`. Offsets are contiguous: `offsets[i][1] == offsets[i+1][0]`, and `offsets[0][0] == 0`, `offsets[-1][1] == len(byte_buffer)`.

### `.get_tokens()` → `dict`

Returns `{"token_ids": [...], "offsets": [...]}`.

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
from tokenizer.base_tokenizer import BaseTokenizer

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

## Potential Improvements

- **Rope / piece-table buffer** — replace `bytearray` with a rope for O(log n) insert/delete on very large prompts (100K+ tokens). The current `bytearray` is O(n) for mid-buffer edits due to memmove, but Python's C-level memmove is fast enough for prompts under ~50K tokens.
- **Batch edits** — accept multiple edits in a single call, sort them, and apply from right to left to avoid cascading offset shifts.
- **KV cache alignment** — expose the longest common prefix length after each edit so the inference engine can reuse the KV cache up to that point.
- **Streaming detokenization** — incremental decode as new tokens are appended.
- **Concurrent access** — TokDelta is not thread-safe. Callers managing concurrent access to the same `PromptState` should synchronize externally (e.g. a per-state lock or an asyncio queue). Internal per-method locking was deliberately avoided because it cannot prevent TOCTOU races (caller reads `.prompt`, computes a position, then calls `.insert()` — another thread could have edited in between).

## Tests

```bash
python tests/test_incremental.py   #  7 core correctness tests
python tests/test_rigorous.py      # 46 edge-case tests (unicode, boundaries, long inputs)
python tests/test_agentic.py       # 24 agentic workflow tests (tool calls, context pruning, etc.)
```

## Project Structure

```
tokdelta/
├── src/
│   ├── __init__.py          # exports PromptState
│   └── state.py             # PromptState: byte buffer, tokens, incremental retokenization
├── tokenizer/
│   ├── __init__.py          # exports TokenizerRegistry
│   ├── base_tokenizer.py    # BaseTokenizer interface
│   ├── tiktoken_tokenizer.py
│   ├── huggingface_tokenizer.py
│   └── tokenizer_registry.py  # lazy-import factory
├── utils.py                 # char_to_byte, byte_to_char, token_index_at_byte
└── tests/
    ├── test_incremental.py
    ├── test_rigorous.py
    └── test_agentic.py
```

