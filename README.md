# TokDelta

TokDelta is a lightweight Python library for keeping tokenizer state consistent across incremental prompt edits.

It is designed for agent-style workflows where prompts change frequently: appending tool outputs, pruning old turns, inserting new context, or rewriting a system prompt without re-tokenizing the entire prompt from scratch.

## Why use it?

Most tokenizer APIs are stateless:

```python
encode(text) -> token_ids
```

If your prompt changes even slightly, you usually have to re-tokenize everything. TokDelta keeps a mutable byte buffer plus token offsets so edits can be applied incrementally.

## Install
<!--
### From PyPI

```bash
pip install tokdelta
```

### With tokenizer backends

```bash
# tiktoken backend
pip install "tokdelta[tiktoken]"

# Hugging Face backend
pip install "tokdelta[huggingface]"
```

### For development-->

```bash
git clone git@github.com:latentvoyage/tokdelta.git
cd tokdelta
python -m venv .venv
source .venv/bin/activate
pip install -e ".[tiktoken,dev]"
```

## Quick start

```python
from tokdelta import PromptState

state = PromptState("You are a helpful assistant.", "huggingface", "gpt2")

print(state.token_ids)
print(state.token_offsets)

state.append("\nUser: What is the capital of France?")
state.insert(0, "[SYSTEM] ")
state.delete(0, 9)

print(state.prompt)
print(state.token_ids)
```

## Core features

- Incremental append / insert / delete support
- Stateful prompt editing without full re-tokenization
- UTF-8 byte-level tracking for tokenizer-safe edits
- Adaptive retokenization windows derived from existing token offsets
- Boundary-stability checks that expand the window only when needed
- Pluggable tokenizer backends
- Works well for agentic prompt workflows such as:
  - appending tool results
  - pruning old conversation turns
  - rewriting system prompts
  - backtracking and regenerating responses

## Supported backends

### tiktoken

```python
state = PromptState("hello", "tiktoken", "gpt-4")
```

### Hugging Face

```python
state = PromptState("hello", "huggingface", "meta-llama/Llama-3-8B")
```

## How it works

TokDelta maintains three pieces of state for a prompt:

- a mutable UTF-8 byte buffer
- a list of token IDs
- per-token byte offsets

When you edit the prompt, TokDelta identifies the token spans that overlap the edited bytes, chooses a local retokenization window from those offsets, and re-tokenizes only that region. If the surrounding boundary tokens are unstable, the window expands until the new segmentation is consistent again. This keeps the common case fast while still remaining robust for tricky edits.

## Performance summary

TokDelta is most useful when prompt edits are frequent and local.

In benchmark runs on a tiktoken GPT-4 setup:

- append operations are typically much faster than full re-tokenization
- insert/delete operations are also significantly faster for many prompt sizes
- the biggest gains show up on long prompts where rebuilding everything from scratch is expensive

Representative results:

- 1,000 tokens: roughly 4–5× faster for inserts than a full encode
- 16,000 tokens: around 5× faster for inserts on the tested setup
- 64,000 tokens: append can be dramatically faster than a full re-encode

These gains come from re-tokenizing only a local region around the edit instead of the full prompt, with the initial window chosen from existing offset spans and expanded only when boundary stability requires it.

## When to use it

Use TokDelta when you need a stateful tokenizer for:

- long prompts that change incrementally
- agent workflows with repeated prompt edits
- streaming prompt construction
- inference systems that want to avoid full re-tokenization on every change
<!-- 
## Limitations and roadmap

TokDelta is a tokenizer-state primitive, not a full prompt-management framework. It is best suited for local edits and prompt mutation, not for full-blown branching, versioning, or general conversation orchestration.

The current implementation already uses a Fenwick tree for O(log N) byte-to-token lookup and an offset-driven retokenization window that expands only when boundary stability requires it. This makes the common case fast while preserving correctness for ambiguous edits.

Remaining work focuses on reducing bookkeeping overhead further, improving backend-specific offset accuracy, and exposing inference-oriented features such as KV-cache alignment or incremental detokenization.
-->

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

