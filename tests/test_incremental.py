"""
Compare incremental PromptState operations against a fresh full
retokenization at every step. If the incremental path ever disagrees
with tokenizing the prompt from scratch, the test fails.
"""

import sys, os

# make sure the repo root is on the path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.state import PromptState


# ── helpers ────────────────────────────────────────────────────────

def fresh_token_ids(prompt: str, tokenizer_name: str, model_name: str) -> list[int]:
    """Tokenize a prompt from scratch (no incremental state)."""
    tmp = PromptState(prompt, tokenizer_name, model_name)
    return tmp.token_ids


def assert_matches_fresh(state: PromptState, label: str):
    expected = fresh_token_ids(state.prompt, state.tokenizer_name, state.model_name)
    actual = state.token_ids
    assert actual == expected, (
        f"[{label}] mismatch\n"
        f"  prompt : {state.prompt!r}\n"
        f"  got    : {actual}\n"
        f"  expect : {expected}"
    )


# ── tests ──────────────────────────────────────────────────────────

def test_initial_tokenization(tok_name, model):
    state = PromptState("Hello world", tok_name, model)
    assert_matches_fresh(state, "init")
    print("  ✓ initial tokenization")


def test_append(tok_name, model):
    state = PromptState("Hello", tok_name, model)
    state.append(" world")
    assert_matches_fresh(state, "append single")

    state.append("! How are you doing today?")
    assert_matches_fresh(state, "append long")

    state.append("")  # no-op
    assert_matches_fresh(state, "append empty")
    print("  ✓ append")


def test_insert(tok_name, model):
    state = PromptState("Hello world", tok_name, model)

    state.insert(5, ",")
    assert_matches_fresh(state, "insert comma")

    state.insert(0, "Oh! ")
    assert_matches_fresh(state, "insert at start")

    state.insert(len(state.prompt), " goodbye")
    assert_matches_fresh(state, "insert at end")
    print("  ✓ insert")


def test_delete(tok_name, model):
    state = PromptState("Hello, beautiful world!", tok_name, model)

    state.delete(5, 16)  # remove ", beautiful"
    assert_matches_fresh(state, "delete middle")

    state.delete(0, 5)  # remove "Hello"
    assert_matches_fresh(state, "delete start")
    print("  ✓ delete")


def test_mixed_operations(tok_name, model):
    state = PromptState("The quick brown fox", tok_name, model)

    state.append(" jumps over the lazy dog")
    assert_matches_fresh(state, "mix: append")

    state.insert(10, "very ")  # "The quick very brown fox …"
    assert_matches_fresh(state, "mix: insert")

    state.delete(0, 4)  # remove "The "
    assert_matches_fresh(state, "mix: delete")

    state.insert(0, "A ")
    assert_matches_fresh(state, "mix: insert at start")

    state.append(".")
    assert_matches_fresh(state, "mix: append period")
    print("  ✓ mixed operations")


def test_unicode(tok_name, model):
    state = PromptState("Hello 🌍", tok_name, model)
    assert_matches_fresh(state, "unicode init")

    state.append(" 🙂 world")
    assert_matches_fresh(state, "unicode append")

    state.insert(6, "🔥")
    assert_matches_fresh(state, "unicode insert")

    # delete the fire emoji (1 char, 4 bytes)
    state.delete(6, 7)
    assert_matches_fresh(state, "unicode delete")
    print("  ✓ unicode")


def test_empty_prompt(tok_name, model):
    state = PromptState("", tok_name, model)
    assert_matches_fresh(state, "empty init")

    state.append("something")
    assert_matches_fresh(state, "empty then append")
    print("  ✓ empty prompt")


# ── runner ─────────────────────────────────────────────────────────

BACKENDS = [
    ("tiktoken", "gpt-4"),
    # ("huggingface", "gpt2"),   # uncomment if you have transformers installed
]

if __name__ == "__main__":
    for tok_name, model in BACKENDS:
        print(f"\n--- {tok_name} / {model} ---")
        test_initial_tokenization(tok_name, model)
        test_append(tok_name, model)
        test_insert(tok_name, model)
        test_delete(tok_name, model)
        test_mixed_operations(tok_name, model)
        test_unicode(tok_name, model)
        test_empty_prompt(tok_name, model)

    print("\nall tests passed")
