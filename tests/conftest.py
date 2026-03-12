"""Shared fixtures for TokDelta tests."""

import pytest


BACKENDS = [
    ("tiktoken", "gpt-4"),
    # ("huggingface", "gpt2"),  # uncomment if you have transformers installed
]


@pytest.fixture(params=BACKENDS, ids=[f"{t}/{m}" for t, m in BACKENDS])
def backend(request):
    """Yield (tok_name, model) pairs for each configured backend."""
    return request.param


@pytest.fixture
def tok_name(backend):
    return backend[0]


@pytest.fixture
def model(backend):
    return backend[1]
