"""Shared test configuration."""

import pytest
import os


@pytest.fixture
def mock_openai_api_key(monkeypatch):
    """Mock OpenAI API key for testing."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


@pytest.fixture
def mock_anthropic_api_key(monkeypatch):
    """Mock Anthropic API key for testing."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


@pytest.fixture
def sample_examples():
    """Sample test examples for evaluation."""
    return [
        type("Example", (), {"question": "What is the capital of France?", "answer": "Paris"})(),
        type("Example", (), {"question": "What is 2+2?", "answer": "4"})(),
    ]


@pytest.fixture
def sample_predictions():
    """Sample predictions corresponding to examples."""
    return [
        type("Pred", (), {"output": "Paris"})(),
        type("Pred", (), {"output": "Four"})(),
    ]
