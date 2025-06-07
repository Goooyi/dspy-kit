#!/usr/bin/env python3
"""
Live test script for the refactored model_graders.py
Tests both sync and async operations with real API calls.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("Environment variables set for testing:")
print(f"OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY', 'NOT SET')[:20]}...")
print(f"OPENAI_API_BASE: {os.getenv('OPENAI_API_BASE', 'NOT SET')}")
print(f"OPENAI_MODEL: {os.getenv('OPENAI_MODEL', 'NOT SET')}")

from dspy_kit.evaluation.graders.model_graders import (
    BinaryClassificationGrader,
    ContextUtilizationGrader,
    FactualAccuracyGrader,
    LabelModelGrader,
    LikertScaleGrader,
    SafetyGrader,
    ScoreModelGrader,
    ToneEvaluationGrader,
    create_customer_support_grader,
    create_qa_grader,
)


# Test data classes
class TestExample:
    def __init__(self, question: str, answer: str, context: str = ""):
        self.question = question
        self.answer = answer
        self.context = context


class TestPrediction:
    def __init__(self, output: str):
        self.output = output


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}")


def print_result(name: str, result: Any, token_usage: Dict[str, Any] = {}):
    """Print test result."""
    print(f"\n{name}:")
    print(f"  Result: {result}")
    if token_usage:
        print(
            f"  Tokens used: {token_usage['total_tokens']} "
            f"(prompt: {token_usage['prompt_tokens']}, "
            f"completion: {token_usage['completion_tokens']})"
        )
        print(f"  Successful calls: {token_usage['successful_calls']}, Failed calls: {token_usage['failed_calls']}")


async def test_score_model_grader():
    """Test ScoreModelGrader with both sync and async."""
    print_section("Testing ScoreModelGrader")

    # Get model configuration from environment
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_API_BASE", None)

    # Create grader
    grader = ScoreModelGrader(
        model=model,
        api_key=api_key,
        api_base=api_base,
        range=[1, 10],
        pass_threshold=7.0,
        prompt_template="""Rate the accuracy of this answer on a scale from 1 to 10.

Question: {{item.question}}
Correct Answer: {{item.reference_answer}}
Model Answer: {{sample.output_text}}

Score (1-10):""",
        include_reasoning=False,
    )

    # Test data
    example = TestExample(question="What is the capital of France?", answer="Paris")
    good_pred = TestPrediction("The capital of France is Paris.")
    bad_pred = TestPrediction("The capital of France is London.")

    # Test sync evaluation mode
    print("\nSync evaluation mode:")
    score1 = grader(example, good_pred)
    print_result("Good answer", score1, grader.get_token_usage())

    score2 = grader(example, bad_pred)
    print_result("Bad answer", score2, grader.get_token_usage())

    # Test sync optimization mode (returns bool)
    print("\nSync optimization mode:")
    pass1 = grader(example, good_pred, trace=True)
    print_result("Good answer (pass/fail)", pass1)

    pass2 = grader(example, bad_pred, trace=True)
    print_result("Bad answer (pass/fail)", pass2)

    # Test async
    print("\nAsync evaluation mode:")
    async_score1 = await grader.acall(example, good_pred)
    print_result("Good answer (async)", async_score1)

    async_score2 = await grader.acall(example, bad_pred)
    print_result("Bad answer (async)", async_score2)

    # Print final token usage
    print(f"\nTotal token usage for ScoreModelGrader: {grader.get_token_usage()}")


async def test_label_model_grader():
    """Test LabelModelGrader."""
    print_section("Testing LabelModelGrader")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY")

    grader = LabelModelGrader(
        model=model,
        api_key=api_key,
        labels=["accurate", "partially_accurate", "inaccurate"],
        passing_labels=["accurate", "partially_accurate"],
        prompt_template="""Classify this answer as: accurate, partially_accurate, or inaccurate

Question: {{item.question}}
Reference: {{item.reference_answer}}
Answer: {{sample.output_text}}

Classification:""",
    )

    example = TestExample(question="What is 2 + 2?", answer="4")

    test_predictions = [
        ("Correct", TestPrediction("2 + 2 equals 4")),
        ("Partially correct", TestPrediction("2 + 2 is approximately 4")),
        ("Incorrect", TestPrediction("2 + 2 equals 5")),
    ]

    for name, pred in test_predictions:
        score = grader(example, pred)
        print_result(name, score)

    print(f"\nTotal tokens: {grader.get_token_usage()['total_tokens']}")


async def test_specialized_graders():
    """Test specialized graders."""
    print_section("Testing Specialized Graders")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY")

    # Test FactualAccuracyGrader
    print("\n--- FactualAccuracyGrader ---")
    factual_grader = FactualAccuracyGrader(model=model, api_key=api_key)

    example = TestExample(
        question="When was the Eiffel Tower built?", answer="The Eiffel Tower was built between 1887 and 1889."
    )

    accurate_pred = TestPrediction("The Eiffel Tower was constructed from 1887 to 1889.")
    inaccurate_pred = TestPrediction("The Eiffel Tower was built in 1950.")

    score1 = factual_grader(example, accurate_pred)
    print_result("Accurate", score1)

    score2 = factual_grader(example, inaccurate_pred)
    print_result("Inaccurate", score2)

    # Test ToneEvaluationGrader
    print("\n--- ToneEvaluationGrader ---")
    tone_grader = ToneEvaluationGrader(model=model, api_key=api_key)

    support_example = TestExample(question="I can't access my account!", answer="Let me help you with that.")

    good_tone = TestPrediction(
        "I understand how frustrating that must be. Let me help you regain access to your account right away."
    )
    bad_tone = TestPrediction("That's your problem. Figure it out yourself.")

    tone1 = tone_grader(support_example, good_tone)
    print_result("Good tone", tone1)

    tone2 = tone_grader(support_example, bad_tone)
    print_result("Bad tone", tone2)

    # Test SafetyGrader
    print("\n--- SafetyGrader ---")
    safety_grader = SafetyGrader(model=model, api_key=api_key)

    safe_pred = TestPrediction("I'd be happy to help you learn about science.")
    unsafe_pred = TestPrediction("Here's how to make dangerous chemicals at home...")

    safety1 = safety_grader(support_example, safe_pred)
    print_result("Safe response", safety1)

    safety2 = safety_grader(support_example, unsafe_pred)
    print_result("Unsafe response", safety2)


async def test_context_utilization():
    """Test ContextUtilizationGrader."""
    print_section("Testing ContextUtilizationGrader")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY")

    grader = ContextUtilizationGrader(model=model, api_key=api_key)

    example = TestExample(
        question="What is the main product mentioned?",
        answer="The AcmeBot 3000",
        context="AcmeBot 3000 is our flagship robotic assistant, featuring advanced AI capabilities and 24-hour battery life.",
    )

    good_context_use = TestPrediction(
        "The main product is the AcmeBot 3000, which is a flagship robotic assistant with AI capabilities."
    )
    poor_context_use = TestPrediction("The main product is a robot.")

    score1 = grader(example, good_context_use)
    print_result("Good context use", score1)

    score2 = grader(example, poor_context_use)
    print_result("Poor context use", score2)


async def test_composite_graders():
    """Test composite grader functions."""
    print_section("Testing Composite Graders")

    # Note: This would require the CompositeGrader to be working
    # For now, just test that the functions create the graders

    try:
        cs_grader = create_customer_support_grader()
        print("✓ Created customer support grader successfully")

        qa_grader = create_qa_grader()
        print("✓ Created QA grader successfully")
    except Exception as e:
        print(f"✗ Error creating composite graders: {e}")


async def test_error_handling():
    """Test error handling and retry logic."""
    print_section("Testing Error Handling")

    # Test with invalid model
    print("\nTesting with invalid model name:")
    grader = ScoreModelGrader(
        model="invalid-model-xyz",
        api_key=os.getenv("OPENAI_API_KEY"),
        max_retries=1,  # Reduce retries for faster testing
    )

    example = TestExample(question="Test", answer="Test")
    pred = TestPrediction("Test")

    try:
        score = grader(example, pred)
        print(f"Unexpected success: {score}")
    except Exception as e:
        print(f"✓ Expected error caught: {type(e).__name__}: {str(e)[:100]}...")

    # Test token estimation
    print("\nTesting token estimation:")
    valid_grader = ScoreModelGrader(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you doing today?"},
    ]

    estimated_tokens = valid_grader.estimate_tokens(messages)
    print(f"Estimated tokens for messages: {estimated_tokens}")


async def test_likert_scale():
    """Test LikertScaleGrader."""
    print_section("Testing LikertScaleGrader")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY")

    grader = LikertScaleGrader(
        model=model, api_key=api_key, criteria="Clarity and comprehensiveness of the explanation"
    )

    example = TestExample(
        question="Explain photosynthesis",
        answer="Photosynthesis is the process by which plants convert light energy into chemical energy.",
    )

    detailed_pred = TestPrediction(
        "Photosynthesis is the biological process where plants, algae, and certain bacteria "
        "convert light energy (usually from the sun) into chemical energy stored in glucose. "
        "This process occurs in chloroplasts and involves two main stages: light reactions "
        "and the Calvin cycle. The overall equation is: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2."
    )

    simple_pred = TestPrediction("Plants make food from sunlight.")

    score1 = grader(example, detailed_pred)
    print_result("Detailed explanation", score1)

    score2 = grader(example, simple_pred)
    print_result("Simple explanation", score2)


async def test_binary_classification():
    """Test BinaryClassificationGrader."""
    print_section("Testing BinaryClassificationGrader")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY")

    grader = BinaryClassificationGrader(
        model=model, api_key=api_key, question="Does this response directly answer the user's question?"
    )

    example = TestExample(question="What is the weather like today?", answer="The weather varies by location.")

    direct_answer = TestPrediction("Today is sunny with a high of 75°F in San Francisco.")
    evasive_answer = TestPrediction("Weather is an interesting topic. Many factors affect it.")

    result1 = grader(example, direct_answer)
    print_result("Direct answer", result1)

    result2 = grader(example, evasive_answer)
    print_result("Evasive answer", result2)


async def main():
    """Run all tests."""
    print("=" * 60)
    print("LIVE TEST OF REFACTORED MODEL GRADERS")
    print("=" * 60)

    # Check environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        print("Please set it in your .env file or environment")
        return

    print(f"\nUsing model: {os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}")
    if os.getenv("OPENAI_API_BASE"):
        print(f"Using API base: {os.getenv('OPENAI_API_BASE')}")

    # Run all tests
    try:
        await test_score_model_grader()
        await test_label_model_grader()
        await test_specialized_graders()
        await test_context_utilization()
        await test_likert_scale()
        await test_binary_classification()
        await test_composite_graders()
        await test_error_handling()

        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)

    except Exception as e:
        print(f"\n\nERROR during testing: {type(e).__name__}: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
