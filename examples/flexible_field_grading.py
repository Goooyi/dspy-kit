"""
Examples demonstrating flexible field grading with DSPy-optimizable graders.

This shows how the new graders solve the field naming flexibility problem
while maintaining DSPy optimization capabilities.
"""

import dspy
from dspy_kit.evaluation.graders.dspy_model_graders import (
    SemanticSimilarityGrader,
    FactualAccuracyGrader,
    RelevanceGrader,
    HelpfulnessGrader,
    CompositeDSPyGrader,
    create_qa_grader,
    create_customer_support_grader,
)


def example_1_basic_flexibility():
    """
    Example 1: Basic field flexibility - same grader works with different field names
    """
    print("=== Example 1: Field Flexibility ===")
    
    # Create a grader that can work with any field names
    grader = SemanticSimilarityGrader(
        pred_field="generated_response",  # Your custom field name
        ideal_field="reference_answer",   # Your custom field name
        threshold=0.8
    )
    
    # Dataset with custom field names
    example_data = [
        {
            "question": "What is Python?",
            "reference_answer": "Python is a high-level programming language known for its simplicity and readability.",
            "some_other_field": "irrelevant"
        }
    ]
    
    # Prediction with custom field names
    prediction = {
        "generated_response": "Python is a popular programming language that's easy to learn and widely used.",
        "confidence": 0.9,
        "metadata": {"model": "gpt-4"}
    }
    
    # Evaluate - works seamlessly with any field names
    score = grader(example_data[0], prediction)
    print(f"Semantic similarity score: {score:.3f}")
    
    # Same grader works with different field names!
    different_grader = SemanticSimilarityGrader(
        pred_field="answer",
        ideal_field="gold_standard",
        threshold=0.8
    )
    
    different_example = {"question": "test", "gold_standard": "Python is a programming language"}
    different_pred = {"answer": "Python is a coding language"}
    
    score2 = different_grader(different_example, different_pred)
    print(f"Same grader, different fields: {score2:.3f}")


def example_2_dspy_program_integration():
    """
    Example 2: Integration with DSPy programs using flexible field names
    """
    print("\n=== Example 2: DSPy Program Integration ===")
    
    # Custom DSPy program with specific output field names
    class CustomQASystem(dspy.Module):
        def __init__(self):
            super().__init__()
            self.qa = dspy.ChainOfThought("question -> detailed_answer")
        
        def forward(self, question):
            result = self.qa(question=question)
            # Return with custom field name
            return dspy.Prediction(
                question=question,
                detailed_answer=result.detailed_answer,
                confidence=0.85
            )
    
    # Create grader that matches your program's output fields
    grader = create_qa_grader(
        answer_field="detailed_answer",     # Matches your program output
        question_field="question",          # Matches your dataset
        expected_field="expected_response"  # Matches your dataset
    )
    
    # Your dataset can have any field names
    sample_data = [
        {
            "question": "Explain machine learning",
            "expected_response": "Machine learning is a subset of AI that enables computers to learn from data",
            "difficulty": "medium",
            "category": "AI"
        }
    ]
    
    # Simulate program execution
    qa_system = CustomQASystem()
    
    # This would work in real DSPy evaluation:
    # evaluator = dspy.Evaluate(devset=sample_data, metric=grader.to_dspy_metric())
    # score = evaluator(qa_system)
    
    print("Grader configured for custom field names:")
    print(f"- Answer field: {grader.graders['accuracy'][0].pred_field}")
    print(f"- Question field: {grader.graders['relevance'][0].query_field}")
    print(f"- Expected field: {grader.graders['accuracy'][0].ideal_field}")


def example_3_customer_support_scenario():
    """
    Example 3: Customer support with domain-specific field names
    """
    print("\n=== Example 3: Customer Support Scenario ===")
    
    # Customer support data with domain-specific field names
    support_data = [
        {
            "customer_inquiry": "I can't log into my account",
            "agent_response": "I'll help you reset your password. Please check your email for reset instructions.",
            "ideal_agent_response": "To reset your password, click the 'Forgot Password' link on the login page and follow the email instructions.",
            "ticket_id": "CS-001",
            "priority": "high"
        }
    ]
    
    # Create grader with customer support field names
    support_grader = create_customer_support_grader(
        response_field="agent_response",
        query_field="customer_inquiry", 
        reference_field="ideal_agent_response"
    )
    
    # Evaluate
    example = support_data[0]
    prediction = {
        "agent_response": example["agent_response"],
        "confidence": 0.9,
        "escalation_needed": False
    }
    
    score = support_grader(example, prediction)
    print(f"Customer support response score: {score:.3f}")
    
    # Show component scores
    for name, (grader, weight) in support_grader.graders.items():
        component_score = grader(example, prediction)
        print(f"- {name}: {component_score:.3f} (weight: {weight})")


def example_4_optimization_capability():
    """
    Example 4: Demonstrating optimization capability while maintaining flexibility
    """
    print("\n=== Example 4: Optimization Capability ===")
    
    # Create an optimizable grader
    grader = FactualAccuracyGrader(
        pred_field="model_output",
        ideal_field="ground_truth",
        threshold=0.8
    )
    
    print("This grader is a dspy.Module and can be optimized!")
    print(f"Is dspy.Module: {isinstance(grader, dspy.Module)}")
    print(f"Has forward method: {hasattr(grader, 'forward')}")
    print(f"Can be used in DSPy compilation: {hasattr(grader, 'parameters')}")
    
    # Example of how it would be optimized (pseudo-code)
    print("\nOptimization workflow:")
    print("1. optimizer = dspy.BootstrapFewShot(metric=grader.to_dspy_metric())")
    print("2. optimized_program = optimizer.compile(your_program, trainset=data)")
    print("3. The grader's prompts and reasoning can be optimized too!")


def example_5_handling_edge_cases():
    """
    Example 5: Graceful handling of different object types and missing fields
    """
    print("\n=== Example 5: Edge Case Handling ===")
    
    grader = SemanticSimilarityGrader(
        pred_field="answer",
        ideal_field="expected"
    )
    
    # Test with different object types
    test_cases = [
        # Case 1: Normal dict
        ({
            "expected": "Python is a programming language"
        }, {
            "answer": "Python is a coding language"
        }),
        
        # Case 2: Object with attributes
        (type('Example', (), {"expected": "Python is great"})(), 
         type('Pred', (), {"answer": "Python is awesome"})()),
        
        # Case 3: Missing fields (graceful fallback)
        ({
            "some_other_field": "irrelevant"
        }, {
            "different_field": "Python is cool"
        }),
        
        # Case 4: Mixed types
        ({
            "expected": "Python programming"
        }, "Python is a language"),  # String fallback
    ]
    
    for i, (example, pred) in enumerate(test_cases, 1):
        try:
            score = grader(example, pred)
            print(f"Test case {i}: {score:.3f}")
        except Exception as e:
            print(f"Test case {i}: Error - {e}")


def example_6_comparison_with_traditional_dspy():
    """
    Example 6: Comparison with traditional DSPy metrics
    """
    print("\n=== Example 6: Traditional vs Flexible Comparison ===")
    
    # Traditional DSPy way (hardcoded fields)
    print("Traditional DSPy metrics require specific field names:")
    print("- example.question (hardcoded)")
    print("- example.response (hardcoded)")  
    print("- pred.response (hardcoded)")
    print("- pred.context (hardcoded)")
    
    print("\nFlexible DSPy graders work with any field names:")
    
    # Your data with custom field names
    custom_data = {
        "user_query": "What is AI?",
        "reference_text": "AI is artificial intelligence",
        "domain": "technology"
    }
    
    custom_prediction = {
        "system_response": "AI refers to artificial intelligence technology",
        "confidence_score": 0.9,
        "processing_time": "50ms"
    }
    
    # Flexible grader adapts to your naming
    flexible_grader = SemanticSimilarityGrader(
        pred_field="system_response",    # Matches your prediction
        ideal_field="reference_text"     # Matches your data
    )
    
    score = flexible_grader(custom_data, custom_prediction)
    print(f"Flexible grader score: {score:.3f}")
    
    # Same grader works with completely different field names
    different_data = {
        "gold_answer": "AI is machine intelligence",
        "category": "tech"
    }
    
    different_pred = {
        "generated_text": "AI means artificial intelligence",
        "model": "gpt-4"
    }
    
    different_grader = SemanticSimilarityGrader(
        pred_field="generated_text",
        ideal_field="gold_answer"
    )
    
    score2 = different_grader(different_data, different_pred)
    print(f"Same logic, different fields: {score2:.3f}")


if __name__ == "__main__":
    # Configure DSPy (uncomment when running with actual models)
    # lm = dspy.OpenAI(model="gpt-4o-mini")
    # dspy.configure(lm=lm)
    
    # Run examples
    example_1_basic_flexibility()
    example_2_dspy_program_integration()
    example_3_customer_support_scenario()
    example_4_optimization_capability()
    example_5_handling_edge_cases()
    example_6_comparison_with_traditional_dspy()
    
    print("\n=== Summary ===")
    print("✅ Flexible field extraction - works with any naming scheme")
    print("✅ DSPy.Module compatibility - can be optimized")
    print("✅ Graceful error handling - handles missing fields")
    print("✅ Composite grading - combine multiple criteria")
    print("✅ Domain-specific helpers - pre-built for common use cases")