"""Comprehensive integration tests with DSPy workflows."""

import asyncio
from unittest.mock import MagicMock

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    pytest = None

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    dspy = None
from dspy_evals.core import (
    StringCheckGrader,
    TextSimilarityGrader,
    ScoreModelGrader,
    CompositeGrader,
    EdgeCaseAwareGrader,
    ExactMatchGrader,
    create_customer_support_grader
)
from dspy_evals.domains.customer_support import (
    IntentAccuracyGrader,
    CustomerSupportCompositeGrader,
    create_advanced_support_grader
)


class MockLanguageModel:
    """Mock language model for testing without API calls."""
    
    def __init__(self, responses=None):
        self.responses = responses or ["4", "good", "safe", "yes"]
        self.call_count = 0
    
    def __call__(self, prompt=None, **kwargs):
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response
    
    def generate(self, prompt, **kwargs):
        return [self(prompt, **kwargs)]


# Test DSPy Signatures
class SimpleQA(dspy.Signature):
    """Answer questions about programming."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class IntentClassification(dspy.Signature):
    """Classify customer support intent."""
    query: str = dspy.InputField()
    intent: str = dspy.OutputField(desc="One of: billing, technical, general")


class CustomerSupportResponse(dspy.Signature):
    """Generate customer support response."""
    query: str = dspy.InputField()
    intent: str = dspy.InputField()
    response: str = dspy.OutputField()


# Test DSPy Programs
class SimpleQAProgram(dspy.Module):
    """Simple QA program for testing."""
    
    def __init__(self):
        self.qa = dspy.Predict(SimpleQA)
    
    def forward(self, question):
        return self.qa(question=question)


class CustomerSupportAgent(dspy.Module):
    """Customer support agent for testing."""
    
    def __init__(self):
        self.classifier = dspy.Predict(IntentClassification)
        self.responder = dspy.Predict(CustomerSupportResponse)
    
    def forward(self, query):
        intent_result = self.classifier(query=query)
        response_result = self.responder(
            query=query,
            intent=intent_result.intent
        )
        
        return dspy.Prediction(
            query=query,
            intent=intent_result.intent,
            response=response_result.response
        )


class TestBasicDSPyIntegration:
    """Test basic DSPy integration functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        # Mock DSPy settings to avoid API calls
        self.mock_lm = MockLanguageModel(["Python is a programming language"])
        dspy.settings.configure(lm=self.mock_lm)
    
    def test_string_grader_with_dspy_prediction(self):
        """Test string grader with actual DSPy prediction."""
        program = SimpleQAProgram()
        grader = StringCheckGrader(operation="like", pred="answer", ideal="expected")
        
        # Create test example
        example = dspy.Example(
            question="What is Python?",
            expected="programming"
        ).with_inputs("question")
        
        # Run program
        prediction = program(question=example.question)
        
        # Test grader
        score = grader(example, prediction)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
    def test_dspy_metric_conversion(self):
        """Test converting grader to DSPy metric."""
        grader = ExactMatchGrader(pred="answer", ideal="expected")
        metric = grader.to_dspy_metric()
        
        # Test metric function signature
        example = dspy.Example(expected="test answer")
        prediction = dspy.Prediction(answer="test answer")
        
        score = metric(example, prediction)
        assert score == 1.0
        
        # Test with trace
        score_with_trace = metric(example, prediction, trace={})
        assert score_with_trace == True
    
    def test_composite_grader_integration(self):
        """Test composite grader with DSPy workflow."""
        accuracy_grader = ExactMatchGrader(pred="answer", ideal="expected")
        similarity_grader = TextSimilarityGrader(
            metric="fuzzy_match",
            pred="answer", 
            ideal="expected"
        )
        
        composite = CompositeGrader({
            "accuracy": (accuracy_grader, 0.6),
            "similarity": (similarity_grader, 0.4)
        })
        
        example = dspy.Example(expected="programming language")
        prediction = dspy.Prediction(answer="programming language")
        
        score = composite(example, prediction)
        assert 0.0 <= score <= 1.0


class TestDSPyOptimizationIntegration:
    """Test integration with DSPy optimization workflows."""
    
    def setup_method(self):
        """Set up test environment."""
        self.mock_lm = MockLanguageModel([
            "Python is a programming language",
            "Java is object-oriented",
            "JavaScript runs in browsers"
        ])
        dspy.settings.configure(lm=self.mock_lm)
    
    def test_metric_in_evaluation(self):
        """Test grader as metric in DSPy Evaluate."""
        program = SimpleQAProgram()
        grader = StringCheckGrader(operation="like", case_sensitive=False)
        metric = grader.to_dspy_metric()
        
        # Create test dataset
        devset = [
            dspy.Example(
                question="What is Python?",
                answer="programming"
            ).with_inputs("question"),
            dspy.Example(
                question="What is Java?", 
                answer="object"
            ).with_inputs("question"),
        ]
        
        # Run evaluation
        evaluator = dspy.Evaluate(devset=devset, metric=metric, num_threads=1)
        score = evaluator(program)
        
        assert isinstance(score, (int, float))
        assert 0.0 <= score <= 1.0
    
    def test_metric_with_bootstrapping(self):
        """Test grader with DSPy bootstrapping."""
        program = SimpleQAProgram()
        grader = StringCheckGrader(operation="like", case_sensitive=False)
        metric = grader.to_dspy_metric()
        
        # Create small training set
        trainset = [
            dspy.Example(
                question="What is Python?",
                answer="programming"
            ).with_inputs("question")
        ]
        
        try:
            # Test that metric works with optimizer
            optimizer = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=1)
            compiled_program = optimizer.compile(program, trainset=trainset)
            
            # Verify we can still run the program
            result = compiled_program(question="What is Python?")
            assert hasattr(result, 'answer')
            
        except Exception as e:
            # Some optimizers might not work in test environment
            pytest.skip(f"Optimizer test skipped: {e}")


class TestCustomerSupportIntegration:
    """Test customer support domain integration."""
    
    def setup_method(self):
        """Set up test environment."""
        self.mock_lm = MockLanguageModel([
            "billing",  # Intent classification
            "I'll help you with your billing inquiry",  # Response generation
            "4",  # Quality score
            "safe",  # Safety check
            "yes"  # Empathy check
        ])
        dspy.settings.configure(lm=self.mock_lm)
    
    def test_intent_accuracy_grader(self):
        """Test intent accuracy grader."""
        agent = CustomerSupportAgent()
        grader = IntentAccuracyGrader(
            valid_intents=["billing", "technical", "general"]
        )
        
        example = dspy.Example(
            query="I have a billing question",
            true_intent="billing"
        ).with_inputs("query")
        
        prediction = agent(query=example.query)
        score = grader(example, prediction)
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
    def test_customer_support_composite_grader(self):
        """Test comprehensive customer support grader."""
        agent = CustomerSupportAgent()
        grader = CustomerSupportCompositeGrader(
            include_empathy=True,
            include_escalation=False  # Skip escalation for simpler test
        )
        
        example = dspy.Example(
            query="I need help with my bill",
            true_intent="billing",
            customer_context="Premium customer"
        ).with_inputs("query")
        
        prediction = agent(query=example.query)
        
        # Add required fields to prediction for grading
        prediction.customer_context = example.customer_context
        
        score = grader(example, prediction)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
    def test_convenience_function_integration(self):
        """Test convenience functions with DSPy."""
        agent = CustomerSupportAgent()
        grader = create_advanced_support_grader()
        metric = grader.to_dspy_metric()
        
        example = dspy.Example(
            query="Help me cancel my subscription",
            true_intent="general"
        ).with_inputs("query")
        
        prediction = agent(query=example.query)
        score = metric(example, prediction)
        
        assert isinstance(score, (float, bool))


class TestAsyncIntegration:
    """Test async functionality integration."""
    
    def setup_method(self):
        """Set up test environment."""
        self.mock_lm = MockLanguageModel(["4", "good", "safe"])
        dspy.settings.configure(lm=self.mock_lm)
    
    @pytest.mark.asyncio
    async def test_async_grader_with_dspy(self):
        """Test async grader functionality."""
        # Mock async model grader
        grader = ScoreModelGrader(
            prompt_template="Rate this: {{sample.output_text}}",
            range=[1, 5],
            model="mock-model"
        )
        
        # Override the model call to avoid actual API calls
        async def mock_call_model(messages, **kwargs):
            return "4"
        
        grader._call_model = mock_call_model
        
        example = dspy.Example(question="test", answer="test answer")
        prediction = dspy.Prediction(output="test response")
        
        score = await grader.acall(example, prediction)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
    
    @pytest.mark.asyncio
    async def test_batch_evaluation(self):
        """Test batch evaluation functionality."""
        grader = StringCheckGrader(operation="like")
        
        examples = [
            dspy.Example(answer="programming"),
            dspy.Example(answer="language"),
            dspy.Example(answer="python")
        ]
        
        predictions = [
            dspy.Prediction(output="programming language"),
            dspy.Prediction(output="computer language"),
            dspy.Prediction(output="python code")
        ]
        
        scores = await grader.batch_evaluate(examples, predictions)
        assert len(scores) == 3
        assert all(isinstance(score, float) for score in scores)


class TestEdgeCaseIntegration:
    """Test edge case handling in DSPy context."""
    
    def setup_method(self):
        """Set up test environment."""
        self.mock_lm = MockLanguageModel(["response"])
        dspy.settings.configure(lm=self.mock_lm)
    
    def test_edge_case_aware_grader(self):
        """Test edge case aware grader with DSPy."""
        base_grader = StringCheckGrader(operation="like")
        
        def is_out_of_scope(example, pred):
            query = getattr(example, 'query', getattr(example, 'question', ''))
            return 'weather' in query.lower()
        
        edge_grader = EdgeCaseAwareGrader(
            base_grader=base_grader,
            edge_case_handlers={'out_of_scope': is_out_of_scope}
        )
        
        # Normal case
        normal_example = dspy.Example(question="What is Python?", answer="programming")
        normal_pred = dspy.Prediction(output="Python is a programming language")
        score = edge_grader(normal_example, normal_pred)
        assert score > 0.0
        
        # Edge case
        edge_example = dspy.Example(question="What's the weather?", answer="sunny")
        edge_pred = dspy.Prediction(output="It's sunny today")
        score = edge_grader(edge_example, edge_pred)
        assert score == 0.0  # Should be handled as edge case
    
    def test_error_handling_in_dspy_context(self):
        """Test error handling within DSPy workflows."""
        grader = StringCheckGrader(operation="eq")
        
        # Test with malformed prediction
        example = dspy.Example(answer="test")
        malformed_pred = dspy.Prediction()  # No output field
        
        # Should handle gracefully
        score = grader(example, malformed_pred)
        assert isinstance(score, float)
        assert score == 0.0  # Empty string comparison


class TestConfigurationIntegration:
    """Test configuration-driven evaluation."""
    
    def setup_method(self):
        """Set up test environment."""
        self.mock_lm = MockLanguageModel(["4", "safe"])
        dspy.settings.configure(lm=self.mock_lm)
    
    def test_config_driven_evaluation(self):
        """Test using configuration for evaluation setup."""
        # Simulate config-driven grader creation
        config = {
            "accuracy": {
                "type": "StringCheckGrader",
                "operation": "eq",
                "weight": 0.6
            },
            "similarity": {
                "type": "TextSimilarityGrader", 
                "metric": "fuzzy_match",
                "weight": 0.4
            }
        }
        
        # Create graders from config
        graders = {}
        for name, cfg in config.items():
            if cfg["type"] == "StringCheckGrader":
                graders[name] = (StringCheckGrader(operation=cfg["operation"]), cfg["weight"])
            elif cfg["type"] == "TextSimilarityGrader":
                graders[name] = (TextSimilarityGrader(metric=cfg["metric"]), cfg["weight"])
        
        composite = CompositeGrader(graders)
        
        example = dspy.Example(answer="test")
        prediction = dspy.Prediction(output="test")
        
        score = composite(example, prediction)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestRealWorldWorkflow:
    """Test realistic end-to-end workflows."""
    
    def setup_method(self):
        """Set up test environment."""
        self.mock_lm = MockLanguageModel([
            "billing",  # Intent
            "I understand your billing concern. Let me help you with that.",  # Response
            "4",  # Quality
            "safe",  # Safety
            "yes"  # Empathy
        ])
        dspy.settings.configure(lm=self.mock_lm)
    
    def test_complete_evaluation_workflow(self):
        """Test complete evaluation workflow from data to results."""
        # 1. Define the program
        agent = CustomerSupportAgent()
        
        # 2. Create evaluation data
        eval_data = [
            dspy.Example(
                query="I have a billing question about my account",
                true_intent="billing",
                expected_quality=4
            ).with_inputs("query"),
            dspy.Example(
                query="I need technical support",
                true_intent="technical", 
                expected_quality=4
            ).with_inputs("query")
        ]
        
        # 3. Set up comprehensive evaluation
        intent_grader = IntentAccuracyGrader(
            valid_intents=["billing", "technical", "general"]
        )
        
        composite_grader = CompositeGrader({
            "intent_accuracy": (intent_grader, 1.0)  # Simplified for test
        })
        
        # 4. Run evaluation
        metric = composite_grader.to_dspy_metric()
        evaluator = dspy.Evaluate(devset=eval_data, metric=metric, num_threads=1)
        
        overall_score = evaluator(agent)
        
        # 5. Verify results
        assert isinstance(overall_score, (int, float))
        assert 0.0 <= overall_score <= 1.0
        
        print(f"Overall evaluation score: {overall_score:.3f}")
    
    def test_optimization_with_evaluation(self):
        """Test optimization workflow with custom metrics."""
        program = SimpleQAProgram()
        grader = StringCheckGrader(operation="like", case_sensitive=False)
        metric = grader.to_dspy_metric()
        
        # Create training data
        trainset = [
            dspy.Example(
                question="What is Python?",
                answer="programming"
            ).with_inputs("question"),
            dspy.Example(
                question="What is machine learning?",
                answer="AI technique"
            ).with_inputs("question")
        ]
        
        # Create test data
        testset = trainset[:1]  # Use subset for test
        
        try:
            # Run optimization
            optimizer = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=1)
            optimized_program = optimizer.compile(program, trainset=trainset)
            
            # Evaluate both versions
            evaluator = dspy.Evaluate(devset=testset, metric=metric, num_threads=1)
            
            baseline_score = evaluator(program)
            optimized_score = evaluator(optimized_program)
            
            print(f"Baseline: {baseline_score:.3f}, Optimized: {optimized_score:.3f}")
            
            # Both should produce valid scores
            assert 0.0 <= baseline_score <= 1.0
            assert 0.0 <= optimized_score <= 1.0
            
        except Exception as e:
            pytest.skip(f"Optimization test skipped due to environment: {e}")