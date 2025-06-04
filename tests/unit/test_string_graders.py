"""Comprehensive unit tests for string-based graders."""

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
from dspy_kit import (
    StringCheckGrader,
    TextSimilarityGrader, 
    ExactMatchGrader,
    ContainsGrader,
    StartsWithGrader,
    RegexGrader,
    MultiFieldGrader,
    create_exact_match,
    create_fuzzy_match,
    create_contains_check
)


class MockExample:
    """Mock example object for testing."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockPrediction:
    """Mock prediction object for testing."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class TestStringCheckGrader:
    """Test StringCheckGrader with all operations."""
    
    @pytest.mark.parametrize("operation,input_val,reference_val,expected", [
        # Exact match (eq)
        ("eq", "hello world", "hello world", 1.0),
        ("eq", "Hello World", "hello world", 0.0),
        ("eq", "hello", "world", 0.0),
        ("eq", "", "", 1.0),
        
        # Not equal (ne)
        ("ne", "hello world", "hello world", 0.0),
        ("ne", "Hello World", "hello world", 1.0),
        ("ne", "hello", "world", 1.0),
        
        # Contains (like)
        ("like", "hello world", "world", 1.0),
        ("like", "hello world", "World", 0.0),
        ("like", "hello world", "goodbye", 0.0),
        ("like", "hello", "hello world", 0.0),
        
        # Case-insensitive contains (ilike)
        ("ilike", "hello world", "World", 1.0),
        ("ilike", "Hello World", "world", 1.0),
        ("ilike", "hello world", "goodbye", 0.0),
        
        # Starts with
        ("startswith", "hello world", "hello", 1.0),
        ("startswith", "hello world", "Hello", 0.0),
        ("startswith", "hello world", "world", 0.0),
        
        # Ends with
        ("endswith", "hello world", "world", 1.0),
        ("endswith", "hello world", "World", 0.0),
        ("endswith", "hello world", "hello", 0.0),
        
        # Regex
        ("regex", "hello123", r"\d+", 1.0),
        ("regex", "hello world", r"\d+", 0.0),
        ("regex", "test@email.com", r".*@.*\..*", 1.0),
    ])
    def test_operations(self, operation, input_val, reference_val, expected):
        """Test all string check operations."""
        grader = StringCheckGrader(
            operation=operation,
            pred="output",
            ideal="answer"
        )
        
        example = MockExample(answer=reference_val)
        pred = MockPrediction(output=input_val)
        
        # Test evaluation mode
        score = grader(example, pred, trace=None)
        assert score == expected
        
        # Test optimization mode
        result = grader(example, pred, trace={})
        assert result == (expected == 1.0)
    
    def test_case_sensitivity(self):
        """Test case sensitivity settings."""
        # Case sensitive (default)
        grader_sensitive = StringCheckGrader(
            operation="eq",
            case_sensitive=True
        )
        
        example = MockExample(answer="Hello")
        pred = MockPrediction(output="hello")
        
        assert grader_sensitive(example, pred) == 0.0
        
        # Case insensitive
        grader_insensitive = StringCheckGrader(
            operation="eq",
            case_sensitive=False
        )
        
        assert grader_insensitive(example, pred) == 1.0
    
    def test_normalization(self):
        """Test text normalization options."""
        grader = StringCheckGrader(
            operation="eq",
            normalize_whitespace=True,
            strip_text=True
        )
        
        example = MockExample(answer="hello world")
        pred = MockPrediction(output="  hello    world  ")
        
        assert grader(example, pred) == 1.0
    
    def test_invalid_operation(self):
        """Test invalid operation raises error."""
        with pytest.raises(ValueError, match="Unsupported operation"):
            StringCheckGrader(operation="invalid_op")
    
    def test_field_extraction(self):
        """Test different field extraction scenarios."""
        grader = StringCheckGrader(operation="eq")
        
        # Object with attributes
        example = MockExample(answer="test")
        pred = MockPrediction(output="test")
        assert grader(example, pred) == 1.0
        
        # Dictionary-like objects
        example_dict = {"answer": "test"}
        pred_dict = {"output": "test"}
        score = grader._extract_and_normalize(example_dict, "answer")
        assert score == "test"
    
    def test_error_handling(self):
        """Test error handling in string comparison."""
        grader = StringCheckGrader(operation="eq")
        
        # Missing fields should be handled gracefully
        example = MockExample()  # No answer field
        pred = MockPrediction()  # No output field
        
        score = grader(example, pred)
        assert score == 1.0  # Empty strings match


class TestTextSimilarityGrader:
    """Test TextSimilarityGrader with different metrics."""
    
    def test_fuzzy_match(self):
        """Test fuzzy matching."""
        grader = TextSimilarityGrader(
            metric="fuzzy_match",
            threshold=0.8
        )
        
        example = MockExample(answer="hello world")
        
        # High similarity
        pred_similar = MockPrediction(output="hello word")
        score = grader(example, pred_similar)
        assert score > 0.7
        
        # Low similarity
        pred_different = MockPrediction(output="completely different")
        score = grader(example, pred_different)
        assert score < 0.5
    
    def test_jaccard_similarity(self):
        """Test Jaccard similarity."""
        grader = TextSimilarityGrader(metric="jaccard")
        
        example = MockExample(answer="hello world test")
        pred = MockPrediction(output="hello test case")
        
        score = grader(example, pred)
        assert 0.0 <= score <= 1.0
    
    def test_levenshtein_similarity(self):
        """Test Levenshtein similarity."""
        grader = TextSimilarityGrader(metric="levenshtein")
        
        example = MockExample(answer="kitten")
        pred = MockPrediction(output="sitting")
        
        score = grader(example, pred)
        assert 0.0 <= score <= 1.0
    
    def test_threshold_behavior(self):
        """Test threshold behavior in trace mode."""
        grader = TextSimilarityGrader(
            metric="fuzzy_match",
            threshold=0.8
        )
        
        example = MockExample(answer="hello world")
        pred_high = MockPrediction(output="hello world!")
        pred_low = MockPrediction(output="goodbye")
        
        # High similarity should pass threshold
        assert grader(example, pred_high, trace={}) == True
        
        # Low similarity should fail threshold
        assert grader(example, pred_low, trace={}) == False
    
    def test_normalization(self):
        """Test text normalization."""
        grader = TextSimilarityGrader(
            metric="fuzzy_match",
            normalize_text=True
        )
        
        example = MockExample(answer="Hello  World")
        pred = MockPrediction(output="hello world")
        
        score = grader(example, pred)
        assert score > 0.9  # Should be very high after normalization
    
    def test_empty_strings(self):
        """Test handling of empty strings."""
        grader = TextSimilarityGrader(metric="fuzzy_match")
        
        example = MockExample(answer="")
        pred = MockPrediction(output="")
        
        score = grader(example, pred)
        assert score == 0.0  # Empty strings should return 0
    
    def test_unsupported_metric(self):
        """Test unsupported metric raises error."""
        with pytest.raises(ValueError, match="Unsupported metric"):
            TextSimilarityGrader(metric="unsupported_metric")
    
    @pytest.mark.parametrize("metric", [
        "fuzzy_match", "jaccard", "levenshtein"
    ])
    def test_all_metrics(self, metric):
        """Test all supported metrics work."""
        grader = TextSimilarityGrader(metric=metric)
        
        example = MockExample(answer="test string")
        pred = MockPrediction(output="test string")
        
        score = grader(example, pred)
        assert 0.0 <= score <= 1.0


class TestSpecializedGraders:
    """Test specialized string graders."""
    
    def test_exact_match_grader(self):
        """Test ExactMatchGrader."""
        grader = ExactMatchGrader()
        
        example = MockExample(answer="exact match")
        pred_match = MockPrediction(output="exact match")
        pred_no_match = MockPrediction(output="no match")
        
        assert grader(example, pred_match) == 1.0
        assert grader(example, pred_no_match) == 0.0
    
    def test_contains_grader(self):
        """Test ContainsGrader."""
        grader = ContainsGrader()
        
        example = MockExample(answer="substring")
        pred_contains = MockPrediction(output="this contains substring here")
        pred_not_contains = MockPrediction(output="this does not have it")
        
        assert grader(example, pred_contains) == 1.0
        assert grader(example, pred_not_contains) == 0.0
    
    def test_starts_with_grader(self):
        """Test StartsWithGrader."""
        grader = StartsWithGrader()
        
        example = MockExample(answer="hello")
        pred_starts = MockPrediction(output="hello world")
        pred_not_starts = MockPrediction(output="world hello")
        
        assert grader(example, pred_starts) == 1.0
        assert grader(example, pred_not_starts) == 0.0
    
    def test_regex_grader(self):
        """Test RegexGrader."""
        grader = RegexGrader()
        
        example = MockExample(answer=r"\d{3}-\d{3}-\d{4}")  # Phone pattern
        pred_match = MockPrediction(output="123-456-7890")
        pred_no_match = MockPrediction(output="not a phone")
        
        assert grader(example, pred_match) == 1.0
        assert grader(example, pred_no_match) == 0.0


class TestMultiFieldGrader:
    """Test MultiFieldGrader."""
    
    def test_multi_field_evaluation(self):
        """Test evaluation across multiple fields."""
        field_graders = {
            "title": {
                "type": "StringCheckGrader",
                "params": {"operation": "eq", "pred": "title", "ideal": "expected_title"}
            },
            "content": {
                "type": "TextSimilarityGrader", 
                "params": {"metric": "fuzzy_match", "pred": "content", "ideal": "expected_content"}
            }
        }
        
        grader = MultiFieldGrader(
            field_graders=field_graders,
            aggregation="average"
        )
        
        example = MockExample(
            expected_title="Test Title",
            expected_content="Test content here"
        )
        pred = MockPrediction(
            title="Test Title",  # Exact match
            content="Test content here!"  # Close match
        )
        
        score = grader(example, pred)
        assert 0.8 <= score <= 1.0  # Should be high
    
    def test_aggregation_methods(self):
        """Test different aggregation methods."""
        field_graders = {
            "field1": {
                "type": "StringCheckGrader",
                "params": {"operation": "eq", "pred": "field1", "ideal": "ref1"}
            },
            "field2": {
                "type": "StringCheckGrader",
                "params": {"operation": "eq", "pred": "field2", "ideal": "ref2"}
            }
        }
        
        example = MockExample(ref1="match", ref2="no match")
        pred = MockPrediction(field1="match", field2="different")
        
        # Test average aggregation
        grader_avg = MultiFieldGrader(field_graders=field_graders, aggregation="average")
        assert grader_avg(example, pred) == 0.5
        
        # Test min aggregation
        grader_min = MultiFieldGrader(field_graders=field_graders, aggregation="min")
        assert grader_min(example, pred) == 0.0
        
        # Test max aggregation
        grader_max = MultiFieldGrader(field_graders=field_graders, aggregation="max")
        assert grader_max(example, pred) == 1.0


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_exact_match(self):
        """Test create_exact_match function."""
        grader = create_exact_match(pred="answer", ideal="gold")
        
        example = MockExample(gold="test")
        pred = MockPrediction(answer="test")
        
        assert grader(example, pred) == 1.0
    
    def test_create_fuzzy_match(self):
        """Test create_fuzzy_match function."""
        grader = create_fuzzy_match(threshold=0.8)
        
        example = MockExample(answer="hello world")
        pred = MockPrediction(output="hello word")  # Typo
        
        score = grader(example, pred)
        assert score > 0.8
    
    def test_create_contains_check(self):
        """Test create_contains_check function."""
        grader = create_contains_check()
        
        example = MockExample(answer="key phrase")
        pred = MockPrediction(output="This contains the key phrase somewhere")
        
        assert grader(example, pred) == 1.0


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_missing_fields_graceful_handling(self):
        """Test graceful handling of missing fields."""
        grader = StringCheckGrader(operation="eq")
        
        # Test with objects that don't have the expected fields
        example = {}
        pred = {}
        
        # Should not crash, should handle gracefully
        try:
            score = grader(example, pred)
            assert isinstance(score, (float, bool))
        except Exception as e:
            # If it raises an exception, it should be informative
            assert "field" in str(e).lower()
    
    def test_none_values(self):
        """Test handling of None values."""
        grader = StringCheckGrader(operation="eq")
        
        example = MockExample(answer=None)
        pred = MockPrediction(output=None)
        
        score = grader(example, pred)
        assert score == 1.0  # None == None should be True
    
    def test_numeric_values(self):
        """Test handling of numeric values."""
        grader = StringCheckGrader(operation="eq")
        
        example = MockExample(answer=123)
        pred = MockPrediction(output="123")
        
        score = grader(example, pred)
        assert score == 1.0  # Should convert to string and match


class TestConfiguration:
    """Test configuration-driven graders."""
    
    def test_configurable_grader_inheritance(self):
        """Test that graders properly inherit from ConfigurableGrader."""
        grader = StringCheckGrader(
            operation="eq",
            pred="custom_field",
            ideal="custom_ref"
        )
        
        # Check that configuration was applied
        assert getattr(grader, 'operation', None) == "eq"
        assert getattr(grader, 'pred', None) == "custom_field"
        assert getattr(grader, 'ideal', None) == "custom_ref"
    
    def test_default_config_override(self):
        """Test overriding default configuration."""
        # Test with custom config
        grader = TextSimilarityGrader(
            metric="jaccard",
            threshold=0.9,
            normalize_text=False
        )
        
        assert getattr(grader, 'metric', None) == "jaccard"
        assert getattr(grader, 'threshold', None) == 0.9
        assert getattr(grader, 'normalize_text', None) == False