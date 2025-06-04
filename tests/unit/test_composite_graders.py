import pytest
from unittest.mock import MagicMock
from dspy_kit import BaseGrader # Assuming CompositeGrader inherits or uses this
from dspy_kit import CompositeGrader # Assuming path
# from .test_string_graders import MockExample, MockPrediction

# Mock Grader
class MockSubGrader(BaseGrader):
    def __init__(self, return_score, name="mock_sub_grader"):
        self.return_score = return_score
        self.name = name
        self.call_history = []

    def __call__(self, example, pred, trace=None):
        self.call_history.append({"example": example, "pred": pred, "trace": trace})
        if trace is not None:
            return self.return_score >= 0.7 # Example strict condition for trace mode
        return self.return_score

@pytest.fixture
def mock_example_pred_comp():
    example = MockExample(field1="a", field2="b")
    pred = MockPrediction(out1="x", out2="y")
    return example, pred

def test_composite_grader_weighted_sum(mock_example_pred_comp):
    example, pred = mock_example_pred_comp

    grader1 = MockSubGrader(return_score=0.8, name="grader1")
    grader2 = MockSubGrader(return_score=0.5, name="grader2")

    composite = CompositeGrader(
        graders={
            "g1": grader1,
            "g2": grader2
        },
        weights={
            "g1": 0.6,
            "g2": 0.4
        }
    )

    expected_score = (0.8 * 0.6) + (0.5 * 0.4) # 0.48 + 0.2 = 0.68
    score = composite(example, pred)
    assert score == pytest.approx(expected_score)

    # Test trace mode
    # g1 trace = True (0.8 >= 0.7)
    # g2 trace = False (0.5 < 0.7)
    # Assuming composite trace mode requires ALL sub-graders to pass in trace mode
    expected_trace_score = True and False
    score_trace_mode = composite(example, pred, trace=True)
    assert isinstance(score_trace_mode, bool)
    assert score_trace_mode == expected_trace_score
    assert grader1.call_history[-1]["trace"] is True
    assert grader2.call_history[-1]["trace"] is True


def test_composite_grader_missing_weights(mock_example_pred_comp):
    example, pred = mock_example_pred_comp
    grader1 = MockSubGrader(return_score=1.0)
    with pytest.raises(ValueError): # Or some default weighting (e.g., equal)
        CompositeGrader(graders={"g1": grader1}, weights={}) # Missing weight for g1

# TODO
# More tests:
# - Different aggregation logic if you support more than weighted sum.
# - Error handling (e.g., sub-grader fails).
# - No graders provided.
