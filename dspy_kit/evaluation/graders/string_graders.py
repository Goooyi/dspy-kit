"""String-based graders for exact and fuzzy matching with comprehensive metrics."""

import re
from typing import Any, Optional, Union

from .base import BaseGrader, ConfigurableGrader

# Optional imports with fallbacks
try:
    from rapidfuzz import fuzz, utils
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

try:
    from nltk.translate.bleu_score import sentence_bleu
    from nltk.tokenize import word_tokenize
    from nltk.translate.meteor_score import meteor_score
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


class StringCheckGrader(ConfigurableGrader):
    """
    String comparison grader following OpenAI's string_check pattern.

    Supports operations:
    - eq: exact match (case-sensitive)
    - ne: not equal (case-sensitive)
    - like: contains (case-sensitive)
    - ilike: contains (case-insensitive)
    - startswith: starts with reference
    - endswith: ends with reference
    - regex: regex pattern match

    String check graders are good for scoring straightforward pass or fail answers.
    """

    DEFAULT_CONFIG = {
        "operation": "eq",
        "pred": "output",
        "ideal": "answer",
        "case_sensitive": True,
        "normalize_whitespace": True,
        "strip_text": True
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        operation = getattr(self, 'operation', self.DEFAULT_CONFIG['operation'])
        valid_operations = ["eq", "ne", "like", "ilike", "startswith", "endswith", "regex"]
        if operation not in valid_operations:
            raise ValueError(f"Unsupported operation: {operation}. Valid: {valid_operations}")

    def __call__(self, example: Any, pred: Any, trace: Optional[Any] = None) -> Union[float, bool]:
        try:
            # Extract strings to compare
            pred_field = getattr(self, 'pred', self.DEFAULT_CONFIG['pred'])
            ideal_field = getattr(self, 'ideal', self.DEFAULT_CONFIG['ideal'])

            input_text = self._extract_and_normalize(pred, pred_field)
            reference_text = self._extract_and_normalize(example, ideal_field)

            # Perform comparison based on operation
            result = self._compare_strings(input_text, reference_text)

            return float(result) if trace is None else result

        except Exception as e:
            print(f"StringCheckGrader error: {e}")
            return 0.0 if trace is None else False

    def _extract_and_normalize(self, obj: Any, field: str) -> str:
        """Extract and normalize text field."""
        text = self.extract_field(obj, field)

        strip_text = getattr(self, 'strip_text', self.DEFAULT_CONFIG['strip_text'])
        if strip_text:
            text = text.strip()

        normalize_whitespace = getattr(self, 'normalize_whitespace', self.DEFAULT_CONFIG['normalize_whitespace'])
        if normalize_whitespace:
            text = re.sub(r'\s+', ' ', text)

        case_sensitive = getattr(self, 'case_sensitive', self.DEFAULT_CONFIG['case_sensitive'])
        if not case_sensitive:
            text = text.lower()

        return text

    def _compare_strings(self, input_text: str, reference_text: str) -> bool:
        """Perform string comparison based on operation."""
        operation = getattr(self, 'operation', self.DEFAULT_CONFIG['operation'])
        case_sensitive = getattr(self, 'case_sensitive', self.DEFAULT_CONFIG['case_sensitive'])

        if operation == "eq":
            return input_text == reference_text
        elif operation == "ne":
            return input_text != reference_text
        elif operation == "like":
            return reference_text in input_text
        elif operation == "ilike":
            ref_lower = reference_text.lower() if case_sensitive else reference_text
            input_lower = input_text.lower() if case_sensitive else input_text
            return ref_lower in input_lower
        elif operation == "startswith":
            return input_text.startswith(reference_text)
        elif operation == "endswith":
            return input_text.endswith(reference_text)
        elif operation == "regex":
            flags = 0 if case_sensitive else re.IGNORECASE
            return bool(re.search(reference_text, input_text, flags))
        else:
            return False


class TextSimilarityGrader(ConfigurableGrader):
    """
    Text similarity grader following OpenAI's text_similarity pattern.

    Supports metrics:
    - fuzzy_match: Fuzzy string matching using rapidfuzz
    - bleu: BLEU score for translation quality
    - rouge_1/rouge_2/rouge_l: ROUGE scores for summarization
    - meteor: METEOR score for machine translation
    - cosine: Cosine similarity using embeddings
    - jaccard: Jaccard similarity coefficient
    - levenshtein: Normalized Levenshtein distance
    """

    DEFAULT_CONFIG = {
        "metric": "fuzzy_match",
        "threshold": 0.8,
        "pred": "output",
        "ideal": "answer",
        "normalize_text": True,
        "embedding_model": "all-MiniLM-L6-v2"
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        metric = getattr(self, 'metric', self.DEFAULT_CONFIG['metric'])
        supported_metrics = [
            "fuzzy_match", "bleu", "rouge_1", "rouge_2", "rouge_l",
            "meteor", "cosine", "jaccard", "levenshtein"
        ]
        if metric not in supported_metrics:
            raise ValueError(f"Unsupported metric: {metric}. Supported: {supported_metrics}")

    def __call__(self, example: Any, pred: Any, trace: Optional[Any] = None) -> Union[float, bool]:
        try:
            pred_field = getattr(self, 'pred', self.DEFAULT_CONFIG['pred'])
            ideal_field = getattr(self, 'ideal', self.DEFAULT_CONFIG['ideal'])

            input_text = self.extract_field(pred, pred_field)
            reference_text = self.extract_field(example, ideal_field)

            normalize_text = getattr(self, 'normalize_text', self.DEFAULT_CONFIG['normalize_text'])
            if normalize_text:
                input_text = self._normalize_text(input_text)
                reference_text = self._normalize_text(reference_text)

            # Calculate similarity score
            score = self._calculate_similarity(input_text, reference_text)

            if trace is None:
                return score
            else:
                threshold = getattr(self, 'threshold', self.DEFAULT_CONFIG['threshold'])
                return score >= threshold

        except Exception as e:
            print(f"TextSimilarityGrader error: {e}")
            return 0.0 if trace is None else False

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Convert to lowercase
        text = text.lower()
        return text

    def _calculate_similarity(self, input_text: str, reference_text: str) -> float:
        """Calculate similarity based on the specified metric."""
        if not input_text or not reference_text:
            return 0.0

        metric = getattr(self, 'metric', self.DEFAULT_CONFIG['metric'])

        if metric == "fuzzy_match":
            return self._fuzzy_match_score(input_text, reference_text)
        elif metric == "bleu":
            return self._bleu_score(input_text, reference_text)
        elif metric.startswith("rouge"):
            return self._rouge_score(input_text, reference_text)
        elif metric == "meteor":
            return self._meteor_score(input_text, reference_text)
        elif metric == "cosine":
            return self._cosine_similarity(input_text, reference_text)
        elif metric == "jaccard":
            return self._jaccard_similarity(input_text, reference_text)
        elif metric == "levenshtein":
            return self._levenshtein_similarity(input_text, reference_text)
        else:
            return 0.0

    def _fuzzy_match_score(self, input_text: str, reference_text: str) -> float:
        """Calculate fuzzy match score."""
        if RAPIDFUZZ_AVAILABLE:
            return fuzz.WRatio(input_text, reference_text, processor=utils.default_process) / 100.0
        else:
            # Fallback to simple word overlap
            return self._simple_word_overlap(input_text, reference_text)

    def _bleu_score(self, candidate: str, reference: str) -> float:
        """Calculate BLEU score."""
        if NLTK_AVAILABLE:
            try:
                ref_tokens = [word_tokenize(reference.lower())]
                cand_tokens = word_tokenize(candidate.lower())
                return sentence_bleu(ref_tokens, cand_tokens)
            except Exception:
                return self._simple_bleu(candidate, reference)
        else:
            return self._simple_bleu(candidate, reference)

    def _simple_bleu(self, candidate: str, reference: str) -> float:
        """Simple BLEU-like score without NLTK."""
        cand_words = set(candidate.lower().split())
        ref_words = set(reference.lower().split())

        if not ref_words:
            return 0.0

        overlap = len(cand_words.intersection(ref_words))
        return overlap / len(ref_words)

    def _rouge_score(self, candidate: str, reference: str) -> float:
        """Calculate ROUGE score."""
        if ROUGE_AVAILABLE:
            try:
                metric = getattr(self, 'metric', self.DEFAULT_CONFIG['metric'])
                rouge_type = "rouge1" if metric == "rouge_1" else \
                            "rouge2" if metric == "rouge_2" else "rougeL"

                scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
                scores = scorer.score(reference, candidate)
                return scores[rouge_type].fmeasure
            except Exception:
                return self._simple_rouge(candidate, reference)
        else:
            return self._simple_rouge(candidate, reference)

    def _simple_rouge(self, candidate: str, reference: str) -> float:
        """Simple ROUGE-like score without rouge-score package."""
        cand_words = candidate.lower().split()
        ref_words = reference.lower().split()

        if not ref_words:
            return 0.0

        overlap = sum(1 for word in cand_words if word in ref_words)
        return overlap / len(ref_words)

    def _meteor_score(self, candidate: str, reference: str) -> float:
        """Calculate METEOR score."""
        if NLTK_AVAILABLE:
            try:
                return meteor_score([reference.split()], candidate.split())
            except Exception:
                return self._fuzzy_match_score(candidate, reference)
        else:
            return self._fuzzy_match_score(candidate, reference)

    def _cosine_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                import numpy as np
                embedding_model = getattr(self, 'embedding_model', self.DEFAULT_CONFIG['embedding_model'])
                model = SentenceTransformer(embedding_model)
                embeddings = model.encode([text1, text2])

                return float(
                    np.dot(embeddings[0], embeddings[1]) /
                    (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
                )
            except Exception:
                return self._tfidf_cosine_similarity(text1, text2)
        else:
            return self._tfidf_cosine_similarity(text1, text2)

    def _tfidf_cosine_similarity(self, text1: str, text2: str) -> float:
        """TF-IDF based cosine similarity fallback."""
        if SKLEARN_AVAILABLE:
            try:
                vectorizer = TfidfVectorizer()
                tfidf = vectorizer.fit_transform([text1, text2])
                return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
            except Exception:
                return self._simple_word_overlap(text1, text2)
        else:
            return self._simple_word_overlap(text1, text2)

    def _simple_word_overlap(self, text1: str, text2: str) -> float:
        """Simple word overlap similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        union = words1.union(words2)
        if not union:
            return 0.0
        return len(words1.intersection(words2)) / len(union)

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def _levenshtein_similarity(self, text1: str, text2: str) -> float:
        """Calculate normalized Levenshtein similarity."""
        def levenshtein_distance(s1, s2):
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)

            if len(s2) == 0:
                return len(s1)

            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row

            return previous_row[-1]

        max_len = max(len(text1), len(text2))
        if max_len == 0:
            return 1.0

        distance = levenshtein_distance(text1, text2)
        return 1.0 - (distance / max_len)


class ExactMatchGrader(StringCheckGrader):
    """Simple exact match grader."""

    DEFAULT_CONFIG = {
        **StringCheckGrader.DEFAULT_CONFIG,
        "operation": "eq"
    }


class ContainsGrader(StringCheckGrader):
    """Check if output contains reference text."""

    DEFAULT_CONFIG = {
        **StringCheckGrader.DEFAULT_CONFIG,
        "operation": "like"
    }


class StartsWithGrader(StringCheckGrader):
    """Check if output starts with reference text."""

    DEFAULT_CONFIG = {
        **StringCheckGrader.DEFAULT_CONFIG,
        "operation": "startswith"
    }


class RegexGrader(StringCheckGrader):
    """Regex pattern matching grader."""

    DEFAULT_CONFIG = {
        **StringCheckGrader.DEFAULT_CONFIG,
        "operation": "regex"
    }


class MultiFieldGrader(ConfigurableGrader):
    """
    Grader that can evaluate multiple fields with different criteria.
    """

    DEFAULT_CONFIG = {
        "field_graders": {},  # Dict of field_name -> grader_config
        "aggregation": "average"  # "average", "min", "max", "all_pass"
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.graders = {}

        # Initialize individual graders for each field
        field_graders = getattr(self, 'field_graders', self.DEFAULT_CONFIG['field_graders'])
        for field_name, grader_config in field_graders.items():
            grader_type = grader_config.get("type", "StringCheckGrader")
            grader_params = grader_config.get("params", {})

            if grader_type == "StringCheckGrader":
                self.graders[field_name] = StringCheckGrader(**grader_params)
            elif grader_type == "TextSimilarityGrader":
                self.graders[field_name] = TextSimilarityGrader(**grader_params)
            else:
                raise ValueError(f"Unsupported grader type: {grader_type}")

    def __call__(self, example: Any, pred: Any, trace: Optional[Any] = None) -> Union[float, bool]:
        scores = []

        for grader in self.graders.values():
            score = grader(example, pred, trace)
            scores.append(score)

        if not scores:
            return 0.0 if trace is None else False

        # Aggregate scores
        aggregation = getattr(self, 'aggregation', self.DEFAULT_CONFIG['aggregation'])
        if aggregation == "average":
            result = sum(scores) / len(scores)
        elif aggregation == "min":
            result = min(scores)
        elif aggregation == "max":
            result = max(scores)
        elif aggregation == "all_pass":
            if trace is None:
                result = 1.0 if all(s >= 0.7 for s in scores) else 0.0
            else:
                result = all(scores)
        else:
            result = sum(scores) / len(scores)

        return result


# Convenience functions
def create_exact_match(pred: str = "output", ideal: str = "answer") -> ExactMatchGrader:
    """Create an exact match grader."""
    return ExactMatchGrader(pred=pred, ideal=ideal)


def create_fuzzy_match(threshold: float = 0.8, pred: str = "output", ideal: str = "answer") -> TextSimilarityGrader:
    """Create a fuzzy match grader."""
    return TextSimilarityGrader(
        metric="fuzzy_match",
        threshold=threshold,
        pred=pred,
        ideal=ideal
    )


def create_contains_check(pred: str = "output", ideal: str = "answer") -> ContainsGrader:
    """Create a contains check grader."""
    return ContainsGrader(pred=pred, ideal=ideal)