"""
LLM-as-Judge template analyzer.

This module provides LLM-powered analysis and optimization suggestions
for prompt templates, leveraging dspy-kit's grader system.
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import dspy
from dspy_kit.evaluation.graders import ConfigurableGrader, CompositeGrader

from ..core.template import PromptTemplate
from ..core.inheritance import InheritablePromptTemplate


@dataclass
class TemplateAnalysisResult:
    """Result of LLM template analysis."""

    template_name: str
    overall_score: float

    # Quality dimensions
    clarity_score: float
    structure_score: float
    efficiency_score: float
    completeness_score: float

    # Detailed feedback
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    # Optimization recommendations
    module_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    priority_adjustments: Dict[str, int] = field(default_factory=dict)

    # LLM-specific optimizations
    model_specific_tips: Dict[str, List[str]] = field(default_factory=dict)


class TemplateQualitySignature(dspy.Signature):
    """Signature for template quality assessment."""

    template_yaml: str = dspy.InputField(desc="The template YAML content to analyze")
    domain: str = dspy.InputField(
        desc="The domain/use case for the template (e.g., chinese_ecommerce, customer_support)"
    )

    clarity_score: float = dspy.OutputField(desc="Score 0-1 for template clarity and understandability")
    clarity_feedback: str = dspy.OutputField(desc="Detailed feedback on template clarity")

    structure_score: float = dspy.OutputField(desc="Score 0-1 for template structure and organization")
    structure_feedback: str = dspy.OutputField(desc="Detailed feedback on template structure")

    efficiency_score: float = dspy.OutputField(desc="Score 0-1 for token efficiency and conciseness")
    efficiency_feedback: str = dspy.OutputField(desc="Detailed feedback on template efficiency")

    completeness_score: float = dspy.OutputField(desc="Score 0-1 for covering all necessary aspects")
    completeness_feedback: str = dspy.OutputField(desc="Detailed feedback on template completeness")


class TemplateOptimizationSignature(dspy.Signature):
    """Signature for template optimization suggestions."""

    template_yaml: str = dspy.InputField(desc="The template YAML content to optimize")
    quality_analysis: str = dspy.InputField(desc="The quality analysis results")
    target_metrics: str = dspy.InputField(
        desc="Target metrics to optimize for (e.g., token_efficiency, response_quality)"
    )

    module_recommendations: str = dspy.OutputField(desc="JSON list of recommended module changes")
    priority_adjustments: str = dspy.OutputField(desc="JSON dict of module priority adjustments")
    optimization_rationale: str = dspy.OutputField(desc="Explanation of optimization recommendations")


class TemplateImprovementSignature(dspy.Signature):
    """Signature for generating improved template versions."""

    original_template: str = dspy.InputField(desc="The original template YAML")
    optimization_suggestions: str = dspy.InputField(desc="The optimization suggestions from analysis")
    improvement_goals: str = dspy.InputField(
        desc="Specific improvement goals (e.g., reduce tokens by 20%, improve clarity)"
    )

    improved_template: str = dspy.OutputField(desc="The improved template YAML maintaining the same structure")
    change_summary: str = dspy.OutputField(desc="Summary of changes made and their benefits")


class TemplateJudge:
    """
    LLM-as-Judge for template analysis and optimization.

    Provides comprehensive template quality assessment, optimization
    suggestions, and automated improvement generation.
    """

    def __init__(self, model: Optional[dspy.LM] = None):
        """
        Initialize template judge.

        Args:
            model: DSPy language model (uses default if not provided)
        """
        self.model = model or dspy.settings.lm

        # Create assessment modules
        self.quality_assessor = dspy.ChainOfThought(TemplateQualitySignature)
        self.optimizer = dspy.ChainOfThought(TemplateOptimizationSignature)
        self.improver = dspy.ChainOfThought(TemplateImprovementSignature)

    def analyze_template(self, template: PromptTemplate, domain: str = "general") -> TemplateAnalysisResult:
        """
        Perform comprehensive template analysis.

        Args:
            template: Template to analyze
            domain: Domain/use case context

        Returns:
            TemplateAnalysisResult with scores and feedback
        """
        # Convert template to YAML for analysis
        template_yaml = self._template_to_yaml(template)

        # Assess quality
        quality_result = self.quality_assessor(template_yaml=template_yaml, domain=domain)

        # Generate optimization suggestions
        quality_summary = self._summarize_quality(quality_result)
        optimization_result = self.optimizer(
            template_yaml=template_yaml,
            quality_analysis=quality_summary,
            target_metrics="token_efficiency,response_quality,clarity",
        )

        # Parse results
        return self._parse_analysis_results(template.name, quality_result, optimization_result)

    def suggest_improvements(self, template: PromptTemplate, goals: List[str]) -> Tuple[str, str]:
        """
        Generate improved version of template.

        Args:
            template: Template to improve
            goals: List of improvement goals

        Returns:
            Tuple of (improved_template_yaml, change_summary)
        """
        # First analyze the template
        analysis = self.analyze_template(template)

        # Generate improvements
        improvement_result = self.improver(
            original_template=self._template_to_yaml(template),
            optimization_suggestions=json.dumps(
                {
                    "module_recommendations": analysis.module_recommendations,
                    "priority_adjustments": analysis.priority_adjustments,
                    "suggestions": analysis.suggestions,
                }
            ),
            improvement_goals=", ".join(goals),
        )

        return improvement_result.improved_template, improvement_result.change_summary

    def compare_templates(
        self, template_a: PromptTemplate, template_b: PromptTemplate, test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare two template versions.

        Args:
            template_a: First template
            template_b: Second template
            test_cases: Test inputs for comparison

        Returns:
            Comparison results with recommendations
        """
        # Analyze both templates
        analysis_a = self.analyze_template(template_a)
        analysis_b = self.analyze_template(template_b)

        # Compare scores
        comparison = {
            "template_a": {
                "name": template_a.name,
                "overall_score": analysis_a.overall_score,
                "scores": {
                    "clarity": analysis_a.clarity_score,
                    "structure": analysis_a.structure_score,
                    "efficiency": analysis_a.efficiency_score,
                    "completeness": analysis_a.completeness_score,
                },
            },
            "template_b": {
                "name": template_b.name,
                "overall_score": analysis_b.overall_score,
                "scores": {
                    "clarity": analysis_b.clarity_score,
                    "structure": analysis_b.structure_score,
                    "efficiency": analysis_b.efficiency_score,
                    "completeness": analysis_b.completeness_score,
                },
            },
            "recommendation": self._generate_comparison_recommendation(analysis_a, analysis_b),
        }

        return comparison

    def _template_to_yaml(self, template: PromptTemplate) -> str:
        """Convert template to YAML string for analysis."""
        import yaml

        # Extract template configuration
        config = {
            "name": template.name,
            "version": template.version,
            "domain": template.domain,
            "language": template.language,
            "input_schema": template.input_schema,
            "output_schema": template.output_schema,
            "modules": template.modules,
            "tools": template.tools,
            "concatenation_style": template.concatenation_style,
        }

        yaml_content = yaml.dump(config, allow_unicode=True, sort_keys=False)
        return f"---\n{yaml_content}---\n{template.content_template}"

    def _summarize_quality(self, quality_result) -> str:
        """Summarize quality assessment results."""
        return f"""
Quality Assessment Summary:
- Clarity: {quality_result.clarity_score}/1.0 - {quality_result.clarity_feedback}
- Structure: {quality_result.structure_score}/1.0 - {quality_result.structure_feedback}
- Efficiency: {quality_result.efficiency_score}/1.0 - {quality_result.efficiency_feedback}
- Completeness: {quality_result.completeness_score}/1.0 - {quality_result.completeness_feedback}
"""

    def _parse_analysis_results(
        self, template_name: str, quality_result, optimization_result
    ) -> TemplateAnalysisResult:
        """Parse LLM outputs into structured result."""
        # Calculate overall score
        overall_score = (
            quality_result.clarity_score
            + quality_result.structure_score
            + quality_result.efficiency_score
            + quality_result.completeness_score
        ) / 4

        # Extract feedback
        strengths = []
        weaknesses = []

        if quality_result.clarity_score > 0.8:
            strengths.append(f"Clear and understandable: {quality_result.clarity_feedback}")
        else:
            weaknesses.append(f"Clarity issues: {quality_result.clarity_feedback}")

        if quality_result.structure_score > 0.8:
            strengths.append(f"Well structured: {quality_result.structure_feedback}")
        else:
            weaknesses.append(f"Structure issues: {quality_result.structure_feedback}")

        # Parse optimization suggestions
        try:
            module_recommendations = json.loads(optimization_result.module_recommendations)
        except:
            module_recommendations = []

        try:
            priority_adjustments = json.loads(optimization_result.priority_adjustments)
        except:
            priority_adjustments = {}

        return TemplateAnalysisResult(
            template_name=template_name,
            overall_score=overall_score,
            clarity_score=quality_result.clarity_score,
            structure_score=quality_result.structure_score,
            efficiency_score=quality_result.efficiency_score,
            completeness_score=quality_result.completeness_score,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=optimization_result.optimization_rationale.split("\n"),
            module_recommendations=module_recommendations,
            priority_adjustments=priority_adjustments,
        )

    def _generate_comparison_recommendation(
        self, analysis_a: TemplateAnalysisResult, analysis_b: TemplateAnalysisResult
    ) -> str:
        """Generate recommendation based on template comparison."""
        if analysis_a.overall_score > analysis_b.overall_score + 0.1:
            return f"Template A ({analysis_a.template_name}) is recommended - higher overall quality"
        elif analysis_b.overall_score > analysis_a.overall_score + 0.1:
            return f"Template B ({analysis_b.template_name}) is recommended - higher overall quality"
        else:
            # Compare specific dimensions
            if analysis_a.efficiency_score > analysis_b.efficiency_score:
                return f"Template A is more efficient, but both are comparable in quality"
            elif analysis_b.efficiency_score > analysis_a.efficiency_score:
                return f"Template B is more efficient, but both are comparable in quality"
            else:
                return "Both templates are comparable - choose based on specific requirements"


class TemplateQualityGrader(ConfigurableGrader):
    """
    Grader that uses LLM-as-Judge to evaluate template quality.

    Can be used with dspy-kit's evaluation framework.
    """

    def __init__(self, judge: Optional[TemplateJudge] = None, **kwargs):
        """Initialize template quality grader."""
        super().__init__(**kwargs)
        self.judge = judge or TemplateJudge()

    def __call__(self, example: Any, pred: Any, trace: Optional[Any] = None) -> float:
        """
        Grade template quality following DSPy grader interface.

        Args:
            example: Expected template (can be None for standalone evaluation)
            pred: Template to evaluate
            trace: Optional trace

        Returns:
            Score as float (0.0-1.0)
        """
        # Extract template from pred
        if hasattr(pred, "template"):
            template = pred.template
        elif isinstance(pred, PromptTemplate):
            template = pred
        else:
            # Try to parse as template
            return 0.0

        # Analyze template
        domain = getattr(template, "domain", "general")
        analysis = self.judge.analyze_template(template, domain)

        # Store detailed results in config for access
        self.config["last_analysis"] = analysis

        return analysis.overall_score


def create_template_judge(model: Optional[dspy.LM] = None) -> TemplateJudge:
    """
    Create a template judge instance.

    Args:
        model: DSPy language model (optional)

    Returns:
        TemplateJudge instance
    """
    return TemplateJudge(model)
