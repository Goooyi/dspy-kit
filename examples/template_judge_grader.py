#!/usr/bin/env python3
"""
Example: Using LLM-as-Judge with dspy-kit's grader system.

This example shows how to:
1. Create a composite grader with LLM-as-Judge
2. Evaluate templates using multiple criteria
3. Track improvements over iterations
"""

import sys
from pathlib import Path

# Add dspy_kit to path
sys.path.insert(0, str(Path(__file__).parent.parent / "dspy_kit"))

import dspy
from dspy_kit.evaluation.graders import CompositeGrader
from dspy_kit.templates import InheritablePromptTemplate
from dspy_kit.templates.analysis.template_judge import TemplateQualityGrader
from dspy_kit.templates.evaluation.template_graders import (
    TemplateFormatComplianceGrader,
    ChineseEcommerceGrader
)


def create_comprehensive_template_grader(template=None):
    """Create a comprehensive grader combining multiple evaluation methods."""
    
    # Create individual graders
    quality_grader = TemplateQualityGrader(
        name="llm_quality_judge",
        threshold=0.8
    )
    
    # Combine graders based on what's available
    graders = {"quality": (quality_grader, 1.0)}
    
    # Add template-specific graders if template provided
    if template:
        format_grader = TemplateFormatComplianceGrader(template)
        graders = {
            "quality": (quality_grader, 0.5),
            "format": (format_grader, 0.5)
        }
    
    # Create composite grader
    comprehensive_grader = CompositeGrader(graders)
    
    return comprehensive_grader


def evaluate_template_iterations():
    """Evaluate template improvements over iterations."""
    print("📈 Template Iteration Evaluation")
    print("=" * 50)
    print()
    
    # Create grader
    grader = create_comprehensive_template_grader()
    
    # Simulate evaluating template versions
    versions = [
        {
            "version": "1.0",
            "description": "Original template",
            "scores": {
                "quality": 0.84,
                "format": 0.95,
                "domain": 0.88
            }
        },
        {
            "version": "1.1", 
            "description": "Added error handling",
            "scores": {
                "quality": 0.87,
                "format": 0.95,
                "domain": 0.90
            }
        },
        {
            "version": "2.0",
            "description": "LLM-optimized version",
            "scores": {
                "quality": 0.92,
                "format": 0.93,
                "domain": 0.91
            }
        }
    ]
    
    print("📊 Evaluation Results:")
    print()
    print("Version | Quality | Format | Domain | Overall | Status")
    print("--------|---------|--------|--------|---------|--------")
    
    for v in versions:
        # Calculate weighted overall score
        overall = (
            v["scores"]["quality"] * 0.4 +
            v["scores"]["format"] * 0.2 +
            v["scores"]["domain"] * 0.4
        )
        
        status = "✅ Pass" if overall >= 0.75 else "❌ Fail"
        
        print(f"{v['version']:7} | {v['scores']['quality']:7.2f} | "
              f"{v['scores']['format']:6.2f} | {v['scores']['domain']:6.2f} | "
              f"{overall:7.2f} | {status}")
    
    print()
    print("📈 Improvement Summary:")
    print("   • Quality: +9.5% (LLM suggestions helped)")
    print("   • Format: -2.1% (Minor trade-off for efficiency)")
    print("   • Domain: +3.4% (Better Chinese support)")
    print("   • Overall: +5.7% improvement")


def demonstrate_grader_integration():
    """Show how LLM-as-Judge integrates with existing graders."""
    print("\n\n🔧 Grader System Integration")
    print("=" * 50)
    print()
    
    print("📋 Available Graders:")
    print()
    print("1. TemplateQualityGrader (LLM-as-Judge)")
    print("   - Uses LLM to assess template quality")
    print("   - Provides detailed feedback and scores")
    print("   - Suggests specific improvements")
    print()
    print("2. TemplateFormatComplianceGrader")
    print("   - Validates YAML structure")
    print("   - Checks required fields")
    print("   - Ensures proper formatting")
    print()
    print("3. ChineseEcommerceGrader")
    print("   - Domain-specific evaluation")
    print("   - Cultural appropriateness")
    print("   - Business logic validation")
    print()
    print("4. FactualRecallGrader")
    print("   - Information accuracy")
    print("   - Fact preservation")
    print("   - Context consistency")
    
    print("\n🔄 Integration Flow:")
    print("```python")
    print("# 1. Load template")
    print("template = InheritablePromptTemplate.from_file('template.yaml')")
    print()
    print("# 2. Create comprehensive grader")
    print("grader = create_comprehensive_template_grader()")
    print()
    print("# 3. Evaluate template")
    print("result = grader.grade(template)")
    print()
    print("# 4. Get improvement suggestions")
    print("if not result.passed:")
    print("    judge = TemplateJudge()")
    print("    improved = judge.suggest_improvements(")
    print("        template,")
    print("        goals=['improve_score', 'reduce_tokens']")
    print("    )")
    print("```")


def show_mlflow_integration():
    """Show how to track template quality with MLflow."""
    print("\n\n📊 MLflow Integration for Template Tracking")
    print("=" * 50)
    print()
    
    print("🔄 Tracking Template Performance:")
    print()
    print("```python")
    print("import mlflow")
    print()
    print("# Start MLflow run")
    print("with mlflow.start_run():")
    print("    # Log template metadata")
    print("    mlflow.log_param('template_name', template.name)")
    print("    mlflow.log_param('template_version', template.version)")
    print("    mlflow.log_param('inheritance_chain', template.get_inheritance_chain())")
    print("    ")
    print("    # Evaluate with LLM-as-Judge")
    print("    analysis = judge.analyze_template(template)")
    print("    ")
    print("    # Log quality metrics")
    print("    mlflow.log_metric('clarity_score', analysis.clarity_score)")
    print("    mlflow.log_metric('structure_score', analysis.structure_score)")
    print("    mlflow.log_metric('efficiency_score', analysis.efficiency_score)")
    print("    mlflow.log_metric('completeness_score', analysis.completeness_score)")
    print("    mlflow.log_metric('overall_score', analysis.overall_score)")
    print("    ")
    print("    # Log suggestions as artifacts")
    print("    with open('suggestions.txt', 'w') as f:")
    print("        f.write('\\n'.join(analysis.suggestions))")
    print("    mlflow.log_artifact('suggestions.txt')")
    print("```")
    
    print("\n📈 Benefits:")
    print("   • Track template quality over time")
    print("   • Compare different versions")
    print("   • Identify improvement trends")
    print("   • Data-driven optimization")


def main():
    """Run the grader integration example."""
    print("🧑‍⚖️ LLM-as-Judge + dspy-kit Grader Integration")
    print("=" * 70)
    print()
    
    try:
        # Demonstrate features
        evaluate_template_iterations()
        demonstrate_grader_integration()
        show_mlflow_integration()
        
        print("\n\n✅ Integration example completed!")
        print("\n💡 Key Takeaways:")
        print("   • LLM-as-Judge seamlessly integrates with dspy-kit graders")
        print("   • Provides comprehensive template evaluation")
        print("   • Enables data-driven template optimization")
        print("   • Works with existing MLflow tracking")
        print("   • Supports continuous improvement workflow")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()