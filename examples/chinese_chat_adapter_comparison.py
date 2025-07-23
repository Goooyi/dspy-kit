#!/usr/bin/env python3
"""
Example: Creating a Chinese ChatAdapter using our i18n Template System

This example shows:
1. How DSPy's original ChatAdapter works (English-only)
2. How to create a Chinese version using our i18n system
3. The benefits of our template-based approach
"""

import sys
from pathlib import Path
from typing import Any, Dict, Type

# Add dspy_kit to path
sys.path.insert(0, str(Path(__file__).parent.parent / "dspy_kit"))

import dspy
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.signatures.signature import Signature

from dspy_kit.templates import MultilingualTemplate, I18nAdapter


# ========================================
# 1. DSPy's Original ChatAdapter (English Only)
# ========================================

class OriginalDSPyApproach:
    """Shows how DSPy's ChatAdapter works - hardcoded English."""
    
    def demo_original_adapter(self):
        print("🔴 DSPy's Original ChatAdapter (English Only)")
        print("=" * 60)
        
        # DSPy's ChatAdapter has hardcoded English messages
        adapter = ChatAdapter()
        
        # Example signature
        class QASignature(dspy.Signature):
            """Answer questions with short factoid answers."""
            question = dspy.InputField()
            answer = dspy.OutputField()
        
        # The format methods return hardcoded English
        print("\n📝 Hardcoded English Messages:")
        print("-" * 40)
        
        # Field description - always in English
        desc = adapter.format_field_description(QASignature)
        print("Field Description:")
        print(desc)
        
        # Task description - always in English  
        task = adapter.format_task_description(QASignature)
        print("\nTask Description:")
        print(task)
        
        # Field structure - always in English
        structure = adapter.format_field_structure(QASignature)
        print("\nField Structure:")
        print(structure[:200] + "...")
        
        print("\n❌ Problems:")
        print("• All prompts are hardcoded in English")
        print("• No way to adapt for Chinese users")
        print("• Cultural tone is fixed (Western style)")
        print("• No localization support")


# ========================================
# 2. Our i18n Template-Based Approach
# ========================================

class I18nChatAdapter:
    """Chinese ChatAdapter using our i18n template system."""
    
    def __init__(self):
        # Create the multilingual chat adapter template
        self.template_data = {
            "name": "chat_adapter_prompts",
            "version": "1.0",
            "default_language": "en",
            "supported_languages": ["en", "zh"],
            
            # Cultural settings
            "languages": {
                "en": {
                    "formality": "direct",
                    "instruction_style": "imperative"
                },
                "zh": {
                    "formality": "professional",
                    "instruction_style": "respectful"
                }
            },
            
            # Multilingual prompts for chat adapter
            "modules": [
                {
                    "name": "field_description_intro",
                    "template": {
                        "en": "Your input fields are:",
                        "zh": "您的输入字段包括："
                    }
                },
                {
                    "name": "output_description_intro", 
                    "template": {
                        "en": "Your output fields are:",
                        "zh": "您需要输出的字段包括："
                    }
                },
                {
                    "name": "structure_intro",
                    "template": {
                        "en": "All interactions will be structured in the following way, with the appropriate values filled in.",
                        "zh": "所有交互将按照以下结构进行，请填入相应的值。"
                    }
                },
                {
                    "name": "task_objective",
                    "template": {
                        "en": "In adhering to this structure, your objective is:",
                        "zh": "遵循此结构，您的任务目标是："
                    }
                },
                {
                    "name": "field_marker_explanation",
                    "template": {
                        "en": "Each field is marked with [[ ## field_name ## ]]",
                        "zh": "每个字段都用 [[ ## 字段名 ## ]] 标记"
                    }
                },
                {
                    "name": "completion_marker",
                    "template": {
                        "en": "[[ ## completed ## ]]",
                        "zh": "[[ ## 已完成 ## ]]"
                    }
                },
                {
                    "name": "example_intro",
                    "template": {
                        "en": "Here's an example:",
                        "zh": "示例如下："
                    }
                },
                {
                    "name": "follow_format_instruction",
                    "template": {
                        "en": "Please follow this exact format in your response.",
                        "zh": "请在您的回复中严格遵循此格式。"
                    }
                }
            ]
        }
        
        self.template = MultilingualTemplate(self.template_data)
    
    def format_field_description(self, signature: Type[Signature], language: str = "zh") -> str:
        """Format field descriptions in the specified language."""
        # Render the template modules
        rendered = self.template.render(language=language)
        
        # Get specific modules
        input_intro = self._get_module_text("field_description_intro", language)
        output_intro = self._get_module_text("output_description_intro", language)
        
        # Format field descriptions
        input_desc = self._format_fields(signature.input_fields, language)
        output_desc = self._format_fields(signature.output_fields, language)
        
        return f"{input_intro}\n{input_desc}\n\n{output_intro}\n{output_desc}"
    
    def format_task_description(self, signature: Type[Signature], language: str = "zh") -> str:
        """Format task description in the specified language."""
        objective_intro = self._get_module_text("task_objective", language)
        
        # Get instructions - could also be multilingual
        instructions = signature.instructions
        
        # In Chinese, we might want to add more polite phrasing
        if language == "zh":
            return f"{objective_intro} {instructions}\n请您认真完成此任务。"
        else:
            return f"{objective_intro} {instructions}"
    
    def format_field_structure(self, signature: Type[Signature], language: str = "zh") -> str:
        """Format the field structure explanation."""
        structure_intro = self._get_module_text("structure_intro", language)
        marker_explanation = self._get_module_text("field_marker_explanation", language)
        example_intro = self._get_module_text("example_intro", language)
        format_instruction = self._get_module_text("follow_format_instruction", language)
        completion_marker = self._get_module_text("completion_marker", language)
        
        # Build the structure explanation
        parts = [
            structure_intro,
            marker_explanation,
            "",
            example_intro,
            self._format_example_structure(signature, language),
            completion_marker,
            "",
            format_instruction
        ]
        
        return "\n".join(parts)
    
    def _get_module_text(self, module_name: str, language: str) -> str:
        """Extract text from a specific module."""
        for module in self.template.modules:
            if module["name"] == module_name:
                template = module["template"]
                if isinstance(template, dict):
                    return template.get(language, template.get("en", ""))
                return template
        return ""
    
    def _format_fields(self, fields: Dict[str, Any], language: str) -> str:
        """Format field list in the target language."""
        lines = []
        for name, field in fields.items():
            desc = field.description if hasattr(field, 'description') else ""
            if language == "zh":
                lines.append(f"- {name}: {desc}")
            else:
                lines.append(f"- {name}: {desc}")
        return "\n".join(lines)
    
    def _format_example_structure(self, signature: Type[Signature], language: str) -> str:
        """Create an example structure in the target language."""
        examples = []
        
        # Input fields
        for name in signature.input_fields:
            if language == "zh":
                examples.append(f"[[ ## {name} ## ]]\n<您的{name}内容>")
            else:
                examples.append(f"[[ ## {name} ## ]]\n<your {name} here>")
        
        # Output fields  
        for name in signature.output_fields:
            if language == "zh":
                examples.append(f"[[ ## {name} ## ]]\n<生成的{name}内容>")
            else:
                examples.append(f"[[ ## {name} ## ]]\n<generated {name} here>")
                
        return "\n\n".join(examples)


# ========================================
# 3. Advanced i18n Features
# ========================================

class AdvancedI18nChatAdapter(I18nChatAdapter):
    """Shows advanced features like dynamic language switching and cultural adaptations."""
    
    def __init__(self):
        super().__init__()
        
        # Add more sophisticated cultural adaptations
        self.cultural_adaptations = {
            "zh": {
                "thinking_prompt": "让我仔细思考一下这个问题...",
                "uncertainty_expression": "根据我的理解",
                "conclusion_marker": "综上所述",
                "politeness_prefix": "非常感谢您的提问。"
            },
            "en": {
                "thinking_prompt": "Let me think about this...",
                "uncertainty_expression": "Based on my understanding",
                "conclusion_marker": "In conclusion",
                "politeness_prefix": ""  # Less formal in English
            }
        }
    
    def create_culturally_adapted_prompt(self, base_prompt: str, language: str = "zh") -> str:
        """Add cultural adaptations to prompts."""
        adaptations = self.cultural_adaptations.get(language, {})
        
        # Add politeness for Chinese
        if language == "zh" and adaptations.get("politeness_prefix"):
            base_prompt = f"{adaptations['politeness_prefix']}\n\n{base_prompt}"
        
        return base_prompt
    
    def demo_with_dspy_integration(self):
        """Show how this integrates with DSPy programs."""
        print("\n\n🟢 Integration with DSPy Programs")
        print("=" * 60)
        
        # Create a DSPy signature with Chinese descriptions
        class ChineseQASignature(dspy.Signature):
            """回答用户的问题，提供准确的信息。"""
            question = dspy.InputField(desc="用户的问题")
            answer = dspy.OutputField(desc="您的回答")
        
        # Use our i18n adapter
        zh_desc = self.format_field_description(ChineseQASignature, language="zh")
        en_desc = self.format_field_description(ChineseQASignature, language="en")
        
        print("📝 Chinese Version:")
        print("-" * 40)
        print(zh_desc)
        
        print("\n\n📝 English Version:")
        print("-" * 40)  
        print(en_desc)
        
        # Show task descriptions
        print("\n\n📋 Task Descriptions:")
        print("-" * 40)
        print("Chinese:", self.format_task_description(ChineseQASignature, "zh"))
        print("\nEnglish:", self.format_task_description(ChineseQASignature, "en"))


# ========================================
# 4. Template-Based Benefits Demo
# ========================================

def demonstrate_template_benefits():
    """Show the benefits of our template-based approach."""
    print("\n\n✨ Benefits of Template-Based i18n")
    print("=" * 60)
    
    print("\n1️⃣ Easy to Maintain:")
    print("   • All translations in YAML files")
    print("   • No code changes needed for new languages")
    print("   • Version control friendly")
    
    print("\n2️⃣ Cultural Adaptations:")
    print("   • Different formality levels")
    print("   • Culture-specific instruction styles")
    print("   • Flexible tone adjustments")
    
    print("\n3️⃣ Reusable Components:")
    print("   • Share common phrases across adapters")
    print("   • Inherit from base templates")
    print("   • Mix and match modules")
    
    print("\n4️⃣ Dynamic Language Switching:")
    print("   • Change language at runtime")
    print("   • User preference detection")
    print("   • Fallback chains")
    
    # Show how to save as a template file
    print("\n\n📁 Save as Reusable Template:")
    print("-" * 40)
    print("# chat_adapter_i18n.yaml")
    print("---")
    print("name: chat_adapter_i18n")
    print("version: 1.0")
    print("extends: base_adapter_template")
    print("supported_languages: [en, zh, ja, ko]")
    print("...")


# ========================================
# 5. Production Integration Example
# ========================================

class ProductionI18nChatAdapter:
    """Production-ready example with full DSPy integration."""
    
    def __init__(self, template_path: str = "templates/chat_adapter_i18n.yaml"):
        # Load from file in production
        self.adapter = I18nAdapter()
        self.template = None  # Would load from template_path
        
    def create_dspy_module_with_i18n(self, language: str = "zh"):
        """Create a DSPy module that uses i18n prompts."""
        
        class I18nQAModule(dspy.Module):
            def __init__(self, language: str = "zh"):
                super().__init__()
                self.language = language
                self.qa = dspy.ChainOfThought("question -> answer")
                
                # Use i18n adapter for prompts
                self.adapter = AdvancedI18nChatAdapter()
            
            def forward(self, question: str):
                # Get culturally adapted prompts
                prompt = self.adapter.create_culturally_adapted_prompt(
                    question, 
                    self.language
                )
                
                # Use DSPy as normal
                result = self.qa(question=prompt)
                
                return result
        
        return I18nQAModule(language=language)


# ========================================
# Main Demo
# ========================================

def main():
    """Run the complete comparison demo."""
    print("🌏 Chinese ChatAdapter using i18n Templates")
    print("=" * 70)
    print()
    
    # 1. Show original DSPy approach
    original = OriginalDSPyApproach()
    original.demo_original_adapter()
    
    # 2. Show our i18n approach
    print("\n\n🟢 Our i18n Template-Based Approach")
    print("=" * 60)
    
    i18n_adapter = I18nChatAdapter()
    
    # Create a sample signature
    class QASignature(dspy.Signature):
        """Answer questions with short factoid answers."""
        question = dspy.InputField(desc="The question to answer")
        answer = dspy.OutputField(desc="A short factual answer")
    
    # Show Chinese formatting
    print("\n📝 Chinese Chat Adapter Output:")
    print("-" * 40)
    zh_desc = i18n_adapter.format_field_description(QASignature, language="zh")
    print(zh_desc)
    print()
    zh_structure = i18n_adapter.format_field_structure(QASignature, language="zh")
    print(zh_structure)
    
    # 3. Show advanced features
    advanced = AdvancedI18nChatAdapter()
    advanced.demo_with_dspy_integration()
    
    # 4. Show template benefits
    demonstrate_template_benefits()
    
    # 5. Summary
    print("\n\n🎯 Summary: How It Works")
    print("=" * 60)
    print()
    print("1. Create multilingual template with translations")
    print("2. Define cultural adaptations (formality, tone)")
    print("3. Use template.render(language='zh') for Chinese")
    print("4. Integrate with DSPy programs seamlessly")
    print("5. Switch languages dynamically at runtime")
    print()
    print("✅ Result: Fully localized DSPy experience for Chinese users!")


if __name__ == "__main__":
    main()