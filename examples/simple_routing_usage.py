#!/usr/bin/env python3
"""
Simple example showing how to use the routing template with DSPy
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "dspy_kit"))

import dspy
from dspy_kit.templates import PromptTemplate, create_dspy_signature, create_dspy_module


def main():
    # Load the complex routing template
    template = PromptTemplate.from_file("templates/examples/customer_support_router.yaml")

    print("1️⃣ System Role ✅")
    print("   Defined as highest priority module in template")

    print("\n2️⃣ Context Module ✅")
    print("   Captures customer query, order ID, product name, etc.")

    print("\n3️⃣ Routing Module ✅")
    print("   Classifies queries into: order_inquiry, product_question, general_support")

    print("\n4️⃣ Task Modules ✅")
    print("   - Order inquiry handler (conditional)")
    print("   - Product question handler (conditional)")
    print("   - General support handler (fallback)")

    print("\n5️⃣ Output Module ✅")
    print("   Structured output with query_type, response, needs_human")

    # Create DSPy signature automatically
    print("\n\n🔧 Creating DSPy Signature...")
    CustomerSupportSignature = create_dspy_signature(template)

    print(f"✅ Generated Signature: {CustomerSupportSignature.__name__}")
    print(f"   Inputs: {list(CustomerSupportSignature.input_fields.keys())}")
    print(f"   Outputs: {list(CustomerSupportSignature.output_fields.keys())}")

    # Create DSPy module automatically
    print("\n🚀 Creating DSPy Module...")
    CustomerSupportModule = create_dspy_module(template)

    print(f"✅ Generated Module: {CustomerSupportModule.__name__}")

    # Example usage
    print("\n\n📝 Example Usage:")
    print("-" * 40)
    print("# Configure DSPy")
    print("lm = dspy.OpenAI(model='gpt-4')")
    print("dspy.configure(lm=lm)")
    print()
    print("# Use the generated module")
    print("support_agent = CustomerSupportModule()")
    print("result = support_agent(")
    print("    customer_query='Where is my order?',")
    print("    order_id='ORD-12345'")
    print(")")
    print()
    print("# Access structured output")
    print("print(f'Query Type: {result.query_type}')")
    print("print(f'Response: {result.response}')")
    print("print(f'Needs Human: {result.needs_human}')")

    # Show the rendered prompt
    print("\n\n📄 Rendered Prompt Example:")
    print("-" * 40)
    example_render = template.render(
        customer_query="Where is my order?",
        order_id="ORD-12345",
        query_type="order_inquiry",  # This would be determined by routing
    )
    print(example_render[:500] + "...\n")

    print("🎉 All features working together seamlessly!")
    print("\n💡 The beauty: Just edit the YAML file to change behavior!")


if __name__ == "__main__":
    main()
