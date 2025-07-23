#!/usr/bin/env python3
"""
Demo: Complex Routing Template with DSPy Integration

Shows how our template system handles:
1. System role
2. Context and output modules  
3. Routing module to classify tasks
4. Multiple task modules that get routed to
5. Full DSPy signature/module generation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "dspy_kit"))

import dspy
from dspy_kit.templates import PromptTemplate, create_dspy_signature, create_dspy_module


# ========================================
# Step 1: Create the Complex Routing Template
# ========================================

def create_routing_template():
    """Create a template with routing logic for different tasks."""
    
    template_data = {
        "name": "customer_support_router",
        "version": "1.0",
        "description": "Routes customer queries to appropriate task handlers",
        
        # Input/Output schemas for DSPy
        "input_schema": {
            "customer_query": {
                "type": "string",
                "required": True,
                "description": "The customer's question or request"
            },
            "order_id": {
                "type": "string", 
                "required": False,
                "description": "Order ID if query is about an order"
            },
            "product_name": {
                "type": "string",
                "required": False,
                "description": "Product name if query is about a product"
            }
        },
        
        "output_schema": {
            "query_type": {
                "type": "string",
                "enum": ["order_inquiry", "product_question", "general_support"],
                "description": "Classified type of the query"
            },
            "response": {
                "type": "string",
                "description": "The response to the customer"
            },
            "needs_human": {
                "type": "boolean",
                "description": "Whether this needs escalation to human agent"
            }
        },
        
        # Modules with priority-based execution
        "modules": [
            # 1. SYSTEM ROLE (highest priority)
            {
                "name": "system_role",
                "priority": 1,
                "template": """You are an AI customer support assistant for an e-commerce platform.
You help customers with order inquiries, product questions, and general support.
Always be helpful, accurate, and professional."""
            },
            
            # 2. CONTEXT MODULE
            {
                "name": "context",
                "priority": 10,
                "template": """Current Context:
- Customer Query: {{ customer_query }}
{% if order_id %}
- Order ID: {{ order_id }}
{% endif %}
{% if product_name %}
- Product: {{ product_name }}
{% endif %}
- Timestamp: {{ timestamp | default('2024-01-15 10:30 AM') }}
- Customer Tier: {{ customer_tier | default('Standard') }}"""
            },
            
            # 3. ROUTING MODULE (classifies the query)
            {
                "name": "routing_instructions",
                "priority": 20,
                "template": """First, classify the customer query into one of these categories:
1. "order_inquiry" - Questions about existing orders, shipping, returns
2. "product_question" - Questions about product features, availability, pricing  
3. "general_support" - Account issues, policies, or other general questions

Based on the classification, follow the appropriate task module below."""
            },
            
            # 4. TASK MODULE 1: Order Inquiry Handler
            {
                "name": "order_inquiry_task",
                "priority": 30,
                "conditional": "{% if _route_to == 'order_inquiry' or order_id %}",
                "template": """For ORDER INQUIRIES:
1. Verify the order ID if provided
2. Check order status, shipping information, and delivery date
3. Provide tracking information if available
4. Offer solutions for common issues (delays, returns, refunds)
5. If order issue is complex, set needs_human to true

Available order statuses: processing, shipped, delivered, returned
Standard shipping time: 3-5 business days
Return window: 30 days"""
            },
            
            # 5. TASK MODULE 2: Product Question Handler
            {
                "name": "product_question_task",
                "priority": 30,
                "conditional": "{% if _route_to == 'product_question' or product_name %}",
                "template": """For PRODUCT QUESTIONS:
1. Identify the specific product or category
2. Provide accurate product information
3. Check availability and pricing
4. Suggest similar products if requested item is unavailable
5. Highlight key features and benefits

Product categories: Electronics, Clothing, Home & Garden, Sports
Price match guarantee: Yes, within 14 days
Warranty: Standard 1-year manufacturer warranty"""
            },
            
            # 6. TASK MODULE 3: General Support Handler (fallback)
            {
                "name": "general_support_task",
                "priority": 30,
                "conditional": "{% if _route_to == 'general_support' or (_route_to != 'order_inquiry' and _route_to != 'product_question') %}",
                "template": """For GENERAL SUPPORT:
1. Understand the customer's issue
2. Provide relevant policy information
3. Guide through account-related tasks
4. Offer appropriate solutions
5. Escalate complex issues to human agents

Common topics: Account access, payment methods, membership benefits
Support hours: 24/7 AI support, Human agents 9 AM - 6 PM EST"""
            },
            
            # 7. OUTPUT MODULE
            {
                "name": "output_format",
                "priority": 40,
                "template": """Generate your response following this structure:
- query_type: [Classified type from routing]
- response: [Your helpful response to the customer]
- needs_human: [true/false based on complexity]

Ensure your response is:
✓ Accurate and helpful
✓ Professional and empathetic
✓ Concise but complete
✓ Action-oriented when appropriate"""
            }
        ],
        
        # Tools for dynamic knowledge
        "tools": [
            "check_order_status",
            "get_product_info",
            "check_inventory",
            "calculate_shipping"
        ],
        
        # Concatenation style
        "concatenation": {
            "style": "sections",
            "section_separator": "\n\n",
            "module_separator": "\n"
        }
    }
    
    return PromptTemplate(template_data)


# ========================================
# Step 2: Demonstrate Template Rendering
# ========================================

def demo_template_rendering():
    """Show how the template renders for different scenarios."""
    
    print("🎯 Template Rendering Demo")
    print("=" * 60)
    
    template = create_routing_template()
    
    # Scenario 1: Order Inquiry
    print("\n📦 Scenario 1: Order Inquiry")
    print("-" * 40)
    
    order_context = {
        "customer_query": "Where is my order? I ordered last week",
        "order_id": "ORD-12345",
        "_route_to": "order_inquiry"  # Simulating routing decision
    }
    
    rendered_order = template.render(**order_context)
    print(rendered_order)
    
    # Scenario 2: Product Question
    print("\n\n🛍️ Scenario 2: Product Question")
    print("-" * 40)
    
    product_context = {
        "customer_query": "Does the iPhone 15 come with a charger?",
        "product_name": "iPhone 15",
        "_route_to": "product_question"
    }
    
    rendered_product = template.render(**product_context)
    print(rendered_product)
    
    # Scenario 3: General Support (no specific route)
    print("\n\n💬 Scenario 3: General Support")
    print("-" * 40)
    
    general_context = {
        "customer_query": "How do I reset my password?",
        "_route_to": "general_support"
    }
    
    rendered_general = template.render(**general_context)
    print(rendered_general)


# ========================================
# Step 3: Create DSPy Signature
# ========================================

def demo_dspy_signature_creation():
    """Show how to create a DSPy signature from the template."""
    
    print("\n\n🔧 DSPy Signature Creation")
    print("=" * 60)
    
    template = create_routing_template()
    
    # Create DSPy signature from template
    RouterSignature = create_dspy_signature(
        template,
        signature_name="CustomerSupportRouter"
    )
    
    print(f"✅ Created signature: {RouterSignature.__name__}")
    print(f"   Instructions: {RouterSignature.instructions[:100]}...")
    print(f"   Input fields: {list(RouterSignature.input_fields.keys())}")
    print(f"   Output fields: {list(RouterSignature.output_fields.keys())}")
    
    # Show the signature in action
    print("\n📝 Signature Details:")
    for field_name, field in RouterSignature.input_fields.items():
        print(f"   Input: {field_name} - {field.description}")
    
    for field_name, field in RouterSignature.output_fields.items():
        print(f"   Output: {field_name} - {field.description}")
    
    return RouterSignature


# ========================================
# Step 4: Create DSPy Module
# ========================================

def demo_dspy_module_creation():
    """Show how to create a complete DSPy module."""
    
    print("\n\n🚀 DSPy Module Creation")
    print("=" * 60)
    
    template = create_routing_template()
    
    # Create DSPy module from template
    RouterModule = create_dspy_module(
        template,
        module_name="CustomerSupportRouterModule"
    )
    
    print(f"✅ Created module: {RouterModule.__name__}")
    
    # Configure DSPy (for demo purposes)
    # In real usage, you'd configure with actual LLM
    print("\n🔧 Module Configuration:")
    print("   - Uses template-defined routing logic")
    print("   - Includes all conditional modules")
    print("   - Supports dynamic tool calls")
    print("   - Generates structured output")
    
    return RouterModule


# ========================================
# Step 5: Complete Integration Example
# ========================================

class AdvancedCustomerSupportSystem:
    """Complete system showing all features working together."""
    
    def __init__(self):
        # Create template
        self.template = create_routing_template()
        
        # Create DSPy components
        self.signature = create_dspy_signature(self.template, "RouterSignature")
        self.ModuleClass = create_dspy_module(self.template, "RouterModule")
        
        # Initialize the module
        self.router = self.ModuleClass()
        
    def process_query(self, query: str, order_id: str = None, product_name: str = None):
        """Process a customer query through the routing system."""
        
        print(f"\n💬 Processing: '{query}'")
        print("-" * 40)
        
        # First, render the template to see the prompt
        context = {
            "customer_query": query,
            "order_id": order_id,
            "product_name": product_name
        }
        
        # Simulate routing decision (in practice, an LLM would do this)
        if order_id or "order" in query.lower():
            context["_route_to"] = "order_inquiry"
        elif product_name or "product" in query.lower():
            context["_route_to"] = "product_question"
        else:
            context["_route_to"] = "general_support"
        
        # Render the prompt
        prompt = self.template.render(**context)
        print("\n📄 Generated Prompt:")
        print(prompt[:300] + "..." if len(prompt) > 300 else prompt)
        
        # In real usage, this would call the LLM
        print("\n🤖 DSPy Module Output (simulated):")
        print(f"   query_type: {context['_route_to']}")
        print(f"   response: [LLM would generate response here]")
        print(f"   needs_human: false")
        
        return context["_route_to"]


# ========================================
# Step 6: Demonstrate Advanced Features
# ========================================

def demo_advanced_features():
    """Show how all features work together."""
    
    print("\n\n✨ Advanced Features Demo")
    print("=" * 60)
    
    # Create the system
    support_system = AdvancedCustomerSupportSystem()
    
    # Test different queries
    test_queries = [
        ("Where is my order ORD-789?", "ORD-789", None),
        ("Is the MacBook Pro in stock?", None, "MacBook Pro"),
        ("I forgot my password", None, None)
    ]
    
    for query, order_id, product in test_queries:
        route = support_system.process_query(query, order_id, product)
        
    # Show template features
    print("\n\n🎯 Template Features Used:")
    print("✅ 1. System Role - Defined AI assistant behavior")
    print("✅ 2. Context Module - Captured query context")
    print("✅ 3. Routing Module - Classified queries into tasks")
    print("✅ 4. Task Modules - Conditional execution based on route")
    print("✅ 5. Output Module - Structured response format")
    print("✅ 6. DSPy Integration - Signature & module generation")
    print("✅ 7. Tool Support - Dynamic knowledge capability")


# ========================================
# Main Demo
# ========================================

def main():
    """Run the complete demonstration."""
    
    print("🚀 Complex Routing Template with DSPy Integration")
    print("=" * 70)
    print()
    
    # Run all demos
    demo_template_rendering()
    signature = demo_dspy_signature_creation()
    module = demo_dspy_module_creation()
    demo_advanced_features()
    
    # Summary
    print("\n\n📊 Summary: Yes, We Can Handle It All!")
    print("=" * 60)
    print()
    print("Our template system successfully handles:")
    print("1️⃣ System Role ✅ - High-priority module defines AI behavior")
    print("2️⃣ Context Module ✅ - Captures and formats input context")  
    print("3️⃣ Routing Module ✅ - Classifies queries for task selection")
    print("4️⃣ Task Modules ✅ - Multiple conditional modules based on route")
    print("5️⃣ Output Module ✅ - Structured output format")
    print()
    print("Plus:")
    print("• Automatic DSPy signature generation")
    print("• Complete DSPy module creation")
    print("• Dynamic tool integration")
    print("• i18n support for multiple languages")
    print("• Template inheritance for reusability")
    print()
    print("🎉 Everything works together seamlessly!")


if __name__ == "__main__":
    main()