# DSPy-Kit Examples

This directory contains practical examples demonstrating the key features of dspy-kit template system.

## 🤝 Template + DSPy Perfect Harmony

**File:** `template_dspy_harmony.py`

**Description:** Comprehensive demonstration of how dspy-kit's template validation system works harmoniously with DSPy's built-in schema validation and runtime features.

### Key Features Demonstrated

#### 🔍 Two-Layer Validation Architecture
- **Layer 1:** Template development-time validation (our system)
- **Layer 2:** DSPy runtime validation and execution

#### 💰 Cost-Effective Development
- Fast template validation without LLM calls
- Early error detection before API costs
- Clear error messages with helpful suggestions

#### 🏪 Domain-Specific Intelligence  
- Chinese e-commerce template validation
- Business-specific tool validation
- Industry best practices enforcement

#### 🔧 Tool Integration
- Template tool definitions
- DSPy tool execution integration
- MCP protocol support

### Running the Example

```bash
# Install dependencies
pip install dspy-ai jsonschema jinja2 python-dotenv

# Run the harmony demonstration
python examples/template_dspy_harmony.py
```

### Environment Setup

For full execution demonstration, set up environment variables:

```bash
# OpenAI-compatible API
export OPENAI_API_KEY="your-api-key"
export OPENAI_API_BASE="your-api-base"  # Optional

# For yuzhoumao proxy (if needed)
export http_proxy="http://localhost:8888"
export https_proxy="http://localhost:8888"
```

### Example Output

The harmony demonstration shows:

1. **Template Validation Results** - Detailed validation with errors, warnings, and suggestions
2. **DSPy Integration** - Seamless signature and module creation
3. **Error Prevention** - Before/after comparison showing benefits
4. **Practical Workflow** - Step-by-step development process
5. **Real Execution** - Live demonstration with actual LLM calls (if configured)

### Key Concepts

#### Template Validation Complements DSPy
- Our validation catches issues **before** runtime
- DSPy handles runtime robustness with fallbacks
- Together they provide comprehensive quality assurance

#### Harmonious Integration Benefits
- ✅ Pre-validated templates have consistent structure  
- ✅ Better type hints improve DSPy's JSON fallback success rate
- ✅ Variable consistency reduces runtime errors
- ✅ Tool dependencies verified before execution
- ✅ Domain-specific validation (Chinese e-commerce)

#### Development Workflow
1. Create template with YAML frontmatter + Jinja2
2. Run `template.validate()` for immediate feedback
3. Fix validation errors and warnings
4. Create DSPy signature and module from validated template
5. Test with real LLM (DSPy handles runtime robustness)
6. Deploy with confidence

### Advanced Features

- **Chinese E-commerce Domain Validation** - Specialized rules for Chinese retail
- **Tool Integration** - Built-in e-commerce tools with MCP support  
- **Multi-language Support** - Template validation for different languages
- **Best Practices** - Automated recommendations for template quality

This example demonstrates how template validation and DSPy work together to create a powerful, cost-effective, and reliable LLM application development experience.