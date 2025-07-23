"""
JSON Schema definitions for template validation.
"""

from typing import Dict, Any

def get_template_schema() -> Dict[str, Any]:
    """Get the main template JSON schema."""
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Prompt Template Schema",
        "description": "Schema for validating prompt template YAML frontmatter",
        "type": "object",
        "required": ["name", "input_schema", "output_schema"],
        "properties": {
            "name": {
                "type": "string",
                "pattern": "^[a-z0-9_]+$",
                "minLength": 1,
                "maxLength": 100,
                "description": "Template name (lowercase, underscore separated)"
            },
            "version": {
                "type": "string",
                "pattern": "^\\d+\\.\\d+(\\.\\d+)?$",
                "default": "1.0",
                "description": "Template version (semantic versioning)"
            },
            "domain": {
                "type": "string",
                "enum": [
                    "general", 
                    "customer_support", 
                    "e_commerce",
                    "intent_classification",
                    "content_generation",
                    "data_analysis"
                ],
                "default": "general",
                "description": "Application domain"
            },
            "language": {
                "type": "string",
                "enum": ["en", "zh-CN", "zh-TW", "ja", "ko", "fr", "de", "es"],
                "default": "en",
                "description": "Primary language of the template"
            },
            "input_schema": {
                "$ref": "#/definitions/schema_definition",
                "description": "Input parameters schema"
            },
            "output_schema": {
                "$ref": "#/definitions/schema_definition", 
                "description": "Expected output schema"
            },
            "modules": {
                "type": "array",
                "items": {
                    "$ref": "#/definitions/module_definition"
                },
                "description": "Modular components of the template"
            },
            "tools": {
                "type": "array",
                "items": {
                    "type": "string",
                    "pattern": "^[a-z_][a-z0-9_]*$"
                },
                "uniqueItems": True,
                "description": "Tools available to the template"
            },
            "concatenation_style": {
                "type": "string",
                "enum": ["sections", "xml", "minimal"],
                "default": "sections",
                "description": "How to concatenate template modules"
            },
            "separator": {
                "type": "string",
                "default": "----",
                "description": "Separator for section-style concatenation"
            },
            "include_headers": {
                "type": "boolean", 
                "default": True,
                "description": "Whether to include section headers"
            },
            "metadata": {
                "type": "object",
                "properties": {
                    "author": {"type": "string"},
                    "description": {"type": "string"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "created_at": {"type": "string", "format": "date-time"},
                    "updated_at": {"type": "string", "format": "date-time"}
                },
                "description": "Template metadata"
            }
        },
        "definitions": {
            "schema_definition": {
                "type": "object",
                "patternProperties": {
                    "^[a-z_][a-z0-9_]*$": {
                        "$ref": "#/definitions/field_definition"
                    }
                },
                "additionalProperties": False
            },
            "field_definition": {
                "type": "object",
                "required": ["type"],
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["string", "number", "integer", "boolean", "array", "object"]
                    },
                    "required": {
                        "type": "boolean",
                        "default": False
                    },
                    "description": {
                        "type": "string",
                        "description": "Field description"
                    },
                    "default": {
                        "description": "Default value for the field"
                    },
                    "enum": {
                        "type": "array",
                        "description": "Allowed values for the field"
                    },
                    "minLength": {
                        "type": "integer",
                        "minimum": 0
                    },
                    "maxLength": {
                        "type": "integer",
                        "minimum": 0
                    },
                    "pattern": {
                        "type": "string",
                        "format": "regex"
                    }
                },
                "additionalProperties": False
            },
            "module_definition": {
                "type": "object",
                "required": ["name", "template"],
                "properties": {
                    "name": {
                        "type": "string",
                        "pattern": "^[a-z_][a-z0-9_]*$"
                    },
                    "description": {
                        "type": "string"
                    },
                    "priority": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 100
                    },
                    "template": {
                        "type": "string",
                        "minLength": 1
                    },
                    "required_tools": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "conditional": {
                        "type": "string",
                        "description": "Jinja2 condition for module inclusion"
                    }
                },
                "additionalProperties": False
            }
        }
    }

def get_chinese_ecommerce_schema() -> Dict[str, Any]:
    """Get schema specific to Chinese e-commerce domain.""" 
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Chinese E-commerce Template Schema",
        "allOf": [
            {"$ref": "#/definitions/base_template"}
        ],
        "properties": {
            "domain": {
                "const": "e_commerce"
            },
            "language": {
                "enum": ["zh-CN", "zh-TW"]
            },
            "metadata": {
                "type": "object",
                "properties": {
                    "shop_type": {
                        "type": "string",
                        "enum": ["旗舰店", "专卖店", "普通店铺"]
                    },
                    "product_categories": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["产品A", "产品B", "产品C", "配件", "其他"]
                        }
                    },
                    "supported_regions": {
                        "type": "array", 
                        "items": {"type": "string"}
                    }
                }
            },
            "tools": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "get_product_info",
                        "get_shop_activities", 
                        "check_inventory",
                        "get_customer_history",
                        "calculate_shipping",
                        "get_reviews"
                    ]
                }
            }
        },
        "definitions": {
            "base_template": get_template_schema()
        }
    }

def get_input_schema_schema() -> Dict[str, Any]:
    """Get schema for input_schema validation."""
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Input Schema Definition",
        "type": "object",
        "patternProperties": {
            "^[a-z_][a-z0-9_]*$": {
                "type": "object",
                "required": ["type"],
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["string", "number", "integer", "boolean", "array", "object"]
                    },
                    "required": {"type": "boolean"},
                    "description": {"type": "string"},
                    "default": True,
                    "enum": {"type": "array"},
                    "minLength": {"type": "integer", "minimum": 0},
                    "maxLength": {"type": "integer", "minimum": 0}
                }
            }
        },
        "additionalProperties": False
    }

def get_output_schema_schema() -> Dict[str, Any]:
    """Get schema for output_schema validation."""
    return get_input_schema_schema()  # Same structure for now

def get_tool_schema() -> Dict[str, Any]:
    """Get schema for tool definitions."""
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Tool Definition Schema",
        "type": "object",
        "required": ["name", "description", "parameters"],
        "properties": {
            "name": {
                "type": "string",
                "pattern": "^[a-z_][a-z0-9_]*$"
            },
            "description": {
                "type": "string",
                "minLength": 1
            },
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {"const": "object"},
                    "properties": {"type": "object"},
                    "required": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["type", "properties"]
            },
            "provider": {
                "type": "string",
                "enum": ["function", "mcp", "builtin"],
                "default": "function"
            }
        }
    }