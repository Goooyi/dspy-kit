#!/usr/bin/env python3
"""Test to examine template structure after loading."""

import sys
from pathlib import Path

# Add dspy_kit to path
sys.path.insert(0, str(Path(__file__).parent / "dspy_kit"))

from dspy_kit.templates import InheritablePromptTemplate
import os

# Change to dspy-kit directory
os.chdir(str(Path(__file__).parent))

# Load a template
template = InheritablePromptTemplate.from_file("templates/shops/example_shop_support.yaml")

# Print all attributes
print("Template attributes:")
for attr in dir(template):
    if not attr.startswith("_"):
        try:
            value = getattr(template, attr)
            if not callable(value):
                print(f"  {attr}: {value if len(str(value)) < 50 else str(value)[:50] + '...'}")
        except:
            print(f"  {attr}: <error accessing>")
