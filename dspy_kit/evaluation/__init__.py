"""DSPy evaluation module with graders and domain-specific evaluators."""

# Import modules to access their __all__
from . import domains, graders
from .domains import *

# Import all public symbols
from .graders import *

# Build __all__ from available modules
__all__ = []

# Add graders exports
if hasattr(graders, "__all__"):
    __all__.extend(graders.__all__)

# Add domains exports if available
if hasattr(domains, "__all__"):
    __all__.extend(domains.__all__)
