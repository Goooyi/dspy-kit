"""Domain-specific evaluation graders for specialized use cases."""

# Import domain-specific graders
__all__ = []

# Try to import customer support domain
try:
    from .customer_support import *
    from . import customer_support
    if hasattr(customer_support, '__all__'):
        __all__.extend(customer_support.__all__)
except ImportError:
    pass

# Try to import QA domain  
try:
    from .qa import *
    from . import qa
    if hasattr(qa, '__all__'):
        __all__.extend(qa.__all__)
except ImportError:
    pass

# Try to import summarization domain
try:
    from .summarization import *
    from . import summarization
    if hasattr(summarization, '__all__'):
        __all__.extend(summarization.__all__)
except ImportError:
    pass

# Version info
__version__ = "0.1.0"