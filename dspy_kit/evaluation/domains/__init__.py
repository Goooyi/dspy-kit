"""Domain-specific evaluation graders for specialized use cases."""

# Import domain-specific graders with proper error handling
__all__ = []

# Import customer support domain (the only one with actual content)
try:
    from . import customer_support
    from .customer_support import *

    if hasattr(customer_support, "__all__"):
        __all__.extend(customer_support.__all__)
except ImportError:
    pass

# Note: QA and summarization domains are placeholders for future development
# They currently have empty __init__.py files, so we don't import them
# to avoid static analysis warnings

# When these domains are implemented, uncomment and add content:
# try:
#     from . import qa
#     from .qa import *
#     if hasattr(qa, "__all__"):
#         __all__.extend(qa.__all__)
# except ImportError:
#     pass

# try:
#     from . import summarization
#     from .summarization import *
#     if hasattr(summarization, "__all__"):
#         __all__.extend(summarization.__all__)
# except ImportError:
#     pass

# Version info
__version__ = "0.1.0"
