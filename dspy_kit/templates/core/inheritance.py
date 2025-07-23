"""
Template inheritance system.

Implements template inheritance allowing base templates to be extended
by domain-specific and shop-specific templates, enabling code reuse
and maintainable template hierarchies.
"""

import copy
from typing import Dict, Any, List, Optional, Set, Union
from pathlib import Path
import yaml

from .template import PromptTemplate


class TemplateInheritanceError(Exception):
    """Raised when template inheritance encounters an error."""
    pass


class CircularInheritanceError(TemplateInheritanceError):
    """Raised when circular template inheritance is detected."""
    pass


class TemplateResolver:
    """
    Resolves template inheritance chains and merges configurations.
    
    Handles loading parent templates, merging schemas, tools, modules,
    and assembling the final template configuration.
    """
    
    def __init__(self, template_directories: Optional[List[Union[str, Path]]] = None):
        """
        Initialize template resolver.
        
        Args:
            template_directories: List of directories to search for templates
        """
        self.template_directories = template_directories or []
        self._template_cache = {}  # Cache loaded templates
        self._resolution_stack = []  # Track resolution to detect cycles
    
    def resolve_template(self, template_config: Dict[str, Any], 
                        template_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Resolve template inheritance chain and return merged configuration.
        
        Args:
            template_config: Template configuration dict
            template_path: Path to template file (for relative resolution)
            
        Returns:
            Merged template configuration
        """
        if "extends" not in template_config:
            # No inheritance, return as-is
            return template_config
        
        parent_name = template_config["extends"]
        template_name = template_config.get("name", "unknown")
        
        # Check for circular inheritance
        if template_name in self._resolution_stack:
            cycle = " -> ".join(self._resolution_stack + [template_name])
            raise CircularInheritanceError(f"Circular inheritance detected: {cycle}")
        
        try:
            self._resolution_stack.append(template_name)
            
            # Load parent template
            parent_config = self._load_parent_template(parent_name, template_path)
            
            # Recursively resolve parent inheritance
            resolved_parent = self.resolve_template(parent_config, template_path)
            
            # Merge child with resolved parent
            merged_config = self._merge_configurations(resolved_parent, template_config)
            
            return merged_config
            
        finally:
            self._resolution_stack.pop()
    
    def _load_parent_template(self, parent_name: str, 
                             template_path: Optional[Path] = None) -> Dict[str, Any]:
        """Load parent template configuration."""
        # Check cache first
        if parent_name in self._template_cache:
            return copy.deepcopy(self._template_cache[parent_name])
        
        # Search for parent template
        parent_path = self._find_template_file(parent_name, template_path)
        if not parent_path:
            raise TemplateInheritanceError(f"Parent template not found: {parent_name}")
        
        # Load parent template
        try:
            with open(parent_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse YAML frontmatter
            if content.startswith('---'):
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    yaml_content = parts[1]
                    template_content = parts[2].strip()
                    
                    parent_config = yaml.safe_load(yaml_content) or {}
                    parent_config["content_template"] = template_content
                else:
                    raise TemplateInheritanceError(f"Invalid template format: {parent_path}")
            else:
                raise TemplateInheritanceError(f"Template missing frontmatter: {parent_path}")
            
            # Cache the loaded template
            self._template_cache[parent_name] = copy.deepcopy(parent_config)
            
            return parent_config
            
        except Exception as e:
            raise TemplateInheritanceError(f"Failed to load parent template {parent_name}: {e}")
    
    def _find_template_file(self, template_name: str, 
                           current_path: Optional[Path] = None) -> Optional[Path]:
        """Find template file by name."""
        # Try various extensions
        extensions = ['.yaml', '.yml']
        
        search_paths = []
        
        # Add current directory if we have a current path
        if current_path:
            search_paths.append(current_path.parent)
        
        # Add configured template directories
        search_paths.extend([Path(d) for d in self.template_directories])
        
        # Add some default search paths
        search_paths.extend([
            Path.cwd(),
            Path.cwd() / "templates",
            Path.cwd() / "base",
            Path.cwd() / "domains"
        ])
        
        for search_path in search_paths:
            for ext in extensions:
                candidate = search_path / f"{template_name}{ext}"
                if candidate.exists():
                    return candidate
                
                # Also try in subdirectories
                for subdir in ["base", "domains", "shops"]:
                    candidate = search_path / subdir / f"{template_name}{ext}"
                    if candidate.exists():
                        return candidate
        
        return None
    
    def _merge_configurations(self, parent: Dict[str, Any], 
                            child: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge child configuration with parent configuration.
        
        Args:
            parent: Parent template configuration
            child: Child template configuration
            
        Returns:
            Merged configuration
        """
        # Start with parent configuration
        merged = copy.deepcopy(parent)
        
        # Update with child configuration (child overrides parent)
        for key, value in child.items():
            if key == "extends":
                # Don't include extends in final config
                continue
            elif key in ["input_schema", "output_schema"]:
                # Deep merge schemas
                merged[key] = self._merge_schemas(
                    merged.get(key, {}), value
                )
            elif key == "modules":
                # Merge modules with priority ordering and overrides
                merged[key] = self._merge_modules(
                    merged.get(key, []), value
                )
            elif key == "tools":
                # Merge tool lists (union)
                parent_tools = set(merged.get(key, []))
                child_tools = set(value) if value else set()
                merged[key] = list(parent_tools | child_tools)
            elif key == "metadata":
                # Merge metadata
                merged[key] = {
                    **merged.get(key, {}),
                    **value
                }
            elif key == "content_template":
                # Handle content template merging
                merged[key] = self._merge_content_templates(
                    merged.get(key, ""), value
                )
            else:
                # Simple override for other fields
                merged[key] = value
        
        return merged
    
    def _merge_schemas(self, parent_schema: Dict[str, Any], 
                      child_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge schemas (child extends/overrides parent)."""
        merged = copy.deepcopy(parent_schema)
        
        for field_name, field_config in child_schema.items():
            if field_name in merged:
                # Merge field configurations
                merged[field_name] = {
                    **merged[field_name],
                    **field_config
                }
            else:
                # Add new field
                merged[field_name] = copy.deepcopy(field_config)
        
        return merged
    
    def _merge_modules(self, parent_modules: List[Dict[str, Any]], 
                      child_modules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge module lists with priority ordering and override support.
        
        Child modules can:
        1. Add new modules
        2. Override parent modules (same name)
        3. Insert modules at specific priorities
        """
        # Create a mapping of modules by name for easy lookup
        merged_modules = {}
        
        # Add parent modules
        for module in parent_modules:
            module_name = module.get("name", f"unnamed_{len(merged_modules)}")
            merged_modules[module_name] = copy.deepcopy(module)
        
        # Process child modules
        for module in child_modules:
            module_name = module.get("name", f"unnamed_{len(merged_modules)}")
            
            if module.get("override", False) or module_name in merged_modules:
                # Override existing module
                merged_modules[module_name] = copy.deepcopy(module)
            else:
                # Add new module
                merged_modules[module_name] = copy.deepcopy(module)
        
        # Sort modules by priority
        sorted_modules = sorted(
            merged_modules.values(),
            key=lambda m: m.get("priority", 100)
        )
        
        return sorted_modules
    
    def _merge_content_templates(self, parent_content: str, 
                               child_content: str) -> str:
        """
        Merge content templates.
        
        If child has content, it replaces parent content.
        If child is empty, parent content is used.
        """
        if child_content.strip():
            return child_content
        else:
            return parent_content


class InheritablePromptTemplate(PromptTemplate):
    """
    Extended PromptTemplate that supports inheritance.
    
    Templates can extend other templates using the 'extends' field,
    creating a hierarchy of reusable base templates and domain-specific
    extensions.
    """
    
    def __init__(self, resolver: Optional[TemplateResolver] = None, **kwargs):
        """
        Initialize inheritable prompt template.
        
        Args:
            resolver: Template resolver for inheritance
            **kwargs: Template configuration
        """
        self.resolver = resolver or TemplateResolver()
        
        # Resolve inheritance if needed
        if "extends" in kwargs:
            kwargs = self.resolver.resolve_template(kwargs)
        
        # Initialize with resolved configuration
        super().__init__(**kwargs)
    
    @classmethod
    def from_file(cls, template_path: Union[str, Path], 
                  resolver: Optional[TemplateResolver] = None) -> "InheritablePromptTemplate":
        """
        Load inheritable template from file.
        
        Args:
            template_path: Path to template file
            resolver: Template resolver for inheritance
            
        Returns:
            InheritablePromptTemplate instance
        """
        path = Path(template_path)
        
        # Create resolver with template directory in search path
        if resolver is None:
            template_dirs = [str(path.parent)]
            resolver = TemplateResolver(template_dirs)
        
        # Parse template file
        from .parser import TemplateParser
        parser = TemplateParser()
        template = parser.parse_file(path)
        
        # Convert to dict for inheritance resolution
        template_config = template.to_dict()
        
        # Resolve inheritance
        resolved_config = resolver.resolve_template(template_config, path)
        
        return cls(resolver=resolver, **resolved_config)
    
    @classmethod
    def from_string(cls, template_string: str, 
                   resolver: Optional[TemplateResolver] = None) -> "InheritablePromptTemplate":
        """
        Load inheritable template from string.
        
        Args:
            template_string: Template string with YAML frontmatter
            resolver: Template resolver for inheritance
            
        Returns:
            InheritablePromptTemplate instance
        """
        # Parse template string
        from .parser import TemplateParser
        parser = TemplateParser()
        template = parser.parse_string(template_string)
        
        # Convert to dict for inheritance resolution
        template_config = template.to_dict()
        
        # Create resolver if needed
        if resolver is None:
            resolver = TemplateResolver()
        
        # Resolve inheritance
        resolved_config = resolver.resolve_template(template_config)
        
        return cls(resolver=resolver, **resolved_config)
    
    def get_inheritance_chain(self) -> List[str]:
        """
        Get the inheritance chain for this template.
        
        Returns:
            List of template names from base to current
        """
        # This would require tracking during resolution
        # For now, return just this template's name
        return [self.name]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without inheritance info)."""
        result = super().to_dict()
        
        # Add inheritance metadata if available
        inheritance_chain = self.get_inheritance_chain()
        if len(inheritance_chain) > 1:
            result["inheritance_chain"] = inheritance_chain
        
        return result


def create_template_resolver(template_directories: Optional[List[Union[str, Path]]] = None) -> TemplateResolver:
    """
    Create a template resolver with default search paths.
    
    Args:
        template_directories: Additional directories to search for templates
        
    Returns:
        TemplateResolver instance
    """
    default_dirs = [
        "templates",
        "templates/base", 
        "templates/domains",
        "templates/shops",
        "base",
        "domains", 
        "shops"
    ]
    
    search_dirs = default_dirs + (template_directories or [])
    return TemplateResolver(search_dirs)