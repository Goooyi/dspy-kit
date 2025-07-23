"""
Migration utility to convert existing prompts to modular template format.
"""

import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

# Import at runtime to avoid circular imports


@dataclass
class PromptSection:
    """Represents a section of a prompt."""
    name: str
    content: str
    variables: List[str]


class PromptMigrator:
    """Migrates existing prompts to modular template format."""
    
    # Common section patterns for Chinese e-commerce prompts
    SECTION_PATTERNS = {
        'role': [r'----\s*Role:?\s*', r'----\s*角色:?\s*'],
        'background': [r'----\s*Background:?\s*', r'----\s*背景:?\s*'],
        'goals': [r'----\s*Goals?:?\s*', r'----\s*目标:?\s*'],
        'profile': [r'----\s*Profile:?\s*', r'----\s*简介:?\s*'],
        'skills': [r'----\s*Skills?:?\s*', r'----\s*技能:?\s*'],
        'examples': [r'----\s*Examples?:?\s*', r'----\s*示例:?\s*'],
        'workflow': [r'----\s*Workflow:?\s*', r'----\s*工作流程:?\s*'],
        'output_format': [r'----\s*OutputFormat:?\s*', r'----\s*输出格式:?\s*']
    }
    
    # Variable pattern
    VARIABLE_PATTERN = re.compile(r'\{([^}]+)\}')
    
    def migrate_file(self, source_path: str, domain: str = "ecommerce", 
                    language: str = "zh-CN"):
        """Migrate a prompt file to template format."""
        with open(source_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return self.migrate_string(content, domain, language)
    
    def migrate_string(self, prompt_content: str, domain: str = "ecommerce",
                      language: str = "zh-CN"):
        """Migrate a prompt string to template format."""
        # Parse sections
        sections = self._parse_sections(prompt_content)
        
        # Extract variables
        all_variables = self._extract_variables(prompt_content)
        
        # Build input schema
        input_schema = self._build_input_schema(all_variables)
        
        # Build output schema (basic for now)
        output_schema = self._build_output_schema()
        
        # Build modules configuration
        modules = self._build_modules(sections)
        
        # Generate content template
        content_template = self._generate_content_template(sections)
        
        from ..core.template import PromptTemplate
        return PromptTemplate(
            name=f"migrated_{domain}_prompt",
            version="1.0",
            domain=domain,
            language=language,
            input_schema=input_schema,
            output_schema=output_schema,
            modules=modules,
            concatenation_style="sections",
            separator="----",
            include_headers=True,
            content_template=content_template
        )
    
    def _parse_sections(self, content: str) -> List[PromptSection]:
        """Parse content into sections."""
        sections = []
        current_section = None
        current_content = []
        
        lines = content.split('\n')
        
        for line in lines:
            # Check if this line starts a new section
            section_name = self._identify_section(line)
            
            if section_name:
                # Save previous section
                if current_section:
                    section_content = '\n'.join(current_content).strip()
                    variables = self._extract_variables(section_content)
                    sections.append(PromptSection(
                        name=current_section,
                        content=section_content,
                        variables=variables
                    ))
                
                # Start new section
                current_section = section_name
                current_content = []
            else:
                # Add to current section
                if current_section:
                    current_content.append(line)
        
        # Don't forget the last section
        if current_section:
            section_content = '\n'.join(current_content).strip()
            variables = self._extract_variables(section_content)
            sections.append(PromptSection(
                name=current_section,
                content=section_content,
                variables=variables
            ))
        
        return sections
    
    def _identify_section(self, line: str) -> Optional[str]:
        """Identify if a line starts a section."""
        line = line.strip()
        
        for section_name, patterns in self.SECTION_PATTERNS.items():
            for pattern in patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    return section_name
        
        return None
    
    def _extract_variables(self, text: str) -> List[str]:
        """Extract variable names from text."""
        matches = self.VARIABLE_PATTERN.findall(text)
        return list(set(matches))  # Remove duplicates
    
    def _build_input_schema(self, variables: List[str]) -> Dict[str, Any]:
        """Build input schema from variables."""
        schema = {}
        
        for var in variables:
            schema[var] = {
                "type": "string",
                "description": f"Input variable: {var}",
                "required": True
            }
        
        return schema
    
    def _build_output_schema(self) -> Dict[str, Any]:
        """Build basic output schema."""
        return {
            "response": {
                "type": "string", 
                "description": "Generated response"
            }
        }
    
    def _build_modules(self, sections: List[PromptSection]) -> List[Dict[str, Any]]:
        """Build modules configuration from sections with Jinja2 format."""
        modules = []
        
        for i, section in enumerate(sections):
            # Convert variables in module template to Jinja2 format
            converted_content = self._convert_to_jinja2_format(section.content)
            
            module = {
                "name": section.name,
                "priority": i + 1,
                "template": converted_content
            }
            modules.append(module)
        
        return modules
    
    def _generate_content_template(self, sections: List[PromptSection]) -> str:
        """Generate the main content template with Jinja2 variable format."""
        template_parts = []
        
        for section in sections:
            # Use concatenation format
            if section.name:
                template_parts.append(f"---- {section.name.title()}")
            
            # Convert {variable} to {{ variable }} for Jinja2
            content = self._convert_to_jinja2_format(section.content)
            template_parts.append(content)
            template_parts.append("")  # Empty line
        
        return '\n'.join(template_parts).strip()
    
    def _convert_to_jinja2_format(self, content: str) -> str:
        """Convert {variable} format to {{ variable }} format for Jinja2."""        
        def replace_variable(match):
            var_name = match.group(1)
            return f"{{{{ {var_name} }}}}"
        
        # Convert {variable} to {{ variable }}
        converted = re.sub(r'\{([^}]+)\}', replace_variable, content)
        return converted


def migrate_prompt(source_path: str, domain: str = "ecommerce", 
                  language: str = "zh-CN"):
    """Convenience function to migrate a prompt file."""
    migrator = PromptMigrator()
    return migrator.migrate_file(source_path, domain, language)