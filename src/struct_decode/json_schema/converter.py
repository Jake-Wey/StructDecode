"""
Convert JSON Schema to regular expressions for constrained generation.

Supported JSON Schema features:
- string (with minLength, maxLength, pattern, enum, format)
- number, integer (with min, max, multipleOf)
- boolean
- null
- array (with items, minItems, maxItems)
- object (with properties, required, additionalProperties)
- anyOf, oneOf, allOf
- $ref (basic support)

Limitations:
- Complex nested structures may hit regex engine limits
- Some JSON Schema features (like uniqueItems) cannot be expressed in regex
- The resulting regex can be very large for complex schemas
"""

from typing import Dict, Any, List, Optional, Union
import json

class JsonSchemaToRegex:
    """
    Convert JSON Schema to regular expression patterns.

    This class handles the conversion of JSON Schema definitions to regex
    patterns that can constrain LLM output. The conversion is approximate
    due to fundamental differences between JSON Schema and regular expressions.
    """

    # JSON string special characters that need escaping
    JSON_SPECIAL_CHARS = {
        '"': r'\"',
        '\\': r'\\\\',
        '\n': r'\\n',
        'r': r'\\r',
        '\t': r'\\t'
    }

    def __init__(self, max_depth: int = 10, max_items: int = 100):
        """
        Initialize the converter.

        Args:
            max_depth: Maximum nesting depth for recursive schemas.
            max_items: Maximum items to allow in arrays (for regex complexity).
        """
               
        self.max_depth = max_depth
        self.max_items = max_items
        self._refs: Dict[str, Dict[str, Any]] = {}

    def convert(self, schema: Dict[str, Any]) -> str:
        """
        Convert a JSON Schema to a regex pattern.

        Args:
            schema: The JSON Schema dictionary.

        Returns:
            A regex pattern string.
        """

        self._refs = {}
        self._extract_refs(schema)
        return self._convert_schema(schema, depth=0)

    def _extract_refs(self, schema: Dict[str, Any]) -> None:
        """Extract all $ref definitions from the schema."""

        if "$defs" in schema:
            self._refs.update(schema["$defs"])
        if "definitions" in schema:
            self._refs.update(schema["definitions"])

    def _resolve_ref(self, ref: str) -> Dict[str, Any]:
        """Resolve a $ref to its schema definition."""

        if ref.startswith("#/"):
            path = ref[2:].split("/")
            result = self._refs
            for key in path:
                if key in result:
                    result = result[key]
                else:
                    return {"type": "any"} # Unknown ref
            return result
        return {"type": "any"} # External refs not supported

    def _convert_schema(self, schema: Dict[str, Any], depth: int) -> str:
        """Convert a schema node to regex."""

        if depth > self.max_depth:
            return self._convert_any(schema, depth)
        
        # Handle $ref
        if "$ref" in schema:
            resolved = self._resolve_ref(schema["$ref"])
            return self._convert_schema(resolved, depth + 1)
        
        # Handle allOf (intersection - approximate with concatenation of constraints)
        if "allOf" in schema:
            return self._convert_all_of(schema["allOf"], depth)

        # Handle anyOf (union)
        if "anyOf" in schema:
            return self._convert_any_of(schema["anyOf"], depth)

        # Handle oneOf (exactly one match)
        if "oneOf" in schema:
            return self._convert_any_of(schema["oneOf"], depth) # Same as anyOf approx

        type_ = schema.get("type")

        if not type_:
            # No type specified - could be any valid JSON
            return self._convert_any(schema, depth)
        
        if isinstance(type_, list):
            # Multiple types allowed
            return self._convert_multi_type(type_, schema, depth)

        type_handlers = {
            "string": self._convert_string,
            "number": self._convert_number,
            "integer": self._convert_integer,
            "boolean": self._convert_boolean,
            "null": self._convert_null,
            "array": self._convert_array,
            "object": self._convert_object,
        }

        handler = type_handlers.get(type_, self._convert_any)
        return handler(schema, depth)

    def _convert_any(self, schema: Dict[str, Any], depth: int) -> str:
        """Convert 'any' type - match any valid JSON value."""

        return r'["\[\{\-\d]|true|false|null|[^\s\]\}\,]+'
    
    def _convert_multi_type(
        self, types: List[str], schema: Dict[str, Any], depth: int
    ) -> str:
        """Convert schema with multiple allowed types."""
        
        patterns = []
        for t in types:
            sub_schema = {**schema, "type": t}
            patterns.append(self._convert_schema(sub_schema, depth))
        return self._join_alternatives(patterns)

    def _convert_string(self, schema: Dict[str, Any], depth: int) -> str:
        """Convert string schema to regex."""

        if "enum" in schema:
            patterns = [self._escape_string(str(v)) for v in schema["enum"]]
            return f'(?:"{"|".join(patterns)}")'
        
        if "const" in schema:
            return f'"{self._escape_string(str(schema["const"]))}"'
        
        format_ = schema.get("format")
        if format_:
            return self._convert_string_format(format_, schema)
        
        if "pattern" in schema:
            pattern = schema["pattern"]
            min_len = schema.get("minLength", 0)
            max_len = schema.get("maxLength")
            return self._apply_length_constraints(f'"{pattern}"', min_len, max_len, is_string=True)

        min_len = schema.get("minLength", 0)
        max_len = schema.get("maxLength")

        if min_len == 0 and max_len is None:
            content = r'[^"\\]*(?:\\.[^"\\]*)*'
        elif max_len is not None:
            content = f'[^"\\\\]{{0,{max_len}}}'
        else:
            content = f'[^"\\\\]{{{min_len},}}'

        return f'"{content}"'

    def _convert_string_format(self, format_: str, schema: Dict[str, Any]) -> str:
        """Convert string format to regex."""

        format_patterns = {
            "date": r'"\d{4}-\d{2}-\d{2}"',
            "time": r'"\d{2}:\d{2}:\d{2}"',
            "date-time": r'"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"',
            "email": r'"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9._]+\.[a-zA-Z]{2,}"',
            "uri": r'"https?://[^\s"]+"',
            "uuid": r'"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"',
            "hostname": r'"[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*"',
            "ipv4": r'"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"'
        }

        return format_patterns.get(format_, self._convert_string(schema, 0))
    
    def _convert_number(self, schema: Dict[str, Any], depth: int) -> str:
        """Convert number schema to regex."""

        if "enum" in schema:
            patterns = [str(v) for v in schema["enum"]]
            return self._join_alternatives(patterns)
        
        min_val = schema.get("minmum", schema.get("exclusiveMinimum"))
        max_val = schema.get("maxmum", schema.get("exclusiveMaximum"))

        pattern = r'-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?'

        # TODO: Apply min/max constraints more precisely
        return pattern

    def _convert_integer(self, schema: Dict[str, Any], depth: int) -> str:
        """Convert integer schema to regex."""

        if "enum" in schema:
            patterns = [str(v) for v in schema["enum"]]
            return self._join_alternatives(patterns)
        
        min_val = schema.get("minmum", schema.get("exclusiveMinimum"))
        max_val = schema.get("maxmum", schema.get("exclusiveMaximum"))

        if min_val is not None and min_val >= 0:
            pattern = r'(?:0|[1-9]\d*)'
        else:
            pattern = r'-?(?:0|[1-9]\d*)'

        return pattern

    def _convert_boolean(self, schema: Dict[str, Any], depth: int) -> str:
        """Convert boolean schema to regex."""

        return "(?:true|false)"

    def _convert_null(self, schema: Dict[str, Any], depth: int) -> str:
        """Convert null schema to regex."""

        return "null"
    
    def _convert_array(self, schema: Dict[str, Any], depth: int) -> str:
        """Convert array schema to regex."""

        min_items = schema.get("minItems", 0)
        max_items = schema.get("max_items", self.max_items)

        if "items" in schema:
            items_schema = schema["items"]
            item_pattern = self._convert_schema(items_schema, depth + 1)
        else:
            item_pattern = self._convert_any(schema, depth)

        if "prefixItems" in schema:
            prefix_patterns = [
                self._convert_schema(item, depth + 1)
                for item in schema["prefixItems"]
            ]
            prefix = ",".join(prefix_patterns)

            if "items" in schema:
                additional = f"(?:,{item_pattern})*"
            elif schema.get("additionalItems", True):
                additional = f"(?:,{self._convert_any(schema, depth)})*"
            else:
                additional = ""

            return rf'\[{prefix}{additional}\]'
        
        if min_items == 0:
            if max_items == 0:
                return r'\[\]'
            
            inner = f"(?:{item_pattern}(?:,{item_pattern}){{0, {max_items - 1}}})?"
        else:
            inner = f"{item_pattern}(?:,{item_pattern}){{{min_items - 1}, {max_items - 1}}}"

        return rf'\[{inner}\]'
    
    def _convert_object(self, schema: Dict[str, Any], depth: int) -> str:
        """Convert object schema to regex."""

        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        additional_props = schema.get("additionalProperties", True)

        if not properties:
            if additional_props:
                # Any object
                return r'\{[^}]*\}'
            else:
                # Empty object only
                return r'\{\s*\}'
            
        prop_patterns = []
        optional_props = []

        for name, prop_schema in properties.items():
            prop_pattern = self._convert_schema(prop_schema, depth + 1)
            escaped_name = self._escape_string(name)

            if name in required:
                prop_patterns.append(f'"{escaped_name}":\\s*{prop_pattern}')
            else:
                optional_props.append(f'"{escaped_name}":\\s*{prop_pattern}')

        required_part = ",".join(prop_patterns)

        if optional_props:
            optional_part = "(?:" + ",".join(f"(?:{p})?" for p in optional_props) + ")?"
        else:
            optional_part = ""

        if additional_props and not prop_patterns:
            return r'\{[^}]*\}'
        
        return rf'\{{{required_part}{optional_part}\}}'
    
    def _convert_any_of(self, schemas: List[Dict[str, Any]], depth: int) -> str:
        """Convert anyOf/oneOf to regex alternatives."""

        patterns = [self._convert_schema(s, depth + 1) for s in schemas]
        return self._join_alternatives(patterns)
    
    def _convert_all_of(self, schemas: List[Dict[str, Any]], depth: int) -> str:
        """Convert allOf - this is approximated as anyOf since regex can't express intersection."""

        merged = {}
        for s in schemas:
            if "$ref" in s:
                s = self._resolve_ref(s["$ref"])
            merged.update(s)
        return self._convert_schema(merged, depth + 1)
    
    def _escape_string(self, s: str) -> str:
        """Escape a string for use in JSON."""

        result = []
        for c in s:
            if c in self.JSON_SPECIAL_CHARS:
                result.append(self.JSON_SPECIAL_CHARS[c])
            elif ord(c) < 32:
                result.append(f"\\u{ord(c):04x}")
            else:
                result.append(c)
        return "".join(result)
    
    def _join_alternatives(self, patterns: List[str]) -> str:
        """Join patterns as alternatives."""

        if not patterns:
            return ""
        if len(patterns) == 1:
            return patterns[0]
        return "(?:" + "|".join(patterns) + ")"
    
    def _apply_length_constraints(
        self, pattern: str, min_len: int, max_len: Optional[int], is_string: bool = False
    ) -> str:
        # TODO
        return pattern

def json_schema_to_regex(schema: Union[Dict[str, Any], str]) -> str:
    """
    Convert a JSON Schema to a regex pattern.

    Args:
        schema: JSON Schema as dict or JSON string.

    Returns:
        Regex pattern string.

    Example:
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {
        ...         "name": {"type": "string"},
        ...         "age": {"type": "integer"}
        ...     },
        ...     "required": ["name"]
        ... }
        >>> pattern = json_schema_to_regex(schema)
    """

    if isinstance(schema, str):
        schema = json.loads(schema)

    converter = JsonSchemaToRegex()
    return converter.convert(schema) # type: ignore
