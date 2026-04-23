#!/usr/bin/env python3
"""
Locate module of the Agentic Mutation Framework (SWE-Mutation).

Restricts the mutation allowlist to files modified in the golden solution,
parses their structure with Tree-sitter to build a structural graph, extracts
execution traces from Fail-to-Pass (F2P) tests, and annotates the graph with
the trace so the Mutation module can target bug-triggering logic precisely.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


LANGUAGE_BY_EXTENSION: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".go": "go",
    ".rs": "rust",
    ".php": "php",
    ".rb": "ruby",
}

# Tree-sitter node type names for class and function/method constructs per language.
# Values: (class_node_type, function_node_type, method_node_type)
# Empty string means the concept does not exist at the top level for that language.
_NODE_TYPES: dict[str, tuple[str, str, str]] = {
    "python":     ("class_definition",  "function_definition", "function_definition"),
    "javascript": ("class_declaration", "function_declaration", "method_definition"),
    "typescript": ("class_declaration", "function_declaration", "method_definition"),
    "java":       ("class_declaration", "",                    "method_declaration"),
    "c":          ("",                  "function_definition", "function_definition"),
    "cpp":        ("class_specifier",   "function_definition", "function_definition"),
    "go":         ("",                  "function_declaration", "method_declaration"),
    "rust":       ("impl_item",         "function_item",       "function_item"),
    "php":        ("class_declaration", "function_definition", "method_declaration"),
    "ruby":       ("class",             "method",              "method"),
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FunctionNode:
    name: str
    start_line: int
    end_line: int
    params: list[str] = field(default_factory=list)
    on_f2p_trace: bool = False


@dataclass
class ClassNode:
    name: str
    start_line: int
    end_line: int
    methods: list[FunctionNode] = field(default_factory=list)
    on_f2p_trace: bool = False


@dataclass
class FileStructure:
    path: str
    language: str
    classes: list[ClassNode] = field(default_factory=list)
    functions: list[FunctionNode] = field(default_factory=list)


@dataclass
class LocateResult:
    allowed_files: list[str]
    structures: list[FileStructure]
    f2p_tests: list[str]

    def format_for_prompt(self) -> str:
        """
        Render the locate result as a human-readable string for inclusion in
        the Mutation module's prompt.

        Nodes on the F2P execution trace are marked [*]; others are marked [ ].
        """
        lines: list[str] = []
        lines.append("=== Locate Module Output ===")
        lines.append("")
        lines.append("Allowed files for mutation:")
        for f in self.allowed_files:
            lines.append(f"  {f}")
        lines.append("")
        lines.append("Fail-to-Pass (F2P) tests:")
        for t in self.f2p_tests:
            lines.append(f"  {t}")
        lines.append("")
        lines.append("Structural graph  ( [*] = on F2P execution trace,  [ ] = not traced )")

        for struct in self.structures:
            lines.append(f"\n  [{struct.language}]  {struct.path}")
            for cls in struct.classes:
                marker = "[*]" if cls.on_f2p_trace else "[ ]"
                lines.append(f"    {marker} class {cls.name}  (L{cls.start_line}-{cls.end_line})")
                for m in cls.methods:
                    m_marker = "[*]" if m.on_f2p_trace else "[ ]"
                    params = ", ".join(m.params)
                    lines.append(
                        f"          {m_marker} def {m.name}({params})"
                        f"  (L{m.start_line}-{m.end_line})"
                    )
            for fn in struct.functions:
                marker = "[*]" if fn.on_f2p_trace else "[ ]"
                params = ", ".join(fn.params)
                lines.append(
                    f"    {marker} def {fn.name}({params})"
                    f"  (L{fn.start_line}-{fn.end_line})"
                )

        lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        def fn_to_dict(fn: FunctionNode) -> dict:
            return {
                "name": fn.name,
                "start_line": fn.start_line,
                "end_line": fn.end_line,
                "params": fn.params,
                "on_f2p_trace": fn.on_f2p_trace,
            }

        def cls_to_dict(cls: ClassNode) -> dict:
            return {
                "name": cls.name,
                "start_line": cls.start_line,
                "end_line": cls.end_line,
                "on_f2p_trace": cls.on_f2p_trace,
                "methods": [fn_to_dict(m) for m in cls.methods],
            }

        def struct_to_dict(s: FileStructure) -> dict:
            return {
                "path": s.path,
                "language": s.language,
                "classes": [cls_to_dict(c) for c in s.classes],
                "functions": [fn_to_dict(f) for f in s.functions],
            }

        return {
            "allowed_files": self.allowed_files,
            "f2p_tests": self.f2p_tests,
            "structures": [struct_to_dict(s) for s in self.structures],
        }


# ---------------------------------------------------------------------------
# Locate module
# ---------------------------------------------------------------------------

class LocateModule:
    """
    Implements the Locate step of the Agentic Mutation Framework.

    Usage::

        module = LocateModule(repo_path="/testbed")
        result = module.run(
            golden_patch=patch_text,
            f2p_tests=["tests/test_widgets.py::TestSplitArray::test_charfield"],
            test_cmd="python -m pytest",
        )
        prompt_section = result.format_for_prompt()
    """

    def __init__(self, repo_path: str | Path):
        self.repo_path = Path(repo_path)

    def run(
        self,
        golden_patch: str,
        f2p_tests: list[str],
        test_cmd: Optional[str] = None,
    ) -> LocateResult:
        allowed_files = self._extract_allowed_files(golden_patch)

        structures: list[FileStructure] = []
        for rel_path in allowed_files:
            abs_path = self.repo_path / rel_path
            if not abs_path.is_file():
                continue
            lang = LANGUAGE_BY_EXTENSION.get(abs_path.suffix, "unknown")
            struct = self._parse_file(rel_path, abs_path, lang)
            structures.append(struct)

        if structures:
            traced_lines = self._get_f2p_trace(f2p_tests, test_cmd)
            for struct in structures:
                self._annotate(struct, traced_lines.get(struct.path, set()))

        return LocateResult(
            allowed_files=allowed_files,
            structures=structures,
            f2p_tests=f2p_tests,
        )

    # ------------------------------------------------------------------
    # Step 1: extract allowed files from the golden patch
    # ------------------------------------------------------------------

    def _extract_allowed_files(self, golden_patch: str) -> list[str]:
        files: list[str] = []
        for line in golden_patch.splitlines():
            if line.startswith("+++ b/"):
                path = line[6:].strip()
                if path and path not in files:
                    files.append(path)
        return files

    # ------------------------------------------------------------------
    # Step 2: parse AST structure with Tree-sitter
    # ------------------------------------------------------------------

    def _parse_file(self, rel_path: str, abs_path: Path, lang: str) -> FileStructure:
        struct = FileStructure(path=rel_path, language=lang)
        try:
            source = abs_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return struct

        if lang == "python":
            self._parse_python(struct, source)
        elif lang != "unknown":
            self._parse_with_treesitter(struct, source, lang)

        return struct

    def _parse_python(self, struct: FileStructure, source: str) -> None:
        try:
            from tree_sitter_languages import get_parser  # type: ignore
            parser = get_parser("python")
            tree = parser.parse(source.encode())
            self._walk_python(struct, tree.root_node)
        except Exception:
            self._parse_python_regex(struct, source)

    def _walk_python(self, struct: FileStructure, node) -> None:
        for child in node.children:
            if child.type == "class_definition":
                cls = ClassNode(
                    name=self._node_name(child),
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                )
                body = child.child_by_field_name("body")
                if body:
                    for item in body.children:
                        fn_node = (
                            item if item.type == "function_definition"
                            else self._unwrap_decorated(item)
                        )
                        if fn_node:
                            m = self._python_fn_node(fn_node)
                            if m:
                                cls.methods.append(m)
                struct.classes.append(cls)

            elif child.type == "function_definition":
                fn = self._python_fn_node(child)
                if fn:
                    struct.functions.append(fn)

    def _unwrap_decorated(self, node) -> Optional[object]:
        for child in node.children:
            if child.type == "function_definition":
                return child
        return None

    def _python_fn_node(self, node) -> Optional[FunctionNode]:
        name_node = node.child_by_field_name("name")
        if not name_node:
            return None
        params_node = node.child_by_field_name("parameters")
        params: list[str] = []
        if params_node:
            for p in params_node.children:
                if p.type in (
                    "identifier", "typed_parameter", "default_parameter",
                    "typed_default_parameter", "list_splat_pattern",
                    "dictionary_splat_pattern",
                ):
                    raw = p.text.decode().split(":")[0].split("=")[0].strip().lstrip("*")
                    if raw:
                        params.append(raw)
        return FunctionNode(
            name=name_node.text.decode(),
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            params=params,
        )

    def _parse_python_regex(self, struct: FileStructure, source: str) -> None:
        """Regex fallback for Python when Tree-sitter is unavailable."""
        class_re = re.compile(r"^class\s+(\w+)", re.MULTILINE)
        func_re = re.compile(r"^(?P<indent>\s*)def\s+(\w+)\s*\(([^)]*)\)", re.MULTILINE)
        lines = source.splitlines()

        current_class: Optional[ClassNode] = None
        current_class_indent = -1

        for i, line in enumerate(lines, start=1):
            cm = class_re.match(line)
            if cm:
                if current_class:
                    struct.classes.append(current_class)
                current_class = ClassNode(name=cm.group(1), start_line=i, end_line=i)
                current_class_indent = 0

            fm = func_re.match(line)
            if fm:
                indent_len = len(fm.group("indent"))
                name = fm.group(2)
                raw_params = fm.group(3)
                params = [
                    p.split("=")[0].split(":")[0].strip().lstrip("*")
                    for p in raw_params.split(",")
                    if p.strip()
                ]
                fn = FunctionNode(name=name, start_line=i, end_line=i, params=params)
                if indent_len > 0 and current_class is not None:
                    current_class.methods.append(fn)
                else:
                    if current_class:
                        struct.classes.append(current_class)
                        current_class = None
                    struct.functions.append(fn)

        if current_class:
            struct.classes.append(current_class)

    def _parse_with_treesitter(self, struct: FileStructure, source: str, lang: str) -> None:
        try:
            from tree_sitter_languages import get_parser  # type: ignore
            parser = get_parser(lang)
            tree = parser.parse(source.encode())
            class_type, fn_type, method_type = _NODE_TYPES.get(lang, ("", "function_definition", "function_definition"))
            self._walk_generic(struct, tree.root_node, class_type, fn_type, method_type)
        except Exception:
            pass

    def _walk_generic(
        self,
        struct: FileStructure,
        node,
        class_type: str,
        fn_type: str,
        method_type: str,
        depth: int = 0,
    ) -> None:
        for child in node.children:
            if class_type and child.type == class_type:
                cls = ClassNode(
                    name=self._node_name(child),
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                )
                for item in child.children:
                    if item.type == method_type:
                        m = FunctionNode(
                            name=self._node_name(item),
                            start_line=item.start_point[0] + 1,
                            end_line=item.end_point[0] + 1,
                        )
                        cls.methods.append(m)
                struct.classes.append(cls)

            elif fn_type and child.type == fn_type and depth == 0:
                fn = FunctionNode(
                    name=self._node_name(child),
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                )
                struct.functions.append(fn)

            else:
                self._walk_generic(struct, child, class_type, fn_type, method_type, depth + 1)

    def _node_name(self, node) -> str:
        name_node = node.child_by_field_name("name")
        if name_node:
            return name_node.text.decode(errors="replace")
        return "?"

    # ------------------------------------------------------------------
    # Step 3: extract F2P execution trace via coverage instrumentation
    # ------------------------------------------------------------------

    def _get_f2p_trace(
        self,
        f2p_tests: list[str],
        test_cmd: Optional[str],
    ) -> dict[str, set[int]]:
        if not f2p_tests:
            return {}
        try:
            import coverage as _  # noqa: F401
        except ImportError:
            return {}

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            cov_json = tmp.name

        test_ids = " ".join(f2p_tests)
        cmd = (
            f"cd {self.repo_path} && "
            f"coverage run --branch -m pytest {test_ids} -q --no-header 2>/dev/null; "
            f"coverage json -o {cov_json} 2>/dev/null"
        )
        try:
            subprocess.run(cmd, shell=True, capture_output=True, timeout=180)
        except subprocess.TimeoutExpired:
            return {}

        traced: dict[str, set[int]] = {}
        try:
            with open(cov_json) as f:
                data = json.load(f)
            for abs_path_str, file_data in data.get("files", {}).items():
                try:
                    rel = str(Path(abs_path_str).relative_to(self.repo_path))
                except ValueError:
                    rel = abs_path_str
                executed = set(file_data.get("executed_lines", []))
                if executed:
                    traced[rel] = executed
        except Exception:
            pass

        return traced

    # ------------------------------------------------------------------
    # Step 4: annotate AST nodes with F2P trace
    # ------------------------------------------------------------------

    def _annotate(self, struct: FileStructure, traced_lines: set[int]) -> None:
        if not traced_lines:
            return

        for cls in struct.classes:
            cls_range = set(range(cls.start_line, cls.end_line + 1))
            if cls_range & traced_lines:
                cls.on_f2p_trace = True
            for method in cls.methods:
                m_range = set(range(method.start_line, method.end_line + 1))
                if m_range & traced_lines:
                    method.on_f2p_trace = True

        for fn in struct.functions:
            fn_range = set(range(fn.start_line, fn.end_line + 1))
            if fn_range & traced_lines:
                fn.on_f2p_trace = True
