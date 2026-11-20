"""Inventory declared library families and benchmark rows without treating missing references as coverage."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


def inventory(repository: Path) -> dict:
    generator = repository / "src/AiDotNet.Generators/TestScaffoldGenerator.cs"
    source = generator.read_text(encoding="utf-8-sig")
    match = re.search(r"private enum TestFamily\s*\{([^}]+)\}", source)
    if not match:
        raise ValueError("Authoritative TestFamily declaration not found; do not emit an empty passing inventory.")
    families = re.findall(r"^\s*([A-Za-z][A-Za-z0-9]*)\s*,?\s*$", match[1], re.MULTILINE)
    program = repository / "benchmarks/AiDotNet.PyTorchParity/Program.cs"
    rows = re.findall(r'"([a-z0-9-]+)"\s*=>\s*new\s+([A-Za-z0-9_]+)', program.read_text(encoding="utf-8-sig"))
    categories: dict[str, list[str]] = {}
    domains: dict[str, list[str]] = {}
    declaration_files = []
    for path in sorted((repository / "src").rglob("*.cs")):
        if {"bin", "obj", "Generated"} & set(path.relative_to(repository).parts):
            continue
        text = path.read_text(encoding="utf-8-sig")
        found_categories = sorted(set(re.findall(r"\[ModelCategory\(ModelCategory\.([A-Za-z0-9_]+)\)", text)))
        if not found_categories:
            continue
        relative = path.relative_to(repository).as_posix()
        declaration_files.append({"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        for category in found_categories:
            categories.setdefault(category, []).append(relative)
        for domain in sorted(set(re.findall(r"\[ModelDomain\(ModelDomain\.([A-Za-z0-9_]+)\)", text))):
            domains.setdefault(domain, []).append(relative)
    if not families or not rows or not declaration_files:
        raise ValueError("Source inventory is unexpectedly empty.")
    return {
        "schema": 1,
        "family_source": generator.relative_to(repository).as_posix() + "#TestFamily",
        "family_source_sha256": hashlib.sha256(generator.read_bytes()).hexdigest(),
        "declared_test_families": [{"name": family, "reference_status": "unverified", "completed_gate_matrix": False}
                                   for family in families],
        "existing_core_benchmark_rows": [{"name": name, "wrapper": wrapper, "completed_gate_matrix": False}
                                         for name, wrapper in rows],
        "model_category_declaration_files": {key: len(value) for key, value in sorted(categories.items())},
        "model_domain_declaration_files": {key: len(value) for key, value in sorted(domains.items())},
        "model_declaration_files": declaration_files,
        "additional_harnesses_to_map": ["ClassicalBench.cs", "TimeSeriesBench.cs", "reinforcement-learning parity harness"],
        "limitations": [
            "Test families and ModelCategory attributes describe different taxonomies; neither is silently equated to the 28 core rows.",
            "File-level attribute inventory is not a semantic claim that every class in a file is a model.",
            "Reference implementations, representative configurations and CPU/six-backend matrices must be mapped before any family can pass.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = inventory(args.repository.resolve())
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{len(result['declared_test_families'])} declared families, "
          f"{len(result['existing_core_benchmark_rows'])} core rows; all completion gates unverified.")


if __name__ == "__main__":
    main()
