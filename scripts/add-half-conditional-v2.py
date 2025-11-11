#!/usr/bin/env python3
"""Add conditional compilation around Half methods in NumericOperations files."""

import sys
import re

def add_conditional(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if already has conditional
    if '#if NET5_0_OR_GREATER' in content and 'ToHalf' in content[content.find('#if NET5_0_OR_GREATER'):content.find('#if NET5_0_OR_GREATER')+500]:
        print(f"- Skipped {file_path} (already has conditionals)")
        return

    # Simple pattern for files with minimal documentation
    simple_pattern = r'(    /// <summary>\s*\n    /// Converts [^\n]+? (?:to|from) Half[^\n]*\n    /// </summary>\s*\n    public [^\n]+ToHalf[^\n]+\n\s*/// <summary>\s*\n    /// Converts [^\n]+? (?:to|from) Half[^\n]*\n    /// </summary>\s*\n    public [^\n]+FromHalf[^\n]+)'

    # Try simple pattern first
    new_content = re.sub(
        simple_pattern,
        r'#if NET5_0_OR_GREATER\n\1\n#endif',
        content,
        flags=re.DOTALL | re.MULTILINE
    )

    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"✓ Updated {file_path}")
    else:
        print(f"⚠ No pattern matched in {file_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 add-half-conditional-v2.py <file_path>")
        sys.exit(1)

    add_conditional(sys.argv[1])
