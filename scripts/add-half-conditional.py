#!/usr/bin/env python3
"""Add conditional compilation around Half methods in NumericOperations files."""

import sys
import re

def add_conditional(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern to match the ToHalf and FromHalf methods together
    pattern = r'(    /// <summary>\s*\n    /// Converts.*?Half.*?\n    /// </summary>.*?public (?:Half|[A-Za-z<>]+) ToHalf\([^)]+\).*?}\s*\n\s*/// <summary>\s*\n    /// Converts.*?Half.*?\n    /// </summary>.*?public [A-Za-z<>]+ FromHalf\([^)]+\).*?})'

    # Check if already has conditional
    if '#if NET5_0_OR_GREATER' in content:
        print(f"- Skipped {file_path} (already has conditionals)")
        return

    # Add conditional compilation
    new_content = re.sub(
        pattern,
        r'#if NET5_0_OR_GREATER\n\1\n#endif',
        content,
        flags=re.DOTALL | re.MULTILINE
    )

    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"✓ Updated {file_path}")
    else:
        print(f"⚠ No changes made to {file_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 add-half-conditional.py <file_path>")
        sys.exit(1)

    add_conditional(sys.argv[1])
