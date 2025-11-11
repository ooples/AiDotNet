#!/usr/bin/env python3
"""
Fixes conditional compilation directives around ToHalf() and FromHalf() methods.
Wraps both methods together in a single #if NET5_0_OR_GREATER block.
"""

import re
from pathlib import Path

def fix_conditionals_in_file(file_path):
    """Fix conditional compilation directives around Half methods."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Remove any existing #if NET5_0_OR_GREATER and #endif around Half methods
    content = re.sub(r'#if NET5_0_OR_GREATER\n', '', content)
    content = re.sub(r'\n#endif(?=\n)', '', content)

    # Find ToHalf method and FromHalf method as a group
    # Pattern matches from the ToHalf summary comment to the end of FromHalf method
    pattern = r'(    /// <summary>\s*\n    /// Converts.*?to Half.*?\n    /// </summary>.*?public [^(]+ ToHalf\([^)]+\)[^}]+}\s*\n\s*/// <summary>\s*\n    /// Converts.*?Half.*?to.*?\n    /// </summary>.*?public [^(]+ FromHalf\([^)]+\)[^}]+})'

    # Wrap both methods together
    content = re.sub(
        pattern,
        r'#if NET5_0_OR_GREATER\n\1\n#endif',
        content,
        flags=re.DOTALL
    )

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✓ Fixed {file_path}")

def main():
    operations_dir = Path("src/NumericOperations")

    # Process all NumericOperations files except HalfOperations.cs
    for file_path in operations_dir.glob("*.cs"):
        if file_path.name != "HalfOperations.cs":
            fix_conditionals_in_file(file_path)

    print("\n✅ All files fixed successfully!")

if __name__ == "__main__":
    main()
