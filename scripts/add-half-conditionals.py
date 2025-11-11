#!/usr/bin/env python3
"""
Adds #if NET5_0_OR_GREATER conditional compilation directives around
ToHalf() and FromHalf() methods in NumericOperations files.
"""

import re
import sys
from pathlib import Path

def add_conditionals_to_file(file_path):
    """Add conditional compilation directives around Half methods."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern to match ToHalf method (with optional XML comments)
    tohalf_pattern = r'(    /// <summary>.*?Converts.*?to Half.*?</summary>.*?public (?:Half|float|double|int|long|byte|sbyte|short|ushort|uint|ulong|decimal|Complex<T>) ToHalf\([^)]+\)[^}]+})'

    # Pattern to match FromHalf method
    fromhalf_pattern = r'(    /// <summary>.*?Converts.*?Half.*?to.*?</summary>.*?public [^(]+ FromHalf\([^)]+\)[^}]+})'

    # Add conditionals around ToHalf
    content = re.sub(
        tohalf_pattern,
        r'#if NET5_0_OR_GREATER\n\1\n#endif',
        content,
        flags=re.DOTALL
    )

    # Add conditionals around FromHalf
    content = re.sub(
        fromhalf_pattern,
        r'#if NET5_0_OR_GREATER\n\1\n#endif',
        content,
        flags=re.DOTALL
    )

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✓ Updated {file_path}")

def main():
    operations_dir = Path("src/NumericOperations")

    # Process all NumericOperations files except HalfOperations.cs
    for file_path in operations_dir.glob("*.cs"):
        if file_path.name != "HalfOperations.cs":
            add_conditionals_to_file(file_path)

    print("\n✅ All files updated successfully!")

if __name__ == "__main__":
    main()
