#!/usr/bin/env python3
"""
Automated UTF-8 encoding fixer for C# documentation.
Fixes common encoding corruption issues where Edit tools convert proper UTF-8 to replacement characters.
"""

import sys
import os
import re
from pathlib import Path

def fix_encoding(content):
    """Fix all common UTF-8 encoding corruptions."""
    original = content
    
    # Fix dimensions and multiplication: "2×3" format
    content = re.sub(r'(\d+)�(\d+)', r'\1×\2', content)
    
    # Fix multiplication in expressions: "5 × 4"
    content = re.sub(r'(\d+(?:\.\d+)?)\s*�\s*(\d+(?:\.\d+)?)', r'\1 × \2', content)
    
    # Fix variable multiplication: "coefficient1 × mean"
    content = re.sub(r'([a-zA-Z]+\d?)\s*�\s*([a-zA-Z]+)', r'\1 × \2', content)
    
    # Fix superscript 2: "x²"
    content = re.sub(r'([a-z0-9\)])�', r'\1²', content)
    
    # Fix superscript 3: "2³"
    content = re.sub(r'2�(?=\s*=\s*8)', '2³', content)
    content = re.sub(r'3�(?=\s*=\s*27)', '3³', content)
    
    # Fix approximately equal: "≈"
    content = re.sub(r'\(�', '(≈', content)
    content = re.sub(r'\s�\s', ' ≈ ', content)
    
    # Fix square root: "√("
    content = re.sub(r'v\(', '√(', content)
    
    # Fix plus-minus: "±"
    content = re.sub(r'�(\d)', r'±\1', content)
    
    # Fix exponential notation
    content = content.replace('e�', 'e^')
    
    # Fix negative exponents
    content = content.replace('e?�', 'e^-1')
    content = content.replace('2?�', '2^-1')
    
    # Fix specific patterns like "2×2×2"
    content = content.replace('2×2�2', '2×2×2')
    content = content.replace('3×3�3', '3×3×3')
    
    # Fix Matrix dimensions in error messages
    content = content.replace('Rows}�{', 'Rows}×{')
    
    return content

def process_file(filepath):
    """Process a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Check if file has corruption
        if '�' not in content:
            return 0
        
        # Fix encoding
        fixed = fix_encoding(content)
        
        # Count remaining issues
        remaining = fixed.count('�')
        original_count = content.count('�')
        fixed_count = original_count - remaining
        
        if fixed_count > 0:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(fixed)
            print(f"✓ {filepath}: fixed {fixed_count} issue(s)" + 
                  (f", {remaining} remaining" if remaining > 0 else ""))
            return fixed_count
        
        return 0
        
    except Exception as e:
        print(f"❌ Error processing {filepath}: {e}", file=sys.stderr)
        return 0

def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        # Process specific files
        files = sys.argv[1:]
    else:
        # Process all .cs files
        files = []
        for root, dirs, filenames in os.walk('.'):
            if '.git' in root or 'bin' in root or 'obj' in root:
                continue
            for filename in filenames:
                if filename.endswith('.cs'):
                    files.append(os.path.join(root, filename))
    
    total_fixed = 0
    for filepath in files:
        if os.path.exists(filepath):
            total_fixed += process_file(filepath)
    
    print(f"\n{'='*60}")
    print(f"Total: Fixed {total_fixed} encoding issue(s)")
    print(f"{'='*60}")
    
    return 0 if total_fixed >= 0 else 1

if __name__ == '__main__':
    sys.exit(main())
