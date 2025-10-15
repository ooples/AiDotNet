#!/usr/bin/env python3
"""Remove duplicate interface definitions from compressor files."""

import re

def remove_duplicate_interface(file_path, interface_name):
    """Remove duplicate interface definition from a file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the duplicate interface definition pattern
    # Look for the interface definition that's followed by other content (not at end of file)
    pattern = rf'(/// <summary>\s*\n/// Interface for .*{interface_name}.*?</remarks>\s*\n)public interface {interface_name}.*?\n}}\s*\n'
    
    # Check if pattern exists
    matches = list(re.finditer(pattern, content, re.DOTALL))
    
    if matches:
        print(f"Found {len(matches)} duplicate interface definitions in {file_path}")
        # Remove the first occurrence (the duplicate)
        content = content[:matches[0].start()] + content[matches[0].end():]
        
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Removed duplicate {interface_name} interface from {file_path}")
    else:
        print(f"No duplicate {interface_name} interface found in {file_path}")

# Fix QuantizationCompressor.cs
remove_duplicate_interface(
    '/home/ooples/AiDotNet/src/Compression/Quantization/QuantizationCompressor.cs',
    'IQuantizedModel<T, TInput, TOutput>'
)

# Fix PruningCompressor.cs
remove_duplicate_interface(
    '/home/ooples/AiDotNet/src/Compression/Pruning/PruningCompressor.cs',
    'IPrunedModel<T, TInput, TOutput>'
)