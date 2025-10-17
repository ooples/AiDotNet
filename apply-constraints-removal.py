#!/usr/bin/env python3
"""
Script to apply CI-001 constraint removal changes from proposal JSON
"""
import json
import sys
import re

def apply_edit(filepath, old_code, new_code, line_start, line_end):
    """Apply a single edit to a file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Convert line numbers (1-indexed) to array indices (0-indexed)
        start_idx = line_start - 1
        end_idx = line_end

        # Extract the lines to replace
        actual_lines = ''.join(lines[start_idx:end_idx])

        # Normalize whitespace for comparison (but keep original in replacement)
        old_normalized = re.sub(r'\s+', ' ', old_code.strip())
        actual_normalized = re.sub(r'\s+', ' ', actual_lines.strip())

        if old_normalized not in actual_normalized:
            print(f"WARNING: Could not find exact match in {filepath} lines {line_start}-{line_end}")
            print(f"Expected (normalized): {old_normalized[:100]}...")
            print(f"Actual (normalized): {actual_normalized[:100]}...")
            return False

        # Replace the lines
        lines[start_idx:end_idx] = [new_code + '\n']

        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        print(f"✓ Applied edit to {filepath}:{line_start}")
        return True

    except Exception as e:
        print(f"ERROR applying edit to {filepath}: {e}")
        return False

def main():
    # Load proposal
    with open('fix-proposals/CI-001-proposal.json', 'r', encoding='utf-8') as f:
        proposal = json.load(f)

    total = 0
    success = 0
    skipped = 0

    for file_change in proposal['solution']['files_to_modify']:
        filepath = file_change['path']

        for change in file_change['changes']:
            if change['type'] == 'note':
                print(f"SKIP: {filepath} - {change['reason']}")
                skipped += 1
                continue

            if change['type'] == 'edit':
                total += 1
                if apply_edit(filepath, change['old_code'], change['new_code'],
                             change['line_start'], change['line_end']):
                    success += 1

    print(f"\n{'='*60}")
    print(f"Results: {success}/{total} edits applied successfully")
    print(f"Skipped: {skipped} (notes/excluded files)")
    print(f"{'='*60}")

    return 0 if success == total else 1

if __name__ == '__main__':
    sys.exit(main())
