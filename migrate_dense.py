#!/usr/bin/env python3
"""
Migrate DenseLayer<T>(inputSize, outputSize, ...) to DenseLayer<T>(outputSize, ...).

Handles:
- Positional: drops first int arg
- Named: drops "inputSize: X, " or "inputSize:X, "
- Multi-line calls
- Nested parens/brackets in arguments

Skips calls with only 1 arg.
"""
import os
import re
import sys


CALL_RE = re.compile(r"new\s+(?:[A-Za-z_][A-Za-z_0-9]*\.)*DenseLayer\s*<")


def find_matching(content: str, start: int, open_ch: str, close_ch: str) -> int:
    """Return index of matching close_ch, scanning from start (which is open_ch)."""
    assert content[start] == open_ch
    depth = 1
    i = start + 1
    n = len(content)
    while i < n and depth > 0:
        c = content[i]
        if c == '"':
            # Skip string literal
            i += 1
            while i < n and content[i] != '"':
                if content[i] == '\\':
                    i += 2
                    continue
                i += 1
            i += 1
            continue
        if c == "'":
            # Skip char literal
            i += 1
            while i < n and content[i] != "'":
                if content[i] == '\\':
                    i += 2
                    continue
                i += 1
            i += 1
            continue
        if c == '/' and i + 1 < n and content[i + 1] == '/':
            # line comment
            while i < n and content[i] != '\n':
                i += 1
            continue
        if c == '/' and i + 1 < n and content[i + 1] == '*':
            # block comment
            i += 2
            while i + 1 < n and not (content[i] == '*' and content[i + 1] == '/'):
                i += 1
            i += 2
            continue
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def split_top_level_args(content: str, start: int, end: int) -> list[tuple[int, int]]:
    """Return [(arg_start, arg_end_exclusive), ...] split on top-level commas in [start, end)."""
    args = []
    depth_paren = 0
    depth_bracket = 0
    depth_brace = 0
    depth_angle = 0
    arg_start = start
    i = start
    while i < end:
        c = content[i]
        if c == '"':
            i += 1
            while i < end and content[i] != '"':
                if content[i] == '\\':
                    i += 2
                    continue
                i += 1
            i += 1
            continue
        if c == "'":
            i += 1
            while i < end and content[i] != "'":
                if content[i] == '\\':
                    i += 2
                    continue
                i += 1
            i += 1
            continue
        if c == '(':
            depth_paren += 1
        elif c == ')':
            depth_paren -= 1
        elif c == '[':
            depth_bracket += 1
        elif c == ']':
            depth_bracket -= 1
        elif c == '{':
            depth_brace += 1
        elif c == '}':
            depth_brace -= 1
        elif c == ',' and depth_paren == 0 and depth_bracket == 0 and depth_brace == 0:
            args.append((arg_start, i))
            arg_start = i + 1
        i += 1
    if arg_start < end:
        args.append((arg_start, end))
    return args


def migrate(content: str) -> tuple[str, int]:
    out = []
    i = 0
    n = len(content)
    count = 0
    while i < n:
        m = CALL_RE.search(content, i)
        if not m:
            out.append(content[i:])
            break
        out.append(content[i:m.start()])
        # Found "new DenseLayer<"
        ang_start = m.end() - 1  # position of '<'
        ang_end = find_matching(content, ang_start, '<', '>')
        if ang_end < 0:
            # Malformed — emit as-is
            out.append(content[m.start():])
            break
        # Skip whitespace after >
        j = ang_end + 1
        while j < n and content[j] in ' \t\r\n':
            j += 1
        if j >= n or content[j] != '(':
            # Not a constructor call (maybe DenseLayer<T> as a type ref) — emit as-is
            out.append(content[m.start():j])
            i = j
            continue
        paren_start = j
        paren_end = find_matching(content, paren_start, '(', ')')
        if paren_end < 0:
            out.append(content[m.start():])
            break
        args = split_top_level_args(content, paren_start + 1, paren_end)
        if len(args) < 2:
            # Only one arg; nothing to drop
            out.append(content[m.start():paren_end + 1])
            i = paren_end + 1
            continue
        # Check first arg for "inputSize:"
        first_text = content[args[0][0]:args[0][1]]
        # Heuristic: if first arg uses named arg "outputSize:" (already migrated) or
        # the second arg looks like an activation/non-int, skip to avoid double-migration.
        if re.match(r'^\s*outputSize\s*:', first_text):
            out.append(content[m.start():paren_end + 1])
            i = paren_end + 1
            continue
        second_text = content[args[1][0]:args[1][1]].strip()
        # Activation-like second arg: starts with "new " for an Activation, or is null,
        # or is a cast to IActivationFunction/IVectorActivationFunction
        is_activation_like = (
            re.search(r'\bActivation\b', second_text) is not None
            or re.match(r'^\s*null\s*$', second_text)
            or re.match(r'^\s*\(\s*IActivationFunction', second_text)
            or re.match(r'^\s*\(\s*IVectorActivationFunction', second_text)
            or re.match(r'^\s*activationFunction\s*:', second_text)
            or re.match(r'^\s*vectorActivation\s*:', second_text)
            or re.match(r'^\s*initializationStrategy\s*:', second_text)
        )
        if is_activation_like:
            out.append(content[m.start():paren_end + 1])
            i = paren_end + 1
            continue
        # Header up to and including '('
        out.append(content[m.start():paren_start + 1])
        if re.match(r'^\s*inputSize\s*:', first_text):
            # Drop first arg entirely; emit rest joined by commas verbatim
            kept = [content[a[0]:a[1]] for a in args[1:]]
            out.append(','.join(kept))
        else:
            # Positional: drop first arg, emit rest verbatim (preserves whitespace/newlines)
            # Find the comma after args[0]
            after_first_comma = args[0][1] + 1  # position after the comma at args[0][1]
            # The remaining region is content[after_first_comma:paren_end] but we want to
            # preserve the leading whitespace pattern for multi-line calls; trim only one space
            tail = content[after_first_comma:paren_end]
            # Strip a single leading space if it exists (often after ", ")
            if tail.startswith(' '):
                tail = tail[1:]
            out.append(tail)
        out.append(')')
        i = paren_end + 1
        count += 1
    return ''.join(out), count


def main():
    roots = sys.argv[1:] if len(sys.argv) > 1 else ['src', 'tests']
    total_files = 0
    total_calls = 0
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            # Skip generated/binary dirs
            parts = dirpath.replace('\\', '/').split('/')
            if any(p in ('bin', 'obj', 'node_modules', '.git') for p in parts):
                continue
            for fname in filenames:
                if not fname.endswith('.cs'):
                    continue
                path = os.path.join(dirpath, fname)
                try:
                    with open(path, 'r', encoding='utf-8-sig') as f:
                        original = f.read()
                except Exception as e:
                    continue
                if 'DenseLayer<' not in original:
                    continue
                migrated, n = migrate(original)
                if n > 0 and migrated != original:
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(migrated)
                    total_files += 1
                    total_calls += n
                    print(f"{path}: {n} migrated")
    print(f"\nTotal: {total_calls} calls migrated across {total_files} files")


if __name__ == '__main__':
    main()
