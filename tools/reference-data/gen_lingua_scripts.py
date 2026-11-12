"""Appends lingua-rs v1.6.2's script tables (src/script.rs) for the alphabets lingua knows, as code-point ranges, to
FireRedTtsTextData.cs."""
import os
import re
import sys

# Usage: python gen_lingua_scripts.py <lingua-rs checkout at v1.6.2>
SCRIPT = os.path.join(sys.argv[1], 'src', 'script.rs')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src', 'TextToSpeech', 'FrontEnd', 'FireRedTtsTextData.cs')
ALPHABETS = ['Arabic', 'Armenian', 'Bengali', 'Cyrillic', 'Devanagari', 'Georgian', 'Greek', 'Gujarati', 'Gurmukhi', 'Han',
             'Hangul', 'Hebrew', 'Hiragana', 'Katakana', 'Latin', 'Tamil', 'Telugu', 'Thai']
src = open(SCRIPT, encoding='utf-8').read()


def char(lit):
    body = lit[1:-1]
    if body.startswith('\\u{'):
        return int(body[3:-1], 16)
    if body.startswith('\\'):
        return {'\\\\': 0x5C, "\\'": 0x27, '\\n': 10, '\\t': 9, '\\r': 13, '\\0': 0}[body]
    assert len(body) == 1, body
    return ord(body)


tables = {}
for name in ALPHABETS:
    m = re.search(r'pub const %s: &\[\(char, char\)\] = &\[(.*?)\];' % name.upper(), src, re.S)
    pairs = re.findall(r"\(('(?:\\u\{[0-9a-fA-F]+\}|\\.|[^'])'), ('(?:\\u\{[0-9a-fA-F]+\}|\\.|[^'])')\)", m.group(1))
    tables[name] = [(char(a), char(b)) for a, b in pairs]

text = open(OUT, encoding='utf-8').read().rstrip()
assert text.endswith('}')
lines = ['', '    /// <summary>lingua-rs 1.6.2\'s script tables (<c>src/script.rs</c>) of the alphabets its rules know, in',
         '    /// <c>Alphabet::iter()</c> order: each a flat list of inclusive (first, last) code points.</summary>',
         '    public static readonly (string Name, int[] Ranges)[] LinguaAlphabets =', '    {']
for name in ALPHABETS:
    flat = [v for pair in tables[name] for v in pair]
    lines.append('        ("%s", new[]' % name)
    lines.append('        {')
    for i in range(0, len(flat), 12):
        lines.append('            ' + ', '.join('0x%X' % v for v in flat[i:i + 12]) + ',')
    lines.append('        }),')
lines.append('    };')
text = text[:-1].rstrip() + '\n' + '\n'.join(lines) + '\n}\n'
open(OUT, 'w', encoding='utf-8', newline='\n').write(text)
print({k: len(v) for k, v in tables.items()})
