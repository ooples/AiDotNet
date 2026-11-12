"""Execute the original released Hindi basic_cleaners branch, preserving quotes and numbers."""
import ast
import json
import os
from pathlib import Path
import re
import sys
from num2words import num2words

source=Path(sys.argv[1])/'TTS/tts/layers/xtts/tokenizer.py'
tree=ast.parse(source.read_text(encoding='utf-8'))
namespace=dict(re=re,num2words=num2words,os=os,__file__=str(source))
exec(compile(ast.Module(body=[node for node in tree.body if isinstance(node,(ast.Assign,ast.FunctionDef))],type_ignores=[]),str(source),'exec'),namespace)
texts=['नमस्ते दुनिया', '  नमस्ते\tदुनिया\n', '"नमस्ते 123" HELLO\tWORLD',
    'मेरे पास 21 रुपये हैं।', 'डॉ. शर्मा 3.14 और $20', 'HELLO नमस्ते ABC',
    'संख्या १२३ और 123', 'हिन्दी\u00a0भाषा\u2003परीक्षण']
cases=[dict(text=text,expected=namespace['basic_cleaners'](text)) for text in texts]
destination=Path(__file__).resolve().parents[2]/'tests/AiDotNet.Tests/TextToSpeech/ReferenceData/xtts_hindi_reference.json'
destination.write_text(json.dumps(dict(source='Original Coqui XTTS Hindi basic_cleaners branch',cases=cases),ensure_ascii=True,separators=(',',':')),encoding='utf-8')
print('Wrote',len(cases),'original Hindi cleaner cases')
