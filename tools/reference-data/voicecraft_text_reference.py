"""Execute VoiceCraft's original TextTokenizer with phonemizer 3.3 and eSpeak NG.

Arguments: VoiceCraft checkout, eSpeak NG shared library. Loading the unmodified
class AST avoids importing unrelated audio codecs from the same tokenizer file.
Raw eSpeak calls are recorded so tests replay the exact front-end transformation
without requiring a system native library on CI.
"""
import ast
import json
from pathlib import Path
import re
import sys
from typing import List, Pattern, Union
from phonemizer.backend import EspeakBackend
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from phonemizer.backend.espeak.language_switch import LanguageSwitch
from phonemizer.backend.espeak.words_mismatch import WordMismatch
from phonemizer.punctuation import Punctuation
from phonemizer.separator import Separator
EspeakWrapper.set_library(sys.argv[2])
source=Path(sys.argv[1])/'data/tokenizer.py'
tree=ast.parse(source.read_text(encoding='utf-8'))
definition=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TextTokenizer')
exec(compile(ast.Module(body=[definition],type_ignores=[]),str(source),'exec'),globals())
tokenizer=TextTokenizer()
calls=[]
original=tokenizer.backend._espeak.text_to_phonemes
def record(text,*args,**kwargs):
    output=original(text,*args,**kwargs)
    calls.append(dict(text=text,output=output))
    return output
tokenizer.backend._espeak.text_to_phonemes=record
cases=[]
for text in ['Hello, world!', 'The quick brown fox.', 'Dr. Smith paid $3.50.', "Don't stop...", '  spaced   words  ',
    'A; B: C?', 'Café déjà vu.', '(Hello) [world]!', '!!!', 'a test of zero-shot voice cloning', 'hello', 'hello world', 'hi',
    'Hello world.', 'a completely different sentence', 'the quick brown fox']:
    calls=[]
    phones=tokenizer(text)[0]
    cases.append(dict(text=text,phones=phones,calls=calls))
out=Path(__file__).resolve().parents[2]/'tests/AiDotNet.Tests/TextToSpeech/ReferenceData/voicecraft_text_reference.json'
out.write_text(json.dumps(dict(source='jasonppy/VoiceCraft data/tokenizer.py TextTokenizer, phonemizer 3.3, eSpeak NG 1.52',cases=cases),ensure_ascii=True,separators=(',',':')),encoding='utf-8')
