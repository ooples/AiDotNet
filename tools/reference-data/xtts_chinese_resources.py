"""Export original pypinyin full TONE3 readings and execute Coqui's Chinese TextNorm.

Argument: coqui-ai/TTS checkout. pypinyin 0.50.0, neutral_tone_with_five=True.
Initial/final styles cannot reconstruct the full TONE3 style (notably y/w syllables).
"""
import ast
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import pypinyin
from pypinyin.constants import PINYIN_DICT
from pypinyin.phrases_dict import phrases_dict
repo=Path(__file__).resolve().parents[2]
resource=repo/'src/TextToSpeech/FrontEnd/Resources'
def read(text):
    return [x[0] for x in pypinyin.pinyin(text,style=pypinyin.Style.TONE3,heteronym=False,neutral_tone_with_five=True)]
with gzip.open(resource/'xtts_pypinyin_chars.tsv.gz','wt',encoding='utf-8',newline='\n') as stream:
    for code in sorted(PINYIN_DICT):stream.write(chr(code)+'\t'+read(chr(code))[0]+'\n')
with gzip.open(resource/'xtts_pypinyin_phrases.tsv.gz','wt',encoding='utf-8',newline='\n') as stream:
    for phrase in sorted(phrases_dict):stream.write(phrase+'\t'+' '.join(read(phrase))+'\n')
upstream=Path(sys.argv[1])/'TTS/tts/layers/xtts'
spec=importlib.util.spec_from_file_location('xtts_zh_num2words',upstream/'zh_num2words.py')
normalizer=importlib.util.module_from_spec(spec);spec.loader.exec_module(normalizer)
tree=ast.parse((upstream/'tokenizer.py').read_text(encoding='utf-8'))
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='chinese_transliterate')
namespace={'pypinyin':pypinyin};exec(compile(ast.Module(body=[function],type_ignores=[]),'tokenizer.py','exec'),namespace)
texts=['你好，世界！','语音与音乐。','月亮和阳光。','女儿去银行。','重庆的长大了。','English 和中文 123。',
    '在12.5秒内','有50名士兵','我的电量为 14%','今天是2024年3月5日。',
    '电话13812345678','比例1/3，价格12.50元','一二三四五六七八九十','妈妈呢？你好！']
cases=[]
for text in texts:
    normalized=normalizer.TextNorm()(text.lower().replace('"',''))
    cases.append(dict(text=text,normalized=normalized,expected=namespace['chinese_transliterate'](normalized)))
(repo/'tests/AiDotNet.Tests/TextToSpeech/ReferenceData/xtts_chinese_reference.json').write_text(json.dumps(dict(
    source='coqui-ai/TTS xtts.zh_num2words.TextNorm, xtts.tokenizer.chinese_transliterate, pypinyin 0.50.0',
    punctuation=normalizer.PUNCS,cases=cases),ensure_ascii=True,separators=(',',':')),encoding='utf-8')
