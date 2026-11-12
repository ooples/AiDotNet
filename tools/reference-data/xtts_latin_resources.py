"""Execute Coqui's original multilingual cleaner and export lexical replacement rules.

Argument: coqui-ai/TTS checkout. Uses num2words 0.5.12. No native code participates
in the expected outputs. Loading assignments/functions avoids unrelated spaCy models.
"""
import ast
import gzip
import json
import os
import importlib
from pathlib import Path
import re
import sys
from num2words import num2words, CONVERTER_CLASSES
from hangul_romanize import Transliter
from hangul_romanize.rule import academic
repo=Path(__file__).resolve().parents[2]
source=Path(sys.argv[1])/'TTS/tts/layers/xtts/tokenizer.py'
tree=ast.parse(source.read_text(encoding='utf-8'))
body=[n for n in tree.body if isinstance(n,(ast.Assign,ast.FunctionDef))]
namespace=dict(re=re,num2words=num2words,os=os,__file__=str(source),Transliter=Transliter,academic=academic)
exec(compile(ast.Module(body=body,type_ignores=[]),str(source),'exec'),namespace)
resources=repo/'src/TextToSpeech/FrontEnd/Resources'
english=CONVERTER_CLASSES['en']
with gzip.open(resources/'num2words_english.json.gz','wt',encoding='utf-8') as stream:
    json.dump(dict(cards=[(str(k),v) for k,v in english.cards.items()],ordinals=english.ords),stream,separators=(',',':'))
currency_path=resources/'num2words_currency.json.gz'
with gzip.open(currency_path,'rt',encoding='utf-8') as stream:currencies=json.load(stream)
currencies['en']=english.CURRENCY_FORMS
european={}
for language in ['de','nl','ko','pt','hu']:
    converter=CONVERTER_CLASSES[language]
    if language != 'ko': currencies[language]=converter.CURRENCY_FORMS
    european[language]=dict(cards=[(str(k),v) for k,v in converter.cards.items()],
        ordinals=list(getattr(converter,'ords',{}).items()) if isinstance(getattr(converter,'ords',{}),dict) else converter.ords,
        negative=converter.negword,point=converter.pointword)
    if language=='pt':european[language].update(hundreds=converter.hundreds,thousands=converter.thousand_separators)
    if language=='hu':european[language].update(partial_ordinals=converter.partial_ords)
currencies['it']=CONVERTER_CLASSES['it'].CURRENCY_FORMS
slavic={}
for language, module_name in [('ru','RU'),('pl','PL'),('cz','CZ')]:
    converter=CONVERTER_CLASSES[language]
    module=importlib.import_module('num2words.lang_'+module_name)
    slavic[language]={name:getattr(module,name) for name in ['ZERO','ONES','TENS','TWENTIES','HUNDREDS','THOUSANDS']}
    slavic[language].update(negative=converter.negword,point=converter.pointword)
    if language=='ru':slavic[language].update(ONES_FEMININE=module.ONES_FEMININE,ordinals=converter.ords,feminine_ordinals=converter.ords_feminine)
    if language=='pl':slavic[language].update(ONES_ORDINALS=module.ONES_ORDINALS,TWENTIES_ORDINALS=module.TWENTIES_ORDINALS,
        HUNDREDS_ORDINALS=module.HUNDREDS_ORDINALS,ordinal_prefixes=module.prefixes_ordinal)
    currencies[language]=converter.CURRENCY_FORMS
with gzip.open(resources/'num2words_slavic.json.gz','wt',encoding='utf-8') as stream:json.dump(slavic,stream,ensure_ascii=False,separators=(',',':'))
arabic_module=importlib.import_module('num2words.lang_AR')
arabic={key:value for key,value in vars(CONVERTER_CLASSES['ar']).items() if key.startswith('arabic') and isinstance(value,list)}
arabic['currencies']={key:getattr(arabic_module,'CURRENCY_'+key) for key in ['SR','EGP','KWD']}
with gzip.open(resources/'num2words_arabic.json.gz','wt',encoding='utf-8') as stream:json.dump(arabic,stream,ensure_ascii=False,separators=(',',':'))
with gzip.open(resources/'num2words_european.json.gz','wt',encoding='utf-8') as stream:json.dump(european,stream,ensure_ascii=False,separators=(',',':'))
with gzip.open(currency_path,'wt',encoding='utf-8') as stream:json.dump(currencies,stream,ensure_ascii=False,separators=(',',':'))
rules={}
for language in namespace['_abbreviations']:
    def export(values):return [dict(pattern=p.pattern,replacement=v,ignore_case=bool(p.flags & re.IGNORECASE)) for p,v in values]
    rules[language]=dict(abbreviations=export(namespace['_abbreviations'][language]),symbols=export(namespace['_symbols_multilingual'][language]))
with gzip.open(resources/'xtts_cleaner_rules.json.gz','wt',encoding='utf-8') as stream:json.dump(rules,stream,ensure_ascii=False,separators=(',',':'))
texts=['Dr. Smith has 12 apples.','Mr. Jones is 21st.','She paid $1.50 and £20.','It costs 2€ or $0.01.',
    '1,234 and 1001 and 1234567.','0.01 12.50 100.001','Mrs. A & Dr. B @ 50% #1',
    '   Extra   whitespace!  ','The 100th and 1000th place.','There are 1000000000 stars.']
cases=[dict(language='en',text=t,expected=namespace['multilingual_cleaners'](t,'en')) for t in texts]
for lang,examples in [('fr',['M. Dupont a 12 pommes.','Le 1er et le 21ème.','Cela coûte 2€ ou $1,50.','1.234 et 1001.','Dr. A & M. B @ 50% #1']),
    ('es',['El Sr. tiene 12 manzanas.','Es el 1º y el 21er.','Cuesta 2€ o $1,50.','1.234 y 1001.','Dr. A & Sr. B @ 50% #1']),
    ('de',['Dr. Müller hat 21 Äpfel.','Am 1. und 100. Tag.','Es kostet 2€ oder $1,50.','1.234 und 1001.','Eine Million: 1000000.']),
    ('nl',['Dr. Jan heeft 21 appels.','De 1ste en 21ste.','Het kost 2€ of $1,50.','1.234 en 1001.','1000000 mensen.']),
    ('pt',['O Sr. tem 21 maçãs.','O 1º e o 21º.','Custa 2€ ou $1,50.','1.234 e 1001.','Um milhão: 1000000.','21€ e 12,50€']),
    ('it',['Il sig. ha 21 mele.','Il 1º e il 23º.','Costa 2€ o $1,50.','1.234 e 1001.','1000000 persone.','100,0 e 100,08','1023 e 2000º']),
    ('ru',['У доктора 21 яблоко.','Это 1-й и 23-й.','Цена 2€ или $1,50.','1,234 и 1001.','1000000 человек.','100.0 и 100.08','2000-й и 300-й']),
    ('pl',['Dr ma 21 jabłek.','To 1st i 23rd.','Cena 2€ lub $1,50.','1.234 i 1001.','1000000 ludzi.','100,0 i 100,08','2000th i 300th']),
    ('cs',['Doktor má 21 jablek','Cena 2€ nebo $1,50','1.234 a 1001','1000000 lidí','100,0 a 100,08']),
    ('hu',['Dr. úrnak 21 almája van.','Az 1. és a 23. hely.','Az ár 2€ vagy $1,50.','1.234 és 1001.','1000000 ember.','100,0 és 100,08','2000. és 300.']),
    ('tr',['Dr. Ali 21 elma aldı.','1. ve 23. sıra.','Fiyat 2€ veya $1,50.','1.234 ve 1001.','1000000 kişi.','100,0 ve 100,08','İYİ ÖĞÜN ÜÇ 101001'])]:
    cases.extend(dict(language=lang,text=t,expected=namespace['multilingual_cleaners'](t,lang)) for t in examples)
for text in ['안녕하세요 한국어.', '한국어를 배워요.', '서울에 12명이 있어요.', '1번째와 21번째', '값은 $1.50', '가각아 가나다 라라', '읽고 값과 앉다']:
    cases.append(dict(language='ko',text=text,expected=namespace['korean_transliterate'](namespace['multilingual_cleaners'](text,'ko'))))
for text in ['لدي 21 تفاحة.', 'الثمن 2€ أو $1.50.', '1000 و 2000 و 1001', '1ر و 21ر و 3',
    '100.0 و 100.08', '1234567 و 2000000', '0.5 و 12.05', '101 و 120 و 161 و 200', '100000000000000000001']:
    # The upstream converter has mutable gender flags; each case represents a fresh process/utterance.
    CONVERTER_CLASSES['ar']=arabic_module.Num2Word_AR()
    cases.append(dict(language='ar',text=text,expected=namespace['multilingual_cleaners'](text,'ar')))
romanizer=Transliter(academic)
boundary_cases=[chr(0xAC00+final)+chr(0xAC00+initial*588) for final in range(28) for initial in range(19)]
boundary_cases.append(''.join(chr(code) for code in range(0xAC00,0xD7A4)))
(repo/'tests/AiDotNet.Tests/TextToSpeech/ReferenceData/hangul_academic_reference.json').write_text(json.dumps(dict(
    source='hangul_romanize.Transliter(rule.academic), every consonant boundary and every precomposed Hangul syllable',
    cases=[dict(text=t,expected=romanizer.translit(t)) for t in boundary_cases]),ensure_ascii=True,separators=(',',':')),encoding='utf-8')
(repo/'tests/AiDotNet.Tests/TextToSpeech/ReferenceData/xtts_latin_reference.json').write_text(json.dumps(dict(
    source='coqui-ai/TTS xtts.tokenizer.multilingual_cleaners and num2words 0.5.12',cases=cases),ensure_ascii=True,separators=(',',':')),encoding='utf-8')
