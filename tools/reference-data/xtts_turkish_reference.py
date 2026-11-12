"""Execute num2words 0.5.12's original Turkish converter on exhaustive triplets and targeted boundary combinations."""
import json
from pathlib import Path
import random
from num2words import CONVERTER_CLASSES
converter=CONVERTER_CLASSES['tr']
cases=set(range(1000))
for prefix in range(1,1000):
    for suffix in [0,1,10,100,999]:cases.add(prefix*1000+suffix)
randomizer=random.Random(312)
cases.update(randomizer.randrange(10**15) for _ in range(1000))
for level in range(2,7):
    for coefficient in [1,2,9,10,11,20,21,99,100,101,111,900,991,999]:
        for tail in [0,1,1001,1000001]:
            value=coefficient*1000**level+tail
            if value<converter.MAXVAL:cases.add(value)
data=dict(source='num2words 0.5.12 Num2Word_TR.to_cardinal/to_ordinal (unmodified)',cases=[dict(
    value=str(value),cardinal=converter.to_cardinal(value),ordinal=converter.to_ordinal(value)) for value in sorted(cases)])
target=Path(__file__).resolve().parents[2]/'tests/AiDotNet.Tests/TextToSpeech/ReferenceData/xtts_turkish_reference.json'
target.write_text(json.dumps(data,ensure_ascii=True,separators=(',',':')),encoding='utf-8')
