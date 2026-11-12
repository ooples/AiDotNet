"""Independent tokenizers-library references for Unicode Whitespace BPE and byte-level regression."""
import json
from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

corpus=['hello world hello', 'नमस्ते दुनिया नमस्ते', 'اللغة العربية', '中文词语', '한국어 단어',
    'café cafe\u0301 a\u20dd', 'Ⅻ 123 foo_bar', 'A\u200dB A\u200cB', '𐐀𐐨𐐀 latin', '🎤🙂 hello',
    'one,two! punctuation... [hi]नमस्ते[SPACE]दुनिया', '[en]hello[SPACE]world']
texts=corpus+['[hi]नमस्ते[SPACE]दुनिया 123', 'unknown 🫠🫠 𐐀𐐨', '\t hello\nworld \u00a0नमस्ते',
    'prefix[SPACE][hi]suffix', 'Ⅻ_123a\u0301𐐀𐐨', '   ', '']
references=[]
for byte_level in [False,True]:
    tokenizer=Tokenizer(models.BPE(unk_token='[UNK]'))
    tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False) if byte_level else pre_tokenizers.Whitespace()
    if byte_level:tokenizer.decoder=decoders.ByteLevel()
    trainer=trainers.BpeTrainer(vocab_size=400,min_frequency=1,show_progress=False,
        special_tokens=['[UNK]','[SPACE]','[en]','[hi]'],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet() if byte_level else [])
    tokenizer.train_from_iterator(corpus,trainer)
    cases=[]
    for text in texts:
        ids=tokenizer.encode(text,add_special_tokens=False).ids
        cases.append(dict(text=text,ids=ids,decoded=tokenizer.decode(ids,skip_special_tokens=False),
            decoded_skip_special=tokenizer.decode(ids,skip_special_tokens=True)))
    references.append(dict(mode='ByteLevel' if byte_level else 'Whitespace',tokenizer=json.loads(tokenizer.to_str()),cases=cases))
tokenizer=Tokenizer(models.BPE(vocab={'[UNK]':0,'a':1,'[UNK]a':2},merges=[('[UNK]','a')],unk_token='[UNK]',fuse_unk=False))
tokenizer.pre_tokenizer=pre_tokenizers.Whitespace()
cases=[]
for text in ['😀a','😀😀a','💡','a a','[UNK]a']:
    ids=tokenizer.encode(text,add_special_tokens=False).ids
    cases.append(dict(text=text,ids=ids,decoded=tokenizer.decode(ids,skip_special_tokens=False),
        decoded_skip_special=tokenizer.decode(ids,skip_special_tokens=True)))
references.append(dict(mode='WhitespaceUnknownMerges',tokenizer=json.loads(tokenizer.to_str()),cases=cases))
tokenizer=Tokenizer(models.BPE(vocab={'[UNK]':0,'a':1,'b':2,'c':3,'ab':4,'bc':5},merges=[('a','b'),('b','c')],unk_token='[UNK]'))
tokenizer.pre_tokenizer=pre_tokenizers.Whitespace()
raw=json.loads(tokenizer.to_str());raw['model']['merges']=[['a','b'],['b','c'],['a','b']]
tokenizer=Tokenizer.from_str(json.dumps(raw))
cases=[]
for text in ['abc','ababc','abc abc','cab']:
    ids=tokenizer.encode(text,add_special_tokens=False).ids
    cases.append(dict(text=text,ids=ids,decoded=tokenizer.decode(ids,skip_special_tokens=False),
        decoded_skip_special=tokenizer.decode(ids,skip_special_tokens=True)))
references.append(dict(mode='WhitespaceDuplicateRanks',tokenizer=raw,cases=cases))
destination=Path(__file__).resolve().parents[2]/'tests/AiDotNet.Tests/Tokenization/ReferenceData/whitespace_bpe_reference.json'
destination.parent.mkdir(parents=True,exist_ok=True)
destination.write_text(json.dumps(dict(source='Hugging Face tokenizers BPE: original Whitespace and ByteLevel pre-tokenizers and decoders',models=references),
    ensure_ascii=True,separators=(',',':')),encoding='utf-8')
print('Wrote',sum(len(item['cases']) for item in references),'independent BPE encode/decode cases')
