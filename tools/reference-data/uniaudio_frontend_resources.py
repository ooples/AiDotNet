"""Package the exact released Kaldi lexicon and pinned UniAudio phone inventory."""
import gzip
import hashlib
from pathlib import Path
import subprocess
import sys
import unicodedata

checkout=Path(sys.argv[1]).resolve()
weights=Path(sys.argv[2]).resolve()
root=Path(__file__).resolve().parents[2]/'src/TextToSpeech/FrontEnd/Resources'
phones=subprocess.check_output(['git','show','HEAD:UniAudio/tools/tokenizer/phone/phone_dict'],cwd=checkout)
lexicon=(weights/'align_lexicon.txt').read_bytes()
uppercase=''.join(f'{code:x}\t{chr(code).upper()}\n' for code in range(0x110000) if chr(code).upper()!=chr(code)).encode('utf-8')
for name,data in [('uniaudio-phones.txt',phones),('uniaudio-align-lexicon.txt',lexicon),
                  ('uniaudio-uppercase.tsv',uppercase)]:
    (root/(name+'.gz')).write_bytes(gzip.compress(data,mtime=0))
    print(name,len(data),hashlib.sha256(data).hexdigest(),flush=True)
print('Python Unicode casing version',unicodedata.unidata_version)
