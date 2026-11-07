"""Export Spark's exact added-token order and real tokenizer cases.

Arguments: original LLM/tokenizer_config.json, original LLM/tokenizer.json.
"""
import gzip
import json
from pathlib import Path
import sys
from tokenizers import Tokenizer

repo = Path(__file__).resolve().parents[2]
config = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
tokens = sorted((int(k), value['content']) for k, value in config['added_tokens_decoder'].items() if int(k) >= 151665)
assert all(index == 151665 + position for position, (index, _) in enumerate(tokens))
with gzip.open(repo / 'src/TextToSpeech/FrontEnd/Resources/spark_added_tokens.json.gz', 'wt', encoding='utf-8') as stream:
    json.dump([token for _, token in tokens], stream, ensure_ascii=False, separators=(',', ':'))
tokenizer = Tokenizer.from_file(sys.argv[2])
texts = ['Hello, world!', '你好，世界！', '日本語と한국어', "Don't stop... 12345", ' Émile 😀 e\u0301 ',
    '<|task_tts|><|start_content|>Hello.<|end_content|><|start_global_token|><|bicodec_global_15|><|end_global_token|>',
    '<|bicodec_global_0|><|bicodec_global_4095|><|bicodec_semantic_0|><|bicodec_semantic_8191|>',
    '<|task_controllable_tts|><|start_style_label|><|gender_0|><|pitch_label_2|><|speed_label_4|><|end_style_label|>']
data = dict(source='SparkAudio/Spark-TTS-0.5B LLM/tokenizer.json and tokenizer_config.json',
    cases=[dict(text=text, ids=tokenizer.encode(text).ids) for text in texts])
(repo / 'tests/AiDotNet.Tests/TextToSpeech/ReferenceData/spark_text_reference.json').write_text(
    json.dumps(data, ensure_ascii=True, separators=(',', ':')), encoding='utf-8')
print(len(tokens), 'Spark added tokens exported')
