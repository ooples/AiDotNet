"""Reference fixture for MeloTTS's text side as the model reads it: utils.get_text_for_tts_infer up to the BERT call
(clean_text, cleaned_text_to_sequence, add_blank's intersperse and word2ph doubling), the BERT tokenizer's input ids,
and split_utils.split_sentence.

Runs MeloTTS's own modules (myshell-ai/MeloTTS at 2091453). Run once in .venv-melo312 for EN, ZH, ZH_MIX_EN, JP, ES and
FR and once in .venv-meloko312 for KR (python-mecab-ko and mecab-python3 cannot share a venv on Windows); each run
merges its languages into tests/AiDotNet.Tests/TextToSpeech/ReferenceData/melo_tts_text_reference.json.
"""
import argparse
import importlib.util
import json
import os
import sys
import types

os.environ.setdefault("LANG", "C.UTF-8")   # gruut's currency symbols, as MeloTTS runs on Linux

HERE = os.path.dirname(os.path.abspath(__file__))
MELO = os.path.normpath(os.path.join(HERE, "..", "..", "..", "refsrc", "MeloTTS", "melo"))
TEXT = os.path.join(MELO, "text")
OUT = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", "melo_tts_text_reference.json")

melo = types.ModuleType("melo")
melo.__path__ = [MELO]
sys.modules["melo"] = melo
package = types.ModuleType("melo.text")
package.__path__ = [TEXT]
sys.modules["melo.text"] = package


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


symbols = load("melo.text.symbols", os.path.join(TEXT, "symbols.py"))
for name in dir(symbols):
    if not name.startswith("_"):
        setattr(package, name, getattr(symbols, name))

SENTENCES = {
    "EN": ["Hello world, this is MeloTTS.", "It costs $5.50 and arrives at 10:30 pm!"],
    "ZH": ["你好，世界！今天天气真不错。", "我有3个苹果和12.5公斤的米。"],
    "ZH_MIX_EN": ["我最近在学习machine learning，希望能够在未来的artificial intelligence领域有所建树。"],
    "JP": ["こんにちは、世界！今日は1789年7月14日です。"],
    "KR": ["안녕하세요, 만나서 반갑습니다!", "사과 2개와 배 10개를 샀다."],
    "ES": ["¡Hola! ¿Cómo estás? Muy bien, gracias.", "El niño comió 21 manzanas."],
    "FR": ["Bonjour, comment allez-vous ? Très bien.", "Les enfants ont 21 amis."],
}
SPLIT = {
    "EN": ("I didn’t know what to do. I said please kill her because it would be better than being kidnapped,” Ben said "
           "on Wednesday. “It’s a nightmare. I said ‘please kill her, don’t take her there.’ " * 3),
    "ZH": "好的，我来给你讲一个故事吧。从前有一个小姑娘，她叫做小红。小红非常喜欢在森林里玩耍。",
    "FR": "Bien sûr ! En quelle matière voudriez-vous que je vous parle en français ? Je peux vous fournir des informations.",
    "KR": "안녕. 나는 학생이다. 반갑다!",
}
MODULES = {"EN": "english", "ZH": "chinese", "ZH_MIX_EN": "chinese_mix", "JP": "japanese", "KR": "korean", "ES": "spanish",
           "FR": "french"}


def module_for(language):
    name = MODULES[language]
    if language == "KR":
        load("melo.text.ko_dictionary", os.path.join(TEXT, "ko_dictionary.py"))
        module = load(f"melo.text.{name}", os.path.join(TEXT, f"{name}.py"))
        import g2pkk.g2pkk as g2pkk_module
        g2pkk_module.platform = types.SimpleNamespace(system=lambda: "Linux")
        return module
    if language == "EN" or language == "ZH_MIX_EN":
        japanese = types.ModuleType("melo.text.japanese")
        import ast
        source = open(os.path.join(TEXT, "japanese.py"), encoding="utf-8").read()
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "distribute_phone")
        exec(compile(ast.Module(body=[node], type_ignores=[]), "japanese.py", "exec"), japanese.__dict__)
        sys.modules.setdefault("melo.text.japanese", japanese)
        load("melo.text.english", os.path.join(TEXT, "english.py"))
    if language in ("ZH", "ZH_MIX_EN"):
        load("melo.text.tone_sandhi", os.path.join(TEXT, "tone_sandhi.py"))
        load("melo.text.chinese", os.path.join(TEXT, "chinese.py"))
    if language in ("ES", "FR"):
        for sub in ("es_phonemizer", "fr_phonemizer"):
            sub_package = types.ModuleType(f"melo.text.{sub}")
            sub_package.__path__ = [os.path.join(TEXT, sub)]
            sys.modules.setdefault(f"melo.text.{sub}", sub_package)
    return load(f"melo.text.{name}", os.path.join(TEXT, f"{name}.py"))


def bert_tokenizer(language, module):
    from transformers import AutoTokenizer
    if language == "ZH":
        return AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
    if language == "ZH_MIX_EN":
        return AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
    return module.tokenizer


def intersperse(sequence):
    result = [0] * (len(sequence) * 2 + 1)
    result[1::2] = sequence
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", required=True)
    args = parser.parse_args()
    import gruut  # noqa: F401  (loads before any module that needs the locale)
    data = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {"languages": {}}
    for language in args.languages:
        module = module_for(language)
        tokenizer = bert_tokenizer(language, module)
        entries = []
        for text in SENTENCES[language]:
            if language in ("ES", "FR"):
                gruut._LOCAL.processors = {}
            norm = module.text_normalize(text)
            phones, tones, word2ph = module.g2p(norm)
            ids = [symbols.symbols.index(p) for p in phones]
            tone_start = symbols.language_tone_start_map[language]
            lang = symbols.language_id_map[language]
            seq_tones = [t + tone_start for t in tones]
            word2ph = [w * 2 for w in word2ph]
            word2ph[0] += 1
            entries.append({"text": text, "normalized": norm, "symbols": intersperse(ids), "tones": intersperse(seq_tones),
                            "languages": intersperse([lang] * len(ids)), "word2ph": word2ph,
                            "bert_ids": tokenizer(norm)["input_ids"]})
        split = None
        if language in SPLIT:
            for unused in ("soundfile", "torchaudio"):   # split_utils imports them; splitting uses neither
                sys.modules.setdefault(unused, types.ModuleType(unused))
            split_utils = load("melo.split_utils", os.path.join(MELO, "split_utils.py"))
            split = {"text": SPLIT[language], "sentences": split_utils.split_sentence(SPLIT[language], language_str=language)}
        data["languages"][language] = {"sentences": entries, "split": split}
    with open(OUT, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=1)
        handle.write("\n")
    print({k: len(v["sentences"]) for k, v in data["languages"].items()})


if __name__ == "__main__":
    main()
