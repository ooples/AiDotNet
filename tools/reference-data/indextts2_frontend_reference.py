"""IndexTTS2's text front end (indextts/utils/front.py) as the release runs it on Linux, and its Chinese grammars.

On Linux the release builds WeTextProcessing 1.0.4.1's Chinese normalizer with remove_interjections=False and
remove_erhua=False into its own (initially empty) cache directory, so the grammars are compiled with those options rather
than read from the wheel; the English normalizer reads the wheel's (the existing wetext_en_tn_* resources). This script
compiles the Chinese grammars the same way and writes them, loaded and optimized as every later run loads them, to
src/TextToSpeech/FrontEnd/Resources/wetext_indextts_zh_tn_{tagger,verbalizer}.fst.gz. It then runs the release's
TextNormalizer and TextTokenizer (IndexTeam/IndexTTS-2's bpe.model) and records in
tests/AiDotNet.Tests/TextToSpeech/ReferenceData/indextts2_frontend_reference.json, per text: the normalized text, the
pieces and ids, and the segments at 120 and at 12 tokens.

Usage (default: verify; --write: regenerate), in the megatts-ref image (pynini 2.1.6, WeTextProcessing 1.0.4.1):
  docker run --rm -v <repo>:/work -v <index-tts>:/indextts:ro -v <IndexTTS-2 dir>:/hf:ro -w /work megatts-ref \
    sh -c "pip install -q sentencepiece && python tools/reference-data/indextts2_frontend_reference.py --write"
"""
import argparse
import gzip
import io
import os
import sys
import tempfile
import types

import pynini

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import torch  # noqa: F401  (_fixture records its version)
except ImportError:
    sys.modules["torch"] = types.SimpleNamespace(__version__="not installed")
import _fixture  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESOURCES = os.path.join(HERE, "..", "..", "src", "TextToSpeech", "FrontEnd", "Resources")
FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/indextts2_frontend_reference.json"

TEXTS = [
    "Hello world.",
    "你好，世界！这是一个测试。",
    "IndexTTS 正式发布1.0版本了，效果666",
    "晕XUAN4是一种GAN3觉",
    "我爱你！",
    "I love you!",
    "“我爱你”的英语是“I love you”",
    "2.5平方电线",
    "共465篇，约315万字",
    "2002年的第一场雪，下在了2003年",
    "速度是10km/h",
    "现在是北京时间2025年01月11日 20:00",
    "他这条裤子是2012年买的，花了200块钱",
    "电话：135-4567-8900",
    "1键3连",
    "他这条视频点赞3000+，评论1000+，收藏500+",
    "这是1024元的手机，你要吗？",
    "受不liao3你了",
    "”衣裳“不读衣chang2，而是读衣shang5",
    "最zhong4要的是：不要chong2蹈覆辙",
    "IndexTTS 正式发布1.0版本了，效果666。最zhong4要的是：不要chong2蹈覆辙",
    "约瑟夫·高登-莱维特（Joseph Gordon-Levitt is an American actor）",
    "蒂莫西·唐纳德·库克（英文名：Timothy Donald Cook），通称蒂姆·库克（Tim Cook），美国商业经理、工业工程师和工业开发商，现任苹果公司首席执行官。",
    "What's up? It's 3:30 pm and there's $12.50 left.",
    "GPT-5-nano and F5-TTS are tech terms; CosyVoice-2 too.",
    "Contact me at alice123@example.com today.",
    "We'll meet on Jan. 5, 2024 -- don't be late!",
    "我的邮箱是test@abc.com",
    "<行|XING2>人请注意。",
    "a",
    "  。 ",
    "The quick brown fox jumps over the lazy dog. " * 6,
    "这是第一句。这是第二句！这是第三句？还有第四句，以及第五句……最后一句。",
    "Mr. Smith paid 1,234 dollars; Dr. Brown paid 99.9% of it (roughly).",
    "【注意】请在2024-03-15前提交~谢谢",
]


def build_resources(cache):
    data = {}
    for name in ("zh_tn_tagger", "zh_tn_verbalizer"):
        fst = pynini.Fst.read(os.path.join(cache, f"{name}.fst")).optimize()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.fst")
            fst.write(path)
            raw = open(path, "rb").read()
        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0, filename="") as g:
            g.write(raw)
        data[name] = buffer.getvalue()
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--indextts", default="/indextts")
    parser.add_argument("--bpe", default="/hf/bpe.model")
    args = parser.parse_args()
    from tn.chinese.normalizer import Normalizer as ZhNormalizer
    from tn.english.normalizer import Normalizer as EnNormalizer
    sys.path.insert(0, args.indextts)
    # front.py's package pulls in torch-based modules through indextts/utils/__init__? Load the two files directly.
    import importlib.util
    utils = types.ModuleType("indextts.utils")
    utils.__path__ = [os.path.join(args.indextts, "indextts", "utils")]
    sys.modules.setdefault("indextts", types.ModuleType("indextts")).__path__ = [os.path.join(args.indextts, "indextts")]
    sys.modules["indextts.utils"] = utils
    spec = importlib.util.spec_from_file_location("indextts.utils.common", os.path.join(args.indextts, "indextts", "utils", "common.py"))
    common = importlib.util.module_from_spec(spec)
    sys.modules["indextts.utils.common"] = common
    try:
        spec.loader.exec_module(common)
    except ImportError:
        # common.py imports torch for unrelated helpers; the two tokenization helpers need only re.
        source = open(os.path.join(args.indextts, "indextts", "utils", "common.py"), encoding="utf-8").read()
        start = source.index("def tokenize_by_CJK_char")
        end = source.index("\ndef ", source.index("def de_tokenized_by_CJK_char") + 1)
        exec("import re\n" + source[start:end], common.__dict__)
    spec = importlib.util.spec_from_file_location("indextts.utils.front", os.path.join(args.indextts, "indextts", "utils", "front.py"))
    front = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(front)

    with tempfile.TemporaryDirectory() as cache:
        zh = ZhNormalizer(cache_dir=cache, remove_interjections=False, remove_erhua=False, overwrite_cache=False)
        resources = build_resources(cache)
        # Every later run reads the cached grammars back and optimizes them again.
        zh = ZhNormalizer(cache_dir=cache, remove_interjections=False, remove_erhua=False, overwrite_cache=False)
    normalizer = front.TextNormalizer(enable_glossary=True)
    normalizer.zh_normalizer = zh
    normalizer.en_normalizer = EnNormalizer(overwrite_cache=False)
    tokenizer = front.TextTokenizer(args.bpe, normalizer)

    cases = []
    for text in TEXTS:
        pieces = tokenizer.tokenize(text)
        cases.append({
            "text": text,
            "normalized": normalizer.normalize(text) if len(text.strip()) != 1 else None,
            "pieces": pieces,
            "ids": tokenizer.convert_tokens_to_ids(pieces),
            "segments": tokenizer.split_segments(pieces, 120),
            "segments12": tokenizer.split_segments(pieces, 12),
        })
    data = {"cases": cases}
    if args.write:
        for name, blob in resources.items():
            with open(os.path.join(RESOURCES, f"wetext_indextts_{name}.fst.gz"), "wb") as f:
                f.write(blob)
        _fixture.save(FIXTURE, data, __file__)
        for c in cases:
            print(repr(c["normalized"]), len(c["pieces"]))
        print("wrote", {k: len(v) for k, v in resources.items()})
        return
    for name, blob in resources.items():
        with open(os.path.join(RESOURCES, f"wetext_indextts_{name}.fst.gz"), "rb") as f:
            assert f.read() == blob, f"{name}: the resource differs from the compiled grammar"
    stored = _fixture.load(FIXTURE)
    check = _fixture.Comparison("IndexTTS2 front end")
    check.exact("cases", stored["cases"], data["cases"])
    check.report(exact=False)


if __name__ == "__main__":
    main()
