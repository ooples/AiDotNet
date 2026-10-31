"""MegaTTS 3's text chunking and audio joining through the release's own code (bytedance/MegaTTS3,
tts/utils/text_utils/split_text.py: chunk_text_english and chunk_text_chinesev2; tts/utils/audio_utils/io.py:
combine_audio_segments). Records in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/megatts3_text_reference.json the
chunks of English, Chinese and mixed texts (the release's own examples among them, at its limits and smaller ones) and
three random segments joined.
Default: verify; --write: regenerate. Pass --megatts3 (checkout).
"""
import argparse
import os
import sys

import numpy as np

import _fixture

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/megatts3_text_reference.json"
ENGLISH = [
    "Washington CNN When President Donald Trump declared in the House Chamber this week that executives at the nation’s top automakers were “so excited” about their prospects amid his new tariff regime, it did not entirely reflect the conversation he’d held with them earlier that day.",
    "Short one. Another: with a colon; and a semicolon!  Double  spaces? Yes.\tTabs,　ideographic space, and an emoji 😀. End",
    "混合的文本。English after it，然后是中文！Another sentence. 最后一句？",
    "",
]
CHINESE = [
    "欢迎收听《TED Talks Daily》，在这里，我们每天为您带来新思想，激发您的好奇心。我是您的主持人，Elise Hugh。当我们去看医生时，医生会评估我们的身体健康状况，检查我们的生命体征，可能还会关注我们的胆固醇水平，确保我们整体处于健康状态。医生可能还会通过一系列问题来检查我们的心理健康。然而，人际交往专家Casley Killam指出，我们在理解健康时忽略了一个关键指标，那就是我们的社会健康。",
    "哇塞！家人们，你们太好运了。我居然发现了一个宝藏零食大礼包，简直适合所有人的口味！有香辣的，让你舌尖跳舞；有盐焗的，咸香可口；还有五香的，香气四溢。就连怀孕的姐妹都吃得津津有味！整整三十包啊！什么手撕蟹柳、辣子鸡、嫩豆干、手撕素肉、鹌鹑蛋、小肉枣肠、猪肉腐、魔芋、魔芋丝等等，应有尽有。",
    "没有标点的一长串中文字符用来测试最后的强制断句情况这里一直没有任何标点符号出现直到很后面才有一个逗号，然后结束",
    "顿号测试、𠀀扩展字符、省略号…破折号—结束.英文句点,逗号!问号?分号;冒号:完",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--megatts3", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, os.path.abspath(args.megatts3))
    from tts.utils.text_utils.split_text import chunk_text_chinesev2, chunk_text_english
    import importlib.util
    spec = importlib.util.spec_from_file_location("io_utils", os.path.join(args.megatts3, "tts", "utils", "audio_utils", "io.py"))
    source = open(spec.origin, encoding="utf-8").read()
    # combine_audio_segments alone (the module imports pydub and others for its other functions).
    start = source.index("def combine_audio_segments")
    end = source.find("\ndef ", start + 1)
    namespace = {"np": np}
    exec(source[start:end if end >= 0 else None], namespace)
    combine = namespace["combine_audio_segments"]

    stored = None if args.write else _fixture.load(FIXTURE)
    english = [{"text": t, "max": m, "chunks": chunk_text_english(t, max_chars=m)} for t in ENGLISH for m in (130, 40)]
    chinese = [{"text": t, "limit": l, "look": a, "chunks": chunk_text_chinesev2(t, limit=l, look_ahead_limit=a)}
               for t in CHINESE for (l, a) in ((60, 30), (12, 5), (20, 3))]
    rng = np.random.default_rng(173)
    segments = [rng.standard_normal(n).tolist() for n in (900, 500, 700)] if stored is None else stored["segments"]
    joined = combine([np.array(s) for s in segments], crossfade_duration=0.02, sr=16000)
    data = {"english": english, "chinese": chinese, "segments": segments, "joined": joined.tolist()}
    if args.write:
        _fixture.save(FIXTURE, data, __file__)
        print("wrote", FIXTURE, [len(e["chunks"]) for e in english], [len(c["chunks"]) for c in chinese], len(joined))
        return
    check = _fixture.Comparison("MegaTTS 3 text")
    check.exact("english", stored["english"], data["english"])
    check.exact("chinese", stored["chinese"], data["chinese"])
    check.floats("joined", stored["joined"], data["joined"], 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
