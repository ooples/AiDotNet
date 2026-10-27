"""Reference fixture for MaskGCT's Mandarin front end (Amphion models/tts/maskgct/g2p at 26f6883): g2p_ of Chinese and
mixed Chinese-English texts (numbers and dates through cn2an, jieba's words, the lexicon, the polyphone BERT, the tone
sandhi and the bopomofo-to-IPA tables), recording each text's phone string and ids, every polyphone prediction
(BertPolyPredict.predict_process: the text, the character's index and the predicted pinyin), the polyphone model's
logits for each of those calls, and every eSpeak call (for the English runs).

Runs in the release's environment (refsrc/maskgct-docker) like maskgct_g2p_reference.py:

    docker run --rm -v <Amphion>:/amphion -v <repo>:/repo -w /amphion maskgct-g2p \
        python /repo/tools/reference-data/maskgct_mandarin_reference.py

Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/maskgct_mandarin_reference.json.
"""
import json
import sys
import types

import numpy as np

sys.path.insert(0, "/amphion")
# As in maskgct_g2p_reference.py: LangSegment's only PyPI release fails to import, and MaskGCT only needs setfilters.
sys.modules["LangSegment"] = types.SimpleNamespace(setfilters=lambda filters: None)
from phonemizer.backend.espeak.wrapper import EspeakWrapper  # noqa: E402

espeak_calls = []
_text_to_phonemes = EspeakWrapper.text_to_phonemes


def recording(self, text, tie=False):
    out = _text_to_phonemes(self, text, tie)
    espeak_calls.append({"voice": self.voice.language, "text": text, "phonemes": out})
    return out


EspeakWrapper.text_to_phonemes = recording

import models.tts.maskgct.g2p.g2p.mandarin as mandarin  # noqa: E402
from models.tts.maskgct.g2p.g2p_generation import chn_eng_g2p, g2p  # noqa: E402

poly_calls = []
predictor = mandarin.g2pw_poly_predict
_predict_process = predictor.predict_process


def recording_predict(txt_list):
    out = _predict_process(txt_list)
    # The model's logits for this call, as predict_onnx computes them.
    words, _, _ = predictor.get_examples_po(txt_list)
    logits = predictor.session.run(output_names=["outputs"], input_feed={"input_ids": np.asarray([words[0]], dtype=np.int32)})[0][0]
    poly_calls.append({"text": txt_list[0], "index": int(txt_list[1]), "pinyin": out, "ids": words[0],
                       "logits": logits[int(txt_list[1])].tolist()})
    return out


predictor.predict_process = recording_predict


def g2p_(text, language):
    # maskgct_utils.g2p_, verbatim.
    if language in ["zh", "en"]:
        return chn_eng_g2p(text)
    else:
        return g2p(text, sentence=None, language=language)


CASES = [
    "今天是2024年10月18日，气温25℃，比昨天高了3.5%。",
    "我们一起去银行，行长说这件事不好办。",
    "他一边看书一边听音乐，觉得这本书很有意思！",
    "小孩儿在公园里玩儿，不要大声喧哗。",
    "我有1/3的把握，第一次参加比赛，心里有点儿紧张？",
    "MaskGCT是一个非自回归的TTS模型，效果很好。",
    "重庆的长江大桥很长，他重新数了一遍数字。",
    "你好吗？我很好，谢谢。",
]


def main():
    cases = []
    for text in CASES:
        e0, p0 = len(espeak_calls), len(poly_calls)
        phonemes, tokens = g2p_(text, "zh")
        cases.append({"text": text, "phonemes": phonemes, "tokens": tokens, "espeak": espeak_calls[e0:], "poly": poly_calls[p0:]})
    path = "/repo/tests/AiDotNet.Tests/TextToSpeech/ReferenceData/maskgct_mandarin_reference.json"
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump({"cases": cases}, handle, ensure_ascii=False)
        handle.write("\n")
    for c in cases:
        print(len(c["tokens"]), len(c["poly"]), len(c["espeak"]))


if __name__ == "__main__":
    main()
