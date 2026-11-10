"""Dicta's Hebrew diacritizer reference (dicta-onnx 1.0.9 Dicta.add_diacritics / OnnxDiacritizationModel.predict) over
dicta-il/dictabert-large-char-menaked at 4dfc20b9, the arg-max logits computed by its BertForDiacritization in float32.
Records each sentence's token ids and character offsets (Hugging Face tokenizers) and its diacritized text.
Usage: python tools/reference-data/dicta_diacritizer_reference.py --model <dictabert-large-char-menaked snapshot> [--write]
       (in .venv-chatterbox; dicta_onnx's model.py is reproduced below without onnxruntime)
"""
import argparse
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fixture  # noqa: E402

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/dicta_diacritizer_reference.json"
SENTENCES = [
    "שלום עולם",
    "בשנת 1948 הוכרזה מדינת ישראל.",
    "הילד הלך לבית הספר בבוקר, ואכל תפוח!",
    "שמש, שמיים ושירים - שלושה דברים.",
    "Hello עברית mixed עם English ו-123 מספרים",
    "לימודיו באוניברסיטה נמשכו ארבע שנים",
    "אני אוהב קפה ☕ בבוקר",
]
MAT_LECT_TOKEN = "<MAT_LECT>"
nikud_pattern = re.compile(r"[ְ-ׇֽׁׂ]")


def predict(model, tokenizer, sentence, nikud_classes, shin_classes):
    # dicta_onnx.model.OnnxDiacritizationModel.predict for one sentence.
    sentence = nikud_pattern.sub("", sentence)
    encoding = tokenizer.encode(sentence)
    ids = torch.tensor([encoding.ids])
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), token_type_ids=torch.zeros_like(ids), return_dict=True).logits
    nikud = np.argmax(out.nikud_logits.numpy(), axis=-1)[0]
    shin = np.argmax(out.shin_logits.numpy(), axis=-1)[0]
    output, prev = [], 0
    for idx, offsets in enumerate(encoding.offsets):
        if offsets[0] > prev:
            output.append(sentence[prev:offsets[0]])
        if offsets[1] - offsets[0] != 1:
            continue
        char = sentence[offsets[0]:offsets[1]]
        prev = offsets[1]
        if not ("א" <= char <= "ת"):
            output.append(char)
            continue
        mark = nikud_classes[nikud[idx]]
        dot = "" if char != "ש" else shin_classes[shin[idx]]
        if mark == MAT_LECT_TOKEN:
            if char not in "אוי":
                mark = ""
            else:
                output.append(char)
                continue
        output.append(char + dot + mark)
    output.append(sentence[prev:])
    return encoding, "".join(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    from tokenizers import Tokenizer
    from transformers import AutoModel
    tokenizer = Tokenizer.from_file(os.path.join(args.model, "tokenizer.json"))
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True).eval()
    cfg = model.config
    rows = []
    for s in SENTENCES:
        encoding, text = predict(model, tokenizer, s, cfg.nikud_classes, cfg.shin_classes)
        rows.append({"text": s, "ids": encoding.ids, "offsets": [list(o) for o in encoding.offsets], "diacritized": text})
    data = {"sentences": rows}
    if args.write:
        _fixture.save(FIXTURE, data, __file__)
        print("wrote", FIXTURE)
        return
    stored = _fixture.load(FIXTURE)
    check = _fixture.Comparison("Dicta")
    check.exact("sentences", stored["sentences"], data["sentences"])
    check.report()


if __name__ == "__main__":
    main()
