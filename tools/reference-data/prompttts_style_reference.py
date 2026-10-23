"""PromptTTS style-encoder reference: THUDM/P-tuning-v2's BertPrefixForSequenceClassification, the P-tuning v2
code PromptTTS fine-tunes its BERT style encoder with (Guo et al. 2023, §4.1; the paper released no code).

Builds tiny randomly initialized models with the reference's code (pass a checkout of THUDM/P-tuning-v2 with
--ptuning; it is not vendored here), stores their weights in float32, runs them in float64 and records, for a style
prompt and its label:
- BERT's last hidden state at [CLS] with the deep prefixes (PromptTTS's style representation, before its projection);
- the classifier's logits and the cross-entropy loss.
for P-tuning v2 with and without the prefix MLP (prefix_projection), in
tests/AiDotNet.Tests/TextToSpeech/ReferenceData/prompttts_style_reference.json. Default: verify; --write: regenerate.

The reference package's model/sequence_classification.py also imports its DeBERTa port, which does not load on
current transformers; it is replaced by an empty module, since the BERT classes do not use it. Run with the Parler-TTS
environment (transformers 4.46.1).
"""
import argparse
import base64
import sys
import types

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

import _fixture

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/prompttts_style_reference.json"
BERT = dict(vocab_size=50, hidden_size=16, num_hidden_layers=2, num_attention_heads=2, intermediate_size=32,
            max_position_embeddings=32, type_vocab_size=2, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)
CASES = {"prefix": dict(prefix_projection=False), "prefix_projection": dict(prefix_projection=True)}
PRE_SEQ_LEN = 3
PREFIX_HIDDEN = 12
NUM_LABELS = 3
PROMPT = [2, 17, 33, 8, 41, 3]               # [CLS] … [SEP] in the tiny vocabulary
LABEL = 2


def load(root):
    sys.path.insert(0, root)
    deberta = types.ModuleType("model.deberta")
    for name in ["DebertaModel", "DebertaPreTrainedModel", "ContextPooler", "StableDropout"]:
        setattr(deberta, name, type(name, (), {}))
    sys.modules["model.deberta"] = deberta
    from model.sequence_classification import BertPrefixForSequenceClassification
    return BertPrefixForSequenceClassification


def build(model_class, case, state=None):
    from transformers import BertConfig
    torch.manual_seed(2294 + list(CASES).index(case))
    config = BertConfig(**BERT, num_labels=NUM_LABELS, pre_seq_len=PRE_SEQ_LEN, prefix_hidden_size=PREFIX_HIDDEN,
                        attn_implementation="eager", **CASES[case])
    model = model_class(config)
    if state is None:
        model.apply(model._init_weights)        # from_pretrained initializes what the checkpoint lacks this way
    else:
        model.load_state_dict(state, strict=True)
    return model.eval()


def run(model):
    ids = torch.tensor([PROMPT])
    with torch.no_grad():
        past = model.get_prompt(batch_size=1)
        mask = torch.ones(1, PRE_SEQ_LEN + len(PROMPT))
        hidden = model.bert(ids, attention_mask=mask, past_key_values=past).last_hidden_state
        out = model(input_ids=ids, attention_mask=torch.ones(1, len(PROMPT)), labels=torch.tensor([LABEL]))
    return {"cls": [float(v) for v in hidden[0, 0].tolist()],
            "logits": [float(v) for v in out.logits[0].tolist()],
            "loss": float(out.loss)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ptuning", required=True, help="a checkout of THUDM/P-tuning-v2")
    parser.add_argument("--write", action="store_true", help="regenerate the fixture (default: verify it)")
    args = parser.parse_args()
    model_class = load(args.ptuning)
    if args.write:
        data = {"bert": BERT, "pre_seq_len": PRE_SEQ_LEN, "prefix_hidden_size": PREFIX_HIDDEN, "num_labels": NUM_LABELS,
                "prompt": PROMPT, "label": LABEL, "cases": {}}
        for case in CASES:
            state = {k: v.float().clone().contiguous() for k, v in build(model_class, case).state_dict().items()}
            model = build(model_class, case, state).double()
            data["cases"][case] = {**CASES[case], "safetensors_base64": base64.b64encode(save_safetensors(state)).decode("ascii"),
                                   **run(model)}
        _fixture.save(FIXTURE, data, __file__)
        print("wrote " + FIXTURE)
        return
    data = _fixture.load(FIXTURE)
    check = _fixture.Comparison("PromptTTS style encoder")
    for case in CASES:
        committed = data["cases"][case]
        again = run(build(model_class, case, load_safetensors(base64.b64decode(committed["safetensors_base64"]))).double())
        for key in ["cls", "logits"]:
            check.floats(f"{case} {key}", committed[key], again[key], 1e-12, float)
        check.floats(f"{case} loss", [committed["loss"]], [again["loss"]], 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
