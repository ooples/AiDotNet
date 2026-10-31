"""Reference fixture for GPT-SoVITS's text-to-semantic model (AR/models/t2s_model.py Text2SemanticDecoder).

Runs the release's own module (RVC-Boss/GPT-SoVITS at b4570b8) in the gptsovits-ref image (torch 2.14 CPU), with a
small configuration in float64: the state dict; the text rows; the teacher-forced logits, forward_old's summed
cross-entropy and forward's DPO loss of a two-example batch (the rejected rows fixed); infer_panel_naive with the
sampling noise fixed (multinomial_sample_one_no_sync's Exp(1) draws recorded per step); and logits_to_probs on a few
logit rows.

    docker run --rm -v <refsrc>:/ref -v <repo>:/repo -w /ref/GPT-SoVITS/GPT_SoVITS gptsovits-ref \
        python /repo/tools/reference-data/gptsovits_t2s_reference.py

Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/gptsovits_t2s_reference.json.
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.getcwd())

import AR.models.t2s_model as t2s  # noqa: E402
import AR.models.utils as utils  # noqa: E402

CONFIG = {"model": {"hidden_dim": 16, "embedding_dim": 16, "head": 2, "n_layer": 2, "vocab_size": 33,
                    "phoneme_vocab_size": 20, "dropout": 0, "EOS": 32}}


def tolist(t):
    return t.detach().double().reshape(-1).tolist()


def main():
    torch.manual_seed(1234)
    torch.set_default_dtype(torch.float64)
    model = t2s.Text2SemanticDecoder(CONFIG, top_k=3).double().eval()
    with torch.no_grad():
        model.ar_text_position.alpha.fill_(0.7)
        model.ar_audio_position.alpha.fill_(1.3)
        for p in model.parameters():
            if p.ndim > 1:
                p.mul_(1.0)
    state = {k: tolist(v) for k, v in model.state_dict().items() if not k.endswith("position_ids")}
    shapes = {k: list(v.shape) for k, v in model.state_dict().items()}

    phonemes_a = [3, 7, 1, 19, 4]
    phonemes_b = [5, 2, 8]
    bert_a = torch.randn(1024, len(phonemes_a))
    bert_b = torch.randn(1024, len(phonemes_b))
    semantic_a = [4, 9, 30, 2, 2, 17]
    semantic_b = [11, 3, 25, 0]

    with torch.no_grad():
        text_rows = model.ar_text_position(model.ar_text_embedding(torch.tensor([phonemes_a])) +
                                           model.bert_proj(bert_a.T.unsqueeze(0)))[0]

    # A batch as the collate builds it: phonemes padded with 0, semantic tokens with EOS, BERT with zeros.
    x = torch.tensor([phonemes_a, phonemes_b + [0, 0]])
    x_lens = torch.tensor([5, 3])
    y = torch.tensor([semantic_a, semantic_b + [32, 32]])
    y_lens = torch.tensor([6, 4])
    bert = torch.zeros(2, 1024, 5)
    bert[0] = bert_a
    bert[1, :, :3] = bert_b
    with torch.no_grad():
        xy_pos, xy_attn_mask, targets = model.make_input_data(x, x_lens, y, y_lens, bert)
        xy_dec, _ = model.h((xy_pos, None), mask=xy_attn_mask)
        logits = model.ar_predict_layer(xy_dec[:, x_lens.max() - 1:])
        ce, _ = model.forward_old(x, x_lens, y, y_lens, bert)
        rejected = [[4, 9, 30, 2, 9, 30, 2, 2, 17], [11, 3, 25, 0, 32]]
        rejected_lens = torch.tensor([len(r) for r in rejected])
        rejected_padded = torch.tensor([r + [0] * (9 - len(r)) for r in rejected])
        t2s.make_reject_y = lambda y_o, lens: (rejected_padded, rejected_lens)
        dpo, _ = model.forward(x, x_lens, y, y_lens, bert)

    # Generation with the sampling noise recorded.
    draws = []
    generator = torch.Generator().manual_seed(99)

    def recorded(probs_sort):
        q = torch.empty_like(probs_sort).exponential_(1, generator=generator)
        draws.append(q[0].tolist())
        return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

    utils.multinomial_sample_one_no_sync = recorded
    t2s.sample = utils.sample
    generations = []
    for prompt, top_k, top_p, temperature, penalty in [([5, 9, 13], 5, 1.0, 1.0, 1.35), ([5, 9, 13], 3, 0.8, 0.7, 1.35),
                                                       (None, 15, 1.0, 1.0, 1.35)]:
        draws.clear()
        with torch.no_grad():
            pred, idx = model.infer_panel(torch.tensor([phonemes_a]), torch.tensor([5]),
                                          None if prompt is None else torch.tensor([prompt]), bert_a.unsqueeze(0),
                                          top_k=top_k, top_p=top_p, temperature=temperature, repetition_penalty=penalty)
        kept = pred[0].tolist() if idx == 0 else pred[0, -idx:].tolist()
        generations.append({"prompt": prompt, "top_k": top_k, "top_p": top_p, "temperature": temperature,
                            "repetition_penalty": penalty, "noise": [list(d) for d in draws], "tokens": kept})

    probs = []
    for row, previous, top_k, top_p, temperature, penalty in [
        ([0.5, -1.2, 2.0, 0.1, -0.3, 1.7], [2, 4, 2], 3, 1.0, 1.0, 1.35),
        ([0.5, -1.2, 2.0, 0.1, -0.3, 1.7], [0], 6, 0.6, 0.8, 1.2),
        ([3.0, 1.0, 1.0, -2.0, 0.0, 0.5], [5, 1], 2, 0.9, 1.5, 1.0),
    ]:
        logits_row = torch.tensor([row])
        p = utils.logits_to_probs(logits_row.clone(), torch.tensor([previous]), temperature=temperature, top_k=top_k,
                                  top_p=top_p, repetition_penalty=penalty)
        probs.append({"logits": row, "previous": previous, "top_k": top_k, "top_p": top_p, "temperature": temperature,
                      "repetition_penalty": penalty, "probabilities": p[0].tolist()})

    out = {
        "config": CONFIG["model"], "state": state, "shapes": shapes,
        "phonemes_a": phonemes_a, "phonemes_b": phonemes_b, "bert_a": tolist(bert_a.T), "bert_b": tolist(bert_b.T),
        "semantic_a": semantic_a, "semantic_b": semantic_b, "text_rows": tolist(text_rows),
        "logits": tolist(logits), "logits_shape": list(logits.shape), "targets": targets.tolist(),
        "ce": ce.item(), "dpo": dpo.item(), "rejected": rejected,
        "generations": generations, "probabilities": probs,
    }
    path = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", "gptsovits_t2s_reference.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(out, handle)
        handle.write("\n")
    print("wrote", path, [g["tokens"][:8] for g in generations])


if __name__ == "__main__":
    main()
