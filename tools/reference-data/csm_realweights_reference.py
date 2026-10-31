"""Real-weights reference for CSM: the release's checkpoint (sesame/csm-1b at c92a71e, ckpt.pt) in SesameAILabs/csm's
models.Model (llama-1B backbone, llama-100M decoder) in float64, prompted as the release's run_csm.py prompts it: speaker
0's conversational_a.wav with its transcript as context, then "Hey how are you doing?" for speaker 0.

The prompt audio is loaded and resampled to 24 kHz with torchaudio as run_csm.py does and encoded by the released Mimi
(kyutai/moshiko-pytorch-bf16, tokenizer-e351c8d8-checkpoint125.safetensors, 32 codebooks) in float64; the text is tokenized
with the checkpoint's tokenizer.json (Llama 3's), BOS and EOS around "[speaker]text" as load_llama3_tokenizer's template adds
them. The float32 arithmetic of torchtune and moshi is replaced as in csm_reference.py and mimi_reference.py.

Records the prompt's text tokens and audio codes, the backbone's last hidden state and codebook-0 logits, and the frames
Model.generate_frame samples with top-k 1 (greedy), with every logit vector it samples from.

Usage: python csm_realweights_reference.py CSM_SNAPSHOT_DIR MIMI_SAFETENSORS (with ~/source/repos/.venv-csm and the csm
clone in refsrc/csm). Writes tests/AiDotNet.Tests/TextToSpeech/ReferenceData/csm_realweights_reference.json.
"""
import json
import os
import sys

import torch
import torchaudio
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import csm_reference  # noqa: E402  (float64 torchtune arithmetic; puts refsrc/csm on the path; before the bitsandbytes stub, which transformers rejects)
import mimi_reference  # noqa: E402,F401  (float64 rotary embedding and attention for Mimi, and the bitsandbytes stub)
from moshi.models import loaders  # noqa: E402

import models  # noqa: E402  (refsrc/csm/models.py)

TEXT = ("like revising for an exam I'd have to try and like keep up the momentum because I'd "
        "start really early I'd be like okay I'm gonna start revising now and then like "
        "you're revising for ages and then I just like start losing steam I didn't do that "
        "for the exam we had recently to be fair that was a more of a last minute scenario "
        "but like yeah I'm trying to like yeah I noticed this yesterday that like Mondays I "
        "sort of start the day with this not like a panic but like a")
TARGET = "Hey how are you doing?"
FRAMES = 4


def main():
    snapshot, mimi_path = sys.argv[1], sys.argv[2]
    tokenizer = Tokenizer.from_file(os.path.join(snapshot, "tokenizer.json"))

    def text_ids(text, speaker):
        return [128000] + tokenizer.encode(f"[{speaker}]{text}", add_special_tokens=False).ids + [128001]

    audio, rate = torchaudio.load(os.path.join(snapshot, "prompts", "conversational_a.wav"))
    audio = torchaudio.functional.resample(audio.squeeze(0), orig_freq=rate, new_freq=24000)
    mimi = loaders.get_mimi(mimi_path, device="cpu", num_codebooks=32).double().eval()
    for module in mimi.modules():
        if hasattr(module, "_embedding"):
            module._embedding = None
    with torch.no_grad():
        codes = mimi.encode(audio.double().view(1, 1, -1))[0]   # (32, frames)
    del mimi
    # mimi_reference replaced torch's attention with Mimi's float64 form (another signature); torchtune needs csm_reference's.
    csm_reference.F.scaled_dot_product_attention = csm_reference.attention64

    model = models.Model(models.ModelArgs(backbone_flavor="llama-1B", decoder_flavor="llama-100M", text_vocab_size=128256,
                                          audio_vocab_size=2051, audio_num_codebooks=32))
    state = torch.load(os.path.join(snapshot, "ckpt.pt"), map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state)   # copied into the float64 parameters
    del state
    model = model.double().eval()

    context_text = text_ids(TEXT, 0)
    target_text = text_ids(TARGET, 0)
    rows, masks = [], []
    for t in context_text:
        rows.append([0] * 32 + [t])
        masks.append([False] * 32 + [True])
    for frame in codes.T.tolist() + [[0] * 32]:
        rows.append(frame + [0])
        masks.append([True] * 32 + [False])
    for t in target_text:
        rows.append([0] * 32 + [t])
        masks.append([False] * 32 + [True])
    tokens = torch.tensor(rows)[None]
    mask = torch.tensor(masks)[None]

    with torch.no_grad():
        embeds = (model._embed_tokens(tokens) * mask.unsqueeze(-1)).sum(dim=2)
        hidden = model.backbone(embeds)[0, -1]
        logits0 = model.codebook0_head(hidden)

        sampled_logits = []
        sample_topk = models.sample_topk

        def recording_sample(logits, topk, temperature):
            sampled_logits.append(logits[0].tolist())
            return sample_topk(logits, topk, temperature)

        models.sample_topk = recording_sample
        model.setup_caches(1)
        curr_tokens, curr_mask = tokens, mask
        curr_pos = torch.arange(tokens.size(1))[None]
        generated = []
        for _ in range(FRAMES):
            sample = model.generate_frame(curr_tokens, curr_mask, curr_pos, 0.9, 1)
            if torch.all(sample == 0):
                break
            generated.append(sample[0].tolist())
            curr_tokens = torch.cat([sample, torch.zeros(1, 1).long()], dim=1).unsqueeze(1)
            curr_mask = torch.cat([torch.ones_like(sample).bool(), torch.zeros(1, 1).bool()], dim=1).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1

    result = {
        "context_text": context_text, "target_text": target_text, "codes": codes.T.tolist(),
        "hidden": hidden.tolist(), "logits0": logits0.tolist(), "generated": generated, "sampled_logits": sampled_logits,
    }
    path = os.path.join(HERE, "..", "..", "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", "csm_realweights_reference.json")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle)
        handle.write("\n")
    print(len(context_text), codes.shape, len(target_text), generated, len(sampled_logits))


if __name__ == "__main__":
    main()
