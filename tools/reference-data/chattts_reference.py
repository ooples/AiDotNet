"""ChatTTS reference: 2noise/ChatTTS's own modules (ChatTTS/model/embed.py Embed, speaker.py Speaker.apply, dvae.py DVAE)
around Hugging Face transformers' LlamaModel, which ChatTTS/model/gpt.py drives with inputs_embeds.

Pass a checkout of 2noise/ChatTTS with --chattts. vector-quantize-pytorch is pinned to 1.14.24, the release current when
ChatTTS shipped (May 2024): later releases scale the residual stages by L^-r instead of (L - 1)^-r and soft-clamp the
input, which decodes ChatTTS's codes to different features.

Builds tiny randomly initialized networks (seeded), runs them in float64 and records in
tests/AiDotNet.Tests/TextToSpeech/ReferenceData/chattts_reference.json:
  - GPT: the embeddings of a sequence of text positions ([Stts] [spk_emb] text [Ptts]) and audio frames, with a speaker
    embedding applied; the final hidden states; the text logits and the 4 code heads' logits; the training objective
    (the mean over the 4 heads of the cross-entropy of each audio frame given the position before it) and that objective
    after one SGD step (lr 0.1) on the GPT and Embed parameters;
  - DVAE: the codes of a waveform (encode: torchaudio mel, downsampling, encoder, GroupedResidualFSQ) and the mel decoded
    from those codes;
  - the hidden-state Decoder (a DVAE without encoder or quantizer): the mel of the GPT's hidden states.
Default: verify; --write: regenerate.
"""
import argparse
import base64
import importlib.util
import sys
import types

import torch
import torch.nn.functional as F
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors
from transformers import LlamaConfig, LlamaModel

import _fixture
from llama_reference import patch_float64

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/chattts_reference.json"
HIDDEN, HEADS, LAYERS, INTERMEDIATE = 16, 2, 2, 32
LEVELS, GROUPS, RESIDUALS = (3, 3), 2, 2
NUM_VQ = GROUPS * RESIDUALS
CODES = 9                                  # 3 x 3
NUM_AUDIO_TOKENS = CODES + 1               # the codes and the end
NUM_TEXT_TOKENS = 21178
SPK_EMB = 21143
# [Stts] [spk_emb] four bert-base-chinese ids [Ptts], then three audio frames and the end frame.
TEXT = [21131, SPK_EMB, 872, 1962, 8024, 102, 21132]
FRAMES = [[1, 4, 7, 2], [0, 8, 3, 5], [6, 2, 2, 1]]
DVAE_DIM = 8
STACK = dict(hidden=8, n_layer=2, bn_dim=8)
WAVE_SAMPLES = 4096
LEARNING_RATE = 0.1


def load_chattts(root):
    """ChatTTS.model.embed / speaker / dvae from a checkout, with ChatTTS.utils stubbed (only load_safetensors)."""
    for package in ["ChatTTS", "ChatTTS.model"]:
        pkg = types.ModuleType(package)
        pkg.__path__ = []
        sys.modules[package] = pkg
    utils = types.ModuleType("ChatTTS.utils")
    utils.load_safetensors = lambda filename: {}
    sys.modules["ChatTTS.utils"] = utils
    loaded = {}
    for name in ["embed", "speaker", "dvae"]:
        spec = importlib.util.spec_from_file_location(f"ChatTTS.model.{name}", f"{root}/ChatTTS/model/{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"ChatTTS.model.{name}"] = module
        spec.loader.exec_module(module)
        loaded[name] = module
    return loaded


def build(m, state=None):
    torch.manual_seed(2294)
    gpt = LlamaModel(LlamaConfig(hidden_size=HIDDEN, intermediate_size=INTERMEDIATE, num_attention_heads=HEADS,
                                 num_hidden_layers=LAYERS, max_position_embeddings=64, vocab_size=8,
                                 attn_implementation="eager")).eval()
    embed = m["embed"].Embed(HIDDEN, NUM_AUDIO_TOKENS, NUM_TEXT_TOKENS, NUM_VQ).eval()
    dvae = m["dvae"].DVAE(decoder_config=dict(idim=DVAE_DIM, odim=DVAE_DIM, **STACK),
                          encoder_config=dict(idim=DVAE_DIM, odim=2 * DVAE_DIM, **STACK),
                          vq_config=dict(dim=2 * DVAE_DIM, levels=LEVELS, G=GROUPS, R=RESIDUALS),
                          dim=DVAE_DIM).eval()
    decoder = m["dvae"].DVAE(decoder_config=dict(idim=HIDDEN // 2, odim=HIDDEN // 2, **STACK), dim=HIDDEN // 2).eval()
    if state is not None:
        for prefix, module in [("gpt.", gpt), ("embed.", embed), ("dvae.", dvae), ("decoder.", decoder)]:
            module.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)
    return gpt, embed, dvae, decoder


def gpt_objective(m, gpt, embed, speaker):
    rows = [[t] * NUM_VQ for t in TEXT] + FRAMES + [[CODES] * NUM_VQ]
    input_ids = torch.tensor([rows])
    text_mask = torch.tensor([[True] * len(TEXT) + [False] * (len(FRAMES) + 1)])
    # Embed.forward runs under inference_mode; call its body directly so the objective keeps its graph.
    emb = m["embed"].Embed.forward.__wrapped__(embed, input_ids, text_mask)
    emb = m["speaker"].Speaker.apply.__wrapped__(None, emb, speaker, input_ids, SPK_EMB, emb.device, inplace=False)
    hidden = gpt(inputs_embeds=emb).last_hidden_state
    text_logits = embed.head_text(hidden)
    code_logits = [head(hidden) for head in embed.head_code]
    first = len(TEXT)
    targets = torch.tensor(FRAMES + [[CODES] * NUM_VQ])                         # [frames + 1, vq]
    loss = sum(F.cross_entropy(code_logits[q][0, first - 1:-1], targets[:, q]) for q in range(NUM_VQ)) / NUM_VQ
    return emb, hidden, text_logits, code_logits, loss


def run(m, gpt, embed, dvae, decoder, speaker, wave):
    emb, hidden, text_logits, code_logits, loss = gpt_objective(m, gpt, embed, speaker)
    parameters = list(gpt.parameters()) + list(embed.parameters())
    grads = torch.autograd.grad(loss, parameters, allow_unused=True)
    with torch.no_grad():
        codes = dvae.forward.__wrapped__(dvae, wave.clone(), "encode")             # [1, G*R, T]
        dvae_mel = dvae.forward.__wrapped__(dvae, codes.clone(), "decode")
        hidden_mel = decoder.forward.__wrapped__(decoder, hidden.detach().permute(0, 2, 1).clone(), "decode")
        for p, g in zip(parameters, grads):
            if g is not None:
                p -= LEARNING_RATE * g
    _, _, _, _, after = gpt_objective(m, gpt, embed, speaker)
    return dict(embeddings=emb, hidden=hidden, text_logits=text_logits, code_logits=torch.stack(code_logits),
                loss=loss, loss_after_sgd=after, codes=codes, dvae_mel=dvae_mel, hidden_mel=hidden_mel)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chattts", required=True, help="a checkout of 2noise/ChatTTS")
    parser.add_argument("--write", action="store_true", help="regenerate the fixture")
    args = parser.parse_args()
    m = load_chattts(args.chattts)
    # LLaMA's float32 steps (RMSNorm variance, rotary tables, softmax) in float64, as llama_reference.py runs them.
    patch_float64()
    if args.write:
        gpt, embed, dvae, decoder = build(m)
        state = {}
        for prefix, module in [("gpt.", gpt), ("embed.", embed), ("dvae.", dvae), ("decoder.", decoder)]:
            for k, v in module.state_dict().items():
                state[prefix + k] = v.detach().clone().contiguous()
        torch.manual_seed(7)
        speaker = torch.randn(HIDDEN)
        wave = 0.3 * torch.randn(WAVE_SAMPLES)
        out = run(m, *(x.double() for x in (gpt, embed, dvae, decoder)), speaker.double(), wave.double())
        data = {"hidden_size": HIDDEN, "heads": HEADS, "layers": LAYERS, "intermediate": INTERMEDIATE, "levels": list(LEVELS),
                "groups": GROUPS, "residuals": RESIDUALS, "dvae_dim": DVAE_DIM, "stack": STACK, "text": TEXT,
                "frames": FRAMES, "learning_rate": LEARNING_RATE,
                "speaker": [float(v) for v in speaker.double().tolist()],
                "wave": [float(v) for v in wave.double().tolist()],
                "safetensors_base64": base64.b64encode(save_safetensors(state)).decode("ascii"),
                "codes": out["codes"][0].t().tolist()}
        for key in ["embeddings", "hidden", "text_logits", "code_logits", "dvae_mel", "hidden_mel"]:
            data[key] = [float(v) for v in out[key].detach().reshape(-1).tolist()]
        data["loss"] = float(out["loss"].detach())
        data["loss_after_sgd"] = float(out["loss_after_sgd"].detach())
        _fixture.save(FIXTURE, data, __file__)
        print("wrote " + FIXTURE)
        return
    data = _fixture.load(FIXTURE)
    gpt, embed, dvae, decoder = build(m, load_safetensors(base64.b64decode(data["safetensors_base64"])))
    out = run(m, *(x.double() for x in (gpt, embed, dvae, decoder)), torch.tensor(data["speaker"], dtype=torch.float64),
              torch.tensor(data["wave"], dtype=torch.float64))
    check = _fixture.Comparison("ChatTTS")
    for key in ["embeddings", "hidden", "text_logits", "code_logits", "dvae_mel", "hidden_mel"]:
        check.floats(key, data[key], out[key].detach().reshape(-1).tolist(), 1e-12, float)
    check.floats("loss", [data["loss"]], [float(out["loss"].detach())], 1e-12, float)
    check.floats("loss_after_sgd", [data["loss_after_sgd"]], [float(out["loss_after_sgd"].detach())], 1e-12, float)
    check.exact("codes", data["codes"], out["codes"][0].t().tolist())
    check.report(exact=False)


if __name__ == "__main__":
    main()
