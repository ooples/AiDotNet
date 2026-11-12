"""Mega-TTS 2's acoustic autoencoder and language models, composed from the authors' modules, on tiny sizes.

Mega-TTS 2 (Jiang et al. 2024) has no public code. This script composes the model of its Sec. 3 and Appendix A from the
authors' code: NATSpeech's FastSpeechEncoder (its feed-forward filter set to Table 5's 2x width), ConvBlocks and
MultiheadAttention between widths; SyntaSpeech's multi-window discriminator; MegaTTS 3's released ARDurPredictor (the
CodePredictor decoder-only Transformer with rotary positions) for the P-LLM and the duration model; and CVQ-VAE's
VectorQuantiser (cosine codes, closest anchors). It records in
tests/AiDotNet.Tests/TextToSpeech/ReferenceData/megatts2_model_reference.json, in float64 on float32 weights:
- the content encoding; the VQ encoder's hidden states (a linear prenet, a ConvBlocks stack, max pooling by r = 4 with a
  partial last window, a second stack), its cosine and Euclidean codes and CVQ-VAE's loss (beta 0.25);
- the MRTE (a prenet, a stack, max pooling by d = 4, a stack, the content attending to it);
- the decoded mel spectrogram (content plus timbre expanded by the durations, plus the prosody repeated r times);
- the L2 mel loss, and the generator's LSGAN term at given window starts;
- CVQ-VAE's training-mode update of the codebook and its usage from the hidden states;
- three sentences of two speakers as the language models read them: the P-LLM's logits and cross-entropy, and the
  duration model's durations and squared error, with positions per speaker and attention within each speaker.
Default: verify; --write: regenerate. Pass --natspeech, --syntaspeech, --megatts3 and --cvqvae (checkouts). Needs torch
and einops.
"""
import argparse
import importlib.util
import math
import os
import sys

import torch
import torch.nn.functional as F

import _fixture

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/megatts2_model_reference.json"

# NatSpeechFrontEnd.EnglishPhones(): g2p_en's ARPAbet with stress, the punctuation, "|" and <BOS>.
ARPABET = ("AA0 AA1 AA2 AE0 AE1 AE2 AH0 AH1 AH2 AO0 AO1 AO2 AW0 AW1 AW2 AY0 AY1 AY2 B CH D DH EH0 EH1 EH2 ER0 ER1 ER2 EY0 EY1 "
           "EY2 F G HH IH0 IH1 IH2 IY0 IY1 IY2 JH K L M N NG OW0 OW1 OW2 OY0 OY1 OY2 P R S SH T TH UH0 UH1 UH2 UW UW0 UW1 UW2 "
           "V W Y Z ZH").split()
VOCABULARY = ["<pad>", "<EOS>", "<UNK>"] + sorted(set(ARPABET + list("!,.?;:") + ["|", "<BOS>"]))
PHONES = ["<BOS>", "HH", "AH0", "L", "OW1", "|", "W", "ER1", "L", "D", ",", "<EOS>"]
DURATIONS = [1, 2, 3, 1, 4, 0, 2, 3, 2, 1, 3, 3]
# The language models' sentences: (speaker, phones, durations).
SENTENCES = [
    (0, ["<BOS>", "HH", "AY1", "|", "DH", "EH1", "R", "<EOS>"], [1, 2, 3, 0, 2, 4, 2, 1]),
    (0, ["<BOS>", "G", "UH1", "D", "<EOS>"], [2, 1, 5, 2, 1]),
    (1, ["<BOS>", "Y", "EH1", "S", ".", "<EOS>"], [1, 3, 2, 4, 1, 2]),
]
MELS, H, FILTER = 10, 16, 32
P_H, R, CODES, CODE_DIM = 12, 4, 12, 8
T_H, D, T_HEADS = 8, 4, 2
LM_H, LM_HEADS, DUR_H, DUR_HEADS, DUR_CODES = 16, 4, 8, 2, 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--natspeech", required=True)
    parser.add_argument("--syntaspeech", required=True)
    parser.add_argument("--megatts3", required=True)
    parser.add_argument("--cvqvae", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, os.path.abspath(args.natspeech))
    from modules.commons.transformer import FastSpeechEncoder, MultiheadAttention, TransformerFFNLayer
    from modules.commons.conv import ConvBlocks
    import modules.commons.transformer as natt

    # Precision patch: the sinusoidal positions are built in float32 (an explicit dtype); the same in float64.
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float64) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float64).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb
    natt.SinusoidalPositionalEmbedding.get_embedding = staticmethod(get_embedding)
    spec = importlib.util.spec_from_file_location("mwd", os.path.join(args.syntaspeech, "modules", "tts", "syntaspeech", "multi_window_disc.py"))
    mwd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mwd)
    # MegaTTS 3's modules live under its own "tts" package, which NATSpeech's path does not shadow.
    sys.path.insert(0, os.path.abspath(args.megatts3))
    import tts.modules.ar_dur.commons.rot_transformer as rot
    from tts.modules.ar_dur.ar_dur_predictor import ARDurPredictor

    # Precision patches (as megatts3_duration_reference.py): the rotary tables and the attention's softmax in float64.
    def create(self, *, width, length):
        theta = torch.pow(10000, -torch.arange(0, width, 2, dtype=torch.float64) / width)
        m_theta = torch.arange(length, dtype=torch.float64).unsqueeze(1) * theta.unsqueeze(0)
        m_theta = torch.cat([m_theta, m_theta], dim=-1)
        self.register_buffer("cos", m_theta.cos().view([length, width]), persistent=False)
        self.register_buffer("sin", m_theta.sin().view([length, width]), persistent=False)
    rot.RotaryEmbeddings._create_rotary_embed = create
    rot.softmax = lambda x, dim: torch.softmax(x, dim=dim)
    qspec = importlib.util.spec_from_file_location("cvq_quantise", os.path.join(args.cvqvae, "quantise.py"))
    quantise = importlib.util.module_from_spec(qspec)
    qspec.loader.exec_module(quantise)

    stored = None if args.write else _fixture.load(FIXTURE)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(331)

    def blocks(c_in, c_out, kernel):
        return ConvBlocks(c_in, c_out, None, kernel, num_layers=2, layers_in_block=2, c_multiple=2, ln_eps=1e-5, post_net_kernel=3)

    def code_model(width, heads, codes, mse):
        hp = dict(lm_use_enc=False, mega_use_char=False, ling_labels=[], ling_label_dict_size=[], use_pos_embed=False,
                  use_post_ln=False, dur_model_type="ar_mse" if mse else "ar_ce", max_tokens=100000)
        # No layers at construction (its default eight heads do not divide the tiny widths); the layers follow.
        model = ARDurPredictor(hp, H, width, 0, len(VOCABULARY), codes, use_rot_embed=True)
        model.layers = torch.nn.ModuleList([rot.RotTransformerDecoderLayer(width, 0.0, kernel_size=1, num_heads=heads, ffn_hidden_size=4 * width)
                                            for _ in range(2)])
        del model.encoder, model.char_empty_embed   # the phone encoder: Mega-TTS 2's models read the content encoder's states
        return model

    content_encoder = FastSpeechEncoder(len(VOCABULARY), hidden_size=H, num_layers=2, kernel_size=5, num_heads=2)
    for layer in content_encoder.layers:
        layer.op.ffn = TransformerFFNLayer(H, FILTER, kernel_size=5, padding="SAME", act="gelu")
    m = torch.nn.ModuleDict({
        "content": content_encoder,
        "prosody_prenet": torch.nn.Linear(MELS, P_H),
        "prosody_frames": blocks(P_H, P_H, 5),
        "prosody_groups": blocks(P_H, CODE_DIM, 5),
        "codebook": torch.nn.Embedding(CODES, CODE_DIM),
        "prosody_out": torch.nn.Linear(CODE_DIM, H),
        "timbre_prenet": torch.nn.Linear(MELS, T_H),
        "timbre_frames": blocks(T_H, T_H, 3),
        "timbre_groups": blocks(T_H, T_H, 3),
        "timbre_attention": MultiheadAttention(H, T_HEADS, kdim=T_H, vdim=T_H, bias=True),
        "decoder": blocks(H, MELS, 5),
        "plm": code_model(LM_H, LM_HEADS, CODES, False),
        "dur": code_model(DUR_H, DUR_HEADS, DUR_CODES, True),
        "disc": mwd.MultiWindowDiscriminator([8, 16], freq_length=MELS, hidden_size=6),
    })
    with torch.no_grad():
        for name, p in m.named_parameters():
            if stored is not None:
                p.copy_(torch.tensor(stored["state"][name], dtype=torch.float32).reshape(p.shape))
            elif name.endswith("code_emb.weight"):
                p.copy_((0.3 * torch.randn_like(p)).float())
                p[0] = 0   # the padding row, as nn.Embedding(padding_idx=0) keeps it
            elif name == "dur.project_out_dim.0.bias":
                p.copy_(torch.full_like(p, 2.0))   # durations of a few frames
            elif name.endswith("weight") and p.dim() == 1:
                p.copy_((1 + 0.1 * torch.randn_like(p)).float())
            else:
                p.copy_((0.3 * torch.randn_like(p)).float())
        for name, b in m.named_buffers():
            if "running" not in name:
                continue
            if stored is not None:
                b.copy_(torch.tensor(stored["state"][name]).reshape(b.shape))
            else:
                b.copy_(((torch.rand_like(b) + 0.5) if "var" in name else 0.1 * torch.randn_like(b)).float())
        state32 = {n: v.float().clone() for n, v in m.named_parameters()}
        state32.update({n: v.float().clone() for n, v in m.named_buffers() if "running" in n})
    m.double().eval()

    g = torch.Generator().manual_seed(337)
    frames = sum(DURATIONS)
    if stored is None:
        mel = torch.randn(frames, MELS, generator=g) - 2
        reference = torch.randn(9, MELS, generator=g) - 2
        mels = [torch.randn(sum(d), MELS, generator=g) - 2 for _, _, d in SENTENCES]
        starts = [int(torch.randint(0, frames - 8 + 1, (1,), generator=g)), int(torch.randint(0, frames - 16 + 1, (1,), generator=g))]
    else:
        mel = torch.tensor(stored["mel"]).reshape(frames, MELS)
        reference = torch.tensor(stored["reference"]).reshape(9, MELS)
        mels = [torch.tensor(v).reshape(sum(s[2]), MELS) for v, s in zip(stored["mels"], SENTENCES)]
        starts = stored["starts"]

    def ids(phones):
        return torch.tensor([[VOCABULARY.index(p) for p in phones]])

    def mel2ph_of(durations):
        return torch.repeat_interleave(torch.arange(1, len(durations) + 1), torch.tensor(durations))[None]

    def expand(h, mel2ph):
        # NATSpeech's expand_states: frame f reads phone mel2ph[f] (1-based; 0 reads zeros).
        h = F.pad(h, [0, 0, 1, 0])
        return torch.gather(h, 1, mel2ph[..., None].repeat([1, 1, h.shape[-1]]))

    def pool(x, r):
        return F.max_pool1d(x.transpose(1, 2), r, r, ceil_mode=True).transpose(1, 2)

    ones = lambda t: torch.ones(1, t, 1)

    def prosody_hidden(y):
        hf = m["prosody_frames"](m["prosody_prenet"](y[None]), nonpadding=ones(y.shape[0]))
        pooled = pool(hf, R)
        return m["prosody_groups"](pooled, nonpadding=ones(pooled.shape[1]))

    def quantizer(distance):
        q = quantise.VectorQuantiser(CODES, CODE_DIM, 0.25, distance=distance, anchor="closest")
        with torch.no_grad():
            q.embedding.weight.copy_(m["codebook"].weight)
        return q

    with torch.no_grad():
        tokens = ids(PHONES)
        L = tokens.shape[1]
        mel2ph = mel2ph_of(DURATIONS)
        content = m["content"](tokens)
        hidden = prosody_hidden(mel)
        n = hidden.shape[1]
        cvq = quantizer("cos").eval()
        zq, vq, (_, _, codes) = cvq(hidden.transpose(1, 2)[..., None])
        zq = zq[..., 0].transpose(1, 2)
        _, _, (_, _, l2_codes) = quantizer("l2").eval()(hidden.transpose(1, 2)[..., None])
        prosody = m["prosody_out"](zq)
        prosody_frames = torch.repeat_interleave(prosody, R, dim=1)[:, :frames]
        th = m["timbre_frames"](m["timbre_prenet"](reference[None]), nonpadding=ones(9))
        tp = pool(th, D)
        zt = m["timbre_groups"](tp, nonpadding=ones(tp.shape[1]))
        timbre, _ = m["timbre_attention"](content.transpose(0, 1), zt.transpose(0, 1), zt.transpose(0, 1))
        timbre = timbre.transpose(0, 1)
        x = expand(content + timbre, mel2ph) + prosody_frames
        decoded = m["decoder"](x, nonpadding=ones(frames))[0]
        mel_loss = ((decoded - mel) ** 2).sum() * (1.0 / mel.numel())
        total = mel_loss + vq
        validity, _, _ = m["disc"](decoded[None, None], torch.tensor([frames]), start_frames_wins=[[starts[0]], [starts[1]]])
        adversarial = ((validity - 1) ** 2).sum() * (1.0 / validity.numel())
        # CVQ-VAE in training mode: the usage and the codebook after its dynamic initialization.
        trained = quantizer("cos").train()
        trained(hidden.transpose(1, 2)[..., None])
        updated = trained.embedding.weight.clone()
        usage = trained.embed_prob.clone()

        # The language models: each sentence's codes and pooled content, read per speaker.
        lm_tokens, lm_feats, lm_pos, lm_spk, lm_targets = [], [], [], [], []
        du_tokens, du_feats, du_pos, du_spk, du_targets = [], [], [], [], []
        positions = {}
        du_positions = {}
        sentence_codes = []
        for (speaker, phones, durations), y in zip(SENTENCES, mels):
            c = m["content"](ids(phones))
            sc = quantizer("cos").eval()(prosody_hidden(y).transpose(1, 2)[..., None])[2][2].tolist()
            sentence_codes.append(sc)
            pooled = pool(expand(c, mel2ph_of(durations)), R)[0]
            lm_tokens += [CODES + 1] + [k + 1 for k in sc]
            lm_feats.append(torch.cat([pooled, torch.zeros(1, H)], 0))
            lm_targets += [k + 1 for k in sc] + [0]
            for _ in range(len(sc) + 1):
                lm_pos.append(positions.get(speaker, 0))
                positions[speaker] = lm_pos[-1] + 1
                lm_spk.append(speaker)
            du_tokens += [DUR_CODES + 1] + [min(d, DUR_CODES) for d in durations[:-1]]
            du_feats.append(c[0])
            du_targets += durations
            for _ in durations:
                du_pos.append(du_positions.get(speaker, 0))
                du_positions[speaker] = du_pos[-1] + 1
                du_spk.append(speaker)

        def run(model, tokens_, feats, pos, spk):
            same = torch.tensor(spk)[:, None] == torch.tensor(spk)[None, :]
            x_ling = model.enc_proj(torch.cat(feats, 0))[None]
            return model(None, None, None, None, None, torch.tensor([tokens_]), x_ling=x_ling, attn_mask=same.double(),
                         spk_pos_ids_flat=torch.tensor([pos]))[0]

        logits = run(m["plm"], lm_tokens, lm_feats, lm_pos, lm_spk)
        ce = F.cross_entropy(logits, torch.tensor(lm_targets))
        dur_pred = run(m["dur"], du_tokens, du_feats, du_pos, du_spk)[:, 0]
        dur_mse = ((dur_pred - torch.tensor(du_targets, dtype=torch.float64)) ** 2).mean()
    torch.set_default_dtype(torch.float32)
    data = {
        "phones": PHONES, "durations": DURATIONS, "sentences": [{"speaker": s, "phones": p, "durations": d} for s, p, d in SENTENCES],
        "mel": mel.reshape(-1).tolist(), "reference": reference.reshape(-1).tolist(), "mels": [y.reshape(-1).tolist() for y in mels],
        "starts": starts, "content": content.reshape(-1).tolist(), "hidden": hidden.reshape(-1).tolist(), "codes": codes.tolist(),
        "l2_codes": l2_codes.tolist(), "vq": vq.item(), "timbre_states": zt.reshape(-1).tolist(), "timbre": timbre.reshape(-1).tolist(),
        "decoded": decoded.reshape(-1).tolist(), "mel_loss": mel_loss.item(), "total": total.item(), "adversarial": adversarial.item(),
        "updated": updated.reshape(-1).tolist(), "usage": usage.tolist(), "sentence_codes": sentence_codes,
        "lm_tokens": lm_tokens, "lm_positions": lm_pos, "lm_targets": lm_targets, "logits": logits.reshape(-1).tolist(), "ce": ce.item(),
        "dur_tokens": du_tokens, "dur_positions": du_pos, "dur_pred": dur_pred.tolist(), "dur_mse": dur_mse.item(),
    }
    if args.write:
        data["state"] = {k: v.reshape(-1).tolist() for k, v in state32.items()}
        _fixture.save(FIXTURE, data, __file__)
        print("wrote", FIXTURE, len(state32), "tensors; codes", data["codes"], "l2", data["l2_codes"], "sentences", sentence_codes,
              "durations", [round(v, 3) for v in data["dur_pred"]], "total", data["total"], "ce", data["ce"])
        return
    check = _fixture.Comparison("Mega-TTS 2 model")
    for key in ("codes", "l2_codes", "sentence_codes", "lm_tokens", "lm_targets", "dur_tokens"):
        check.exact(key, stored[key], data[key])
    for key in ("content", "hidden", "timbre_states", "timbre", "decoded", "updated", "usage", "logits", "dur_pred"):
        check.floats(key, stored[key], data[key], 1e-12, float)
    for key in ("vq", "mel_loss", "total", "adversarial", "ce", "dur_mse"):
        check.floats(key, [stored[key]], [data[key]], 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
