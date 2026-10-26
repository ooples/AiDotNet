"""MegaTTS 3's latent diffusion transformer through the release's own code (bytedance/MegaTTS3,
tts/modules/llm_dit/dit.py, transformer.py, time_embedding.py, cfm.py, and the ar_dur commons it uses), on a tiny model.

The release's Diffusion.__init__ hardcodes its sizes (width 1024, 24 layers, 16 heads, a 302-phone vocabulary); the test
builds the same modules in the same order at width 32 with two layers, two heads and the release's vocabularies (a
test-size patch: the construction is restated with smaller numbers, nothing else). Records in
tests/AiDotNet.Tests/TextToSpeech/ReferenceData/megatts3_dit_reference.json, in float64:
- the phone encoder's output (forward_ling_encoder: tone embedding, sinusoidal phone positions, the relative-attention
  encoder with its prenet);
- one velocity prediction (_forward) for the three-way batch the release builds (speaker and text, text, neither) and its
  multi-condition CFG combination;
- the training pair (forward): the prediction and target for given noise and time;
- inference with 4 steps, the sway schedule and the release's amo sampler, its draws recorded.
Default: verify; --write: regenerate. Pass --megatts3 (checkout). Needs the .venv-kani environment.
"""
import argparse
import os
import sys

import torch

import _fixture

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/megatts3_dit_reference.json"
WIDTH, LAYERS, HEADS, LATENT, LOCAL, CTX_MASK = 32, 2, 2, 8, 16, 4
PHONES, FRAMES = 12, 24


def build(dit_module, ar, stored):
    from torch import nn
    from tts.modules.llm_dit.transformer import Transformer
    from tts.modules.llm_dit.time_embedding import TimestepEmbedding
    from tts.modules.llm_dit.cfm import ConditionalFlowMatcher
    from tts.modules.ar_dur.commons.layers import Embedding
    from tts.modules.ar_dur.commons.nar_tts_modules import PosEmb
    from tts.modules.ar_dur.commons.rel_transformer import RelTransformerEncoder

    class TinyDiffusion(dit_module.Diffusion):
        def __init__(self):
            nn.Module.__init__(self)
            # Diffusion.__init__ at test size, module for module.
            self.local_cond_dim = LOCAL
            self.ctx_mask_dim = CTX_MASK
            self.in_channels = LATENT
            self.out_channels = LATENT
            self.encoder_dim = WIDTH
            self.encoder_n_layers = LAYERS
            self.encoder_n_heads = HEADS
            self.max_seq_len = 512
            self.multiple_of = 256
            self.ctx_mask_proj = nn.Linear(1, self.ctx_mask_dim)
            self.local_cond_project = nn.Linear(self.out_channels + self.ctx_mask_dim, self.local_cond_dim)
            self.encoder = Transformer(self.encoder_n_layers, self.encoder_dim, self.encoder_n_heads, self.max_seq_len)
            self.x_prenet = nn.Linear(self.in_channels, self.encoder_dim)
            self.prenet = nn.Linear(self.local_cond_dim, self.encoder_dim)
            self.postnet = nn.Linear(self.encoder_dim, self.out_channels)
            self.flow_matcher = ConditionalFlowMatcher(sigma=0.0)
            self.f5_time_embed = TimestepEmbedding(self.encoder_dim)
            self.ph_encoder = RelTransformerEncoder(302, self.encoder_dim, self.encoder_dim, self.encoder_dim * 2, HEADS, LAYERS, 3, 0.0,
                                                    prenet=True, pre_ln=True)
            self.tone_embed = Embedding(32, self.encoder_dim, padding_idx=0)
            self.ph_pos_embed = PosEmb(self.encoder_dim)
            self.ling_pre_net = torch.nn.Sequential(*[
                torch.nn.Conv1d(self.encoder_dim, self.encoder_dim, kernel_size=s * 2, stride=s, padding=s // 2)
                for i, s in enumerate([2, 2])])

    torch.manual_seed(107)
    model = TinyDiffusion()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if stored is not None:
                p.copy_(torch.tensor(stored["state"][name], dtype=torch.float32).reshape(p.shape))
            elif "gamma" in name:
                p.copy_((1 + 0.1 * torch.randn_like(p)).float())
            elif p.dim() == 1:
                p.copy_((0.1 * torch.randn_like(p)).float())
            else:
                p.copy_((torch.randn_like(p) / max(1.0, p.shape[-1]) ** 0.5).float())
    return model


def inference(self, ar, inputs, timesteps, seq_cfg_w):
    """Diffusion.inference restated line for line without its float32 casts (a precision patch): the release casts the
    latent to float32 before each step and back to the model's dtype after it."""
    x_ling = self.forward_ling_encoder(inputs["phone"], inputs["tone"])
    x_ling = self.ling_pre_net(ar.expand_states(x_ling, inputs['dur']).transpose(1, 2)).transpose(1, 2)
    ctx_feature = inputs['lat_ctx']
    ctx_feature[1:, :, :] = 0
    ctx_mask_emb = self.ctx_mask_proj(inputs['ctx_mask'])
    local_cond = torch.cat([ctx_feature, ctx_mask_emb], dim=-1)
    local_cond = self.local_cond_project(local_cond)
    bsz, frm_len = local_cond.size(0), local_cond.size(1)
    sway_sampling_coef = -1.0
    t_schedule = torch.linspace(0, 1, timesteps + 1, dtype=x_ling.dtype)
    t_schedule = t_schedule + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_schedule) - 1 + t_schedule)

    def amo_sampling(z_t, t, t_next, v):
        s = t_next
        c = 3
        o = min(t_next + c * (t_next - t), 1)
        pred_z_o = z_t + (o - t) * v
        a = s / o
        b = ((1 - s) ** 2 - (a * (1 - o)) ** 2) ** 0.5
        noise_i = torch.randn(size=z_t.shape)
        return a * pred_z_o + b * noise_i

    x = torch.randn([1, frm_len, self.out_channels])
    for step_index in range(timesteps):
        sigma = t_schedule[step_index]
        sigma_next = t_schedule[step_index + 1]
        model_out = self._forward(torch.cat([x] * bsz), local_cond, x_ling, timesteps=sigma.unsqueeze(0), ctx_mask=inputs['ctx_mask'],
                                  dur=inputs['dur'], seq_cfg_w=seq_cfg_w)
        x = amo_sampling(x, sigma, sigma_next, model_out)
    return x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--megatts3", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, os.path.abspath(args.megatts3))
    import tts.modules.llm_dit.dit as dit_module
    import tts.modules.ar_dur.ar_dur_predictor as ar
    import tts.modules.llm_dit.transformer as transformer
    stored = None if args.write else _fixture.load(FIXTURE)

    # Precision patches: the rotary table and the timestep embedding are built in float32; the same in float64.
    def precompute(dim, end, theta=10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64)[: (dim // 2)] / dim))
        t = torch.arange(end, dtype=torch.float64)
        freqs = torch.outer(t, freqs)
        return torch.polar(torch.ones_like(freqs), freqs)
    transformer.precompute_freqs_cis = precompute

    def apply_rotary(xq, xk, freqs_cis):
        xq_ = torch.view_as_complex(xq.double().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.double().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis = transformer.reshape_for_broadcast(freqs_cis, xq_)
        return torch.view_as_real(xq_ * freqs_cis).flatten(3).type_as(xq), torch.view_as_real(xk_ * freqs_cis).flatten(3).type_as(xk)
    transformer.apply_rotary_emb = apply_rotary
    original_forward = transformer.Transformer.forward

    # Diffusion.forward passes do_checkpoint=False, which the release's Transformer.forward does not take (so its
    # training forward raises as released); it changes nothing in the computation and is ignored here.
    def transformer_forward(self, x, t, attn_mask, start_pos=0, do_checkpoint=False):
        freqs_cis = torch.view_as_complex(self.freqs_cis.double())[start_pos: start_pos + x.size(1)]
        for layer in self.layers:
            x = layer(x, t, start_pos, freqs_cis, attn_mask)
        return self.out_proj(self.norm(x, t))
    transformer.Transformer.forward = transformer_forward
    import tts.modules.llm_dit.time_embedding as te

    def sinus(self, x, scale=1000):
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float64) * -(torch.log(torch.tensor(10000.0, dtype=torch.float64)) / (half_dim - 1)))
        emb = scale * x.double().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)
    te.SinusPositionEmbedding.forward = sinus

    torch.set_default_dtype(torch.float64)
    model = build(dit_module, ar, stored)
    state32 = {k: v.float().clone() for k, v in model.named_parameters()}
    model = model.double().eval()
    g = torch.Generator().manual_seed(109)
    if args.write:
        phone = torch.randint(1, 302, (1, PHONES), generator=g)
        tone = torch.randint(1, 32, (1, PHONES), generator=g)
        durations = torch.randint(1, 4, (PHONES,), generator=g)
        durations[-1] += (4 * FRAMES) - int(durations.sum()) if int(durations.sum()) < 4 * FRAMES else 0
        mel2ph = torch.repeat_interleave(torch.arange(1, PHONES + 1), durations)[: 4 * FRAMES][None]
        latent_prompt = torch.randn(1, 6, LATENT, generator=g).float().double()
    else:
        phone = torch.tensor([stored["phone"]])
        tone = torch.tensor([stored["tone"]])
        mel2ph = torch.tensor([stored["mel2ph"]])
        latent_prompt = torch.tensor(stored["latent_prompt"]).reshape(1, 6, LATENT)

    with torch.no_grad():
        x_ling_phone = model.forward_ling_encoder(phone, tone)
        # The release's three-way CFG batch: speaker and text, text, neither (cfg mask tokens 301 and 31).
        ph_seq = torch.cat([phone, phone, torch.full_like(phone, 301)], 0)
        tone_seq = torch.cat([tone, tone, torch.full_like(tone, 31)], 0)
        dur = mel2ph.repeat(3, 1)
        target = FRAMES
        lat = latent_prompt.repeat(3, 1, 1)
        ctx_mask = torch.ones_like(lat[:, :, 0:1])
        lat = torch.nn.functional.pad(lat, (0, 0, 0, target - latent_prompt.size(1)))
        lat[1:] = 0.0
        ctx_mask = torch.nn.functional.pad(ctx_mask, (0, 0, 0, target - latent_prompt.size(1)))
        inputs = {"phone": ph_seq, "tone": tone_seq, "lat_ctx": lat * ctx_mask, "ctx_mask": ctx_mask, "dur": dur}
        x_ling = model.forward_ling_encoder(inputs["phone"], inputs["tone"])
        x_ling = model.ling_pre_net(ar.expand_states(x_ling, inputs["dur"]).transpose(1, 2)).transpose(1, 2)
        ctx_feature = inputs["lat_ctx"].clone()
        ctx_feature[1:, :, :] = 0
        local_cond = model.local_cond_project(torch.cat([ctx_feature, model.ctx_mask_proj(inputs["ctx_mask"])], dim=-1))
        g2 = torch.Generator().manual_seed(113)
        x_probe = torch.randn(1, target, LATENT, generator=g2).float().double() if args.write else torch.tensor(stored["x_probe"]).reshape(1, target, LATENT)
        velocity = model._forward(torch.cat([x_probe] * 3), local_cond, x_ling, timesteps=torch.tensor([0.3]), ctx_mask=inputs["ctx_mask"],
                                  seq_cfg_w=[1.6, 2.5])

        # Inference with the draws recorded (the initial latent and each amo step's noise).
        draws = []
        real_randn = torch.randn

        def recording_randn(*a, **k):
            if stored is not None:
                value = torch.tensor(stored["draws"][len(draws)], dtype=torch.float64).reshape(*(a[0] if len(a) == 1 else a) if a else k["size"])
            else:
                value = real_randn(*a, **{kk: vv for kk, vv in k.items() if kk != "device"}).float().double()
            draws.append(value.reshape(-1).tolist())
            return value
        dit_module.torch.randn = recording_randn
        try:
            generated = inference(model, ar, {k: v.clone() for k, v in inputs.items()}, timesteps=4, seq_cfg_w=[1.6, 2.5])
        finally:
            dit_module.torch.randn = real_randn

        # The training pair (Diffusion.forward) for given noise, time and target.
        g3 = torch.Generator().manual_seed(127)
        x1 = torch.randn(1, target, LATENT, generator=g3).float().double() if args.write else torch.tensor(stored["x1"]).reshape(1, target, LATENT)
        x0 = torch.randn(1, target, LATENT, generator=g3).float().double() if args.write else torch.tensor(stored["x0"]).reshape(1, target, LATENT)
        t = torch.tensor([0.37])
        mask1 = torch.zeros(1, target, 1)
        mask1[:, :6] = 1
        train_inputs = {"ctx_mask": mask1, "lat_ctx": x1 * mask1, "lat": x1, "phone": phone, "tone": tone, "mel2ph": mel2ph,
                        "text_mel_mask": torch.ones(1, target)}
        model.flow_matcher.sample_location_and_conditional_flow = lambda a, b: (t, t[:, None, None] * b + (1 - t[:, None, None]) * x0, b - x0)
        real_randn_like = torch.randn_like
        dit_module.torch.randn_like = lambda x: x0
        # The release casts t and the noisy input to bfloat16 for its mixed-precision run; the float64 path keeps them.
        real_bf16 = torch.Tensor.bfloat16
        torch.Tensor.bfloat16 = lambda self: self
        try:
            train_pred, train_target = model(train_inputs)
        finally:
            dit_module.torch.randn_like = real_randn_like
            torch.Tensor.bfloat16 = real_bf16
    torch.set_default_dtype(torch.float32)

    data = {"width": WIDTH, "layers": LAYERS, "heads": HEADS, "latent": LATENT, "local": LOCAL, "ctx_mask_dim": CTX_MASK,
            "phone": phone[0].tolist(), "tone": tone[0].tolist(), "mel2ph": mel2ph[0].tolist(), "latent_prompt": latent_prompt.reshape(-1).tolist(),
            "x_ling_phone": x_ling_phone.reshape(-1).tolist(), "x_probe": x_probe.reshape(-1).tolist(), "velocity": velocity.reshape(-1).tolist(),
            "draws": draws, "generated": generated.reshape(-1).tolist(), "x0": x0.reshape(-1).tolist(), "x1": x1.reshape(-1).tolist(),
            "train_pred": train_pred.reshape(-1).tolist(), "train_target": train_target.reshape(-1).tolist()}
    if args.write:
        data["state"] = {k: v.reshape(-1).tolist() for k, v in state32.items()}
        _fixture.save(FIXTURE, data, __file__)
        print("wrote", FIXTURE, "velocity", velocity.shape, "generated", generated.shape, "draws", len(draws))
        return
    check = _fixture.Comparison("MegaTTS 3 DiT")
    for key in ("x_ling_phone", "velocity", "generated", "train_pred", "train_target"):
        check.floats(key, stored[key], data[key], 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
