"""Grounding DINO reference: Hugging Face transformers' GroundingDinoForObjectDetection (a port of the official
IDEA-Research/GroundingDINO, with converted checkpoints), run in float64 on CPU.

A small model with the released architecture: a Swin backbone (embed 16, depths 1-1-2-1, heads 1-2-2-4, window 4,
stages 2-4 out) and a BERT text backbone (width 32, 2 layers, 2 heads, vocabulary 2100), d_model 32 (one channel per
GroupNorm group), 2 encoder and 2 decoder layers of 4 heads with 2 sampling points, 4 feature levels, 10 queries and a
16-token contrastive head. All weights are random (state dict recorded). The reference computes its sine position
embeddings with float32 intermediates even in a float64 model; they are patched to float64 here, as the C# side computes
them, so the comparison is not limited to float32 precision. The port also casts the text position embedding to the
integer dtype of the position ids, truncating it to 0s and 1s; the official code (get_sine_pos_embed) keeps the floats,
and so does the patch.

For two 3x64x64 images and two prompts (the second padded), the fixture records the final layer's logits and boxes,
every decoder layer's logits and boxes, and the loss: the mean squared box plus the mean squared logit over the scored
(unmasked) prompt tokens, summed over decoder layers; and the loss after one plain SGD step (lr 0.001) on every parameter.
Default: verify; --write: regenerate tests/AiDotNet.Tests/VisionLanguage/ReferenceData/grounding_dino_reference.json.
"""
import argparse
import json
import math
import os
import sys

FIXTURE = "tests/AiDotNet.Tests/VisionLanguage/ReferenceData/grounding_dino_reference.json"

IDS = [[101, 1500, 1600, 1012, 1700, 1012, 102], [101, 1800, 1012, 102, 0, 0, 0]]
MASK = [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0]]


def patch_float64():
    import torch
    import transformers.models.grounding_dino.modeling_grounding_dino as g

    def encode(pos_tensor, num_pos_feats=128, temperature=10000):
        scale = 2 * math.pi
        dim_t = torch.arange(num_pos_feats, dtype=torch.float64, device=pos_tensor.device)
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)
        coords = pos_tensor.double().unbind(-1)
        embeddings = [coord[..., None] * scale / dim_t for coord in coords]
        embeddings = [torch.stack((e[..., 0::2].sin(), e[..., 1::2].cos()), dim=-1).flatten(-2) for e in embeddings]
        if len(embeddings) >= 2:
            embeddings[0], embeddings[1] = embeddings[1], embeddings[0]
        out = torch.cat(embeddings, dim=-1)
        # The port casts to the coordinates' dtype, which for the text encoder's integer position ids truncates every
        # sine and cosine to an integer; the official get_sine_pos_embed returns them as floats.
        return out if not pos_tensor.is_floating_point() else out.to(pos_tensor.dtype)

    def sine_forward(self, pixel_values, pixel_mask):
        y_embed = pixel_mask.cumsum(1, dtype=torch.float64)
        x_embed = pixel_mask.cumsum(2, dtype=torch.float64)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.embedding_dim, dtype=torch.float64, device=pixel_values.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.embedding_dim)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

    g.encode_sinusoidal_position_embedding = encode
    g.GroundingDinoSinePositionEmbedding.forward = sine_forward


def build():
    import torch
    from transformers import BertConfig, GroundingDinoConfig, GroundingDinoForObjectDetection, SwinConfig
    patch_float64()
    torch.manual_seed(7)
    backbone = SwinConfig(embed_dim=16, depths=[1, 1, 2, 1], num_heads=[1, 2, 2, 4], window_size=4, image_size=64,
                          out_indices=[2, 3, 4], drop_path_rate=0.0)
    text = BertConfig(vocab_size=2100, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                      max_position_embeddings=64)
    config = GroundingDinoConfig(backbone_config=backbone, text_config=text, d_model=32, encoder_layers=2, decoder_layers=2,
                                 encoder_attention_heads=4, decoder_attention_heads=4, encoder_ffn_dim=64, decoder_ffn_dim=64,
                                 encoder_n_points=2, decoder_n_points=2, num_queries=10, max_text_len=16,
                                 use_timm_backbone=False, backbone=None, use_pretrained_backbone=False)
    model = GroundingDinoForObjectDetection(config).double().eval()
    with torch.no_grad():  # the zero-initialized heads and gates would hide wiring mistakes
        for name, p in model.named_parameters():
            if p.abs().sum() == 0 or name.endswith("vision_param") or name.endswith("text_param"):
                p.normal_(0, 0.05)
    return model


def compute():
    import torch
    torch.set_default_dtype(torch.float64)
    model = build()
    state = {k: v.detach().flatten().tolist() for k, v in model.state_dict().items()
             if not k.endswith("relative_position_index") and not k.endswith("position_ids")}
    torch.manual_seed(3)
    images = torch.randn(2, 3, 64, 64, dtype=torch.float64)
    ids = torch.tensor(IDS)
    mask = torch.tensor(MASK)

    def run():
        out = model(pixel_values=images, input_ids=ids, attention_mask=mask, output_hidden_states=False)
        hidden = out
        logits, boxes = [], []
        # Every decoder layer's head outputs (the reference's auxiliary outputs, recomputed as it does).
        o = model.model(pixel_values=images, input_ids=ids, attention_mask=mask)
        for level in range(o.intermediate_hidden_states.shape[1]):
            reference = o.init_reference_points if level == 0 else o.intermediate_reference_points[:, level - 1]
            reference = torch.special.logit(reference, eps=1e-5)
            logits.append(model.class_embed[level](o.intermediate_hidden_states[:, level], o.encoder_last_hidden_state_text, mask.bool()))
            boxes.append((model.bbox_embed[level](o.intermediate_hidden_states[:, level]) + reference).sigmoid())
        return out, logits, boxes

    def loss_of(logits, boxes):
        total = 0
        for lg, bx in zip(logits, boxes):
            valid = torch.zeros_like(lg, dtype=torch.bool)
            valid[:, :, :mask.shape[1]] = mask.bool()[:, None, :]
            total = total + (bx ** 2).mean() + (lg[valid] ** 2).mean()
        return total

    trace = {}
    def bert_hook(module, inputs, output):
        trace["bert"] = output.last_hidden_state

    def swin_hook(module, inputs, output):
        for j, (feature, _) in enumerate(output):
            trace[f"swin{j}"] = feature

    layer0 = model.model.encoder.layers[0]

    def fusion_hook(module, inputs, output):
        trace["fusion0_vision"], trace["fusion0_text"] = output[0][0], output[1][0]

    def enhancer_hook(module, inputs, output):
        trace["enhancer0"] = output[0]

    def deformable_hook(module, inputs, output):
        trace["deformable0"] = output[0]

    def projection_hook(module, inputs, output):
        trace["text_projected"] = output

    hooks = [model.model.text_backbone.register_forward_hook(bert_hook),
             model.model.backbone.conv_encoder.register_forward_hook(swin_hook),
             model.model.text_projection.register_forward_hook(projection_hook),
             layer0.fusion_layer.register_forward_hook(fusion_hook),
             layer0.text_enhancer_layer.register_forward_hook(enhancer_hook),
             layer0.deformable_layer.register_forward_hook(deformable_hook)]
    for level in range(4):
        def level_hook(module, inputs, output, level=level):
            trace[f"projected{level}"] = output
        hooks.append(model.model.input_proj_vision[level].register_forward_hook(level_hook))
    with torch.no_grad():
        o = model.model(pixel_values=images, input_ids=ids, attention_mask=mask)
    for h in hooks:
        h.remove()
    trace["encoder_vision"] = o.encoder_last_hidden_state_vision
    trace["encoder_text"] = o.encoder_last_hidden_state_text
    trace["init_reference"] = o.init_reference_points
    trace["intermediate"] = o.intermediate_hidden_states
    out, logits, boxes = run()
    loss = loss_of(logits, boxes)
    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p -= 0.001 * p.grad
        _, logits2, boxes2 = run()
        after = loss_of(logits2, boxes2).item()
    finite = lambda t: [v if math.isfinite(v) else None for v in t.detach().flatten().tolist()]
    return {"state": state, "images": images.flatten().tolist(), "ids": IDS, "mask": MASK,
            "logits": [finite(t) for t in logits], "boxes": [t.detach().flatten().tolist() for t in boxes],
            "final_logits": finite(out.logits), "final_boxes": out.pred_boxes.detach().flatten().tolist(),
            "loss": loss.item(), "loss_after_sgd": after,
            "trace": {k: v.detach().flatten().tolist() for k, v in trace.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = compute()
    if args.write:
        os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
        with open(FIXTURE, "w", encoding="utf-8", newline="\n") as f:
            json.dump(result, f)
        print("wrote", round(result["loss"], 9), round(result["loss_after_sgd"], 9))
        return
    with open(FIXTURE, encoding="utf-8") as f:
        stored = json.load(f)
    same = abs(stored["loss"] - result["loss"]) < 1e-12 and abs(stored["loss_after_sgd"] - result["loss_after_sgd"]) < 1e-12
    print("fixture matches" if same else "fixture differs")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
