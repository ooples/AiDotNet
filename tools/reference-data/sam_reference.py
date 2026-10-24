"""Segment Anything reference: the official segment_anything package (facebookresearch/segment-anything, PyPI 1.0).

A small SAM is built from the official classes the way build_sam builds the released ones: ImageEncoderViT (96-pixel
images, 16-pixel patches -> a 6x6 grid, width 32, 2 heads, 3 blocks, windowed attention with window 4 -- so the 6x6
grid is padded to 8x8 -- except block 1, which is global; decomposed relative positions; 16 output channels),
PromptEncoder (width 16, mask input channels 8) and MaskDecoder over a two-layer TwoWayTransformer (2 heads, MLP 32,
IoU head hidden 16). Weights are float32 values run in float64; the zero-initialized position tables are randomized.

For one 3x80x72 image (padded to 96 by Sam.preprocess) the fixture records the state dict, the image embedding, the
dense positional encoding, and for four prompt sets -- two points (one positive, one negative), a box, a box with a
point, and a box with a low-resolution mask -- the low-resolution logits and IoU predictions with multimask output on
and off, and the masks postprocessed to the original size before thresholding. The loss is the mean of every recorded
low-resolution logit squared plus every IoU prediction squared; the fixture records it and its value after one plain SGD
step (lr 0.1) on every parameter.
Default: verify; --write: regenerate tests/AiDotNet.Tests/ComputerVision/ReferenceData/sam_reference.json.
"""
import argparse
import json
import os
import sys

FIXTURE = "tests/AiDotNet.Tests/ComputerVision/ReferenceData/sam_reference.json"


def build():
    from functools import partial
    import torch
    from segment_anything.modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer
    torch.manual_seed(3)
    encoder = ImageEncoderViT(img_size=96, patch_size=16, in_chans=3, embed_dim=32, depth=3, num_heads=2, mlp_ratio=4.0,
                              out_chans=16, qkv_bias=True, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                              use_abs_pos=True, use_rel_pos=True, rel_pos_zero_init=True, window_size=4,
                              global_attn_indexes=(1,))
    prompt = PromptEncoder(embed_dim=16, image_embedding_size=(6, 6), input_image_size=(96, 96), mask_in_chans=8)
    decoder = MaskDecoder(num_multimask_outputs=3, transformer=TwoWayTransformer(depth=2, embedding_dim=16, mlp_dim=32,
                                                                              num_heads=2),
                          transformer_dim=16, iou_head_depth=3, iou_head_hidden_dim=16)
    return Sam(image_encoder=encoder, prompt_encoder=prompt, mask_decoder=decoder).double()


PROMPTS = [
    {"name": "points", "points": [[10.0, 20.0], [50.5, 33.0]], "labels": [1, 0]},
    {"name": "box", "box": [8.0, 12.0, 60.0, 70.0]},
    {"name": "box_point", "box": [8.0, 12.0, 60.0, 70.0], "points": [[30.0, 40.0]], "labels": [1]},
    {"name": "box_mask", "box": [8.0, 12.0, 60.0, 70.0], "mask": True},
]


def compute():
    import torch
    import torch.nn.functional as F
    torch.set_default_dtype(torch.float64)
    # The prompt encoder casts point coordinates and its position grid to float32 (.to(torch.float), dtype=float32):
    # floors for half-precision training. In this float64 run they stay float64.
    to, ones = torch.Tensor.to, torch.ones
    torch.Tensor.to = lambda self, *a, **k: self if (a and a[0] is torch.float32 and self.dtype == torch.float64) else to(self, *a, **k)
    torch.ones = lambda *a, **k: ones(*a, **{**k, "dtype": torch.float64} if k.get("dtype") is torch.float32 else k)
    sam = build()
    g = torch.Generator().manual_seed(4)
    with torch.no_grad():
        for name, p in sam.named_parameters():
            if "rel_pos" in name or name == "image_encoder.pos_embed":
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
            p.copy_(p.float().double())
        sam.prompt_encoder.pe_layer.positional_encoding_gaussian_matrix.copy_(
            sam.prompt_encoder.pe_layer.positional_encoding_gaussian_matrix.float().double())
    state = {k: v.detach().flatten().tolist() for k, v in sam.state_dict().items()}
    image = (255 * torch.rand(3, 80, 72, generator=g)).float().double()
    low_mask = torch.randn(1, 1, 24, 24, generator=g).float().double()

    def run():
        embedding = sam.image_encoder(sam.preprocess(image)[None])
        dense_pe = sam.prompt_encoder.get_dense_pe()
        results = []
        for p in PROMPTS:
            points = None
            if "points" in p:
                points = (torch.tensor([p["points"]]), torch.tensor([p["labels"]]))
            boxes = torch.tensor([p["box"]]) if "box" in p else None
            masks = low_mask if p.get("mask") else None
            sparse, dense = sam.prompt_encoder(points=points, boxes=boxes, masks=masks)
            for multi in (True, False):
                logits, iou = sam.mask_decoder(image_embeddings=embedding, image_pe=dense_pe,
                                               sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                                               multimask_output=multi)
                results.append((p["name"], multi, logits, iou,
                                sam.postprocess_masks(logits, input_size=(80, 72), original_size=(80, 72))))
        return embedding, dense_pe, results

    embedding, dense_pe, results = run()
    loss = sum((r[2] ** 2).mean() + (r[3] ** 2).mean() for r in results)
    loss.backward()
    with torch.no_grad():
        for p in sam.parameters():
            if p.grad is not None:
                p -= 0.1 * p.grad
        after = sum((r[2] ** 2).mean() + (r[3] ** 2).mean() for r in run()[2]).item()
    return {"state": state, "image": image.flatten().tolist(), "low_res_mask": low_mask.flatten().tolist(),
            "prompts": PROMPTS, "embedding": embedding.detach().flatten().tolist(),
            "dense_pe": dense_pe.flatten().tolist(),
            "results": [{"prompt": n, "multimask": m, "logits": l.detach().flatten().tolist(),
                         "logits_shape": list(l.shape), "iou": i.detach().flatten().tolist(),
                         "masks": u.detach().flatten().tolist()} for n, m, l, i, u in results],
            "loss": loss.item(), "loss_after_sgd": after}


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
