"""SAM 2 image reference: Hugging Face transformers' Sam2Model (a port of facebookresearch/sam2, with converted
checkpoints), run in float64 on CPU.

A small model with the released architecture: Hiera (embed 16-32-64-128, heads 1-2-4-8, blocks 1-1-2-1, windows
4-4-4-2, block 3 global, query pooling on the first block of stages 2-4), the FPN neck (width 32, top-down levels 2 and
3), the prompt encoder (mask input channels 8) and the mask decoder (2 two-way layers, 2 heads, MLP 64, IoU head hidden
16), on 128-pixel images (an 8x8 image embedding). Weights are random (state dict recorded); the zero-initialized
position tables and tokens are randomized. Two reference artifacts are removed: the neck rounds its top-down features
through float32 (patched to stay float64), and the port keeps two random Fourier matrices where the official model has
one (the image-wide one is set to the prompt encoder's).

For one normalized 3x128x128 image the fixture records the image embedding and the two high-resolution features, and
for four prompt sets (two points, a box, a box with a point, a box with a mask) the low-resolution logits, IoU scores
and object-score logits with multimask output on and off (off selects by stability, as in evaluation). The loss is the
mean squared logits plus the mean squared IoU over every output; the fixture records it and its value after one plain
SGD step (lr 0.01).
Default: verify; --write: regenerate tests/AiDotNet.Tests/ComputerVision/ReferenceData/sam2_reference.json.
"""
import argparse
import json
import os
import sys

FIXTURE = "tests/AiDotNet.Tests/ComputerVision/ReferenceData/sam2_reference.json"

PROMPTS = [
    {"name": "points", "points": [[40.0, 52.0], [90.0, 70.0]], "labels": [1, 0]},
    {"name": "box", "box": [20.0, 30.0, 100.0, 96.0]},
    {"name": "box+point", "box": [20.0, 30.0, 100.0, 96.0], "points": [[60.0, 60.0]], "labels": [1]},
    {"name": "box+mask", "box": [10.0, 12.0, 70.0, 80.0], "mask": True},
]


def patch_neck():
    import torch
    import torch.nn.functional as F
    import transformers.models.sam2.modeling_sam2 as s

    def forward(self, hidden_states):
        fpn_hidden_states = ()
        fpn_position_encoding = ()
        n = len(self.convs) - 1
        for i in range(n, -1, -1):
            lateral_features = self.convs[n - i](hidden_states[i].permute(0, 3, 1, 2))
            if i not in self.fpn_top_down_levels or i == n:
                prev_features = lateral_features
            else:
                prev_features = lateral_features + F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
            fpn_hidden_states += (prev_features,)
            fpn_position_encoding += (self.position_encoding(prev_features.shape, prev_features.device, prev_features.dtype),)
        return fpn_hidden_states, fpn_position_encoding

    s.Sam2VisionNeck.forward = forward


def build():
    import torch
    from transformers import Sam2Config, Sam2HieraDetConfig, Sam2MaskDecoderConfig, Sam2Model, Sam2PromptEncoderConfig, Sam2VisionConfig
    patch_neck()
    torch.manual_seed(5)
    backbone = Sam2HieraDetConfig(hidden_size=16, embed_dim_per_stage=[16, 32, 64, 128], num_attention_heads_per_stage=[1, 2, 4, 8],
                                  blocks_per_stage=[1, 1, 2, 1], window_size_per_stage=[4, 4, 4, 2], global_attention_blocks=[3],
                                  image_size=[128, 128], window_positional_embedding_background_size=[7, 7])
    vision = Sam2VisionConfig(backbone_config=backbone, backbone_channel_list=[128, 64, 32, 16],
                              backbone_feature_sizes=[[32, 32], [16, 16], [8, 8]], fpn_hidden_size=32)
    prompt = Sam2PromptEncoderConfig(hidden_size=32, image_size=128, patch_size=16, mask_input_channels=8)
    decoder = Sam2MaskDecoderConfig(hidden_size=32, mlp_dim=64, num_hidden_layers=2, num_attention_heads=2, iou_head_hidden_dim=16)
    model = Sam2Model(Sam2Config(vision_config=vision, prompt_encoder_config=prompt, mask_decoder_config=decoder)).double().eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.abs().sum() == 0:
                p.normal_(0, 0.1)
        model.shared_image_embedding.positional_embedding.copy_(model.prompt_encoder.shared_embedding.positional_embedding)
    return model


def compute():
    import torch
    torch.set_default_dtype(torch.float64)
    model = build()
    state = {k: v.detach().flatten().tolist() for k, v in model.state_dict().items()}
    torch.manual_seed(9)
    image = torch.randn(1, 3, 128, 128, dtype=torch.float64)
    low_mask = torch.randn(1, 1, 32, 32, dtype=torch.float64)

    def run():
        results = []
        embeddings = None
        for prompt in PROMPTS:
            for multimask in (True, False):
                kwargs = {}
                if "points" in prompt:
                    kwargs["input_points"] = torch.tensor([[prompt["points"]]], dtype=torch.float64)
                    kwargs["input_labels"] = torch.tensor([[prompt["labels"]]])
                if "box" in prompt:
                    kwargs["input_boxes"] = torch.tensor([[prompt["box"]]], dtype=torch.float64)
                if "mask" in prompt:
                    kwargs["input_masks"] = low_mask
                out = model(pixel_values=image, multimask_output=multimask, **kwargs)
                embeddings = out.image_embeddings
                results.append((prompt["name"], multimask, out.pred_masks[0, 0], out.iou_scores[0, 0], out.object_score_logits[0, 0]))
        return embeddings, results

    embeddings, results = run()
    loss = sum((r[2] ** 2).mean() + (r[3] ** 2).mean() for r in results)
    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p -= 0.01 * p.grad
        after = sum((r[2] ** 2).mean() + (r[3] ** 2).mean() for r in run()[1]).item()
    return {"state": state, "image": image.flatten().tolist(), "low_res_mask": low_mask.flatten().tolist(), "prompts": PROMPTS,
            "embedding": embeddings[-1].detach().flatten().tolist(), "high_res0": embeddings[0].detach().flatten().tolist(),
            "high_res1": embeddings[1].detach().flatten().tolist(),
            "results": [{"prompt": n, "multimask": m, "logits": l.detach().flatten().tolist(), "logits_shape": list(l.shape),
                         "iou": i.detach().flatten().tolist(), "object_score": o.detach().flatten().tolist()} for n, m, l, i, o in results],
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
