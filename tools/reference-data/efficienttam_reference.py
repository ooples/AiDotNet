"""EfficientTAM reference: the official yformer/EfficientTAM modules (EfficientTAMBase with its ViTDet encoder), run in
float64 on CPU.

The official package is copied from EFFICIENTTAM_DIR (default ~/source/repos/refsrc/EfficientTAM) to a temporary folder and
its float32 casts are made float64 (the rotary encodings, position encodings and mask casts), the reference's only
artifacts. A small model with the released architecture: a ViT (width 48, 3 heads, 4 blocks, 3x3 windows on blocks 0 and 2,
global attention on 1 and 3, a 4x4 pretraining position grid resized bicubically to the 8x8 grid), the ViTDet neck to width
128, SAM's prompt encoder and mask decoder without high-resolution features (2 layers, 2 heads, MLP 64, IoU hidden 16), and
SAM 2's memory modules (2 memory-attention layers, FFN 64, memory width 64, a 3-slot bank, 4 object pointers), on 128-pixel
frames. Weights are random (rounded to 1e-4; the zero-initialized tables randomized and the object-score bias raised by 2 so
tracked frames find the object); the state dict, shared by the three variants, is recorded under the official names.

The video predictor's build overrides are applied as build_efficienttam_video_predictor applies them: dynamic multimask via
stability (0.05, 0.98) and binarized memory masks for clicked frames.

For each memory cross-attention (SAM 2's full attention, and the efficient 2x2 pooled one in its two official forms,
EfficientRoPEAttention1 and 2), five random normalized frames are tracked as the official track_step does in evaluation: a
click on frame 0 and a mask on frame 2 decoded on their own, then frames 1, 3 and 4 conditioned on the memory. The fixture
records every frame's low-resolution logits and object score.
Default: verify; --write: regenerate tests/AiDotNet.Tests/ComputerVision/ReferenceData/efficienttam_reference.json.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile

FIXTURE = "tests/AiDotNet.Tests/ComputerVision/ReferenceData/efficienttam_reference.json"
S = 128


def load_official():
    source = os.environ.get("EFFICIENTTAM_DIR", os.path.expanduser("~/source/repos/refsrc/EfficientTAM"))
    target = tempfile.mkdtemp()
    package = os.path.join(target, "efficient_track_anything")
    shutil.copytree(os.path.join(source, "efficient_track_anything"), package, ignore=shutil.ignore_patterns("csrc", "*.so", "configs"))
    for folder, _, files in os.walk(os.path.join(package, "modeling")):
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(folder, name)
            with open(path, encoding="utf-8") as f:
                text = f.read()
            text = text.replace("torch.float32", "torch.float64").replace(".float()", ".double()").replace("dtype=torch.float)", "dtype=torch.float64)").replace(".to(torch.float)", ".to(torch.float64)")
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
    sys.path.insert(0, target)
    import efficient_track_anything.modeling as m  # noqa: F401
    return target


def build(attention):
    import torch
    from efficient_track_anything.modeling.backbones.vitdet import ViT
    from efficient_track_anything.modeling.backbones.image_encoder import ImageEncoder, ViTDetNeck
    from efficient_track_anything.modeling.position_encoding import PositionEmbeddingSine
    from efficient_track_anything.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
    from efficient_track_anything.modeling.memory_encoder import MemoryEncoder, MaskDownSampler, Fuser, CXBlock
    from efficient_track_anything.modeling.sam import transformer as tr
    from efficient_track_anything.modeling.sam.mask_decoder import MaskDecoder
    from efficient_track_anything.modeling.efficienttam_base import EfficientTAMBase

    torch.manual_seed(5)
    trunk = ViT(img_size=S, patch_size=16, embed_dim=48, depth=4, num_heads=3, window_size=3, window_block_indexes=[0, 2],
                pretrain_img_size=64)
    neck = ViTDetNeck(position_encoding=PositionEmbeddingSine(num_pos_feats=128, normalize=True), d_model=128,
                      backbone_channel_list=[48], neck_norm="LN")
    encoder = ImageEncoder(trunk=trunk, neck=neck, scalp=0)
    cross = {"full": tr.RoPEAttention, "scores": tr.EfficientRoPEAttention1, "keys": tr.EfficientRoPEAttention2}[attention]
    layer = MemoryAttentionLayer(
        activation="relu", dim_feedforward=64, dropout=0.0, pos_enc_at_attn=False,
        self_attention=tr.RoPEAttention(rope_theta=10000.0, feat_sizes=[8, 8], embedding_dim=128, num_heads=1, downsample_rate=1, dropout=0.0),
        d_model=128, pos_enc_at_cross_attn_keys=True, pos_enc_at_cross_attn_queries=False,
        cross_attention=cross(rope_theta=10000.0, feat_sizes=[8, 8], rope_k_repeat=True, embedding_dim=128, num_heads=1, downsample_rate=1,
                              dropout=0.0, kv_in_dim=64))
    memory_attention = MemoryAttention(d_model=128, pos_enc_at_input=True, layer=layer, num_layers=2)
    memory_encoder = MemoryEncoder(
        out_dim=64, position_encoding=PositionEmbeddingSine(num_pos_feats=64, normalize=True),
        mask_downsampler=MaskDownSampler(embed_dim=128, kernel_size=3, stride=2, padding=1),
        fuser=Fuser(CXBlock(dim=128, kernel_size=7, padding=3, layer_scale_init_value=1e-6, use_dwconv=True), num_layers=2),
        in_dim=128)
    model = EfficientTAMBase(
        image_encoder=encoder, memory_attention=memory_attention, memory_encoder=memory_encoder, num_maskmem=3, image_size=S,
        sigmoid_scale_for_mem_enc=20.0, sigmoid_bias_for_mem_enc=-10.0, use_mask_input_as_output_without_sam=True,
        directly_add_no_mem_embed=True, multimask_output_in_sam=True, iou_prediction_use_sigmoid=True, use_obj_ptrs_in_encoder=True,
        add_tpos_enc_to_obj_ptrs=False, only_obj_ptrs_in_the_past_for_eval=True, pred_obj_scores=True, pred_obj_scores_mlp=True,
        fixed_no_obj_ptr=True, multimask_output_for_tracking=True, use_multimask_token_for_obj_ptr=True, multimask_min_pt_num=0,
        multimask_max_pt_num=1, use_mlp_for_obj_ptr_proj=True, max_obj_ptrs_in_encoder=4, binarize_mask_from_pts_for_mem_enc=True)
    # A smaller decoder than _build_sam_heads hard-codes (MLP 2048, 8 heads, IoU hidden 256).
    model.sam_mask_decoder = MaskDecoder(
        num_multimask_outputs=3, transformer=tr.TwoWayTransformer(depth=2, embedding_dim=128, mlp_dim=64, num_heads=2),
        transformer_dim=128, iou_head_depth=3, iou_head_hidden_dim=16, use_high_res_features=False, iou_prediction_use_sigmoid=True,
        pred_obj_scores=True, pred_obj_scores_mlp=True, use_multimask_token_for_obj_ptr=True, dynamic_multimask_via_stability=True,
        dynamic_multimask_stability_delta=0.05, dynamic_multimask_stability_thresh=0.98)
    model = model.double().eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.abs().sum() == 0:
                p.normal_(0, 0.1)
        model.sam_mask_decoder.pred_obj_score_head.layers[-1].bias += 2.0
        for p in model.parameters():
            p.copy_(torch.round(p * 1e4) / 1e4)
    return model


def track(model, frames, mask):
    import torch
    count = frames.shape[0]
    output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    prompts = {0: {"point_coords": torch.tensor([[[60.0, 50.0]]]), "point_labels": torch.tensor([[1]], dtype=torch.int32)},
               2: mask[None, None]}
    results = {}

    def step(t, prompted):
        backbone_out = model.forward_image(frames[t:t + 1])
        _, vision_feats, vision_pos_embeds, feat_sizes = model._prepare_backbone_features(backbone_out)
        prompt = prompts.get(t)
        out = model.track_step(frame_idx=t, is_init_cond_frame=prompted, current_vision_feats=vision_feats,
                               current_vision_pos_embeds=vision_pos_embeds, feat_sizes=feat_sizes,
                               point_inputs=prompt if isinstance(prompt, dict) else None,
                               mask_inputs=prompt if prompt is not None and not isinstance(prompt, dict) else None,
                               output_dict=output_dict, num_frames=count)
        output_dict["cond_frame_outputs" if prompted else "non_cond_frame_outputs"][t] = out
        results[t] = out

    with torch.no_grad():
        for t in sorted(prompts):
            step(t, True)
        for t in range(count):
            if t not in prompts:
                step(t, False)
    return [{"logits": results[t]["pred_masks"].flatten().tolist(), "object_score": results[t]["object_score_logits"].flatten().tolist()}
            for t in range(count)]


def compute():
    import torch
    torch.set_default_dtype(torch.float64)
    load_official()
    torch.manual_seed(9)
    frames = torch.round(torch.randn(5, 3, S, S) * 1e4) / 1e4
    mask = torch.zeros(S, S)
    mask[30:90, 40:100] = 1.0
    result = {"frames": frames.flatten().tolist(), "mask": mask.flatten().tolist(), "variants": {}}
    shared = None
    for attention in ("full", "scores", "keys"):
        model = build(attention)
        if shared is None:
            shared = model.state_dict()
            result["state"] = {k: v.detach().flatten().tolist() for k, v in shared.items()}
        else:
            model.load_state_dict(shared)
        result["variants"][attention] = {"frames": track(model, frames, mask)}
    return result


def summary(result):
    return [f["object_score"][0] for v in result["variants"].values() for f in v["frames"]] + \
        [result["variants"][a]["frames"][4]["logits"][0] for a in ("full", "scores", "keys")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = compute()
    if args.write:
        os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
        with open(FIXTURE, "w", encoding="utf-8", newline="\n") as f:
            json.dump(result, f, separators=(",", ":"))
        print("wrote", [round(x, 6) for x in summary(result)])
        return
    with open(FIXTURE, encoding="utf-8") as f:
        stored = json.load(f)
    same = all(abs(a - b) < 1e-12 for a, b in zip(summary(stored), summary(result)))
    print("fixture matches" if same else "fixture differs")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
