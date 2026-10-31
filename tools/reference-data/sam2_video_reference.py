"""SAM 2 video reference: Hugging Face transformers' Sam2VideoModel (a port of facebookresearch/sam2, with converted
checkpoints), run in float64 on CPU.

A small model with the released architecture, as in sam2_reference.py but 128 wide (the memory is 64 wide, so each object
pointer is two memory tokens): Hiera (embed 8-16-32-64, blocks 1-1-2-1, windows 4-4-4-2, block 3 global), the FPN neck,
the prompt encoder and mask decoder (2 layers, 2 heads, MLP 64), the memory attention (2 layers, FFN 64, 2D RoPE), the
memory encoder (mask downsampler, two ConvNeXt fuser blocks of hidden width 64), a 3-slot memory bank and 4 object
pointers, on 64-pixel frames (a 4x4 embedding). Weights and frames are random, rounded to 1e-4 so the fixture stays small (state dict
recorded; the zero-initialized tables are randomized, and the object-score bias is raised by 2 so tracked frames find the
object). It is built twice, sharing the weights: as SAM 2, and as SAM 2.1 (the temporal encoding of the object pointers and
the occlusion embedding, whose extra weights are recorded separately).

Reference artifacts removed: the neck's float32 rounding (as in sam2_reference.py), the float32 rotary and pointer
encodings and masks, and the bfloat16 storage of the memories (a storage choice of the video predictor) all run in
float64; and a prompted mask's memory is encoded from its sigmoid, as the official track_step does
(is_mask_from_pts = point_inputs is not None), where the port also binarizes it.

Inference: six random (already normalized) frames, a click on frame 1, a mask on frame 3 and an empty mask on frame 5 (the
object absent); each prompted frame is decoded, then the object is propagated forward from frame 1 and back to frame 0. The fixture records every frame's low-resolution
logits and object score, and the memory and pointer of frames 1 and 2.

Training (SAM 2 only, in train mode with no dropout): three frames, a click on frame 0 with one corrective click (the first
prediction's low-resolution logits as the mask input), the object present on frames 0 and 1 and absent on frame 2, every
prediction's loss as the official MultiStepMultiMasksAndIous computes it (focal 20, dice 1, l1 IoU 1 on all masks,
occlusion 1), and the loss after one plain SGD step (lr 0.01).
Default: verify; --write: regenerate tests/AiDotNet.Tests/ComputerVision/ReferenceData/sam2_video_reference.json.
"""
import argparse
import inspect
import json
import os
import sys

FIXTURE = "tests/AiDotNet.Tests/ComputerVision/ReferenceData/sam2_video_reference.json"
S = 64


def patch():
    import torch.nn.functional as F
    import transformers.models.sam2.modeling_sam2 as s2
    import transformers.models.sam2_video.modeling_sam2_video as sv

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

    s2.Sam2VisionNeck.forward = forward
    src = inspect.getsource(sv)
    for a, b in [
        ("dtype=torch.float)", "dtype=torch.float64)"),
        (".float()", ".double()"),
        ("torch.float32", "torch.float64"),
        (".to(torch.bfloat16)", ""),
        ("is_mask_from_pts_per_obj.append(point_inputs is not None or mask_inputs is not None)",
         "is_mask_from_pts_per_obj.append(point_inputs is not None)"),
        ("        sam_output_token = sam_output_tokens[:, :, 0]\n",
         "        self._last_high_res_multimasks = high_res_multimasks\n        sam_output_token = sam_output_tokens[:, :, 0]\n"),
    ]:
        assert a in src, a
        src = src.replace(a, b)
    exec(compile(src, sv.__file__, "exec"), sv.__dict__)
    return sv


def build(sv, sam21, state=None):
    import torch
    from transformers import Sam2HieraDetConfig, Sam2VisionConfig
    torch.manual_seed(5)
    backbone = Sam2HieraDetConfig(hidden_size=8, embed_dim_per_stage=[8, 16, 32, 64], num_attention_heads_per_stage=[1, 2, 4, 8],
                                  blocks_per_stage=[1, 1, 2, 1], window_size_per_stage=[4, 4, 4, 2], global_attention_blocks=[3],
                                  image_size=[S, S], window_positional_embedding_background_size=[7, 7])
    vision = Sam2VisionConfig(backbone_config=backbone, backbone_channel_list=[64, 32, 16, 8],
                              backbone_feature_sizes=[[S // 4, S // 4], [S // 8, S // 8], [S // 16, S // 16]], fpn_hidden_size=128)
    prompt = sv.Sam2VideoPromptEncoderConfig(hidden_size=128, image_size=S, patch_size=16, mask_input_channels=8)
    decoder = sv.Sam2VideoMaskDecoderConfig(hidden_size=128, mlp_dim=64, num_hidden_layers=2, num_attention_heads=2, iou_head_hidden_dim=16)
    config = sv.Sam2VideoConfig(
        vision_config=vision, prompt_encoder_config=prompt, mask_decoder_config=decoder, image_size=S,
        memory_attention_hidden_size=128, memory_attention_num_layers=2, memory_attention_feed_forward_hidden_size=64,
        memory_attention_rope_feat_sizes=[S // 16, S // 16], memory_attention_dropout=0.0, memory_attention_rope_dropout=0.0,
        memory_encoder_hidden_size=128, memory_encoder_output_channels=64, mask_downsampler_embed_dim=128,
        memory_fuser_embed_dim=128, memory_fuser_intermediate_dim=64, num_maskmem=3, max_object_pointers_in_encoder=4,
        enable_temporal_pos_encoding_for_object_pointers=sam21, enable_occlusion_spatial_embedding=sam21)
    model = sv.Sam2VideoModel(config).double().eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.abs().sum() == 0:
                p.normal_(0, 0.1)
        model.mask_decoder.pred_obj_score_head.proj_out.bias += 2.0
        for p in model.parameters():
            p.copy_(torch.round(p * 1e4) / 1e4)
        model.shared_image_embedding.positional_embedding.copy_(model.prompt_encoder.shared_embedding.positional_embedding)
        if state is not None:
            model.load_state_dict(state, strict=False)
    return model


def inference(sv, model, frames, mask, empty):
    import torch
    session = sv.Sam2VideoInferenceSession(video=frames, video_height=S, video_width=S, dtype=torch.float64)
    obj = session.obj_id_to_idx(1)
    session.add_point_inputs(obj, 1, {"point_coords": torch.tensor([[[[30.0, 25.0]]]]),
                                      "point_labels": torch.tensor([[[1]]], dtype=torch.int32)})
    session.obj_with_new_inputs = [1]
    model(session, frame_idx=1)
    session.add_mask_inputs(obj, 3, mask[None, None])
    session.obj_with_new_inputs = [1]
    model(session, frame_idx=3)
    session.add_mask_inputs(obj, 5, empty[None, None])
    session.obj_with_new_inputs = [1]
    model(session, frame_idx=5)
    outputs = {}
    for o in model.propagate_in_video_iterator(session):
        outputs[o.frame_idx] = o
    for o in model.propagate_in_video_iterator(session, start_frame_idx=1, reverse=True):
        outputs[o.frame_idx] = o
    store = session.output_dict_per_obj[obj]
    frames_out = []
    for t in range(frames.shape[0]):
        o = outputs[t]
        frames_out.append({"logits": o.pred_masks.flatten().tolist(), "object_score": o.object_score_logits.flatten().tolist()})
    detail = {}
    for t, key in [(1, "cond_frame_outputs"), (2, "non_cond_frame_outputs")]:
        d = store[key][t]
        detail[str(t)] = {"memory": d["maskmem_features"].permute(1, 2, 0).flatten().tolist(),
                          "pointer": d["object_pointer"].flatten().tolist()}
    return frames_out, detail


def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2, loss_on_multimask=False):
    import torch.nn.functional as F
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if loss_on_multimask:
        return loss.flatten(2).mean(-1)
    return loss.mean(1).sum()


def dice_loss(inputs, targets):
    inputs = inputs.sigmoid().flatten(2)
    targets = targets.flatten(2)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    return 1 - (numerator + 1) / (denominator + 1)


def iou_loss(inputs, targets, pred_ious):
    import torch
    import torch.nn.functional as F
    pred_mask = inputs.flatten(2) > 0
    gt_mask = targets.flatten(2) > 0
    area_i = torch.sum(pred_mask & gt_mask, dim=-1).double()
    area_u = torch.sum(pred_mask | gt_mask, dim=-1).double()
    actual_ious = area_i / torch.clamp(area_u, min=1.0)
    return F.l1_loss(pred_ious, actual_ious, reduction="none")


def prediction_loss(src, target, ious, object_score):
    """MultiStepMultiMasksAndIous._update_losses with one object, weights 20/1/1/1, supervise_all_iou, l1 IoU."""
    import torch
    target = target[None, None].expand_as(src)
    focal = sigmoid_focal_loss(src, target, loss_on_multimask=True)
    dice = dice_loss(src, target)
    target_obj = torch.any((target[:, 0] > 0).flatten(1), dim=-1)[..., None].double()
    cls = sigmoid_focal_loss(object_score.reshape(1, 1), target_obj, alpha=-1, gamma=0)
    iou = iou_loss(src, target, ious)
    if focal.size(1) > 1:
        best = torch.argmin(focal * 20 + dice, dim=-1)
        b = torch.arange(1)
        focal, dice, iou = focal[b, best].unsqueeze(1), dice[b, best].unsqueeze(1), iou.mean(dim=-1).unsqueeze(1)
    return 20 * (focal * target_obj).sum() + (dice * target_obj).sum() + (iou * target_obj).sum() + cls


def training_loss(sv, model, frames, gts, point, click):
    import torch
    session = sv.Sam2VideoInferenceSession(video=frames, video_height=S, video_width=S, dtype=torch.float64)
    obj = session.obj_id_to_idx(1)
    count = frames.shape[0]
    total = 0.0
    for t in range(count):
        feats, pos = model._prepare_vision_features(session, t, 1)
        high_res = [x.permute(1, 2, 0).view(x.size(1), x.size(2), *s) for x, s in zip(feats[:-1], model.backbone_feature_sizes[:-1])]
        prompted = t == 0
        pix = model._prepare_memory_conditioned_features(
            inference_session=session, frame_idx=t, obj_idx=obj, is_initial_conditioning_frame=prompted,
            current_vision_features=feats[-1], current_vision_positional_embeddings=pos[-1], num_total_frames=count)

        def run(points, labels, mask_input):
            point_inputs = {"point_labels": labels} if points is not None else None
            multimask = model._use_multimask(prompted, point_inputs)
            out = model._single_frame_forward(input_points=points, input_labels=labels, input_masks=mask_input,
                                              image_embeddings=high_res + [pix], multimask_output=multimask)
            multi = model._last_high_res_multimasks.squeeze(1)
            return out, (multi, out.iou_scores.reshape(1, -1), out.object_score_logits)

        predictions = []
        points = labels = None
        if prompted:
            points = torch.tensor([[[point]]])
            labels = torch.tensor([[[1]]], dtype=torch.int32)
        out, p = run(points, labels, None)
        predictions.append(p)
        if prompted:
            points = torch.cat([points, torch.tensor([[[click[:2]]]])], 2)
            labels = torch.cat([labels, torch.tensor([[[click[2]]]], dtype=torch.int32)], 2)
            out, p = run(points, labels, out.pred_masks)
            predictions.append(p)
        session.store_output(obj, t, output_value={"pred_masks": out.pred_masks, "object_pointer": out.object_pointer},
                             is_conditioning_frame=prompted)
        memory, memory_pos = model._encode_new_memory(feats[-1], out.high_res_masks, out.object_score_logits, is_mask_from_pts=prompted)
        d = session.output_dict_per_obj[obj]["cond_frame_outputs" if prompted else "non_cond_frame_outputs"][t]
        d["maskmem_features"] = memory
        d["maskmem_pos_enc"] = memory_pos
        for multi, ious, score in predictions:
            total = total + prediction_loss(multi, gts[t], ious, score)
    return total


def compute():
    import torch
    torch.set_default_dtype(torch.float64)
    sv = patch()
    torch.manual_seed(9)
    frames = torch.round(torch.randn(6, 3, S, S) * 1e4) / 1e4
    mask = torch.zeros(S, S)
    mask[15:45, 20:50] = 1.0
    empty = torch.zeros(S, S)
    result = {"frames": frames.flatten().tolist(), "mask": mask.flatten().tolist(), "variants": {}}
    base = build(sv, False)
    shared = base.state_dict()
    result["state"] = {k: v.detach().flatten().tolist() for k, v in shared.items()}
    outputs, detail = inference(sv, base, frames, mask, empty)
    result["variants"]["sam2"] = {"frames": outputs, "detail": detail}
    model = build(sv, True, shared)
    result["state21_extra"] = {k: v.detach().flatten().tolist() for k, v in model.state_dict().items() if k not in shared}
    outputs, detail = inference(sv, model, frames, mask, empty)
    result["variants"]["sam21"] = {"frames": outputs, "detail": detail}

    model = build(sv, False, shared)
    model.train()
    gts = torch.zeros(3, S, S)
    gts[0, 10:40, 15:45] = 1.0
    gts[1, 13:43, 18:48] = 1.0
    point, click = [25.0, 25.0], [50.0, 10.0, 0]
    train_frames = frames[:3]
    loss = training_loss(sv, model, train_frames, gts, point, click)
    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p -= 0.01 * p.grad
        after = training_loss(sv, model, train_frames, gts, point, click).item()
    result["training"] = {"masks": gts.flatten().tolist(), "point": point, "click": click, "loss": loss.item(), "loss_after_sgd": after}
    return result


def summary(result):
    v = result["variants"]
    return [v[n]["frames"][t]["object_score"][0] for n in ("sam2", "sam21") for t in range(6)] + \
        [result["training"]["loss"], result["training"]["loss_after_sgd"]]


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
