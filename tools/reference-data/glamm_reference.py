"""GLaMM reference: one training sample through GLaMM as the official code (mbzuai-oryx/groundingLMM) computes it, run
in float64 on CPU, assembled from the reference implementations of its parts:

- the global image encoder, Hugging Face's CLIPVisionModel, and LLaVA's projector (Linear, exact GELU, Linear) on the
  second-to-last hidden state's patch tokens;
- the language model, Hugging Face's LlamaForCausalLM with peft LoRA on q_proj and v_proj;
- the region encoder, the transcription of the official MLVLROIQueryModule in glamm_region_reference.py, reading CLIP
  hidden states -2, -5, -8 and -11;
- the grounding encoder, the official segment_anything SAM. GLaMM's fork passes the [SEG] embeddings to the prompt
  encoder as text_embeds, which only appends them to the (empty) sparse prompts beside the no-mask dense embedding; that
  is written out here, since the official SAM has no text_embeds argument;
- transcriptions of the official LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal (mm_use_im_start_end, region
  features replacing <bbox> embeddings), the [SEG] selection (input_ids[:, 1:] == [SEG] after the image's patches minus
  one), GLaMM's CE + 2 BCE + 0.5 DICE loss, and LLaVA's label masking (_process_conversation), with the tokenizer of
  glamm_tokenizer_reference.py.

Small sizes keep the structure: CLIP of 28 px in 7 px patches (a 4x4 grid), width 64 and 10 layers (so the region encoder
has its four levels); a LLaMA of width 32, two layers and the test tokenizer's 400 pieces plus GLaMM's seven tokens, with
LoRA rank 4; the region encoder at the CLIP width with a region width of 32; SAM of 64 px (16 px patches, width 32, two
layers) with a 16-wide prompt. Every parameter and buffer is generated from its name on both sides (synthetic()), so the
fixture stores no weights. The float32 steps of the references (CLIP's and LLaMA's softmax, RMSNorm, rotary tables, the
loss cast, SAM's prompt encoder casts) are patched to float64, as in their own fixture scripts.

The sample: one image, a region box, and a conversation whose instruction holds <bbox> and whose answer holds two [SEG]
tokens, with two ground-truth masks at 24x32 (the SAM input is the resized 48x64 image padded to 64). The fixture records
the ids and labels, the [SEG] mask logits, each loss term and the total, and the total after one SGD step (lr 0.001) on
every parameter.
Default: verify; --write: regenerate tests/AiDotNet.Tests/VisionLanguage/ReferenceData/glamm_reference.json.
"""
import argparse
import base64
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clip_vision_reference  # noqa: E402
import glamm_region_reference  # noqa: E402
import glamm_tokenizer_reference  # noqa: E402
import llama_reference  # noqa: E402

FIXTURE = "tests/AiDotNet.Tests/VisionLanguage/ReferenceData/glamm_reference.json"
TOKENIZER = "tests/AiDotNet.Tests/VisionLanguage/ReferenceData/glamm_tokenizer_reference.json"
SYSTEM = ("A chat between a curious human and an artificial intelligence assistant. "
          "The assistant gives helpful, detailed, and polite answers to the human's questions.")
INSTRUCTION = "The <im_start><image><im_end> provides an overview of the picture.\nCan you segment the dog near region1 <bbox>?"
ANSWER = "Sure, the dog [SEG] and its shadow [SEG]."
BOXES = [[0.10, 0.20, 0.60, 0.75]]
CLIP_HIDDEN, LLM_HIDDEN, SAM_DIM = 64, 32, 16


def uniforms(name, count):
    return glamm_region_reference.uniforms(name, count)


def synthetic(name, shape):
    """Biases small, one-dimensional weights near 1, the rest Kaiming-uniform (identical rule in the C# test)."""
    count = 1
    for d in shape:
        count *= d
    u = uniforms(name, count)
    if name.endswith(".bias") or name.endswith("_bias"):
        return [0.1 * (2 * v - 1) for v in u]
    if len(shape) == 1:
        return [0.5 + v for v in u]
    bound = math.sqrt(6.0 / (count // shape[0]))
    return [bound * (2 * v - 1) for v in u]


def assign(module, prefix, skip=()):
    import torch
    with torch.no_grad():
        for name, t in list(module.state_dict(keep_vars=True).items()):
            if any(s in name for s in skip) or not t.is_floating_point():
                continue
            full = prefix + name
            t.copy_(torch.tensor(synthetic(full, list(t.shape))).reshape(t.shape))


def build():
    import torch
    from functools import partial
    from peft import LoraConfig, get_peft_model
    from segment_anything.modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer
    from transformers import CLIPVisionConfig, CLIPVisionModel, LlamaConfig, LlamaForCausalLM
    clip = CLIPVisionModel(CLIPVisionConfig(hidden_size=CLIP_HIDDEN, intermediate_size=128, num_hidden_layers=10, num_attention_heads=4,
                                            image_size=28, patch_size=7, hidden_act="quick_gelu", attn_implementation="eager")).double()
    assign(clip, "model.vision_tower.vision_tower.vision_model.", skip=("position_ids",))
    projector = torch.nn.Sequential(torch.nn.Linear(CLIP_HIDDEN, LLM_HIDDEN), torch.nn.GELU(), torch.nn.Linear(LLM_HIDDEN, LLM_HIDDEN)).double()
    assign(projector, "model.mm_projector.")
    vocab = json.load(open(TOKENIZER, encoding="utf-8"))["pieces"] + 7
    llm = LlamaForCausalLM(LlamaConfig(vocab_size=vocab, hidden_size=LLM_HIDDEN, intermediate_size=64, num_hidden_layers=2,
                                       num_attention_heads=4, num_key_value_heads=4, rms_norm_eps=1e-5, rope_theta=10000.0,
                                       max_position_embeddings=256, tie_word_embeddings=False, attn_implementation="eager")).double()
    llm = get_peft_model(llm, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["q_proj", "v_proj"]))
    with torch.no_grad():
        for name, p in llm.named_parameters():
            mapped = llama_reference.mapped(name)
            p.copy_(torch.tensor(synthetic(mapped, list(p.shape))).reshape(p.shape))
            p.requires_grad_(True)
    glamm_region_reference.EMBED, glamm_region_reference.REGION, glamm_region_reference.OUT = CLIP_HIDDEN, 32, LLM_HIDDEN
    region = glamm_region_reference.build()
    assign(region, "model.region_encoder.")
    encoder = ImageEncoderViT(img_size=64, patch_size=16, in_chans=3, embed_dim=32, depth=2, num_heads=2, mlp_ratio=4.0,
                              out_chans=SAM_DIM, qkv_bias=True, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), use_abs_pos=True,
                              use_rel_pos=True, rel_pos_zero_init=True, window_size=2, global_attn_indexes=(1,))
    prompt = PromptEncoder(embed_dim=SAM_DIM, image_embedding_size=(4, 4), input_image_size=(64, 64), mask_in_chans=8)
    decoder = MaskDecoder(num_multimask_outputs=3, transformer=TwoWayTransformer(depth=2, embedding_dim=SAM_DIM, mlp_dim=32, num_heads=2),
                          transformer_dim=SAM_DIM, iou_head_depth=3, iou_head_hidden_dim=16)
    sam = Sam(image_encoder=encoder, prompt_encoder=prompt, mask_decoder=decoder).double()
    assign(sam, "model.grounding_encoder.", skip=("pixel_mean", "pixel_std"))
    text_fcs = torch.nn.Sequential(torch.nn.Linear(LLM_HIDDEN, LLM_HIDDEN), torch.nn.ReLU(), torch.nn.Linear(LLM_HIDDEN, SAM_DIM)).double()
    assign(text_fcs, "model.text_hidden_fcs.0.")
    return clip, projector, llm, region, sam, text_fcs


def patch_float64():
    import torch
    llama_reference.patch_float64()
    clip_vision_reference.patch_float64()
    # SAM's prompt encoder casts coordinates and its position grid to float32 (as in sam_reference.py).
    to, ones = torch.Tensor.to, torch.ones
    torch.Tensor.to = lambda self, *a, **k: self if (a and a[0] is torch.float32 and self.dtype == torch.float64) else to(self, *a, **k)
    torch.ones = lambda *a, **k: ones(*a, **{**k, "dtype": torch.float64} if k.get("dtype") is torch.float32 else k)


def tokenized():
    stored = json.load(open(TOKENIZER, encoding="utf-8"))
    sp, encode, image_token, decode = glamm_tokenizer_reference.wrap(base64.b64decode(stored["model"]))
    conversation = SYSTEM + " USER: " + INSTRUCTION + " ASSISTANT: " + ANSWER + "</s>"
    ids = image_token(conversation)
    labels = list(ids)
    sep, sep2 = " ASSISTANT: ", "</s>"
    cur = 1
    labels[:1] = [-100]
    for rou in conversation.split(sep2):  # LLaVA's _process_conversation (an <image> is present)
        if not rou:
            break
        parts = rou.split(sep)
        parts[0] += sep
        round_len = len(image_token(rou))
        instruction_len = len(image_token(parts[0])) - 2
        labels[cur:cur + instruction_len] = [-100] * instruction_len
        cur += round_len
    labels[cur:] = [-100] * (len(labels) - cur)
    base = sp.get_piece_size()
    return conversation, ids, labels, base + glamm_tokenizer_reference.ADDED.index("[SEG]"), base + glamm_tokenizer_reference.ADDED.index("<bbox>")


def forward(parts, ids, labels, seg_id, bbox_id, global_image, sam_image, gt_masks):
    import torch
    import torch.nn.functional as F
    clip, projector, llm, region, sam, text_fcs = parts
    states = clip(pixel_values=global_image, output_hidden_states=True).hidden_states
    select = -2
    features = projector(states[select][:, 1:])                                       # [1, 16, H]
    levels = [s[:, 1:] for s in states[select::-3][::-1][-4:]]
    reg_feat = region(levels, [torch.tensor(BOXES)])[0]

    # prepare_inputs_labels_for_multimodal, mm_use_im_start_end (tune_mm_mlp_adapter off), one image.
    embed = llm.get_input_embeddings()
    cur_ids = torch.tensor(ids)
    cur_labels = torch.tensor(labels)
    start = int((cur_ids == -200).nonzero()[0])
    image = features[0]
    new_embeds = [embed(cur_ids[:start]), image, embed(cur_ids[start + 1:start + 2]), embed(cur_ids[start + 2:])]
    full_ids = torch.cat([cur_ids[:start], torch.full((image.shape[0],), start), cur_ids[start + 1:start + 2], cur_ids[start + 2:]])
    new_labels = torch.cat([cur_labels[:start], torch.full((image.shape[0],), -100), cur_labels[start + 1:start + 2], cur_labels[start + 2:]])
    embeds = torch.cat(new_embeds, dim=0)
    reg_mask = full_ids == bbox_id
    reg_embeds = torch.zeros_like(embeds)
    reg_embeds[reg_mask] = reg_feat[:int(reg_mask.sum())]
    embeds = embeds * (~reg_mask).to(embeds.dtype)[:, None] + reg_embeds
    out = llm(inputs_embeds=embeds[None], attention_mask=torch.ones(1, embeds.shape[0], dtype=torch.long), output_hidden_states=True)
    logits = out.logits
    ce = F.cross_entropy(logits[0, :-1], new_labels[1:], ignore_index=-100)

    seg_mask = torch.cat([torch.zeros(image.shape[0] - 1, dtype=torch.bool), cur_ids[1:] == seg_id, torch.zeros(1, dtype=torch.bool)])
    pred_embeddings = text_fcs(out.hidden_states[-1][0])[seg_mask]
    image_embeddings = sam.image_encoder(sam_image)
    sparse = pred_embeddings.unsqueeze(1)
    dense = sam.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(sparse.shape[0], -1, 4, 4)
    low_res, _ = sam.mask_decoder(image_embeddings=image_embeddings, image_pe=sam.prompt_encoder.get_dense_pe(),
                                  sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, multimask_output=False)
    pred = sam.postprocess_masks(low_res, input_size=(48, 64), original_size=(24, 32))[:, 0]

    n = gt_masks.shape[0]
    bce = F.binary_cross_entropy_with_logits(pred, gt_masks, reduction="none").flatten(1, 2).mean(1).sum() / (n + 1e-8) * n
    p = pred.sigmoid().flatten(1, 2)
    g = gt_masks.flatten(1, 2)
    dice = (1 - (2 * (p / 1000 * g).sum(-1) + 1e-6) / ((p / 1000).sum(-1) + (g / 1000).sum(-1) + 1e-6)).sum() / (n + 1e-8) * n
    bce, dice = 2.0 * bce / (n + 1e-8), 0.5 * dice / (n + 1e-8)
    return ce, bce, dice, ce + bce + dice, pred


def compute():
    import torch
    torch.set_default_dtype(torch.float64)
    patch_float64()
    parts = build()
    conversation, ids, labels, seg_id, bbox_id = tokenized()
    global_image = torch.tensor([2 * v - 1 for v in uniforms("global_image", 3 * 28 * 28)]).reshape(1, 3, 28, 28)
    sam_image = torch.tensor([2 * v - 1 for v in uniforms("sam_image", 3 * 64 * 64)]).reshape(1, 3, 64, 64)
    gt_masks = torch.tensor([1.0 if v > 0.5 else 0.0 for v in uniforms("gt_masks", 2 * 24 * 32)]).reshape(2, 24, 32)
    ce, bce, dice, total, pred = forward(parts, ids, labels, seg_id, bbox_id, global_image, sam_image, gt_masks)
    params = [p for module in parts for p in module.parameters()]
    for p in params:
        p.grad = None
    total.backward()
    with torch.no_grad():
        for p in params:
            if p.grad is not None:
                p -= 0.001 * p.grad
        _, _, _, after, _ = forward(parts, ids, labels, seg_id, bbox_id, global_image, sam_image, gt_masks)
    return {"conversation": conversation, "ids": ids, "labels": labels, "boxes": BOXES, "mask_logits": pred.detach().flatten().tolist(),
            "ce": ce.item(), "bce": bce.item(), "dice": dice.item(), "loss": total.item(), "loss_after_sgd": after.item()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = compute()
    if args.write:
        os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
        with open(FIXTURE, "w", encoding="utf-8", newline="\n") as f:
            json.dump(result, f)
        print("wrote", result["ce"], result["bce"], result["dice"], result["loss"], result["loss_after_sgd"])
        return
    with open(FIXTURE, encoding="utf-8") as f:
        stored = json.load(f)
    same = all(abs(stored[k] - result[k]) < 1e-12 for k in ("loss", "loss_after_sgd")) and stored["ids"] == result["ids"]
    print("fixture matches" if same else "fixture differs")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
