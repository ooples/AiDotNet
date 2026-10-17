"""wav2vec 2.0 pre-training reference: transformers' Wav2Vec2ForPreTraining.

The BASE-style small network of wav2vec2_reference.py with a Gumbel quantizer of 2 codebooks of 8 entries
(codevector width 8), projections to 6, and 4 distractors, in float64 and eval mode (the quantizer takes the argmax
entry; training's Gumbel noise is random). Two 400-sample waveforms, a fixed time mask and fixed distractor indices
(transformers' _compute_mask_indices / _sample_negative_indices with a seeded NumPy generator) give the contrastive
loss, the diversity loss and their sum (weight 0.1); then one plain SGD step (lr 0.1) on every parameter and the loss
after it. fairseq's feature-encoder activation penalty is not part of transformers' loss and is not recorded.
transformers computes the cosine logits and the cross-entropy in float32 (.float()), so the fixture recomputes the
same expressions (Wav2Vec2ForPreTraining.forward's loss block) in float64 from the model's outputs.
Default: verify; --write: regenerate tests/AiDotNet.Tests/Audio/ReferenceData/wav2vec2_pretraining_reference.json.
"""
import argparse
import json
import os
import sys

FIXTURE = "tests/AiDotNet.Tests/Audio/ReferenceData/wav2vec2_pretraining_reference.json"


def compute():
    import numpy as np
    import torch
    from transformers import Wav2Vec2Config, Wav2Vec2ForPreTraining
    from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices
    torch.set_default_dtype(torch.float64)
    config = Wav2Vec2Config(conv_dim=(8, 8, 8), conv_kernel=(10, 3, 2), conv_stride=(5, 2, 2), conv_bias=False,
                            feat_extract_norm="group", do_stable_layer_norm=False, hidden_size=16, num_hidden_layers=2,
                            num_attention_heads=2, intermediate_size=32, num_conv_pos_embeddings=8,
                            num_conv_pos_embedding_groups=4, hidden_act="gelu", feat_extract_activation="gelu",
                            layer_norm_eps=1e-5, num_codevector_groups=2, num_codevectors_per_group=8, codevector_dim=8,
                            proj_codevector_dim=6, num_negatives=4, contrastive_logits_temperature=0.1,
                            diversity_loss_weight=0.1)
    torch.manual_seed(13)
    model = Wav2Vec2ForPreTraining(config).double().eval()
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(p.float())
    g = torch.Generator().manual_seed(14)
    audio = torch.randn(2, 400, generator=g)
    frames = int(model.wav2vec2._get_feat_extract_output_lengths(400))
    np.random.seed(15)
    mask = _compute_mask_indices((2, frames), mask_prob=0.3, mask_length=3)
    negatives = _sample_negative_indices((2, frames), config.num_negatives, mask_time_indices=mask)
    state = {k: v.detach().flatten().tolist() for k, v in model.state_dict().items()}
    mask_t = torch.tensor(mask, dtype=torch.long)
    neg_t = torch.tensor(negatives, dtype=torch.long)
    def losses(out):
        quantized, predicted = out.projected_quantized_states, out.projected_states
        batch, length, width = quantized.shape
        negative = quantized.view(-1, width)[neg_t.view(-1)].view(batch, length, -1, width).permute(2, 0, 1, 3)
        targets = torch.cat([quantized[None, :], negative], dim=0)
        logits = torch.cosine_similarity(predicted, targets, dim=-1) / config.contrastive_logits_temperature
        neg_is_pos = (quantized == negative).all(-1)
        if neg_is_pos.any():
            logits = logits.clone()
            logits[1:][neg_is_pos] = float("-inf")
        logits = logits.transpose(0, 2).reshape(-1, logits.size(0))
        target = ((1 - mask_t) * -100).transpose(0, 1).flatten()
        contrastive = torch.nn.functional.cross_entropy(logits, target, reduction="sum")
        entries = config.num_codevectors_per_group * config.num_codevector_groups
        diversity = ((entries - out.codevector_perplexity) / entries) * mask_t.sum()
        return contrastive, diversity, contrastive + config.diversity_loss_weight * diversity

    out = model(audio, mask_time_indices=mask_t, sampled_negative_indices=neg_t)
    contrastive, diversity, loss = losses(out)
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p -= 0.1 * p.grad
        after = losses(model(audio, mask_time_indices=mask_t, sampled_negative_indices=neg_t))[2]
    return {"state": state, "audio": audio.flatten().tolist(), "frames": frames, "mask": mask.astype(int).tolist(),
            "negatives": negatives.tolist(), "loss": loss.item(), "contrastive": contrastive.item(),
            "diversity": diversity.item(), "perplexity": out.codevector_perplexity.item(), "loss_after_sgd": after.item()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = compute()
    if args.write:
        os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
        with open(FIXTURE, "w", encoding="utf-8", newline="\n") as f:
            json.dump(result, f)
        print("wrote", result["frames"], round(result["loss"], 9), round(result["contrastive"], 9), round(result["diversity"], 9),
              round(result["loss_after_sgd"], 9))
        return
    with open(FIXTURE, encoding="utf-8") as f:
        stored = json.load(f)
    same = abs(stored["loss"] - result["loss"]) < 1e-12 and abs(stored["loss_after_sgd"] - result["loss_after_sgd"]) < 1e-12
    print("fixture matches" if same else "fixture differs")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
