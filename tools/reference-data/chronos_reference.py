"""Chronos-T5 reference: the official amazon-science/chronos-forecasting code (src/chronos/chronos.py) with Hugging Face
transformers' T5ForConditionalGeneration, as the official training (scripts/training/train.py) builds it.

- The MeanScaleUniformBins tokenizer of a small configuration (64 tokens on [-15, 15], context length 12, prediction
  length 4): the bin centers; a context of 16 values with missing ones (truncated to its last 12) and the 4 values
  after it, with a missing one: token ids, masks and the scale; and the value of every token. Also the centers of the
  released 4096-token tokenizer.
- A tiny T5 (64 tokens, d_model 16, d_ff 32, d_kv 8, 2 heads, 2 layers each, 8 relative buckets with maximum distance
  16, no dropout, tied embeddings, decoder start = pad = 0, EOS = 1) initialized as the official training does
  (initializer_factor 0.05), run in float64 in eval mode with the layer norms' variance in float64 (Hugging Face
  computes it in float32 even in a float64 model). The fixture records its state dict; the loss of a batch of two
  windows of different context lengths (left-padded, as the official InstanceSplitter pads, with the padding and the
  missing values masked and missing labels ignored), the loss after one plain SGD step (lr 0.1), and the tokens that
  top-k = 1 sampling generates (min_new_tokens = max_new_tokens = 4, as the pipeline asks) from the first window, with
  a second model of the same sizes at initializer factor 1, whose logits are spread enough for the path to vary.
Pass a chronos-forecasting checkout with --chronos. Default: verify; --write: regenerate
tests/AiDotNet.Tests/UnitTests/TimeSeries/ReferenceData/chronos_reference.json.
"""
import argparse
import json
import math
import os
import sys

FIXTURE = "tests/AiDotNet.Tests/UnitTests/TimeSeries/ReferenceData/chronos_reference.json"
NAN = float("nan")
SERIES = [3.0, NAN, 5.5, 4.0, 6.25, 7.0, NAN, 5.0, 4.5, 8.0, 9.5, 7.25, 6.0, 5.75, NAN, 10.0, 11.0, NAN, 9.0, 12.5]
SHORT = [2.0, 2.5, NAN, 3.0, 4.0, 3.5]
T5 = dict(vocab_size=64, d_model=16, d_ff=32, d_kv=8, num_heads=2, num_layers=2, num_decoder_layers=2,
          relative_attention_num_buckets=8, relative_attention_max_distance=16, dropout_rate=0.0,
          feed_forward_proj="relu", layer_norm_epsilon=1e-6, decoder_start_token_id=0, eos_token_id=1, pad_token_id=0,
          tie_word_embeddings=True, initializer_factor=0.05)


def compute(chronos_src):
    sys.path.insert(0, chronos_src)
    import torch
    import transformers.models.t5.modeling_t5 as modeling_t5
    from transformers import T5Config, T5ForConditionalGeneration
    from chronos.chronos import ChronosConfig, MeanScaleUniformBins

    def layer_norm(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        return self.weight * (hidden_states * torch.rsqrt(variance + self.variance_epsilon))

    modeling_t5.T5LayerNorm.forward = layer_norm

    def config(n_tokens, context, prediction):
        return ChronosConfig(tokenizer_class="MeanScaleUniformBins", tokenizer_kwargs=dict(low_limit=-15.0, high_limit=15.0),
                             context_length=context, prediction_length=prediction, n_tokens=n_tokens, n_special_tokens=2,
                             pad_token_id=0, eos_token_id=1, use_eos_token=True, model_type="seq2seq", num_samples=20,
                             temperature=1.0, top_k=50, top_p=1.0)

    released = MeanScaleUniformBins(-15.0, 15.0, config(4096, 512, 64))
    tok = MeanScaleUniformBins(-15.0, 15.0, config(64, 12, 4))

    context = torch.tensor([SERIES[:16]], dtype=torch.float64)
    future = torch.tensor([SERIES[16:20]], dtype=torch.float64)
    ids, mask, scale = tok.context_input_transform(context)
    label_ids, label_mask = tok.label_input_transform(future, scale)
    values = tok.output_transform(torch.arange(64).reshape(1, 1, 64), scale)

    # The second window: a 6-value context, left-padded to 12 with NaN, and the next 4 values (from the long series).
    short = torch.tensor([[NAN] * 6 + SHORT], dtype=torch.float64)
    short_future = torch.tensor([[4.5, NAN, 5.0, 5.5]], dtype=torch.float64)
    ids2, mask2, scale2 = tok.context_input_transform(short)
    labels2, label_mask2 = tok.label_input_transform(short_future, scale2)

    input_ids = torch.cat([ids, ids2])
    attention_mask = torch.cat([mask, mask2])
    labels = torch.cat([label_ids, labels2]).clone()
    labels[torch.cat([label_mask, label_mask2]) == 0] = -100

    torch.manual_seed(2024)
    model = T5ForConditionalGeneration(T5Config(**T5)).double().eval()
    state = {k: v.detach().flatten().tolist() for k, v in model.state_dict().items()
             if k not in ("encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight")}
    loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            p -= 0.1 * p.grad
        after = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss

    # Generation from a model at initializer factor 1, whose logits are spread enough for the path to be informative.
    torch.manual_seed(8)
    generator = T5ForConditionalGeneration(T5Config(**{**T5, "initializer_factor": 1.0})).double().eval()
    generator_state = {k: v.detach().flatten().tolist() for k, v in generator.state_dict().items()
                       if k not in ("encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight")}
    with torch.no_grad():
        generated = generator.generate(input_ids=ids, attention_mask=mask, min_new_tokens=4, max_new_tokens=4,
                                       do_sample=True, top_k=1, eos_token_id=1, pad_token_id=0)[0, 1:].tolist()

    return {
        "released_centers": released.centers.tolist(),
        "centers": tok.centers.tolist(),
        "boundaries": tok.boundaries[1:-1].tolist(),
        "series": [None if math.isnan(v) else v for v in SERIES],
        "short": [None if math.isnan(v) else v for v in SHORT],
        "short_future": [4.5, None, 5.0, 5.5],
        "context_ids": ids[0].tolist(), "context_mask": mask[0].tolist(), "scale": scale.item(),
        "label_ids": label_ids[0].tolist(), "label_mask": label_mask[0].tolist(),
        "short_ids": ids2[0].tolist(), "short_mask": mask2[0].tolist(), "short_scale": scale2.item(),
        "short_label_ids": labels2[0].tolist(), "short_label_mask": label_mask2[0].tolist(),
        "values": values[0, 0].tolist(),
        "t5": {k: v for k, v in T5.items()},
        "state": state,
        "loss": loss.item(), "loss_after_sgd": after.item(),
        "generator_state": generator_state,
        "generated": generated,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--chronos", default=os.path.expanduser("~/source/repos/ref/chronos-forecasting/src"))
    args = parser.parse_args()
    result = compute(args.chronos)
    if args.write:
        os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
        with open(FIXTURE, "w", encoding="utf-8", newline="\n") as f:
            json.dump(result, f)
        print(f"wrote loss {result['loss']:.12f}, after SGD {result['loss_after_sgd']:.12f}, generated {result['generated']}")
        return
    with open(FIXTURE, encoding="utf-8") as f:
        stored = json.load(f)
    same = (abs(stored["loss"] - result["loss"]) < 1e-12 and abs(stored["loss_after_sgd"] - result["loss_after_sgd"]) < 1e-12
            and stored["generated"] == result["generated"] and stored["context_ids"] == result["context_ids"])
    print("fixture matches" if same else "fixture differs")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
