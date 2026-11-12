"""T5 v1.0 reference (Pheme's text-to-semantic model): Hugging Face transformers' T5ForConditionalGeneration.

Builds a tiny randomly initialized T5 (seeded; ReLU feed-forward, tied embeddings, 8 relative buckets with maximum
distance 16 so the logarithmic buckets are exercised on 12-token inputs), runs it in float64, and records its weights
(safetensors), an input and label sequence, the encoder states, the decoder logits and the loss in
tests/AiDotNet.Tests/TextToSpeech/ReferenceData/t5_reference.json. Weights are float32; the run is float64, with the
layer norms' variance in float64 too (see _layer_norm_in_model_precision). Default: verify; --write: regenerate.
"""
import base64

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors
from transformers import T5Config, T5ForConditionalGeneration

import _fixture
import transformers.models.t5.modeling_t5 as modeling_t5


def _layer_norm_in_model_precision(self, hidden_states):
    """T5LayerNorm without its float32 cast of the variance. Hugging Face computes the RMS variance in float32 even in a
    float64 model (to protect fp16 training); in float64 that cast is the only rounding left between the reference and
    a float64 port, so the fixture computes it in the model's precision instead. Weights are unchanged."""
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    return self.weight * (hidden_states * torch.rsqrt(variance + self.variance_epsilon))


modeling_t5.T5LayerNorm.forward = _layer_norm_in_model_precision

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/t5_reference.json"
CONFIG = dict(vocab_size=20, d_model=8, d_ff=12, d_kv=4, num_heads=2, num_layers=2, num_decoder_layers=2,
              relative_attention_num_buckets=8, relative_attention_max_distance=16, dropout_rate=0.0,
              feed_forward_proj="relu", layer_norm_epsilon=1e-6, decoder_start_token_id=0, eos_token_id=2,
              pad_token_id=0, tie_word_embeddings=True)
INPUT_IDS = [1, 3, 7, 11, 5, 9, 13, 4, 17, 6, 8, 2]
LABELS = [1, 3, 15, 12, 19, 10, 14, 16, 18, 2]


def model_from(state=None):
    torch.manual_seed(2294)
    # Initialized and stored in float32 (the loaders read float32), then run in float64.
    model = T5ForConditionalGeneration(T5Config(**CONFIG)).eval()
    if state is not None:
        model.load_state_dict(state, strict=False)
    return model


def run(model):
    input_ids = torch.tensor([INPUT_IDS])
    labels = torch.tensor([LABELS])
    with torch.no_grad():
        encoder = model.encoder(input_ids=input_ids).last_hidden_state
        out = model(input_ids=input_ids, labels=labels)
    return encoder, out.logits, out.loss


def main():
    args = _fixture.parse_args(__doc__)
    if args.write:
        model = model_from()
        # encoder/decoder.embed_tokens and lm_head alias shared.weight (tied); store the one table.
        aliases = {"encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"}
        state = {k: v.clone().contiguous() for k, v in model.state_dict().items() if k not in aliases}
        encoder, logits, loss = run(model.double())
        data = {"config": CONFIG, "input_ids": INPUT_IDS, "labels": LABELS,
                "safetensors_base64": base64.b64encode(save_safetensors(state)).decode("ascii"),
                "encoder": [float(v) for v in encoder.reshape(-1).tolist()],
                "logits": [float(v) for v in logits.reshape(-1).tolist()], "loss": float(loss)}
        _fixture.save(FIXTURE, data, __file__)
        print("wrote " + FIXTURE)
        return
    data = _fixture.load(FIXTURE)
    model = model_from(load_safetensors(base64.b64decode(data["safetensors_base64"])))
    encoder, logits, loss = run(model.double())
    check = _fixture.Comparison("T5")
    check.floats("encoder", data["encoder"], encoder.reshape(-1).tolist(), 1e-12, float)
    check.floats("logits", data["logits"], logits.reshape(-1).tolist(), 1e-12, float)
    check.floats("loss", [data["loss"]], [float(loss)], 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
