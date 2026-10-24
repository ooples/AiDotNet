"""SPEAR-TTS's reading model (S1) reference: Hugging Face transformers' T5ForConditionalGeneration (T5 v1.0, as the
paper's T5-Large) with the encoder reading its own embedding table, which SPEAR-TTS's finetuning updates while the
decoder (and its tied table) stays frozen (Kharitonov et al. 2023, Sec. 7.2).

Builds a tiny randomly initialized T5 (seeded; ReLU feed-forward, tied decoder embedding and output, a separate encoder
table, 8 relative buckets with maximum distance 16), runs it in float64 and records in
tests/AiDotNet.Tests/TextToSpeech/ReferenceData/speartts_reference.json:
  - the label-smoothed loss of a text-to-semantic pair, as torch.nn.functional.cross_entropy(label_smoothing=0.1)
    computes it (Sec. 7.2: label smoothing 0.1), next to the unsmoothed loss;
  - the smoothed loss after one SGD step (lr 0.1) on the reading stage's parameters only: the encoder's table and its
    lowest block (the paper's "encoder embeddings and the lower layers", with one lower layer here);
  - the beam search's result (beam 3, Sec. 7.4 uses 10) from the decoder's logits, restricted to semantic ids and the
    end token (the end not as the first token): summed log-probabilities without a length penalty, as
    lucidrains/spear-tts-pytorch ranks its beams, with a beam that has emitted the end token kept as it is.
Weights are float32; the run is float64, with the layer norms' variance in float64 too. Default: verify; --write:
regenerate.
"""
import base64

import torch
import torch.nn.functional as F
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors
from transformers import T5Config, T5ForConditionalGeneration

import _fixture
import transformers.models.t5.modeling_t5 as modeling_t5


def _layer_norm_in_model_precision(self, hidden_states):
    """T5LayerNorm without its float32 cast of the variance (see t5_reference.py)."""
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    return self.weight * (hidden_states * torch.rsqrt(variance + self.variance_epsilon))


modeling_t5.T5LayerNorm.forward = _layer_norm_in_model_precision

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/speartts_reference.json"
CONFIG = dict(vocab_size=24, d_model=8, d_ff=12, d_kv=4, num_heads=2, num_layers=2, num_decoder_layers=2,
              relative_attention_num_buckets=8, relative_attention_max_distance=16, dropout_rate=0.0,
              feed_forward_proj="relu", layer_norm_epsilon=1e-6, decoder_start_token_id=0, eos_token_id=1,
              pad_token_id=0, tie_word_embeddings=True)
# Ids 2..11 play the phonemes and 12..23 the semantic tokens; 0 is <pad> (the decoder's start) and 1 is </s>.
SEMANTIC_OFFSET = 12
INPUT_IDS = [3, 7, 11, 5, 9, 4, 6, 10, 2, 1]
LABELS = [15, 18, 12, 20, 22, 13, 1]
LABEL_SMOOTHING = 0.1
LEARNING_RATE = 0.1
BEAM_SIZE = 3
MAX_NEW_TOKENS = 6


def model_from(state=None, seed=2294):
    torch.manual_seed(seed)
    # Initialized and stored in float32 (the loaders read float32), then run in float64.
    model = T5ForConditionalGeneration(T5Config(**CONFIG)).eval()
    # The encoder's own table, drawn as T5 draws `shared` (N(0, 1) at initializer factor 1).
    table = torch.nn.Embedding(CONFIG["vocab_size"], CONFIG["d_model"])
    torch.nn.init.normal_(table.weight, mean=0.0, std=1.0)
    model.encoder.set_input_embeddings(table)
    if state is not None:
        model.load_state_dict(state, strict=False)
    return model


def smoothed_loss(model, input_ids, labels):
    logits = model(input_ids=input_ids, labels=labels).logits
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), label_smoothing=LABEL_SMOOTHING)


def reading_parameters(model):
    """The reading stage's trainable parameters: the encoder's table and its lowest block."""
    return [model.encoder.embed_tokens.weight] + list(model.encoder.block[0].parameters())


def beam_search(model, input_ids, beam_size):
    with torch.no_grad():
        memory = model.encoder(input_ids=input_ids).last_hidden_state
        start = [CONFIG["decoder_start_token_id"]]
        beams = [(start, 0.0, False)]
        eos = CONFIG["eos_token_id"]
        for step in range(MAX_NEW_TOKENS):
            if all(done for _, _, done in beams):
                break
            candidates = []
            for sequence, score, done in beams:
                if done:
                    candidates.append((sequence, score, done))
                    continue
                logits = model(encoder_outputs=(memory,), decoder_input_ids=torch.tensor([sequence])).logits[0, -1]
                allowed = torch.tensor([v >= SEMANTIC_OFFSET or (v == eos and step > 0) for v in range(logits.shape[0])])
                masked = logits.masked_fill(~allowed, float("-inf"))
                log_probs = torch.log_softmax(masked, dim=-1)
                order = sorted([v for v in range(logits.shape[0]) if allowed[v]], key=lambda v: -float(log_probs[v]))
                for v in order[:beam_size]:
                    candidates.append((sequence + [v], score + float(log_probs[v]), v == eos))
            ranked = sorted(enumerate(candidates), key=lambda item: (-item[1][1], item[0]))
            beams = [candidate for _, candidate in ranked[:beam_size]]
        best = beams[0][0]
        end = len(best) - 1 if len(best) > 1 and best[-1] == eos else len(best)
        return best[1:end]


def run(model):
    input_ids = torch.tensor([INPUT_IDS])
    labels = torch.tensor([LABELS])
    with torch.no_grad():
        plain = model(input_ids=input_ids, labels=labels).loss
    loss = smoothed_loss(model, input_ids, labels)
    parameters = reading_parameters(model)
    grads = torch.autograd.grad(loss, parameters)
    beam = beam_search(model, input_ids, BEAM_SIZE)
    greedy = beam_search(model, input_ids, 1)
    with torch.no_grad():
        for p, g in zip(parameters, grads):
            p -= LEARNING_RATE * g
        after = smoothed_loss(model, input_ids, labels)
    return float(plain), float(loss.detach()), float(after), beam, greedy


def discriminating_seed():
    """The first seed from 2294 whose beam search and greedy search (beam 1) find different sequences, the beam's of
    at least two distinct tokens: an untrained tiny model often repeats one token, and then the fixture could not tell
    a beam search from a greedy one."""
    for seed in range(2294, 2294 + 500):
        _, _, _, beam, greedy = run(model_from(seed=seed).double())
        if beam != greedy and len(set(beam)) >= 2:
            return seed
    raise RuntimeError("no seed in range separates beam search from greedy search")


def main():
    args = _fixture.parse_args(__doc__)
    if args.write:
        seed = discriminating_seed()
        model = model_from(seed=seed)
        # decoder.embed_tokens and lm_head alias shared.weight (tied); the encoder's table is its own.
        aliases = {"decoder.embed_tokens.weight", "lm_head.weight"}
        state = {k: v.clone().contiguous() for k, v in model.state_dict().items() if k not in aliases}
        assert not torch.equal(state["encoder.embed_tokens.weight"], state["shared.weight"])
        plain, loss, after, beam, greedy = run(model.double())
        data = {"config": CONFIG, "seed": seed, "semantic_offset": SEMANTIC_OFFSET, "input_ids": INPUT_IDS, "labels": LABELS,
                "label_smoothing": LABEL_SMOOTHING, "learning_rate": LEARNING_RATE, "beam_size": BEAM_SIZE,
                "max_new_tokens": MAX_NEW_TOKENS,
                "safetensors_base64": base64.b64encode(save_safetensors(state)).decode("ascii"),
                "loss": plain, "smoothed_loss": loss, "smoothed_loss_after_sgd": after, "beam": beam, "greedy": greedy}
        _fixture.save(FIXTURE, data, __file__)
        print("wrote " + FIXTURE)
        return
    data = _fixture.load(FIXTURE)
    model = model_from(load_safetensors(base64.b64decode(data["safetensors_base64"])))
    plain, loss, after, beam, greedy = run(model.double())
    check = _fixture.Comparison("SPEAR-TTS reading")
    check.exact("greedy", data["greedy"], greedy)
    check.floats("loss", [data["loss"]], [plain], 1e-12, float)
    check.floats("smoothed_loss", [data["smoothed_loss"]], [loss], 1e-12, float)
    check.floats("smoothed_loss_after_sgd", [data["smoothed_loss_after_sgd"]], [after], 1e-12, float)
    check.exact("beam", data["beam"], beam)
    check.report(exact=False)


if __name__ == "__main__":
    main()
