"""CosyVoice 1 language model reference: FunAudioLLM/CosyVoice's TransformerLM (cosyvoice/llm/llm.py) with its causal
Conformer text encoder and causal Transformer LM. Pass checkouts with --cosyvoice and --matcha (cosyvoice imports
Matcha-TTS on the way).

Builds a tiny randomly initialized model (seeded), runs it in float64 and records in
tests/AiDotNet.Tests/TextToSpeech/ReferenceData/cosyvoice_lm_reference.json the LM's input embeddings and log-probabilities
for a text, x-vector and speech tokens, the training loss TransformerLM.forward computes, and that loss after one SGD
step (lr 0.1). The ESPnet position table is float64 here (the reference builds it in float32). Default: verify;
--write: regenerate.
"""
import argparse
import base64

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

import _fixture
from cosyvoice_flow_reference import stub_imports

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/cosyvoice_lm_reference.json"
CONFIG = dict(text_encoder_input_size=6, llm_input_size=8, llm_output_size=8, text_token_size=20, speech_token_size=12,
              spk_embed_dim=5)
TEXT_ENCODER = dict(output_size=8, attention_heads=2, linear_units=12, num_blocks=2)
LLM = dict(output_size=8, attention_heads=2, linear_units=12, num_blocks=2)
TEXT = [3, 17, 8, 1, 11]
SPEECH = [4, 9, 0, 7, 2, 11]
LEARNING_RATE = 0.1


def build(state=None):
    from cosyvoice.llm.llm import TransformerLM
    from cosyvoice.transformer.encoder import ConformerEncoder, TransformerEncoder
    torch.manual_seed(2294)
    text_encoder = ConformerEncoder(input_size=CONFIG["text_encoder_input_size"], dropout_rate=0.1, positional_dropout_rate=0.1,
                                    attention_dropout_rate=0.0, normalize_before=True, input_layer="linear",
                                    pos_enc_layer_type="rel_pos_espnet", selfattention_layer_type="rel_selfattn",
                                    use_cnn_module=False, macaron_style=False, use_dynamic_chunk=False,
                                    use_dynamic_left_chunk=False, static_chunk_size=1, **TEXT_ENCODER)
    llm = TransformerEncoder(input_size=CONFIG["llm_input_size"], dropout_rate=0.1, positional_dropout_rate=0.1,
                             attention_dropout_rate=0.0, input_layer="linear_legacy", pos_enc_layer_type="rel_pos_espnet",
                             selfattention_layer_type="rel_selfattn", static_chunk_size=1, **LLM)
    model = TransformerLM(text_encoder=text_encoder, llm=llm, sampling=None, length_normalized_loss=True, lsm_weight=0,
                          **CONFIG).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.add_(0.1 * torch.randn_like(p))
    if state is not None:
        model.load_state_dict(state, strict=True)
    return model


def run(model, xvector):
    batch = {"text_token": torch.tensor([TEXT]), "text_token_len": torch.tensor([len(TEXT)]),
             "speech_token": torch.tensor([SPEECH]), "speech_token_len": torch.tensor([len(SPEECH)]), "embedding": xvector}
    loss = model(batch, torch.device("cpu"))["loss"]
    with torch.no_grad():
        text, text_len = model.encode(model.text_embedding(batch["text_token"]), batch["text_token_len"])
        embedding = model.spk_embed_affine_layer(torch.nn.functional.normalize(xvector, dim=1)).unsqueeze(1)
        sos = model.llm_embedding.weight[model.sos].reshape(1, 1, -1)
        task = model.llm_embedding.weight[model.task_id].reshape(1, 1, -1)
        lm_input = torch.concat([sos, embedding, text, task, model.speech_embedding(batch["speech_token"])], dim=1)
        out, _ = model.llm(lm_input, torch.tensor([lm_input.size(1)]))
        logp = model.llm_decoder(out).log_softmax(dim=-1)
    parameters = list(model.parameters())
    grads = torch.autograd.grad(loss, parameters, allow_unused=True)
    with torch.no_grad():
        for p, g in zip(parameters, grads):
            if g is not None:
                p -= LEARNING_RATE * g
        after = model(batch, torch.device("cpu"))["loss"]
    return lm_input, logp, loss, after


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cosyvoice", required=True)
    parser.add_argument("--matcha", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    stub_imports(args.cosyvoice, args.matcha)
    if args.write:
        model = build()
        state = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}
        torch.manual_seed(7)
        xvector = torch.randn(1, CONFIG["spk_embed_dim"]).double()
        lm_input, logp, loss, after = run(model.double(), xvector)
        data = {"config": CONFIG, "text_encoder": TEXT_ENCODER, "llm": LLM, "text": TEXT, "speech": SPEECH,
                "learning_rate": LEARNING_RATE, "safetensors_base64": base64.b64encode(save_safetensors(state)).decode("ascii"),
                "xvector": [float(v) for v in xvector.reshape(-1).tolist()],
                "lm_input": [float(v) for v in lm_input.reshape(-1).tolist()],
                "logp": [float(v) for v in logp.reshape(-1).tolist()],
                "loss": float(loss.detach()), "loss_after_sgd": float(after.detach())}
        _fixture.save(FIXTURE, data, __file__)
        print("wrote " + FIXTURE)
        return
    data = _fixture.load(FIXTURE)
    model = build(load_safetensors(base64.b64decode(data["safetensors_base64"]))).double()
    xvector = torch.tensor(data["xvector"], dtype=torch.float64).reshape(1, CONFIG["spk_embed_dim"])
    lm_input, logp, loss, after = run(model, xvector)
    check = _fixture.Comparison("CosyVoice LM")
    check.floats("lm_input", data["lm_input"], lm_input.reshape(-1).tolist(), 1e-12, float)
    check.floats("logp", data["logp"], logp.reshape(-1).tolist(), 1e-12, float)
    check.floats("loss", [data["loss"]], [float(loss.detach())], 1e-12, float)
    check.floats("loss_after_sgd", [data["loss_after_sgd"]], [float(after.detach())], 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
