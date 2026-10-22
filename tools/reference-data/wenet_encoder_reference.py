"""wenet encoder reference: FunAudioLLM/CosyVoice's ConformerEncoder and TransformerEncoder (cosyvoice/transformer) with
ESPnet relative positions, as CosyVoice 1 configures its text encoder (Conformer, linear input, Swish, causal), its
language model (Transformer, linear_legacy input, ReLU, causal) and its flow encoder (Conformer, full attention).
Pass a checkout of FunAudioLLM/CosyVoice with --cosyvoice.

Builds tiny randomly initialized encoders (seeded), runs them in float64 and records in
tests/AiDotNet.Tests/TextToSpeech/ReferenceData/wenet_encoder_reference.json each one's output for one input, and the
first one's output after one SGD step (lr 0.1) on the mean square of its output. The reference computes its position
table in float32 (EspnetRelPositionalEncoding.extend_pe); it is computed in float64 here. Default: verify; --write:
regenerate.
"""
import argparse
import base64
import math
import sys
import types

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

import _fixture

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/wenet_encoder_reference.json"
LENGTH = 7
ENCODERS = [
    dict(name="text_encoder", kind="conformer", input_size=6, output_size=8, attention_heads=2, linear_units=12,
         num_blocks=2, input_layer="linear", static_chunk_size=1),
    dict(name="llm", kind="transformer", input_size=8, output_size=8, attention_heads=2, linear_units=12, num_blocks=2,
         input_layer="linear_legacy", static_chunk_size=1),
    dict(name="flow_encoder", kind="conformer", input_size=6, output_size=8, attention_heads=2, linear_units=12,
         num_blocks=2, input_layer="linear", static_chunk_size=0),
]
LEARNING_RATE = 0.1


def patch_float64_positions(embedding):
    def extend_pe(self, x):
        size = x.size(1)
        if self.pe is not None and self.pe.size(1) >= size * 2 - 1 and self.pe.dtype == torch.float64:
            return
        position = torch.arange(0, size, dtype=torch.float64).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float64) * -(math.log(10000.0) / self.d_model))
        positive = torch.zeros(size, self.d_model, dtype=torch.float64)
        negative = torch.zeros(size, self.d_model, dtype=torch.float64)
        positive[:, 0::2] = torch.sin(position * div_term)
        positive[:, 1::2] = torch.cos(position * div_term)
        negative[:, 0::2] = torch.sin(-1 * position * div_term)
        negative[:, 1::2] = torch.cos(-1 * position * div_term)
        self.pe = torch.cat([torch.flip(positive, [0]).unsqueeze(0), negative[1:].unsqueeze(0)], dim=1).to(x.device)
    embedding.EspnetRelPositionalEncoding.extend_pe = extend_pe


def build(root, states=None):
    sys.path.insert(0, root)
    # cosyvoice.utils.onnx (speech-token extraction through onnxruntime) is imported along the way but not used here.
    onnx_stub = types.ModuleType("cosyvoice.utils.onnx")
    onnx_stub.SpeechTokenExtractor = onnx_stub.online_feature = onnx_stub.onnx_path = None
    sys.modules.setdefault("cosyvoice.utils.onnx", onnx_stub)
    # cosyvoice.utils.class_utils with its model imports (llm, flow, hifigan, cli) removed: the encoder needs only its
    # activation, subsampling, position and attention tables, and the model modules need packages it does not use.
    with open(f"{root}/cosyvoice/utils/class_utils.py", encoding="utf-8") as handle:
        source = "\n".join(line for line in handle.read().splitlines()
                           if not line.startswith(("from cosyvoice.llm", "from cosyvoice.flow", "from cosyvoice.hifigan",
                                                   "from cosyvoice.cli")))
    class_utils = types.ModuleType("cosyvoice.utils.class_utils")
    sys.modules["cosyvoice.utils.class_utils"] = class_utils
    exec(compile(source.split("def get_model_type")[0], "class_utils.py", "exec"), class_utils.__dict__)
    import cosyvoice.transformer.embedding as embedding
    from cosyvoice.transformer.encoder import ConformerEncoder, TransformerEncoder
    patch_float64_positions(embedding)
    torch.manual_seed(2294)
    models = []
    for spec in ENCODERS:
        args = dict(input_size=spec["input_size"], output_size=spec["output_size"], attention_heads=spec["attention_heads"],
                    linear_units=spec["linear_units"], num_blocks=spec["num_blocks"], dropout_rate=0.1,
                    positional_dropout_rate=0.1, attention_dropout_rate=0.0, input_layer=spec["input_layer"],
                    pos_enc_layer_type="rel_pos_espnet", selfattention_layer_type="rel_selfattn",
                    static_chunk_size=spec["static_chunk_size"])
        if spec["kind"] == "conformer":
            model = ConformerEncoder(**args, normalize_before=True, use_cnn_module=False, macaron_style=False)
        else:
            model = TransformerEncoder(**args)
        model = model.eval()
        with torch.no_grad():
            # pos_bias_u / pos_bias_v start from xavier_uniform; layer norms from (1, 0): spread both.
            for name, p in model.named_parameters():
                if "norm" in name:
                    p.add_(0.1 * torch.randn_like(p))
        if states is not None:
            model.load_state_dict(states[spec["name"]], strict=True)
        models.append(model.double())
    return models


def encode(model, x):
    out, _ = model(x, torch.tensor([x.size(1)]))
    return out


def run(models, inputs):
    outputs = [encode(m, x) for m, x in zip(models, inputs)]
    loss = (outputs[0] ** 2).mean()
    parameters = list(models[0].parameters())
    grads = torch.autograd.grad(loss, parameters, allow_unused=True)
    with torch.no_grad():
        for p, g in zip(parameters, grads):
            if g is not None:
                p -= LEARNING_RATE * g
        after = encode(models[0], inputs[0])
    return outputs, after


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cosyvoice", required=True, help="a checkout of FunAudioLLM/CosyVoice")
    parser.add_argument("--write", action="store_true", help="regenerate the fixture")
    args = parser.parse_args()
    if args.write:
        models = build(args.cosyvoice)
        state = {}
        for spec, model in zip(ENCODERS, models):
            for k, v in model.state_dict().items():
                state[f"{spec['name']}.{k}"] = v.detach().float().clone().contiguous()
        torch.manual_seed(7)
        inputs = [torch.randn(1, LENGTH, spec["input_size"]) for spec in ENCODERS]
        # Reload the float32-stored weights so the run uses exactly what the fixture stores.
        states = {spec["name"]: {k[len(spec["name"]) + 1:]: v for k, v in state.items() if k.startswith(spec["name"] + ".")} for spec in ENCODERS}
        models = build(args.cosyvoice, states)
        outputs, after = run(models, [x.double() for x in inputs])
        data = {"encoders": ENCODERS, "length": LENGTH, "learning_rate": LEARNING_RATE,
                "safetensors_base64": base64.b64encode(save_safetensors(state)).decode("ascii"),
                "inputs": [[float(v) for v in x.double().reshape(-1).tolist()] for x in inputs],
                "outputs": [[float(v) for v in o.detach().reshape(-1).tolist()] for o in outputs],
                "after_sgd": [float(v) for v in after.detach().reshape(-1).tolist()]}
        _fixture.save(FIXTURE, data, __file__)
        print("wrote " + FIXTURE)
        return
    data = _fixture.load(FIXTURE)
    state = load_safetensors(base64.b64decode(data["safetensors_base64"]))
    states = {spec["name"]: {k[len(spec["name"]) + 1:]: v for k, v in state.items() if k.startswith(spec["name"] + ".")} for spec in ENCODERS}
    models = build(args.cosyvoice, states)
    inputs = [torch.tensor(x, dtype=torch.float64).reshape(1, LENGTH, spec["input_size"]) for x, spec in zip(data["inputs"], ENCODERS)]
    outputs, after = run(models, inputs)
    check = _fixture.Comparison("wenet encoder")
    for i, spec in enumerate(ENCODERS):
        check.floats(spec["name"], data["outputs"][i], outputs[i].detach().reshape(-1).tolist(), 1e-12, float)
    check.floats("after_sgd", data["after_sgd"], after.detach().reshape(-1).tolist(), 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
