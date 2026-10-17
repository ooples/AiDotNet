"""VALL-E reference: lifeiteng/vall-e's VALLE (valle/models/valle.py), the reproduction the paper's gaps follow.

Builds a tiny randomly initialized VALLE with the reference's code (pass a checkout of lifeiteng/vall-e with --valle;
it is not vendored here), stores its weights in float32, runs it in float64 and records:
- the AR logits for a phoneme sequence and first-codebook codes (teacher-forced, train_stage 1);
- the NAR logits and loss of one training step (train_stage 2, prefix_mode 2: a random prompt segment of the same
  utterance), with the stage and segment the model drew;
- greedy inference (top_k 1) from an acoustic prompt with an enrolled transcript: the generated codes of all
  codebooks.
in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/valle_reference.json. Default: verify; --write: regenerate.

Only the model files are loaded (the package __init__ imports the data pipeline: lhotse, icefall); make_pad_mask is
icefall's one-liner. SinePositionalEmbedding builds its table in float32; the float64 run rebuilds it in float64,
which is what the C# port computes (the released models' float32 table differs in the last float32 digit).
"""
import argparse
import base64
import importlib.util
import math
import random
import sys
import types

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

import _fixture

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/valle_reference.json"
CONFIG = dict(d_model=16, nhead=2, num_layers=2, norm_first=True, add_prenet=False, prefix_mode=2,
              share_embedding=True, nar_scale_factor=1.0, prepend_bos=False, num_quantizers=3)
TEXT = [1, 17, 45, 62, 11, 33, 2]                 # <bos> … <eos>
ENROLLED = 4                                      # <bos> + two prompt phonemes + "_" … as the collater counts it
FRAMES = 12
PROMPT_FRAMES = 5


def load(valle_root):
    def module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    for package in ["valle", "valle.models", "valle.modules", "valle.data", "icefall"]:
        pkg = types.ModuleType(package)
        pkg.__path__ = []
        sys.modules[package] = pkg
    icefall_utils = types.ModuleType("icefall.utils")
    icefall_utils.make_pad_mask = lambda lengths, max_len=0: (
        torch.arange(max(max_len, int(lengths.max())), device=lengths.device).expand(len(lengths), -1)
        >= lengths.unsqueeze(-1))
    sys.modules["icefall.utils"] = icefall_utils
    strategies = types.ModuleType("valle.data.input_strategies")
    strategies.PromptedFeatures = type("PromptedFeatures", (), {})
    sys.modules["valle.data.input_strategies"] = strategies
    visualizer = types.ModuleType("valle.models.visualizer")
    visualizer.visualize = lambda *a, **k: None
    sys.modules["valle.models.visualizer"] = visualizer
    spec = importlib.util.spec_from_file_location("valle.utils", f"{valle_root}/valle/utils/__init__.py",
                                                  submodule_search_locations=[f"{valle_root}/valle/utils"])
    utils = importlib.util.module_from_spec(spec)
    sys.modules["valle.utils"] = utils
    spec.loader.exec_module(utils)
    for name in ["scaling", "activation", "embedding", "transformer"]:
        module(f"valle.modules.{name}", f"{valle_root}/valle/modules/{name}.py")
    module("valle.models.macros", f"{valle_root}/valle/models/macros.py")
    valle = module("valle.models.valle", f"{valle_root}/valle/models/valle.py")
    embedding = sys.modules["valle.modules.embedding"]

    def extend_pe_float64(self, x):
        length = x.size(1)
        if self.pe is not None and self.pe.size(1) >= length and self.pe.dtype == torch.float64:
            return
        position = torch.arange(0, length, dtype=torch.float64).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.dim_model, 2, dtype=torch.float64) * -(math.log(10000.0) / self.dim_model))
        pe = torch.zeros(length, self.dim_model, dtype=torch.float64)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    embedding.SinePositionalEmbedding.extend_pe = extend_pe_float64
    return valle


def build(valle, state=None):
    torch.manual_seed(2294)
    model = valle.VALLE(**CONFIG)
    if state is None:
        with torch.no_grad():
            model.ar_text_position.alpha.fill_(0.7)
            model.ar_audio_position.alpha.fill_(1.3)
    else:
        model.load_state_dict(state, strict=True)
    return model.eval()


def inputs():
    g = torch.Generator().manual_seed(7)
    codes = torch.randint(0, 1024, (1, FRAMES, CONFIG["num_quantizers"]), generator=g)
    prompt = torch.randint(0, 1024, (1, PROMPT_FRAMES, CONFIG["num_quantizers"]), generator=g)
    return torch.tensor([TEXT]), codes, prompt


def run(valle, model):
    x, codes, prompt = inputs()
    x_lens, y_lens = torch.tensor([len(TEXT)]), torch.tensor([FRAMES])
    captured = {}
    hook = model.ar_predict_layer.register_forward_hook(lambda m, i, o: captured.__setitem__("ar", o))
    with torch.no_grad():
        _, ar_loss, _ = model(x, x_lens, codes, y_lens, reduction="sum", train_stage=1)
    hook.remove()

    # The NAR stage and prompt segment come from model.rng; record what it draws.
    drawn = {}
    rng = random.Random(5)
    original_choices, original_randint = rng.choices, rng.randint
    rng.choices = lambda *a, **k: drawn.setdefault("stage", original_choices(*a, **k))
    rng.randint = lambda *a, **k: drawn.setdefault("start", original_randint(*a, **k))
    model.rng = rng
    nar_logits = {}
    # The NAR logits leave through nar_predict_layers[stage − 1]; hook them all.
    hooks = [layer.register_forward_hook(lambda m, i, o: nar_logits.__setitem__("logits", o)) for layer in model.nar_predict_layers]
    with torch.no_grad():
        _, nar_loss, _ = model(x, x_lens, codes.clone(), y_lens, reduction="sum", train_stage=2)
    for hook in hooks:
        hook.remove()

    with torch.no_grad():
        generated = model.inference(x, x_lens, prompt, enroll_x_lens=torch.tensor([ENROLLED]), top_k=1)
    return {
        "ar_logits": [float(v) for v in captured["ar"].reshape(-1).tolist()],
        "ar_loss": float(ar_loss),
        "nar_stage": int(drawn["stage"][0]),
        "nar_prompt_start": int(drawn["start"]),
        "nar_logits": [float(v) for v in nar_logits["logits"].reshape(-1).tolist()],
        "nar_loss": float(nar_loss),
        "generated": generated[0].tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valle", required=True, help="a checkout of lifeiteng/vall-e")
    parser.add_argument("--write", action="store_true", help="regenerate the fixture (default: verify it)")
    args = parser.parse_args()
    valle = load(args.valle)
    x, codes, prompt = inputs()
    if args.write:
        model = build(valle)
        state = {k: v.float().clone().contiguous() for k, v in model.state_dict().items()
                 if not any(k == f"nar_predict_layers.{j}.weight" for j in range(CONFIG["num_quantizers"] - 2))}
        model = build(valle, {**state, **{f"nar_predict_layers.{j}.weight": state[f"nar_audio_embeddings.{j + 2}.word_embeddings.weight"]
                                          for j in range(CONFIG["num_quantizers"] - 2)}}).double()
        data = {"config": CONFIG, "text": TEXT, "enrolled": ENROLLED,
                "codes": codes[0].tolist(), "prompt": prompt[0].tolist(),
                "safetensors_base64": base64.b64encode(save_safetensors(state)).decode("ascii"), **run(valle, model)}
        _fixture.save(FIXTURE, data, __file__)
        print("wrote " + FIXTURE)
        return
    data = _fixture.load(FIXTURE)
    state = load_safetensors(base64.b64decode(data["safetensors_base64"]))
    state.update({f"nar_predict_layers.{j}.weight": state[f"nar_audio_embeddings.{j + 2}.word_embeddings.weight"]
                  for j in range(CONFIG["num_quantizers"] - 2)})
    again = run(valle, build(valle, state).double())
    check = _fixture.Comparison("VALL-E")
    for key in ["ar_logits", "nar_logits"]:
        check.floats(key, data[key], again[key], 1e-12, float)
    check.floats("losses", [data["ar_loss"], data["nar_loss"]], [again["ar_loss"], again["nar_loss"]], 1e-12, float)
    if [data["nar_stage"], data["nar_prompt_start"], data["generated"]] != [again["nar_stage"], again["nar_prompt_start"], again["generated"]]:
        sys.exit("MISMATCH: the drawn stage, prompt segment or generated codes differ")
    check.report(exact=False)


if __name__ == "__main__":
    main()
