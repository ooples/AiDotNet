"""Unit-based HiFi-GAN reference: fairseq's CodeGenerator (fairseq/models/text_to_speech/codehifigan.py over hifigan.py's
Generator and fastspeech2.py's VariancePredictor), as CodeHiFiGANVocoder runs it. Pass the fairseq checkout with --fairseq.

Records in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/code_hifigan_reference.json, in float64:
  - a tiny generator (seeded): the log durations of 10 units, and the waveforms with and without duration prediction;
  - with --vocoder (the released mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj g_00500000), the same for 30 units, in
    code_hifigan_realweights_reference.json.
fairseq is not importable whole here: hifigan.py and codehifigan.py are loaded from the checkout, and VariancePredictor's
source is executed from fastspeech2.py with FairseqDropout as torch's Dropout (the vocoder runs in eval mode).
Default: verify; --write: regenerate.
"""
import argparse
import base64
import importlib.util
import os
import re
import sys
import types

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fixture  # noqa: E402

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/code_hifigan_reference.json"
REAL_FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/code_hifigan_realweights_reference.json"
CONFIG = {"upsample_rates": [5, 4], "upsample_kernel_sizes": [11, 8], "upsample_initial_channel": 16,
          "resblock_kernel_sizes": [3, 7], "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5]], "num_embeddings": 20,
          "embedding_dim": 8, "model_in_dim": 8,
          "dur_predictor_params": {"encoder_embed_dim": 8, "var_pred_hidden_dim": 8, "var_pred_kernel_size": 3, "var_pred_dropout": 0.5}}
UNITS = [3, 17, 0, 9, 9, 12, 5, 19, 1, 7]


def load_fairseq(root):
    base = os.path.join(root, "fairseq", "models", "text_to_speech")
    for name in ["fairseq", "fairseq.models", "fairseq.models.text_to_speech"]:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    spec = importlib.util.spec_from_file_location("fairseq.models.text_to_speech.hifigan", os.path.join(base, "hifigan.py"))
    hifigan = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = hifigan
    spec.loader.exec_module(hifigan)
    with open(os.path.join(base, "fastspeech2.py"), encoding="utf-8") as handle:
        source = handle.read()
    variance = re.search(r"^class VariancePredictor\(nn\.Module\):.*?(?=^class )", source, re.S | re.M).group(0)
    fastspeech2 = types.ModuleType("fairseq.models.text_to_speech.fastspeech2")
    fastspeech2.__dict__.update(nn=torch.nn, FairseqDropout=lambda p, module_name=None: torch.nn.Dropout(p))
    exec(compile(variance, "fastspeech2.py", "exec"), fastspeech2.__dict__)
    sys.modules[fastspeech2.__name__] = fastspeech2
    spec = importlib.util.spec_from_file_location("fairseq.models.text_to_speech.codehifigan", os.path.join(base, "codehifigan.py"))
    codehifigan = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(codehifigan)
    return codehifigan.CodeGenerator


def run(model, units):
    with torch.no_grad():
        code = torch.tensor([units])
        x = model.dict(code)
        log_durations = model.dur_predictor(x)[0]
        with_durations = model(code=code, dur_prediction=True).reshape(-1)
        without = model(code=code, dur_prediction=False).reshape(-1)
    return log_durations, with_durations, without


def floats(t):
    return [float(v) for v in t.reshape(-1).tolist()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fairseq", required=True)
    parser.add_argument("--vocoder")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    CodeGenerator = load_fairseq(args.fairseq)
    torch.set_default_dtype(torch.float64)

    if args.vocoder:
        import json
        config = json.load(open(os.path.join(os.path.dirname(args.vocoder), "code_hifigan_config.json")))
        model = CodeGenerator(config)
        model.load_state_dict(torch.load(args.vocoder, map_location="cpu", weights_only=False)["generator"])
        model = model.double().eval()
        stored = None if args.write else _fixture.load(REAL_FIXTURE)
        units = stored["units"] if stored else [(37 * i + 11) % 1000 for i in range(30)]
        log_durations, with_durations, without = run(model, units)
        results = {"log_durations": floats(log_durations), "with_durations": floats(with_durations), "without": floats(without)}
        if args.write:
            _fixture.save(REAL_FIXTURE, {"units": units, **results}, __file__)
            print("wrote", REAL_FIXTURE, len(results["with_durations"]), "samples")
            return
        check = _fixture.Comparison("CodeHiFiGAN, released vocoder")
        for k, v in results.items():
            check.floats(k, stored[k], v, 1e-12, float)
        check.report(exact=False)
        return

    stored = None if args.write else _fixture.load(FIXTURE)
    torch.manual_seed(2294)
    model = CodeGenerator(CONFIG)
    with torch.no_grad():
        # Durations of a few frames each, so the repetition is exercised.
        model.dur_predictor.proj.bias.fill_(1.4)
        for p in model.parameters():
            p.copy_(p.float().double())
    if stored:
        model.load_state_dict(load_safetensors(base64.b64decode(stored["safetensors_base64"])))
    model = model.double().eval()
    log_durations, with_durations, without = run(model, UNITS)
    results = {"log_durations": floats(log_durations), "with_durations": floats(with_durations), "without": floats(without)}
    if args.write:
        state = {k: v.detach().float().contiguous() for k, v in model.state_dict().items()}
        _fixture.save(FIXTURE, {"config": CONFIG, "units": UNITS, "safetensors_base64": base64.b64encode(save_safetensors(state)).decode("ascii"),
                                **results}, __file__)
        print("wrote", FIXTURE, "durations", [round(float(torch.exp(d) - 1)) for d in log_durations])
        return
    check = _fixture.Comparison("CodeHiFiGAN")
    for k, v in results.items():
        check.floats(k, stored[k], v, 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
