"""Multi-speaker unit HiFi-GAN reference: speech-resynthesis' duration-predicting unit generator with speaker embeddings
(examples/expresso/models.py MultiSpkrMultiStyleCodeGenerator over examples/speech_to_speech_translation/models.py
DurationCodeGenerator; the latter alone calls an _upsample only its subclasses define), the unit vocoder SpeechGPT's paper trains (Polyak et al.
2021). Pass the facebookresearch/speech-resynthesis checkout with --resynthesis.

Records in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/code_hifigan_multispeaker_reference.json, in float64, for a
tiny generator (seeded; dropout 0 so the training branch is deterministic):
  - the training forward on frame-level codes with repeats (the waveform, and the duration loss over the middle runs);
  - the same with only two runs (every run kept);
  - inference on deduplicated units with duration prediction, for two speakers.
utils.py imports matplotlib for plotting only; it is stubbed. process_duration's float32 run lengths are made float64. Default: verify; --write: regenerate.
"""
import argparse
import base64
import os
import sys
import types

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fixture  # noqa: E402

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/code_hifigan_multispeaker_reference.json"
CONFIG = {"resblock": "1", "upsample_rates": [5, 4], "upsample_kernel_sizes": [11, 8], "upsample_initial_channel": 16,
          "resblock_kernel_sizes": [3, 7], "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5]], "num_embeddings": 20,
          "embedding_dim": 8, "model_in_dim": 16, "multispkr": "_", "num_speakers": 200, "dur_prediction_weight": 1.0,
          "dur_predictor_params": {"encoder_embed_dim": 8, "var_pred_hidden_dim": 8, "var_pred_kernel_size": 3, "var_pred_dropout": 0.0}}
CODES = [3, 3, 3, 17, 17, 0, 9, 9, 9, 9, 12, 5, 5, 19]
TWO_RUNS = [4, 4, 4, 11, 11]
UNITS = [3, 17, 0, 9, 12, 5]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resynthesis", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    for name in ("matplotlib", "matplotlib.pylab"):
        stub = types.ModuleType(name)
        stub.use = lambda *a, **k: None
        sys.modules[name] = stub
    sys.modules["matplotlib"].pylab = sys.modules["matplotlib.pylab"]
    sys.path.insert(0, args.resynthesis)
    from utils import AttrDict
    from examples.expresso.models import MultiSpkrMultiStyleCodeGenerator as DurationCodeGenerator
    torch.set_default_dtype(torch.float64)
    stored = None if args.write else _fixture.load(FIXTURE)
    torch.manual_seed(2294)
    model = DurationCodeGenerator(AttrDict(CONFIG))
    with torch.no_grad():
        model.dur_predictor.proj.bias.fill_(1.4)
        for p in model.parameters():
            p.copy_(p.float().double())
    if stored:
        model.load_state_dict(load_safetensors(base64.b64decode(stored["safetensors_base64"])))
    model = model.double()

    def training(codes, speaker):
        # process_duration returns the run lengths as .float() (float32); kept in float64.
        model.train()
        original_float = torch.Tensor.float
        torch.Tensor.float = lambda self, *a, **k: self.double()
        try:
            wave, loss = model(code=torch.tensor([codes]), spkr=torch.tensor([[speaker]]))
        finally:
            torch.Tensor.float = original_float
        return [float(v) for v in wave.detach().reshape(-1).tolist()], float(loss)

    wave, loss = training(CODES, 7)
    wave_two, loss_two = training(TWO_RUNS, 2)
    model.eval()
    with torch.no_grad():
        inferred = {f"inferred_{s}": [float(v) for v in model(code=torch.tensor([UNITS]), spkr=torch.tensor([[s]]), dur_prediction=True)[0].reshape(-1).tolist()]
                    for s in (0, 7)}
    data = {"training_wave": wave, "training_loss": loss, "two_runs_wave": wave_two, "two_runs_loss": loss_two, **inferred}
    if args.write:
        state = {k: v.detach().float().contiguous() for k, v in model.state_dict().items()}
        _fixture.save(FIXTURE, {"config": CONFIG, "codes": CODES, "two_runs": TWO_RUNS, "units": UNITS,
                                "safetensors_base64": base64.b64encode(save_safetensors(state)).decode("ascii"), **data}, __file__)
        print("wrote", FIXTURE, loss, loss_two, len(inferred["inferred_0"]))
        return
    check = _fixture.Comparison("multi-speaker CodeHiFiGAN")
    for k, v in data.items():
        check.floats(k, stored[k] if isinstance(stored[k], list) else [stored[k]], v if isinstance(v, list) else [v], 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
