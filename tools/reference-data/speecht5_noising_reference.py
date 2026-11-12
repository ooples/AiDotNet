"""SpeechT5's pre-training noise as the official code draws it, with every random draw recorded.

Records in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/speecht5_noising_reference.json: for token sequences of
several lengths, the official TextPretrainDataset.add_whole_word_mask (BART's text infilling with the recipe's settings:
mask 0.3, mask-random 0.1, Poisson λ 3.5, replace-length 1, no BPE so every inner token starts a word) and, for frame
counts, fairseq's compute_mask_indices (mask-prob 0.8, length 10, static, min-masks 2); each case with the draws it made
(categorical samples, permutations, uniforms, integers, choices) in order, which the C# port replays.

Usage (default: verify; --write: regenerate), in the speecht5-ref image:
  docker run --rm -v <repo>:/work -v <SpeechT5>/SpeechT5/speecht5:/code/speecht5:ro -w /work speecht5-ref \\
    python tools/reference-data/speecht5_noising_reference.py [--write]
"""
import argparse
import importlib.util
import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fixture  # noqa: E402

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/speecht5_noising_reference.json"
VOCAB = 81
MASK = 79

draws = []


class TorchProxy:
    """The real torch, with randperm, randint and FloatTensor(n).uniform_() recorded."""

    def __getattr__(self, name):
        return getattr(torch, name)

    @staticmethod
    def randperm(n, *a, **k):
        p = torch.randperm(n, *a, **k)
        draws.append(["permutation", n, p.tolist()])
        return p

    @staticmethod
    def randint(low, high, size, *a, **k):
        v = torch.randint(low, high, size, *a, **k)
        draws.append(["integers", [low, high], v.tolist()])
        return v

    @staticmethod
    def FloatTensor(n):
        class Uniform:
            def uniform_(self):
                u = torch.FloatTensor(n).uniform_()
                draws.append(["uniforms", n, [float(x) for x in u.tolist()]])
                return u
        return Uniform()


class RecordingCategorical:
    def __init__(self, inner):
        self.inner = inner

    def sample(self, sample_shape):
        s = self.inner.sample(sample_shape=sample_shape)
        draws.append(["categorical", int(sample_shape[0]), s.tolist()])
        return s


class NumpyRandomProxy:
    def __getattr__(self, name):
        return getattr(np.random, name)

    @staticmethod
    def rand():
        u = np.random.rand()
        draws.append(["uniforms", 1, [float(u)]])
        return u

    @staticmethod
    def choice(n, size, replace=True, p=None):
        c = np.random.choice(n, size, replace=replace, p=p)
        draws.append(["choice", [int(n), int(size)], [int(x) for x in c]])
        return c


def load_text_dataset():
    sys.modules.setdefault("speecht5", types.ModuleType("speecht5"))
    sys.modules["speecht5"].__path__ = ["/code/speecht5"]
    data = types.ModuleType("speecht5.data")
    data.__path__ = ["/code/speecht5/data"]
    sys.modules["speecht5.data"] = data
    spec = importlib.util.spec_from_file_location("speecht5.data.text_dataset", "/code/speecht5/data/text_dataset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.torch = TorchProxy()
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    stored = None if args.write else _fixture.load(FIXTURE)
    module = load_text_dataset()
    import fairseq.data.data_utils as data_utils
    data_utils.np = types.SimpleNamespace(**{k: getattr(np, k) for k in dir(np) if not k.startswith("__")})
    data_utils.np.random = NumpyRandomProxy()

    dataset = module.TextPretrainDataset.__new__(module.TextPretrainDataset)
    dataset.vocab = list(range(VOCAB))
    dataset.mask_idx = MASK
    dataset.mask_whole_word = None
    dataset.random_ratio = 0.1
    dataset.replace_length = 1
    dataset.iid_noise_target = False
    weights, power, factorial, e = [], 1, 1, np.exp(-3.5)
    for k in range(128):
        weights.append(e * power / factorial)
        power *= 3.5
        factorial *= k + 1
        if weights[-1] < 0.0000001:
            break
    dataset.mask_span_distribution = RecordingCategorical(torch.distributions.Categorical(torch.FloatTensor(weights)))

    text_cases, span_cases = [], []
    for case, length in enumerate([3, 4, 6, 9, 14, 20, 27, 33, 41, 52, 64]):
        for seed in range(3):
            torch.manual_seed(1000 * case + seed)
            source = torch.tensor([0] + list(torch.randint(4, 79, (length - 2,)).tolist()) + [2])
            draws.clear()
            noised = dataset.add_whole_word_mask(source.clone(), 0.3)
            noised = noised[0] if isinstance(noised, tuple) else noised
            text_cases.append({"source": source.tolist(), "noised": noised.tolist(), "draws": [list(d) for d in draws]})
    for case, frames in enumerate([23, 31, 49, 50, 77, 120, 249, 400]):
        np.random.seed(77 + case)
        draws.clear()
        mask = data_utils.compute_mask_indices((1, frames), None, 0.8, 10, "static", 0.0, min_masks=2, no_overlap=False, min_space=1)
        span_cases.append({"frames": frames, "mask": [bool(x) for x in mask[0]], "draws": [list(d) for d in draws]})

    if args.write:
        _fixture.save(FIXTURE, {"text": text_cases, "spans": span_cases}, __file__)
        print("wrote", FIXTURE, len(text_cases), "text cases,", len(span_cases), "span cases;",
              sum(1 for c in text_cases if any(d[0] == "permutation" and d[1] != len(c["source"]) - 2 for d in c["draws"])), "with insertions")
        return
    check = _fixture.Comparison("SpeechT5 noising")
    check.exact("text", stored["text"], text_cases)
    check.exact("spans", stored["spans"], span_cases)
    check.report()


if __name__ == "__main__":
    main()
