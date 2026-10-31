"""Step-1 reference (Step-Audio's LLM, stepfun-ai/Step-Audio-TTS-3B modeling_step1.py): a tiny Step1ForCausalLM (width 24,
6 heads in 2 groups (not a power of two, so both slope series run), 2 layers, intermediate 40, vocabulary 50) with seeded random weights in float64 (StepRMSNorm's float32 mean square patched out), through the
release's non-flash path (repeat_interleave groups, build_alibi_cache's √-distance ALiBi, scaled_dot_product_attention).
Records the logits of a 7-token sequence and of the same sequence fed 4 + 1 + 1 + 1 tokens through the KV cache.

transformers 4.48.3 (the release's pin) leaves lm_head untied (Step1ForCausalLM defines no get_output_embeddings);
newer versions would tie it, so tie_word_embeddings is set False to reproduce the release.
Usage: python tools/reference-data/step1_reference.py --modeling <dir with modeling_step1.py, configuration_step1.py> [--write]
"""
import argparse
import os
import sys
import types
import importlib.util

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fixture  # noqa: E402

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/step1_reference.json"


def load(directory):
    package = types.ModuleType("step1pkg")
    package.__path__ = [directory]
    sys.modules["step1pkg"] = package
    for name, file in (("configuration_step1", "tts3b_configuration_step1.py"), ("modeling_step1", "tts3b_modeling_step1.py")):
        spec = importlib.util.spec_from_file_location(f"step1pkg.{name}", os.path.join(directory, file))
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"step1pkg.{name}"] = module
        spec.loader.exec_module(module)
    modeling = sys.modules["step1pkg.modeling_step1"]

    # StepRMSNorm takes the mean square in float32; the comparison keeps float64.
    def rms_forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

    modeling.StepRMSNorm.forward = rms_forward
    return sys.modules["step1pkg.configuration_step1"], modeling


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    configuration, modeling = load(args.modeling)
    config = configuration.Step1Config(hidden_size=24, intermediate_size=40, num_attention_heads=6, num_attention_groups=2,
                                       num_hidden_layers=2, vocab_size=50, tie_word_embeddings=False, initializer_range=0.02)
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    model = modeling.Step1ForCausalLM(config).double().eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(1 + 0.2 * torch.randn_like(p) if "norm" in name else 0.3 * torch.randn_like(p))
    assert model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr(), "lm_head must stay untied"
    ids = torch.tensor([[1, 4, 17, 9, 33, 3, 21]])
    with torch.no_grad():
        full = model(input_ids=ids, use_cache=False).logits[0]
        cache = modeling.DynamicCache()
        steps = []
        for start, end in ((0, 4), (4, 5), (5, 6), (6, 7)):
            out = model(input_ids=ids[:, start:end], past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            steps.append(out.logits[0].reshape(-1).tolist())
    data = {"logits": full.reshape(-1).tolist(), "steps": steps}
    if args.write:
        state = {k: v.reshape(-1).tolist() for k, v in model.state_dict().items()}
        _fixture.save(FIXTURE, {"ids": ids[0].tolist(), "state": state, **data}, __file__)
        print("wrote", FIXTURE, full.shape)
        return
    stored = _fixture.load(FIXTURE)
    check = _fixture.Comparison("Step-1")
    check.floats("logits", stored["logits"], data["logits"], 1e-12, float)
    for i, (a, b) in enumerate(zip(stored["steps"], data["steps"])):
        check.floats(f"step {i}", a, b, 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
