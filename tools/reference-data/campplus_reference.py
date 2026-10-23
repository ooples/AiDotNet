"""CAM++ speaker encoder reference: modelscope/3D-Speaker's CAMPPlus (commit 065629c313eaf1a01c65c640c46d77e61e9607b4,
speakerlab/models/campplus/DTDNN.py and layers.py), the x-vector extractor CosyVoice ships as campplus.onnx. Pass a
directory holding the speakerlab package with --speakerlab.

Builds a small randomly initialized encoder (seeded, float64; the D-TDNN's 12/24/16 layers kept, narrow widths) with
randomized BatchNorm statistics, and records in tests/AiDotNet.Tests/TextToSpeech/ReferenceData/campplus_reference.json:
- the embeddings of a batch of two filterbank sequences in eval mode (230 frames: after the TDNN's stride 2, the
  context-aware masking pools a full 100-frame segment and a short 15-frame one);
- the training-mode loss sum(embedding * target) with batch statistics;
- the eval-mode embeddings after one SGD step, which also read the running statistics the training forward updated.
  The step is lr 1e-3: the training-mode embedding norm divides by a small batch variance, so larger steps diverge.
Default: verify; --write: regenerate.
"""
import argparse
import base64
import sys

import torch
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

import _fixture

FIXTURE = "tests/AiDotNet.Tests/TextToSpeech/ReferenceData/campplus_reference.json"
CONFIG = dict(feat_dim=16, embedding_size=6, growth_rate=4, bn_size=2, init_channels=8)
FRAMES = 230
BATCH = 2
LEARNING_RATE = 1e-3


def flat(tensor):
    return [float(v) for v in tensor.detach().reshape(-1).tolist()]


def build(root, state=None):
    sys.path.insert(0, root)
    from speakerlab.models.campplus.DTDNN import CAMPPlus
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(2294)
    model = CAMPPlus(**CONFIG, memory_efficient=False)
    with torch.no_grad():
        for name, buffer in model.named_buffers():
            if name.endswith("running_mean"):
                buffer.copy_(0.1 * torch.randn_like(buffer))
            elif name.endswith("running_var"):
                buffer.copy_(0.5 + torch.rand_like(buffer))
        for name, p in model.named_parameters():
            if "batchnorm" in name or ".bn" in name or "shortcut.1" in name:
                p.add_(0.1 * torch.randn_like(p))
    if state is not None:
        model.load_state_dict(state, strict=True)
    return model


def run(model, features, target):
    model.eval()
    with torch.no_grad():
        embedding = model(features)
    model.train()
    loss = (model(features) * target).sum()
    parameters = list(model.parameters())
    grads = torch.autograd.grad(loss, parameters)
    with torch.no_grad():
        for p, g in zip(parameters, grads):
            p -= LEARNING_RATE * g
    model.eval()
    with torch.no_grad():
        after = model(features)
    return embedding, loss, after


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speakerlab", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.write:
        model = build(args.speakerlab)
        state = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}
        generator = torch.Generator().manual_seed(7)
        features = torch.randn(BATCH, FRAMES, CONFIG["feat_dim"], generator=generator)
        target = torch.randn(BATCH, CONFIG["embedding_size"], generator=generator)
        embedding, loss, after = run(model, features, target)
        data = {"config": CONFIG, "frames": FRAMES, "batch": BATCH, "learning_rate": LEARNING_RATE,
                "safetensors_base64": base64.b64encode(save_safetensors(state)).decode("ascii"),
                "features": flat(features), "target": flat(target), "embedding": flat(embedding),
                "loss": float(loss.detach()), "embedding_after_sgd": flat(after)}
        _fixture.save(FIXTURE, data, __file__)
        print("wrote " + FIXTURE)
        return
    data = _fixture.load(FIXTURE)
    model = build(args.speakerlab, load_safetensors(base64.b64decode(data["safetensors_base64"])))
    features = torch.tensor(data["features"]).reshape(BATCH, FRAMES, CONFIG["feat_dim"])
    target = torch.tensor(data["target"]).reshape(BATCH, CONFIG["embedding_size"])
    embedding, loss, after = run(model, features, target)
    check = _fixture.Comparison("CAM++")
    check.floats("embedding", data["embedding"], flat(embedding), 1e-12, float)
    check.floats("loss", [data["loss"]], [float(loss.detach())], 1e-12, float)
    check.floats("embedding_after_sgd", data["embedding_after_sgd"], flat(after), 1e-12, float)
    check.report(exact=False)


if __name__ == "__main__":
    main()
