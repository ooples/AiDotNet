"""EAST reference: a PyTorch transcription of the paper's network (Zhou et al. 2017, "EAST: An Efficient and Accurate
Scene Text Detector", Sec. 3.2 and Fig. 3) with the VGG16 stem.

No official code was released. The transcription follows the paper where it is explicit and the public
reimplementation (argman/EAST) where it is not:
  * stem: torchvision's vgg16().features (configuration D, no batch norm); the feature maps after pooling-2 to
    pooling-5 are f4 (1/4) to f1 (1/32);
  * merging branch (Eq. 1-2): h1 = f1; for stages 2-4, g = unpool(h), h = conv3x3(conv1x1([g; f])) with 128, 64 and
    32 channels (Fig. 3); then g4 = conv3x3(h4) with 32 channels. Each convolution is followed by batch norm and ReLU
    (argman); unpool is a x2 bilinear resize with TF1's resize_bilinear (asymmetric: src = dst * in / out), as argman's
    tf.image.resize_bilinear;
  * output layer: conv1x1 to a sigmoid score map, and to 5 RBOX channels: four distances sigmoid * text scale (the
    map side here) and the angle (sigmoid - 0.5) * pi / 2 (argman model.py).
VGG16 has 14.7 million weights, too many to record, so every parameter is generated from an integer hash of its name
and index (synthetic(), reproduced exactly by the C# test) and the network runs in eval mode on two 3x64x96 images
generated the same way. Recorded: the score and geometry maps, the loss (the mean squared score map plus the mean squared geometry map) and the
loss after one plain SGD step (lr 0.01) on every parameter.
Default: verify; --write: regenerate tests/AiDotNet.Tests/ComputerVision/ReferenceData/east_reference.json.
"""
import argparse
import json
import math
import os
import sys

FIXTURE = "tests/AiDotNet.Tests/ComputerVision/ReferenceData/east_reference.json"


def build():
    import torch
    import torch.nn as nn
    import torchvision

    def resize_asymmetric(x, out_h, out_w):
        def taps(size_in, size_out):
            src = torch.arange(size_out, dtype=torch.float64) * size_in / size_out
            i0 = src.floor().long()
            i1 = torch.clamp(i0 + 1, max=size_in - 1)
            w1 = src - i0.double()
            return i0, i1, 1 - w1, w1
        y0, y1, wy0, wy1 = taps(x.shape[2], out_h)
        x0, x1, wx0, wx1 = taps(x.shape[3], out_w)
        rows = x[:, :, y0, :] * wy0[None, None, :, None] + x[:, :, y1, :] * wy1[None, None, :, None]
        return rows[:, :, :, x0] * wx0 + rows[:, :, :, x1] * wx1

    class Stage(nn.Module):
        def __init__(self, inputs, channels):
            super().__init__()
            self.reduce = nn.Conv2d(inputs, channels, 1)
            self.reduce_norm = nn.BatchNorm2d(channels)
            self.fuse = nn.Conv2d(channels, channels, 3, padding=1)
            self.fuse_norm = nn.BatchNorm2d(channels)

        def forward(self, x):
            x = torch.relu(self.reduce_norm(self.reduce(x)))
            return torch.relu(self.fuse_norm(self.fuse(x)))

    class East(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = torchvision.models.vgg16(weights=None)
            del self.stem.classifier, self.stem.avgpool
            self.merge = nn.ModuleList([Stage(512 + 512, 128), Stage(128 + 256, 64), Stage(64 + 128, 32)])
            self.final = nn.Conv2d(32, 32, 3, padding=1)
            self.final_norm = nn.BatchNorm2d(32)
            self.score = nn.Conv2d(32, 1, 1)
            self.geometry = nn.Conv2d(32, 5, 1)

        def forward(self, images):
            features, x = [], images
            for i, layer in enumerate(self.stem.features):
                x = layer(x)
                if i in (9, 16, 23, 30):           # pool2 .. pool5
                    features.append(x)
            h = features[-1]
            for i, stage in enumerate(self.merge):
                skip = features[-2 - i]
                h = stage(torch.cat([resize_asymmetric(h, skip.shape[2], skip.shape[3]), skip], dim=1))
            g = torch.relu(self.final_norm(self.final(h)))
            score = torch.sigmoid(self.score(g))
            raw = self.geometry(g)
            side = max(raw.shape[2], raw.shape[3])
            geometry = torch.cat([torch.sigmoid(raw[:, :4]) * side, (torch.sigmoid(raw[:, 4:]) - 0.5) * math.pi / 2], dim=1)
            return score, geometry

    model = East().double()
    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            if name.endswith("num_batches_tracked"):
                continue
            tensor.copy_(torch.tensor(synthetic(name, list(tensor.shape)), dtype=torch.float64).reshape(tensor.shape))
    return model.eval()


def uniforms(name, count):
    """u in [0, 1) from FNV-1a of the name and a xorshift of the index; exact integer arithmetic."""
    h = 2166136261
    for ch in name.encode("ascii"):
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    out = []
    for j in range(count):
        x = (h ^ ((j * 2654435761) & 0xFFFFFFFF)) & 0xFFFFFFFF
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        out.append(x / 4294967296.0)
    return out


def synthetic(name, shape):
    """A parameter's values: uniform with the variance of Kaiming init for weights, near-identity batch norms."""
    count = 1
    for d in shape:
        count *= d
    u = uniforms(name, count)
    if name.endswith("running_var") or (name.endswith("norm.weight")):
        return [0.5 + v for v in u]
    if name.endswith("running_mean") or name.endswith(".bias"):
        return [0.1 * (2 * v - 1) for v in u]
    fan_in = count // shape[0]
    bound = math.sqrt(6.0 / fan_in)
    return [bound * (2 * v - 1) for v in u]


def compute():
    import torch
    model = build()
    images = torch.tensor(uniforms("images", 2 * 3 * 64 * 96), dtype=torch.float64).reshape(2, 3, 64, 96)

    def loss_of(score, geometry):
        return (score ** 2).mean() + (geometry ** 2).mean()

    score, geometry = model(images)
    loss = loss_of(score, geometry)
    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p -= 0.01 * p.grad
        after = loss_of(*model(images)).item()
    return {"score": score.detach().flatten().tolist(),
            "geometry": geometry.detach().flatten().tolist(), "loss": loss.item(), "loss_after_sgd": after}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = compute()
    if args.write:
        os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
        with open(FIXTURE, "w", encoding="utf-8", newline="\n") as f:
            json.dump(result, f)
        print("wrote", round(result["loss"], 9), round(result["loss_after_sgd"], 9))
        return
    with open(FIXTURE, encoding="utf-8") as f:
        stored = json.load(f)
    same = abs(stored["loss"] - result["loss"]) < 1e-12 and abs(stored["loss_after_sgd"] - result["loss_after_sgd"]) < 1e-12
    print("fixture matches" if same else "fixture differs")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
