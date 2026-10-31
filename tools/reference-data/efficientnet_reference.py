"""EfficientNet reference: timm's tf_efficientnet, the PyTorch port of the official TensorFlow EfficientNet
(tpu/models/official/efficientnet) with its "SAME" padding and BatchNorm epsilon 1e-3.

A narrow network is built by timm's own generator, _gen_efficientnet('tf_efficientnet_b0', ...), with width multiplier
0.25 and depth multiplier 0.3 (stem 8, stages 8/8/8/16/24/32/48 over 1/1/1/1/1/2/1 blocks, head 320), 5 classes, the
paper's BatchNorm momentum 0.99 (PyTorch momentum 0.01), and no dropout or stochastic depth so the training-mode
forward is deterministic. It runs in float64 from float32 weights and random BatchNorm statistics on two 3x34x30
images (odd strided sizes exercise the asymmetric padding). The fixture records the state dict, the images, the seven
stage outputs and the head output in eval mode, the logits, the cross-entropy against targets (1, 3), that loss after
one plain SGD step (lr 0.1) on every parameter, and a training-mode forward's logits and updated running statistics
(bn1 and the head's bn2). It also records the parameter counts of timm's full tf_efficientnet_b0 to b7 (1000 classes).
Default: verify; --write: regenerate tests/AiDotNet.Tests/ComputerVision/ReferenceData/efficientnet_reference.json.
"""
import argparse
import json
import os
import sys

FIXTURE = "tests/AiDotNet.Tests/ComputerVision/ReferenceData/efficientnet_reference.json"


def compute():
    import timm
    import torch
    from timm.models.efficientnet import _gen_efficientnet
    torch.set_default_dtype(torch.float64)

    counts = {}
    for b in range(8):
        model = timm.create_model(f"tf_efficientnet_b{b}", pretrained=False)
        counts[f"b{b}"] = sum(p.numel() for p in model.parameters())

    torch.manual_seed(3)
    model = _gen_efficientnet("tf_efficientnet_b0", channel_multiplier=0.25, depth_multiplier=0.3, num_classes=5,
                              bn_eps=1e-3, bn_momentum=0.01, pad_type="same", drop_rate=0.0,
                              drop_path_rate=0.0).double()
    g = torch.Generator().manual_seed(4)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(p.float())
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                c = m.num_features
                m.weight.copy_((1 + 0.2 * torch.randn(c, generator=g)).float())
                m.bias.copy_((0.1 * torch.randn(c, generator=g)).float())
                m.running_mean.copy_((0.1 * torch.randn(c, generator=g)).float())
                m.running_var.copy_((1 + 0.5 * torch.rand(c, generator=g)).float())
    state = {k: v.detach().flatten().tolist() for k, v in model.state_dict().items() if "num_batches" not in k}
    images = torch.randn(2, 3, 34, 30, generator=g).float().double()
    targets = torch.tensor([1, 3])

    def stages(x):
        x = model.bn1(model.conv_stem(x))
        out = []
        for stage in model.blocks:
            x = stage(x)
            out.append(x)
        out.append(model.bn2(model.conv_head(x)))
        return out

    model.eval()
    outputs = stages(images)
    features = [f.detach().flatten().tolist() for f in outputs]
    feature_shapes = [list(f.shape) for f in outputs]
    logits = model(images)
    loss = torch.nn.functional.cross_entropy(logits, targets)
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            p -= 0.1 * p.grad
        after = torch.nn.functional.cross_entropy(model(images), targets).item()
    model.train()
    with torch.no_grad():
        train_logits = model(images)
    return {"parameter_counts": counts, "state": state, "images": images.flatten().tolist(), "targets": targets.tolist(),
            "features": features, "feature_shapes": feature_shapes, "logits": logits.detach().flatten().tolist(),
            "loss": loss.item(), "loss_after_sgd": after, "train_logits": train_logits.flatten().tolist(),
            "train_bn1_running_mean": model.bn1.running_mean.tolist(),
            "train_bn1_running_var": model.bn1.running_var.tolist(),
            "train_head_running_var": model.bn2.running_var.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = compute()
    if args.write:
        os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
        with open(FIXTURE, "w", encoding="utf-8", newline="\n") as f:
            json.dump(result, f)
        print("wrote", result["parameter_counts"], round(result["loss"], 9), round(result["loss_after_sgd"], 9),
              result["feature_shapes"])
        return
    with open(FIXTURE, encoding="utf-8") as f:
        stored = json.load(f)
    same = stored["parameter_counts"] == result["parameter_counts"] and abs(stored["loss"] - result["loss"]) < 1e-12 \
        and abs(stored["loss_after_sgd"] - result["loss_after_sgd"]) < 1e-12
    print("fixture matches" if same else "fixture differs")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
