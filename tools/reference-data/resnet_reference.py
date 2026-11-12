"""ResNet reference: torchvision's ResNet (torchvision.models.resnet), the most widely used port of He et al. 2016.

The full networks are too large for a fixture, so two narrow ones are built from torchvision's own classes: its
ResNet.__init__ with the stem and stage widths scaled down, its _make_layer, BasicBlock, Bottleneck and forward. A
basic-block network (blocks 2/1/1/1, widths 8 then 8/16/24/32) and a bottleneck network (blocks 1/2/1/1, widths 8 then
4/8/8/16), each with 5 classes, run in float64 from float32 weights and random BatchNorm statistics on two 3x40x36
images. Per network the fixture records the state dict, the images, the C2-C5 stage outputs and the logits in eval
mode, the cross-entropy against targets (1, 3), that loss after one plain SGD step (lr 0.1) on every parameter, and a
training-mode forward's logits and updated running statistics (bn1 and the last block's last BatchNorm). It also
records the parameter counts of torchvision's full resnet18 to resnet152 (1000 classes), and checks that the narrow
construction at full width reproduces resnet18's and resnet50's state-dict shapes.
Default: verify; --write: regenerate tests/AiDotNet.Tests/ComputerVision/ReferenceData/resnet_reference.json.
"""
import argparse
import json
import os
import sys

FIXTURE = "tests/AiDotNet.Tests/ComputerVision/ReferenceData/resnet_reference.json"


def narrow(block, layers, widths, num_classes):
    """torchvision's ResNet.__init__ with the widths 64, 64, 128, 256, 512 replaced by `widths`."""
    from torch import nn
    from torchvision.models.resnet import ResNet
    model = ResNet.__new__(ResNet)
    nn.Module.__init__(model)
    model._norm_layer = nn.BatchNorm2d
    model.inplanes = widths[0]
    model.dilation = 1
    model.groups = 1
    model.base_width = 64
    model.conv1 = nn.Conv2d(3, widths[0], kernel_size=7, stride=2, padding=3, bias=False)
    model.bn1 = nn.BatchNorm2d(widths[0])
    model.relu = nn.ReLU(inplace=True)
    model.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    model.layer1 = model._make_layer(block, widths[1], layers[0])
    model.layer2 = model._make_layer(block, widths[2], layers[1], stride=2)
    model.layer3 = model._make_layer(block, widths[3], layers[2], stride=2)
    model.layer4 = model._make_layer(block, widths[4], layers[3], stride=2)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model.fc = nn.Linear(widths[4] * block.expansion, num_classes)
    return model


def compute():
    import torch
    from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
    from torchvision.models.resnet import BasicBlock, Bottleneck
    torch.set_default_dtype(torch.float64)

    full = [(18, BasicBlock, [2, 2, 2, 2], resnet18), (50, Bottleneck, [3, 4, 6, 3], resnet50)]
    for depth, block, layers, factory in full:
        ours = {k: tuple(v.shape) for k, v in narrow(block, layers, [64, 64, 128, 256, 512], 1000).state_dict().items()}
        theirs = {k: tuple(v.shape) for k, v in factory().state_dict().items()}
        assert ours == theirs, f"the narrow construction differs from resnet{depth}"
    counts = {f"resnet{d}": sum(p.numel() for p in f().parameters())
              for d, f in [(18, resnet18), (34, resnet34), (50, resnet50), (101, resnet101), (152, resnet152)]}

    def case(name, block, layers, widths, seed):
        torch.manual_seed(seed)
        model = narrow(block, layers, widths, 5).double()
        g = torch.Generator().manual_seed(seed + 1)
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
        images = torch.randn(2, 3, 40, 36, generator=g).float().double()
        targets = torch.tensor([1, 3])

        def stages(x):
            x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
            out = []
            for layer in (model.layer1, model.layer2, model.layer3, model.layer4):
                x = layer(x)
                out.append(x)
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
        # Training mode from the stepped weights: batch statistics, and the running statistics' update.
        model.train()
        with torch.no_grad():
            train_logits = model(images)
        last = model.layer4[-1]
        last_norm = last.bn3 if hasattr(last, "bn3") else last.bn2
        return {"name": name, "bottleneck": block is Bottleneck, "blocks": layers, "widths": widths, "state": state,
                "images": images.flatten().tolist(), "targets": targets.tolist(), "features": features,
                "feature_shapes": feature_shapes,
                "logits": logits.detach().flatten().tolist(), "loss": loss.item(), "loss_after_sgd": after,
                "train_logits": train_logits.flatten().tolist(),
                "train_bn1_running_mean": model.bn1.running_mean.tolist(),
                "train_bn1_running_var": model.bn1.running_var.tolist(),
                "train_last_running_var": last_norm.running_var.tolist()}

    return {"parameter_counts": counts,
            "cases": [case("basic", BasicBlock, [2, 1, 1, 1], [8, 8, 16, 24, 32], 3),
                      case("bottleneck", Bottleneck, [1, 2, 1, 1], [8, 4, 8, 8, 16], 5)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = compute()
    if args.write:
        os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
        with open(FIXTURE, "w", encoding="utf-8", newline="\n") as f:
            json.dump(result, f)
        print("wrote", result["parameter_counts"], [(c["name"], round(c["loss"], 9), round(c["loss_after_sgd"], 9))
                                                    for c in result["cases"]])
        return
    with open(FIXTURE, encoding="utf-8") as f:
        stored = json.load(f)
    same = stored["parameter_counts"] == result["parameter_counts"] and all(
        abs(a["loss"] - b["loss"]) < 1e-12 and abs(a["loss_after_sgd"] - b["loss_after_sgd"]) < 1e-12
        for a, b in zip(stored["cases"], result["cases"]))
    print("fixture matches" if same else "fixture differs")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
