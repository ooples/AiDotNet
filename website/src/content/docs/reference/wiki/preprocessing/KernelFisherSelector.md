---
title: "KernelFisherSelector<T>"
description: "Kernel Fisher Discriminant-based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Kernel`

Kernel Fisher Discriminant-based Feature Selection.

## For Beginners

Fisher's criterion tries to maximize the distance
between class means while minimizing the spread within each class. Doing this
in kernel space allows us to find non-linear boundaries between classes,
selecting features that help separate classes in complex ways.

## How It Works

Applies the Fisher criterion in kernel space to find features that maximize
class separation using non-linear transformations.

