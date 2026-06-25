---
title: "SignalToNoiseRatio<T>"
description: "Signal-to-Noise Ratio (SNR) for binary classification feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter`

Signal-to-Noise Ratio (SNR) for binary classification feature selection.

## For Beginners

Think of the "signal" as the difference between
class averages (what distinguishes them) and "noise" as the variation within
each class (what makes it hard to tell them apart). A high SNR means the
distinguishing signal is much stronger than the confusing noise.

## How It Works

SNR measures the ratio of the difference in class means to the sum of their
standard deviations. Higher SNR indicates features where classes are well
separated relative to their internal variation.

