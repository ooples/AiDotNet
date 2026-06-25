---
title: "CTCLoss"
description: "Implements the Connectionist Temporal Classification (CTC) loss function for sequence-to-sequence learning."
section: "Reference"
---

_Loss Functions_

Implements the Connectionist Temporal Classification (CTC) loss function for sequence-to-sequence learning.

## For Beginners

Connectionist Temporal Classification (CTC) is a loss function designed for sequence-to-sequence learning problems where the alignment between input and output sequences is unknown. For example, in speech recognition, we have: - Input: An audio waveform (long sequence of sound samples) - Output: Text transcript (shorter sequence of characters) The key challenge is that we don't know exactly which parts of the audio correspond to each character. CTC solves this by considering all possible alignments between the input and output sequences. CTC introduces a special "blank" token to handle: - Repetitions of characters (e.g., "hello" vs "hheellloo") - Silence or transitions between sounds This loss function is commonly used in: - Speech recognition - Handwriting recognition - Any task where input and output sequences have different lengths and unknown alignment

## Example

```csharp
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;

var loss = new CTCLoss<float>();
var predicted = new Vector<float>(new[] { 0.9f, 0.2f, 0.7f });
var actual = new Vector<float>(new[] { 1.0f, 0.0f, 1.0f });

float value = loss.CalculateLoss(predicted, actual);
Console.WriteLine($"CTCLoss = {value:F4}");
```

