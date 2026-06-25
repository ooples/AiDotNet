---
title: "ConstellationNetAlgorithm<T, TInput, TOutput>"
description: "Implementation of ConstellationNet (structured part-based few-shot learning) (Xu et al., ICLR 2021)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of ConstellationNet (structured part-based few-shot learning) (Xu et al., ICLR 2021).

## For Beginners

ConstellationNet recognizes objects by their PARTS and ARRANGEMENT:

**The insight:**
A face isn't just "eyes + nose + mouth" - it's the specific ARRANGEMENT of these parts.
Similarly, a bird isn't just "beak + wing + tail" - it's how they're positioned.
ConstellationNet captures both the parts AND their spatial arrangement.

**How it works:**

1. **Part detection:** For each example, detect K discriminative parts
- Each part has a feature vector (what it looks like)
- Each part has a position (where it is)
2. **Constellation formation:** Model relationships between parts
- Pairwise spatial relationships between all parts
- Creates a "constellation" = graph of parts + spatial edges
3. **Constellation matching:** Compare query and support constellations
- Match parts between query and support (feature similarity)
- Compare spatial arrangements (structural similarity)
- Combined score determines classification

**Why constellations help:**
Two classes of birds might share similar colors but differ in proportions.
A constellation captures "beak-to-eye distance" and "wing-to-tail ratio" naturally.

## How It Works

ConstellationNet detects discriminative parts in examples and models their spatial
relationships as "constellations." Classification is performed by matching the
constellation structure between queries and support examples.

**Algorithm - ConstellationNet:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ConstellationNetAlgorithm(ConstellationNetOptions<,,>)` | Initializes a new ConstellationNet meta-learner. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `ComputeAuxLoss(TaskBatch<,,>)` | Computes the average loss over a task batch using part detection and constellation scoring. |
| `ComputeConstellationScores(Vector<>)` | Computes constellation scores: pairwise spatial relationships between detected parts. |
| `DetectParts(Vector<>)` | Detects K discriminative parts from features using attention-based selection. |
| `InitializePartDetector` | Initializes part detector and relation module parameters. |
| `MetaTrain(TaskBatch<,,>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_partDetectorParams` | Parameters for the part detection module. |
| `_relationParams` | Parameters for the spatial relation module. |

