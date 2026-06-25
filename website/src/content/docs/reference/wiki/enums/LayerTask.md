---
title: "LayerTask"
description: "Tasks or architectural roles that a neural network layer performs."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Tasks or architectural roles that a neural network layer performs.

## Fields

| Field | Summary |
|:-----|:--------|
| `ActivationNormalization` | Normalizing activations for training stability. |
| `AttentionComputation` | Computing attention scores and weighted sums. |
| `CrossModalAttention` | Cross-modal attention between different input types. |
| `DownSampling` | Reducing spatial/temporal resolution (pooling, strided convolution). |
| `FeatureExtraction` | Extracting features from input data (Conv, attention, embedding). |
| `FeatureFusion` | Combining multiple input streams (concat, add, multiply). |
| `GraphProcessing` | Processing graph-structured data (GCN, GAT, message passing). |
| `PositionalEncoding` | Encoding position information into representations. |
| `Projection` | Transforming between representation spaces (linear projection). |
| `Regularization` | Preventing overfitting (Dropout, BatchNorm, weight decay). |
| `Routing` | Routing information between capsules or experts. |
| `SequenceModeling` | Modeling sequential/temporal dependencies (RNN, LSTM, SSM, Transformer). |
| `SpatialProcessing` | Processing 2D spatial data (Conv2D, pooling, spatial attention). |
| `TemporalProcessing` | Processing temporal/time-series data (RNN, 1D conv, causal attention). |
| `UpSampling` | Increasing spatial/temporal resolution (upsample, transpose convolution). |
| `VolumetricProcessing` | Processing 3D volumetric data (Conv3D, 3D pooling). |

