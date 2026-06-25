---
title: "AutoMLTaskFamily"
description: "Defines the high-level task family for an AutoML run."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the high-level task family for an AutoML run.

## For Beginners

This is the kind of problem you're solving:

- `Regression` predicts numbers.
- `BinaryClassification` predicts one of two outcomes.
- `MultiClassClassification` predicts one of many outcomes.
- `TimeSeriesForecasting` predicts future values from past time-ordered values.
- `ReinforcementLearning` learns by interacting with an environment to maximize reward.

## How It Works

AutoML uses the task family to choose sensible defaults for metrics, evaluation protocols, and candidate selection.

## Fields

| Field | Summary |
|:-----|:--------|
| `BinaryClassification` | Binary (two-class) classification. |
| `DepthEstimation` | Depth estimation (predicting depth from 2D images). |
| `GraphClassification` | Graph (whole-graph) classification. |
| `GraphGeneration` | Graph generation. |
| `GraphLinkPrediction` | Graph link prediction. |
| `GraphNodeClassification` | Graph node classification. |
| `ImageClassification` | Image classification. |
| `ImageSegmentation` | Image segmentation. |
| `MeshClassification` | Mesh classification (classifying 3D mesh objects). |
| `MeshSegmentation` | Mesh segmentation (per-face or per-vertex classification). |
| `MultiClassClassification` | Multi-class (single-label) classification. |
| `MultiLabelClassification` | Multi-label classification (multiple labels can be true for one sample). |
| `ObjectDetection` | Object detection. |
| `PointCloudClassification` | Point cloud classification (classifying entire point clouds into categories). |
| `PointCloudCompletion` | Point cloud completion (reconstructing missing parts of point clouds). |
| `PointCloudSegmentation` | Point cloud segmentation (per-point classification/labeling). |
| `RadianceFieldReconstruction` | Neural radiance field reconstruction (novel view synthesis from images). |
| `Ranking` | Ranking (ordering items by relevance, e.g., search results). |
| `Recommendation` | Recommendation (ranking or scoring items for users). |
| `Regression` | Supervised regression (predicting continuous numeric values). |
| `ReinforcementLearning` | Reinforcement learning. |
| `SequenceTagging` | Sequence tagging (token-level labels like NER, POS). |
| `SpeechRecognition` | Speech recognition (ASR). |
| `TextClassification` | Text classification. |
| `TextGeneration` | Text generation (language modeling / free-form generation). |
| `ThreeDObjectDetection` | 3D object detection (detecting and localizing objects in 3D space). |
| `TimeSeriesAnomalyDetection` | Time-series anomaly detection (detecting rare/abnormal events in time-ordered data). |
| `TimeSeriesForecasting` | Time-series forecasting (predicting future values from past time-ordered values). |
| `Translation` | Machine translation. |
| `VolumetricClassification` | Volumetric classification (classifying 3D voxel grids). |
| `VolumetricSegmentation` | Volumetric segmentation (per-voxel classification). |

