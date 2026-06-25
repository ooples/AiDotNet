---
title: "ObjectDetectorBase<T>"
description: "Base class for all object detection models."
section: "API Reference"
---

`Base Classes` · `AiDotNet.ComputerVision.Detection.ObjectDetection`

Base class for all object detection models.

## For Beginners

An object detector takes an image and finds all objects in it,
returning their locations (bounding boxes), types (class labels), and confidence scores.
This base class provides the common structure and methods that all detection models share.

## How It Works

A typical detector has three parts:

- Backbone: Extracts features from the image
- Neck: Combines features at multiple scales
- Head: Produces final predictions (boxes, classes, scores)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ObjectDetectorBase(ObjectDetectionOptions<>)` | Creates a new object detector with the specified options. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Backbone` | The backbone network for feature extraction. |
| `ClassNames` | Class names for detection labels. |
| `DefaultLossFunction` |  |
| `EnsureBackbone` | Gets the backbone network, throwing if not initialized. |
| `EnsureNeck` | Gets the neck module, throwing if not initialized. |
| `Name` | Name of this detector architecture. |
| `Neck` | The neck module for feature fusion. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DeepCopy` |  |
| `Detect(Tensor<>)` | Detects objects in an image. |
| `Detect(Tensor<>,Double,Double)` | Detects objects in an image with custom thresholds. |
| `DetectBatch(Tensor<>)` | Detects objects in a batch of images. |
| `DetectBatch(Tensor<>,Double,Double)` | Detects objects in a batch of images with custom thresholds. |
| `ExtractBatchItem(Tensor<>,Int32)` | Extracts a single image from a batch. |
| `ExtractBatchOutputs(List<Tensor<>>,Int32)` | Extracts the outputs for a single batch item from batch outputs. |
| `Forward(Tensor<>)` | Performs forward pass through the network. |
| `GetCocoClassNames` | Gets the default COCO class names. |
| `GetHeadParameterCount` | Gets the number of parameters in the detection head. |
| `GetParameterCount` | Gets the total number of parameters in the model. |
| `GetParameters` |  |
| `LoadPretrainedWeightsAsync(CancellationToken)` | Loads default pre-trained weights for this architecture and size. |
| `LoadWeightsAsync(String,CancellationToken)` | Loads pre-trained weights from a file or URL. |
| `Normalize(Tensor<>)` | Normalizes image values to [0, 1] range. |
| `PostProcess(List<Tensor<>>,Int32,Int32,Double,Double)` | Post-processes raw network outputs into detections. |
| `Predict(Tensor<>)` | Predicts by running the forward pass and returning raw network outputs concatenated. |
| `Preprocess(Tensor<>)` | Preprocesses an image for input to the network. |
| `ResizeImage(Tensor<>,Int32,Int32)` | Resizes an image tensor to the specified dimensions. |
| `SaveWeights(String)` | Saves model weights to a file. |
| `SetParameters(Vector<>)` |  |
| `SetTrainingMode(Boolean)` | Sets the model to training or inference mode. |
| `Train(Tensor<>,Tensor<>)` | Training object detectors requires specialized loss. |
| `WithParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `IsTrainingMode` | Whether the model is in training mode. |
| `Nms` | NMS algorithm for removing duplicate detections. |
| `Options` | Configuration options for this detector. |
| `WeightDownloader` | Weight downloader for fetching pre-trained weights. |

