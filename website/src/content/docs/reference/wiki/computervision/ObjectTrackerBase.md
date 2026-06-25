---
title: "ObjectTrackerBase<T>"
description: "Base class for object trackers."
section: "API Reference"
---

`Base Classes` · `AiDotNet.ComputerVision.Tracking`

Base class for object trackers.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ObjectTrackerBase(TrackingOptions<>)` | Creates a new object tracker. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Name of this tracker. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeCosineSimilarity(Tensor<>,Tensor<>)` | Computes cosine similarity between two feature vectors. |
| `ComputeIoU(BoundingBox<>,BoundingBox<>)` | Computes IoU between two bounding boxes. |
| `CreateTrack(Detection<>)` | Creates a new track from detection. |
| `GetConfirmedTracks` | Gets confirmed tracks only. |
| `HungarianAssignment(Double[0:,0:])` | Solves linear assignment problem (Hungarian algorithm). |
| `PredictNextPosition(Track<>)` | Predicts track position for next frame using velocity. |
| `Reset` | Resets the tracker state. |
| `Update(List<Detection<>>)` | Updates tracks with new detections. |
| `Update(List<Detection<>>,Tensor<>)` | Updates tracks with new detections and image for appearance features. |
| `UpdateTrack(Track<>,Detection<>)` | Updates a track with a matched detection. |

