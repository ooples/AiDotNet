---
title: "Track<T>"
description: "Represents a tracked object across frames."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Tracking`

Represents a tracked object across frames.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Track(Int32,BoundingBox<>,Int32,)` | Creates a new track. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Age` | Total number of frames this track has been active. |
| `AppearanceFeature` | Optional appearance feature embedding. |
| `Box` | Current bounding box. |
| `ClassId` | Class ID of the tracked object. |
| `Confidence` | Current confidence score. |
| `Hits` | Number of frames with detection hits. |
| `State` | Track state (tentative, confirmed, deleted). |
| `TimeSinceUpdate` | Number of consecutive frames without detection. |
| `TrackId` | Unique track identifier. |
| `VelocityX` | Velocity in x direction (pixels/frame). |
| `VelocityY` | Velocity in y direction (pixels/frame). |

