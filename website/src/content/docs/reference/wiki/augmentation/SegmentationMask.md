---
title: "SegmentationMask<T>"
description: "Represents a segmentation mask for pixel-level annotations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation.Image`

Represents a segmentation mask for pixel-level annotations.

## For Beginners

While a bounding box draws a rectangle around an object,
a segmentation mask precisely outlines the object's shape. When you rotate or
flip an image, the mask must be transformed identically.

## How It Works

Segmentation masks provide pixel-level object delineation, used for:

- Semantic segmentation (what class is each pixel)
- Instance segmentation (which object instance is each pixel)
- Panoptic segmentation (both semantic and instance)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SegmentationMask` | Creates an empty segmentation mask. |
| `SegmentationMask(IList<IList<ValueTuple<Double,Double>>>,Int32,Int32,MaskType)` | Creates a segmentation mask from polygon vertices. |
| `SegmentationMask(Int32[],Int32,Int32,MaskType)` | Creates a segmentation mask from RLE encoding. |
| `SegmentationMask([0:,0:],MaskType,Int32)` | Creates a segmentation mask from dense data. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ClassIndex` | Gets or sets the class index (for semantic/instance masks). |
| `ClassName` | Gets or sets the class name. |
| `Confidence` | Gets or sets the confidence score (if from a segmenter). |
| `Encoding` | Gets or sets the current encoding format. |
| `Height` | Gets or sets the mask height. |
| `InstanceId` | Gets or sets the instance ID (for instance/panoptic masks). |
| `MaskData` | Gets or sets the mask data as a 2D array [height, width]. |
| `Metadata` | Gets or sets additional metadata. |
| `Polygons` | Gets or sets the polygon vertices (if using polygon encoding). |
| `RLECounts` | Gets or sets the RLE-encoded mask (if using RLE encoding). |
| `Type` | Gets or sets the mask type. |
| `Width` | Gets or sets the mask width. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Area` | Calculates the area (number of foreground pixels). |
| `Clone` | Creates a deep copy of this mask. |
| `DecodeRLE` | Decodes RLE to dense format. |
| `Dice(SegmentationMask<>)` | Calculates the Dice coefficient with another mask. |
| `EncodeRLE([0:,0:])` | Encodes dense mask to RLE format. |
| `FillPolygon([0:,0:],IList<ValueTuple<Double,Double>>)` | Fills a polygon using scanline algorithm. |
| `GetBoundingBox` | Calculates the bounding box of the mask. |
| `IoU(SegmentationMask<>)` | Calculates the IoU (Intersection over Union) with another mask. |
| `RasterizePolygons` | Rasterizes polygons to dense format. |
| `ToDense` | Converts this mask to dense format. |
| `ToRLE` | Converts this mask to RLE format. |

