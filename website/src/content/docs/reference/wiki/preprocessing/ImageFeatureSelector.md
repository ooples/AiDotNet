---
title: "ImageFeatureSelector<T>"
description: "Feature selection for image data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.DomainSpecific`

Feature selection for image data.

## For Beginners

Images have special structure - nearby pixels are usually
similar. This selector understands that and picks features while considering their
spatial relationships. It's like choosing representative spots on a photo rather
than random pixels.

## How It Works

ImageFeatureSelector is designed for selecting features from image data,
considering spatial locality and the correlation patterns typical of image features.
It can handle flattened image data and respects the original image structure.

