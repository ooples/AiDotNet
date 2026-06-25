---
title: "JaccardScoreMetric<T>"
description: "Computes Jaccard Score (Intersection over Union): measures similarity between prediction and actual sets."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Classification`

Computes Jaccard Score (Intersection over Union): measures similarity between prediction and actual sets.

## For Beginners

Jaccard score measures how much overlap exists between predicted positives
and actual positives. Perfect score of 1 means complete overlap; 0 means no overlap at all.
Also known as the Jaccard Index or IoU (Intersection over Union) in object detection.

## How It Works

Jaccard = |TP| / (|TP| + |FP| + |FN|) = Intersection / Union

