---
title: "ImageSafetyConfig"
description: "Configuration for image safety classification modules."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Safety.Image`

Configuration for image safety classification modules.

## For Beginners

Use this to configure which types of harmful image content
to detect and how strict the detection should be.

## Properties

| Property | Summary |
|:-----|:--------|
| `ClassifierType` | Classifier type to use. |
| `CsamDetection` | Whether to detect CSAM content. |
| `NsfwThreshold` | NSFW detection threshold (0.0-1.0). |
| `ViolenceThreshold` | Violence detection threshold (0.0-1.0). |

