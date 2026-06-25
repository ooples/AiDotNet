---
title: "WeightMapping"
description: "Maps weight names between different model formats."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ModelLoading`

Maps weight names between different model formats.

## For Beginners

Model weights have names like paths in a file system.

For example, in Stable Diffusion:

- HuggingFace: "model.diffusion_model.input_blocks.0.0.weight"
- Our model: "unet.inputConv.weight"

This class translates between these naming conventions so we can
load weights from any source into our models.

## How It Works

Different ML frameworks and model releases use different naming conventions.
This class provides mappings to translate between them.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WeightMapping(Dictionary<String,String>)` | Initializes a new instance with custom mappings. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DirectMappingCount` | Gets the count of direct mappings. |
| `DirectMappings` | Gets all direct mappings. |
| `PatternMappingCount` | Gets the count of pattern mappings. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddMapping(String,String)` | Adds a direct name mapping. |
| `AddPatternMapping(String,String)` | Adds a pattern-based mapping. |
| `CreateCLIPTextEncoder` | Creates weight mapping for CLIP text encoder. |
| `CreateSDXLVAE` | Creates weight mapping for SDXL VAE. |
| `CreateStableDiffusionV1UNet` | Creates weight mapping for Stable Diffusion v1.x UNet. |
| `CreateStableDiffusionV1VAE` | Creates weight mapping for Stable Diffusion v1.x VAE. |
| `Map(String)` | Maps a source weight name to the target name. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_mappings` | Direct name-to-name mappings. |
| `_patternMappings` | Pattern-based mappings (regex -> replacement). |

