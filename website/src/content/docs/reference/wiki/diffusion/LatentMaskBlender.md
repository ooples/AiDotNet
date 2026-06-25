---
title: "LatentMaskBlender<T>"
description: "Blends latent representations using a mask for seamless inpainting and region editing."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.MaskUtilities`

Blends latent representations using a mask for seamless inpainting and region editing.

## For Beginners

When editing just part of an image (like replacing a face or
removing an object), you need to smoothly blend the new content with the original.
LatentMaskBlender does this in the model's internal representation, mixing old and new
content according to a mask that says "use new content here, keep old content there."

## How It Works

Performs per-element blending of two latent tensors using a mask that defines the mixing
ratio at each spatial location. Operates in latent space (after VAE encoding) for
efficient inpainting. Supports both hard (binary) and soft (feathered) masks for
smooth transitions between inpainted and original regions.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LatentMaskBlender(Double)` | Initializes a new latent mask blender. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BlendSharpness` | Gets the blend sharpness (higher = sharper mask transitions). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Blend(Vector<>,Vector<>,Vector<>)` | Blends two latent vectors using a mask. |
| `BlendWithSchedule(Vector<>,Vector<>,Vector<>,Double)` | Blends with noise-aware scheduling for diffusion inpainting. |

