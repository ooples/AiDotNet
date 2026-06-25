---
title: "SCOREFramework<T>"
description: "SCORE: Selective Concept Obliteration for Responsible Editing in diffusion models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Safety`

SCORE: Selective Concept Obliteration for Responsible Editing in diffusion models.

## For Beginners

SCORE works like a content filter built into the model itself.
Instead of blocking outputs after generation, it changes the model so it can't
generate certain content in the first place. When the model encounters a prompt for
erased content, it automatically redirects to generating something neutral instead.

## How It Works

SCORE provides a structured framework for selectively removing unsafe or undesirable
concepts from diffusion models while preserving the model's overall generation quality.
It operates by fine-tuning cross-attention layers to redirect concept associations,
effectively "remapping" the erased concept to a neutral alternative.

Reference: Lu et al., "SCORE: Selective Concept Obliteration for Responsible Editing", 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SCOREFramework(Double,Double)` | Initializes a new SCORE framework instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `PreservationWeight` | Gets the preservation weight. |
| `RemappingCount` | Gets the number of registered concept remappings. |
| `RemappingStrength` | Gets the remapping strength. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddConceptRemapping(String,String)` | Registers a concept to be remapped to a neutral alternative. |
| `ComputeSCORELoss(Vector<>,Vector<>,Vector<>,Vector<>)` | Computes the SCORE training loss for cross-attention remapping. |
| `GetReplacement(String)` | Gets the neutral replacement for an erased concept. |
| `IsConceptErased(String)` | Checks whether a concept has a registered remapping. |

