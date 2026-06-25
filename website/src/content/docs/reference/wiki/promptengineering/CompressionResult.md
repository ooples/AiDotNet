---
title: "CompressionResult"
description: "Contains the result of a prompt compression operation including metrics."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PromptEngineering.Compression`

Contains the result of a prompt compression operation including metrics.

## For Beginners

This is the result of compressing a prompt, with before/after stats.

When you compress a prompt, you want to know:

- What's the compressed text?
- How much shorter is it?
- How much money did we save?

Example:

## How It Works

This class encapsulates the compressed prompt along with detailed metrics
about the compression operation, including token counts, compression ratio,
and estimated cost savings.

## Properties

| Property | Summary |
|:-----|:--------|
| `CompressedAt` | Gets or sets the timestamp when compression was performed. |
| `CompressedPrompt` | Gets or sets the compressed prompt. |
| `CompressedTokenCount` | Gets or sets the token count of the compressed prompt. |
| `CompressionMethod` | Gets or sets the compression method used. |
| `CompressionRatio` | Gets the compression ratio (0.0 to 1.0, where higher means more compression). |
| `EstimatedCostSavings` | Gets or sets the estimated cost savings in USD. |
| `IsSuccessful` | Gets whether the compression was successful (reduced token count). |
| `OriginalPrompt` | Gets or sets the original prompt before compression. |
| `OriginalTokenCount` | Gets or sets the token count of the original prompt. |
| `TokensSaved` | Gets the number of tokens saved by compression. |
| `Warnings` | Gets or sets any warnings generated during compression. |

