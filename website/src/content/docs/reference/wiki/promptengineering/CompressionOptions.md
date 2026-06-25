---
title: "CompressionOptions"
description: "Options for controlling prompt compression behavior."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.PromptEngineering.Compression`

Options for controlling prompt compression behavior.

## For Beginners

Settings for how to compress your prompt.

Example:

## How It Works

Configures how aggressive compression should be, what techniques to use,
and constraints on the output.

## Properties

| Property | Summary |
|:-----|:--------|
| `Aggressive` | Gets aggressive compression options for maximum reduction. |
| `Conservative` | Gets conservative compression options that preserve more content. |
| `Default` | Gets default compression options with moderate settings. |
| `MaxTokens` | Gets or sets the maximum number of tokens in the output. |
| `MinTokenCount` | Gets or sets the minimum number of tokens in the output. |
| `ModelName` | Gets or sets the model name for accurate token counting. |
| `PreserveCodeBlocks` | Gets or sets whether to preserve code blocks during compression. |
| `PreserveVariables` | Gets or sets whether to preserve template variables during compression. |
| `TargetReduction` | Gets or sets the target reduction ratio (0.0 to 1.0). |

