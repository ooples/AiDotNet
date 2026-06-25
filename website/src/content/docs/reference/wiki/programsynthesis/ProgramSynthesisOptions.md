---
title: "ProgramSynthesisOptions"
description: "Configuration options for enabling Program Synthesis / Code Tasks via the primary builder/result APIs."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.ProgramSynthesis.Options`

Configuration options for enabling Program Synthesis / Code Tasks via the primary builder/result APIs.

## For Beginners

These settings control which built-in code model to use and how big it is.
You can usually accept the defaults and only change the language and model kind.

## How It Works

This options type is intended for use with `AiModelBuilder` to configure a code model and
safe defaults (tokenization and architecture) without requiring users to construct low-level engines.

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultTask` | Gets or sets the default code task type used for the configured architecture. |
| `MaxSequenceLength` | Gets or sets the maximum sequence length (in tokens) that the model can process. |
| `ModelKind` | Gets or sets the built-in code model kind to configure when no explicit model is provided. |
| `NumDecoderLayers` | Gets or sets the number of decoder layers used by the configured architecture. |
| `NumEncoderLayers` | Gets or sets the number of encoder layers used by the configured architecture. |
| `SynthesisType` | Gets or sets the synthesis approach type used for the configured architecture. |
| `TargetLanguage` | Gets or sets the target programming language for the configured model. |
| `Tokenizer` | Gets or sets an optional tokenizer to use for code tasks. |
| `VocabularySize` | Gets or sets the vocabulary size used by the configured architecture. |

