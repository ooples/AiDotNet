---
title: "LSTMCRF<T>"
description: "LSTM-CRF: Unidirectional LSTM with Conditional Random Field for Named Entity Recognition."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NER.SequenceLabeling`

LSTM-CRF: Unidirectional LSTM with Conditional Random Field for Named Entity Recognition.

## For Beginners

LSTM-CRF is a faster but slightly less accurate version of BiLSTM-CRF.
Instead of reading the sentence both forwards and backwards, it only reads forwards (left to right).

Think of it like reading a mystery novel: BiLSTM-CRF reads the whole book before deciding who
the suspects are, while LSTM-CRF identifies suspects as it reads, without looking ahead.
Both are good at their job, but having the full picture (BiLSTM) helps with tricky cases.

Use this model when speed matters more than getting every last bit of accuracy, such as:

- Processing live chat messages for entity extraction
- Real-time speech-to-text NER (identifying entities as words are spoken)
- Edge/mobile deployment where compute is limited

## How It Works

LSTM-CRF (Huang, Xu, and Yu, 2015 - "Bidirectional LSTM-CRF Models for Sequence Tagging")
is a simpler variant of BiLSTM-CRF that uses a unidirectional (left-to-right) LSTM encoder.
While the original paper proposed both unidirectional and bidirectional variants, this class
implements the unidirectional version for scenarios requiring lower latency or streaming inference.

**Architecture:****Key Differences from BiLSTM-CRF:**

- Only processes text left-to-right (no backward pass)
- Each token's representation only captures preceding context (not following context)
- ~50% fewer LSTM parameters (one direction instead of two)
- ~2x faster inference per token
- 1-2% lower F1 score compared to BiLSTM-CRF

**When to Use LSTM-CRF vs BiLSTM-CRF:**

- **LSTM-CRF:** Real-time/streaming NER, edge deployment, latency-sensitive applications
- **BiLSTM-CRF:** Offline processing, maximum accuracy, standard NER benchmarks

The CRF layer is particularly important for LSTM-CRF because it partially compensates for the
lack of right-context. Without the CRF, the unidirectional LSTM would have no way to enforce
constraints like "I-PER can only follow B-PER or I-PER", which depend on future label decisions.
The CRF's transition matrix captures these patterns, effectively providing a form of right-context
at the label level.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LSTMCRF(NeuralNetworkArchitecture<>,LSTMCRFOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates an LSTM-CRF model in native training mode with C# layers. |
| `LSTMCRF(NeuralNetworkArchitecture<>,String,LSTMCRFOptions)` | Creates an LSTM-CRF model in ONNX inference mode using a pre-trained model file. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ExpectedInputShape` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AiDotNet#NER#Interfaces#INERModel{T}#GetModelSummary` |  |
| `AiDotNet#NER#Interfaces#INERModel{T}#PredictBatch(IEnumerable<Tensor<>>)` |  |
| `AiDotNet#NER#Interfaces#INERModel{T}#TrainAsync(Tensor<>,Tensor<>,Int32,IProgress<NERTrainingProgress>,CancellationToken)` |  |
| `AiDotNet#NER#Interfaces#INERModel{T}#ValidateInputShape(Tensor<>)` |  |
| `ComputeEmissionScores(Tensor<>)` |  |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Dispose(Boolean)` |  |
| `GetModelMetadata` |  |
| `InitializeLayers` |  |
| `PostprocessOutput(Tensor<>)` |  |
| `PredictLabels(Tensor<>)` |  |
| `PreprocessTokens(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

