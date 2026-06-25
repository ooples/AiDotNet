---
title: "BiLSTMCRF<T>"
description: "BiLSTM-CRF: Bidirectional LSTM with Conditional Random Field for Named Entity Recognition."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NER.SequenceLabeling`

BiLSTM-CRF: Bidirectional LSTM with Conditional Random Field for Named Entity Recognition.

## For Beginners

BiLSTM-CRF is the go-to model for finding names, places, organizations,
and other entities in text. Think of it as a two-step process:

**Step 1 - Reading with Context (BiLSTM):** The model reads each word while considering
the words around it. It reads the sentence both forward and backward, like reading a mystery
novel twice - once normally and once from the end. This gives each word a rich understanding
of its context. For example, "Apple" in "Apple Inc. announced..." gets a different
representation than "apple" in "She ate an apple."

**Step 2 - Making Consistent Labels (CRF):** After understanding each word's context,
the CRF ensures the labels make sense as a sequence. It's like a spell-checker for entity
labels. Without CRF, the model might label "Obama" as I-ORG after labeling "Barack" as
B-PER, which doesn't make sense. The CRF knows that I-ORG can't follow B-PER and would
correct this to "Barack"=B-PER, "Obama"=I-PER.

**Example:****When to use BiLSTM-CRF:**

- Named entity recognition (people, organizations, locations, etc.)
- Part-of-speech tagging
- Chunking and shallow parsing
- Any sequence labeling task where label dependencies matter

**Key advantage over simpler approaches:** The CRF layer typically improves F1 score
by 1-2% over independent per-token classification, and more importantly, it guarantees
that the output label sequence is always structurally valid.

## How It Works

BiLSTM-CRF (Huang et al., 2015; Lample et al., NAACL 2016) is the foundational neural architecture
for Named Entity Recognition that combines two powerful, complementary components into a unified model:

**1. Bidirectional LSTM (BiLSTM):**
A pair of Long Short-Term Memory networks that process the token sequence in both directions simultaneously.
The forward LSTM reads left-to-right (e.g., "Barack" -> "Obama" -> "was" -> "born" -> "in" -> "Honolulu"),
capturing preceding context for each token. The backward LSTM reads right-to-left, capturing following
context. Their hidden states are merged at each position, giving the model a complete view of the
entire sentence context around every token. This bidirectional context is crucial for NER because entity
recognition often depends on both left and right context (e.g., "Washington" could be a person, location,
or organization depending on surrounding words).

**2. Conditional Random Field (CRF):**
A structured prediction layer that models label-level transition dependencies to produce globally optimal
label sequences. While the BiLSTM produces per-token scores for each possible label (called "emission
scores"), the CRF adds a learned transition matrix that captures which label transitions are valid.
For example, it learns that I-PER (Inside Person) can only follow B-PER (Begin Person) or I-PER,
never B-ORG (Begin Organization). During inference, the Viterbi algorithm efficiently finds the
highest-scoring label sequence that respects all transition constraints.

**Architecture Pipeline:**

- Token embeddings (e.g., 100d GloVe vectors) are fed into the BiLSTM encoder
- BiLSTM produces contextualized hidden states for each token position
- A linear projection maps hidden states to emission scores (one score per label per token)
- Dropout regularization prevents overfitting during training
- CRF layer combines emission scores with learned transition scores to decode the optimal label sequence

**Mathematical Formulation:**
For input sequence x = (x_1, ..., x_n) and label sequence y = (y_1, ..., y_n):

- Emission score: e(x_t, y_t) = W * h_t + b, where h_t is the BiLSTM hidden state at position t
- Transition score: T[y_{t-1}, y_t] from the learned CRF transition matrix
- Sequence score: S(x, y) = SUM_t [e(x_t, y_t) + T[y_{t-1}, y_t]]
- Training objective: maximize log P(y|x) = S(x, y) - log(SUM_{y'} exp(S(x, y')))
- Inference: y* = argmax_y S(x, y), solved efficiently by Viterbi algorithm in O(n * L^2) time

This model serves as the golden example for the NER model family in AiDotNet, establishing the
architectural patterns (base class hierarchy, layer initialization via LayerHelper, dual ONNX/native
mode, serialization, etc.) that all other NER models follow.

**References:**

- "Bidirectional LSTM-CRF Models for Sequence Tagging" (Huang, Xu, Yu, 2015)
- "Neural Architectures for Named Entity Recognition" (Lample et al., NAACL 2016)
- "End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF" (Ma and Hovy, ACL 2016)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BiLSTMCRF(NeuralNetworkArchitecture<>,BiLSTMCRFOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a BiLSTM-CRF model in native training mode with full forward/backward pass support. |
| `BiLSTMCRF(NeuralNetworkArchitecture<>,String,BiLSTMCRFOptions)` | Creates a BiLSTM-CRF model in ONNX inference mode, loading a pre-trained model from disk. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AiDotNet#NER#Interfaces#INERModel{T}#EmbeddingDimension` | Gets the dimensionality of input token embeddings expected by this model. |
| `AiDotNet#NER#Interfaces#INERModel{T}#NumLabels` | Gets the number of BIO labels this model can predict. |
| `ExpectedInputShape` | Gets the expected input tensor shape as [maxSequenceLength, embeddingDimension]. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AiDotNet#NER#Interfaces#INERModel{T}#GetModelSummary` | Returns a human-readable summary of the model's architecture and configuration. |
| `AiDotNet#NER#Interfaces#INERModel{T}#PredictBatch(IEnumerable<Tensor<>>)` | Predicts entity labels for multiple input sequences in batch, yielding results lazily. |
| `AiDotNet#NER#Interfaces#INERModel{T}#TrainAsync(Tensor<>,Tensor<>,Int32,IProgress<NERTrainingProgress>,CancellationToken)` | Trains the BiLSTM-CRF model asynchronously over multiple epochs with progress reporting. |
| `AiDotNet#NER#Interfaces#INERModel{T}#ValidateInputShape(Tensor<>)` | Validates that an input tensor has the correct shape, embedding dimension, and sequence length for this model. |
| `ComputeEmissionScores(Tensor<>)` | Computes emission scores from token embeddings by running the BiLSTM and linear projection layers, stopping before the CRF layer. |
| `CreateNewInstance` | Creates a new, uninitialized instance of this model with the same configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes model-specific data from a binary stream, restoring the model to its saved state. |
| `Dispose(Boolean)` | Releases resources held by the model, including ONNX runtime sessions and native layers. |
| `GetModelMetadata` | Returns metadata describing this model for serialization, logging, and model registry purposes. |
| `GetNamedLayerActivations(Tensor<>)` |  |
| `GetOptions` | Returns the model's configuration options. |
| `InitializeLayers` | Initializes the BiLSTM-CRF layer stack using research-paper-validated defaults from `Boolean)`. |
| `PostprocessOutput(Tensor<>)` | Postprocesses the model output by applying argmax decoding to produce label indices. |
| `PredictLabels(Tensor<>)` | Predicts the optimal BIO label sequence for input token embeddings using the full BiLSTM-CRF pipeline (or ONNX inference if in ONNX mode). |
| `PreprocessTokens(Tensor<>)` | Preprocesses token embeddings by padding or truncating to MaxSequenceLength before feeding them to the BiLSTM. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes model-specific data to a binary stream for saving model checkpoints. |
| `ThrowIfDisposed` | Throws `ObjectDisposedException` if this model has been disposed. |
| `Train(Tensor<>,Tensor<>)` | Performs one training step: forward pass, loss computation, backward pass, and parameter update. |
| `UpdateParameters(Vector<>)` | Updates all model parameters from a flat parameter vector. |
| `ValidateOptions` | Validates the options for consistency and supported feature combinations. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_disposed` | Whether this model instance has been disposed and should no longer be used. |
| `_optimizer` | The gradient-based optimizer used for updating model weights during training. |
| `_options` | The configuration options controlling all aspects of the BiLSTM-CRF architecture, including embedding dimensions, hidden sizes, number of LSTM layers, dropout rates, and CRF settings. |
| `_useNativeMode` | Whether the model operates in native training mode (true) or ONNX inference-only mode (false). |

