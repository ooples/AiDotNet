---
title: "CNNBiLSTMCRF<T>"
description: "CNN-BiLSTM-CRF: Character CNN + Bidirectional LSTM + Conditional Random Field for Named Entity Recognition."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NER.SequenceLabeling`

CNN-BiLSTM-CRF: Character CNN + Bidirectional LSTM + Conditional Random Field for Named Entity Recognition.

## For Beginners

CNN-BiLSTM-CRF is one of the best-performing non-transformer NER models.
It works in three steps:

1. **Character CNN:** Looks at the letters within each word to detect patterns. For example,

it can notice that "Google" starts with a capital letter and "Inc." ends with a period,
which are clues that these might be parts of a company name.

2. **BiLSTM:** Reads the sentence both forwards and backwards to understand each word in

context. For example, "Apple" after "bought" is likely a company, but "Apple" after "ate" is
likely a fruit.

3. **CRF:** Makes sure the final labels are consistent. For example, if one word is labeled

as the start of a person name (B-PER), the next word should be either a continuation (I-PER)
or a new entity/non-entity, not the start of an organization.

This model is a good choice when:

- You need high accuracy without transformer-level compute costs
- Your text contains many rare or out-of-vocabulary words (names, technical terms)
- You need fast training and inference compared to BERT-based models

## How It Works

CNN-BiLSTM-CRF (Ma and Hovy, ACL 2016 - "End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF")
is a state-of-the-art sequence labeling architecture that extends BiLSTM-CRF with character-level CNN
embeddings. The model has three key components:

**1. Character-level CNN:**
A 1D Convolutional Neural Network that processes each word's character sequence to extract morphological
features. Unlike the character BiLSTM used in Lample et al. (2016), a CNN is:

- Faster: Processes all character positions in parallel (no sequential dependency)
- Better at local patterns: Captures character trigrams/n-grams like "-tion", "Dr.", "co-"
- Simpler: Fewer parameters and easier to train

The CNN operates as follows:
(a) Each character is mapped to a 30-dimensional embedding vector
(b) A bank of 30 convolutional filters with kernel size 3 slides over the character sequence
(c) Max-pooling over the character sequence produces a fixed-size 30-dimensional feature vector
(d) This character feature vector is concatenated with the word embedding

This captures features that word embeddings miss:

- Capitalization: "Apple" (entity) vs "apple" (fruit)
- Suffixes/prefixes: "-burg" (city), "-son" (person), "Dr." (title), "un-" (negation)
- Out-of-vocabulary words: Rare names recognized by character patterns
- Number patterns: "2023" recognized as a potential date

**2. Bidirectional LSTM (BiLSTM):**
Processes the concatenated [word_embedding; char_CNN_features] in both directions.
The forward LSTM reads left-to-right and the backward LSTM reads right-to-left.
Their hidden states are merged at each position via element-wise addition, giving each
token a context-aware representation informed by both preceding and following words.

**3. Conditional Random Field (CRF):**
Models label-level transition dependencies using a learned transition matrix. During inference,
the Viterbi algorithm finds the globally optimal label sequence that maximizes both emission
scores (from the BiLSTM) and transition scores (learned label-to-label preferences).

**Performance (CoNLL-2003):**

- CNN-BiLSTM-CRF achieves 91.21% F1 (Ma and Hovy, 2016)
- BiLSTM-CRF achieves 90.94% F1 (Lample et al., 2016)
- The character CNN provides a modest but consistent improvement over character BiLSTM

**Architecture diagram:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CNNBiLSTMCRF(NeuralNetworkArchitecture<>,CNNBiLSTMCRFOptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a CNN-BiLSTM-CRF model in native training mode with C# layers. |
| `CNNBiLSTMCRF(NeuralNetworkArchitecture<>,String,CNNBiLSTMCRFOptions)` | Creates a CNN-BiLSTM-CRF model in ONNX inference mode using a pre-trained model file. |

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
| `GetNamedLayerActivations(Tensor<>)` |  |
| `InitializeLayers` |  |
| `PostprocessOutput(Tensor<>)` |  |
| `PredictLabels(Tensor<>)` |  |
| `PreprocessTokens(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

