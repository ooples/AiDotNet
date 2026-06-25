---
title: "FeatureTokenizer<T>"
description: "Implements the Feature Tokenizer that converts tabular features into embeddings for FT-Transformer."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

Implements the Feature Tokenizer that converts tabular features into embeddings for FT-Transformer.

## For Beginners

The Feature Tokenizer is like a translator that converts your
spreadsheet columns into a format the Transformer can understand.

What it does:

1. Takes each column value and converts it to a vector (list of numbers)
2. For numbers (like age=25): Multiplies by learned weights and adds bias
3. For categories (like color="red"): Looks up a learned embedding vector
4. Adds a special [CLS] token that will aggregate all feature information

Why this matters:

- Transformers work on sequences of vectors, not raw numbers
- This conversion allows the model to learn rich representations of each feature
- The [CLS] token provides a single representation for the final prediction

Example: If you have 5 features with embedding dimension 64:
Input: [age, income, zip_code, gender, marital_status]
Output: Tensor of shape [batch, 6, 64] (5 features + 1 CLS token, each as 64-dim vector)

## How It Works

The Feature Tokenizer converts each input feature into a d-dimensional embedding:

- For numerical features: embedding = x * w + b (linear projection)
- For categorical features: embedding lookup from learned embedding table

A learnable [CLS] token is prepended to the sequence for classification/regression.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FeatureTokenizer(Int32,Int32,Int32[],Boolean,Double)` | Initializes a new instance of the FeatureTokenizer class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbeddingDimension` | Gets the embedding dimension. |
| `ParameterCount` | Gets the total number of trainable parameters. |
| `SequenceLength` | Gets the sequence length including [CLS] token. |
| `TotalFeatures` | Gets the total number of features (numerical + categorical). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` | Performs the forward pass with numerical features only. |
| `Forward(Tensor<>,Matrix<Int32>)` | Performs the forward pass to tokenize features into embeddings. |
| `GetClsToken` | Gets the [CLS] token embedding. |
| `GetNumericalBias` | Gets the numerical feature biases. |
| `GetNumericalWeights` | Gets the numerical feature weights. |
| `GetParameterGradients` | Gets the parameter gradients as a single vector. |
| `GetParameters` | Gets all trainable parameters as a single vector. |
| `GetTrainableTensors` | Returns the tokenizer's trainable weight TENSORS (the live field instances, not a flat copy) so a caller can register them with a `GradientTape` and apply gradients in place. |
| `InitializeXavier(Tensor<>,Double)` | Initializes a tensor with Xavier/Glorot initialization. |
| `ResetGradients` | Resets all gradients to zero. |
| `SetParameters(Vector<>)` | Sets the trainable parameters from a vector. |
| `UpdateParameters()` | Updates parameters using the calculated gradients. |

