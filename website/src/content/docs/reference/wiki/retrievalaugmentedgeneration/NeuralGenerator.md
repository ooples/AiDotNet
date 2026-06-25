---
title: "NeuralGenerator<T>"
description: "Neural network-based text generator for RAG systems using LSTM architecture."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Generators`

Neural network-based text generator for RAG systems using LSTM architecture.

## For Beginners

This generator uses a trained neural network to create answers.

Think of it like a smart writer:

- Takes your question and retrieved documents
- Converts text into numbers (tokens) the network understands
- Uses an LSTM neural network to predict next words
- Samples from probability distributions with temperature control
- Converts numbers back to readable text

How it works:

1. Tokenizes input text into numerical IDs
2. Feeds tokens through LSTM network layer-by-layer
3. Network outputs probability distribution over vocabulary
4. Samples next token using temperature (higher = more creative)
5. Repeats until generating enough tokens or reaching end token
6. Detokenizes back to human-readable text

Production considerations:

- Requires pre-trained LSTM network (not included)
- Actual LSTM forward pass for each token (computationally intensive)
- Temperature-based sampling for controlled randomness
- Configurable context and generation limits
- Proper error handling and edge cases
- Memory-efficient sequential processing

Note: This generator requires a pre-trained LSTM network with vocabulary matching
the configured vocabulary size. Without proper training, output will be meaningless.
For production use, you must:

1. Train the LSTM on your domain data, OR
2. Use transfer learning from a pre-trained language model

For production RAG systems, consider using LLM-based generators with pre-trained
models instead of training your own LSTM.

## How It Works

This generator uses an LSTM neural network for token-by-token text generation in
retrieval-augmented generation tasks. It processes context through the trained LSTM
and generates responses using temperature-based sampling from the network's probability
distributions.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NeuralGenerator(LSTMNeuralNetwork<>,Int32,Int32,Int32,Int32,Double,IDictionary<String,Int32>)` | Initializes a new instance of the NeuralGenerator class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxContextTokens` | Gets the maximum number of tokens this generator can process in a single request. |
| `MaxGenerationTokens` | Gets the maximum number of tokens this generator can generate in a response. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Generate(String)` | Generates a text response based on a prompt. |
| `GenerateGrounded(String,IEnumerable<Document<>>)` | Generates a grounded answer using provided context documents. |

