---
title: "ChronosFoundationModel<T>"
description: "Implements the Chronos foundation model for zero-shot time series forecasting."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TimeSeries`

Implements the Chronos foundation model for zero-shot time series forecasting.

## For Beginners

Imagine you've read thousands of different books about weather,
stock prices, store sales, and website traffic. After reading all these, you develop an
intuition for how numbers change over time. When someone shows you a new sequence of numbers
you've never seen, you can make educated guesses about what comes next.

Chronos does exactly this but with neural networks. It "reads" millions of time series during
training and learns patterns. Then it can forecast new time series without being specifically
trained on that type of data. This is incredibly powerful for real-world applications where
you might not have enough historical data to train a specialized model.

## How It Works

**What is a Foundation Model?**
A foundation model is a large neural network pretrained on vast amounts of data that can be
applied to new tasks without task-specific training (zero-shot) or with minimal fine-tuning.
GPT-3/4 are foundation models for text; Chronos is a foundation model for time series.

**The Chronos Approach:**
Chronos (Ansari et al., 2024) treats time series forecasting as a language modeling task.
The key insight is that if we can tokenize continuous time series values into discrete
tokens, we can apply the same powerful transformer architectures that work so well for text.

**Mean-Scaling Tokenization:**
Before tokenization, values are normalized by the mean absolute value of the context:
x_normalized = x / (mean(|context|) + epsilon)
This makes the model scale-invariant - it can handle time series of any magnitude.
Normalized values are then mapped to discrete tokens using a fixed vocabulary of
uniformly-spaced bins covering a reasonable range (e.g., -15 to 15).

**Causal Transformer Architecture:**
Chronos uses a decoder-only transformer (like GPT) with causal masking. Each position
can only attend to itself and previous positions, enabling autoregressive generation.
The architecture includes:

- Token embeddings mapping discrete tokens to dense vectors
- Sinusoidal positional encoding for temporal awareness
- Multiple transformer layers with multi-head causal self-attention
- Layer normalization and feed-forward networks
- Output projection to vocabulary logits

**Zero-Shot Forecasting:**
Once pretrained on diverse time series data (synthetic and real), Chronos can forecast
new time series it has never seen. The model learns general patterns of temporal dynamics
that transfer across domains - seasonality, trends, noise patterns, etc.

## Example

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.TimeSeries;
using AiDotNet.Tensors.LinearAlgebra;

double[] series =
{
    120, 135, 148, 160, 155, 170, 180, 195, 210, 198, 220, 235,
    140, 155, 165, 178, 172, 190, 200, 215, 230, 218, 245, 260
};
var x = new Matrix<double>(series.Length, 1);
for (int i = 0; i < series.Length; i++) x[i, 0] = i;

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new ChronosFoundationModel<double>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(x, new Vector<double>(series)))
    .BuildAsync();

var forecast = result.Predict(new Matrix<double>(6, 1));
Console.WriteLine($"ChronosFoundationModel: forecast {forecast.Length} steps.");
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ChronosFoundationModel(ChronosOptions<>)` | Initializes a new instance of the Chronos foundation model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeGradients(Vector<>,)` | Computes gradients using backpropagation through the entire network. |
| `ForecastWithQuantiles(Vector<>,Double[],Int32)` | Generates probabilistic forecasts by sampling from the model. |
| `Predict(Matrix<>)` | Predicts the next value in a time series. |
| `TrainCore(Matrix<>,Vector<>)` | Trains the Chronos model using proper backpropagation through all parameters. |

