---
title: "SSLResult<T>"
description: "Result from SSL pretraining containing the trained encoder and metrics."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SelfSupervisedLearning`

Result from SSL pretraining containing the trained encoder and metrics.

## For Beginners

After SSL pretraining, this result object contains
everything you need: the trained encoder, training history, and evaluation metrics.
You can use the encoder for downstream tasks or continue training.

## Properties

| Property | Summary |
|:-----|:--------|
| `BestEpoch` | Gets the epoch at which the best validation metric was achieved. |
| `BestValidationMetric` | Gets the best validation metric achieved during training. |
| `CheckpointPath` | Gets or sets the path to saved checkpoint (if saved). |
| `Config` | Gets or sets the training configuration used. |
| `Encoder` | Gets or sets the pretrained encoder network. |
| `EpochsTrained` | Gets or sets the number of epochs trained. |
| `ErrorMessage` | Gets or sets any error message if training failed. |
| `FinalMetrics` | Gets or sets the final SSL metrics. |
| `History` | Gets or sets the training history. |
| `IsSuccess` | Gets or sets whether training was successful. |
| `KNNAccuracy` | Gets or sets the k-NN evaluation accuracy (if performed). |
| `LinearEvaluation` | Gets or sets the linear evaluation result (if performed). |
| `Method` | Gets or sets the SSL method that was used for training. |
| `TrainingTimeSeconds` | Gets or sets the total training time in seconds. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Failure(String)` | Creates a failed SSL result. |
| `Success(INeuralNetwork<>,SSLMethodType,SSLConfig,SSLTrainingHistory<>)` | Creates a successful SSL result. |

