---
title: "SSLSession<T>"
description: "Manages a self-supervised learning training session."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SelfSupervisedLearning`

Manages a self-supervised learning training session.

## For Beginners

An SSL session manages the entire training lifecycle:
initialization, training loop, evaluation, and checkpointing. It provides
callbacks for monitoring progress and supports resuming from checkpoints.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SSLSession(ISSLMethod<>,SSLConfig)` | Initializes a new SSL training session. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CurrentEpoch` | Gets the current epoch number. |
| `GlobalStep` | Gets the global step counter. |
| `IsDistributed` | Gets whether this session is using distributed training. |
| `IsTraining` | Gets whether training is in progress. |
| `Method` | Gets the SSL method being used. |
| `Rank` | Gets the rank of this worker in distributed training. |
| `WorldSize` | Gets the world size (number of workers) for distributed training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CacheTrainingFeaturesForKNN(IEnumerable<Tensor<>>,Int32[])` | Caches training data for k-NN evaluation. |
| `ComputeKNNAccuracy(Tensor<>,Int32[],Int32)` | Computes k-NN accuracy on validation data using cached training features. |
| `CreateCommunicationBackend(SSLDistributedConfig)` | Creates the communication backend based on configuration. |
| `Evaluate(Tensor<>)` | Runs evaluation on the current encoder. |
| `FromCheckpoint(String,INeuralNetwork<>,Func<INeuralNetwork<>,ISSLMethod<>>)` | Creates a session from a pretrained checkpoint. |
| `GetEffectiveBatchSize` | Gets the effective batch size considering distributed training. |
| `GetHistory` | Gets the current training history. |
| `Reset` | Resets the session for a new training run. |
| `SaveCheckpoint(String)` | Saves a checkpoint to disk. |
| `Stop` | Stops training gracefully. |
| `SynchronizeGradients` | Synchronizes gradients across all distributed workers using AllReduce. |
| `SynchronizeParameters` | Synchronizes model parameters across all distributed workers. |
| `Train(Func<IEnumerable<Tensor<>>>,Tensor<>,Int32[])` | Trains the SSL method for the specified number of epochs. |
| `TrainEpoch(Func<IEnumerable<Tensor<>>>,Tensor<>,Int32[])` | Trains for a single epoch. |

## Events

| Event | Summary |
|:-----|:--------|
| `OnCollapseDetected` | Event raised when collapse is detected. |
| `OnEpochEnd` | Event raised at the end of each epoch. |
| `OnEpochStart` | Event raised at the start of each epoch. |
| `OnStepComplete` | Event raised after each training step. |

