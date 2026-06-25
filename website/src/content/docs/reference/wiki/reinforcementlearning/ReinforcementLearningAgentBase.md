---
title: "ReinforcementLearningAgentBase<T>"
description: "Base class for all reinforcement learning agents, providing common functionality and structure."
section: "API Reference"
---

`Base Classes` · `AiDotNet.ReinforcementLearning.Agents`

Base class for all reinforcement learning agents, providing common functionality and structure.

## For Beginners

This is the foundation for all RL agents in AiDotNet.

Think of this base class as the blueprint that defines what every RL agent must be able to do:

- Select actions based on observations
- Store experiences for learning
- Train/update from experiences
- Save and load trained models
- Integrate with AiDotNet's neural networks and optimizers

All specific RL algorithms (DQN, PPO, SAC, etc.) inherit from this base and implement
their own unique learning logic while sharing common functionality.

## How It Works

This abstract base class defines the core structure that all RL agents must follow, ensuring
consistency across different RL algorithms while allowing for specialized implementations.
It integrates deeply with AiDotNet's existing architecture, using Vector, Matrix, and Tensor types,
and following established patterns like OptimizerBase and NeuralNetworkBase.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ReinforcementLearningAgentBase(ReinforcementLearningOptions<>)` | Initializes a new instance of the ReinforcementLearningAgentBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ActionDimension` | Gets the number of action dimensions. |
| `DefaultLossFunction` | Gets the default loss function for this agent. |
| `FeatureCount` | Gets the number of input features (state dimensions). |
| `FeatureNames` | Gets the names of input features. |
| `ParameterCount` | Gets the number of parameters in the agent. |
| `SupportsParameterInitialization` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyGradients(Vector<>,)` | Applies gradients to update the agent. |
| `Clone` | Clones the agent. |
| `ComputeAverage(IEnumerable<>)` | Computes the average of a collection of values. |
| `ComputeGradients(Vector<>,Vector<>,ILossFunction<>)` | Computes gradients for the agent. |
| `DeepCopy` | Creates a deep copy of the agent. |
| `Deserialize(Byte[])` | Deserializes the agent from bytes. |
| `Dispose` | Disposes of resources used by the agent. |
| `GetActiveFeatureIndices` | Gets the indices of active features. |
| `GetDynamicShapeInfo` |  |
| `GetFeatureImportance` | Gets feature importance scores. |
| `GetInputShape` |  |
| `GetMetrics` | Gets the current training metrics. |
| `GetModelMetadata` | Gets model metadata. |
| `GetOptions` |  |
| `GetOutputShape` |  |
| `GetParameters` | Gets the agent's parameters. |
| `HashStateToAction(String,Int32)` | Computes a deterministic, state-dependent fallback action index for tabular agents whose Q-values are tied (typical for unvisited states with zero-init). |
| `IsFeatureUsed(Int32)` | Checks if a feature is used by the agent. |
| `LoadModel(String)` | Loads the agent's state from a file, stripping the AIMF header if present. |
| `LoadState(Stream)` | Loads the agent's state (parameters and configuration) from a stream. |
| `Predict(Vector<>)` | Makes a prediction using the trained agent. |
| `ResetEpisode` | Resets episode-specific state (if any). |
| `SanitizeParameters(Vector<>)` |  |
| `SaveModel(String)` | Saves the agent's state to a file with an AIMF envelope header. |
| `SaveState(Stream)` | Saves the agent's current state (parameters and configuration) to a stream. |
| `SelectAction(Vector<>,Boolean)` | Selects an action given the current state observation. |
| `Serialize` | Serializes the agent to bytes. |
| `SetActiveFeatureIndices(IEnumerable<Int32>)` | Sets the active feature indices. |
| `SetParameters(Vector<>)` | Sets the agent's parameters. |
| `StoreExperience(Vector<>,Vector<>,,Vector<>,Boolean)` | Stores an experience tuple for later learning. |
| `Train` | Performs one training step, updating the agent's policy/value function. |
| `Train(Vector<>,Vector<>)` | Trains the agent on a single (state, target) supervised pair by translating it into a one-step RL transition and dispatching through the standard `Boolean)` + `Train` pipeline. |
| `WithParameters(Vector<>)` | Creates a new instance with the specified parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `DiscountFactor` | Discount factor (gamma) for future rewards. |
| `Episodes` | Number of episodes completed. |
| `LearningRate` | Learning rate for gradient updates. |
| `LossFunction` | Loss function used for training. |
| `LossHistory` | History of losses during training. |
| `NumOps` | Numeric operations provider for type T. |
| `Options` | Configuration options for this agent. |
| `Random` | Random number generator for stochastic operations. |
| `RewardHistory` | History of episode rewards. |
| `TrainingSteps` | Number of training steps completed. |

