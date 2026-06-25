---
title: "DeepQNetwork<T>"
description: "Represents a Deep Q-Network (DQN), a reinforcement learning algorithm that combines Q-learning with deep neural networks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a Deep Q-Network (DQN), a reinforcement learning algorithm that combines Q-learning with deep neural networks.

## For Beginners

A Deep Q-Network is like a smart decision-maker that learns through trial and error.

Imagine you're teaching a robot to play a video game:

- The robot needs to learn which actions (button presses) are best in each situation (game screen)
- At first, the robot makes many random moves to explore the game
- Over time, it remembers which moves led to high scores and which led to game over
- The "Deep" part means it uses a neural network to recognize patterns in complex situations
- The "Q" refers to "Quality" - how good an action is in a specific situation

For example, in a maze game, the network learns that moving toward the exit is usually better than moving away from it,
even if it hasn't seen that exact maze position before.

## How It Works

A Deep Q-Network (DQN) is a reinforcement learning algorithm that uses a neural network to approximate the Q-function,
which represents the expected future rewards for taking specific actions in specific states. DQNs overcome the limitations
of traditional Q-learning by using neural networks to generalize across states, allowing them to handle problems with large
or continuous state spaces. Key features of DQNs include experience replay (storing and randomly sampling past experiences)
and the use of a separate target network to stabilize learning.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DeepQNetwork` | Initializes a new instance of the `DeepQNetwork` class with the specified architecture and exploration rate. |
| `DeepQNetwork(NeuralNetworkArchitecture<>,ILossFunction<>,Double,Boolean,DeepQNetworkOptions)` | Private constructor used to create the target network without infinite recursion. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddExperience(Tensor<>,Int32,,Tensor<>,Boolean)` | Adds a new experience to the replay buffer. |
| `CreateNewInstance` | Creates a new instance of the Deep Q-Network with the same architecture and configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Loads Deep Q-Network specific data from a binary stream. |
| `ForwardWithMemoryBatch(Tensor<>)` | Performs a forward pass for a batch of inputs while storing intermediate values for backpropagation. |
| `GetAction(Tensor<>)` | Gets an action to take in the given state, balancing exploration and exploitation. |
| `GetBestAction(Tensor<>)` | Gets the best action to take in the given state based on current Q-values. |
| `GetModelMetadata` | Gets metadata about this Deep Q-Network model. |
| `GetOptions` |  |
| `GetQValues(Tensor<>)` | Gets the Q-values for all possible actions in the given state. |
| `InitializeLayers` | Initializes the layers of the Deep Q-Network based on the architecture. |
| `PredictCore(Tensor<>)` | Performs a forward pass with a tensor input. |
| `SampleBatch(Int32)` | Samples a batch of experiences from the replay buffer. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Saves Deep Q-Network specific data to a binary stream. |
| `UpdateParameters()` | Updates the parameters of all layers in the network using the calculated gradients. |
| `UpdateParameters(Vector<>)` | Updates the parameters of all layers in the Deep Q-Network. |
| `UpdateTargetNetwork` | Updates the target network to match the current state of the main network. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_actionSpace` | Gets or sets the number of possible actions the agent can take in the environment. |
| `_batchSize` | Gets the number of experiences to sample from the replay buffer during each training step. |
| `_epsilon` | Gets the exploration rate, which controls how often the agent takes random actions versus exploiting learned knowledge. |
| `_replayBuffer` | Gets the buffer that stores past experiences for experience replay. |
| `_targetNetwork` | Gets the target network, a copy of the main network used to generate target Q-values during training. |
| `_trainOptimizer` | Trains the network using experience replay. |

