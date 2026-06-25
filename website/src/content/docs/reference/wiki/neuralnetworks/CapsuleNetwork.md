---
title: "CapsuleNetwork<T>"
description: "Represents a Capsule Network, a type of neural network that preserves spatial relationships between features."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a Capsule Network, a type of neural network that preserves spatial relationships between features.

## For Beginners

A Capsule Network is like a more advanced version of traditional neural networks.

Think of it this way:

- Traditional networks detect features like edges or textures, but lose information about how these features relate to each other
- Capsule Networks not only detect features, but also understand their relationships, orientations, and positions
- This is like the difference between recognizing individual puzzle pieces versus understanding how they fit together

For example, a traditional network might recognize an eye, a nose, and a mouth separately, but a Capsule Network
can better understand that these features need to be in a specific arrangement to make a face. This makes
Capsule Networks particularly good at recognizing objects from different angles or when parts are arranged differently.

## How It Works

A Capsule Network is a neural network architecture designed to address limitations of traditional convolutional
neural networks. Instead of using scalar-output feature detectors (neurons), Capsule Networks use vector-output
capsules. Each capsule's output vector represents the presence of an entity and its instantiation parameters
(like position, orientation, and scale). This architecture helps to preserve hierarchical relationships
between features, making it particularly effective for tasks requiring understanding of spatial relationships.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CapsuleNetwork` | Initializes a new instance of the `CapsuleNetwork` class with the specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AuxiliaryLossWeight` | Gets or sets the weight for reconstruction loss. |
| `UseAuxiliaryLoss` | Gets or sets whether to use auxiliary loss (reconstruction regularization) during training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyCapsuleMask(Tensor<>,Nullable<Int32>)` | Applies masking to capsule outputs, zeroing out all capsules except the target. |
| `ComputeAuxiliaryLoss` | Computes the auxiliary loss for the CapsuleNetwork, which is the reconstruction regularization. |
| `ComputeReconstructionLoss(Tensor<>,Nullable<Int32>)` | Computes the reconstruction loss for capsule network regularization. |
| `ComputeReconstructionLoss(Tensor<>,Tensor<>)` | Computes the reconstruction loss by measuring how well the input can be reconstructed from capsule outputs. |
| `CreateNewInstance` | Creates a new instance of the capsule network model. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes Capsule Network-specific data from a binary reader. |
| `ForwardForTraining(Tensor<>)` | Tape-tracked forward pass that also captures output for reconstruction loss. |
| `GetAuxiliaryLossDiagnostics` | Gets diagnostic information about the auxiliary losses. |
| `GetDiagnostics` | Gets diagnostic information about this component's state and behavior. |
| `GetModelMetadata` | Retrieves metadata about the Capsule Network model. |
| `GetOptions` |  |
| `GetPredictedClass(Tensor<>)` | Gets the predicted class from capsule outputs based on capsule norms. |
| `InitializeLayers` | Initializes the layers of the Capsule Network based on the architecture. |
| `PredictCore(Tensor<>)` | Performs a forward pass through the Capsule Network to make a prediction. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes Capsule Network-specific data to a binary writer. |
| `Train(Tensor<>,Tensor<>)` | Trains the Capsule Network using the provided input and expected output. |
| `UpdateParameters(Vector<>)` | Updates the parameters of all layers in the Capsule Network. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_lastCapsuleOutputs` | Stores the last capsule outputs for reconstruction loss computation. |
| `_lastInput` | Stores the last input for reconstruction loss computation. |
| `_lastMarginLoss` | Stores the last computed margin loss for diagnostics. |
| `_lastReconstructionLoss` | Stores the last computed reconstruction loss for diagnostics. |
| `_optimizer` | Optimizer used by the tape-based training path. |

