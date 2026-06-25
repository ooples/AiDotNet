---
title: "FourierNeuralOperator<T>"
description: "Implements the Fourier Neural Operator (FNO) for learning operators between function spaces."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PhysicsInformed.NeuralOperators`

Implements the Fourier Neural Operator (FNO) for learning operators between function spaces.

## How It Works

For Beginners:
A Neural Operator learns mappings between entire functions, not just inputs to outputs.

Traditional Neural Networks:

- Learn: point → point mappings
- Input: a vector (x, y, z)
- Output: a vector (u, v, w)
- Example: (temperature, pressure) → (velocity)

Neural Operators:

- Learn: function → function mappings
- Input: an entire function a(x)
- Output: an entire function u(x)
- Example: initial condition → solution after time T

Why This Matters:
Many problems in physics involve operators:

- PDE solution operator: (initial/boundary conditions) → (solution)
- Green's function: (source) → (response)
- Transfer function: (input signal) → (output signal)

Traditionally, you'd need to solve the PDE from scratch for each new set of conditions.
With neural operators, you train once, then can instantly evaluate for new conditions!

Fourier Neural Operator (FNO):
The key innovation is doing computations in Fourier space.

How FNO Works:

1. Lift: Embed input function into higher-dimensional space
2. Fourier Layers (repeated):

a) Apply FFT to transform to frequency domain
b) Linear transformation in frequency space (learn weights)
c) Apply inverse FFT to return to physical space
d) Add skip connection and activation

3. Project: Map back to output function

Why Fourier Space?

- Many PDEs have simple form in frequency domain
- Derivatives → multiplication (∂/∂x in physical space = ik in Fourier space)
- Captures global information efficiently
- Natural for periodic problems
- Computational efficiency (FFT is O(n log n))

Key Advantages:

1. Resolution-invariant: Train at one resolution, evaluate at another
2. Fast: Instant evaluation after training (vs. solving PDE each time)
3. Mesh-free: No discretization needed
4. Generalizes well: Works for different parameter values
5. Captures long-range dependencies naturally

Applications:

- Fluid dynamics (Navier-Stokes)
- Climate modeling (weather prediction)
- Material science (stress-strain)
- Seismic imaging
- Quantum chemistry (electron density)

Example Use Case:
Problem: Solve 2D Navier-Stokes for different initial vorticity fields
Traditional: Solve PDE numerically for each initial condition (slow)
FNO: Train once on many examples, then instantly predict solution for new initial conditions

Historical Context:
FNO was introduced by Li et al. in 2021 and has achieved remarkable success
in learning solution operators for PDEs, often matching or exceeding traditional
numerical methods in accuracy while being orders of magnitude faster.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FourierNeuralOperator` | Initializes a new instance of the Fourier Neural Operator with default configuration. |
| `FourierNeuralOperator(NeuralNetworkArchitecture<>,Int32,Int32,Int32[],Int32,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,FourierNeuralOperatorOptions)` | Initializes a new instance of the Fourier Neural Operator. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets the total parameter count for lift, Fourier, and projection layers. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new instance with the same configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes FNO-specific data. |
| `FlattenPointwiseInput(Tensor<>,Int32[])` | Flattens a channel-first tensor `[B, C, d_1, ..., d_N]` into the `[B * d_1 * ... |
| `Forward(Tensor<>)` | Forward pass through the FNO. |
| `ForwardForTraining(Tensor<>)` | Forward pass used by tape-based training. |
| `GetModelMetadata` | Gets metadata about the FNO model. |
| `GetOptions` |  |
| `GetParameters` | Gets the trainable parameters as a flattened vector. |
| `InitializeFourierLayers(Int32)` | Initializes the Fourier layers. |
| `PredictCore(Tensor<>)` | Makes a prediction using the FNO. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes FNO-specific data. |
| `TapeTrainStep(Tensor<>,Tensor<>)` | Tape-based FNO training step. |
| `Train(Tensor<>,Tensor<>)` | Performs a basic supervised training step using MSE loss. |
| `Train(Tensor<>[],Tensor<>[],Int32,Double)` | Trains the FNO on input-output function pairs. |
| `UnflattenPointwiseOutput(Tensor<>,Int32,Int32[])` | Inverse of `Int32[])`. |
| `UpdateParameters(Vector<>)` | Updates the trainable parameters from a flattened vector. |

