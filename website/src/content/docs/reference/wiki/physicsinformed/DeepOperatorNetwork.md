---
title: "DeepOperatorNetwork<T>"
description: "Implements Deep Operator Network (DeepONet) for learning operators."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PhysicsInformed.NeuralOperators`

Implements Deep Operator Network (DeepONet) for learning operators.

## How It Works

For Beginners:
DeepONet is another approach to learning operators (like FNO), but with a different architecture.

Universal Approximation Theorem for Operators:
Just as neural networks can approximate any function, DeepONet can approximate any operator!
This is based on a theorem by Chen and Chen (1995).

The Key Idea - Decomposition:
DeepONet represents an operator G as:
G(u)(y) = Σᵢ bᵢ(u) * tᵢ(y)

Where:

- u is the input function
- y is the query location
- bᵢ(u) are "basis functions" of the input (learned by Branch Net)
- tᵢ(y) are "basis functions" of the location (learned by Trunk Net)

Architecture:
DeepONet has TWO networks:

1. Branch Network:
- Input: The entire input function u(x) (sampled at sensors)
- Output: Coefficients b₁, b₂, ..., bₚ
- Role: Encodes information about the input function

2. Trunk Network:
- Input: Query location y (where we want to evaluate output)
- Output: Basis functions t₁(y), t₂(y), ..., tₚ(y)
- Role: Encodes spatial/temporal patterns

3. Combination:
- Output: G(u)(y) = b · t = Σᵢ bᵢ * tᵢ(y)
- Simple dot product of the two network outputs

Analogy:
Think of it like a bilinear form or low-rank factorization:

- Branch net learns "what" information matters in the input
- Trunk net learns "where" patterns occur spatially
- Their interaction gives the output

Example - Heat Equation:
Problem: Given initial temperature u(x,0), find temperature u(x,t)

Branch Net:

- Input: u(x,0) sampled at many points → [u(x₁,0), u(x₂,0), ..., u(xₙ,0)]
- Learns: "This initial condition is smooth/peaked/oscillatory"
- Output: Coefficients [b₁, b₂, ..., bₚ]

Trunk Net:

- Input: (x, t) where we want to know the temperature
- Learns: Spatial-temporal basis functions
- Output: Basis values [t₁(x,t), t₂(x,t), ..., tₚ(x,t)]

Result:
u(x,t) = Σᵢ bᵢ * tᵢ(x,t)

Key Advantages:

1. Sensor flexibility: Can use different sensor locations at test time
2. Query flexibility: Can evaluate at any location y
3. Theoretical foundation: Universal approximation theorem
4. Efficient: Once trained, very fast evaluation
5. Interpretable: Decomposition into branch/trunk has clear meaning

Comparison with FNO:
DeepONet:

- Works on unstructured data (any sensor locations)
- More flexible for irregular domains
- Requires specifying sensor locations
- Good for problems with sparse/irregular data

FNO:

- Works on structured grids
- Uses FFT (very efficient)
- Resolution-invariant
- Good for periodic/regular problems

Both are powerful, choice depends on your problem!

Applications:

- Same as FNO: PDEs, climate, fluids, etc.
- Particularly good for:
* Inverse problems (finding unknown parameters)
* Problems with sparse measurements
* Irregular geometries
* Multi-scale phenomena

Historical Note:
DeepONet was introduced by Lu et al. (2021) and has been highly successful
in learning solution operators for PDEs with theoretical guarantees.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DeepOperatorNetwork(NeuralNetworkArchitecture<>,NeuralNetworkArchitecture<>,NeuralNetworkArchitecture<>,Int32,Int32,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,DeepOperatorNetworkOptions)` | Initializes a new instance of DeepONet. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets the total number of parameters across branch and trunk networks. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new instance with the same configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes DeepONet-specific data. |
| `Evaluate([],[])` | Evaluates the operator: G(u)(y) = branch(u) · trunk(y). |
| `EvaluateMultiple([],[0:,0:])` | Evaluates the operator at multiple query locations efficiently. |
| `GetModelMetadata` | Gets metadata about the DeepONet model. |
| `GetOptions` |  |
| `GetParameters` | Gets the trainable parameters as a flattened vector. |
| `PredictCore(Tensor<>)` | Makes a prediction using the DeepONet for a batch of input/query pairs. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes DeepONet-specific data. |
| `Train(Tensor<>,Tensor<>)` | Performs a basic supervised training step using MSE loss. |
| `Train([0:,0:],[0:,0:,0:],[0:,0:],Int32,Double,Boolean)` | Trains DeepONet on input-output function pairs. |
| `UpdateParameters(Vector<>)` | Updates the branch and trunk network parameters from a flattened vector. |

