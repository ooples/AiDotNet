---
title: "DomainDecompositionPINN<T>"
description: "Domain Decomposition Physics-Informed Neural Network for large-scale problems."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PhysicsInformed.PINNs`

Domain Decomposition Physics-Informed Neural Network for large-scale problems.

## How It Works

For Beginners:
Domain decomposition is a strategy for solving large problems by breaking them
into smaller, manageable pieces. Think of it like solving a jigsaw puzzle:

- Each piece (subdomain) is solved separately
- The pieces must fit together at the edges (interface conditions)

Why Use Domain Decomposition?

1. Memory: Large domains require too much memory for one network
2. Parallelism: Subdomains can be trained independently (partially)
3. Accuracy: Local networks can specialize for local behavior
4. Geometry: Complex domains can be split into simpler shapes

Types of Decomposition:

1. Non-overlapping (used here):

|-------|-------|
| D1 | D2 | Interface at boundary
|-------|-------|

2. Overlapping:

|----------|
| D1 |---|
|----------| | D2 |
|----|------|
Overlap region

Interface Conditions:
At subdomain interfaces, we enforce:

1. Continuity: u_1 = u_2 (solutions match)
2. Flux continuity: du_1/dn = du_2/dn (derivatives match)

Training Strategy:

1. Train each subdomain network on its local domain
2. Enforce interface conditions at boundaries
3. Iterate until global convergence

References:

- Jagtap, A.D., and Karniadakis, G.E. "Extended Physics-Informed Neural Networks

(XPINNs): A Generalized Space-Time Domain Decomposition Based Deep Learning Framework"
Communications in Computational Physics, 2020.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DomainDecompositionPINN(NeuralNetworkArchitecture<>,IPDESpecification<>,IBoundaryCondition<>[],List<SubdomainDefinition<>>,List<PhysicsInformedNeuralNetwork<>>,IInitialCondition<>,Int32,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,Double,Double,Double,Double,Int32,DomainDecompositionPINNOptions)` | Creates a Domain Decomposition PINN with specified subdomains. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Interfaces` | Gets all interface definitions. |
| `SubdomainCount` | Gets the number of subdomains. |
| `Subdomains` | Gets all subdomain definitions. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DetectInterfaces(List<SubdomainDefinition<>>)` | Detects interfaces between adjacent subdomains. |
| `FindSharedBoundary(SubdomainDefinition<>,SubdomainDefinition<>)` | Finds the shared boundary between two subdomains. |
| `GenerateBoundaryPoints(,Int32,SubdomainDefinition<>,Int32)` | Generates points along a shared boundary. |
| `GetGlobalSolution([])` | Gets the solution at a point by finding the appropriate subdomain. |
| `GetOptions` |  |
| `GetSubdomainNetwork(Int32)` | Gets a specific subdomain network. |
| `SolveWithDecomposition(Int32,Double,Boolean,Int32)` | Solves the PDE using domain decomposition. |

