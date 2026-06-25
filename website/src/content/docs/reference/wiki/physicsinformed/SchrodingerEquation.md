---
title: "SchrodingerEquation<T>"
description: "Represents the time-dependent Schrodinger equation for quantum mechanics."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PhysicsInformed.PDEs`

Represents the time-dependent Schrodinger equation for quantum mechanics.

## How It Works

For Beginners:
The Schrodinger equation is the fundamental equation of quantum mechanics.
It describes how the quantum state (wavefunction) of a physical system evolves over time.

The equation in 1D (with normalized units ℏ = m = 1):
i ∂ψ/∂t = -½ ∂²ψ/∂x² + V(x)ψ

Since ψ is complex (ψ = ψ_r + i*ψ_i), we split into real/imaginary parts:
∂ψ_r/∂t = -½ ∂²ψ_i/∂x² + V*ψ_i
∂ψ_i/∂t = ½ ∂²ψ_r/∂x² - V*ψ_r

Key Concepts:

- ψ(x,t): Wavefunction - its square magnitude |ψ|² gives probability density
- V(x): Potential energy function (e.g., harmonic oscillator, particle in box)
- ℏ: Reduced Planck's constant (set to 1 in normalized units)
- m: Particle mass (set to 1 in normalized units)

Physical Interpretation:

- The wavefunction encodes all quantum information about the system
- |ψ(x,t)|² = ψ_r² + ψ_i² is the probability density at position x, time t
- Total probability is conserved: ∫|ψ|²dx = 1

Applications:

- Atomic and molecular physics
- Quantum chemistry
- Semiconductor physics
- Quantum computing simulations
- Tunneling phenomena

This implementation supports a user-defined potential function V(x).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SchrodingerEquation` | Creates a Schrodinger equation with zero potential (free particle). |
| `SchrodingerEquation(Func<,>)` | Initializes the Schrodinger equation with a specified potential. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InputDimension` |  |
| `Name` |  |
| `OutputDimension` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeResidual(Vector<>,Vector<>,PDEDerivatives<>)` |  |
| `ComputeResidualGradient(Vector<>,Vector<>,PDEDerivatives<>)` |  |
| `CreateZeroPotential` | Creates a zero potential function (free particle case). |

