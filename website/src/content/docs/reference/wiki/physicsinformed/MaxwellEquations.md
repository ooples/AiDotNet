---
title: "MaxwellEquations<T>"
description: "Represents Maxwell's equations for electromagnetic wave propagation (2D TE mode)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PhysicsInformed.PDEs`

Represents Maxwell's equations for electromagnetic wave propagation (2D TE mode).

## How It Works

For Beginners:
Maxwell's equations are the foundation of all electromagnetic phenomena including:

- Light, radio waves, microwaves, X-rays (all forms of electromagnetic radiation)
- Electric motors and generators
- Wireless communication
- Optical fibers

The equations describe how electric and magnetic fields interact and propagate.
This implementation uses the 2D Transverse Electric (TE) mode where:

- Electric field lies in the x-y plane: (Ex, Ey)
- Magnetic field points in z-direction: Bz

The equations are:
**Faraday's Law** (changing magnetic field creates electric field):
∂Bz/∂t = -(∂Ey/∂x - ∂Ex/∂y)

**Ampere's Law** (changing electric field creates magnetic field):
Using B = μH, the curl of B gives: ∇×B = εμ ∂E/∂t
∂Ex/∂t = (1/εμ) ∂Bz/∂y
∂Ey/∂t = -(1/εμ) ∂Bz/∂x

Key Parameters:

- ε (epsilon): Electric permittivity - how easily a material polarizes
- μ (mu): Magnetic permeability - how easily a material magnetizes
- c = 1/sqrt(εμ): Speed of light in the medium

Physical Interpretation:

- Electromagnetic waves are self-sustaining oscillations of E and B fields
- The wave equation shows they propagate at the speed of light
- Energy flows perpendicular to both E and B (Poynting vector)

Applications:

- Antenna design
- Optical waveguides
- Photonic crystals
- Metamaterials
- Radar and communication systems

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MaxwellEquations(,)` | Initializes Maxwell's equations with specified electromagnetic properties. |
| `MaxwellEquations(Double,Double)` | Initializes Maxwell's equations with double parameters. |

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

