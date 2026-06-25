---
title: "LinearElasticityEquation<T>"
description: "Represents the 2D Linear Elasticity Equations (Navier-Cauchy equations): (λ + μ)∂(∂u/∂x + ∂v/∂y)/∂x + μ∇²u + fₓ = 0 (λ + μ)∂(∂u/∂x + ∂v/∂y)/∂y + μ∇²v + fᵧ = 0"
section: "API Reference"
---

`Models & Types` · `AiDotNet.PhysicsInformed.PDEs`

Represents the 2D Linear Elasticity Equations (Navier-Cauchy equations):
(λ + μ)∂(∂u/∂x + ∂v/∂y)/∂x + μ∇²u + fₓ = 0
(λ + μ)∂(∂u/∂x + ∂v/∂y)/∂y + μ∇²v + fᵧ = 0

## How It Works

For Beginners:
The Linear Elasticity equations describe how solid materials deform under stress.

Variables:

- u(x,y) = Displacement in x-direction
- v(x,y) = Displacement in y-direction
- λ (lambda) = First Lamé parameter (related to bulk modulus)
- μ (mu) = Second Lamé parameter (shear modulus, measures resistance to shearing)
- fₓ, fᵧ = Body forces (like gravity)

Physical Interpretation:

- When you push or pull on a solid object, it deforms
- The equations balance internal stresses with external forces
- The Lamé parameters describe how stiff the material is

Material Properties:

- λ and μ can be computed from Young's modulus E and Poisson's ratio ν:
* λ = Eν / ((1+ν)(1-2ν))
* μ = E / (2(1+ν))

Applications:

- Structural engineering (buildings, bridges)
- Mechanical design (stress analysis)
- Geology (tectonic plate deformation)
- Biomechanics (bone and tissue mechanics)

Example: A beam bending under load, a pressure vessel expanding,
or a rubber band stretching.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LinearElasticityEquation(,,,)` | Initializes a new instance of the Linear Elasticity Equation. |
| `LinearElasticityEquation(Double,Double,Double,Double)` | Initializes a new instance of the Linear Elasticity Equation with double parameters. |

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
| `FromEngineeringConstants(Double,Double,Double,Double)` | Creates a Linear Elasticity Equation from Young's modulus and Poisson's ratio. |

