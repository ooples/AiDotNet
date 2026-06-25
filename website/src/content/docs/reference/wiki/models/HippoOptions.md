---
title: "HippoOptions<T>"
description: "Configuration options for HiPPO (High-order Polynomial Projection Operators) model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for HiPPO (High-order Polynomial Projection Operators) model.

## For Beginners

HiPPO answers a fundamental question in sequence modeling:
"How do we optimally remember a continuous history in a fixed-size memory?"

**The Key Insight:**
When processing a sequence, we want our hidden state to be an "optimal summary" of the past.
HiPPO shows that by projecting the input history onto polynomial bases (like Legendre polynomials),
we can create hidden states that provably capture the history optimally.

**How It Works:**

1. **Polynomial Basis:** Choose a basis (Legendre, Laguerre, Fourier, etc.)
2. **Optimal Projection:** The state x(t) represents coefficients of the polynomial

approximation of the input history

3. **Online Update:** The state can be updated efficiently as new inputs arrive
4. **Memory Matrix A:** Defines how the state evolves (different for each basis)

**The Math (simplified):**
State Space Model: dx/dt = Ax + Bu

- A is the "HiPPO matrix" derived from the chosen polynomial basis
- x(t) contains coefficients: history ≈ Σ x_i(t) * P_i(τ)
- Different A matrices give different memory properties:
- LegS: Sliding window over recent history
- LegT: Fixed window over [0, t]
- LagT: Exponential decay (older = less weight)

**Why HiPPO Matters:**

- Provides principled initialization for state space models
- Enables models to handle very long sequences
- Foundation for S4, Mamba, and other modern SSMs
- Mathematically optimal for memory compression

## How It Works

HiPPO provides the theoretical foundation for efficient state space models like S4 and Mamba.
It defines optimal state matrices for compressing sequential input history into a fixed-size state.

**Reference:** Gu et al., "HiPPO: Recurrent Memory with Optimal Polynomial Projections", 2020.
https://arxiv.org/abs/2008.07669

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HippoOptions` | Initializes a new instance of the `HippoOptions` class with default values. |
| `HippoOptions(HippoOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ContextLength` | Gets or sets the context length (input sequence length). |
| `DiscretizationMethod` | Gets or sets the discretization method for converting continuous to discrete SSM. |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `ForecastHorizon` | Gets or sets the forecast horizon (prediction length). |
| `HippoMethod` | Gets or sets the HiPPO method for state matrix initialization. |
| `ModelDimension` | Gets or sets the model dimension (d_model). |
| `NumLayers` | Gets or sets the number of HiPPO layers. |
| `StateDimension` | Gets or sets the state dimension (N in the paper). |
| `TimescaleMax` | Gets or sets the maximum timescale for the SSM. |
| `TimescaleMin` | Gets or sets the minimum timescale for the SSM. |
| `UseNormalization` | Gets or sets whether to use normalization between HiPPO layers. |

