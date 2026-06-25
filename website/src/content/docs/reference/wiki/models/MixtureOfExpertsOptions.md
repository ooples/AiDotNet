---
title: "MixtureOfExpertsOptions<T>"
description: "Configuration options for the Mixture-of-Experts (MoE) neural network model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Mixture-of-Experts (MoE) neural network model.

## For Beginners

Mixture-of-Experts (MoE) is like having a team of specialists rather than one generalist.

Imagine you're running a hospital:

- Instead of one doctor handling everything, you have specialists (cardiologist, neurologist, etc.)
- A triage system (gating network) decides which specialist(s) should see each patient
- Each specialist only handles cases they're best suited for

In a MoE neural network:

- Multiple "expert" networks specialize in different patterns in your data
- A "gating network" learns to route each input to the best expert(s)
- Only a few experts process each input (sparse activation), making it efficient
- The final prediction combines the outputs from the selected experts

This class lets you configure:

- How many expert networks to use
- How many experts process each input (Top-K)
- Dimensions of the expert networks
- Whether to use load balancing to ensure all experts are utilized

## How It Works

Mixture-of-Experts is a neural network architecture that employs multiple specialist networks (experts)
and a gating mechanism to route inputs to the most appropriate experts. This approach enables:

- Increased model capacity without proportional compute cost (sparse activation)
- Specialization of different experts on different aspects of the problem
- Improved scalability for large-scale problems

## Properties

| Property | Summary |
|:-----|:--------|
| `HiddenExpansion` | Gets or sets the hidden layer expansion factor for each expert's feed-forward network. |
| `InputDim` | Gets or sets the input dimension for each expert network. |
| `LoadBalancingWeight` | Gets or sets the weight of the auxiliary load balancing loss. |
| `NumExperts` | Gets or sets the number of expert networks in the mixture. |
| `OutputDim` | Gets or sets the output dimension for each expert network. |
| `RandomSeed` | Gets or sets the random seed for expert initialization. |
| `TopK` | Gets or sets the number of experts to activate for each input (Top-K routing). |
| `UseLoadBalancing` | Gets or sets whether to enable auxiliary load balancing loss. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates that all option values are within acceptable ranges. |

