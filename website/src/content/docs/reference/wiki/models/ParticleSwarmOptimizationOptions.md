---
title: "ParticleSwarmOptimizationOptions<T, TInput, TOutput>"
description: "Configuration options for Particle Swarm Optimization (PSO), a population-based stochastic optimization technique inspired by social behavior of bird flocking or fish schooling."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Particle Swarm Optimization (PSO), a population-based stochastic optimization
technique inspired by social behavior of bird flocking or fish schooling.

## For Beginners

Particle Swarm Optimization is a way to find the best solution by mimicking how birds flock or fish swim in groups.

Imagine you and your friends are searching for the highest point in a hilly landscape while blindfolded:

- Each person (particle) can only feel the height at their current position
- Everyone remembers the highest point they personally have found so far
- The group also shares information about the highest point anyone has found

When deciding where to step next:

- You consider continuing in your current direction (inertia)
- You're pulled toward the best spot you've personally found (cognitive component)
- You're also pulled toward the best spot anyone in the group has found (social component)
- You combine these influences to decide your next move

This creates a smart search pattern because:

- People explore different areas (maintaining diversity in the search)
- The group gradually converges on promising regions
- The method balances exploration (finding new areas) with exploitation (refining good solutions)

This class provides extensive options to fine-tune how the particles move and interact,
allowing you to customize the algorithm for different types of optimization problems.

## How It Works

Particle Swarm Optimization is a computational method that optimizes a problem by iteratively
improving candidate solutions (particles) with regard to a given measure of quality. The algorithm
maintains a population of particles, where each particle represents a potential solution to the
optimization problem. Particles move through the solution space guided by their own best known position
and the swarm's best known position. This social interaction leads to emergent intelligence that
efficiently explores complex solution spaces. PSO is particularly effective for continuous optimization
problems and has advantages in terms of simplicity, flexibility, and minimal parameter tuning requirements.

## Properties

| Property | Summary |
|:-----|:--------|
| `CognitiveParameter` | Gets or sets the cognitive parameter that controls the influence of the particle's personal best position. |
| `CognitiveWeightAdaptationRate` | Gets or sets the rate at which the cognitive weight adapts when using adaptive weights. |
| `InertiaDecayRate` | Gets or sets the rate at which inertia weight decays when using adaptive inertia. |
| `InertiaWeight` | Gets or sets the inertia weight that controls the influence of the particle's previous velocity. |
| `InitialCognitiveWeight` | Gets or sets the initial cognitive weight when using adaptive weights. |
| `InitialInertia` | Gets or sets the initial inertia weight when using adaptive inertia. |
| `InitialSocialWeight` | Gets or sets the initial social weight when using adaptive weights. |
| `MaxCognitiveWeight` | Gets or sets the maximum cognitive weight when using adaptive weights. |
| `MaxInertia` | Gets or sets the maximum inertia weight when using adaptive inertia. |
| `MaxSocialWeight` | Gets or sets the maximum social weight when using adaptive weights. |
| `MinCognitiveWeight` | Gets or sets the minimum cognitive weight when using adaptive weights. |
| `MinInertia` | Gets or sets the minimum inertia weight when using adaptive inertia. |
| `MinSocialWeight` | Gets or sets the minimum social weight when using adaptive weights. |
| `SocialParameter` | Gets or sets the social parameter that controls the influence of the swarm's global best position. |
| `SocialWeightAdaptationRate` | Gets or sets the rate at which the social weight adapts when using adaptive weights. |
| `SwarmSize` | Gets or sets the number of particles in the swarm. |
| `UseAdaptiveInertia` | Gets or sets whether to use adaptive inertia weight that changes throughout the optimization process. |
| `UseAdaptiveWeights` | Gets or sets whether to use adaptive cognitive and social parameters that change throughout the optimization. |

