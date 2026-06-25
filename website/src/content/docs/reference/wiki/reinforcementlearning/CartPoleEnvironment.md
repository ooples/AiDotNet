---
title: "CartPoleEnvironment<T>"
description: "Classic CartPole-v1 environment for reinforcement learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ReinforcementLearning.Environments`

Classic CartPole-v1 environment for reinforcement learning.

## For Beginners

Think of this like balancing a broomstick on your hand - you move your hand left and right
to keep the stick upright. The CartPole is a classic RL problem that's simple to understand
but requires learning to balance competing forces.

State (4 dimensions):

- Cart position: where the cart is (-2.4 to 2.4)
- Cart velocity: how fast it's moving
- Pole angle: how tilted the pole is (-12° to 12°)
- Pole angular velocity: how fast it's rotating

Actions (2 discrete):

- 0: Push cart left
- 1: Push cart right

Reward: +1 for each timestep the pole remains balanced

## How It Works

The CartPole environment simulates balancing a pole on a cart. The agent must move the cart
left or right to keep the pole balanced. The episode ends if:

- The pole angle exceeds ±12 degrees
- The cart position exceeds ±2.4 units
- The maximum number of steps is reached

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CartPoleEnvironment(Int32,Nullable<Int32>)` | Initializes a new instance of the CartPoleEnvironment class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ActionSpaceSize` |  |
| `IsContinuousActionSpace` |  |
| `ObservationSpaceDimension` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Close` |  |
| `Reset` |  |
| `Seed(Int32)` |  |
| `Step(Vector<>)` |  |

