---
title: "SymbolicPhysicsLearner<T>"
description: "Implements Symbolic Physics Learning for discovering interpretable equations from data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PhysicsInformed.ScientificML`

Implements Symbolic Physics Learning for discovering interpretable equations from data.

## How It Works

For Beginners:
Symbolic Physics Learner discovers equations in symbolic form (like f = ma, E = mc^2).

Traditional ML:

- Neural networks are "black boxes"
- Learn complex functions but hard to interpret
- Can't extract simple equations

Symbolic Regression:

- Discovers actual mathematical equations
- Interpretable results (can publish in papers!)
- Can rediscover known physics laws
- Can discover NEW laws

Example:
Input: Data of planetary positions vs. time
Output: F = G*m1*m2/r^2 (Newton's law of gravitation)

How It Works:

1. Search space: Library of operators (+, -, *, /, sin, exp, etc.)
2. Search algorithm: Genetic programming, reinforcement learning, etc.
3. Fitness: Balance between accuracy and simplicity (Occam's razor)
4. Output: Symbolic expression

Applications:

- Discovering physical laws from experiments
- Automating scientific discovery
- Interpretable AI for science
- Finding conservation laws

Famous Success:

- Rediscovered Kepler's laws from planetary data
- Found new equations in materials science
- Discovered patterns in quantum mechanics

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultLossFunction` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DeepCopy` |  |
| `DiscoverEquation([0:,0:],[],Int32,Int32)` | Discovers a symbolic equation from data using genetic programming. |
| `GetParameters` |  |
| `Predict(Matrix<>)` | Predicts outputs using the discovered symbolic equation. |
| `SetParameters(Vector<>)` |  |
| `Simplify(SymbolicExpression<>)` | Simplifies an expression using symbolic algebra rules. |
| `ToLatex(SymbolicExpression<>)` | Converts expression to human-readable string. |
| `Train(Matrix<>,Vector<>)` | Trains the symbolic physics learner by discovering an equation from the data. |
| `WithParameters(Vector<>)` |  |

