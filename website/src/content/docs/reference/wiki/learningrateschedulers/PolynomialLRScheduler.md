---
title: "PolynomialLRScheduler"
description: "Decays the learning rate using a polynomial function."
section: "API Reference"
---

`Models & Types` · `AiDotNet.LearningRateSchedulers`

Decays the learning rate using a polynomial function.

## For Beginners

This scheduler provides flexible control over how fast the learning
rate decreases. With power=1, it's a straight line decrease. With power=2, it decreases slowly
at first then more rapidly. With power=0.5, it decreases rapidly at first then slows down.
This flexibility lets you customize the decay curve to your specific training needs.

## How It Works

PolynomialLR decays the learning rate from the initial value to a minimum value using
a polynomial function. The decay curve can be controlled by the power parameter -
power=1 gives linear decay, power>1 gives faster initial decay, power<1 gives slower initial decay.

Formula: lr = (base_lr - end_lr) * (1 - step/total_steps)^power + end_lr

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PolynomialLRScheduler(Double,Int32,Double,Double)` | Initializes a new instance of the PolynomialLRScheduler class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EndLearningRate` | Gets the end learning rate. |
| `Power` | Gets the polynomial power. |
| `TotalSteps` | Gets the total number of steps. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeLearningRate(Int32)` |  |
| `GetState` |  |

