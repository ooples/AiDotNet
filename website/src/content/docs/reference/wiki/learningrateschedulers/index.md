---
title: "Learning Rate Schedulers"
description: "All 24 public types in the AiDotNet.learningrateschedulers namespace, organized by kind."
section: "API Reference"
---

**24** public types in this namespace, organized by kind.

## Models & Types (14)

| Type | Summary |
|:-----|:--------|
| [`ConstantLRScheduler`](/docs/reference/wiki/learningrateschedulers/constantlrscheduler/) | Maintains a constant learning rate throughout training. |
| [`CosineAnnealingLRScheduler`](/docs/reference/wiki/learningrateschedulers/cosineannealinglrscheduler/) | Sets the learning rate using a cosine annealing schedule. |
| [`CosineAnnealingWarmRestartsScheduler`](/docs/reference/wiki/learningrateschedulers/cosineannealingwarmrestartsscheduler/) | Sets the learning rate using cosine annealing with warm restarts. |
| [`CyclicLRScheduler`](/docs/reference/wiki/learningrateschedulers/cycliclrscheduler/) | Implements cyclical learning rate policy. |
| [`ExponentialLRScheduler`](/docs/reference/wiki/learningrateschedulers/exponentiallrscheduler/) | Decays the learning rate exponentially every step. |
| [`LambdaLRScheduler`](/docs/reference/wiki/learningrateschedulers/lambdalrscheduler/) | Sets the learning rate using a user-defined lambda function. |
| [`LinearWarmupScheduler`](/docs/reference/wiki/learningrateschedulers/linearwarmupscheduler/) | Implements linear learning rate warmup followed by constant or decay schedule. |
| [`MultiStepLRScheduler`](/docs/reference/wiki/learningrateschedulers/multisteplrscheduler/) | Decays the learning rate by gamma at each milestone step. |
| [`NoamSchedule`](/docs/reference/wiki/learningrateschedulers/noamschedule/) | Implements the Noam learning rate schedule from "Attention Is All You Need" (Vaswani et al. |
| [`OneCycleLRScheduler`](/docs/reference/wiki/learningrateschedulers/onecyclelrscheduler/) | Implements the 1cycle learning rate policy. |
| [`PolynomialLRScheduler`](/docs/reference/wiki/learningrateschedulers/polynomiallrscheduler/) | Decays the learning rate using a polynomial function. |
| [`ReduceOnPlateauScheduler`](/docs/reference/wiki/learningrateschedulers/reduceonplateauscheduler/) | Reduces learning rate when a metric has stopped improving. |
| [`SequentialLRScheduler`](/docs/reference/wiki/learningrateschedulers/sequentiallrscheduler/) | Chains multiple learning rate schedulers together in sequence. |
| [`StepLRScheduler`](/docs/reference/wiki/learningrateschedulers/steplrscheduler/) | Decays the learning rate by a factor (gamma) every specified number of steps. |

## Base Classes (1)

| Type | Summary |
|:-----|:--------|
| [`LearningRateSchedulerBase`](/docs/reference/wiki/learningrateschedulers/learningrateschedulerbase/) | Base class for learning rate schedulers providing common functionality. |

## Interfaces (1)

| Type | Summary |
|:-----|:--------|
| [`ILearningRateScheduler`](/docs/reference/wiki/learningrateschedulers/ilearningratescheduler/) | Interface for learning rate schedulers that adjust the learning rate during training. |

## Enums (7)

| Type | Summary |
|:-----|:--------|
| [`AnnealingStrategy`](/docs/reference/wiki/learningrateschedulers/annealingstrategy/) | Annealing strategy for the decay phase. |
| [`CyclicMode`](/docs/reference/wiki/learningrateschedulers/cyclicmode/) | Mode for cyclic learning rate. |
| [`DecayMode`](/docs/reference/wiki/learningrateschedulers/decaymode/) | Decay mode after warmup phase. |
| [`LearningRateSchedulerType`](/docs/reference/wiki/learningrateschedulers/learningrateschedulertype/) | Enumeration of available learning rate scheduler types. |
| [`Mode`](/docs/reference/wiki/learningrateschedulers/mode/) | Optimization mode. |
| [`SchedulerStepMode`](/docs/reference/wiki/learningrateschedulers/schedulerstepmode/) | Specifies when the learning rate scheduler should be stepped during training. |
| [`ThresholdMode`](/docs/reference/wiki/learningrateschedulers/thresholdmode/) | Threshold comparison mode. |

## Helpers & Utilities (1)

| Type | Summary |
|:-----|:--------|
| [`LearningRateSchedulerFactory`](/docs/reference/wiki/learningrateschedulers/learningrateschedulerfactory/) | Factory for creating learning rate schedulers with common configurations. |

