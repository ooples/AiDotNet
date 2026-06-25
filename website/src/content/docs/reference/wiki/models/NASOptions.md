---
title: "NASOptions<T>"
description: "Configuration options for Neural Architecture Search (NAS)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Neural Architecture Search (NAS).

## For Beginners

NAS automatically discovers optimal neural network architectures.
Instead of manually designing layers and connections, NAS explores the design space
to find architectures that maximize accuracy while meeting hardware constraints.

## How It Works

**Quick Start Example:**

**Available Strategies:**

- **DARTS:** Fast gradient-based search (~1-2 GPU days)
- **GDAS:** Improved DARTS with better discretization
- **OnceForAll:** Train once, deploy anywhere with elastic networks
- **NeuralArchitectureSearch:** Auto-selects best algorithm

## Properties

| Property | Summary |
|:-----|:--------|
| `ArchitectureLearningRate` | Gets or sets the learning rate for architecture parameters. |
| `CheckpointDirectory` | Gets or sets the checkpoint directory path. |
| `ElasticDepths` | Gets or sets the elastic depth values for OFA networks. |
| `ElasticExpansionRatios` | Gets or sets the elastic expansion ratios for OFA inverted residuals. |
| `ElasticKernelSizes` | Gets or sets the elastic kernel sizes for OFA networks. |
| `ElasticWidths` | Gets or sets the elastic width multipliers for OFA networks. |
| `Generations` | Gets or sets the number of generations for evolutionary search. |
| `HardwareConstraints` | Gets or sets the hardware constraints for the architecture search. |
| `InputChannels` | Gets or sets the number of input channels. |
| `MaxEpochs` | Gets or sets the maximum number of search epochs. |
| `MaxSearchTime` | Gets or sets the maximum time for the architecture search. |
| `MutationProbability` | Gets or sets the mutation probability for evolutionary search. |
| `NumClasses` | Gets or sets the number of output classes. |
| `OnEpochComplete` | Gets or sets the callback invoked after each search epoch. |
| `PopulationSize` | Gets or sets the population size for evolutionary search. |
| `QuantizationAware` | Gets or sets whether to use quantization-aware training during NAS. |
| `QuantizationMode` | Gets or sets the quantization mode for quantization-aware NAS. |
| `RandomSeed` | Gets or sets the random seed for reproducibility. |
| `SaveCheckpoints` | Gets or sets whether to save architecture checkpoints during search. |
| `SearchSpace` | Gets or sets the search space configuration. |
| `SpatialSize` | Gets or sets the spatial size of the input (assuming square inputs). |
| `Strategy` | Gets or sets the NAS strategy to use. |
| `TargetPlatform` | Gets or sets the target hardware platform for optimization. |
| `Verbose` | Gets or sets whether to enable verbose logging during search. |
| `WeightLearningRate` | Gets or sets the learning rate for network weights. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates the options and throws if any are invalid. |
| `With(Action<NASOptions<>>)` | Creates a copy of these options with the specified modifications. |

