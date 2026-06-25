---
title: "GraphGenerationTask<T>"
description: "Represents a graph generation task where the goal is to generate new valid graphs."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Structures`

Represents a graph generation task where the goal is to generate new valid graphs.

## For Beginners

Graph generation is like creating new objects that look realistic.

**Real-world examples:**

**Drug Discovery:**

- Task: Generate novel drug-like molecules
- Input: Training set of known drugs
- Output: New molecular structures with desired properties
- Goal: Discover new drug candidates automatically
- Example: Generate molecules that bind to a specific protein target

**Material Design:**

- Task: Generate new material structures
- Input: Database of materials with known properties
- Output: Novel material configurations
- Goal: Design materials with specific properties (strength, conductivity, etc.)

**Synthetic Data Generation:**

- Task: Create realistic social network graphs
- Input: Real social network data
- Output: Synthetic networks preserving statistical properties
- Goal: Generate data for testing while preserving privacy

**Molecular Optimization:**

- Task: Modify molecules to improve properties
- Input: Starting molecule
- Output: Similar molecules with better properties
- Example: Improve drug efficacy while maintaining safety

**Approaches:**

- **Autoregressive**: Generate nodes/edges one at a time
- **VAE**: Learn latent space of graphs, sample new ones
- **GAN**: Generator creates graphs, discriminator evaluates them
- **Flow-based**: Learn invertible transformations of graph distributions

## How It Works

Graph generation creates new graph structures that follow learned patterns from training data.
This is useful for generating novel molecules, designing new materials, creating synthetic
networks, and other generative tasks.

## Properties

| Property | Summary |
|:-----|:--------|
| `EdgeTypes` | Possible edge types/labels (for categorical edge features). |
| `GenerationBatchSize` | Number of graphs to generate per batch during training. |
| `GenerationMetrics` | Metrics to track during generation (e.g., validity rate, uniqueness, novelty). |
| `IsDirected` | Whether to generate directed graphs. |
| `MaxNumEdges` | Maximum number of edges allowed in generated graphs. |
| `MaxNumNodes` | Maximum number of nodes allowed in generated graphs. |
| `NodeTypes` | Possible node types/labels (for categorical node features). |
| `NumEdgeFeatures` | Number of edge feature dimensions (0 if no edge features). |
| `NumNodeFeatures` | Number of node feature dimensions. |
| `TrainingGraphs` | Training graphs used to learn the distribution. |
| `ValidationGraphs` | Validation graphs for monitoring generation quality. |
| `ValidityChecker` | Validity constraints for generated graphs. |

