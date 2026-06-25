---
title: "ReasoningConfig"
description: "Configuration options for reasoning strategies that control how problems are solved."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Reasoning.Models`

Configuration options for reasoning strategies that control how problems are solved.

## For Beginners

Think of this class as a control panel with knobs and switches that adjust
how the AI thinks about problems. Just like you might adjust the difficulty level in a video game,
these settings let you control things like:

- How many steps the AI should take when thinking
- How thoroughly it should explore different solution paths
- Whether it should verify its work
- How much computing power to use

Different problems might need different settings. A simple math problem might only need a few steps,
while a complex reasoning task might benefit from exploring many different approaches.

## How It Works

**Example Usage:**

## Properties

| Property | Summary |
|:-----|:--------|
| `BeamWidth` | Beam width for beam search algorithms. |
| `BranchingFactor` | Number of alternative thoughts to generate at each step (for Tree-of-Thoughts). |
| `ComputeScalingFactor` | Multiplier for compute resources based on estimated problem difficulty (1.0 = baseline, 2.0 = double). |
| `EnableContradictionDetection` | Whether to enable contradiction detection across reasoning steps. |
| `EnableDiversitySampling` | Whether to enable diversity sampling for exploring varied reasoning paths. |
| `EnableExternalVerification` | Whether to enable external tool verification (calculators, code execution, etc.). |
| `EnableSelfRefinement` | Whether to enable self-refinement when verification fails. |
| `EnableTestTimeCompute` | Whether to enable test-time compute scaling (adaptive computation based on problem difficulty). |
| `EnableVerification` | Whether to enable step-by-step verification with critic models. |
| `ExplorationDepth` | Maximum depth for tree-based reasoning strategies (Tree-of-Thoughts, MCTS). |
| `MaxReasoningTimeSeconds` | Maximum total reasoning time in seconds (0 = no limit). |
| `MaxRefinementAttempts` | Maximum number of refinement attempts per step. |
| `MaxSteps` | Maximum number of reasoning steps to generate. |
| `NumSamples` | Number of independent reasoning attempts for self-consistency (majority voting). |
| `Temperature` | Temperature for sampling diverse reasoning paths (0.0 = deterministic, 1.0+ = creative). |
| `VerificationThreshold` | Minimum verification score to accept a reasoning step (0.0 to 1.0). |

