---
title: "Reasoning"
description: "All 41 public types in the AiDotNet.reasoning namespace, organized by kind."
section: "API Reference"
---

**41** public types in this namespace, organized by kind.

## Models & Types (33)

| Type | Summary |
|:-----|:--------|
| [`ARCAGIBenchmark<T>`](/docs/reference/wiki/reasoning/arcagibenchmark/) | ARC-AGI (Abstract Reasoning Corpus - Artificial General Intelligence) benchmark. |
| [`BenchmarkProblem`](/docs/reference/wiki/reasoning/benchmarkproblem/) | Represents a single problem in a benchmark dataset. |
| [`BenchmarkResult<T>`](/docs/reference/wiki/reasoning/benchmarkresult/) | Results from evaluating a reasoning system on a benchmark. |
| [`BoolQBenchmark<T>`](/docs/reference/wiki/reasoning/boolqbenchmark/) | BoolQ benchmark for evaluating yes/no question answering. |
| [`ChainOfThoughtStrategy<T>`](/docs/reference/wiki/reasoning/chainofthoughtstrategy/) | Implements Chain-of-Thought (CoT) reasoning that solves problems through explicit step-by-step thinking. |
| [`CodeReasoner<T>`](/docs/reference/wiki/reasoning/codereasoner/) |  |
| [`CodeXGlueBenchmark<T>`](/docs/reference/wiki/reasoning/codexgluebenchmark/) | CodeXGLUE benchmark harness (dataset-loader + metric computation). |
| [`CodeXGlueProblem`](/docs/reference/wiki/reasoning/codexglueproblem/) | A single CodeXGLUE problem record (task-agnostic). |
| [`CommonsenseQABenchmark<T>`](/docs/reference/wiki/reasoning/commonsenseqabenchmark/) | CommonsenseQA benchmark for evaluating commonsense knowledge and reasoning. |
| [`DROPBenchmark<T>`](/docs/reference/wiki/reasoning/dropbenchmark/) | DROP (Discrete Reasoning Over Paragraphs) benchmark for numerical and discrete reasoning. |
| [`GSM8KBenchmark<T>`](/docs/reference/wiki/reasoning/gsm8kbenchmark/) | Grade School Math 8K (GSM8K) benchmark for evaluating mathematical reasoning. |
| [`GSM8KDataLoader`](/docs/reference/wiki/reasoning/gsm8kdataloader/) | Loader for GSM8K benchmark dataset. |
| [`GSM8KProblem`](/docs/reference/wiki/reasoning/gsm8kproblem/) |  |
| [`HellaSwagBenchmark<T>`](/docs/reference/wiki/reasoning/hellaswagbenchmark/) | HellaSwag benchmark for evaluating commonsense natural language inference. |
| [`HumanEvalBenchmark<T>`](/docs/reference/wiki/reasoning/humanevalbenchmark/) | HumanEval benchmark for evaluating Python code generation capabilities. |
| [`HumanEvalDataLoader`](/docs/reference/wiki/reasoning/humanevaldataloader/) | Loader for HumanEval benchmark dataset. |
| [`HumanEvalProblem`](/docs/reference/wiki/reasoning/humanevalproblem/) |  |
| [`LogiQABenchmark<T>`](/docs/reference/wiki/reasoning/logiqabenchmark/) | LogiQA benchmark for evaluating logical reasoning abilities. |
| [`LogicalReasoner<T>`](/docs/reference/wiki/reasoning/logicalreasoner/) |  |
| [`MATHBenchmark<T>`](/docs/reference/wiki/reasoning/mathbenchmark/) | MATH benchmark for evaluating advanced mathematical reasoning. |
| [`MBPPBenchmark<T>`](/docs/reference/wiki/reasoning/mbppbenchmark/) | MBPP (Mostly Basic Python Problems) benchmark for evaluating Python code generation. |
| [`MMLUBenchmark<T>`](/docs/reference/wiki/reasoning/mmlubenchmark/) | MMLU (Massive Multitask Language Understanding) benchmark for evaluating world knowledge. |
| [`MathematicalReasoner<T>`](/docs/reference/wiki/reasoning/mathematicalreasoner/) | Specialized reasoner for mathematical problems using verified reasoning and external verification. |
| [`PIQABenchmark<T>`](/docs/reference/wiki/reasoning/piqabenchmark/) | PIQA (Physical Interaction Question Answering) benchmark for physical commonsense reasoning. |
| [`ProblemEvaluation<T>`](/docs/reference/wiki/reasoning/problemevaluation/) | Result for a single problem evaluation. |
| [`ReasoningChain<T>`](/docs/reference/wiki/reasoning/reasoningchain/) | Represents a complete chain of reasoning steps from problem to solution. |
| [`ReasoningResult<T>`](/docs/reference/wiki/reasoning/reasoningresult/) | Represents the complete result of a reasoning process, including the answer, reasoning chain, and performance metrics. |
| [`ReasoningStep<T>`](/docs/reference/wiki/reasoning/reasoningstep/) | Represents a single step in a reasoning chain, capturing the thought process and evaluation. |
| [`ScientificReasoner<T>`](/docs/reference/wiki/reasoning/scientificreasoner/) |  |
| [`SelfConsistencyStrategy<T>`](/docs/reference/wiki/reasoning/selfconsistencystrategy/) | Implements Self-Consistency reasoning by sampling multiple reasoning paths and using majority voting. |
| [`TreeOfThoughtsStrategy<T>`](/docs/reference/wiki/reasoning/treeofthoughtsstrategy/) | Implements Tree-of-Thoughts (ToT) reasoning that explores multiple reasoning paths in a tree structure. |
| [`TruthfulQABenchmark<T>`](/docs/reference/wiki/reasoning/truthfulqabenchmark/) | TruthfulQA benchmark for evaluating truthfulness and resistance to falsehoods. |
| [`WinoGrandeBenchmark<T>`](/docs/reference/wiki/reasoning/winograndebenchmark/) | WinoGrande benchmark for evaluating commonsense reasoning through pronoun resolution. |

## Base Classes (1)

| Type | Summary |
|:-----|:--------|
| [`ReasoningStrategyBase<T>`](/docs/reference/wiki/reasoning/reasoningstrategybase/) | Abstract base class for reasoning strategies that solve problems through structured thinking. |

## Enums (2)

| Type | Summary |
|:-----|:--------|
| [`ReasoningMode`](/docs/reference/wiki/reasoning/reasoningmode/) | Available reasoning modes that determine how problems are solved. |
| [`SearchAlgorithmType`](/docs/reference/wiki/reasoning/searchalgorithmtype/) | Types of search algorithms available for Tree-of-Thoughts. |

## Options & Configuration (3)

| Type | Summary |
|:-----|:--------|
| [`CodeXGlueBenchmarkOptions`](/docs/reference/wiki/reasoning/codexgluebenchmarkoptions/) | Options for configuring CodeXGLUE dataset loading. |
| [`HumanEvalBenchmarkOptions`](/docs/reference/wiki/reasoning/humanevalbenchmarkoptions/) |  |
| [`ReasoningConfig`](/docs/reference/wiki/reasoning/reasoningconfig/) | Configuration options for reasoning strategies that control how problems are solved. |

## Helpers & Utilities (2)

| Type | Summary |
|:-----|:--------|
| [`CodeXGlueDataLoader`](/docs/reference/wiki/reasoning/codexgluedataloader/) | Loads CodeXGLUE-style datasets from JSONL (one JSON object per line). |
| [`ThoughtNode<T>`](/docs/reference/wiki/reasoning/thoughtnode/) | Represents a node in a tree of thoughts, used for exploring multiple reasoning paths. |

