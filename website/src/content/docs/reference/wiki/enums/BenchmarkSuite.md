---
title: "BenchmarkSuite"
description: "Defines the supported benchmark suites available through the AiDotNet facade."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the supported benchmark suites available through the AiDotNet facade.

## For Beginners

A benchmark suite is like a standardized test you can run to measure how well your
model performs on a specific family of problems. You select a suite using this enum, and AiDotNet runs
the benchmark and returns a structured report.

## Fields

| Field | Summary |
|:-----|:--------|
| `ARCAGI` | ARC-AGI - abstract reasoning puzzles. |
| `BoolQ` | BoolQ - yes/no question answering. |
| `CIFAR10` | CIFAR-10 - federated image classification (synthetic partitioning of standard CIFAR-10). |
| `CIFAR100` | CIFAR-100 - federated image classification (synthetic partitioning of standard CIFAR-100). |
| `CommonsenseQA` | CommonsenseQA - commonsense multiple-choice QA. |
| `DROP` | DROP - reading comprehension with discrete reasoning over paragraphs. |
| `FEMNIST` | FEMNIST - LEAF federated handwritten character classification (per-writer partitioning). |
| `GSM8K` | Grade School Math 8K (GSM8K) - multi-step math word problems. |
| `HellaSwag` | HellaSwag - commonsense inference in narrative completion. |
| `HumanEval` | HumanEval - code generation / program synthesis evaluation. |
| `LEAF` | LEAF - federated benchmark suite (JSON-based train/test splits). |
| `LogiQA` | LogiQA - logical reasoning benchmark. |
| `MATH` | MATH - competition-style mathematics problems. |
| `MBPP` | MBPP - mostly basic programming problems. |
| `MMLU` | MMLU - broad multi-subject multiple-choice benchmark. |
| `PIQA` | PIQA - physical commonsense reasoning. |
| `Reddit` | Reddit - federated next-token prediction benchmark (LEAF Reddit dataset). |
| `Sent140` | Sent140 - LEAF federated sentiment classification benchmark based on tweets. |
| `Shakespeare` | Shakespeare - LEAF federated next-character prediction benchmark. |
| `StackOverflow` | StackOverflow - federated next-token prediction benchmark (StackOverflow corpus). |
| `TabularNonIID` | Generic tabular suite with synthetic non-IID client partitions. |
| `TruthfulQA` | TruthfulQA - evaluates truthfulness and resistance to hallucination. |
| `WinoGrande` | WinoGrande - pronoun resolution / commonsense reasoning. |

