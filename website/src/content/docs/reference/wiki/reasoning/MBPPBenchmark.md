---
title: "MBPPBenchmark<T>"
description: "MBPP (Mostly Basic Python Problems) benchmark for evaluating Python code generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Reasoning.Benchmarks`

MBPP (Mostly Basic Python Problems) benchmark for evaluating Python code generation.

## For Beginners

MBPP is a collection of basic Python programming problems,
designed to test fundamental programming skills and algorithmic thinking.

**What is MBPP?**
MBPP contains 974 short Python programming problems at an entry-level difficulty.
Each problem includes:

- A natural language description
- Code solution
- 3 test cases

**MBPP vs HumanEval:**

- **MBPP**: 974 problems, more basic, multiple test cases provided
- **HumanEval**: 164 problems, more challenging, function signature given
- **Overlap**: Both test code generation, but MBPP is more comprehensive

**Example problems:**

*Problem 1: Sum of numbers*
```
Task: Write a function to find the sum of all numbers in a list.
Test Cases:

- sum_list([1, 2, 3]) == 6
- sum_list([10, 20]) == 30
- sum_list([]) == 0

```

*Problem 2: Check palindrome*
```
Task: Write a function to check if a string is a palindrome.
Test Cases:

- is_palindrome("racecar") == True
- is_palindrome("hello") == False
- is_palindrome("") == True

```

*Problem 3: Remove duplicates*
```
Task: Write a function to remove duplicate elements from a list.
Test Cases:

- remove_duplicates([1, 2, 2, 3]) == [1, 2, 3]
- remove_duplicates([]) == []
- remove_duplicates([1, 1, 1]) == [1]

```

**Categories:**

- List operations (sorting, filtering, searching)
- String manipulation
- Mathematical operations
- Basic algorithms (searching, sorting)
- Data structure operations (lists, dictionaries)
- Boolean logic

**Difficulty levels:**

- Basic: Simple operations (50% of problems)
- Intermediate: Multiple steps (40% of problems)
- Advanced: Complex logic (10% of problems)

**Performance levels:**

- GPT-3 (Codex): ~59%
- GPT-3.5: ~70%
- GPT-4: ~82%
- Claude 3 Opus: ~78%
- Claude 3.5 Sonnet: ~85%
- ChatGPT o1: ~90%
- AlphaCode: ~75%
- CodeGen: ~65%

**Why it's useful:**

- Tests basic programming competency
- More comprehensive than HumanEval (974 vs 164)
- Includes test cases (can verify correctness)
- Entry-level difficulty (good for beginners)
- Real-world relevance (common programming tasks)

**Research:**

- "Program Synthesis with Large Language Models" (Austin et al., 2021)
- https://arxiv.org/abs/2108.07732
- Dataset: 974 problems with solutions and test cases
- Used by Google Research for code generation evaluation

**Integration with CodeExecutionVerifier:**
MBPP works particularly well with CodeExecutionVerifier since each problem
includes test cases that can be executed to verify correctness.

## Properties

| Property | Summary |
|:-----|:--------|
| `BenchmarkName` |  |
| `Description` |  |
| `TotalProblems` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateAsync(Func<String,Task<String>>,Nullable<Int32>,CancellationToken)` |  |
| `LoadProblemsAsync(Nullable<Int32>)` |  |

