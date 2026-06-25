---
title: "WinoGrandeBenchmark<T>"
description: "WinoGrande benchmark for evaluating commonsense reasoning through pronoun resolution."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Reasoning.Benchmarks`

WinoGrande benchmark for evaluating commonsense reasoning through pronoun resolution.

## For Beginners

WinoGrande tests whether AI can figure out what pronouns
(like "it", "they", "he", "she") refer to in sentences, which requires common sense.

**What is WinoGrande?**
Based on the classic Winograd Schema Challenge, WinoGrande presents sentences with
pronouns where you need common sense to understand what the pronoun refers to.

**Example 1:**
```
Sentence: "The trophy doesn't fit in the suitcase because it is too big."
Question: What is too big?
A) The trophy
B) The suitcase
Answer: A (The trophy is too big to fit)
```

**Example 2:**
```
Sentence: "The trophy doesn't fit in the suitcase because it is too small."
Question: What is too small?
A) The trophy
B) The suitcase
Answer: B (The suitcase is too small to hold the trophy)
```

Notice how just changing one word ("big" → "small") completely flips the answer!

**Example 3:**
```
Sentence: "The city councilmen refused the demonstrators a permit because they feared violence."
Question: Who feared violence?
A) The city councilmen
B) The demonstrators
Answer: A (The councilmen feared, so they refused)
```

**Example 4:**
```
Sentence: "The city councilmen refused the demonstrators a permit because they advocated violence."
Question: Who advocated violence?
A) The city councilmen
B) The demonstrators
Answer: B (The demonstrators advocated, so permit was refused)
```

**Why it's called "Winograd"?**
Named after Terry Winograd, a pioneer in natural language understanding who created
the original Winograd Schema Challenge in 1972 as a better alternative to the Turing Test.

**Why it requires common sense:**

- Can't be solved by word associations alone
- Need to understand cause and effect
- Require knowledge about how the world works
- Must reason about physical properties, social situations, etc.

**Performance levels:**

- Random guessing: 50%
- Humans: 94.0%
- BERT: 59.4%
- RoBERTa: 79.1%
- GPT-3: 70.2%
- GPT-4: 87.5%
- Claude 3 Opus: 86.8%
- Claude 3.5 Sonnet: 88.5%
- ChatGPT o1: 90.8%

**WinoGrande improvements over original:**

- 44,000 examples (vs 273 in original)
- Adversarially generated to be harder
- More diverse scenarios
- Less prone to statistical biases

**Research:**

- "WinoGrande: An Adversarial Winograd Schema Challenge at Scale" (Sakaguchi et al., 2020)
- https://arxiv.org/abs/1907.10641
- Dataset: 44,000 problems with adversarial filtering
- Part of SuperGLUE benchmark

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

