---
title: "LogiQABenchmark<T>"
description: "LogiQA benchmark for evaluating logical reasoning abilities."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Reasoning.Benchmarks`

LogiQA benchmark for evaluating logical reasoning abilities.

## For Beginners

LogiQA tests whether AI can solve logic puzzles similar to
those on standardized tests like the LSAT or GRE.

**What is LogiQA?**
LogiQA contains logical reasoning questions from Chinese civil service exams, translated
to English. These questions test formal logic, deductive reasoning, and analytical skills.

**Question types:**

*1. Categorical reasoning:*
```
All programmers are logical.
Some logical people are creative.
John is a programmer.

Which must be true?
A) John is creative
B) John is logical
C) All creative people are programmers
D) Some programmers are creative
Answer: B
```

*2. Conditional reasoning:*
```
If it rains, the picnic will be cancelled.
The picnic was not cancelled.

What can we conclude?
A) It rained
B) It did not rain
C) The picnic happened
D) Cannot determine
Answer: B (contrapositive: not cancelled → not rain)
```

*3. Assumption identification:*
```
Premise: Companies that provide good customer service are successful.
Conclusion: Therefore, our company should invest in customer service.

What assumption is made?
A) Our company wants to be successful
B) Customer service is expensive
C) Other companies have good service
D) Success is important
Answer: A
```

*4. Weaken/Strengthen arguments:*
```
Argument: "This medicine works because 80% of patients improved."

Which weakens this argument?
A) The medicine is expensive
B) 80% is a high percentage
C) 85% of patients improve without medicine
D) The medicine has side effects
Answer: C (natural improvement rate is higher)
```

*5. Paradox resolution:*
```
Paradox: "Sales of umbrellas decreased, but rainfall increased."

Which explains this?
A) Umbrellas became more expensive
B) People stayed indoors more during rain
C) Rainfall occurred at night when stores are closed
D) Other rain gear became popular
Answer: Could be B, C, or D depending on context
```

**Logic types tested:**

- Deductive reasoning (must be true)
- Inductive reasoning (likely to be true)
- Abductive reasoning (best explanation)
- Formal logic (syllogisms, conditionals)
- Critical thinking (assumptions, flaws)

**Performance levels:**

- Random guessing: 25%
- Humans (average): ~65%
- Humans (trained in logic): ~85%
- BERT: 34.2%
- RoBERTa: 37.1%
- GPT-3: 29.8%
- GPT-4: 43.5%
- Claude 3 Opus: 44.2%
- Claude 3.5 Sonnet: 48.0%
- ChatGPT o1: 61.2% (significant improvement with CoT)

**Why it's hard:**

- Requires formal logical reasoning
- Can't rely on pattern matching
- Need to track complex relationships
- Must avoid logical fallacies
- Tests rigorous thinking, not just knowledge

**Research:**

- "LogiQA: A Challenge Dataset for Machine Reading Comprehension with Logical Reasoning" (Liu et al., 2020)
- https://arxiv.org/abs/2007.08124
- Dataset: 8,678 questions from Chinese exams
- Tests multiple types of logical reasoning

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

