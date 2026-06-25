---
title: "DROPBenchmark<T>"
description: "DROP (Discrete Reasoning Over Paragraphs) benchmark for numerical and discrete reasoning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Reasoning.Benchmarks`

DROP (Discrete Reasoning Over Paragraphs) benchmark for numerical and discrete reasoning.

## For Beginners

DROP tests whether AI can read a paragraph and answer questions
that require counting, comparing, sorting, or doing arithmetic on numbers in the text.

**What is DROP?**
DROP presents passages containing numbers, dates, and quantities, then asks questions
requiring discrete reasoning operations on this information.

**Question types:**

*1. Addition/Subtraction:*
```
Passage: "In 2019, the company had 500 employees. In 2020, they hired 150 more
and 50 left."

Q: How many employees did they have at the end of 2020?
A: 600 (500 + 150 - 50)
```

*2. Counting:*
```
Passage: "The team scored touchdowns in the 1st, 3rd, and 4th quarters."

Q: How many quarters did they score touchdowns in?
A: 3
```

*3. Comparison:*
```
Passage: "Team A scored 28 points. Team B scored 21 points."

Q: Which team scored more?
A: Team A
```

*4. Sorting:*
```
Passage: "Alice is 25, Bob is 30, and Carol is 22 years old."

Q: Who is oldest?
A: Bob
```

*5. Multi-step reasoning:*
```
Passage: "In the first half, they scored 14 points. In the third quarter,
they scored 7 more. In the fourth quarter, they scored 10 points."

Q: What was their total score?
A: 31 (14 + 7 + 10)
```

*6. Date arithmetic:*
```
Passage: "The war started in 1939 and ended in 1945."

Q: How long did the war last?
A: 6 years
```

**Why it's challenging:**

- Requires extracting multiple numbers from text
- Need to understand what operation to perform
- Must track relationships between entities
- Often requires multi-step reasoning
- Can't just pattern match - must actually compute

**Performance levels:**

- Humans: ~96% F1 score
- BERT: ~43% F1
- RoBERTa: ~58% F1
- GPT-3: ~52% F1
- GPT-4: ~79% F1
- Claude 3 Opus: ~77% F1
- Claude 3.5 Sonnet: ~82% F1
- ChatGPT o1: ~87% F1 (reasoning helps significantly)

**Reasoning operations:**

- Addition, subtraction
- Counting occurrences
- Finding maximum/minimum
- Sorting by value
- Comparing quantities
- Date/time arithmetic
- Percentage calculations

**Research:**

- "DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs" (Dua et al., 2019)
- https://arxiv.org/abs/1903.00161
- Dataset: 96,000 questions from Wikipedia passages
- Focus on numerical reasoning in natural language

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

