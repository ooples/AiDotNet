---
title: "MMLUBenchmark<T>"
description: "MMLU (Massive Multitask Language Understanding) benchmark for evaluating world knowledge."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Reasoning.Benchmarks`

MMLU (Massive Multitask Language Understanding) benchmark for evaluating world knowledge.

## For Beginners

MMLU is like a comprehensive standardized test for AI,
covering 57 subjects from elementary to professional level.

**What is MMLU?**
MMLU tests knowledge across diverse academic and professional domains:

- STEM: Mathematics, Physics, Chemistry, Biology, Computer Science
- Humanities: History, Philosophy, Law
- Social Sciences: Psychology, Economics, Sociology
- Other: Medicine, Business, Professional Knowledge

**Format:**
Multiple choice questions (A, B, C, D) spanning different difficulty levels:

- Elementary
- High School
- College
- Professional

**Example questions:**

*Elementary Math:*
Q: What is 7 × 8?
A) 54 B) 56 C) 64 D) 48
Answer: B

*College Physics:*
Q: What is the ground state energy of a hydrogen atom?
A) -13.6 eV B) -27.2 eV C) -6.8 eV D) 0 eV
Answer: A

*Professional Medicine:*
Q: A 45-year-old presents with sudden chest pain. What is the most appropriate first test?
A) CT scan B) ECG C) Blood test D) X-ray
Answer: B

**Why it's important:**

- Comprehensive knowledge evaluation
- Tests reasoning + memorization
- Standard benchmark for LLMs
- Measures real-world applicability

**Performance levels:**

- Random guessing: 25%
- Average human expert: ~90% (in their domain)
- GPT-3.5: ~70%
- GPT-4: ~86%
- Claude 3 Opus: ~87%
- Claude 3.5 Sonnet: ~89%
- ChatGPT o1: ~91%
- Gemini Pro 1.5: ~90%

**57 Subject categories:**

**STEM (18 subjects):**

- Abstract Algebra, Astronomy, College Biology, College Chemistry
- College Computer Science, College Mathematics, College Physics
- Conceptual Physics, Electrical Engineering, Elementary Mathematics
- High School Biology, High School Chemistry, High School Computer Science
- High School Mathematics, High School Physics, High School Statistics
- Machine Learning

**Humanities (13 subjects):**

- Formal Logic, High School European History, High School US History
- High School World History, International Law, Jurisprudence
- Logical Fallacies, Moral Disputes, Moral Scenarios
- Philosophy, Prehistory, Professional Law, World Religions

**Social Sciences (12 subjects):**

- Econometrics, High School Geography, High School Government and Politics
- High School Macroeconomics, High School Microeconomics
- High School Psychology, Human Sexuality, Professional Psychology
- Public Relations, Security Studies, Sociology, US Foreign Policy

**Other (14 subjects):**

- Anatomy, Business Ethics, Clinical Knowledge, College Medicine
- Global Facts, Human Aging, Management, Marketing
- Medical Genetics, Miscellaneous, Nutrition, Professional Accounting
- Professional Medicine, Virology

**Research:**

- "Measuring Massive Multitask Language Understanding" (Hendrycks et al., 2021)
- https://arxiv.org/abs/2009.03300
- Dataset: 15,908 questions across 57 tasks

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

