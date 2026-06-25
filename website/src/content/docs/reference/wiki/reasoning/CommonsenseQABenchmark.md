---
title: "CommonsenseQABenchmark<T>"
description: "CommonsenseQA benchmark for evaluating commonsense knowledge and reasoning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Reasoning.Benchmarks`

CommonsenseQA benchmark for evaluating commonsense knowledge and reasoning.

## For Beginners

CommonsenseQA tests everyday knowledge that humans take
for granted but AI often struggles with.

**What is CommonsenseQA?**
CommonsenseQA contains multiple-choice questions requiring common sense about everyday
situations, objects, and concepts.

**Example questions:**

*Physical world:*
```
Q: Where would you put uncooked food that you want to cook soon?
A) pantry B) shelf C) refrigerator D) kitchen cabinet E) oven
Answer: C (refrigerator keeps food fresh until cooking)
```

*Social understanding:*
```
Q: What happens when people get tired?
A) they sleep B) go to movies C) feel energetic D) stay awake E) study
Answer: A (tired people need sleep)
```

*Cause and effect:*
```
Q: What can happen to someone who doesn't get enough sleep?
A) lazy B) insomnia C) get tired D) snore E) have fun
Answer: C (lack of sleep causes tiredness)
```

*Object properties:*
```
Q: What is likely to be found in a book?
A) pictures B) words C) pages D) cover E) all of the above
Answer: E (books have all these features)
```

*Spatial reasoning:*
```
Q: Where do you typically find a handle?
A) door B) briefcase C) suitcase D) cup E) all of the above
Answer: E (all these objects have handles)
```

**Knowledge types:**

- Physical properties (hot, cold, heavy, fragile)
- Spatial relationships (inside, on top of, next to)
- Temporal understanding (before, after, during)
- Causal relationships (causes, prevents, enables)
- Social norms (polite, rude, appropriate)
- Functional roles (what things are used for)
- Typical locations (where things are usually found)

**Why it's important:**

- Tests implicit knowledge humans use daily
- Can't be answered by facts alone
- Requires understanding of how the world works
- Foundation for real-world AI applications

**Performance levels:**

- Random guessing: 20%
- Humans (crowd workers): 88.9%
- Humans (expert): 95.3%
- BERT: 57.0%
- RoBERTa: 73.1%
- GPT-3: 65.2%
- GPT-4: 82.4%
- Claude 3 Opus: 81.7%
- Claude 3.5 Sonnet: 85.9%
- ChatGPT o1: 88.1%

**Why LLMs struggle:**

- Lack embodied experience (can't touch/see/hear)
- No direct interaction with physical world
- Must infer common sense from text alone
- Training data may lack obvious implicit knowledge
- Difficulty distinguishing common from rare situations

**How it's created:**

1. Start with concept from ConceptNet (knowledge graph)
2. Generate question about the concept
3. Use crowd workers to create wrong but plausible options
4. Adversarial filtering to ensure quality

**ConceptNet integration:**
Questions are based on ConceptNet relations like:

- UsedFor: knife UsedFor cutting
- AtLocation: book AtLocation library
- Causes: exercise Causes tiredness
- CapableOf: bird CapableOf flying

**Research:**

- "CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge" (Talmor et al., 2019)
- https://arxiv.org/abs/1811.00937
- Dataset: 12,247 questions with 5 answer choices each
- Based on ConceptNet knowledge graph

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

