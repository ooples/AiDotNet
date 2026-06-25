---
title: "IFewShotExampleSelector<T>"
description: "Defines the contract for selecting few-shot examples to include in prompts."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for selecting few-shot examples to include in prompts.

## For Beginners

An example selector picks which examples to show the language model.

Think of it like a teacher choosing practice problems:

- You have 100 practice problems in the textbook
- You can only show 3-5 examples in class
- Which ones do you choose?

Different strategies:

- Random: Pick any 3 problems
- Similar to homework: Pick problems like tonight's homework
- Diverse: Pick problems covering different concepts
- Best of both: Pick relevant problems that are also diverse

The examples you choose significantly affect how well students (or the LLM) learn!

Example - Sentiment classification:
Available examples: 1,000 labeled movie reviews
Current query: "This movie was fantastic, loved every minute!"

Selector's job:

1. Look at the query
2. Choose 3-5 most helpful examples
3. Return them to include in the prompt

Good examples → Better LLM performance

## How It Works

A few-shot example selector chooses which examples to include in a prompt to guide the language model's behavior.
Different selection strategies (random, semantic similarity, diversity, MMR) optimize for different goals.

## Methods

| Method | Summary |
|:-----|:--------|
| `AddExample(FewShotExample)` | Adds an example to the selector's pool of available examples. |
| `GetAllExamples` | Gets all examples currently in the selector's pool. |
| `RemoveExample(FewShotExample)` | Removes an example from the selector's pool. |
| `SelectExamples(String,Int32)` | Selects the most appropriate examples for the given query. |

