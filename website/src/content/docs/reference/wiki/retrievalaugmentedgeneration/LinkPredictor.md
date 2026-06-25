---
title: "LinkPredictor<T>"
description: "Link prediction engine that uses trained KG embeddings to predict missing triples and evaluate model quality."
section: "API Reference"
---

`Models & Types` · `AiDotNet.RetrievalAugmentedGeneration.Graph.Embeddings`

Link prediction engine that uses trained KG embeddings to predict missing triples
and evaluate model quality.

## For Beginners

Link prediction fills in missing facts in a knowledge graph.

Example: Your graph has "Einstein born_in ?" — the predictor:

1. Scores every entity as a possible answer: (Einstein, born_in, Germany) = 0.95, (Einstein, born_in, France) = 0.12
2. Ranks them: Germany #1, Switzerland #2, Austria #3...
3. Returns the top-K most plausible completions

Evaluation (EvaluateModel) tests how well the model can "guess" known facts by temporarily
hiding them and checking if they rank highly.

## How It Works

Link prediction answers questions like "Given (Einstein, born_in, ?), what is the most likely tail entity?"
It works by scoring all candidate entities and ranking them by plausibility.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LinkPredictor(IKnowledgeGraphEmbedding<>)` | Creates a new link predictor using a trained embedding model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateModel(KnowledgeGraph<>,IEnumerable<ValueTuple<String,String,String>>,Int32[])` | Evaluates the embedding model using standard link prediction metrics (MRR, Hits@K, MeanRank) in the filtered setting. |
| `PredictHeads(KnowledgeGraph<>,String,String,Int32,Boolean)` | Predicts the most plausible head entities for a given (?, relation, tail) query. |
| `PredictTails(KnowledgeGraph<>,String,String,Int32,Boolean)` | Predicts the most plausible tail entities for a given (head, relation, ?) query. |

