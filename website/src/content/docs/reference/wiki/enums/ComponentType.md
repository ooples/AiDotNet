---
title: "ComponentType"
description: "Defines the type of an AI pipeline component (Tier 2 metadata)."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the type of an AI pipeline component (Tier 2 metadata).

## For Beginners

This tells you what kind of building block a component is.
Unlike models (which learn and predict), components transform, route, or process data
as part of a larger AI pipeline.

## Fields

| Field | Summary |
|:-----|:--------|
| `ActiveLearner` | Active learning strategies that select the most informative samples to label. |
| `BenchmarkUtility` | Benchmark and evaluation utilities. |
| `Chunker` | Splits documents into semantically meaningful chunks for indexing. |
| `ContextCompressor` | Compresses or filters retrieved context before generation. |
| `ContinualLearner` | Continual learning strategies that prevent catastrophic forgetting. |
| `CryptoPrimitive` | Cryptographic primitives used by secure computation protocols. |
| `DataLoader` | Data loading and dataset management components. |
| `DimensionReducer` | Dimensionality reduction components. |
| `DistillationStrategy` | Knowledge distillation strategies for model compression. |
| `DocumentStore` | Stores and retrieves documents or embeddings for a RAG pipeline. |
| `DomainAdapter` | Domain adaptation components for cross-domain learning. |
| `Encoder` | Categorical or feature encoding components. |
| `EntityRecognizer` | Named entity recognition components for document processing. |
| `Evaluator` | Evaluation metrics and benchmarking components. |
| `FeatureGenerator` | Feature generation/engineering components. |
| `FeatureSelector` | Feature selection components. |
| `FederatedAggregator` | Federated learning aggregation strategies. |
| `FederatedTrainer` | Federated learning trainers that coordinate distributed training. |
| `FederatedUnlearner` | Federated unlearning strategies for selective data removal. |
| `Generator` | Generates responses using retrieved context and a language model. |
| `GraphPartitioner` | Graph partitioning and topology management components. |
| `MetaLearner` | Meta-learning algorithms that learn how to learn. |
| `Optimizer` | Optimization algorithms for training. |
| `PSIProtocol` | Private set intersection protocols for secure data matching. |
| `PersonalizationStrategy` | Personalization strategies for per-client model adaptation. |
| `PrivacyMechanism` | Privacy mechanisms for differential privacy and secure computation. |
| `QueryExpander` | Expands a single query into multiple related queries for broader retrieval. |
| `QueryProcessor` | Transforms or expands queries before retrieval. |
| `Regularizer` | Regularization techniques. |
| `Reranker` | Re-scores and reorders retrieved documents for improved relevance. |
| `Retriever` | Retrieves relevant documents or passages from a corpus given a query. |
| `Scaler` | Data scaling/normalization components. |
| `Scheduler` | Learning rate schedulers. |
| `TransferAlgorithm` | Transfer learning algorithms for domain adaptation. |
| `VectorIndex` | Searches vector embeddings for nearest neighbors. |
| `VerificationScheme` | Verification and commitment schemes for secure protocols. |

