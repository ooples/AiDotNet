# Basic RAG - Retrieval-Augmented Generation

This sample demonstrates how to build a question-answering system using Retrieval-Augmented Generation (RAG).

## What You'll Learn

- How to configure RAG with `PredictionModelBuilder`
- How to use vector stores for document embedding
- How to configure retrievers and rerankers
- How to generate answers from retrieved context

## What is RAG?

RAG combines retrieval and generation:
1. **Embed documents** into a vector store
2. **Retrieve** relevant documents for a query
3. **Generate** an answer using the retrieved context

This allows the model to answer questions about documents it wasn't trained on.

## Running the Sample

```bash
dotnet run
```

## Expected Output

```
=== AiDotNet Basic RAG ===

Loading documents...
  Loaded 5 documents about AI/ML topics

Building RAG pipeline...
  - Vector store: In-memory
  - Retriever: Dense retriever with top-k=3
  - Generator: LLM-based

Indexing documents...
  Indexed 5 documents

Query: "What is machine learning?"

Retrieved documents:
  1. [0.92] Machine learning is a subset of AI...
  2. [0.78] ML algorithms learn patterns from data...
  3. [0.71] Supervised learning uses labeled data...

Generated answer:
  Machine learning is a subset of artificial intelligence that enables
  systems to learn and improve from experience without being explicitly
  programmed. It focuses on developing algorithms that can access data
  and use it to learn for themselves.
```

## Code Highlights

```csharp
var result = await new PredictionModelBuilder<float, string, string>()
    .ConfigureRetrievalAugmentedGeneration(
        retriever: new DenseRetriever<float>(embeddingModel, topK: 3),
        reranker: new CrossEncoderReranker<float>(),
        generator: new LLMGenerator<float>(llmClient))
    .BuildAsync();

var answer = await result.Model.QueryAsync(question, documents);
```

## Architecture

```
┌─────────────────┐
│     Query       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Embedder      │ ← Convert query to vector
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Retriever     │ ← Find similar documents
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Reranker      │ ← Re-score for relevance
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Generator     │ ← Generate answer
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Answer      │
└─────────────────┘
```

## Next Steps

- [GraphRAG](../GraphRAG/) - Knowledge graph-enhanced RAG
- [Embeddings](../../Embeddings/) - Learn about text embeddings
