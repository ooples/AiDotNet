# Retrieval-Augmented Generation (RAG) Framework

## Overview

The RAG framework provides a complete implementation of retrieval-augmented generation for building AI systems that ground their responses in retrieved documents. This enables more accurate, verifiable, and trustworthy AI-generated content.

## Quick Start

```csharp
using AiDotNet.RetrievalAugmentedGeneration;
using AiDotNet.RetrievalAugmentedGeneration.Embeddings;
using AiDotNet.RetrievalAugmentedGeneration.DocumentStores;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;
using AiDotNet.RetrievalAugmentedGeneration.Rerankers;
using AiDotNet.RetrievalAugmentedGeneration.Generators;
using AiDotNet.RetrievalAugmentedGeneration.Models;

// 1. Create components
var embeddingModel = new StubEmbeddingModel<double>();
var documentStore = new InMemoryDocumentStore<double>();
var retriever = new VectorRetriever<double>(documentStore, embeddingModel);
var reranker = new IdentityReranker();
var generator = new StubGenerator();

// 2. Index some documents
var docs = new[]
{
    new Document("doc1", "Photosynthesis is the process by which plants convert sunlight into energy."),
    new Document("doc2", "Plants use chlorophyll to absorb light during photosynthesis."),
    new Document("doc3", "The products of photosynthesis are glucose and oxygen.")
};

foreach (var doc in docs)
{
    var embedding = embeddingModel.Embed(doc.Content);
    documentStore.Add(new VectorDocument<double>(doc, embedding));
}

// 3. Create RAG pipeline
var pipeline = new RagPipeline(retriever, reranker, generator);

// 4. Ask questions!
var answer = pipeline.Generate("What is photosynthesis?");

Console.WriteLine($"Answer: {answer.Answer}");
Console.WriteLine($"Confidence: {answer.ConfidenceScore:P0}");
Console.WriteLine($"\nSources:");
foreach (var citation in answer.Citations)
{
    Console.WriteLine($"  {citation}");
}
```

## Architecture

The RAG framework follows the **Interface → Base Class → Concrete Implementation** pattern:

### Core Interfaces

- **IEmbeddingModel<T>**: Text → Vector embedding
- **IDocumentStore<T>**: Vector storage and similarity search
- **IRetriever**: Document retrieval strategies
- **IReranker**: Relevance reranking
- **IGenerator**: Text generation with grounding
- **IChunkingStrategy**: Text splitting strategies

### Base Classes

All base classes use the Template Method pattern to provide common functionality while allowing customization:

- `EmbeddingModelBase<T>`
- `DocumentStoreBase<T>`
- `RetrieverBase`
- `RerankerBase`
- `ChunkingStrategyBase`

### MVP Implementations

Current implementations (Issue #284):

| Component | Implementation | Purpose |
|-----------|---------------|---------|
| Embedding | `StubEmbeddingModel<T>` | Hash-based deterministic embeddings for testing |
| Document Store | `InMemoryDocumentStore<T>` | Fast in-memory storage with cosine similarity |
| Retriever | `VectorRetriever<T>` | Dense vector similarity search |
| Reranker | `IdentityReranker` | Pass-through (no reranking) |
| Chunking | `FixedSizeChunkingStrategy` | Simple character-based splitting |
| Generator | `StubGenerator` | Template-based answer generation |

## Future Implementations

See follow-up issues for planned implementations:

- **Sparse Retrieval**: BM25, TF-IDF, ElasticSearch
- **Hybrid Retrieval**: Combining dense + sparse methods
- **Advanced Reranking**: CrossEncoder, ColBERT, MonoT5, LLM-based
- **Vector Databases**: FAISS, Milvus, Pinecone, Weaviate, Qdrant
- **Advanced Chunking**: Recursive, semantic, markdown-aware, code-aware
- **RAG Metrics**: Precision, recall, BLEU, ROUGE, BERTScore
- **Real Embeddings**: Integration with Issue #12 (Transformer embeddings)

## Key Features

✅ **Generic Types**: Uses `Vector<T>` and `Matrix<T>` instead of arrays  
✅ **SOLID Principles**: Interface-driven design with dependency injection  
✅ **DRY**: Common functionality in base classes  
✅ **Extensible**: Easy to add new implementations  
✅ **Type-Safe**: Compile-time type checking with generics  
✅ **Well-Documented**: Comprehensive XML docs with beginner explanations  

## Data Flow

```
User Query
    ↓
[IChunkingStrategy] → Split into searchable chunks
    ↓
[IEmbeddingModel] → Convert to query vector
    ↓
[IDocumentStore.GetSimilar()] → Find top-K similar documents
    ↓
[IReranker.Rerank()] → Improve relevance ranking
    ↓
[Context Packing] → Prepare documents for generation
    ↓
[IGenerator.GenerateGrounded()] → Generate answer with citations
    ↓
GroundedAnswer (Answer + Sources + Citations + Confidence)
```

## Testing

The framework includes stub implementations specifically designed for testing:

- **StubEmbeddingModel**: Deterministic hash-based embeddings
- **StubGenerator**: Template-based generation
- **IdentityReranker**: Pass-through for baseline testing

These allow full pipeline testing without requiring production models.

## Performance Considerations

### InMemoryDocumentStore<T>

- **Best for**: < 100K documents
- **Search complexity**: O(n) linear scan (future: approximate nearest neighbor)
- **Memory**: ~1-2KB per document + embedding size
- **Typical**: 10K documents @ 768-dim = ~50MB RAM

### Scaling Up

For larger collections, implement:
- **FAISS**: Approximate nearest neighbor (millions of vectors)
- **Milvus**: Distributed vector database
- **Pinecone**: Managed cloud service

## Contributing

When adding new implementations:

1. Implement the interface
2. Extend the base class
3. Follow existing documentation patterns
4. Include "For Beginners" sections in XML docs
5. Add unit tests
6. Update this README

## Related Issues

- **Issue #284**: RAG Framework (this implementation)
- **Issue #12**: Transformer Embeddings (future integration)

See `FUTURE-ISSUES.md` for planned enhancements.
