# Issue #284: RAG Framework Implementation Plan

## Executive Summary
This document outlines the comprehensive implementation plan for Issue #284: RAG Framework – Indexing, Retrieval, and Task Evals, based on deep analysis of the AiDotNet codebase using Google Gemini's 2M token context window.

## Issue Requirements Recap

**Background**: 2023–2025 RAG systems need robust indexing, retrieval, and evaluation of grounded answers.

**Problem**: No end-to-end RAG utilities (index build, query routing, grounded answer evaluation).

**User Story**: As an engineer, I want to build and evaluate RAG pipelines using the existing Transformer model as the generator.

**Acceptance Criteria**:
- Indexes: Interfaces for vector indexes (pluggable backend), chunking strategies; metadata filters
- Retrieval: Hybrid scoring (sparse+dense) interface; reranking hooks
- Generation: Context packing; citation extraction; grounded answer format
- Evals: Faithfulness, answer similarity, and context coverage metrics
- Docs: Patterns for enterprise data sources

**Dependencies**: Issue #12 (embeddings) recommended; can start with stubs
**Effort**: 2–3 weeks

---

## Architecture Analysis - Key Findings

### Existing Components to Leverage

1. **`Transformer.cs`** - Solid implementation that will serve as the generator in RAG pipeline
2. **`EmbeddingLayer.cs`** - Can create vector embeddings for text (core requirement for retrieval)
3. **`Vector.cs`** - Robust vector class with operations for similarity calculations
4. **`IModel.cs` & `IModelEvaluator.cs`** - Clear patterns for integrating RAG pipeline and evaluator
5. **`JsonConverterRegistry.cs`** - Existing serialization support for Vector, Matrix, Tensor

### Architecture Patterns Identified

- **Interface-driven design**: Strong emphasis on interfaces and abstractions
- **Generic components**: Modular architecture facilitates integration
- **Namespace organization**: Clear separation by functionality (Models, NeuralNetworks, Evaluation, etc.)
- **Serialization**: Comprehensive JSON serialization infrastructure
- **Testing**: Unit tests in `tests/UnitTests/` directory structure

---

## Proposed Implementation

### 1. Namespace and File Structure

**Namespace**: `AiDotNet.RetrievalAugmentedGeneration`

```
src/RetrievalAugmentedGeneration/
├── Interfaces/
│   ├── IDocumentStore.cs
│   ├── IRetriever.cs
│   ├── IReranker.cs
│   ├── IGenerator.cs
│   ├── ITextSplitter.cs
│   └── IChunkingStrategy.cs
├── Models/
│   ├── Document.cs
│   ├── VectorDocument.cs
│   ├── GroundedAnswer.cs
│   └── RetrievalContext.cs
├── Retrievers/
│   ├── VectorRetriever.cs
│   └── HybridRetriever.cs (sparse+dense)
├── DocumentStores/
│   └── InMemoryDocumentStore.cs
├── TextSplitters/
│   ├── CharacterTextSplitter.cs
│   └── RecursiveTextSplitter.cs
├── Rerankers/
│   ├── SimpleReranker.cs
│   └── CrossEncoderReranker.cs
├── Evaluation/
│   ├── RAGEvaluator.cs
│   ├── FaithfulnessMetric.cs
│   ├── AnswerSimilarityMetric.cs
│   └── ContextCoverageMetric.cs
└── RagPipeline.cs

tests/UnitTests/RetrievalAugmentedGeneration/
├── RagPipelineTests.cs
├── VectorRetrieverTests.cs
├── InMemoryDocumentStoreTests.cs
├── CharacterTextSplitterTests.cs
└── Evaluation/
    ├── FaithfulnessMetricTests.cs
    ├── AnswerSimilarityMetricTests.cs
    └── ContextCoverageMetricTests.cs
```

### 2. Class Hierarchy

```
[IModel] (existing)
  ^
  |
[IGenerator] (new interface)
  ^
  |
[Transformer] (existing - implements IGenerator)
  |
  +---- [RagPipeline] uses ----> [IRetriever]
                                    ^
                                    |
                                  [VectorRetriever] uses --> [IDocumentStore]
                                                               ^
                                                               |
                                                             [InMemoryDocumentStore] 
                                                                uses --> [Vector]
                                                                         [EmbeddingLayer]

[IModelEvaluator] (existing)
  ^
  |
[RAGEvaluator] (new)
  uses --> [FaithfulnessMetric]
           [AnswerSimilarityMetric]
           [ContextCoverageMetric]
```

### 3. Core Interface Definitions

#### IDocumentStore
```csharp
namespace AiDotNet.RetrievalAugmentedGeneration.Interfaces;

/// <summary>
/// Defines the contract for a document store that holds and retrieves vectorized documents.
/// Supports pluggable backends for different vector database implementations.
/// </summary>
public interface IDocumentStore
{
    /// <summary>
    /// Adds a collection of vectorized documents to the store.
    /// </summary>
    void Add(IEnumerable<VectorDocument> documents);

    /// <summary>
    /// Retrieves the top 'k' most similar documents to a given query vector.
    /// </summary>
    IEnumerable<Document> GetSimilar(Vector queryVector, int topK);

    /// <summary>
    /// Retrieves documents with metadata filtering.
    /// </summary>
    IEnumerable<Document> GetSimilarWithFilters(Vector queryVector, int topK, 
        IDictionary<string, object> metadataFilters);
}
```

#### IRetriever
```csharp
namespace AiDotNet.RetrievalAugmentedGeneration.Interfaces;

/// <summary>
/// Defines the contract for a retriever that fetches relevant documents based on a query.
/// Supports hybrid retrieval strategies (sparse + dense).
/// </summary>
public interface IRetriever
{
    /// <summary>
    /// Retrieves relevant documents for a given query string.
    /// </summary>
    IEnumerable<Document> Retrieve(string query);

    /// <summary>
    /// Retrieves relevant documents with custom top-k parameter.
    /// </summary>
    IEnumerable<Document> Retrieve(string query, int topK);
}
```

#### IReranker
```csharp
namespace AiDotNet.RetrievalAugmentedGeneration.Interfaces;

/// <summary>
/// Defines the contract for reranking retrieved documents.
/// </summary>
public interface IReranker
{
    /// <summary>
    /// Reranks documents based on relevance to the query.
    /// </summary>
    IEnumerable<Document> Rerank(string query, IEnumerable<Document> documents);
}
```

#### IGenerator
```csharp
namespace AiDotNet.RetrievalAugmentedGeneration.Interfaces;

/// <summary>
/// Defines the contract for a text generator, typically a large language model.
/// This will be implemented by the existing Transformer class.
/// </summary>
public interface IGenerator : IModel
{
    /// <summary>
    /// Generates a response based on a given prompt.
    /// </summary>
    string Generate(string prompt);

    /// <summary>
    /// Generates a response with citations extracted from context.
    /// </summary>
    GroundedAnswer GenerateGrounded(string prompt, IEnumerable<Document> context);
}
```

#### IChunkingStrategy
```csharp
namespace AiDotNet.RetrievalAugmentedGeneration.Interfaces;

/// <summary>
/// Defines the contract for text chunking strategies.
/// </summary>
public interface IChunkingStrategy
{
    /// <summary>
    /// Splits text into chunks according to the strategy.
    /// </summary>
    IEnumerable<string> Chunk(string text);

    /// <summary>
    /// Gets the chunk size used by this strategy.
    /// </summary>
    int ChunkSize { get; }

    /// <summary>
    /// Gets the overlap between chunks.
    /// </summary>
    int ChunkOverlap { get; }
}
```

### 4. Core Model Classes

#### Document
```csharp
namespace AiDotNet.RetrievalAugmentedGeneration.Models;

public class Document
{
    public string Id { get; set; }
    public string Content { get; set; }
    public Dictionary<string, object> Metadata { get; set; } = new();
    public double? RelevanceScore { get; set; }
}
```

#### VectorDocument
```csharp
namespace AiDotNet.RetrievalAugmentedGeneration.Models;

public class VectorDocument
{
    public Document Document { get; set; }
    public Vector Embedding { get; set; }
}
```

#### GroundedAnswer
```csharp
namespace AiDotNet.RetrievalAugmentedGeneration.Models;

public class GroundedAnswer
{
    public string Answer { get; set; }
    public IEnumerable<Document> SourceDocuments { get; set; }
    public IEnumerable<string> Citations { get; set; }
    public double ConfidenceScore { get; set; }
}
```

### 5. Implementation Phases

#### Phase 1: Core Infrastructure (Week 1, Days 1-3)
1. Create namespace and directory structure
2. Define all interfaces (IDocumentStore, IRetriever, IReranker, IGenerator, IChunkingStrategy)
3. Implement core model classes (Document, VectorDocument, GroundedAnswer)
4. Stub EmbeddingLayer for testing (simple hash-based or random vectors)

#### Phase 2: Indexing & Chunking (Week 1, Days 4-5)
5. Implement CharacterTextSplitter
6. Implement RecursiveTextSplitter
7. Implement InMemoryDocumentStore with cosine similarity
8. Write unit tests for text splitters and document store

#### Phase 3: Retrieval (Week 2, Days 1-3)
9. Implement VectorRetriever
10. Implement HybridRetriever (sparse + dense scoring)
11. Implement SimpleReranker (pass-through)
12. Add metadata filtering support to document store
13. Write unit tests for retrievers

#### Phase 4: Generation Integration (Week 2, Days 4-5)
14. Extend Transformer to implement IGenerator
15. Implement RagPipeline class
16. Implement context packing logic
17. Implement citation extraction
18. Write unit tests for RAG pipeline

#### Phase 5: Evaluation (Week 3, Days 1-3)
19. Implement RAGEvaluator
20. Implement FaithfulnessMetric
21. Implement AnswerSimilarityMetric
22. Implement ContextCoverageMetric
23. Write unit tests for all evaluation metrics

#### Phase 6: Documentation & Polish (Week 3, Days 4-5)
24. Write XML documentation for all public APIs
25. Create usage examples
26. Write integration tests
27. Create enterprise data source patterns documentation
28. Update README with RAG examples

### 6. Integration with Existing Transformer

The existing `Transformer.cs` will be extended to implement `IGenerator`:

```csharp
public class Transformer : NeuralNetworkBase, IGenerator
{
    // Existing code...

    public string Generate(string prompt)
    {
        // Adapter method that wraps existing Run/Predict logic
        // Convert string prompt -> input tensor
        // Execute transformer forward pass
        // Convert output tensor -> string response
    }

    public GroundedAnswer GenerateGrounded(string prompt, IEnumerable<Document> context)
    {
        // Construct augmented prompt with context
        var augmentedPrompt = BuildPromptWithContext(prompt, context);
        var answer = Generate(augmentedPrompt);
        
        // Extract citations from answer
        var citations = ExtractCitations(answer, context);
        
        return new GroundedAnswer
        {
            Answer = answer,
            SourceDocuments = context,
            Citations = citations
        };
    }
}
```

### 7. Handling Issue #12 Dependency (Embeddings)

**Stub Implementation Strategy**:

```csharp
// Temporary stub until Issue #12 is complete
namespace AiDotNet.NeuralNetworks.Layers;

public class EmbeddingLayer
{
    private readonly int _dimension;
    
    public EmbeddingLayer(int dimension = 768)
    {
        _dimension = dimension;
    }
    
    public Vector Embed(string text)
    {
        // Simple deterministic hash-based embedding for testing
        // Will be replaced with real semantic embeddings from Issue #12
        var hash = text.GetHashCode();
        var random = new Random(hash);
        var values = new double[_dimension];
        for (int i = 0; i < _dimension; i++)
        {
            values[i] = random.NextDouble();
        }
        var vector = new Vector(values);
        return vector.Normalize(); // Return unit vector
    }
}
```

**Migration Path**:
- Once Issue #12 completes, replace stub with real implementation
- Interface remains identical: `Vector Embed(string text)`
- No changes needed to RAG framework code

### 8. Testing Strategy

Following existing patterns in `AiDotNetTests`:

**Unit Tests**:
- Each class has corresponding test class
- Mock dependencies using interfaces
- Test edge cases and error conditions

**Integration Tests**:
- End-to-end RAG pipeline testing
- Test with sample documents and queries
- Verify retrieval accuracy
- Validate generation quality

**Performance Tests**:
- Benchmark vector similarity search
- Test with large document collections
- Memory usage profiling

### 9. Evaluation Metrics Implementation

#### Faithfulness Metric
Measures if generated answer is supported by retrieved context.
- Uses NLI (Natural Language Inference) approach
- Breaks answer into claims
- Verifies each claim against context

#### Answer Similarity Metric
Measures semantic similarity between generated answer and ground truth.
- Uses embedding similarity (cosine)
- Falls back to BLEU/ROUGE if embeddings unavailable

#### Context Coverage Metric
Measures how much of retrieved context was used in answer.
- Token overlap analysis
- Identifies unused context segments

### 10. Next Steps & Discussion Points

**Questions for Discussion**:

1. **Embedding Strategy**: Should we wait for Issue #12 or proceed with stubs? Any preference for stub implementation?

2. **Vector Database Backend**: Start with in-memory only, or should we plan for pluggable backends (e.g., FAISS, Milvus) from the start?

3. **Hybrid Retrieval**: What sparse retrieval method should we support? BM25? TF-IDF?

4. **Reranking**: Should we implement cross-encoder reranking immediately or start with simple pass-through?

5. **Metadata Filtering**: What filter operators do we need? (equals, contains, range, etc.)

6. **Citation Format**: How should citations be formatted in the grounded answer? Numbered references? Inline?

7. **Testing Data**: Do we have sample enterprise data sources for testing, or should we create synthetic datasets?

8. **Performance Targets**: Any specific requirements for retrieval latency or throughput?

9. **Serialization**: Should RAG pipeline state be serializable (for save/load)?

10. **Multi-threading**: Should retrieval support parallel processing for large document sets?

---

## Conclusion

This implementation plan provides a comprehensive roadmap for building a production-ready RAG framework that integrates seamlessly with the existing AiDotNet architecture. The modular, interface-driven design allows for incremental implementation and testing while maintaining compatibility with existing code patterns.

The 2-3 week estimate is achievable given the clear architectural foundation and existing utilities (Vector, Transformer, serialization) that can be leveraged.

**Ready to proceed with detailed implementation upon approval of architectural decisions.**
