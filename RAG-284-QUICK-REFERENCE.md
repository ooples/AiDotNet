# RAG Framework Architecture Quick Reference

## Component Overview

### 1. Core Interfaces

```
IDocumentStore       - Vector storage & similarity search (pluggable backends)
IRetriever          - Document retrieval interface (hybrid sparse+dense)
IReranker           - Document reranking hooks
IGenerator          - Text generation interface (implemented by Transformer)
IChunkingStrategy   - Text splitting strategies
```

### 2. Main Classes

```
RagPipeline          - Orchestrates retrieval -> generation flow
VectorRetriever      - Dense vector-based retrieval
HybridRetriever      - Combines sparse (BM25) + dense retrieval
InMemoryDocumentStore - In-memory vector index
Document             - Text document with metadata
VectorDocument       - Document + embedding
GroundedAnswer       - Generated answer + citations + source docs
RAGEvaluator         - Evaluates RAG quality (faithfulness, similarity, coverage)
```

### 3. Data Flow

```
Query (string)
    ↓
[Text Splitter] → Chunks
    ↓
[EmbeddingLayer] → Query Vector
    ↓
[IDocumentStore.GetSimilar()] → Retrieved Documents
    ↓
[IReranker.Rerank()] → Reranked Documents
    ↓
[Context Packing] → Augmented Prompt
    ↓
[Transformer.Generate()] → Generated Answer
    ↓
[Citation Extraction] → GroundedAnswer
    ↓
[RAGEvaluator] → Metrics (faithfulness, similarity, coverage)
```

### 4. Integration Points with Existing Code

| Existing Component | How RAG Uses It |
|-------------------|-----------------|
| `Transformer.cs` | Implements IGenerator for answer generation |
| `EmbeddingLayer.cs` | Converts text to vectors (Issue #12 dependency) |
| `Vector.cs` | Stores embeddings, computes similarity |
| `IModel` | RAGPipeline can wrap as IModel for ecosystem integration |
| `IModelEvaluator` | RAGEvaluator follows this pattern |
| `JsonConverterRegistry` | Serializes RAG components |

### 5. File Structure

```
src/RetrievalAugmentedGeneration/
├── Interfaces/
│   ├── IDocumentStore.cs
│   ├── IRetriever.cs
│   ├── IReranker.cs
│   ├── IGenerator.cs
│   └── IChunkingStrategy.cs
├── Models/
│   ├── Document.cs
│   ├── VectorDocument.cs
│   └── GroundedAnswer.cs
├── Retrievers/
│   ├── VectorRetriever.cs
│   └── HybridRetriever.cs
├── DocumentStores/
│   └── InMemoryDocumentStore.cs
├── TextSplitters/
│   ├── CharacterTextSplitter.cs
│   └── RecursiveTextSplitter.cs
├── Rerankers/
│   └── SimpleReranker.cs
├── Evaluation/
│   ├── RAGEvaluator.cs
│   ├── FaithfulnessMetric.cs
│   ├── AnswerSimilarityMetric.cs
│   └── ContextCoverageMetric.cs
└── RagPipeline.cs
```

### 6. Implementation Priority

**Must-Have (MVP)**:
- IDocumentStore, IRetriever, IGenerator interfaces
- InMemoryDocumentStore with cosine similarity
- VectorRetriever
- CharacterTextSplitter
- RagPipeline
- Basic RAGEvaluator with one metric

**Should-Have (Week 2)**:
- HybridRetriever (sparse+dense)
- Metadata filtering
- All three evaluation metrics
- Citation extraction
- Transformer integration

**Nice-to-Have (Week 3)**:
- CrossEncoderReranker
- RecursiveTextSplitter
- Advanced context packing
- Enterprise data source examples
- Performance optimizations

### 7. Key Design Decisions

1. **Namespace**: `AiDotNet.RetrievalAugmentedGeneration` (follows existing pattern)
2. **Embedding Dimension**: 768 (stub), configurable for real embeddings
3. **Default TopK**: 5 documents
4. **Similarity Metric**: Cosine similarity (can extend to others)
5. **Serialization**: Use existing JsonConverterRegistry pattern
6. **Testing**: Unit tests for each class, integration tests for full pipeline

### 8. Dependencies

**Hard Dependencies**:
- `AiDotNet.LinearAlgebra.Vector` (existing)
- `AiDotNet.NeuralNetworks.Transformer` (existing)
- `AiDotNet.Interfaces.IModel` (existing)

**Soft Dependencies**:
- `AiDotNet.NeuralNetworks.Layers.EmbeddingLayer` (Issue #12 - can stub)

### 9. Usage Example

```csharp
// Setup
var embeddingLayer = new EmbeddingLayer(dimension: 768);
var documentStore = new InMemoryDocumentStore(embeddingLayer);
var retriever = new VectorRetriever(documentStore, embeddingLayer, topK: 5);
var transformer = new Transformer(/* config */);
var pipeline = new RagPipeline(retriever, transformer);

// Index documents
var documents = new[] {
    new Document { Id = "1", Content = "AI is transforming healthcare...", Metadata = {...} },
    new Document { Id = "2", Content = "Machine learning improves diagnostics...", Metadata = {...} }
};
var vectorDocs = documents.Select(d => new VectorDocument {
    Document = d,
    Embedding = embeddingLayer.Embed(d.Content)
});
documentStore.Add(vectorDocs);

// Query
var query = "How is AI used in healthcare?";
var answer = pipeline.Execute(query);

// Evaluate
var evaluator = new RAGEvaluator();
var metrics = evaluator.Evaluate(query, answer, groundTruth);
```

### 10. Testing Strategy

```csharp
// Unit Test Example
[TestClass]
public class VectorRetrieverTests
{
    [TestMethod]
    public void Retrieve_ReturnsTopKDocuments()
    {
        // Arrange
        var mockStore = new Mock<IDocumentStore>();
        var mockEmbedding = new Mock<EmbeddingLayer>();
        var retriever = new VectorRetriever(mockStore.Object, mockEmbedding.Object, topK: 3);
        
        // Act
        var results = retriever.Retrieve("test query");
        
        // Assert
        Assert.AreEqual(3, results.Count());
        mockStore.Verify(s => s.GetSimilar(It.IsAny<Vector>(), 3), Times.Once);
    }
}
```

### 11. Evaluation Metrics Details

**Faithfulness** (Answer grounded in context?):
- Extract claims from answer
- Check each claim against retrieved context
- Score: % of claims supported by context

**Answer Similarity** (How close to ground truth?):
- Embedding cosine similarity if available
- Fall back to BLEU/ROUGE scores
- Score: 0-1 similarity metric

**Context Coverage** (How much context used?):
- Token overlap between answer and context
- Identify unused context segments
- Score: % of context tokens appearing in answer

### 12. Performance Considerations

**Vector Search**:
- In-memory: O(n) linear scan for MVP
- Future: O(log n) with FAISS/approximate nearest neighbor

**Context Packing**:
- Limit total tokens to transformer max length
- Prioritize top-ranked documents
- Truncate documents if needed

**Batch Processing**:
- Support batch embeddings
- Parallel document retrieval
- Cache query embeddings

### 13. Future Enhancements (Post-MVP)

- [ ] FAISS/Milvus vector database backends
- [ ] Cross-encoder reranking
- [ ] Multi-query retrieval strategies
- [ ] Document chunking with semantic boundaries
- [ ] Async/streaming generation
- [ ] Distributed document indexing
- [ ] Advanced citation formatting
- [ ] Query expansion/rephrasing
- [ ] Negative sampling for evaluation
- [ ] A/B testing framework

---

## Quick Decision Matrix

| Decision Point | Recommended Choice | Rationale |
|---------------|-------------------|-----------|
| Start with stubs? | Yes | Don't block on Issue #12 |
| Vector backend | In-memory first | Simplest for MVP, pluggable later |
| Sparse retrieval | BM25 | Industry standard for hybrid RAG |
| Reranking | Simple first | Add cross-encoder in Phase 2 |
| Metadata filters | Equality + range | Covers 80% of use cases |
| Citation format | Numbered refs | Easiest to parse and validate |
| Parallel retrieval | Not in MVP | Add after profiling shows need |
| Serialization | Yes | Consistent with existing patterns |

---

**Status**: Ready for implementation after architectural approval
**Next Step**: Discuss 10 key questions in main implementation plan
