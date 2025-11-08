# Junior Developer Implementation Guide: Issue #371
## RAG Retrieval Strategies - Comprehensive Unit Testing

### Overview
Test 11 retrieval strategies: Dense (vector), Sparse (keyword), Hybrid, Multi-query, Multi-vector, Parent-document, BM25, TF-IDF, ColBERT, Graph, and Vector retrievers. Implementations exist - ensure thorough testing.

---

## For Beginners: Retrieval Strategies

### The Problem
**Question:** "How does photosynthesis work?"
**Document Store:** 10,000 scientific articles

Which articles should we retrieve?

### Retrieval Strategies

**1. Dense Retrieval (Vector Similarity)**
- Convert query → embedding vector
- Find documents with similar embedding vectors
- Understands meaning: "photosynthesis" matches "plants convert sunlight"

**2. Sparse Retrieval (Keyword Matching - BM25/TF-IDF)**
- Look for exact keyword matches
- Fast but literal: Misses "sunlight conversion" when searching "photosynthesis"

**3. Hybrid Retrieval**
- Combine dense + sparse
- Best of both: Semantic understanding + keyword precision

**4. Multi-Query Retrieval**
- Expand "photosynthesis" → ["photosynthesis", "plant energy production", "chlorophyll function"]
- Retrieve for all variants, combine results

**5. Multi-Vector Retrieval**
- Create multiple embeddings per document
- Better for long/complex documents

**6. Parent-Document Retrieval**
- Retrieve small chunks, return full parent documents
- Precise retrieval, comprehensive context

**7. ColBERT (Contextualized Late Interaction)**
- Token-level embeddings for fine-grained matching

**8. Graph Retrieval**
- Use knowledge graph connections
- Already tested in Issue #342 (GraphRAG)

---

## What EXISTS

### Retriever Implementations (11 Total)

**Core:**
1. `RetrieverBase<T>` - Base class with common functionality
2. `DenseRetriever<T>` - Vector similarity search
3. `VectorRetriever<T>` - Alternative dense implementation

**Sparse:**
4. `BM25Retriever<T>` - Best Match 25 algorithm
5. `TFIDFRetriever<T>` - Term Frequency-Inverse Document Frequency

**Hybrid:**
6. `HybridRetriever<T>` - Combines dense + sparse

**Advanced:**
7. `MultiQueryRetriever<T>` - Query expansion/reformulation
8. `MultiVectorRetriever<T>` - Multiple embeddings per document
9. `ParentDocumentRetriever<T>` - Chunk-based with parent retrieval
10. `ColBERTRetriever<T>` - Contextualized late interaction
11. `GraphRetriever<T>` - Graph-based retrieval

---

## What's MISSING

### Test Coverage Gaps

**For Each Retriever:**
- Constructor validation
- Basic retrieval with valid query
- TopK parameter enforcement
- Metadata filtering integration
- Empty query handling
- Empty document store handling
- Relevance score assignment

**Strategy-Specific:**
- **BM25**: Term frequency calculations, IDF scoring
- **Hybrid**: Score fusion methods (RRF, weighted sum)
- **Multi-Query**: Query generation quality
- **Parent-Document**: Chunk-to-parent mapping

---

## Step-by-Step Implementation

### Step 1: Retriever Test Base

```csharp
// File: tests/RetrievalAugmentedGeneration/Retrievers/RetrieverTestBase.cs

using Xunit;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.RetrievalAugmentedGeneration.DocumentStores;
using AiDotNet.RetrievalAugmentedGeneration.Embeddings;

namespace AiDotNet.Tests.RetrievalAugmentedGeneration.Retrievers;

public abstract class RetrieverTestBase<TRetriever, T>
    where TRetriever : IRetriever<T>
{
    protected abstract TRetriever CreateRetriever(IDocumentStore<T> store);

    protected InMemoryDocumentStore<T> CreatePopulatedStore()
    {
        var store = new InMemoryDocumentStore<T>();
        var embeddingModel = new StubEmbeddingModel<T>(dimension: 384);

        var documents = new[]
        {
            new Document<T> { Id = "1", Content = "Machine learning is AI subset.", Metadata = new() },
            new Document<T> { Id = "2", Content = "Deep learning uses neural networks.", Metadata = new() },
            new Document<T> { Id = "3", Content = "NLP enables language understanding.", Metadata = new() }
        };

        foreach (var doc in documents)
        {
            var embedding = embeddingModel.Embed(doc.Content);
            store.Add(new VectorDocument<T> { Document = doc, Embedding = embedding });
        }

        return store;
    }

    [Fact]
    public void Retrieve_WithValidQuery_ReturnsResults()
    {
        var store = CreatePopulatedStore();
        var retriever = CreateRetriever(store);

        var results = retriever.Retrieve("machine learning", topK: 2).ToList();

        Assert.NotEmpty(results);
        Assert.True(results.Count <= 2);
    }

    [Fact]
    public void Retrieve_WithTopK_ReturnsCorrectCount()
    {
        var store = CreatePopulatedStore();
        var retriever = CreateRetriever(store);

        var results = retriever.Retrieve("test query", topK: 1).ToList();

        Assert.Single(results);
    }

    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    public void Retrieve_WithEmptyQuery_ThrowsArgumentException(string query)
    {
        var store = CreatePopulatedStore();
        var retriever = CreateRetriever(store);

        Assert.Throws<ArgumentException>(() => retriever.Retrieve(query, topK: 5));
    }

    [Fact]
    public void Retrieve_WithNullQuery_ThrowsArgumentException()
    {
        var store = CreatePopulatedStore();
        var retriever = CreateRetriever(store);

        Assert.Throws<ArgumentException>(() => retriever.Retrieve(null, topK: 5));
    }

    [Fact]
    public void Retrieve_WithMetadataFilters_AppliesFilters()
    {
        var store = new InMemoryDocumentStore<T>();
        var embeddingModel = new StubEmbeddingModel<T>(dimension: 384);

        var doc1 = new Document<T>
        {
            Id = "1",
            Content = "ML content",
            Metadata = new() { ["year"] = 2024, ["category"] = "AI" }
        };
        var doc2 = new Document<T>
        {
            Id = "2",
            Content = "ML content",
            Metadata = new() { ["year"] = 2020, ["category"] = "AI" }
        };

        store.Add(new VectorDocument<T> { Document = doc1, Embedding = embeddingModel.Embed(doc1.Content) });
        store.Add(new VectorDocument<T> { Document = doc2, Embedding = embeddingModel.Embed(doc2.Content) });

        var retriever = CreateRetriever(store);
        var filters = new Dictionary<string, object> { ["year"] = 2024 };

        var results = retriever.Retrieve("ML", topK: 5, filters).ToList();

        Assert.All(results, doc => Assert.Equal(2024, doc.Metadata["year"]));
    }
}
```

### Step 2: Dense Retriever Tests

```csharp
// File: tests/RetrievalAugmentedGeneration/Retrievers/DenseRetrieverTests.cs

using Xunit;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;
using AiDotNet.RetrievalAugmentedGeneration.DocumentStores;
using AiDotNet.RetrievalAugmentedGeneration.Embeddings;

namespace AiDotNet.Tests.RetrievalAugmentedGeneration.Retrievers;

public class DenseRetrieverTests : RetrieverTestBase<DenseRetriever<double>, double>
{
    protected override DenseRetriever<double> CreateRetriever(IDocumentStore<double> store)
    {
        var embeddingModel = new StubEmbeddingModel<double>(dimension: 384);
        return new DenseRetriever<double>(store, embeddingModel, defaultTopK: 5);
    }

    [Fact]
    public void Constructor_WithNullStore_ThrowsArgumentNullException()
    {
        var embeddingModel = new StubEmbeddingModel<double>(dimension: 384);

        Assert.Throws<ArgumentNullException>(() =>
            new DenseRetriever<double>(documentStore: null, embeddingModel, defaultTopK: 5));
    }

    [Fact]
    public void Constructor_WithNullEmbeddingModel_ThrowsArgumentNullException()
    {
        var store = new InMemoryDocumentStore<double>();

        Assert.Throws<ArgumentNullException>(() =>
            new DenseRetriever<double>(store, embeddingModel: null, defaultTopK: 5));
    }

    [Fact]
    public void Retrieve_UsesSimilaritySearch()
    {
        var store = CreatePopulatedStore();
        var retriever = CreateRetriever(store);

        // Query semantically related to first document
        var results = retriever.Retrieve("AI and ML", topK: 3).ToList();

        Assert.NotEmpty(results);
        Assert.All(results, doc => Assert.True(doc.HasRelevanceScore));
    }
}
```

### Step 3: BM25 Retriever Tests

```csharp
// File: tests/RetrievalAugmentedGeneration/Retrievers/BM25RetrieverTests.cs

using Xunit;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;

namespace AiDotNet.Tests.RetrievalAugmentedGeneration.Retrievers;

public class BM25RetrieverTests
{
    [Fact]
    public void Constructor_WithValidParameters_Initializes()
    {
        var store = new InMemoryDocumentStore<double>();

        var retriever = new BM25Retriever<double>(
            store,
            k1: 1.5,
            b: 0.75,
            defaultTopK: 5
        );

        Assert.NotNull(retriever);
    }

    [Fact]
    public void Retrieve_PerformsBM25Scoring()
    {
        var store = CreateStoreWithDocuments();
        var retriever = new BM25Retriever<double>(store);

        // BM25 should rank documents by term frequency and document length
        var results = retriever.Retrieve("machine learning neural", topK: 3).ToList();

        Assert.NotEmpty(results);

        // Documents with more matching terms should rank higher
        Assert.All(results, doc => Assert.True(doc.HasRelevanceScore));
    }

    [Fact]
    public void Retrieve_WithCommonTerms_AppliesIDFWeighting()
    {
        // BM25 should reduce weight of common terms (low IDF)
        // and increase weight of rare terms (high IDF)
    }

    private InMemoryDocumentStore<double> CreateStoreWithDocuments()
    {
        var store = new InMemoryDocumentStore<double>();
        // Add sample documents
        return store;
    }
}
```

### Step 4: Hybrid Retriever Tests

```csharp
// File: tests/RetrievalAugmentedGeneration/Retrievers/HybridRetrieverTests.cs

using Xunit;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;

namespace AiDotNet.Tests.RetrievalAugmentedGeneration.Retrievers;

public class HybridRetrieverTests
{
    [Fact]
    public void Constructor_WithDenseAndSparseRetrievers_Initializes()
    {
        var store = new InMemoryDocumentStore<double>();
        var embeddingModel = new StubEmbeddingModel<double>(dimension: 384);

        var denseRetriever = new DenseRetriever<double>(store, embeddingModel);
        var sparseRetriever = new BM25Retriever<double>(store);

        var hybridRetriever = new HybridRetriever<double>(
            denseRetriever,
            sparseRetriever,
            denseWeight: 0.7,
            sparseWeight: 0.3
        );

        Assert.NotNull(hybridRetriever);
    }

    [Fact]
    public void Retrieve_CombinesDenseAndSparseResults()
    {
        var hybrid = CreateHybridRetriever();

        var results = hybrid.Retrieve("machine learning", topK: 5).ToList();

        // Should combine results from both retrievers
        Assert.NotEmpty(results);

        // Scores should be fusion of dense and sparse
        Assert.All(results, doc => Assert.True(doc.HasRelevanceScore));
    }

    [Fact]
    public void Retrieve_WithDifferentWeights_ProducesDifferentRankings()
    {
        // Test that changing dense vs sparse weights affects ranking
    }

    private HybridRetriever<double> CreateHybridRetriever()
    {
        var store = new InMemoryDocumentStore<double>();
        var embeddingModel = new StubEmbeddingModel<double>(dimension: 384);

        var dense = new DenseRetriever<double>(store, embeddingModel);
        var sparse = new BM25Retriever<double>(store);

        return new HybridRetriever<double>(dense, sparse, 0.5, 0.5);
    }
}
```

---

## Testing Strategy

### Coverage Targets
- Dense/Vector retrievers: 90%+
- Sparse (BM25/TF-IDF): 85%+
- Hybrid: 85%+
- Advanced (Multi-Query, etc.): 80%+

### Performance Tests
```csharp
[Fact]
public void Retrieve_From10KDocuments_CompletesUnder5Seconds()
{
    // Benchmark retrieval speed
}
```

---

## Common Pitfalls

### Pitfall 1: Not Testing Relevance Scores

**Wrong:**
```csharp
var results = retriever.Retrieve("query", topK: 5);
Assert.NotEmpty(results);
```

**Correct:**
```csharp
var results = retriever.Retrieve("query", topK: 5).ToList();
Assert.NotEmpty(results);
Assert.All(results, doc =>
{
    Assert.True(doc.HasRelevanceScore);
    Assert.InRange(Convert.ToDouble(doc.RelevanceScore), 0.0, 1.0);
});

// Verify results are sorted by relevance
for (int i = 0; i < results.Count - 1; i++)
{
    var score1 = Convert.ToDouble(results[i].RelevanceScore);
    var score2 = Convert.ToDouble(results[i+1].RelevanceScore);
    Assert.True(score1 >= score2, "Results should be sorted by relevance");
}
```

### Pitfall 2: Not Testing Strategy-Specific Behavior

**Wrong:**
```csharp
// Same tests for all retrievers
```

**Correct:**
```csharp
// BM25-specific test
[Fact]
public void BM25_WithRepeatedTerms_AppliesTermFrequencyCorrectly()
{
    // Test BM25's TF component
}

// Dense-specific test
[Fact]
public void Dense_WithSynonyms_FindsSemanticMatches()
{
    // Test that "automobile" retrieves "car" documents
}
```

---

## Testing Checklist

### For Each Retriever
- [ ] Constructor validation
- [ ] Basic retrieval works
- [ ] TopK parameter enforced
- [ ] Results sorted by relevance
- [ ] Relevance scores assigned
- [ ] Metadata filtering works
- [ ] Empty query validation
- [ ] Empty store handling
- [ ] Performance acceptable

### Strategy-Specific
- [ ] BM25: TF-IDF scoring correct
- [ ] Hybrid: Score fusion works
- [ ] Multi-Query: Query expansion quality
- [ ] Parent-Document: Mapping correct

---

## Next Steps

1. Implement tests for all 11 retrievers (150+ test methods)
2. Achieve 80%+ coverage
3. Performance benchmarks
4. Move to **Issue #372** (Document Store Integration)

---

## Resources

### Retrieval Algorithms
- **BM25**: Probabilistic ranking
- **TF-IDF**: Statistical weighting
- **Dense**: Vector similarity (cosine, dot product)
- **RRF**: Reciprocal Rank Fusion for hybrid

Good luck! Retrieval is the core of RAG - these tests ensure quality results!
