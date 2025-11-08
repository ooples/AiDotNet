# Junior Developer Implementation Guide: Issue #372
## RAG Document Store Integration - Comprehensive Unit Testing

### Overview
Test 13 document store implementations including InMemory, FAISS, Pinecone, Weaviate, Qdrant, Milvus, Chroma, Redis, Postgres, Elasticsearch, Azure Search, and Hybrid stores. Implementations exist - ensure thorough testing.

---

## For Beginners: What Is a Document Store?

### The Concept

A document store is a **smart database** for RAG systems that:
1. Stores documents with their vector embeddings
2. Enables fast similarity search
3. Supports metadata filtering
4. Scales to millions of documents

### Real-World Analogy

**Traditional Database (SQL):**
```
SELECT * FROM documents WHERE title LIKE '%machine learning%';
```
- Finds exact keyword matches only
- Can't understand "What documents discuss AI?" (without keyword "AI")

**Vector Document Store:**
```
store.GetSimilar(queryEmbedding, topK: 5);
```
- Finds semantically similar documents
- "AI" query finds "machine learning", "neural networks", "deep learning" documents
- Uses mathematical similarity in vector space

### Store Types

**In-Memory Stores:**
- `InMemoryDocumentStore` - Simple, fast, non-persistent
- Best for: Testing, prototypes, small datasets (<10K docs)

**Specialized Vector Databases:**
- `FAISSDocumentStore` - Facebook's similarity search library
- `PineconeDocumentStore` - Managed cloud vector database
- `WeaviateDocumentStore` - Open-source vector search engine
- `QdrantDocumentStore` - High-performance vector database
- `MilvusDocumentStore` - Scalable vector database
- `ChromaDBDocumentStore` - Embeddings database

**General-Purpose with Vector Support:**
- `PostgresVectorDocumentStore` - PostgreSQL with pgvector extension
- `RedisVLDocumentStore` - Redis with vector search
- `ElasticsearchDocumentStore` - Elasticsearch with vector fields
- `AzureSearchDocumentStore` - Azure Cognitive Search

**Hybrid:**
- `HybridDocumentStore` - Combines vector + keyword search

---

## What EXISTS

### Document Store Implementations (13 Total)

**Base Infrastructure:**
- `IDocumentStore<T>` - Interface defining contract
- `DocumentStoreBase<T>` - Base class with common functionality

**Stores:**
1. **InMemoryDocumentStore** - In-memory storage
2. **FAISSDocumentStore** - FAISS library integration
3. **PineconeDocumentStore** - Pinecone cloud service
4. **WeaviateDocumentStore** - Weaviate integration
5. **QdrantDocumentStore** - Qdrant integration
6. **MilvusDocumentStore** - Milvus integration
7. **ChromaDBDocumentStore** - ChromaDB integration
8. **RedisVLDocumentStore** - Redis Vector Library
9. **PostgresVectorDocumentStore** - PostgreSQL + pgvector
10. **ElasticsearchDocumentStore** - Elasticsearch
11. **AzureSearchDocumentStore** - Azure Cognitive Search
12. **HybridDocumentStore** - Hybrid search
13. *(Additional stores as needed)*

---

## What's MISSING

### Test Coverage Gaps

**Core Functionality (All Stores):**
- Add document (single)
- Add documents (batch)
- Get similar documents (vector search)
- Get similar with filters (metadata filtering)
- Get document by ID
- Remove document by ID
- Clear all documents
- Get all documents
- Document count property
- Vector dimension property

**Edge Cases:**
- Add document with mismatched vector dimension
- Add duplicate document ID
- Search empty store
- Search with invalid vector dimension
- Null/empty parameter validation
- Very large batch additions (1000+ documents)
- Concurrent operations (thread safety)

**Store-Specific:**
- **FAISS**: Index types (Flat, IVF, HNSW)
- **Pinecone**: API key validation, namespace handling
- **Postgres**: Connection string validation, pgvector extension check
- **Hybrid**: Score fusion methods

---

## Step-by-Step Implementation

### Step 1: Document Store Test Base

```csharp
// File: tests/RetrievalAugmentedGeneration/DocumentStores/DocumentStoreTestBase.cs

using Xunit;
using AiDotNet.Interfaces;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.RetrievalAugmentedGeneration.Embeddings;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Tests.RetrievalAugmentedGeneration.DocumentStores;

public abstract class DocumentStoreTestBase<TStore, T>
    where TStore : IDocumentStore<T>
{
    protected abstract TStore CreateStore(int vectorDimension);

    protected VectorDocument<T> CreateVectorDocument(string id, string content, int dimension)
    {
        var embeddingModel = new StubEmbeddingModel<T>(dimension);
        var doc = new Document<T>
        {
            Id = id,
            Content = content,
            Metadata = new Dictionary<string, object>()
        };

        return new VectorDocument<T>
        {
            Document = doc,
            Embedding = embeddingModel.Embed(content)
        };
    }

    [Fact]
    public void Add_WithValidDocument_AddsSuccessfully()
    {
        var store = CreateStore(vectorDimension: 384);
        var vectorDoc = CreateVectorDocument("doc1", "Test content", 384);

        store.Add(vectorDoc);

        Assert.Equal(1, store.DocumentCount);
    }

    [Fact]
    public void Add_WithNullDocument_ThrowsArgumentNullException()
    {
        var store = CreateStore(384);

        Assert.Throws<ArgumentNullException>(() => store.Add(null));
    }

    [Fact]
    public void AddBatch_WithMultipleDocuments_AddsAll()
    {
        var store = CreateStore(384);
        var docs = new[]
        {
            CreateVectorDocument("doc1", "Content 1", 384),
            CreateVectorDocument("doc2", "Content 2", 384),
            CreateVectorDocument("doc3", "Content 3", 384)
        };

        store.AddBatch(docs);

        Assert.Equal(3, store.DocumentCount);
    }

    [Fact]
    public void AddBatch_WithNullCollection_ThrowsArgumentNullException()
    {
        var store = CreateStore(384);

        Assert.Throws<ArgumentNullException>(() => store.AddBatch(null));
    }

    [Fact]
    public void AddBatch_WithEmptyCollection_ThrowsArgumentException()
    {
        var store = CreateStore(384);

        Assert.Throws<ArgumentException>(() => store.AddBatch(Array.Empty<VectorDocument<T>>()));
    }

    [Fact]
    public void GetSimilar_WithValidQuery_ReturnsResults()
    {
        var store = CreateStore(384);
        var embeddingModel = new StubEmbeddingModel<T>(384);

        // Add documents
        store.AddBatch(new[]
        {
            CreateVectorDocument("doc1", "Machine learning is AI", 384),
            CreateVectorDocument("doc2", "Deep learning uses neural networks", 384),
            CreateVectorDocument("doc3", "NLP enables language understanding", 384)
        });

        // Query
        var queryEmbedding = embeddingModel.Embed("artificial intelligence");
        var results = store.GetSimilar(queryEmbedding, topK: 2).ToList();

        Assert.Equal(2, results.Count);
        Assert.All(results, doc => Assert.True(doc.HasRelevanceScore));
    }

    [Fact]
    public void GetSimilar_ReturnsResultsSortedByRelevance()
    {
        var store = CreateStore(384);
        var embeddingModel = new StubEmbeddingModel<T>(384);

        store.AddBatch(new[]
        {
            CreateVectorDocument("doc1", "Content 1", 384),
            CreateVectorDocument("doc2", "Content 2", 384),
            CreateVectorDocument("doc3", "Content 3", 384)
        });

        var queryEmbedding = embeddingModel.Embed("test");
        var results = store.GetSimilar(queryEmbedding, topK: 3).ToList();

        // Verify descending order
        for (int i = 0; i < results.Count - 1; i++)
        {
            var score1 = Convert.ToDouble(results[i].RelevanceScore);
            var score2 = Convert.ToDouble(results[i + 1].RelevanceScore);
            Assert.True(score1 >= score2, "Results should be sorted by relevance");
        }
    }

    [Fact]
    public void GetSimilarWithFilters_AppliesMetadataFilters()
    {
        var store = CreateStore(384);

        var doc1 = CreateVectorDocument("doc1", "AI content", 384);
        doc1.Document.Metadata["year"] = 2024;
        doc1.Document.Metadata["category"] = "AI";

        var doc2 = CreateVectorDocument("doc2", "AI content", 384);
        doc2.Document.Metadata["year"] = 2020;
        doc2.Document.Metadata["category"] = "AI";

        store.AddBatch(new[] { doc1, doc2 });

        var embeddingModel = new StubEmbeddingModel<T>(384);
        var queryEmbedding = embeddingModel.Embed("AI");
        var filters = new Dictionary<string, object> { ["year"] = 2024 };

        var results = store.GetSimilarWithFilters(queryEmbedding, topK: 5, filters).ToList();

        Assert.All(results, doc => Assert.Equal(2024, doc.Metadata["year"]));
    }

    [Fact]
    public void GetById_WithExistingId_ReturnsDocument()
    {
        var store = CreateStore(384);
        var vectorDoc = CreateVectorDocument("doc1", "Test", 384);

        store.Add(vectorDoc);

        var retrieved = store.GetById("doc1");

        Assert.NotNull(retrieved);
        Assert.Equal("doc1", retrieved.Id);
    }

    [Fact]
    public void GetById_WithNonExistentId_ReturnsNull()
    {
        var store = CreateStore(384);

        var retrieved = store.GetById("nonexistent");

        Assert.Null(retrieved);
    }

    [Fact]
    public void Remove_WithExistingId_RemovesDocument()
    {
        var store = CreateStore(384);
        var vectorDoc = CreateVectorDocument("doc1", "Test", 384);

        store.Add(vectorDoc);
        var removed = store.Remove("doc1");

        Assert.True(removed);
        Assert.Equal(0, store.DocumentCount);
    }

    [Fact]
    public void Remove_WithNonExistentId_ReturnsFalse()
    {
        var store = CreateStore(384);

        var removed = store.Remove("nonexistent");

        Assert.False(removed);
    }

    [Fact]
    public void Clear_RemovesAllDocuments()
    {
        var store = CreateStore(384);

        store.AddBatch(new[]
        {
            CreateVectorDocument("doc1", "Test1", 384),
            CreateVectorDocument("doc2", "Test2", 384)
        });

        store.Clear();

        Assert.Equal(0, store.DocumentCount);
    }

    [Fact]
    public void GetAll_ReturnsAllDocuments()
    {
        var store = CreateStore(384);

        store.AddBatch(new[]
        {
            CreateVectorDocument("doc1", "Test1", 384),
            CreateVectorDocument("doc2", "Test2", 384),
            CreateVectorDocument("doc3", "Test3", 384)
        });

        var all = store.GetAll().ToList();

        Assert.Equal(3, all.Count);
    }

    [Fact]
    public void VectorDimension_ReturnsCorrectDimension()
    {
        var store = CreateStore(768);

        // Add a document to set dimension
        store.Add(CreateVectorDocument("doc1", "Test", 768));

        Assert.Equal(768, store.VectorDimension);
    }

    [Fact]
    public void Add_WithMismatchedDimension_ThrowsArgumentException()
    {
        var store = CreateStore(384);

        // Add first document with dimension 384
        store.Add(CreateVectorDocument("doc1", "Test", 384));

        // Try to add document with different dimension
        var wrongDimDoc = CreateVectorDocument("doc2", "Test", 512);

        Assert.Throws<ArgumentException>(() => store.Add(wrongDimDoc));
    }

    [Fact]
    public void AddBatch_WithLargeBatch_HandlesEfficiently()
    {
        var store = CreateStore(384);

        var largeBatch = Enumerable.Range(0, 1000)
            .Select(i => CreateVectorDocument($"doc{i}", $"Content {i}", 384))
            .ToArray();

        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        store.AddBatch(largeBatch);
        stopwatch.Stop();

        Assert.Equal(1000, store.DocumentCount);
        Assert.True(stopwatch.ElapsedMilliseconds < 10000,
            $"Batch add took {stopwatch.ElapsedMilliseconds}ms (should be < 10s)");
    }

    [Fact]
    public void GetSimilar_FromEmptyStore_ReturnsEmptyList()
    {
        var store = CreateStore(384);
        var embeddingModel = new StubEmbeddingModel<T>(384);

        var queryEmbedding = embeddingModel.Embed("test");
        var results = store.GetSimilar(queryEmbedding, topK: 5).ToList();

        Assert.Empty(results);
    }
}
```

### Step 2: InMemoryDocumentStore Tests

```csharp
// File: tests/RetrievalAugmentedGeneration/DocumentStores/InMemoryDocumentStoreTests.cs

using Xunit;
using AiDotNet.RetrievalAugmentedGeneration.DocumentStores;

namespace AiDotNet.Tests.RetrievalAugmentedGeneration.DocumentStores;

public class InMemoryDocumentStoreTests : DocumentStoreTestBase<InMemoryDocumentStore<double>, double>
{
    protected override InMemoryDocumentStore<double> CreateStore(int vectorDimension)
    {
        return new InMemoryDocumentStore<double>();
    }

    [Fact]
    public void Constructor_Initializes()
    {
        var store = new InMemoryDocumentStore<double>();

        Assert.NotNull(store);
        Assert.Equal(0, store.DocumentCount);
    }

    [Fact]
    public void GetSimilar_UsesCosineSimilarity()
    {
        var store = CreateStore(384);

        // Add documents and verify cosine similarity is used
        // (Test similarity metric specifically)
    }

    [Fact]
    public void Clear_IsInstantaneous()
    {
        var store = CreateStore(384);

        // Add many documents
        var docs = Enumerable.Range(0, 10000)
            .Select(i => CreateVectorDocument($"doc{i}", $"Content {i}", 384))
            .ToArray();

        store.AddBatch(docs);

        // Clear should be fast
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        store.Clear();
        stopwatch.Stop();

        Assert.Equal(0, store.DocumentCount);
        Assert.True(stopwatch.ElapsedMilliseconds < 100);
    }
}
```

### Step 3: FAISS DocumentStore Tests

```csharp
// File: tests/RetrievalAugmentedGeneration/DocumentStores/FAISSDocumentStoreTests.cs

using Xunit;
using AiDotNet.RetrievalAugmentedGeneration.DocumentStores;

namespace AiDotNet.Tests.RetrievalAugmentedGeneration.DocumentStores;

public class FAISSDocumentStoreTests : DocumentStoreTestBase<FAISSDocumentStore<double>, double>
{
    protected override FAISSDocumentStore<double> CreateStore(int vectorDimension)
    {
        return new FAISSDocumentStore<double>(
            dimension: vectorDimension,
            indexType: "Flat"  // Simple index for testing
        );
    }

    [Theory]
    [InlineData("Flat")]
    [InlineData("IVF")]
    [InlineData("HNSW")]
    public void Constructor_WithDifferentIndexTypes_Initializes(string indexType)
    {
        var store = new FAISSDocumentStore<double>(dimension: 384, indexType: indexType);

        Assert.NotNull(store);
    }

    [Fact]
    public void GetSimilar_WithHNSWIndex_IsFast()
    {
        var store = new FAISSDocumentStore<double>(dimension: 384, indexType: "HNSW");

        // Add many documents
        var docs = Enumerable.Range(0, 10000)
            .Select(i => CreateVectorDocument($"doc{i}", $"Content {i}", 384))
            .ToArray();

        store.AddBatch(docs);

        // Search should be fast with HNSW
        var embeddingModel = new StubEmbeddingModel<double>(384);
        var queryEmbedding = embeddingModel.Embed("test");

        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        var results = store.GetSimilar(queryEmbedding, topK: 10).ToList();
        stopwatch.Stop();

        Assert.Equal(10, results.Count);
        Assert.True(stopwatch.ElapsedMilliseconds < 100,
            $"HNSW search took {stopwatch.ElapsedMilliseconds}ms (should be < 100ms)");
    }
}
```

---

## Testing Strategy

### Coverage Targets
- **InMemoryDocumentStore**: 95%+ (simplest to test)
- **FAISS, local stores**: 85%+
- **Cloud stores (Pinecone, etc.)**: 75%+ unit tests, integration separate
- **Hybrid stores**: 80%+

### Performance Benchmarks

```csharp
[Fact]
public void GetSimilar_From100KDocuments_CompletesUnder1Second()
{
    // Benchmark search performance
}

[Fact]
public void AddBatch_10KDocuments_CompletesUnder10Seconds()
{
    // Benchmark insertion performance
}
```

---

## Common Pitfalls

### Pitfall 1: Testing Cloud Stores Without Mocks

**Wrong:**
```csharp
[Fact]
public void Pinecone_Add_MakesAPICall()
{
    var store = new PineconeDocumentStore<double>("real-api-key", ...);
    store.Add(doc);  // Makes real API call!
}
```

**Correct:**
```csharp
[Fact]
public void Pinecone_Constructor_ValidatesApiKey()
{
    Assert.Throws<ArgumentException>(() =>
        new PineconeDocumentStore<double>(apiKey: null, ...));
}

// Real API tests go in integration test suite
```

### Pitfall 2: Not Testing Vector Dimension Validation

**Wrong:**
```csharp
// Assume dimension validation works
```

**Correct:**
```csharp
[Fact]
public void Add_WithMismatchedDimension_ThrowsArgumentException()
{
    var store = CreateStore(384);
    store.Add(CreateVectorDocument("doc1", "test", 384));

    var wrongDoc = CreateVectorDocument("doc2", "test", 512);
    Assert.Throws<ArgumentException>(() => store.Add(wrongDoc));
}
```

---

## Testing Checklist

### For Each Store
- [ ] Constructor validation
- [ ] Add document works
- [ ] Add batch works
- [ ] GetSimilar returns correct count
- [ ] GetSimilar results sorted by relevance
- [ ] GetSimilarWithFilters applies filters
- [ ] GetById works
- [ ] Remove works
- [ ] Clear works
- [ ] GetAll works
- [ ] DocumentCount accurate
- [ ] VectorDimension correct
- [ ] Dimension mismatch rejected
- [ ] Null parameter validation
- [ ] Empty store handling
- [ ] Large batch performance

### Store-Specific
- [ ] FAISS: Index types (Flat, IVF, HNSW)
- [ ] Pinecone/cloud: API validation
- [ ] Postgres: Connection string
- [ ] Hybrid: Score fusion

---

## Next Steps

1. Implement tests for all 13 stores (250+ test methods)
2. Achieve 80%+ coverage
3. Integration tests (separate PR)
4. Performance benchmarks
5. Move to **Issue #373** (Vector Search)

---

## Resources

### Document Store Types
- **In-Memory**: Fastest, no persistence
- **FAISS**: Local, very fast, production-ready
- **Cloud (Pinecone, Weaviate)**: Managed, scalable
- **Hybrid**: Best of vector + keyword

### Similarity Metrics
- **Cosine similarity**: Angle between vectors
- **Euclidean distance**: Straight-line distance
- **Dot product**: Inner product of vectors

Good luck! Document stores are the foundation of RAG systems - thorough testing ensures reliability at scale!
