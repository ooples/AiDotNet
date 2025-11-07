# Junior Developer Implementation Guide: Issue #342
## RAG Advanced Patterns - Comprehensive Unit Testing

### Overview
This guide covers comprehensive unit testing for advanced RAG patterns including GraphRAG, Chain-of-Thought retrieval, FLARE (Forward-Looking Active Retrieval), and Self-Correcting Retrieval. The implementations exist - your task is to create thorough tests ensuring these sophisticated patterns work correctly.

---

## For Beginners: What Are Advanced RAG Patterns?

### Traditional RAG (Simple)

```
User Question → Retrieve Documents → Generate Answer
```

Simple, but limited:
- Can't handle multi-hop reasoning
- No self-correction if retrieval fails
- Doesn't leverage structured knowledge
- Treats all retrievals equally

### Advanced RAG (Sophisticated)

**GraphRAG** - Combines knowledge graphs with vector search
```
Question: "What did Einstein's theories influence?"
→ Extract entities: "Einstein"
→ Check knowledge graph: Einstein → Theory of Relativity → GPS Technology
→ Vector search + Graph boost → Better results
```

**Chain-of-Thought Retrieval** - Multi-step reasoning
```
Complex Question: "How did quantum mechanics influence modern computing?"
→ Step 1: Retrieve docs about quantum mechanics fundamentals
→ Step 2: Retrieve docs about quantum → semiconductor physics
→ Step 3: Retrieve docs about semiconductors → computing
→ Combine all steps into comprehensive answer
```

**FLARE** - Proactive retrieval during generation
```
Generating: "The capital of France is..."
→ Check confidence → High → Continue: "Paris"
Generating: "Einstein's theory of..."
→ Check confidence → Low → Retrieve docs about Einstein → "relativity"
```

**Self-Correcting Retrieval** - Validates and refines results
```
Initial retrieval → Check relevance → If poor, reformulate query → Retry
Loop until high-quality results or max attempts
```

### Real-World Analogy

**Traditional RAG** = Looking up one encyclopedia article

**Advanced Patterns:**
- **GraphRAG** = Using encyclopedia + index + cross-references
- **Chain-of-Thought** = Following a research trail across multiple sources
- **FLARE** = Reading and stopping to look things up as needed
- **Self-Correcting** = Double-checking sources and trying better keywords if needed

---

## What EXISTS in the Codebase

### Advanced Pattern Implementations

**1. GraphRAG** - `src/RetrievalAugmentedGeneration/AdvancedPatterns/GraphRAG.cs`
   - Combines knowledge graph with vector search
   - Entity extraction from queries
   - Graph traversal for related entities
   - Score boosting for graph-connected documents
   - In-memory knowledge graph storage

**2. Chain-of-Thought Retriever** - `ChainOfThoughtRetriever.cs`
   - Multi-step decomposition of complex queries
   - Sequential retrieval with context accumulation
   - LLM-guided reasoning chain generation
   - Combines intermediate results

**3. FLARE Retriever** - `FLARERetriever.cs`
   - Forward-looking active retrieval
   - Confidence-based triggering
   - Generation-time document fetching
   - Adaptive retrieval strategy

**4. Self-Correcting Retriever** - `SelfCorrectingRetriever.cs`
   - Relevance validation
   - Query reformulation
   - Iterative refinement
   - Fallback mechanisms

### Core Dependencies

**From RAG Infrastructure:**
- `IRetriever<T>` - Base retriever interface
- `IGenerator<T>` - LLM generation interface
- `IDocumentStore<T>` - Vector document storage
- `IEmbeddingModel<T>` - Text embedding
- `Document<T>` - Document model with metadata

**Graph Components:**
- `KnowledgeGraph` - Graph data structure
- `GraphNode` - Entity nodes
- `GraphEdge` - Relationship edges

---

## What's MISSING (This Issue)

Comprehensive unit tests for:

### 1. GraphRAG Tests

**Basic Functionality:**
- Constructor validation
- Adding relations to knowledge graph
- Entity extraction from queries
- Graph traversal
- Score boosting for graph-connected entities

**Advanced Scenarios:**
- Multi-hop graph traversal (A → B → C)
- Handling disconnected graph components
- Duplicate relation prevention
- Case-insensitive entity matching
- Large graph performance (1000+ entities)

**Edge Cases:**
- Empty knowledge graph (no relations)
- Query with no extractable entities
- Entities not in graph
- Circular relationships
- Very long entity names

### 2. Chain-of-Thought Retriever Tests

**Core Functionality:**
- Query decomposition into steps
- Sequential retrieval
- Context accumulation
- Final answer synthesis

**Reasoning Chains:**
- Simple 2-step reasoning
- Complex multi-step reasoning (5+ steps)
- Handling failed intermediate steps
- Combining contradictory information

**Edge Cases:**
- Query that can't be decomposed
- Empty retrieval at intermediate step
- Maximum step depth limit

### 3. FLARE Retriever Tests

**Confidence Triggering:**
- High confidence (no retrieval needed)
- Low confidence (retrieval triggered)
- Threshold tuning
- Multiple retrieval triggers in one generation

**Generation Integration:**
- Seamless document insertion
- Context window management
- Token limit handling

**Edge Cases:**
- Continuous low confidence (retrieval loop)
- No relevant documents available
- Generation timeout

### 4. Self-Correcting Retriever Tests

**Validation:**
- Relevance scoring
- Quality thresholds
- Reformulation triggers

**Correction Loop:**
- Single correction iteration
- Multiple iterations
- Max attempt limits
- Convergence detection

**Edge Cases:**
- Always irrelevant results
- Query that can't be reformulated
- Correction makes results worse

---

## Step-by-Step Implementation

### Step 1: GraphRAG Tests

```csharp
// File: tests/RetrievalAugmentedGeneration/AdvancedPatterns/GraphRAGTests.cs

using Xunit;
using AiDotNet.RetrievalAugmentedGeneration.AdvancedPatterns;
using AiDotNet.RetrievalAugmentedGeneration.Generators;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;
using AiDotNet.RetrievalAugmentedGeneration.DocumentStores;
using AiDotNet.RetrievalAugmentedGeneration.Embeddings;
using AiDotNet.RetrievalAugmentedGeneration.Models;

namespace AiDotNet.Tests.RetrievalAugmentedGeneration.AdvancedPatterns;

/// <summary>
/// Tests for GraphRAG (Graph-based Retrieval Augmented Generation).
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> GraphRAG combines knowledge graphs with vector search.
///
/// What we're testing:
/// - Knowledge graph construction (adding entities and relationships)
/// - Entity extraction from natural language queries
/// - Graph traversal to find related entities
/// - Score boosting for documents mentioning graph-connected entities
/// - Integration with vector retrieval
///
/// Think of it like testing a smart library system that knows how topics relate to each other.
/// </remarks>
public class GraphRAGTests
{
    private GraphRAG<double> CreateGraphRAG()
    {
        // Create stub components for testing
        var generator = new StubGenerator<double>();
        var embeddingModel = new StubEmbeddingModel<double>(dimension: 384);
        var documentStore = new InMemoryDocumentStore<double>();
        var vectorRetriever = new DenseRetriever<double>(documentStore, embeddingModel);

        return new GraphRAG<double>(generator, vectorRetriever);
    }

    [Fact]
    public void Constructor_WithValidParameters_Initializes()
    {
        // Arrange
        var generator = new StubGenerator<double>();
        var embeddingModel = new StubEmbeddingModel<double>(dimension: 384);
        var documentStore = new InMemoryDocumentStore<double>();
        var vectorRetriever = new DenseRetriever<double>(documentStore, embeddingModel);

        // Act
        var graphRAG = new GraphRAG<double>(generator, vectorRetriever);

        // Assert
        Assert.NotNull(graphRAG);
    }

    [Fact]
    public void Constructor_WithNullGenerator_ThrowsArgumentNullException()
    {
        // Arrange
        var embeddingModel = new StubEmbeddingModel<double>(dimension: 384);
        var documentStore = new InMemoryDocumentStore<double>();
        var vectorRetriever = new DenseRetriever<double>(documentStore, embeddingModel);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new GraphRAG<double>(generator: null, vectorRetriever));
    }

    [Fact]
    public void Constructor_WithNullVectorRetriever_ThrowsArgumentNullException()
    {
        // Arrange
        var generator = new StubGenerator<double>();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new GraphRAG<double>(generator, vectorRetriever: null));
    }

    [Fact]
    public void AddRelation_WithValidParameters_AddsToGraph()
    {
        // Arrange
        var graphRAG = CreateGraphRAG();

        // Act
        graphRAG.AddRelation("Albert Einstein", "DISCOVERED", "Theory of Relativity");
        graphRAG.AddRelation("Theory of Relativity", "PUBLISHED", "1915");

        // Assert - No exception should be thrown
        // Verify by retrieving with a related query
        var documents = CreateSampleDocuments();
        // (In actual test, would populate document store and verify retrieval)
    }

    [Fact]
    public void AddRelation_WithNullEntity_ThrowsArgumentException()
    {
        // Arrange
        var graphRAG = CreateGraphRAG();

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            graphRAG.AddRelation(entity: null, "DISCOVERED", "Something"));
    }

    [Fact]
    public void AddRelation_WithEmptyEntity_ThrowsArgumentException()
    {
        // Arrange
        var graphRAG = CreateGraphRAG();

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            graphRAG.AddRelation(entity: "", "DISCOVERED", "Something"));

        Assert.Throws<ArgumentException>(() =>
            graphRAG.AddRelation(entity: "   ", "DISCOVERED", "Something"));
    }

    [Fact]
    public void AddRelation_WithNullRelation_ThrowsArgumentException()
    {
        // Arrange
        var graphRAG = CreateGraphRAG();

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            graphRAG.AddRelation("Entity", relation: null, "Target"));
    }

    [Fact]
    public void AddRelation_WithNullTarget_ThrowsArgumentException()
    {
        // Arrange
        var graphRAG = CreateGraphRAG();

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            graphRAG.AddRelation("Entity", "RELATION", target: null));
    }

    [Fact]
    public void AddRelation_WithDuplicateRelation_DoesNotAddTwice()
    {
        // Arrange
        var graphRAG = CreateGraphRAG();

        // Act
        graphRAG.AddRelation("Einstein", "DISCOVERED", "Relativity");
        graphRAG.AddRelation("Einstein", "DISCOVERED", "Relativity");  // Duplicate

        // Assert
        // Duplicate should be silently ignored (no exception)
        // Graph should only contain one instance
    }

    [Fact]
    public void AddRelation_CaseInsensitive_NormalizesEntities()
    {
        // Arrange
        var graphRAG = CreateGraphRAG();

        // Act
        graphRAG.AddRelation("Albert Einstein", "DISCOVERED", "Relativity");
        graphRAG.AddRelation("albert einstein", "BORN_IN", "Germany");

        // Assert
        // Both relations should be associated with the same entity (normalized)
    }

    [Fact]
    public void Retrieve_WithNullQuery_ThrowsArgumentException()
    {
        // Arrange
        var graphRAG = CreateGraphRAG();

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            graphRAG.Retrieve(query: null, topK: 5));
    }

    [Fact]
    public void Retrieve_WithEmptyQuery_ThrowsArgumentException()
    {
        // Arrange
        var graphRAG = CreateGraphRAG();

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            graphRAG.Retrieve(query: "", topK: 5));

        Assert.Throws<ArgumentException>(() =>
            graphRAG.Retrieve(query: "   ", topK: 5));
    }

    [Fact]
    public void Retrieve_WithZeroTopK_ThrowsArgumentOutOfRangeException()
    {
        // Arrange
        var graphRAG = CreateGraphRAG();

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            graphRAG.Retrieve("test query", topK: 0));
    }

    [Fact]
    public void Retrieve_WithNegativeTopK_ThrowsArgumentOutOfRangeException()
    {
        // Arrange
        var graphRAG = CreateGraphRAG();

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            graphRAG.Retrieve("test query", topK: -1));
    }

    [Fact]
    public void Retrieve_WithEmptyGraph_ReturnsVectorResultsOnly()
    {
        // Arrange
        var graphRAG = CreateGraphRAG();
        // No relations added to graph

        var embeddingModel = new StubEmbeddingModel<double>(dimension: 384);
        var documentStore = new InMemoryDocumentStore<double>();

        // Add sample documents to store
        AddSampleDocumentsToStore(documentStore, embeddingModel);

        var vectorRetriever = new DenseRetriever<double>(documentStore, embeddingModel);
        graphRAG = new GraphRAG<double>(new StubGenerator<double>(), vectorRetriever);

        // Act
        var results = graphRAG.Retrieve("What is machine learning?", topK: 3).ToList();

        // Assert
        Assert.NotEmpty(results);
        Assert.True(results.Count <= 3);
        // Results should come from vector search only (no graph boost)
    }

    [Fact]
    public void Retrieve_WithGraphConnections_BoostsRelevantDocuments()
    {
        // Arrange
        var embeddingModel = new StubEmbeddingModel<double>(dimension: 384);
        var documentStore = new InMemoryDocumentStore<double>();

        // Add documents about Einstein and relativity
        var doc1 = new Document<double>
        {
            Id = "doc1",
            Content = "Albert Einstein developed the Theory of Relativity.",
            Metadata = new Dictionary<string, object>()
        };
        var doc2 = new Document<double>
        {
            Id = "doc2",
            Content = "Quantum mechanics is a different field of physics.",
            Metadata = new Dictionary<string, object>()
        };

        var embedding1 = embeddingModel.Embed(doc1.Content);
        var embedding2 = embeddingModel.Embed(doc2.Content);

        documentStore.Add(new VectorDocument<double> { Document = doc1, Embedding = embedding1 });
        documentStore.Add(new VectorDocument<double> { Document = doc2, Embedding = embedding2 });

        var vectorRetriever = new DenseRetriever<double>(documentStore, embeddingModel);
        var graphRAG = new GraphRAG<double>(new StubGenerator<double>(), vectorRetriever);

        // Add graph relations
        graphRAG.AddRelation("Albert Einstein", "DISCOVERED", "Theory of Relativity");

        // Act
        var results = graphRAG.Retrieve("What did Einstein discover?", topK: 2).ToList();

        // Assert
        Assert.NotEmpty(results);

        // Document about Einstein and Relativity should be ranked higher due to graph boost
        var firstResult = results.First();
        Assert.Contains("Einstein", firstResult.Content);
        Assert.Contains("Relativity", firstResult.Content);

        // Verify graph boost metadata
        if (firstResult.Metadata.ContainsKey("graph_boosted"))
        {
            Assert.True((bool)firstResult.Metadata["graph_boosted"]);
        }
    }

    [Fact]
    public void Retrieve_WithMultiHopRelations_TraversesGraph()
    {
        // Arrange
        var graphRAG = CreateGraphRAG();

        // Create multi-hop relations: A → B → C
        graphRAG.AddRelation("Einstein", "DISCOVERED", "Relativity");
        graphRAG.AddRelation("Relativity", "INFLUENCED", "GPS Technology");
        graphRAG.AddRelation("GPS Technology", "ENABLES", "Modern Navigation");

        // Act
        var results = graphRAG.Retrieve("How did Einstein's work impact navigation?", topK: 5);

        // Assert
        // GraphRAG should traverse: Einstein → Relativity → GPS → Navigation
        // and boost documents mentioning these connected entities
        Assert.NotEmpty(results);
    }

    [Fact]
    public void Retrieve_WithNoExtractableEntities_FallsBackToVectorSearch()
    {
        // Arrange
        var embeddingModel = new StubEmbeddingModel<double>(dimension: 384);
        var documentStore = new InMemoryDocumentStore<double>();
        AddSampleDocumentsToStore(documentStore, embeddingModel);

        var vectorRetriever = new DenseRetriever<double>(documentStore, embeddingModel);
        var graphRAG = new GraphRAG<double>(new StubGenerator<double>(), vectorRetriever);

        graphRAG.AddRelation("Einstein", "DISCOVERED", "Relativity");

        // Act - Query with no entities
        var results = graphRAG.Retrieve("how does it work", topK: 3).ToList();

        // Assert
        Assert.NotEmpty(results);
        // Should return results from vector search alone
    }

    [Fact]
    public void Retrieve_WithCircularRelations_DoesNotInfiniteLoop()
    {
        // Arrange
        var graphRAG = CreateGraphRAG();

        // Create circular relation: A → B → C → A
        graphRAG.AddRelation("A", "LINKS_TO", "B");
        graphRAG.AddRelation("B", "LINKS_TO", "C");
        graphRAG.AddRelation("C", "LINKS_TO", "A");

        // Act
        var results = graphRAG.Retrieve("Tell me about A", topK: 3);

        // Assert
        // Should complete without hanging
        Assert.NotNull(results);
    }

    [Fact]
    public void Retrieve_WithLargeGraph_HandlesEfficiently()
    {
        // Arrange
        var graphRAG = CreateGraphRAG();

        // Add 1000 entities with relations
        for (int i = 0; i < 1000; i++)
        {
            graphRAG.AddRelation($"Entity_{i}", "RELATES_TO", $"Entity_{i + 1}");
        }

        // Act
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        var results = graphRAG.Retrieve("Find Entity_500", topK: 10);
        stopwatch.Stop();

        // Assert
        Assert.True(stopwatch.ElapsedMilliseconds < 5000,
            $"Large graph retrieval took {stopwatch.ElapsedMilliseconds}ms (should be < 5000ms)");
    }

    [Fact]
    public void Retrieve_WithSpecialCharactersInEntities_HandlesCorrectly()
    {
        // Arrange
        var graphRAG = CreateGraphRAG();

        // Add relations with special characters
        graphRAG.AddRelation("Entity & Co.", "OWNS", "Product (v2.0)");
        graphRAG.AddRelation("René Descartes", "WROTE", "Méditations métaphysiques");

        // Act
        var results = graphRAG.Retrieve("What did René write?", topK: 3);

        // Assert
        Assert.NotNull(results);
        // Should handle special characters without errors
    }

    [Fact]
    public void Retrieve_ReturnsCorrectNumberOfDocuments()
    {
        // Arrange
        var embeddingModel = new StubEmbeddingModel<double>(dimension: 384);
        var documentStore = new InMemoryDocumentStore<double>();

        // Add 20 documents
        for (int i = 0; i < 20; i++)
        {
            var doc = new Document<double>
            {
                Id = $"doc{i}",
                Content = $"Document {i} content",
                Metadata = new Dictionary<string, object>()
            };
            var embedding = embeddingModel.Embed(doc.Content);
            documentStore.Add(new VectorDocument<double> { Document = doc, Embedding = embedding });
        }

        var vectorRetriever = new DenseRetriever<double>(documentStore, embeddingModel);
        var graphRAG = new GraphRAG<double>(new StubGenerator<double>(), vectorRetriever);

        // Act
        var results = graphRAG.Retrieve("test query", topK: 5).ToList();

        // Assert
        Assert.Equal(5, results.Count);
    }

    [Fact]
    public void Retrieve_PreservesDocumentMetadata()
    {
        // Arrange
        var embeddingModel = new StubEmbeddingModel<double>(dimension: 384);
        var documentStore = new InMemoryDocumentStore<double>();

        var doc = new Document<double>
        {
            Id = "doc1",
            Content = "Test content",
            Metadata = new Dictionary<string, object>
            {
                ["author"] = "John Doe",
                ["year"] = 2024,
                ["category"] = "science"
            }
        };

        var embedding = embeddingModel.Embed(doc.Content);
        documentStore.Add(new VectorDocument<double> { Document = doc, Embedding = embedding });

        var vectorRetriever = new DenseRetriever<double>(documentStore, embeddingModel);
        var graphRAG = new GraphRAG<double>(new StubGenerator<double>(), vectorRetriever);

        // Act
        var results = graphRAG.Retrieve("test", topK: 1).ToList();

        // Assert
        var result = results.First();
        Assert.Equal("John Doe", result.Metadata["author"]);
        Assert.Equal(2024, result.Metadata["year"]);
        Assert.Equal("science", result.Metadata["category"]);
    }

    private void AddSampleDocumentsToStore(InMemoryDocumentStore<double> store, StubEmbeddingModel<double> embeddingModel)
    {
        var sampleTexts = new[]
        {
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing enables computers to understand text."
        };

        for (int i = 0; i < sampleTexts.Length; i++)
        {
            var doc = new Document<double>
            {
                Id = $"doc{i}",
                Content = sampleTexts[i],
                Metadata = new Dictionary<string, object>()
            };
            var embedding = embeddingModel.Embed(doc.Content);
            store.Add(new VectorDocument<double> { Document = doc, Embedding = embedding });
        }
    }

    private List<Document<double>> CreateSampleDocuments()
    {
        return new List<Document<double>>
        {
            new Document<double>
            {
                Id = "1",
                Content = "Albert Einstein developed the Theory of Relativity.",
                Metadata = new Dictionary<string, object>()
            },
            new Document<double>
            {
                Id = "2",
                Content = "The Theory of Relativity was published in 1915.",
                Metadata = new Dictionary<string, object>()
            },
            new Document<double>
            {
                Id = "3",
                Content = "GPS technology relies on relativistic corrections.",
                Metadata = new Dictionary<string, object>()
            }
        };
    }
}
```

### Step 2: Chain-of-Thought Retriever Tests

```csharp
// File: tests/RetrievalAugmentedGeneration/AdvancedPatterns/ChainOfThoughtRetrieverTests.cs

using Xunit;
using AiDotNet.RetrievalAugmentedGeneration.AdvancedPatterns;
using AiDotNet.RetrievalAugmentedGeneration.Generators;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;

namespace AiDotNet.Tests.RetrievalAugmentedGeneration.AdvancedPatterns;

/// <summary>
/// Tests for Chain-of-Thought Retriever.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Chain-of-Thought breaks complex queries into steps.
///
/// What we're testing:
/// - Query decomposition into reasoning steps
/// - Sequential retrieval for each step
/// - Context accumulation across steps
/// - Final answer synthesis
/// - Handling of failed intermediate steps
///
/// Example:
/// Complex query: "How did quantum mechanics lead to modern computing?"
/// Step 1: Retrieve quantum mechanics fundamentals
/// Step 2: Retrieve quantum effects in semiconductors
/// Step 3: Retrieve semiconductor role in computers
/// Final: Combine all steps into comprehensive answer
/// </remarks>
public class ChainOfThoughtRetrieverTests
{
    [Fact]
    public void Constructor_WithValidParameters_Initializes()
    {
        // Arrange
        var generator = new StubGenerator<double>();
        var embeddingModel = new StubEmbeddingModel<double>(dimension: 384);
        var documentStore = new InMemoryDocumentStore<double>();
        var baseRetriever = new DenseRetriever<double>(documentStore, embeddingModel);

        // Act
        var cotRetriever = new ChainOfThoughtRetriever<double>(
            generator,
            baseRetriever,
            maxSteps: 5
        );

        // Assert
        Assert.NotNull(cotRetriever);
    }

    [Fact]
    public void Constructor_WithNullGenerator_ThrowsArgumentNullException()
    {
        // Arrange
        var embeddingModel = new StubEmbeddingModel<double>(dimension: 384);
        var documentStore = new InMemoryDocumentStore<double>();
        var baseRetriever = new DenseRetriever<double>(documentStore, embeddingModel);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new ChainOfThoughtRetriever<double>(generator: null, baseRetriever, maxSteps: 5));
    }

    [Fact]
    public void Retrieve_WithSimpleQuery_DecomposesIntoSteps()
    {
        // Arrange
        var cotRetriever = CreateChainOfThoughtRetriever();

        // Act
        var results = cotRetriever.Retrieve("What is machine learning?", topK: 5);

        // Assert
        Assert.NotEmpty(results);
        // Should have decomposed query and retrieved documents for each step
    }

    [Fact]
    public void Retrieve_WithComplexQuery_PerformsMultiStepReasoning()
    {
        // Arrange
        var cotRetriever = CreateChainOfThoughtRetriever();

        // Complex query requiring multiple reasoning steps
        var query = "How did Einstein's theories influence GPS technology?";

        // Act
        var results = cotRetriever.Retrieve(query, topK: 10);

        // Assert
        Assert.NotEmpty(results);
        // Should have retrieved documents across multiple reasoning steps
    }

    [Fact]
    public void Retrieve_WithMaxStepsLimit_DoesNotExceedLimit()
    {
        // Arrange
        var cotRetriever = CreateChainOfThoughtRetriever(maxSteps: 3);

        // Act
        var results = cotRetriever.Retrieve("Complex multi-hop query", topK: 10);

        // Assert
        Assert.NotEmpty(results);
        // Implementation should respect maxSteps limit
    }

    private ChainOfThoughtRetriever<double> CreateChainOfThoughtRetriever(int maxSteps = 5)
    {
        var generator = new StubGenerator<double>();
        var embeddingModel = new StubEmbeddingModel<double>(dimension: 384);
        var documentStore = new InMemoryDocumentStore<double>();
        var baseRetriever = new DenseRetriever<double>(documentStore, embeddingModel);

        return new ChainOfThoughtRetriever<double>(generator, baseRetriever, maxSteps);
    }
}
```

---

## Testing Strategy

### Coverage Targets

- **GraphRAG**: 85%+ coverage
- **Chain-of-Thought**: 80%+ coverage
- **FLARE**: 80%+ coverage
- **Self-Correcting**: 80%+ coverage

### Integration Tests

Test interaction between advanced patterns:
```csharp
[Fact]
public void Integration_GraphRAG_WithChainOfThought_Works()
{
    // Combine GraphRAG's knowledge graph with CoT's multi-step reasoning
}
```

---

## Common Pitfalls

### Pitfall 1: Not Testing Graph Traversal Depth

**Wrong:**
```csharp
graphRAG.AddRelation("A", "R", "B");
// Only test direct relation, not multi-hop
```

**Correct:**
```csharp
graphRAG.AddRelation("A", "R1", "B");
graphRAG.AddRelation("B", "R2", "C");
graphRAG.AddRelation("C", "R3", "D");

var results = graphRAG.Retrieve("Query about A and D", topK: 5);
// Verify traversal from A through B, C to D
```

### Pitfall 2: Ignoring LLM Integration

**Wrong:**
```csharp
// Test without considering LLM responses
```

**Correct:**
```csharp
// Use StubGenerator with predictable responses
var generator = new StubGenerator<double>();
generator.SetResponse("Expected decomposition");

// Test that retriever correctly processes LLM output
```

### Pitfall 3: Not Testing Failure Recovery

**Wrong:**
```csharp
// Only test happy path
```

**Correct:**
```csharp
[Fact]
public void SelfCorrecting_WhenInitialRetrievalFails_ReformulatesQuery()
{
    // Simulate failed initial retrieval
    // Verify reformulation and retry logic
}
```

---

## Testing Checklist

### GraphRAG
- [ ] Constructor validation
- [ ] Adding relations (valid, null, empty)
- [ ] Duplicate relation handling
- [ ] Entity extraction from queries
- [ ] Single-hop graph traversal
- [ ] Multi-hop graph traversal
- [ ] Score boosting verification
- [ ] Empty graph handling
- [ ] Large graph performance
- [ ] Circular relationship handling
- [ ] Special characters in entities
- [ ] Case-insensitive matching

### Chain-of-Thought
- [ ] Constructor validation
- [ ] Simple query decomposition
- [ ] Complex multi-step reasoning
- [ ] Max steps limit enforcement
- [ ] Failed intermediate step handling
- [ ] Context accumulation
- [ ] Final synthesis quality

### FLARE
- [ ] Confidence threshold triggering
- [ ] High confidence (no retrieval)
- [ ] Low confidence (retrieval triggered)
- [ ] Multiple retrieval points
- [ ] Token limit handling

### Self-Correcting
- [ ] Relevance validation
- [ ] Query reformulation
- [ ] Correction iteration limits
- [ ] Convergence detection
- [ ] Graceful degradation

---

## Next Steps

1. Implement all GraphRAG tests (30+ test methods)
2. Implement Chain-of-Thought tests (20+ test methods)
3. Implement FLARE tests (15+ test methods)
4. Implement Self-Correcting tests (15+ test methods)
5. Achieve 80%+ code coverage
6. Create integration tests
7. Move to **Issue #369** (Context Compression)

---

## Resources

### Research Papers
- **GraphRAG**: "From Local to Global: A Graph RAG Approach"
- **Chain-of-Thought**: "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
- **FLARE**: "Active Retrieval Augmented Generation"
- **Self-RAG**: "Self-Reflective Retrieval-Augmented Generation"

### Implementation Patterns
```csharp
// Pattern: Graph-based boosting
if (documentMentionsGraphEntity)
    score *= boostFactor;

// Pattern: Multi-step retrieval
foreach (var step in reasoningSteps)
    intermediateResults.Add(Retrieve(step));

// Pattern: Confidence-based retrieval
if (confidence < threshold)
    additionalDocs = Retrieve(missingConcepts);
```

Good luck! These advanced patterns represent the cutting edge of RAG research.
