# AiDotNet Code Patterns & Standards for RAG Implementation

## Overview
This guide documents the existing code patterns found in AiDotNet that must be followed when implementing the RAG framework (Issue #284).

---

## 1. XML Documentation Standards

### Pattern Found in Transformer.cs

```csharp
/// <summary>
/// Brief one-line summary of the class/method.
/// </summary>
/// <remarks>
/// <para>
/// Detailed technical explanation for advanced users.
/// Multiple paragraphs explaining the concept in depth.
/// </para>
/// <para><b>For Beginners:</b> Simple explanation using analogies.
/// 
/// Think of it like [analogy]:
/// - Bullet point explaining concept 1
/// - Bullet point explaining concept 2
/// - Real-world example
/// </para>
/// </remarks>
/// <param name="paramName">Parameter description</param>
/// <returns>Return value description</returns>
```

### Apply to RAG Classes

```csharp
/// <summary>
/// Represents a document store that indexes and retrieves vectorized documents using similarity search.
/// </summary>
/// <remarks>
/// <para>
/// This implementation uses cosine similarity to find the most relevant documents to a query vector.
/// Documents are stored with their vector embeddings and metadata for efficient retrieval.
/// </para>
/// <para><b>For Beginners:</b> A document store is like a smart library catalog.
/// 
/// Think of it like organizing books in a library:
/// - Each document is converted to a set of numbers (a vector) that represents its meaning
/// - When you search, the store finds documents whose vectors are most similar to your query
/// - It's faster than reading every document because it uses mathematical similarity
/// </para>
/// </remarks>
public class InMemoryDocumentStore : IDocumentStore
{
    // Implementation
}
```

---

## 2. Interface Design Pattern

### Pattern Found in IModel.cs

```csharp
namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the core functionality for [purpose].
/// </summary>
/// <remarks>
/// [Technical description]
/// 
/// <b>For Beginners:</b> [Simple explanation with analogy]
/// </remarks>
/// <typeparam name="TInput">The input type</typeparam>
/// <typeparam name="TOutput">The output type</typeparam>
public interface IModel<TInput, TOutput, TMetadata>
{
    /// <summary>
    /// [Method summary]
    /// </summary>
    /// <remarks>
    /// [Technical details]
    /// 
    /// <b>For Beginners:</b> [Simple explanation]
    /// </remarks>
    void Train(TInput input, TOutput expectedOutput);
    
    TOutput Predict(TInput input);
    
    TMetadata GetModelMetadata();
}
```

### Apply to RAG Interfaces

```csharp
namespace AiDotNet.RetrievalAugmentedGeneration.Interfaces;

/// <summary>
/// Defines the contract for retrieving relevant documents based on a query.
/// </summary>
/// <remarks>
/// This interface enables pluggable retrieval strategies, supporting both
/// dense vector search and hybrid approaches combining sparse and dense methods.
/// 
/// <b>For Beginners:</b> A retriever finds the most relevant documents for your question.
/// 
/// Think of it like a smart search engine:
/// - You provide a question or search query
/// - The retriever looks through all documents and finds the most relevant ones
/// - It returns the top matches that are most likely to contain the answer
/// </remarks>
public interface IRetriever
{
    /// <summary>
    /// Retrieves relevant documents for a given query string.
    /// </summary>
    /// <param name="query">The query text.</param>
    /// <returns>A collection of relevant documents ordered by relevance.</returns>
    IEnumerable<Document> Retrieve(string query);
}
```

---

## 3. Class Implementation Pattern

### Pattern Found in Transformer.cs

```csharp
public class Transformer<T> : NeuralNetworkBase<T>
{
    // Private fields with descriptive names
    private readonly TransformerArchitecture<T> _transformerArchitecture;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    
    // Public properties with XML docs
    public Tensor<T>? AttentionMask { get; set; }
    
    // Constructor with detailed XML docs
    /// <summary>
    /// Creates a new Transformer neural network with the specified architecture.
    /// </summary>
    public Transformer(TransformerArchitecture<T> architecture) : base(architecture)
    {
        _transformerArchitecture = architecture;
        // Initialization
    }
    
    // Public methods
    public override void Train(/* params */)
    {
        // Implementation
    }
}
```

### Apply to RAG Classes

```csharp
namespace AiDotNet.RetrievalAugmentedGeneration;

/// <summary>
/// Orchestrates the complete RAG pipeline from query to grounded answer.
/// </summary>
/// <remarks>
/// <para>
/// The RAG pipeline coordinates retrieval of relevant documents and generation
/// of grounded answers using the retrieved context.
/// </para>
/// <para><b>For Beginners:</b> The RAG pipeline is like a research assistant.
/// 
/// When you ask a question, it:
/// - Searches through documents to find relevant information
/// - Reads the relevant parts carefully
/// - Writes an answer based on what it found
/// - Shows you where the information came from (citations)
/// </para>
/// </remarks>
public class RagPipeline
{
    private readonly IRetriever _retriever;
    private readonly IGenerator _generator;
    private readonly IReranker? _reranker;
    
    /// <summary>
    /// Gets or sets the maximum number of documents to retrieve.
    /// </summary>
    public int TopK { get; set; } = 5;
    
    /// <summary>
    /// Initializes a new instance of the RAG pipeline.
    /// </summary>
    /// <param name="retriever">The document retriever.</param>
    /// <param name="generator">The text generator.</param>
    /// <param name="reranker">Optional reranker for retrieved documents.</param>
    public RagPipeline(IRetriever retriever, IGenerator generator, IReranker? reranker = null)
    {
        _retriever = retriever ?? throw new ArgumentNullException(nameof(retriever));
        _generator = generator ?? throw new ArgumentNullException(nameof(generator));
        _reranker = reranker;
    }
    
    /// <summary>
    /// Executes the RAG pipeline for a given query.
    /// </summary>
    public GroundedAnswer Execute(string query)
    {
        // Implementation
    }
}
```

---

## 4. Namespace Organization

### Existing Pattern
```
AiDotNet.NeuralNetworks
AiDotNet.NeuralNetworks.Layers
AiDotNet.Interfaces
AiDotNet.Models
AiDotNet.Evaluation
AiDotNet.LinearAlgebra
```

### RAG Namespace Structure
```
AiDotNet.RetrievalAugmentedGeneration
AiDotNet.RetrievalAugmentedGeneration.Interfaces
AiDotNet.RetrievalAugmentedGeneration.Models
AiDotNet.RetrievalAugmentedGeneration.Retrievers
AiDotNet.RetrievalAugmentedGeneration.DocumentStores
AiDotNet.RetrievalAugmentedGeneration.TextSplitters
AiDotNet.RetrievalAugmentedGeneration.Rerankers
AiDotNet.RetrievalAugmentedGeneration.Evaluation
```

---

## 5. Error Handling Pattern

### From existing code
```csharp
public SomeClass(RequiredParam param1, OptionalParam? param2 = null)
{
    // Null checks with descriptive exceptions
    _param1 = param1 ?? throw new ArgumentNullException(nameof(param1));
    _param2 = param2 ?? new DefaultOptionalParam();
}

public void SomeMethod(Input input)
{
    if (input == null)
        throw new ArgumentNullException(nameof(input));
        
    if (input.Value < 0)
        throw new ArgumentException("Value must be non-negative", nameof(input));
}
```

### Apply to RAG
```csharp
public VectorRetriever(IDocumentStore documentStore, EmbeddingLayer embeddingLayer, int topK = 5)
{
    _documentStore = documentStore ?? throw new ArgumentNullException(nameof(documentStore));
    _embeddingLayer = embeddingLayer ?? throw new ArgumentNullException(nameof(embeddingLayer));
    
    if (topK <= 0)
        throw new ArgumentException("TopK must be greater than zero", nameof(topK));
        
    _topK = topK;
}
```

---

## 6. Model Evaluator Pattern

### From DefaultModelEvaluator.cs

```csharp
public class DefaultModelEvaluator<T, TInput, TOutput> : IModelEvaluator<T, TInput, TOutput>
{
    protected readonly PredictionStatsOptions _predictionOptions;
    
    public DefaultModelEvaluator(PredictionStatsOptions? predictionOptions = null)
    {
        _predictionOptions = predictionOptions ?? new PredictionStatsOptions();
    }
    
    public ModelEvaluationData<T, TInput, TOutput> EvaluateModel(
        ModelEvaluationInput<T, TInput, TOutput> input)
    {
        var evaluationData = new ModelEvaluationData<T, TInput, TOutput>
        {
            TrainingSet = CalculateDataSetStats(input.InputData.XTrain, input.InputData.YTrain, input.Model),
            ValidationSet = CalculateDataSetStats(input.InputData.XValidation, input.InputData.YValidation, input.Model),
            TestSet = CalculateDataSetStats(input.InputData.XTest, input.InputData.YTest, input.Model)
        };
        
        return evaluationData;
    }
}
```

### Apply to RAG Evaluator

```csharp
namespace AiDotNet.RetrievalAugmentedGeneration.Evaluation;

/// <summary>
/// Evaluates RAG pipeline performance using faithfulness, similarity, and coverage metrics.
/// </summary>
public class RAGEvaluator
{
    private readonly FaithfulnessMetric _faithfulnessMetric;
    private readonly AnswerSimilarityMetric _similarityMetric;
    private readonly ContextCoverageMetric _coverageMetric;
    
    public RAGEvaluator(
        FaithfulnessMetric? faithfulnessMetric = null,
        AnswerSimilarityMetric? similarityMetric = null,
        ContextCoverageMetric? coverageMetric = null)
    {
        _faithfulnessMetric = faithfulnessMetric ?? new FaithfulnessMetric();
        _similarityMetric = similarityMetric ?? new AnswerSimilarityMetric();
        _coverageMetric = coverageMetric ?? new ContextCoverageMetric();
    }
    
    public RAGEvaluationResult Evaluate(string query, GroundedAnswer answer, string? groundTruth = null)
    {
        return new RAGEvaluationResult
        {
            Faithfulness = _faithfulnessMetric.Calculate(answer),
            Similarity = groundTruth != null ? _similarityMetric.Calculate(answer.Answer, groundTruth) : null,
            Coverage = _coverageMetric.Calculate(answer)
        };
    }
}
```

---

## 7. Testing Pattern

### Structure
```
tests/UnitTests/
    └── [FeatureArea]/
        └── [ClassName]Tests.cs
```

### Test Class Pattern
```csharp
using Microsoft.VisualStudio.TestTools.UnitTesting;
using AiDotNet.RetrievalAugmentedGeneration;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration;

[TestClass]
public class VectorRetrieverTests
{
    [TestMethod]
    public void Retrieve_WithValidQuery_ReturnsTopKDocuments()
    {
        // Arrange
        var documentStore = CreateMockDocumentStore();
        var embeddingLayer = CreateMockEmbeddingLayer();
        var retriever = new VectorRetriever(documentStore, embeddingLayer, topK: 3);
        var query = "test query";
        
        // Act
        var results = retriever.Retrieve(query);
        
        // Assert
        Assert.IsNotNull(results);
        Assert.AreEqual(3, results.Count());
    }
    
    [TestMethod]
    [ExpectedException(typeof(ArgumentNullException))]
    public void Constructor_WithNullDocumentStore_ThrowsArgumentNullException()
    {
        // Arrange & Act
        var retriever = new VectorRetriever(null, CreateMockEmbeddingLayer());
        
        // Assert - ExpectedException
    }
    
    private IDocumentStore CreateMockDocumentStore()
    {
        // Mock creation
    }
}
```

---

## 8. Naming Conventions

### Fields & Properties
```csharp
// Private fields: camelCase with underscore prefix
private readonly IRetriever _retriever;
private int _topK;

// Public properties: PascalCase
public int TopK { get; set; }
public bool IsEnabled { get; private set; }
```

### Methods
```csharp
// Public methods: PascalCase, verb-noun pattern
public IEnumerable<Document> Retrieve(string query);
public void AddDocuments(IEnumerable<Document> documents);
public double CalculateSimilarity(Vector v1, Vector v2);

// Private methods: PascalCase
private string BuildPromptWithContext(string query, IEnumerable<Document> context);
private IEnumerable<string> ExtractCitations(string answer, IEnumerable<Document> context);
```

### Interfaces
```csharp
// Interface names: PascalCase with 'I' prefix
public interface IRetriever { }
public interface IDocumentStore { }
public interface IChunkingStrategy { }
```

---

## 9. Generic Type Patterns

### When to Use Generics

**Use generics for numeric operations** (like existing Transformer<T>):
```csharp
public class VectorSimilarityCalculator<T> where T : struct
{
    public T CosineSimilarity(Vector<T> v1, Vector<T> v2);
}
```

**Don't use generics for business logic** (RAG pipeline operates on strings/documents):
```csharp
// Good - concrete types
public class RagPipeline
{
    public GroundedAnswer Execute(string query);
}

// Avoid over-genericization
// public class RagPipeline<TQuery, TAnswer> // Too abstract
```

---

## 10. Dependency Injection Pattern

### From existing code
```csharp
public class SomeClass
{
    // Inject dependencies via constructor
    private readonly IDependency _dependency;
    
    public SomeClass(IDependency dependency)
    {
        _dependency = dependency ?? throw new ArgumentNullException(nameof(dependency));
    }
}
```

### Apply to RAG
```csharp
public class RagPipeline
{
    private readonly IRetriever _retriever;
    private readonly IGenerator _generator;
    private readonly IReranker? _reranker; // Optional dependency
    
    public RagPipeline(IRetriever retriever, IGenerator generator, IReranker? reranker = null)
    {
        _retriever = retriever ?? throw new ArgumentNullException(nameof(retriever));
        _generator = generator ?? throw new ArgumentNullException(nameof(generator));
        _reranker = reranker; // Optional, can be null
    }
}
```

---

## 11. File Header Pattern

### Standard file structure
```csharp
// Optional using directives
using System;
using System.Collections.Generic;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.RetrievalAugmentedGeneration.Retrievers;

/// <summary>
/// Class documentation
/// </summary>
public class VectorRetriever : IRetriever
{
    // Implementation
}
```

---

## 12. Properties vs Fields

### Pattern from existing code
```csharp
public class Example
{
    // Private backing fields for complex properties
    private Vector? _cachedEmbedding;
    
    // Auto-properties for simple cases
    public int TopK { get; set; }
    
    // Properties with logic
    public Vector QueryEmbedding
    {
        get => _cachedEmbedding ??= ComputeEmbedding();
        set => _cachedEmbedding = value;
    }
    
    // Read-only properties
    public int DocumentCount => _documents.Count;
}
```

---

## Summary Checklist for RAG Implementation

✅ **Documentation**
- [ ] All public classes have summary + remarks with "For Beginners" section
- [ ] All public methods documented with params and returns
- [ ] Real-world analogies provided for complex concepts

✅ **Code Structure**
- [ ] Namespace follows pattern: AiDotNet.RetrievalAugmentedGeneration.[SubArea]
- [ ] Interfaces in Interfaces/ subdirectory
- [ ] Models in Models/ subdirectory
- [ ] Private fields use _camelCase
- [ ] Public members use PascalCase

✅ **Error Handling**
- [ ] Constructor parameters validated with ArgumentNullException
- [ ] Invalid arguments throw ArgumentException with descriptive messages
- [ ] Required dependencies injected via constructor

✅ **Testing**
- [ ] Unit tests in tests/UnitTests/RetrievalAugmentedGeneration/
- [ ] Test class name matches implementation class with "Tests" suffix
- [ ] Arrange-Act-Assert pattern used
- [ ] Edge cases and exceptions tested

✅ **Integration**
- [ ] Follows existing IModel/IModelEvaluator patterns
- [ ] Leverages Vector, Transformer, EmbeddingLayer from existing code
- [ ] Serialization compatible with JsonConverterRegistry

---

**This guide ensures the RAG implementation will be consistent with the existing AiDotNet codebase quality and style standards.**
