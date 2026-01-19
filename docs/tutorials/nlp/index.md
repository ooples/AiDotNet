---
layout: default
title: NLP & RAG
parent: Tutorials
nav_order: 5
has_children: true
permalink: /tutorials/nlp/
---

# NLP & RAG Tutorial
{: .no_toc }

Build powerful text processing and retrieval-augmented generation systems.
{: .fs-6 .fw-300 }

---

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

AiDotNet provides 50+ RAG components for building intelligent document systems:
- **Embeddings**: Sentence Transformers, custom models
- **Vector Stores**: In-memory, FAISS, distributed
- **Retrievers**: Dense, sparse, hybrid
- **Rerankers**: Cross-encoder, ColBERT

---

## Text Embeddings

### Generate Embeddings

```csharp
using AiDotNet.RetrievalAugmentedGeneration;

// Create embedding model
var embedder = new SentenceTransformerEmbeddings<float>(
    model: "all-MiniLM-L6-v2");

// Single text
var embedding = embedder.Encode("Hello, world!");
Console.WriteLine($"Embedding dimension: {embedding.Length}"); // 384

// Batch encoding
var texts = new[] { "First document", "Second document" };
var embeddings = embedder.EncodeBatch(texts);
```

### Similarity Search

```csharp
// Compute cosine similarity
float similarity = embedder.ComputeSimilarity(
    "machine learning",
    "deep learning");
Console.WriteLine($"Similarity: {similarity:F3}"); // ~0.85
```

---

## Vector Stores

### In-Memory Store

```csharp
var vectorStore = new InMemoryVectorStore<float>(
    dimension: 384);

// Add documents
var docs = new[]
{
    new Document { Id = "1", Text = "First document about AI" },
    new Document { Id = "2", Text = "Second document about ML" }
};

await vectorStore.AddAsync(docs, embedder);

// Search
var results = await vectorStore.SearchAsync(
    query: "artificial intelligence",
    embedder: embedder,
    topK: 5);
```

### FAISS Store (For Large Datasets)

```csharp
var faissStore = new FAISSVectorStore<float>(
    dimension: 384,
    indexType: FAISSIndexType.IVFFlat,
    nlist: 100);

// Index millions of documents
await faissStore.AddAsync(documents, embedder);

// Fast search
var results = await faissStore.SearchAsync(query, embedder, topK: 10);
```

---

## Retrieval-Augmented Generation (RAG)

### Basic RAG Pipeline

```csharp
using AiDotNet.RetrievalAugmentedGeneration;

// Build RAG pipeline
var rag = new RAGPipeline<float>()
    .WithEmbeddings(new SentenceTransformerEmbeddings<float>())
    .WithVectorStore(new InMemoryVectorStore<float>())
    .WithRetriever(new DenseRetriever<float>(topK: 5))
    .Build();

// Index documents
var documents = new[]
{
    "AiDotNet is a .NET machine learning framework.",
    "It supports 100+ neural network architectures.",
    "GPU acceleration is available via CUDA."
};
await rag.IndexDocumentsAsync(documents);

// Query
var response = await rag.QueryAsync("What is AiDotNet?");
Console.WriteLine(response.Answer);
Console.WriteLine($"Sources: {string.Join(", ", response.SourceDocuments)}");
```

### Advanced RAG with Reranking

```csharp
var advancedRag = new RAGPipeline<float>()
    .WithEmbeddings(new SentenceTransformerEmbeddings<float>("all-mpnet-base-v2"))
    .WithVectorStore(new FAISSVectorStore<float>(768))
    .WithRetriever(new HybridRetriever<float>(
        denseWeight: 0.7f,
        sparseWeight: 0.3f,
        topK: 20))
    .WithReranker(new CrossEncoderReranker<float>(
        model: "cross-encoder/ms-marco-MiniLM-L-6-v2",
        topK: 5))
    .WithChunker(new RecursiveChunker(
        chunkSize: 512,
        chunkOverlap: 50))
    .Build();
```

---

## GraphRAG

Enhance RAG with knowledge graphs:

```csharp
using AiDotNet.RetrievalAugmentedGeneration.Graph;

var graphRag = new GraphRAGPipeline<float>()
    .WithEmbeddings(new SentenceTransformerEmbeddings<float>())
    .WithKnowledgeGraph(new InMemoryKnowledgeGraph<float>())
    .WithEntityExtractor(new NEREntityExtractor<float>())
    .WithRelationExtractor(new RelationExtractor<float>())
    .Build();

// Index with entity extraction
await graphRag.IndexDocumentsAsync(documents);

// Query with graph traversal
var response = await graphRag.QueryAsync(
    "How are AI and machine learning related?");
```

---

## Text Classification

```csharp
using AiDotNet.Classification;

var texts = new[]
{
    "This movie was amazing!",
    "Terrible waste of time.",
    "Pretty good overall."
};
var labels = new[] { 1, 0, 1 }; // positive/negative

var result = await new AiModelBuilder<float, string, int>()
    .ConfigureModel(new TextClassifier<float>(
        backbone: "distilbert-base-uncased",
        numClasses: 2))
    .ConfigureTokenizer(new BertTokenizer())
    .BuildAsync(texts, labels);

// Use result.Predict() directly (facade pattern)
var prediction = result.Predict("Great film!");
```

---

## Named Entity Recognition

```csharp
using AiDotNet.NLP;

var ner = new NamedEntityRecognizer<float>("bert-base-NER");

var text = "Apple Inc. was founded by Steve Jobs in California.";
var entities = ner.Extract(text);

foreach (var entity in entities)
{
    Console.WriteLine($"{entity.Text} [{entity.Label}]");
    // Apple Inc. [ORG]
    // Steve Jobs [PERSON]
    // California [LOCATION]
}
```

---

## Document Chunking

### Strategies

```csharp
// Fixed size chunks
var fixedChunker = new FixedSizeChunker(
    chunkSize: 512,
    overlap: 50);

// Sentence-based
var sentenceChunker = new SentenceChunker(
    maxSentencesPerChunk: 5);

// Recursive (respects document structure)
var recursiveChunker = new RecursiveChunker(
    chunkSize: 512,
    separators: ["\n\n", "\n", ". ", " "]);

// Semantic (groups similar content)
var semanticChunker = new SemanticChunker<float>(
    embedder: embedder,
    similarityThreshold: 0.7f);
```

---

## Best Practices

### Embedding Selection

| Model | Dimension | Speed | Quality |
|:------|:----------|:------|:--------|
| all-MiniLM-L6-v2 | 384 | ⚡⚡⚡ | ⭐⭐⭐ |
| all-mpnet-base-v2 | 768 | ⚡⚡ | ⭐⭐⭐⭐ |
| e5-large-v2 | 1024 | ⚡ | ⭐⭐⭐⭐⭐ |

### RAG Tips

1. **Chunk size matters**: Too small loses context, too large adds noise
2. **Use reranking**: Significantly improves retrieval quality
3. **Hybrid retrieval**: Combine dense + sparse for best results
4. **Metadata filtering**: Filter by date, source, type before retrieval
5. **Evaluate systematically**: Use RAGAS or similar frameworks

---

## Next Steps

- [BasicRAG Sample](/samples/nlp/RAG/BasicRAG/)
- [GraphRAG Sample](/samples/nlp/RAG/GraphRAG/)
- [Embeddings Sample](/samples/nlp/Embeddings/)
- [RAG API Reference](/api/AiDotNet.RetrievalAugmentedGeneration/)
