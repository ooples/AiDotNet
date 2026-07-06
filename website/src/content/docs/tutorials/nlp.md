---
title: "NLP"
description: "Process text and classify documents with AiDotNet."
order: 5
section: "Tutorials"
---

Process text through the `AiModelBuilder` facade. A text vectorizer turns raw strings into features, any classifier or regressor trains on them, and `result.PredictText(...)` predicts straight from text.

## The Text Pattern

Two text-specific pieces sit on top of the usual facade:

- **`ConfigureTextVectorizer(vectorizer)`** hands the result a fitted vectorizer so new text is converted the same way as training text.
- **`DataLoaders.FromTextDocuments(texts, labels, vectorizer)`** fits the vectorizer on your documents and produces the numeric features the model trains on.

## Text Classification

```csharp
using AiDotNet;
using AiDotNet.Classification.Ensemble;
using AiDotNet.Data.Loaders;
using AiDotNet.Models.Options;
using AiDotNet.Preprocessing.TextVectorizers;
using AiDotNet.Tensors.LinearAlgebra;

// Categories: 0 = Technology, 1 = Sports, 2 = Business
string[] documents =
{
    "New smartphone launches with a faster AI chip",
    "The team won the championship final in overtime",
    "Quarterly earnings beat analyst expectations",
    "Cloud provider expands its global data centers",
    "Star striker signs a record transfer deal",
    "Central bank raises interest rates again",
};
double[] labels = { 0, 1, 2, 0, 1, 2 };

var vectorizer = new TfidfVectorizer<double>();
var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new RandomForestClassifier<double>(
        new RandomForestClassifierOptions<double> { NEstimators = 100 }))
    .ConfigureTextVectorizer(vectorizer)
    .ConfigureDataLoader(DataLoaders.FromTextDocuments(documents, labels, vectorizer))
    .BuildAsync();

// Classify new text directly.
var prediction = result.PredictText(new[] { "The league announced a new playoff format" });
Console.WriteLine($"Predicted category: {(int)prediction[0]}");
```

## Sentiment Analysis

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Preprocessing.TextVectorizers;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;

string[] reviews =
{
    "This product is amazing, I love it!",
    "Terrible quality, a complete waste of money",
    "Good value for the price",
    "Not what I expected, very disappointed",
    "Excellent service and fast shipping",
    "Broke after one day, would not recommend",
};
double[] sentiment = { 1, 0, 1, 0, 1, 0 };

var vectorizer = new TfidfVectorizer<double>();
var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new LogisticRegression<double>())
    .ConfigureTextVectorizer(vectorizer)
    .ConfigureDataLoader(DataLoaders.FromTextDocuments(reviews, sentiment, vectorizer))
    .BuildAsync();

var score = result.PredictText(new[] { "I really enjoyed this purchase!" });
Console.WriteLine($"Sentiment: {(score[0] > 0.5 ? "Positive" : "Negative")}");

// Read classification metrics off the facade.
var features = vectorizer.Transform(reviews);
var stats = result.GetDataSetStats(features, new Vector<double>(sentiment));
Console.WriteLine($"Accuracy: {stats.ErrorStats.Accuracy:P1}, F1: {stats.ErrorStats.F1Score:P1}");
```

## Vectorizer Options

`CountVectorizer` and `TfidfVectorizer` control how text becomes features — n-grams, vocabulary size, lowercasing, and stop words.

```csharp
using AiDotNet.Preprocessing.TextVectorizers;
using System.Collections.Generic;

// Unigrams + bigrams, capped vocabulary, custom stop words.
var vectorizer = new CountVectorizer<double>(
    maxFeatures: 1000,
    nGramRange: (1, 2),
    lowercase: true,
    stopWords: new HashSet<string> { "the", "a", "an", "of", "and" });

Console.WriteLine("Configured a TF/count vectorizer.");
```

## Best Practices

1. **Start with TF-IDF**: `TfidfVectorizer` is a strong default for document classification.
2. **Tune the vocabulary**: cap `maxFeatures` and add stop words to cut noise.
3. **Add n-grams**: `nGramRange: (1, 2)` captures short phrases.
4. **Predict from text**: keep the fitted vectorizer and use `result.PredictText(...)`.
5. **Measure**: read accuracy/F1 from `result.GetDataSetStats(...).ErrorStats`.

## Retrieval-Augmented Generation (RAG)

AiDotNet has a full RAG stack — document stores (`InMemoryDocumentStore`, `FAISSDocumentStore`, `Chroma`/`Pinecone`/`Milvus`), retrievers (`BM25Retriever`, `DenseRetriever`, `HybridRetriever`, `ColBERTRetriever`, …), rerankers, and generators — wired onto a build with `ConfigureRetrievalAugmentedGeneration(retriever, reranker, generator)`.

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Regression;
using AiDotNet.RetrievalAugmentedGeneration.DocumentStores;
using AiDotNet.RetrievalAugmentedGeneration.Generators;
using AiDotNet.RetrievalAugmentedGeneration.Models;
using AiDotNet.RetrievalAugmentedGeneration.Retrievers;
using AiDotNet.Tensors.LinearAlgebra;

const int embeddingDim = 64;

// 1. A document store holds your corpus.
var store = new InMemoryDocumentStore<float>(vectorDimension: embeddingDim);
(string Id, string Content)[] corpus =
{
    ("doc1", "Deep learning uses multi-layer neural networks."),
    ("doc2", "Reinforcement learning trains agents through rewards."),
    ("doc3", "Supervised learning maps inputs to labeled outputs."),
};
foreach (var (id, content) in corpus)
    store.Add(new VectorDocument<float>(
        new Document<float> { Id = id, Content = content }, new Vector<float>(embeddingDim)));

// 2. A retriever + generator form the pipeline (swap BM25 for DenseRetriever to use embeddings).
var retriever = new BM25Retriever<float>(store, defaultTopK: 2);
var generator = new StubGenerator<float>();

// 3. RAG is configured onto a build via ConfigureRetrievalAugmentedGeneration.
var baseX = new Matrix<float>(8, 1);
var baseY = new Vector<float>(8);
for (int i = 0; i < 8; i++) { baseX[i, 0] = i; baseY[i] = i; }

var result = await new AiModelBuilder<float, Matrix<float>, Vector<float>>()
    .ConfigureModel(new RidgeRegression<float>())
    .ConfigureDataLoader(DataLoaders.FromMatrixVector(baseX, baseY))
    .ConfigureRetrievalAugmentedGeneration(retriever: retriever, generator: generator)
    .BuildAsync();

Console.WriteLine("RAG pipeline configured through the facade.");
```

For embedding-based retrieval, use `DenseRetriever<T>(store, embeddingModel, topK)` with `ConfigureEmbeddingModel(...)`. `GraphRetriever` / a knowledge graph power GraphRAG.

## Notes

The facade covers text classification, sentiment, and similarity (`ConfigureTextVectorizer`) and full RAG (`ConfigureRetrievalAugmentedGeneration`). Token-level tasks — NER, extractive QA, and free-form generation — use the lower-level sequence models directly.

## Next Steps

- [Text Classification & NLP Examples](/docs/examples/transformer/)
- [Classification Tutorial](/docs/tutorials/classification/)
