---
title: "Text Classification & NLP"
description: "Classify and score text through the AiModelBuilder facade."
order: 4
section: "Examples"
---


This guide demonstrates text/NLP tasks through AiDotNet's `AiModelBuilder` facade. A text vectorizer turns raw strings into features, you train any classifier or regressor on them, and you predict straight from text with `result.PredictText(...)`.

## Overview

The pattern is the same as any other model, plus two text-specific pieces:

- **`ConfigureTextVectorizer(vectorizer)`** hands the result a fitted vectorizer so it can convert new text the same way it converted training text.
- **`DataLoaders.FromTextDocuments(texts, labels, vectorizer)`** fits the vectorizer on your documents and produces the numeric features the model trains on.

Then `result.PredictText(strings)` goes straight from text to a prediction.

## Text Classification (News Categorization)

```csharp
using AiDotNet;
using AiDotNet.Classification.Ensemble;
using AiDotNet.Data.Loaders;
using AiDotNet.Models.Options;
using AiDotNet.Preprocessing.TextVectorizers;
using AiDotNet.Tensors.LinearAlgebra;

// Categories: 0 = Technology, 1 = Sports, 2 = Business
string[] articles =
{
    "New smartphone launches with a faster AI chip and better camera",
    "The team won the championship final in overtime last night",
    "Quarterly earnings beat analyst expectations as revenue grew",
    "Cloud computing provider expands its global data centers",
    "Star striker signs a record transfer to the rival club",
    "Central bank raises interest rates to curb rising inflation",
};
double[] labels = { 0, 1, 2, 0, 1, 2 };

// FromTextDocuments fits the vectorizer; ConfigureTextVectorizer lets the result reuse it.
var vectorizer = new TfidfVectorizer<double>();
var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new RandomForestClassifier<double>(
        new RandomForestClassifierOptions<double> { NEstimators = 100 }))
    .ConfigureTextVectorizer(vectorizer)
    .ConfigureDataLoader(DataLoaders.FromTextDocuments(articles, labels, vectorizer))
    .BuildAsync();

// Categorize new text directly — no manual feature engineering.
var prediction = result.PredictText(new[] { "The league announced a new playoff format" });
Console.WriteLine($"Predicted category: {(int)prediction[0]}");
```

## Sentiment Analysis (Binary)

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
    "Good value for the price, happy with it",
    "Not what I expected, very disappointed",
    "Excellent service and fast shipping",
    "Broke after one day, would not recommend",
};
double[] sentiment = { 1, 0, 1, 0, 1, 0 }; // 1 = positive, 0 = negative

var vectorizer = new TfidfVectorizer<double>();
var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new LogisticRegression<double>())
    .ConfigureTextVectorizer(vectorizer)
    .ConfigureDataLoader(DataLoaders.FromTextDocuments(reviews, sentiment, vectorizer))
    .BuildAsync();

var score = result.PredictText(new[] { "I really enjoyed this purchase!" });
Console.WriteLine($"Sentiment: {(score[0] > 0.5 ? "Positive" : "Negative")}");

// Read classification metrics off the facade (no hand-rolled math).
var features = vectorizer.Transform(reviews);
var stats = result.GetDataSetStats(features, new Vector<double>(sentiment));
Console.WriteLine($"Accuracy: {stats.ErrorStats.Accuracy:P1}, F1: {stats.ErrorStats.F1Score:P1}");
```

## Text Similarity

Score how similar two texts are by concatenating each pair and training a regressor on the similarity target.

```csharp
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Preprocessing.TextVectorizers;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using System.Linq;

string[] textA = { "The cat sat on the mat", "I love programming", "The weather is nice today" };
string[] textB = { "A cat was sitting on a rug", "Coding is my passion", "It is a beautiful sunny day" };
double[] similarity = { 0.9, 0.85, 0.8 };

// Combine each pair into one document the vectorizer can featurize.
var paired = textA.Zip(textB, (a, b) => $"{a} {b}").ToArray();

var vectorizer = new TfidfVectorizer<double>();
var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new MultipleRegression<double>())
    .ConfigureTextVectorizer(vectorizer)
    .ConfigureDataLoader(DataLoaders.FromTextDocuments(paired, similarity, vectorizer))
    .BuildAsync();

var comparison = result.PredictText(new[] { "Hello world Hi there" });
Console.WriteLine($"Similarity: {comparison[0]:F2}");
```

## Configuration Options

The vectorizer controls how text becomes features — n-grams, vocabulary size, lowercasing, and stop words are all real knobs.

```csharp
using AiDotNet;
using AiDotNet.Classification.Ensemble;
using AiDotNet.Data.Loaders;
using AiDotNet.Models.Options;
using AiDotNet.Preprocessing.TextVectorizers;
using AiDotNet.Tensors.LinearAlgebra;
using System.Collections.Generic;

string[] texts = { "I love this product", "Terrible experience", "Great service", "Waste of money" };
double[] labels = { 1, 0, 1, 0 };

// Unigrams + bigrams, capped vocabulary, custom stop words.
var vectorizer = new CountVectorizer<double>(
    maxFeatures: 1000,
    nGramRange: (1, 2),
    lowercase: true,
    stopWords: new HashSet<string> { "the", "a", "an", "of", "and" });

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureModel(new RandomForestClassifier<double>(
        new RandomForestClassifierOptions<double> { NEstimators = 200, MaxDepth = 10 }))
    .ConfigureTextVectorizer(vectorizer)
    .ConfigureDataLoader(DataLoaders.FromTextDocuments(texts, labels, vectorizer))
    .BuildAsync();

Console.WriteLine($"Class for new text: {(int)result.PredictText(new[] { "great product" })[0]}");
```

## Error Handling

Log failures for developers and surface a sanitized message to users; never swallow exceptions silently.

```csharp
using AiDotNet;
using AiDotNet.Classification.Ensemble;
using AiDotNet.Data.Loaders;
using AiDotNet.Models.Options;
using AiDotNet.Preprocessing.TextVectorizers;
using AiDotNet.Tensors.LinearAlgebra;

string[] texts = { "I love this product", "Terrible experience", "Great service", "Waste of money" };
double[] labels = { 1, 0, 1, 0 };

try
{
    var vectorizer = new TfidfVectorizer<double>();
    var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
        .ConfigureModel(new RandomForestClassifier<double>(
            new RandomForestClassifierOptions<double> { NEstimators = 100 }))
        .ConfigureTextVectorizer(vectorizer)
        .ConfigureDataLoader(DataLoaders.FromTextDocuments(texts, labels, vectorizer))
        .BuildAsync();

    var features = vectorizer.Transform(texts);
    var stats = result.GetDataSetStats(features, new Vector<double>(labels));
    Console.WriteLine($"Training complete. Accuracy: {stats.ErrorStats.Accuracy:P2}");
}
catch (ArgumentException)
{
    // Log the exception for developers via your ILogger; show users a safe message.
    Console.WriteLine("Could not train the text model with the provided data.");
}
```

## Advanced: Sequence Models

Token-level tasks — named-entity recognition (a label per token), text generation (next-token prediction), and extractive question answering (answer spans) — need a true sequence model rather than a bag-of-words vectorizer. AiDotNet ships a real `Transformer<T>` (`AiDotNet.NeuralNetworks`) that you build from a `TransformerArchitecture<T>` and train on tokenized `Tensor<T>` inputs. That path is heavier than the one-call facade flow above and is covered in the neural-network guides.

## Summary

For text classification, sentiment, and similarity, the facade gives you a one-call flow:

- `ConfigureTextVectorizer(vectorizer)` + `DataLoaders.FromTextDocuments(...)` to turn text into features
- Any classifier or regressor via `ConfigureModel(...)`
- Metrics through `result.GetDataSetStats(...).ErrorStats`
- Predictions straight from raw strings with `result.PredictText(...)`

Token-level sequence tasks use the lower-level `Transformer<T>` directly.
