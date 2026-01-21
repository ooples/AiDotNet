# Transformer Models with AiModelBuilder

This guide demonstrates how to use transformer-based models for NLP tasks using AiDotNet's simplified API.

## Overview

AiDotNet provides powerful transformer capabilities through the `AiModelBuilder` facade, hiding the complexity of transformer architecture while giving you full control over configuration.

## Text Classification

```csharp
using AiDotNet;

// Prepare your text data
var texts = new string[]
{
    "This product is amazing, I love it!",
    "Terrible quality, waste of money",
    "Good value for the price",
    "Not what I expected, disappointed",
    "Excellent service and fast shipping"
};

var sentiments = new double[] { 1, 0, 1, 0, 1 }; // 1 = positive, 0 = negative

// Build and train a transformer model for text classification
var result = await new AiModelBuilder<double, string[], double[]>()
    .ConfigureNlp(config =>
    {
        config.TaskType = NlpTaskType.TextClassification;
        config.ModelType = NlpModelType.Transformer;
        config.MaxSequenceLength = 128;
        config.VocabSize = 30000;
    })
    .ConfigurePreprocessing()
    .BuildAsync(texts, sentiments);

// Make predictions
var newTexts = new[] { "I really enjoyed this purchase!" };
var predictions = result.Predict(newTexts);
Console.WriteLine($"Sentiment: {(predictions[0] > 0.5 ? "Positive" : "Negative")}");

// View training metrics
Console.WriteLine($"Training Accuracy: {result.TrainingAccuracy:P2}");
Console.WriteLine($"Validation Accuracy: {result.ValidationAccuracy:P2}");
```

## Named Entity Recognition

```csharp
using AiDotNet;

// Prepare training data with entity labels
var sentences = new string[]
{
    "John Smith works at Microsoft in Seattle.",
    "Apple released a new iPhone yesterday.",
    "Dr. Jane Doe will speak at Harvard University."
};

// Entity labels (per token)
var entityLabels = new int[][]
{
    new[] { 1, 1, 0, 0, 2, 0, 3 }, // PERSON, O, ORG, O, LOC
    new[] { 2, 0, 0, 0, 4, 0 },     // ORG, O, PRODUCT
    new[] { 1, 1, 1, 0, 0, 0, 2, 2 } // PERSON, O, ORG
};

// Build NER model
var result = await new AiModelBuilder<double, string[], int[][]>()
    .ConfigureNlp(config =>
    {
        config.TaskType = NlpTaskType.NamedEntityRecognition;
        config.ModelType = NlpModelType.Transformer;
        config.NumLabels = 5; // O, PERSON, ORG, LOC, PRODUCT
    })
    .ConfigurePreprocessing()
    .BuildAsync(sentences, entityLabels);

// Extract entities from new text
var newSentence = new[] { "Elon Musk founded SpaceX in California." };
var entities = result.Predict(newSentence);
Console.WriteLine($"Detected entities: {string.Join(", ", entities[0])}");
```

## Text Generation

```csharp
using AiDotNet;

// Training corpus for text generation
var trainingTexts = new string[]
{
    "Once upon a time in a faraway kingdom",
    "The scientist discovered a new element",
    "In the depths of the ocean lives a creature",
    // ... more training texts
};

// Build generative model
var result = await new AiModelBuilder<double, string[], string[]>()
    .ConfigureNlp(config =>
    {
        config.TaskType = NlpTaskType.TextGeneration;
        config.ModelType = NlpModelType.Transformer;
        config.MaxSequenceLength = 256;
        config.Temperature = 0.7;
    })
    .ConfigurePreprocessing()
    .BuildAsync(trainingTexts, trainingTexts);

// Generate new text
var prompt = new[] { "The robot looked at the sunset and" };
var generated = result.Predict(prompt);
Console.WriteLine($"Generated: {generated[0]}");
```

## Question Answering

```csharp
using AiDotNet;
using System.Linq;

// Context-question-answer triplets
var contexts = new string[]
{
    "The Eiffel Tower is located in Paris, France. It was built in 1889.",
    "Python is a programming language created by Guido van Rossum.",
};

var questions = new string[]
{
    "Where is the Eiffel Tower located?",
    "Who created Python?"
};

var answers = new string[]
{
    "Paris, France",
    "Guido van Rossum"
};

// Combine context and question as input
var inputs = contexts.Zip(questions, (c, q) => $"{c} [SEP] {q}").ToArray();

// Build QA model
var result = await new AiModelBuilder<double, string[], string[]>()
    .ConfigureNlp(config =>
    {
        config.TaskType = NlpTaskType.QuestionAnswering;
        config.ModelType = NlpModelType.Transformer;
        config.MaxSequenceLength = 512;
    })
    .ConfigurePreprocessing()
    .BuildAsync(inputs, answers);

// Answer new questions
var newContext = "Albert Einstein developed the theory of relativity.";
var newQuestion = "What did Einstein develop?";
var newInput = new[] { $"{newContext} [SEP] {newQuestion}" };

var answer = result.Predict(newInput);
Console.WriteLine($"Answer: {answer[0]}");
```

## Text Similarity / Embeddings

```csharp
using AiDotNet;

// Pairs of similar texts
var text1 = new string[]
{
    "The cat sat on the mat",
    "I love programming",
    "The weather is nice today"
};

var text2 = new string[]
{
    "A cat was sitting on a rug",
    "Coding is my passion",
    "It's a beautiful sunny day"
};

var similarityScores = new double[] { 0.9, 0.85, 0.8 };

// Build similarity model
var result = await new AiModelBuilder<double, (string, string)[], double[]>()
    .ConfigureNlp(config =>
    {
        config.TaskType = NlpTaskType.SemanticSimilarity;
        config.ModelType = NlpModelType.Transformer;
    })
    .ConfigurePreprocessing()
    .BuildAsync(
        text1.Zip(text2, (a, b) => (a, b)).ToArray(),
        similarityScores
    );

// Compare new text pairs
var comparison = result.Predict(new[] { ("Hello world", "Hi there") });
Console.WriteLine($"Similarity: {comparison[0]:F2}");
```

## Configuration Options

```csharp
using AiDotNet;

// Sample training data
var texts = new string[] { "I love this product", "Terrible experience", "Great service", "Waste of money" };
var labels = new double[] { 1.0, 0.0, 1.0, 0.0 };  // 1.0 = positive, 0.0 = negative

// Full configuration example
var result = await new AiModelBuilder<double, string[], double[]>()
    .ConfigureNlp(config =>
    {
        // Model architecture
        config.TaskType = NlpTaskType.TextClassification;
        config.ModelType = NlpModelType.Transformer;
        config.MaxSequenceLength = 256;
        config.VocabSize = 32000;

        // Training parameters
        config.LearningRate = 2e-5;
        config.BatchSize = 16;
        config.Epochs = 5;
        config.WarmupSteps = 500;

        // Regularization
        config.Dropout = 0.1;
        config.WeightDecay = 0.01;

        // Tokenization
        config.TokenizerType = TokenizerType.BPE;
        config.LowercaseInput = true;
    })
    .ConfigurePreprocessing()
    .ConfigureValidation(validationSplit: 0.15)
    .BuildAsync(texts, labels);

// Access training history
foreach (var epoch in result.TrainingHistory)
{
    Console.WriteLine($"Epoch {epoch.EpochNumber}: Loss={epoch.Loss:F4}, Accuracy={epoch.Accuracy:P2}");
}
```

## Best Practices

1. **Use appropriate sequence length**: Shorter sequences train faster but may truncate important information
2. **Adjust batch size for memory**: Transformer models are memory-intensive; reduce batch size if needed
3. **Use warmup steps**: Gradually increasing learning rate helps training stability
4. **Monitor validation metrics**: Watch for overfitting on small datasets

## Summary

The `AiModelBuilder` provides a clean interface for transformer-based NLP tasks:
- Text classification and sentiment analysis
- Named entity recognition
- Text generation
- Question answering
- Semantic similarity

All complexity is handled internally, letting you focus on your data and results.
