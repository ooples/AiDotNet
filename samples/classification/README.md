# Classification Samples

This directory contains examples of classification algorithms in AiDotNet.

## Available Samples

### Binary Classification
| Sample | Description |
|--------|-------------|
| [SentimentAnalysis](./BinaryClassification/SentimentAnalysis/) | Classify text as positive or negative sentiment |
| [SpamDetection](./BinaryClassification/SpamDetection/) | Detect spam emails using ML |
| [FraudDetection](./BinaryClassification/FraudDetection/) | Identify fraudulent transactions |

### Multi-class Classification
| Sample | Description |
|--------|-------------|
| [IrisClassification](./MultiClassification/IrisClassification/) | Classic Iris flower classification |
| [ImageClassification](./MultiClassification/ImageClassification/) | Classify images into categories |

## Quick Start

```csharp
using AiDotNet;
using AiDotNet.Classification;

var features = new double[][] { /* training data */ };
var labels = new double[] { /* class labels */ };

var result = await new PredictionModelBuilder<double, double[], double>()
    .ConfigureModel(new RandomForestClassifier<double>(nEstimators: 100))
    .ConfigurePreprocessing()
    .BuildAsync(features, labels);

var prediction = result.Model.Predict(newSample);
```

## Available Classifiers

AiDotNet includes 28+ classification algorithms:
- Random Forest
- Gradient Boosting
- SVM (Support Vector Machines)
- Naive Bayes (Gaussian, Multinomial, Bernoulli)
- K-Nearest Neighbors
- Decision Trees
- AdaBoost
- XGBoost
- LightGBM
- Neural Network classifiers

## Learn More

- [Classification Tutorial](/docs/tutorials/classification/)
- [API Reference](/api/AiDotNet.Classification/)
