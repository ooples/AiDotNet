# Getting Started with AiDotNet

This guide demonstrates the fundamentals of using AiDotNet for machine learning tasks.

## Overview

AiDotNet provides a simple, unified API for machine learning through the `AiModelBuilder` class. All complexity is handled internally, so you can focus on your data and results.

## Quick Start: Linear Regression

```csharp
using AiDotNet;

// Your training data
var features = new double[][]
{
    new[] { 1.0 },
    new[] { 2.0 },
    new[] { 3.0 },
    new[] { 4.0 },
    new[] { 5.0 }
};

var targets = new double[] { 2.1, 4.0, 5.9, 8.1, 10.0 };

// Build and train a model
var result = await new AiModelBuilder<double, double[][], double[]>()
    .ConfigureRegression()
    .ConfigurePreprocessing()
    .BuildAsync(features, targets);

// Make predictions
var prediction = result.Predict(new double[][] { new[] { 6.0 } });
Console.WriteLine($"Prediction for x=6: {prediction[0]:F2}");
// Output: Prediction for x=6: 12.01

// View model performance
Console.WriteLine($"R-Squared: {result.RSquared:F4}");
Console.WriteLine($"Mean Squared Error: {result.MeanSquaredError:F4}");
```

## Multiple Features (Multivariate Regression)

```csharp
using AiDotNet;

// House price prediction: [sqft, bedrooms, age]
var houseFeatures = new double[][]
{
    new[] { 1500.0, 3.0, 10.0 },
    new[] { 2000.0, 4.0, 5.0 },
    new[] { 1200.0, 2.0, 20.0 },
    new[] { 1800.0, 3.0, 8.0 },
    new[] { 2500.0, 5.0, 2.0 }
};

var prices = new double[] { 300000, 450000, 200000, 380000, 550000 };

// Build model
var result = await new AiModelBuilder<double, double[][], double[]>()
    .ConfigureRegression(config =>
    {
        config.ModelType = RegressionModelType.Ridge;
        config.RegularizationStrength = 0.1;
    })
    .ConfigurePreprocessing(config =>
    {
        config.NormalizeFeatures = true;
        config.HandleMissingValues = true;
    })
    .BuildAsync(houseFeatures, prices);

// Predict new house price
var newHouse = new double[][] { new[] { 1700.0, 3.0, 12.0 } };
var predictedPrice = result.Predict(newHouse);
Console.WriteLine($"Estimated price: ${predictedPrice[0]:N0}");

// View feature importance
Console.WriteLine("\nFeature Importance:");
Console.WriteLine($"  Square Feet: {result.FeatureImportance[0]:F3}");
Console.WriteLine($"  Bedrooms: {result.FeatureImportance[1]:F3}");
Console.WriteLine($"  Age: {result.FeatureImportance[2]:F3}");
```

## Binary Classification

```csharp
using AiDotNet;

// Customer churn prediction
var customerData = new double[][]
{
    new[] { 12.0, 50.0, 1.0 },   // tenure, monthly charges, has contract
    new[] { 1.0, 80.0, 0.0 },
    new[] { 24.0, 45.0, 1.0 },
    new[] { 3.0, 95.0, 0.0 },
    new[] { 36.0, 40.0, 1.0 }
};

var churned = new double[] { 0, 1, 0, 1, 0 };

// Build classification model
var result = await new AiModelBuilder<double, double[][], double[]>()
    .ConfigureClassification(config =>
    {
        config.ModelType = ClassificationModelType.LogisticRegression;
        config.ClassCount = 2;
    })
    .ConfigurePreprocessing()
    .BuildAsync(customerData, churned);

// Predict churn probability
var newCustomer = new double[][] { new[] { 6.0, 70.0, 0.0 } };
var churnProbability = result.PredictProbability(newCustomer);
Console.WriteLine($"Churn probability: {churnProbability[0]:P1}");

// View model metrics
Console.WriteLine($"Accuracy: {result.Accuracy:P2}");
Console.WriteLine($"Precision: {result.Precision:P2}");
Console.WriteLine($"Recall: {result.Recall:P2}");
Console.WriteLine($"F1 Score: {result.F1Score:P2}");
```

## Multi-Class Classification

```csharp
using AiDotNet;

// Iris flower classification
var irisFeatures = new double[][]
{
    new[] { 5.1, 3.5, 1.4, 0.2 },
    new[] { 7.0, 3.2, 4.7, 1.4 },
    new[] { 6.3, 3.3, 6.0, 2.5 },
    // ... more samples
};

var species = new double[] { 0, 1, 2 }; // 0=setosa, 1=versicolor, 2=virginica

// Build model
var result = await new AiModelBuilder<double, double[][], double[]>()
    .ConfigureClassification(config =>
    {
        config.ModelType = ClassificationModelType.RandomForest;
        config.ClassCount = 3;
    })
    .ConfigurePreprocessing()
    .BuildAsync(irisFeatures, species);

// Predict species
var newFlower = new double[][] { new[] { 5.9, 3.0, 5.1, 1.8 } };
var predicted = result.Predict(newFlower);
var probabilities = result.PredictProbability(newFlower);

Console.WriteLine($"Predicted species: {predicted[0]}");
Console.WriteLine($"Confidence: {probabilities.Max():P1}");
```

## Working with Data

### Automatic Preprocessing

```csharp
using AiDotNet;

// Data with missing values and different scales
var rawData = new double[][]
{
    new[] { 1000.0, 25.0, double.NaN },
    new[] { 2000.0, 30.0, 3.0 },
    new[] { double.NaN, 35.0, 5.0 },
    new[] { 1500.0, double.NaN, 4.0 }
};

var targets = new double[] { 100, 200, 180, 150 };

// AiModelBuilder handles preprocessing automatically
var result = await new AiModelBuilder<double, double[][], double[]>()
    .ConfigureRegression()
    .ConfigurePreprocessing(config =>
    {
        config.HandleMissingValues = true;         // Impute missing values
        config.NormalizeFeatures = true;           // Scale to 0-1 range
        config.RemoveOutliers = true;              // Remove statistical outliers
        config.EncodeCategories = true;            // One-hot encode categories
    })
    .BuildAsync(rawData, targets);
```

### Train/Test Split

```csharp
using AiDotNet;

var result = await new AiModelBuilder<double, double[][], double[]>()
    .ConfigureRegression()
    .ConfigurePreprocessing()
    .ConfigureValidation(config =>
    {
        config.ValidationSplit = 0.2;    // 20% for validation
        config.Shuffle = true;
        config.RandomSeed = 42;
    })
    .BuildAsync(features, targets);

Console.WriteLine($"Training R-Squared: {result.TrainingRSquared:F4}");
Console.WriteLine($"Validation R-Squared: {result.ValidationRSquared:F4}");
```

### Cross-Validation

```csharp
using AiDotNet;

var result = await new AiModelBuilder<double, double[][], double[]>()
    .ConfigureClassification()
    .ConfigurePreprocessing()
    .ConfigureValidation(config =>
    {
        config.CrossValidationFolds = 5;
    })
    .BuildAsync(features, labels);

Console.WriteLine($"CV Accuracy: {result.CrossValidationAccuracy:P2}");
Console.WriteLine($"CV Std Dev: {result.CrossValidationStdDev:F4}");
```

## Saving and Loading Models

```csharp
using AiDotNet;

// Train and save
var result = await new AiModelBuilder<double, double[][], double[]>()
    .ConfigureRegression()
    .BuildAsync(features, targets);

result.SaveModel("my_model.aimodel");
Console.WriteLine("Model saved!");

// Load and use later
var loadedModel = AiModelResult<double>.Load("my_model.aimodel");
var prediction = loadedModel.Predict(newData);
Console.WriteLine($"Prediction: {prediction[0]}");
```

## Batch Predictions

```csharp
using AiDotNet;

// Make predictions on multiple samples at once
var testData = new double[][]
{
    new[] { 1.5 },
    new[] { 2.5 },
    new[] { 3.5 },
    new[] { 4.5 }
};

var predictions = result.Predict(testData);

Console.WriteLine("Batch Predictions:");
for (int i = 0; i < predictions.Length; i++)
{
    Console.WriteLine($"  Input {testData[i][0]}: {predictions[i]:F2}");
}
```

## Model Comparison

```csharp
using AiDotNet;

// Compare different model types
var models = new[]
{
    RegressionModelType.Linear,
    RegressionModelType.Ridge,
    RegressionModelType.Lasso,
    RegressionModelType.ElasticNet
};

Console.WriteLine("Model Comparison:");
Console.WriteLine("Model\t\t\tR-Squared\tMSE");
Console.WriteLine("-----\t\t\t---------\t---");

foreach (var modelType in models)
{
    var result = await new AiModelBuilder<double, double[][], double[]>()
        .ConfigureRegression(c => c.ModelType = modelType)
        .ConfigurePreprocessing()
        .ConfigureValidation(c => c.ValidationSplit = 0.2)
        .BuildAsync(features, targets);

    Console.WriteLine($"{modelType}\t\t{result.ValidationRSquared:F4}\t\t{result.ValidationMse:F4}");
}
```

## Summary

AiDotNet's `AiModelBuilder` provides:
- Simple, fluent API for all ML tasks
- Automatic preprocessing and feature engineering
- Built-in validation and cross-validation
- Model saving and loading
- Performance metrics and feature importance

All complexity is handled internally. You focus on your data and business problem.
