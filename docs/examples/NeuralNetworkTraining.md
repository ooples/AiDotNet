# Neural Networks with AiModelBuilder

This guide demonstrates how to train neural networks for various tasks using AiDotNet's simplified API.

## Overview

AiDotNet provides powerful neural network capabilities through the `AiModelBuilder` facade. You configure what you want, and the system handles the architecture details.

## Image Classification

```csharp
using AiDotNet;
using System.Linq;

// Load image data (28x28 grayscale images as flat arrays)
var images = LoadMnistImages();  // double[][] with 784 features each
var labels = LoadMnistLabels();  // double[] with values 0-9

// Build and train a neural network
var result = await new AiModelBuilder<double, double[][], double[]>()
    .ConfigureNeuralNetwork(config =>
    {
        config.TaskType = NeuralNetworkTaskType.ImageClassification;
        config.InputShape = new[] { 28, 28, 1 };  // Height, Width, Channels
        config.NumClasses = 10;
    })
    .ConfigureTraining(config =>
    {
        config.Epochs = 10;
        config.BatchSize = 64;
        config.LearningRate = 0.001;
    })
    .ConfigurePreprocessing()
    .BuildAsync(images, labels);

// Make predictions
var newImage = LoadTestImage();
var prediction = result.Predict(new[] { newImage });
var probabilities = result.PredictProbability(new[] { newImage });

Console.WriteLine($"Predicted digit: {prediction[0]}");
Console.WriteLine($"Confidence: {probabilities.Max():P1}");

// View training metrics
Console.WriteLine($"\nTraining Accuracy: {result.TrainingAccuracy:P2}");
Console.WriteLine($"Validation Accuracy: {result.ValidationAccuracy:P2}");
```

## Binary Classification (Spam Detection)

```csharp
using AiDotNet;

// Email features (word frequencies, etc.)
var emailFeatures = new double[][]
{
    new[] { 0.1, 0.8, 0.0, 0.9 },  // high frequency of spam words
    new[] { 0.9, 0.1, 0.7, 0.0 },  // normal email
    new[] { 0.2, 0.7, 0.1, 0.8 },
    // ... more samples
};

var isSpam = new double[] { 1, 0, 1 };

// Build neural network for binary classification
var result = await new AiModelBuilder<double, double[][], double[]>()
    .ConfigureNeuralNetwork(config =>
    {
        config.TaskType = NeuralNetworkTaskType.BinaryClassification;
        config.InputSize = 4;
    })
    .ConfigureTraining(config =>
    {
        config.Epochs = 50;
        config.BatchSize = 32;
        config.ValidationSplit = 0.2;
    })
    .BuildAsync(emailFeatures, isSpam);

// Classify new email
var newEmail = new double[][] { new[] { 0.3, 0.6, 0.2, 0.7 } };
var spamProbability = result.PredictProbability(newEmail);
Console.WriteLine($"Spam probability: {spamProbability[0]:P1}");

// View metrics
Console.WriteLine($"Accuracy: {result.Accuracy:P2}");
Console.WriteLine($"AUC-ROC: {result.AucRoc:F4}");
```

## Regression (Price Prediction)

```csharp
using AiDotNet;

// Product features for price prediction
var productFeatures = new double[][]
{
    new[] { 100.0, 4.5, 1000.0, 2.0 },  // weight, rating, reviews, category
    new[] { 250.0, 4.2, 500.0, 1.0 },
    new[] { 50.0, 4.8, 2000.0, 3.0 },
    // ... more samples
};

var prices = new double[] { 29.99, 59.99, 19.99 };

// Build neural network for regression
var result = await new AiModelBuilder<double, double[][], double[]>()
    .ConfigureNeuralNetwork(config =>
    {
        config.TaskType = NeuralNetworkTaskType.Regression;
        config.InputSize = 4;
    })
    .ConfigureTraining(config =>
    {
        config.Epochs = 100;
        config.BatchSize = 16;
        config.LearningRate = 0.0001;
    })
    .ConfigurePreprocessing(config =>
    {
        config.NormalizeFeatures = true;
        config.NormalizeTargets = true;
    })
    .BuildAsync(productFeatures, prices);

// Predict price
var newProduct = new double[][] { new[] { 150.0, 4.3, 750.0, 2.0 } };
var predictedPrice = result.Predict(newProduct);
Console.WriteLine($"Predicted price: ${predictedPrice[0]:F2}");

// View metrics
Console.WriteLine($"R-Squared: {result.RSquared:F4}");
Console.WriteLine($"MAE: ${result.MeanAbsoluteError:F2}");
```

## Time Series Forecasting

```csharp
using AiDotNet;

// Historical sales data
var historicalData = new double[][]
{
    new[] { 100.0, 110.0, 105.0, 115.0, 120.0 },  // past 5 days
    new[] { 110.0, 105.0, 115.0, 120.0, 125.0 },
    new[] { 105.0, 115.0, 120.0, 125.0, 130.0 },
    // ... more sequences
};

var nextDaySales = new double[] { 125, 130, 135 };

// Build neural network for time series
var result = await new AiModelBuilder<double, double[][], double[]>()
    .ConfigureNeuralNetwork(config =>
    {
        config.TaskType = NeuralNetworkTaskType.TimeSeriesForecasting;
        config.SequenceLength = 5;
        config.ForecastHorizon = 1;
    })
    .ConfigureTraining(config =>
    {
        config.Epochs = 100;
        config.BatchSize = 32;
    })
    .BuildAsync(historicalData, nextDaySales);

// Forecast next day
var recentSales = new double[][] { new[] { 125.0, 130.0, 135.0, 140.0, 145.0 } };
var forecast = result.Predict(recentSales);
Console.WriteLine($"Forecasted sales: ${forecast[0]:F0}");
```

## Multi-Output Prediction

```csharp
using AiDotNet;

// Predict multiple targets at once
var features = new double[][]
{
    new[] { 1.0, 2.0, 3.0 },
    new[] { 2.0, 3.0, 4.0 },
    new[] { 3.0, 4.0, 5.0 }
};

// Multiple targets: [price, quantity, rating]
var multiTargets = new double[][]
{
    new[] { 10.0, 100.0, 4.5 },
    new[] { 15.0, 80.0, 4.2 },
    new[] { 12.0, 90.0, 4.7 }
};

var result = await new AiModelBuilder<double, double[][], double[][]>()
    .ConfigureNeuralNetwork(config =>
    {
        config.TaskType = NeuralNetworkTaskType.MultiOutputRegression;
        config.InputSize = 3;
        config.OutputSize = 3;
    })
    .ConfigureTraining(config =>
    {
        config.Epochs = 50;
        config.BatchSize = 16;
    })
    .BuildAsync(features, multiTargets);

// Predict multiple outputs
var newFeatures = new double[][] { new[] { 2.5, 3.5, 4.5 } };
var predictions = result.Predict(newFeatures);
Console.WriteLine($"Price: ${predictions[0][0]:F2}");
Console.WriteLine($"Quantity: {predictions[0][1]:F0}");
Console.WriteLine($"Rating: {predictions[0][2]:F1}");
```

## Training Configuration

```csharp
using AiDotNet;

// Sample training data (e.g., MNIST-like: 784 features per sample, 10 classes)
double[][] features = /* your feature matrix */;
double[] labels = /* your label array (0-9 for 10-class classification) */;

var result = await new AiModelBuilder<double, double[][], double[]>()
    .ConfigureNeuralNetwork(config =>
    {
        config.TaskType = NeuralNetworkTaskType.MultiClassClassification;
        config.InputSize = 784;
        config.NumClasses = 10;
    })
    .ConfigureTraining(config =>
    {
        // Basic training parameters
        config.Epochs = 100;
        config.BatchSize = 64;
        config.LearningRate = 0.001;
        config.ValidationSplit = 0.15;

        // Optimizer settings
        config.Optimizer = OptimizerType.Adam;
        config.WeightDecay = 0.0001;

        // Learning rate scheduling
        config.UseLearningRateScheduler = true;
        config.SchedulerType = SchedulerType.CosineAnnealing;

        // Early stopping
        config.UseEarlyStopping = true;
        config.Patience = 10;
        config.MinDelta = 0.001;

        // Regularization
        config.DropoutRate = 0.3;
    })
    .ConfigurePreprocessing()
    .BuildAsync(features, labels);

// Access training history
Console.WriteLine("\nTraining History:");
foreach (var epoch in result.TrainingHistory)
{
    Console.WriteLine($"Epoch {epoch.EpochNumber}: " +
        $"Loss={epoch.Loss:F4}, Acc={epoch.Accuracy:P2}, " +
        $"ValLoss={epoch.ValidationLoss:F4}, ValAcc={epoch.ValidationAccuracy:P2}");
}
```

## Data Augmentation

```csharp
using AiDotNet;

var result = await new AiModelBuilder<double, double[][], double[]>()
    .ConfigureNeuralNetwork(config =>
    {
        config.TaskType = NeuralNetworkTaskType.ImageClassification;
        config.InputShape = new[] { 32, 32, 3 };  // CIFAR-like images
        config.NumClasses = 10;
    })
    .ConfigurePreprocessing(config =>
    {
        config.NormalizeFeatures = true;
    })
    .ConfigureAugmentation(config =>
    {
        config.HorizontalFlip = true;
        config.RotationRange = 15;
        config.WidthShiftRange = 0.1;
        config.HeightShiftRange = 0.1;
        config.ZoomRange = 0.1;
    })
    .ConfigureTraining(config =>
    {
        config.Epochs = 50;
        config.BatchSize = 32;
    })
    .BuildAsync(images, labels);
```

## GPU Acceleration

```csharp
using AiDotNet;

var result = await new AiModelBuilder<double, double[][], double[]>()
    .ConfigureNeuralNetwork(config =>
    {
        config.TaskType = NeuralNetworkTaskType.ImageClassification;
        config.InputShape = new[] { 224, 224, 3 };
        config.NumClasses = 1000;
    })
    .ConfigureCompute(config =>
    {
        config.UseGpu = true;
        config.GpuDeviceId = 0;
        config.MixedPrecision = true;  // FP16 for faster training
    })
    .ConfigureTraining(config =>
    {
        config.Epochs = 100;
        config.BatchSize = 128;
    })
    .BuildAsync(images, labels);

Console.WriteLine($"Training completed on: {result.TrainingDevice}");
```

## Model Checkpointing and Resume

```csharp
using AiDotNet;

// Train with checkpoints
var result = await new AiModelBuilder<double, double[][], double[]>()
    .ConfigureNeuralNetwork(config =>
    {
        config.TaskType = NeuralNetworkTaskType.ImageClassification;
    })
    .ConfigureTraining(config =>
    {
        config.Epochs = 100;
        config.SaveCheckpoints = true;
        config.CheckpointPath = "./checkpoints";
        config.CheckpointFrequency = 10;  // Every 10 epochs
    })
    .BuildAsync(images, labels);

// Resume from checkpoint
var resumedResult = await new AiModelBuilder<double, double[][], double[]>()
    .ConfigureNeuralNetwork(config =>
    {
        config.TaskType = NeuralNetworkTaskType.ImageClassification;
    })
    .ConfigureTraining(config =>
    {
        config.Epochs = 50;  // Additional epochs
        config.ResumeFromCheckpoint = "./checkpoints/epoch_100.ckpt";
    })
    .BuildAsync(images, labels);
```

## Best Practices

1. **Start with small models**: Begin simple and increase complexity only if needed
2. **Use validation data**: Always monitor validation metrics to detect overfitting
3. **Normalize your data**: Neural networks train better with normalized inputs
4. **Use early stopping**: Prevent overfitting by stopping when validation loss increases
5. **Experiment with learning rates**: The learning rate is often the most important hyperparameter
6. **Use data augmentation**: Especially helpful for image tasks with limited data

## Troubleshooting

### Training too slow?
- Reduce batch size if memory is limited
- Enable GPU acceleration
- Use mixed precision training

### Overfitting?
- Add dropout via `config.DropoutRate`
- Enable early stopping
- Use data augmentation
- Reduce model complexity

### Underfitting?
- Train for more epochs
- Increase learning rate
- Increase model complexity
- Check data quality

## Summary

AiDotNet's `AiModelBuilder` makes neural network training accessible:
- Configure task type and the system builds the appropriate architecture
- Built-in training loop with validation and early stopping
- Automatic preprocessing and data augmentation
- GPU acceleration support
- Model checkpointing and resumption

All complexity is handled internally. You focus on your data and results.
