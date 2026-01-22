namespace AiDotNet.Playground.Services;

/// <summary>
/// Service providing interactive code examples for the playground.
/// All examples use the AiModelBuilder facade pattern.
/// </summary>
public class ExampleService
{
    private readonly Dictionary<string, List<CodeExample>> _examples;

    public ExampleService()
    {
        _examples = InitializeExamples();
    }

    public IEnumerable<string> GetCategories() => _examples.Keys;

    public IEnumerable<CodeExample> GetExamples(string category)
    {
        return _examples.TryGetValue(category, out var examples) ? examples : [];
    }

    public CodeExample? GetExample(string id)
    {
        return _examples.Values.SelectMany(x => x).FirstOrDefault(e => e.Id == id);
    }

    private Dictionary<string, List<CodeExample>> InitializeExamples()
    {
        return new Dictionary<string, List<CodeExample>>
        {
            ["Getting Started"] = new()
            {
                new CodeExample
                {
                    Id = "hello-world",
                    Name = "Hello World",
                    Description = "Your first AiDotNet program",
                    Difficulty = "Beginner",
                    Tags = ["basics", "introduction"],
                    Code = @"// Hello World - Your first AiDotNet program
using System;

Console.WriteLine(""Welcome to AiDotNet!"");
Console.WriteLine(""The comprehensive AI/ML framework for .NET"");
Console.WriteLine();
Console.WriteLine(""Key Features:"");
Console.WriteLine(""  - Simple facade pattern via AiModelBuilder"");
Console.WriteLine(""  - Build, train, and deploy ML models easily"");
Console.WriteLine(""  - Cross-platform: Windows, Linux, macOS"");
"
                },
                new CodeExample
                {
                    Id = "linear-regression",
                    Name = "Linear Regression",
                    Description = "Train a linear regression model using AiModelBuilder",
                    Difficulty = "Beginner",
                    Tags = ["regression", "linear", "basics"],
                    Code = @"// Linear Regression with AiModelBuilder
using AiDotNet;
using AiDotNet.Regression;

// Training data: X = input features, y = target values
var features = new double[,]
{
    { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 }, { 5.0 }
};
var labels = new double[] { 2.1, 4.0, 5.9, 8.1, 10.0 };

// Build and train the model using the facade
var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new LinearRegression<double>())
    .ConfigurePreprocessing()
    .BuildAsync(features, labels);

// Make predictions
var prediction = result.Predict(new double[] { 6.0 });
Console.WriteLine($""Prediction for x=6: {prediction:F2}"");
Console.WriteLine($""Model R² Score: {result.Metrics.RSquared:F4}"");
"
                },
                new CodeExample
                {
                    Id = "classification",
                    Name = "Classification",
                    Description = "Train a classifier using AiModelBuilder",
                    Difficulty = "Beginner",
                    Tags = ["classification", "basics"],
                    Code = @"// Classification with AiModelBuilder
using AiDotNet;
using AiDotNet.Classification;

// Training data: features and class labels
var features = new double[,]
{
    { 5.1, 3.5 }, { 4.9, 3.0 }, { 7.0, 3.2 }, { 6.4, 3.2 }
};
var labels = new double[] { 0, 0, 1, 1 }; // Binary classes

// Build and train using the facade pattern
var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new RandomForestClassifier<double>(nEstimators: 100))
    .ConfigurePreprocessing()
    .ConfigureCrossValidation(new KFoldCrossValidator<double>(k: 5))
    .BuildAsync(features, labels);

// Make predictions
var prediction = result.Predict(new double[] { 6.0, 3.1 });
Console.WriteLine($""Predicted class: {prediction}"");
Console.WriteLine($""Model Accuracy: {result.Metrics.Accuracy:P2}"");
"
                }
            },

            ["Regression"] = new()
            {
                new CodeExample
                {
                    Id = "ridge-regression",
                    Name = "Ridge Regression",
                    Description = "Linear regression with L2 regularization",
                    Difficulty = "Intermediate",
                    Tags = ["regression", "regularization"],
                    Code = @"// Ridge Regression with AiModelBuilder
using AiDotNet;
using AiDotNet.Regression;

var features = new double[,]
{
    { 1.0, 2.0 }, { 2.0, 3.0 }, { 3.0, 4.0 }, { 4.0, 5.0 }
};
var labels = new double[] { 3.0, 5.0, 7.0, 9.0 };

// Ridge regression with regularization
var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new RidgeRegression<double>(alpha: 1.0))
    .ConfigurePreprocessing()
    .BuildAsync(features, labels);

var prediction = result.Predict(new double[] { 5.0, 6.0 });
Console.WriteLine($""Prediction: {prediction:F2}"");
Console.WriteLine($""R² Score: {result.Metrics.RSquared:F4}"");
"
                },
                new CodeExample
                {
                    Id = "polynomial-regression",
                    Name = "Polynomial Regression",
                    Description = "Fit non-linear data with polynomial features",
                    Difficulty = "Intermediate",
                    Tags = ["regression", "polynomial"],
                    Code = @"// Polynomial Regression with AiModelBuilder
using AiDotNet;
using AiDotNet.Regression;

// Non-linear data (quadratic relationship)
var features = new double[,]
{
    { 1.0 }, { 2.0 }, { 3.0 }, { 4.0 }, { 5.0 }
};
var labels = new double[] { 1.0, 4.0, 9.0, 16.0, 25.0 }; // y = x²

var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new PolynomialRegression<double>(degree: 2))
    .ConfigurePreprocessing()
    .BuildAsync(features, labels);

var prediction = result.Predict(new double[] { 6.0 });
Console.WriteLine($""Prediction for x=6: {prediction:F2}"");
Console.WriteLine($""Expected (6²): 36.00"");
"
                },
                new CodeExample
                {
                    Id = "gradient-boosting-regressor",
                    Name = "Gradient Boosting",
                    Description = "Ensemble method for regression",
                    Difficulty = "Advanced",
                    Tags = ["regression", "ensemble", "boosting"],
                    Code = @"// Gradient Boosting Regression with AiModelBuilder
using AiDotNet;
using AiDotNet.Regression.Ensemble;

var features = new double[,]
{
    { 1.0, 0.5 }, { 2.0, 1.0 }, { 3.0, 1.5 }, { 4.0, 2.0 }, { 5.0, 2.5 }
};
var labels = new double[] { 1.5, 3.0, 4.5, 6.0, 7.5 };

var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new GradientBoostingRegressor<double>(
        nEstimators: 100,
        learningRate: 0.1,
        maxDepth: 3))
    .ConfigurePreprocessing()
    .BuildAsync(features, labels);

var prediction = result.Predict(new double[] { 6.0, 3.0 });
Console.WriteLine($""Prediction: {prediction:F2}"");
"
                }
            },

            ["Classification"] = new()
            {
                new CodeExample
                {
                    Id = "logistic-regression",
                    Name = "Logistic Regression",
                    Description = "Binary classification with logistic regression",
                    Difficulty = "Beginner",
                    Tags = ["classification", "logistic", "binary"],
                    Code = @"// Logistic Regression with AiModelBuilder
using AiDotNet;
using AiDotNet.Classification;

var features = new double[,]
{
    { 1.0, 2.0 }, { 2.0, 1.0 }, { 3.0, 3.0 }, { 4.0, 2.0 }
};
var labels = new double[] { 0, 0, 1, 1 };

var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new LogisticRegression<double>())
    .ConfigurePreprocessing()
    .BuildAsync(features, labels);

var prediction = result.Predict(new double[] { 3.5, 2.5 });
Console.WriteLine($""Predicted class: {prediction}"");
Console.WriteLine($""Accuracy: {result.Metrics.Accuracy:P2}"");
"
                },
                new CodeExample
                {
                    Id = "svm-classifier",
                    Name = "Support Vector Machine",
                    Description = "SVM classifier for classification tasks",
                    Difficulty = "Intermediate",
                    Tags = ["classification", "svm"],
                    Code = @"// SVM Classification with AiModelBuilder
using AiDotNet;
using AiDotNet.Classification.SVM;

var features = new double[,]
{
    { 1.0, 1.0 }, { 1.5, 2.0 }, { 3.0, 3.0 }, { 3.5, 3.5 }
};
var labels = new double[] { 0, 0, 1, 1 };

var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new SupportVectorClassifier<double>(kernel: ""rbf""))
    .ConfigurePreprocessing()
    .BuildAsync(features, labels);

var prediction = result.Predict(new double[] { 2.5, 2.5 });
Console.WriteLine($""Predicted class: {prediction}"");
"
                },
                new CodeExample
                {
                    Id = "naive-bayes",
                    Name = "Naive Bayes",
                    Description = "Probabilistic classifier using Bayes theorem",
                    Difficulty = "Beginner",
                    Tags = ["classification", "probabilistic"],
                    Code = @"// Naive Bayes Classification with AiModelBuilder
using AiDotNet;
using AiDotNet.Classification.NaiveBayes;

var features = new double[,]
{
    { 1.0, 2.0 }, { 1.5, 1.8 }, { 5.0, 8.0 }, { 6.0, 9.0 }
};
var labels = new double[] { 0, 0, 1, 1 };

var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new GaussianNaiveBayes<double>())
    .ConfigurePreprocessing()
    .BuildAsync(features, labels);

var prediction = result.Predict(new double[] { 3.0, 4.0 });
Console.WriteLine($""Predicted class: {prediction}"");
"
                }
            },

            ["Clustering"] = new()
            {
                new CodeExample
                {
                    Id = "kmeans",
                    Name = "K-Means Clustering",
                    Description = "Partition data into K clusters",
                    Difficulty = "Beginner",
                    Tags = ["clustering", "kmeans", "unsupervised"],
                    Code = @"// K-Means Clustering with AiModelBuilder
using AiDotNet;
using AiDotNet.Clustering.Partitioning;

var data = new double[,]
{
    { 1.0, 1.0 }, { 1.5, 2.0 }, { 3.0, 4.0 },
    { 5.0, 7.0 }, { 3.5, 5.0 }, { 4.5, 5.0 }
};

var result = await new AiModelBuilder<double, double[], int>()
    .ConfigureModel(new KMeans<double>(nClusters: 2))
    .BuildAsync(data);

Console.WriteLine(""Cluster assignments:"");
for (int i = 0; i < data.GetLength(0); i++)
{
    var cluster = result.Predict(new double[] { data[i, 0], data[i, 1] });
    Console.WriteLine($""  Point ({data[i, 0]}, {data[i, 1]}) -> Cluster {cluster}"");
}
"
                },
                new CodeExample
                {
                    Id = "dbscan",
                    Name = "DBSCAN",
                    Description = "Density-based clustering that finds arbitrarily shaped clusters",
                    Difficulty = "Intermediate",
                    Tags = ["clustering", "density", "unsupervised"],
                    Code = @"// DBSCAN Clustering with AiModelBuilder
using AiDotNet;
using AiDotNet.Clustering.Density;

var data = new double[,]
{
    { 1.0, 1.0 }, { 1.1, 1.1 }, { 0.9, 1.0 },
    { 5.0, 5.0 }, { 5.1, 5.1 }, { 4.9, 5.0 },
    { 10.0, 10.0 } // Outlier/noise point
};

var result = await new AiModelBuilder<double, double[], int>()
    .ConfigureModel(new DBSCAN<double>(eps: 0.5, minSamples: 2))
    .BuildAsync(data);

Console.WriteLine(""Cluster assignments (−1 = noise):"");
for (int i = 0; i < data.GetLength(0); i++)
{
    var cluster = result.Predict(new double[] { data[i, 0], data[i, 1] });
    Console.WriteLine($""  Point ({data[i, 0]}, {data[i, 1]}) -> Cluster {cluster}"");
}
"
                },
                new CodeExample
                {
                    Id = "hierarchical",
                    Name = "Hierarchical Clustering",
                    Description = "Build a hierarchy of clusters",
                    Difficulty = "Intermediate",
                    Tags = ["clustering", "hierarchical", "unsupervised"],
                    Code = @"// Hierarchical Clustering with AiModelBuilder
using AiDotNet;
using AiDotNet.Clustering.Hierarchical;

var data = new double[,]
{
    { 1.0, 1.0 }, { 1.5, 1.5 }, { 5.0, 5.0 }, { 5.5, 5.5 }
};

var result = await new AiModelBuilder<double, double[], int>()
    .ConfigureModel(new AgglomerativeClustering<double>(
        nClusters: 2,
        linkage: LinkageMethod.Ward))
    .BuildAsync(data);

Console.WriteLine(""Hierarchical cluster assignments:"");
for (int i = 0; i < data.GetLength(0); i++)
{
    var cluster = result.Predict(new double[] { data[i, 0], data[i, 1] });
    Console.WriteLine($""  Point ({data[i, 0]}, {data[i, 1]}) -> Cluster {cluster}"");
}
"
                }
            },

            ["Neural Networks"] = new()
            {
                new CodeExample
                {
                    Id = "neural-net-basic",
                    Name = "Basic Neural Network",
                    Description = "Simple feedforward neural network for classification",
                    Difficulty = "Intermediate",
                    Tags = ["neural-network", "classification"],
                    Code = @"// Neural Network with AiModelBuilder
using AiDotNet;
using AiDotNet.NeuralNetworks;

var features = new double[,]
{
    { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 }
};
var labels = new double[] { 0, 1, 1, 0 }; // XOR problem

var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new NeuralNetworkBuilder<double>()
        .AddDenseLayer(inputSize: 2, outputSize: 4)
        .AddActivation(ActivationType.ReLU)
        .AddDenseLayer(inputSize: 4, outputSize: 1)
        .AddActivation(ActivationType.Sigmoid)
        .Build())
    .ConfigureOptimizer(new Adam<double>(learningRate: 0.01))
    .ConfigureTraining(epochs: 1000)
    .BuildAsync(features, labels);

Console.WriteLine(""XOR Neural Network Results:"");
foreach (var input in new[] { new[] { 0.0, 0.0 }, new[] { 0.0, 1.0 }, new[] { 1.0, 0.0 }, new[] { 1.0, 1.0 } })
{
    var pred = result.Predict(input);
    Console.WriteLine($""  {input[0]} XOR {input[1]} = {pred:F2}"");
}
"
                },
                new CodeExample
                {
                    Id = "neural-net-regression",
                    Name = "Neural Network Regression",
                    Description = "Neural network for continuous value prediction",
                    Difficulty = "Intermediate",
                    Tags = ["neural-network", "regression"],
                    Code = @"// Neural Network Regression with AiModelBuilder
using AiDotNet;
using AiDotNet.NeuralNetworks;

// Function approximation: y = sin(x)
var features = new double[100, 1];
var labels = new double[100];
for (int i = 0; i < 100; i++)
{
    double x = i * 0.1;
    features[i, 0] = x;
    labels[i] = Math.Sin(x);
}

var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new NeuralNetworkBuilder<double>()
        .AddDenseLayer(inputSize: 1, outputSize: 32)
        .AddActivation(ActivationType.ReLU)
        .AddDenseLayer(inputSize: 32, outputSize: 1)
        .Build())
    .ConfigureOptimizer(new Adam<double>(learningRate: 0.001))
    .ConfigureTraining(epochs: 500)
    .BuildAsync(features, labels);

Console.WriteLine(""Sin(x) approximation:"");
Console.WriteLine($""  Sin(1.57) predicted: {result.Predict(new[] { 1.57 }):F4}"");
Console.WriteLine($""  Sin(1.57) actual:    {Math.Sin(1.57):F4}"");
"
                }
            },

            ["Time Series"] = new()
            {
                new CodeExample
                {
                    Id = "arima",
                    Name = "ARIMA Forecasting",
                    Description = "Time series forecasting with ARIMA",
                    Difficulty = "Intermediate",
                    Tags = ["time-series", "forecasting", "arima"],
                    Code = @"// ARIMA Time Series Forecasting with AiModelBuilder
using AiDotNet;
using AiDotNet.TimeSeries;

// Monthly sales data
var data = new double[] { 100, 120, 130, 125, 140, 150, 160, 155, 170, 180, 190, 200 };

var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new ARIMA<double>(p: 1, d: 1, q: 1))
    .BuildAsync(data);

Console.WriteLine(""ARIMA Forecast (next 3 periods):"");
var forecast = result.Forecast(steps: 3);
for (int i = 0; i < forecast.Length; i++)
{
    Console.WriteLine($""  Period {data.Length + i + 1}: {forecast[i]:F2}"");
}
"
                },
                new CodeExample
                {
                    Id = "exponential-smoothing",
                    Name = "Exponential Smoothing",
                    Description = "Simple exponential smoothing for forecasting",
                    Difficulty = "Beginner",
                    Tags = ["time-series", "forecasting", "smoothing"],
                    Code = @"// Exponential Smoothing with AiModelBuilder
using AiDotNet;
using AiDotNet.TimeSeries;

var data = new double[] { 10, 12, 13, 15, 14, 16, 18, 17, 19, 20 };

var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new ExponentialSmoothing<double>(alpha: 0.3))
    .BuildAsync(data);

Console.WriteLine(""Smoothed values and forecast:"");
var smoothed = result.Transform(data);
Console.WriteLine($""  Last smoothed: {smoothed[^1]:F2}"");
Console.WriteLine($""  Next forecast: {result.Forecast(steps: 1)[0]:F2}"");
"
                }
            },

            ["Advanced"] = new()
            {
                new CodeExample
                {
                    Id = "cross-validation",
                    Name = "Cross-Validation",
                    Description = "Evaluate model performance with K-Fold cross-validation",
                    Difficulty = "Intermediate",
                    Tags = ["validation", "evaluation"],
                    Code = @"// Cross-Validation with AiModelBuilder
using AiDotNet;
using AiDotNet.Classification;

var features = new double[100, 2];
var labels = new double[100];
var rng = new Random(42);
for (int i = 0; i < 100; i++)
{
    features[i, 0] = rng.NextDouble() * 10;
    features[i, 1] = rng.NextDouble() * 10;
    labels[i] = features[i, 0] + features[i, 1] > 10 ? 1 : 0;
}

var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new RandomForestClassifier<double>(nEstimators: 50))
    .ConfigurePreprocessing()
    .ConfigureCrossValidation(new KFoldCrossValidator<double>(k: 5))
    .BuildAsync(features, labels);

Console.WriteLine(""5-Fold Cross-Validation Results:"");
Console.WriteLine($""  Mean Accuracy: {result.CrossValidationMetrics.MeanAccuracy:P2}"");
Console.WriteLine($""  Std Deviation: {result.CrossValidationMetrics.StdAccuracy:F4}"");
"
                },
                new CodeExample
                {
                    Id = "hyperparameter-tuning",
                    Name = "Hyperparameter Tuning",
                    Description = "Automatically find the best model parameters",
                    Difficulty = "Advanced",
                    Tags = ["automl", "hyperparameter", "optimization"],
                    Code = @"// Hyperparameter Tuning with AiModelBuilder
using AiDotNet;
using AiDotNet.Classification;

var features = new double[200, 4];
var labels = new double[200];
var rng = new Random(42);
for (int i = 0; i < 200; i++)
{
    for (int j = 0; j < 4; j++)
        features[i, j] = rng.NextDouble();
    labels[i] = features[i, 0] > 0.5 ? 1 : 0;
}

var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new RandomForestClassifier<double>())
    .ConfigurePreprocessing()
    .ConfigureHyperparameterOptimization(new HyperparameterSearchSpace()
        .AddIntParameter(""nEstimators"", 10, 200)
        .AddIntParameter(""maxDepth"", 3, 20)
        .AddFloatParameter(""minSamplesSplit"", 0.01, 0.5))
    .BuildAsync(features, labels);

Console.WriteLine(""Best hyperparameters found:"");
Console.WriteLine($""  n_estimators: {result.BestParameters[""nEstimators""]}"");
Console.WriteLine($""  max_depth: {result.BestParameters[""maxDepth""]}"");
Console.WriteLine($""  Best Accuracy: {result.Metrics.Accuracy:P2}"");
"
                },
                new CodeExample
                {
                    Id = "model-persistence",
                    Name = "Save and Load Models",
                    Description = "Persist trained models for later use",
                    Difficulty = "Beginner",
                    Tags = ["persistence", "save", "load"],
                    Code = @"// Model Persistence with AiModelBuilder
using AiDotNet;
using AiDotNet.Classification;

// Train a model
var features = new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 }, { 7, 8 } };
var labels = new double[] { 0, 0, 1, 1 };

var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new RandomForestClassifier<double>(nEstimators: 50))
    .BuildAsync(features, labels);

// Save the trained model
await result.SaveAsync(""my_model.aidotnet"");
Console.WriteLine(""Model saved to my_model.aidotnet"");

// Load the model later
var loadedResult = await AiModelResult<double, double[], double>
    .LoadAsync(""my_model.aidotnet"");
Console.WriteLine(""Model loaded successfully"");

// Use the loaded model
var prediction = loadedResult.Predict(new double[] { 6, 7 });
Console.WriteLine($""Prediction: {prediction}"");
"
                }
            },

            ["Computer Vision"] = new()
            {
                new CodeExample
                {
                    Id = "image-classification-cnn",
                    Name = "Image Classification (CNN)",
                    Description = "Classify images using a Convolutional Neural Network",
                    Difficulty = "Intermediate",
                    Tags = ["cnn", "image", "classification", "deep-learning"],
                    Code = @"// Image Classification with CNN using AiModelBuilder
using AiDotNet;
using AiDotNet.NeuralNetworks;

// Simulated image data: 100 samples of 28x28 grayscale images (flattened)
var images = new double[100, 784]; // 28*28 = 784
var labels = new double[100];
var rng = new Random(42);
for (int i = 0; i < 100; i++)
{
    for (int j = 0; j < 784; j++)
        images[i, j] = rng.NextDouble();
    labels[i] = i % 10; // 10 classes (digits 0-9)
}

var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new NeuralNetworkBuilder<double>()
        .AddConv2DLayer(inputChannels: 1, outputChannels: 32, kernelSize: 3)
        .AddActivation(ActivationType.ReLU)
        .AddMaxPooling2D(poolSize: 2)
        .AddConv2DLayer(inputChannels: 32, outputChannels: 64, kernelSize: 3)
        .AddActivation(ActivationType.ReLU)
        .AddFlatten()
        .AddDenseLayer(inputSize: 64 * 5 * 5, outputSize: 128)
        .AddActivation(ActivationType.ReLU)
        .AddDenseLayer(inputSize: 128, outputSize: 10)
        .AddActivation(ActivationType.Softmax)
        .Build())
    .ConfigureOptimizer(new Adam<double>(learningRate: 0.001))
    .ConfigureTraining(epochs: 10, batchSize: 32)
    .BuildAsync(images, labels);

Console.WriteLine($""Training Accuracy: {result.Metrics.Accuracy:P2}"");
"
                },
                new CodeExample
                {
                    Id = "resnet-transfer",
                    Name = "ResNet Transfer Learning",
                    Description = "Use pre-trained ResNet for image classification",
                    Difficulty = "Advanced",
                    Tags = ["resnet", "transfer-learning", "pretrained"],
                    Code = @"// ResNet Transfer Learning with AiModelBuilder
using AiDotNet;
using AiDotNet.NeuralNetworks.Architectures;

// Load pre-trained ResNet and fine-tune for custom classes
var images = new double[50, 224 * 224 * 3]; // 224x224 RGB images
var labels = new double[50];
var rng = new Random(42);
for (int i = 0; i < 50; i++)
{
    for (int j = 0; j < 224 * 224 * 3; j++)
        images[i, j] = rng.NextDouble();
    labels[i] = i % 5; // 5 custom classes
}

var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new ResNetBuilder<double>()
        .UseResNet18(pretrained: true)
        .FreezeBaseLayers()
        .ReplaceClassificationHead(numClasses: 5)
        .Build())
    .ConfigureOptimizer(new Adam<double>(learningRate: 0.0001))
    .ConfigureTraining(epochs: 5)
    .BuildAsync(images, labels);

Console.WriteLine($""Fine-tuned Accuracy: {result.Metrics.Accuracy:P2}"");
"
                },
                new CodeExample
                {
                    Id = "object-detection-yolo",
                    Name = "Object Detection (YOLO)",
                    Description = "Detect objects in images using YOLO",
                    Difficulty = "Advanced",
                    Tags = ["yolo", "object-detection", "deep-learning"],
                    Code = @"// Object Detection with YOLO using AiModelBuilder
using AiDotNet;
using AiDotNet.ComputerVision;

// Simulated image for object detection
var image = new double[1, 416 * 416 * 3]; // 416x416 RGB image
var rng = new Random(42);
for (int i = 0; i < 416 * 416 * 3; i++)
    image[0, i] = rng.NextDouble();

var result = await new AiModelBuilder<double, double[], DetectionResult>()
    .ConfigureModel(new YOLOv8<double>(modelSize: YOLOSize.Small))
    .BuildAsync(image);

Console.WriteLine(""Detected Objects:"");
foreach (var detection in result.Detections)
{
    Console.WriteLine($""  {detection.ClassName}: {detection.Confidence:P1}"");
    Console.WriteLine($""    Box: ({detection.X}, {detection.Y}, {detection.Width}, {detection.Height})"");
}
"
                }
            },

            ["Preprocessing"] = new()
            {
                new CodeExample
                {
                    Id = "standard-scaler",
                    Name = "Standard Scaling",
                    Description = "Normalize features to zero mean and unit variance",
                    Difficulty = "Beginner",
                    Tags = ["preprocessing", "scaling", "normalization"],
                    Code = @"// Standard Scaling with AiModelBuilder
using AiDotNet;
using AiDotNet.Preprocessing;

var features = new double[,]
{
    { 100, 0.001 }, { 200, 0.002 }, { 150, 0.0015 }, { 300, 0.003 }
};
var labels = new double[] { 1, 2, 1.5, 3 };

// StandardScaler normalizes to mean=0, std=1
var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new LinearRegression<double>())
    .ConfigurePreprocessing(new PreprocessingPipeline<double>()
        .AddScaler(new StandardScaler<double>()))
    .BuildAsync(features, labels);

Console.WriteLine(""Model trained with standardized features"");
Console.WriteLine($""R² Score: {result.Metrics.RSquared:F4}"");
"
                },
                new CodeExample
                {
                    Id = "minmax-scaler",
                    Name = "Min-Max Scaling",
                    Description = "Scale features to a specified range [0, 1]",
                    Difficulty = "Beginner",
                    Tags = ["preprocessing", "scaling", "minmax"],
                    Code = @"// Min-Max Scaling with AiModelBuilder
using AiDotNet;
using AiDotNet.Preprocessing;

var features = new double[,]
{
    { 10, 100 }, { 20, 200 }, { 15, 150 }, { 30, 300 }
};
var labels = new double[] { 0, 1, 0, 1 };

// MinMaxScaler scales to [0, 1] range
var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new LogisticRegression<double>())
    .ConfigurePreprocessing(new PreprocessingPipeline<double>()
        .AddScaler(new MinMaxScaler<double>(featureRange: (0, 1))))
    .BuildAsync(features, labels);

Console.WriteLine(""Model trained with min-max scaled features"");
Console.WriteLine($""Accuracy: {result.Metrics.Accuracy:P2}"");
"
                },
                new CodeExample
                {
                    Id = "preprocessing-pipeline",
                    Name = "Preprocessing Pipeline",
                    Description = "Chain multiple preprocessing steps together",
                    Difficulty = "Intermediate",
                    Tags = ["preprocessing", "pipeline", "feature-engineering"],
                    Code = @"// Preprocessing Pipeline with AiModelBuilder
using AiDotNet;
using AiDotNet.Preprocessing;

var features = new double[100, 10];
var labels = new double[100];
var rng = new Random(42);
for (int i = 0; i < 100; i++)
{
    for (int j = 0; j < 10; j++)
        features[i, j] = rng.NextDouble() * 100;
    labels[i] = features[i, 0] > 50 ? 1 : 0;
}

// Chain multiple preprocessing steps
var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new RandomForestClassifier<double>())
    .ConfigurePreprocessing(new PreprocessingPipeline<double>()
        .AddImputer(new SimpleImputer<double>(strategy: ImputeStrategy.Mean))
        .AddScaler(new StandardScaler<double>())
        .AddFeatureSelector(new VarianceThreshold<double>(threshold: 0.01)))
    .BuildAsync(features, labels);

Console.WriteLine(""Pipeline: Impute -> Scale -> Feature Selection"");
Console.WriteLine($""Accuracy: {result.Metrics.Accuracy:P2}"");
"
                }
            },

            ["Anomaly Detection"] = new()
            {
                new CodeExample
                {
                    Id = "isolation-forest",
                    Name = "Isolation Forest",
                    Description = "Detect anomalies using isolation-based method",
                    Difficulty = "Intermediate",
                    Tags = ["anomaly", "outlier", "isolation-forest"],
                    Code = @"// Isolation Forest with AiModelBuilder
using AiDotNet;
using AiDotNet.AnomalyDetection;

// Normal data with some anomalies
var data = new double[110, 2];
var rng = new Random(42);
// 100 normal points clustered around (0, 0)
for (int i = 0; i < 100; i++)
{
    data[i, 0] = rng.NextDouble() * 2 - 1;
    data[i, 1] = rng.NextDouble() * 2 - 1;
}
// 10 anomalies far from the cluster
for (int i = 100; i < 110; i++)
{
    data[i, 0] = rng.NextDouble() * 10 + 5;
    data[i, 1] = rng.NextDouble() * 10 + 5;
}

var result = await new AiModelBuilder<double, double[], int>()
    .ConfigureModel(new IsolationForest<double>(
        nEstimators: 100,
        contamination: 0.1))
    .BuildAsync(data);

Console.WriteLine(""Anomaly Detection Results:"");
int anomalyCount = 0;
for (int i = 0; i < 110; i++)
{
    var pred = result.Predict(new[] { data[i, 0], data[i, 1] });
    if (pred == -1) anomalyCount++;
}
Console.WriteLine($""  Detected {anomalyCount} anomalies out of 110 samples"");
"
                },
                new CodeExample
                {
                    Id = "one-class-svm",
                    Name = "One-Class SVM",
                    Description = "Novelty detection using One-Class SVM",
                    Difficulty = "Intermediate",
                    Tags = ["anomaly", "svm", "novelty"],
                    Code = @"// One-Class SVM with AiModelBuilder
using AiDotNet;
using AiDotNet.AnomalyDetection;

// Train on normal data only
var normalData = new double[100, 2];
var rng = new Random(42);
for (int i = 0; i < 100; i++)
{
    // Box-Muller transform for Gaussian random numbers
    double u1 = 1.0 - rng.NextDouble();
    double u2 = 1.0 - rng.NextDouble();
    double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    normalData[i, 0] = randStdNormal;
    normalData[i, 1] = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
}

var result = await new AiModelBuilder<double, double[], int>()
    .ConfigureModel(new OneClassSVM<double>(
        kernel: ""rbf"",
        nu: 0.1))
    .BuildAsync(normalData);

// Test on new data
var normalTest = new double[] { 0.5, 0.3 };
var anomalyTest = new double[] { 10.0, 10.0 };

Console.WriteLine($""Normal point (0.5, 0.3): {(result.Predict(normalTest) == 1 ? ""Normal"" : ""Anomaly"")}"");
Console.WriteLine($""Anomaly point (10, 10): {(result.Predict(anomalyTest) == 1 ? ""Normal"" : ""Anomaly"")}"");
"
                }
            },

            ["Dimensionality Reduction"] = new()
            {
                new CodeExample
                {
                    Id = "pca",
                    Name = "Principal Component Analysis",
                    Description = "Reduce dimensionality while preserving variance",
                    Difficulty = "Intermediate",
                    Tags = ["pca", "dimensionality", "feature-extraction"],
                    Code = @"// PCA with AiModelBuilder
using AiDotNet;
using AiDotNet.DimensionalityReduction;

// High-dimensional data
var data = new double[100, 50]; // 50 features
var rng = new Random(42);
for (int i = 0; i < 100; i++)
    for (int j = 0; j < 50; j++)
        data[i, j] = rng.NextDouble();

var result = await new AiModelBuilder<double, double[], double[]>()
    .ConfigureModel(new PCA<double>(nComponents: 10))
    .BuildAsync(data);

Console.WriteLine(""PCA Results:"");
Console.WriteLine($""  Original dimensions: 50"");
Console.WriteLine($""  Reduced dimensions: 10"");
Console.WriteLine($""  Explained variance ratio: {result.Metrics.ExplainedVarianceRatio:P1}"");

// Transform new data
var newSample = new double[50];
var transformed = result.Transform(newSample);
Console.WriteLine($""  New sample transformed: {transformed.Length} dimensions"");
"
                },
                new CodeExample
                {
                    Id = "tsne",
                    Name = "t-SNE Visualization",
                    Description = "Visualize high-dimensional data in 2D",
                    Difficulty = "Advanced",
                    Tags = ["tsne", "visualization", "embedding"],
                    Code = @"// t-SNE with AiModelBuilder
using AiDotNet;
using AiDotNet.DimensionalityReduction;

// High-dimensional data for visualization
var data = new double[200, 100]; // 100 features
var labels = new double[200];
var rng = new Random(42);
for (int i = 0; i < 200; i++)
{
    labels[i] = i % 4; // 4 classes
    double offset = labels[i] * 2;
    for (int j = 0; j < 100; j++)
        data[i, j] = rng.NextDouble() + offset;
}

var result = await new AiModelBuilder<double, double[], double[]>()
    .ConfigureModel(new TSNE<double>(
        nComponents: 2,
        perplexity: 30,
        learningRate: 200))
    .BuildAsync(data);

Console.WriteLine(""t-SNE Embedding Results:"");
Console.WriteLine($""  Reduced to 2D for visualization"");
Console.WriteLine($""  KL Divergence: {result.Metrics.KLDivergence:F4}"");
"
                }
            },

            ["NLP & Text Processing"] = new()
            {
                new CodeExample
                {
                    Id = "text-classification",
                    Name = "Text Classification",
                    Description = "Classify text documents using TF-IDF and machine learning",
                    Difficulty = "Intermediate",
                    Tags = ["nlp", "text", "classification", "tfidf"],
                    Code = @"// Text Classification with AiModelBuilder
using AiDotNet;
using AiDotNet.NLP;
using AiDotNet.Classification;

// Sample text documents
var documents = new string[]
{
    ""I love this product, it's amazing!"",
    ""Great quality and fast shipping"",
    ""Terrible experience, would not recommend"",
    ""Worst purchase ever, complete waste"",
    ""Excellent service, highly satisfied"",
    ""Poor quality, very disappointed""
};
var labels = new double[] { 1, 1, 0, 0, 1, 0 }; // 1 = positive, 0 = negative

var result = await new AiModelBuilder<double, string, double>()
    .ConfigureModel(new LogisticRegression<double>())
    .ConfigurePreprocessing(new TextPreprocessingPipeline<double>()
        .AddTokenizer(new WordTokenizer())
        .AddVectorizer(new TfidfVectorizer<double>(maxFeatures: 1000)))
    .BuildAsync(documents, labels);

// Classify new text
var newReview = ""This is an excellent product!"";
var prediction = result.Predict(newReview);
Console.WriteLine($""Review: '{newReview}'"");
Console.WriteLine($""Sentiment: {(prediction == 1 ? ""Positive"" : ""Negative"")}"");
"
                },
                new CodeExample
                {
                    Id = "word-embeddings",
                    Name = "Word Embeddings",
                    Description = "Create and use word embeddings for semantic similarity",
                    Difficulty = "Intermediate",
                    Tags = ["nlp", "embeddings", "word2vec"],
                    Code = @"// Word Embeddings with AiModelBuilder
using AiDotNet;
using AiDotNet.NLP.Embeddings;

// Training corpus
var corpus = new string[]
{
    ""The cat sat on the mat"",
    ""The dog ran in the park"",
    ""Cats and dogs are pets"",
    ""Machine learning is fascinating"",
    ""Deep learning uses neural networks""
};

var result = await new AiModelBuilder<double, string[], double[,]>()
    .ConfigureModel(new Word2Vec<double>(
        vectorSize: 100,
        windowSize: 5,
        minCount: 1))
    .BuildAsync(corpus);

// Get word vectors
var catVector = result.GetWordVector(""cat"");
var dogVector = result.GetWordVector(""dog"");

// Calculate similarity
var similarity = result.CosineSimilarity(""cat"", ""dog"");
Console.WriteLine($""Similarity between 'cat' and 'dog': {similarity:F4}"");

// Find similar words
var similarWords = result.MostSimilar(""learning"", topN: 3);
Console.WriteLine(""Words similar to 'learning':"");
foreach (var (word, score) in similarWords)
    Console.WriteLine($""  {word}: {score:F4}"");
"
                },
                new CodeExample
                {
                    Id = "transformer-text",
                    Name = "Transformer for Text",
                    Description = "Use transformer architecture for text processing",
                    Difficulty = "Advanced",
                    Tags = ["nlp", "transformer", "attention", "deep-learning"],
                    Code = @"// Transformer for Text with AiModelBuilder
using AiDotNet;
using AiDotNet.NeuralNetworks;
using AiDotNet.NLP;

// Tokenized text sequences (simulated)
var sequences = new double[100, 128]; // 100 samples, max length 128
var labels = new double[100];
var rng = new Random(42);
for (int i = 0; i < 100; i++)
{
    for (int j = 0; j < 128; j++)
        sequences[i, j] = rng.Next(0, 10000); // vocab size 10000
    labels[i] = i % 2; // binary classification
}

var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new NeuralNetworkBuilder<double>()
        .AddEmbeddingLayer(vocabSize: 10000, embeddingDim: 256)
        .AddTransformerEncoder(
            numHeads: 8,
            hiddenDim: 512,
            numLayers: 4,
            dropout: 0.1)
        .AddGlobalAveragePooling()
        .AddDenseLayer(inputSize: 256, outputSize: 64)
        .AddActivation(ActivationType.ReLU)
        .AddDenseLayer(inputSize: 64, outputSize: 1)
        .AddActivation(ActivationType.Sigmoid)
        .Build())
    .ConfigureOptimizer(new Adam<double>(learningRate: 0.0001))
    .ConfigureTraining(epochs: 10, batchSize: 16)
    .BuildAsync(sequences, labels);

Console.WriteLine($""Transformer Text Classifier Accuracy: {result.Metrics.Accuracy:P2}"");
"
                },
                new CodeExample
                {
                    Id = "named-entity-recognition",
                    Name = "Named Entity Recognition",
                    Description = "Extract named entities from text",
                    Difficulty = "Advanced",
                    Tags = ["nlp", "ner", "sequence-labeling"],
                    Code = @"// Named Entity Recognition with AiModelBuilder
using AiDotNet;
using AiDotNet.NLP;

// Training data: tokens and their entity labels
var tokens = new string[][]
{
    new[] { ""John"", ""works"", ""at"", ""Microsoft"", ""in"", ""Seattle"" },
    new[] { ""Apple"", ""CEO"", ""Tim"", ""Cook"", ""announced"", ""new"", ""products"" }
};
var entityLabels = new string[][]
{
    new[] { ""B-PER"", ""O"", ""O"", ""B-ORG"", ""O"", ""B-LOC"" },
    new[] { ""B-ORG"", ""O"", ""B-PER"", ""I-PER"", ""O"", ""O"", ""O"" }
};

var result = await new AiModelBuilder<double, string[], string[]>()
    .ConfigureModel(new BiLSTMCRF<double>(
        hiddenSize: 128,
        numLayers: 2))
    .ConfigurePreprocessing(new NERPreprocessor<double>())
    .BuildAsync(tokens, entityLabels);

// Extract entities from new text
var newTokens = new[] { ""Elon"", ""Musk"", ""founded"", ""SpaceX"" };
var entities = result.Predict(newTokens);
Console.WriteLine(""Named Entity Recognition:"");
for (int i = 0; i < newTokens.Length; i++)
    Console.WriteLine($""  {newTokens[i]}: {entities[i]}"");
"
                }
            },

            ["Audio Processing"] = new()
            {
                new CodeExample
                {
                    Id = "speech-recognition",
                    Name = "Speech Recognition",
                    Description = "Transcribe audio to text using Whisper",
                    Difficulty = "Advanced",
                    Tags = ["audio", "speech", "whisper", "asr"],
                    Code = @"// Speech Recognition with AiModelBuilder
using AiDotNet;
using AiDotNet.Audio.Whisper;

// Simulated audio waveform (in practice, load from file)
var sampleRate = 16000;
var duration = 5.0; // seconds
var audioSamples = new double[(int)(sampleRate * duration)];
var rng = new Random(42);
for (int i = 0; i < audioSamples.Length; i++)
    audioSamples[i] = rng.NextDouble() * 2 - 1;

var result = await new AiModelBuilder<double, double[], string>()
    .ConfigureModel(new WhisperModel<double>(
        modelSize: WhisperModelSize.Base,
        language: ""en""))
    .BuildAsync(audioSamples);

Console.WriteLine(""Speech Recognition Result:"");
Console.WriteLine($""  Transcription: {result.Transcription}"");
Console.WriteLine($""  Confidence: {result.Confidence:P2}"");
Console.WriteLine($""  Detected Language: {result.DetectedLanguage}"");
"
                },
                new CodeExample
                {
                    Id = "audio-classification",
                    Name = "Audio Classification",
                    Description = "Classify audio clips by content type",
                    Difficulty = "Intermediate",
                    Tags = ["audio", "classification", "mfcc"],
                    Code = @"// Audio Classification with AiModelBuilder
using AiDotNet;
using AiDotNet.Audio;
using AiDotNet.Audio.Features;

// Simulated MFCC features from audio
var audioFeatures = new double[100, 40]; // 100 samples, 40 MFCC features
var labels = new double[100];
var rng = new Random(42);
for (int i = 0; i < 100; i++)
{
    for (int j = 0; j < 40; j++)
        audioFeatures[i, j] = rng.NextDouble();
    labels[i] = i % 3; // 3 classes: speech, music, noise
}

var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new NeuralNetworkBuilder<double>()
        .AddDenseLayer(inputSize: 40, outputSize: 128)
        .AddActivation(ActivationType.ReLU)
        .AddDropout(0.3)
        .AddDenseLayer(inputSize: 128, outputSize: 64)
        .AddActivation(ActivationType.ReLU)
        .AddDenseLayer(inputSize: 64, outputSize: 3)
        .AddActivation(ActivationType.Softmax)
        .Build())
    .ConfigureOptimizer(new Adam<double>(learningRate: 0.001))
    .ConfigureTraining(epochs: 50)
    .BuildAsync(audioFeatures, labels);

Console.WriteLine(""Audio Classification Results:"");
Console.WriteLine($""  Classes: Speech, Music, Noise"");
Console.WriteLine($""  Accuracy: {result.Metrics.Accuracy:P2}"");
"
                },
                new CodeExample
                {
                    Id = "text-to-speech",
                    Name = "Text-to-Speech",
                    Description = "Generate speech audio from text",
                    Difficulty = "Advanced",
                    Tags = ["audio", "tts", "speech-synthesis"],
                    Code = @"// Text-to-Speech with AiModelBuilder
using AiDotNet;
using AiDotNet.Audio.TextToSpeech;

var text = ""Hello, welcome to AiDotNet!"";

var result = await new AiModelBuilder<double, string, double[]>()
    .ConfigureModel(new TtsModel<double>(
        modelType: TtsModelType.VITS,
        sampleRate: 22050))
    .BuildAsync(text);

Console.WriteLine(""Text-to-Speech Generation:"");
Console.WriteLine($""  Input text: '{text}'"");
Console.WriteLine($""  Audio length: {result.AudioSamples.Length / 22050.0:F2} seconds"");
Console.WriteLine($""  Sample rate: 22050 Hz"");

// In practice, save to file:
// await result.SaveAsync(""output.wav"");
"
                },
                new CodeExample
                {
                    Id = "music-generation",
                    Name = "Music Generation",
                    Description = "Generate music using MusicGen model",
                    Difficulty = "Advanced",
                    Tags = ["audio", "music", "generation", "ai"],
                    Code = @"// Music Generation with AiModelBuilder
using AiDotNet;
using AiDotNet.Audio.MusicGen;

var prompt = ""A calm piano melody with soft strings"";

var result = await new AiModelBuilder<double, string, double[]>()
    .ConfigureModel(new MusicGenModel<double>(
        modelSize: MusicGenModelSize.Medium,
        sampleRate: 32000))
    .ConfigureGeneration(durationSeconds: 10)
    .BuildAsync(prompt);

Console.WriteLine(""Music Generation Result:"");
Console.WriteLine($""  Prompt: '{prompt}'"");
Console.WriteLine($""  Duration: {result.AudioSamples.Length / 32000.0:F2} seconds"");
Console.WriteLine($""  Sample rate: 32000 Hz"");
"
                }
            },

            ["Reinforcement Learning"] = new()
            {
                new CodeExample
                {
                    Id = "dqn-agent",
                    Name = "DQN Agent",
                    Description = "Deep Q-Network for discrete action spaces",
                    Difficulty = "Advanced",
                    Tags = ["rl", "dqn", "deep-learning", "q-learning"],
                    Code = @"// DQN Agent with AiModelBuilder
using AiDotNet;
using AiDotNet.ReinforcementLearning;
using AiDotNet.ReinforcementLearning.Agents;

// Define environment interface
var stateSize = 4;
var actionSize = 2;

var result = await new AiModelBuilder<double, double[], int>()
    .ConfigureModel(new DQNAgent<double>(
        stateSize: stateSize,
        actionSize: actionSize,
        hiddenSizes: new[] { 128, 128 },
        learningRate: 0.001,
        gamma: 0.99,
        epsilon: 1.0,
        epsilonDecay: 0.995,
        epsilonMin: 0.01,
        bufferSize: 10000,
        batchSize: 64))
    .BuildAsync();

Console.WriteLine(""DQN Agent Created:"");
Console.WriteLine($""  State size: {stateSize}"");
Console.WriteLine($""  Action size: {actionSize}"");
Console.WriteLine($""  Architecture: Input -> 128 -> 128 -> Output"");

// Training loop example
Console.WriteLine(""\nTraining loop:"");
Console.WriteLine(""  1. Get state from environment"");
Console.WriteLine(""  2. Select action: agent.SelectAction(state)"");
Console.WriteLine(""  3. Execute action, get reward and next_state"");
Console.WriteLine(""  4. Store: agent.Remember(state, action, reward, next_state, done)"");
Console.WriteLine(""  5. Train: agent.Replay()"");
"
                },
                new CodeExample
                {
                    Id = "ppo-agent",
                    Name = "PPO Agent",
                    Description = "Proximal Policy Optimization for continuous control",
                    Difficulty = "Advanced",
                    Tags = ["rl", "ppo", "policy-gradient", "continuous"],
                    Code = @"// PPO Agent with AiModelBuilder
using AiDotNet;
using AiDotNet.ReinforcementLearning;
using AiDotNet.ReinforcementLearning.Agents;

// Continuous action space environment
var stateSize = 8;
var actionSize = 2;

var result = await new AiModelBuilder<double, double[], double[]>()
    .ConfigureModel(new PPOAgent<double>(
        stateSize: stateSize,
        actionSize: actionSize,
        hiddenSizes: new[] { 64, 64 },
        actorLearningRate: 0.0003,
        criticLearningRate: 0.001,
        gamma: 0.99,
        gaeλ: 0.95,
        clipEpsilon: 0.2,
        entropyCoeff: 0.01,
        epochs: 10,
        batchSize: 64))
    .BuildAsync();

Console.WriteLine(""PPO Agent Created:"");
Console.WriteLine($""  State size: {stateSize}"");
Console.WriteLine($""  Action size: {actionSize} (continuous)"");
Console.WriteLine($""  Clip epsilon: 0.2"");
Console.WriteLine($""  GAE λ: 0.95"");

// Get action for a state
var state = new double[] { 0.1, -0.2, 0.3, 0.4, -0.1, 0.2, 0.0, 0.5 };
var action = result.SelectAction(state);
Console.WriteLine($""  Sample action: [{action[0]:F3}, {action[1]:F3}]"");
"
                },
                new CodeExample
                {
                    Id = "sac-agent",
                    Name = "SAC Agent",
                    Description = "Soft Actor-Critic for sample-efficient learning",
                    Difficulty = "Advanced",
                    Tags = ["rl", "sac", "actor-critic", "entropy"],
                    Code = @"// SAC Agent with AiModelBuilder
using AiDotNet;
using AiDotNet.ReinforcementLearning;
using AiDotNet.ReinforcementLearning.Agents;

var stateSize = 11;
var actionSize = 3;

var result = await new AiModelBuilder<double, double[], double[]>()
    .ConfigureModel(new SACAgent<double>(
        stateSize: stateSize,
        actionSize: actionSize,
        hiddenSizes: new[] { 256, 256 },
        learningRate: 0.0003,
        gamma: 0.99,
        tau: 0.005,
        alpha: 0.2,
        bufferSize: 1000000,
        batchSize: 256,
        autoAlpha: true))
    .BuildAsync();

Console.WriteLine(""SAC Agent Created:"");
Console.WriteLine($""  State size: {stateSize}"");
Console.WriteLine($""  Action size: {actionSize}"");
Console.WriteLine($""  Architecture: 256 -> 256"");
Console.WriteLine($""  Auto-tuning entropy: enabled"");
Console.WriteLine($""  Soft update τ: 0.005"");
"
                },
                new CodeExample
                {
                    Id = "multi-agent-rl",
                    Name = "Multi-Agent RL",
                    Description = "Train multiple agents in a shared environment",
                    Difficulty = "Expert",
                    Tags = ["rl", "multi-agent", "marl", "cooperative"],
                    Code = @"// Multi-Agent RL with AiModelBuilder
using AiDotNet;
using AiDotNet.ReinforcementLearning;
using AiDotNet.ReinforcementLearning.MultiAgent;

var numAgents = 3;
var stateSize = 16;
var actionSize = 4;

var result = await new AiModelBuilder<double, double[][], int[]>()
    .ConfigureModel(new MADDPGSystem<double>(
        numAgents: numAgents,
        stateSize: stateSize,
        actionSize: actionSize,
        hiddenSizes: new[] { 128, 128 },
        actorLearningRate: 0.001,
        criticLearningRate: 0.001,
        gamma: 0.95,
        tau: 0.01))
    .BuildAsync();

Console.WriteLine(""Multi-Agent System Created (MADDPG):"");
Console.WriteLine($""  Number of agents: {numAgents}"");
Console.WriteLine($""  State size per agent: {stateSize}"");
Console.WriteLine($""  Action size per agent: {actionSize}"");
Console.WriteLine($""  Centralized training, decentralized execution"");
"
                }
            },

            ["RAG & Embeddings"] = new()
            {
                new CodeExample
                {
                    Id = "vector-store",
                    Name = "Vector Store",
                    Description = "Create and query a vector database for similarity search",
                    Difficulty = "Intermediate",
                    Tags = ["rag", "vector", "similarity", "search"],
                    Code = @"// Vector Store with AiModelBuilder
using AiDotNet;
using AiDotNet.RAG;
using AiDotNet.RAG.VectorStores;

// Documents to index
var documents = new[]
{
    ""Machine learning is a subset of artificial intelligence."",
    ""Neural networks are inspired by biological neurons."",
    ""Deep learning uses multiple layers of neural networks."",
    ""Natural language processing deals with text data."",
    ""Computer vision processes visual information.""
};

var result = await new AiModelBuilder<double, string[], VectorStore<double>>()
    .ConfigureModel(new VectorStoreBuilder<double>()
        .UseEmbeddingModel(new SentenceTransformer<double>(model: ""all-MiniLM-L6-v2""))
        .UseIndex(new HNSWIndex<double>(efConstruction: 200, m: 16)))
    .BuildAsync(documents);

// Search for similar documents
var query = ""What is deep learning?"";
var results = result.Search(query, topK: 3);

Console.WriteLine($""Query: '{query}'"");
Console.WriteLine(""Top 3 similar documents:"");
foreach (var (doc, score) in results)
    Console.WriteLine($""  [{score:F4}] {doc}"");
"
                },
                new CodeExample
                {
                    Id = "rag-pipeline",
                    Name = "RAG Pipeline",
                    Description = "Retrieval-Augmented Generation for Q&A",
                    Difficulty = "Advanced",
                    Tags = ["rag", "retrieval", "generation", "qa"],
                    Code = @"// RAG Pipeline with AiModelBuilder
using AiDotNet;
using AiDotNet.RAG;
using AiDotNet.RAG.Retrievers;

// Knowledge base documents
var knowledgeBase = new[]
{
    ""AiDotNet is the most comprehensive AI/ML framework for .NET."",
    ""AiDotNet supports 100+ neural network architectures."",
    ""AiDotNet includes 106+ classical ML algorithms."",
    ""The AiModelBuilder provides a simple facade pattern."",
    ""AiDotNet supports distributed training with DDP and FSDP.""
};

var result = await new AiModelBuilder<double, string[], RAGPipeline<double>>()
    .ConfigureModel(new RAGPipelineBuilder<double>()
        .UseRetriever(new DenseRetriever<double>(
            embeddingModel: ""all-MiniLM-L6-v2"",
            topK: 3))
        .UseReranker(new CrossEncoderReranker<double>(
            model: ""cross-encoder/ms-marco-MiniLM-L-6-v2""))
        .UseGenerator(new LLMGenerator<double>(
            model: ""gpt-4"",
            maxTokens: 500)))
    .BuildAsync(knowledgeBase);

// Ask a question
var question = ""What is AiModelBuilder?"";
var answer = await result.QueryAsync(question);

Console.WriteLine($""Question: {question}"");
Console.WriteLine($""Answer: {answer.Response}"");
Console.WriteLine($""Sources: {string.Join("", "", answer.Sources)}"");
"
                },
                new CodeExample
                {
                    Id = "document-chunking",
                    Name = "Document Chunking",
                    Description = "Split documents into optimal chunks for retrieval",
                    Difficulty = "Intermediate",
                    Tags = ["rag", "chunking", "preprocessing"],
                    Code = @"// Document Chunking with AiModelBuilder
using AiDotNet;
using AiDotNet.RAG;
using AiDotNet.RAG.Chunkers;

var longDocument = @""
AiDotNet is the most comprehensive AI/ML framework for .NET.
It provides 4,300+ implementations across 60+ feature categories.

The framework includes neural networks, classical ML, computer vision,
audio processing, reinforcement learning, and much more.

Key features include the AiModelBuilder facade pattern which simplifies
model creation and training. The library supports both net8.0 and net471.
"";

var result = await new AiModelBuilder<double, string, string[]>()
    .ConfigureModel(new DocumentChunker<double>(
        chunkSize: 200,
        chunkOverlap: 50,
        strategy: ChunkingStrategy.RecursiveCharacter,
        separators: new[] { ""\n\n"", ""\n"", "". "", "" "" }))
    .BuildAsync(longDocument);

Console.WriteLine($""Original document length: {longDocument.Length} chars"");
Console.WriteLine($""Number of chunks: {result.Chunks.Length}"");
Console.WriteLine(""\nChunks:"");
for (int i = 0; i < result.Chunks.Length; i++)
    Console.WriteLine($""  [{i}] {result.Chunks[i][..Math.Min(50, result.Chunks[i].Length)]}..."");
"
                },
                new CodeExample
                {
                    Id = "semantic-search",
                    Name = "Semantic Search",
                    Description = "Search using semantic meaning rather than keywords",
                    Difficulty = "Intermediate",
                    Tags = ["rag", "semantic", "search", "embeddings"],
                    Code = @"// Semantic Search with AiModelBuilder
using AiDotNet;
using AiDotNet.RAG;
using AiDotNet.NLP.Embeddings;

var corpus = new[]
{
    ""The quick brown fox jumps over the lazy dog"",
    ""A fast auburn canine leaps above the idle hound"",
    ""Machine learning algorithms learn from data"",
    ""Artificial intelligence mimics human cognition"",
    ""The weather today is sunny and warm""
};

var result = await new AiModelBuilder<double, string[], SemanticSearchEngine<double>>()
    .ConfigureModel(new SemanticSearchBuilder<double>()
        .UseEmbeddings(new SentenceTransformer<double>(""all-mpnet-base-v2"")))
    .BuildAsync(corpus);

// Semantic search finds meaning, not just keywords
var query = ""canine jumping"";
var matches = result.Search(query, topK: 2);

Console.WriteLine($""Query: '{query}'"");
Console.WriteLine(""Semantic matches (finds similar meaning):"");
foreach (var (text, score) in matches)
    Console.WriteLine($""  [{score:F4}] {text}"");
"
                }
            },

            ["LoRA & Fine-tuning"] = new()
            {
                new CodeExample
                {
                    Id = "lora-basic",
                    Name = "LoRA Fine-tuning",
                    Description = "Low-Rank Adaptation for efficient model fine-tuning",
                    Difficulty = "Advanced",
                    Tags = ["lora", "fine-tuning", "peft", "llm"],
                    Code = @"// LoRA Fine-tuning with AiModelBuilder
using AiDotNet;
using AiDotNet.LoRA;

// Prepare training data (instruction-response pairs)
var trainingData = new[]
{
    (""Explain quantum computing"", ""Quantum computing uses qubits...""),
    (""What is machine learning?"", ""Machine learning is a subset of AI...""),
    (""Define neural network"", ""A neural network is a computational model..."")
};

// Load base model and apply LoRA
var result = await new AiModelBuilder<double, string, string>()
    .ConfigureModel(new LoRAModelBuilder<double>()
        .UseBaseModel(""llama-7b"")
        .ConfigureLoRA(
            rank: 8,
            alpha: 16,
            dropout: 0.1,
            targetModules: new[] { ""q_proj"", ""v_proj"" }))
    .ConfigureTraining(
        epochs: 3,
        batchSize: 4,
        learningRate: 2e-4)
    .BuildAsync(trainingData);

Console.WriteLine(""LoRA Fine-tuning Complete:"");
Console.WriteLine($""  Base model: llama-7b"");
Console.WriteLine($""  LoRA rank: 8"");
Console.WriteLine($""  Trainable params: {result.TrainableParameters:N0}"");
Console.WriteLine($""  Total params: {result.TotalParameters:N0}"");
Console.WriteLine($""  % trainable: {result.TrainableParameters * 100.0 / result.TotalParameters:F2}%"");
"
                },
                new CodeExample
                {
                    Id = "qlora",
                    Name = "QLoRA (Quantized LoRA)",
                    Description = "4-bit quantized LoRA for memory-efficient fine-tuning",
                    Difficulty = "Advanced",
                    Tags = ["qlora", "quantization", "fine-tuning", "memory-efficient"],
                    Code = @"// QLoRA Fine-tuning with AiModelBuilder
using AiDotNet;
using AiDotNet.LoRA;
using AiDotNet.Quantization;

var result = await new AiModelBuilder<double, string, string>()
    .ConfigureModel(new QLoRAModelBuilder<double>()
        .UseBaseModel(""llama-13b"")
        .ConfigureQuantization(
            bits: 4,
            quantType: QuantizationType.NF4,
            computeDtype: ComputeType.BFloat16)
        .ConfigureLoRA(
            rank: 64,
            alpha: 16,
            dropout: 0.05,
            targetModules: new[] { ""q_proj"", ""k_proj"", ""v_proj"", ""o_proj"" }))
    .ConfigureTraining(
        epochs: 1,
        batchSize: 1,
        gradientAccumulation: 16,
        learningRate: 2e-4)
    .BuildAsync(trainingData);

Console.WriteLine(""QLoRA Fine-tuning Complete:"");
Console.WriteLine($""  Quantization: 4-bit NF4"");
Console.WriteLine($""  Memory usage: ~4GB (vs ~26GB full precision)"");
Console.WriteLine($""  LoRA rank: 64"");
"
                },
                new CodeExample
                {
                    Id = "dora",
                    Name = "DoRA (Weight-Decomposed LoRA)",
                    Description = "Improved LoRA with weight decomposition",
                    Difficulty = "Expert",
                    Tags = ["dora", "fine-tuning", "peft", "advanced"],
                    Code = @"// DoRA Fine-tuning with AiModelBuilder
using AiDotNet;
using AiDotNet.LoRA;

var result = await new AiModelBuilder<double, string, string>()
    .ConfigureModel(new DoRAModelBuilder<double>()
        .UseBaseModel(""mistral-7b"")
        .ConfigureDoRA(
            rank: 16,
            alpha: 32,
            dropout: 0.1,
            targetModules: new[] { ""q_proj"", ""k_proj"", ""v_proj"", ""o_proj"", ""gate_proj"", ""up_proj"", ""down_proj"" },
            decomposeMagnitude: true))
    .ConfigureTraining(
        epochs: 2,
        batchSize: 2,
        learningRate: 1e-4)
    .BuildAsync(trainingData);

Console.WriteLine(""DoRA Fine-tuning Complete:"");
Console.WriteLine($""  Method: Weight-Decomposed Low-Rank Adaptation"");
Console.WriteLine($""  Decomposes weights into magnitude and direction"");
Console.WriteLine($""  Better performance than standard LoRA"");
"
                },
                new CodeExample
                {
                    Id = "adalora",
                    Name = "AdaLoRA (Adaptive LoRA)",
                    Description = "Automatically adjust LoRA rank during training",
                    Difficulty = "Expert",
                    Tags = ["adalora", "adaptive", "fine-tuning", "dynamic"],
                    Code = @"// AdaLoRA Fine-tuning with AiModelBuilder
using AiDotNet;
using AiDotNet.LoRA;

var result = await new AiModelBuilder<double, string, string>()
    .ConfigureModel(new AdaLoRAModelBuilder<double>()
        .UseBaseModel(""llama-7b"")
        .ConfigureAdaLoRA(
            initRank: 12,
            targetRank: 8,
            alpha: 32,
            dropout: 0.1,
            betaStart: 0.85,
            betaEnd: 0.85,
            deltaT: 10,
            rankPattern: RankPattern.Decreasing))
    .ConfigureTraining(
        epochs: 3,
        batchSize: 4,
        learningRate: 1e-4,
        warmupSteps: 100)
    .BuildAsync(trainingData);

Console.WriteLine(""AdaLoRA Fine-tuning Complete:"");
Console.WriteLine($""  Initial rank: 12 -> Target rank: 8"");
Console.WriteLine($""  Adaptive rank allocation per layer"");
Console.WriteLine($""  Final ranks: {string.Join("", "", result.FinalRanks)}"");
"
                }
            },

            ["AutoML"] = new()
            {
                new CodeExample
                {
                    Id = "automl-classification",
                    Name = "AutoML Classification",
                    Description = "Automatically find the best model for classification",
                    Difficulty = "Intermediate",
                    Tags = ["automl", "classification", "model-selection"],
                    Code = @"// AutoML Classification with AiModelBuilder
using AiDotNet;
using AiDotNet.AutoML;

var features = new double[1000, 20];
var labels = new double[1000];
var rng = new Random(42);
for (int i = 0; i < 1000; i++)
{
    for (int j = 0; j < 20; j++)
        features[i, j] = rng.NextDouble();
    labels[i] = features[i, 0] + features[i, 1] > 1 ? 1 : 0;
}

var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureAutoML(new AutoMLConfig()
        .SetTask(MLTask.Classification)
        .SetTimeLimit(TimeSpan.FromMinutes(5))
        .SetMetric(OptimizationMetric.Accuracy)
        .SearchModels(new[]
        {
            typeof(RandomForestClassifier<double>),
            typeof(GradientBoostingClassifier<double>),
            typeof(LogisticRegression<double>),
            typeof(SupportVectorClassifier<double>)
        }))
    .BuildAsync(features, labels);

Console.WriteLine(""AutoML Classification Results:"");
Console.WriteLine($""  Best model: {result.BestModel.GetType().Name}"");
Console.WriteLine($""  Best accuracy: {result.BestScore:P2}"");
Console.WriteLine($""  Models evaluated: {result.ModelsEvaluated}"");
Console.WriteLine($""  Total time: {result.ElapsedTime.TotalSeconds:F1}s"");
"
                },
                new CodeExample
                {
                    Id = "automl-regression",
                    Name = "AutoML Regression",
                    Description = "Automatically find the best model for regression",
                    Difficulty = "Intermediate",
                    Tags = ["automl", "regression", "model-selection"],
                    Code = @"// AutoML Regression with AiModelBuilder
using AiDotNet;
using AiDotNet.AutoML;

var features = new double[500, 10];
var labels = new double[500];
var rng = new Random(42);
for (int i = 0; i < 500; i++)
{
    for (int j = 0; j < 10; j++)
        features[i, j] = rng.NextDouble() * 10;
    labels[i] = features[i, 0] * 2 + features[i, 1] * 3 + rng.NextDouble();
}

var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureAutoML(new AutoMLConfig()
        .SetTask(MLTask.Regression)
        .SetTimeLimit(TimeSpan.FromMinutes(3))
        .SetMetric(OptimizationMetric.R2Score)
        .SetCrossValidationFolds(5)
        .EnableEarlyTermination(patience: 10))
    .BuildAsync(features, labels);

Console.WriteLine(""AutoML Regression Results:"");
Console.WriteLine($""  Best model: {result.BestModel.GetType().Name}"");
Console.WriteLine($""  Best R² score: {result.BestScore:F4}"");
Console.WriteLine($""  Best hyperparameters:"");
foreach (var param in result.BestHyperparameters)
    Console.WriteLine($""    {param.Key}: {param.Value}"");
"
                },
                new CodeExample
                {
                    Id = "neural-architecture-search",
                    Name = "Neural Architecture Search",
                    Description = "Automatically design optimal neural network architectures",
                    Difficulty = "Expert",
                    Tags = ["automl", "nas", "neural-network", "architecture"],
                    Code = @"// Neural Architecture Search with AiModelBuilder
using AiDotNet;
using AiDotNet.AutoML;
using AiDotNet.AutoML.NAS;

var features = new double[1000, 784]; // MNIST-like data
var labels = new double[1000];
var rng = new Random(42);
for (int i = 0; i < 1000; i++)
{
    for (int j = 0; j < 784; j++)
        features[i, j] = rng.NextDouble();
    labels[i] = i % 10;
}

var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureAutoML(new NeuralArchitectureSearch<double>()
        .SetSearchSpace(new NASSearchSpace()
            .AddLayerChoice(""conv"", new[] { 16, 32, 64 })
            .AddLayerChoice(""dense"", new[] { 128, 256, 512 })
            .AddActivationChoice(new[] { ActivationType.ReLU, ActivationType.GELU })
            .AddDropoutRange(0.1, 0.5))
        .SetSearchStrategy(SearchStrategy.ENAS)
        .SetTimeLimit(TimeSpan.FromMinutes(30))
        .SetMaxTrials(100))
    .BuildAsync(features, labels);

Console.WriteLine(""Neural Architecture Search Results:"");
Console.WriteLine($""  Best architecture found:"");
Console.WriteLine($""    {result.BestArchitecture}"");
Console.WriteLine($""  Validation accuracy: {result.BestScore:P2}"");
Console.WriteLine($""  Parameters: {result.BestModel.ParameterCount:N0}"");
"
                },
                new CodeExample
                {
                    Id = "bayesian-optimization",
                    Name = "Bayesian Hyperparameter Optimization",
                    Description = "Smart hyperparameter tuning using Bayesian optimization",
                    Difficulty = "Advanced",
                    Tags = ["automl", "bayesian", "hyperparameter", "optimization"],
                    Code = @"// Bayesian Optimization with AiModelBuilder
using AiDotNet;
using AiDotNet.AutoML;

var features = new double[800, 15];
var labels = new double[800];
var rng = new Random(42);
for (int i = 0; i < 800; i++)
{
    for (int j = 0; j < 15; j++)
        features[i, j] = rng.NextDouble();
    labels[i] = i % 3;
}

var result = await new AiModelBuilder<double, double[], double>()
    .ConfigureModel(new GradientBoostingClassifier<double>())
    .ConfigureHyperparameterOptimization(new BayesianOptimization<double>()
        .AddIntParameter(""n_estimators"", 50, 500)
        .AddFloatParameter(""learning_rate"", 0.01, 0.3, log: true)
        .AddIntParameter(""max_depth"", 3, 15)
        .AddFloatParameter(""subsample"", 0.5, 1.0)
        .SetAcquisitionFunction(AcquisitionFunction.ExpectedImprovement)
        .SetNumInitialPoints(10)
        .SetMaxIterations(50))
    .BuildAsync(features, labels);

Console.WriteLine(""Bayesian Optimization Results:"");
Console.WriteLine($""  Best parameters:"");
Console.WriteLine($""    n_estimators: {result.BestParameters[""n_estimators""]}"");
Console.WriteLine($""    learning_rate: {result.BestParameters[""learning_rate""]}"");
Console.WriteLine($""    max_depth: {result.BestParameters[""max_depth""]}"");
Console.WriteLine($""  Best accuracy: {result.BestScore:P2}"");
Console.WriteLine($""  Iterations: {result.Iterations}"");
"
                }
            }
        };
    }
}

/// <summary>
/// Represents a code example in the playground.
/// </summary>
public class CodeExample
{
    public string Id { get; set; } = "";
    public string Name { get; set; } = "";
    public string Description { get; set; } = "";
    public string Difficulty { get; set; } = "Beginner";
    public string[] Tags { get; set; } = [];
    public string Code { get; set; } = "";
}
