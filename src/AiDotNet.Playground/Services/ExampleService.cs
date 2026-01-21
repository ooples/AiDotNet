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
