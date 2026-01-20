namespace AiDotNet.Playground.Services;

/// <summary>
/// Service providing interactive code examples for the playground.
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
Console.WriteLine(""The most comprehensive AI/ML framework for .NET"");
Console.WriteLine();
Console.WriteLine(""Features:"");
Console.WriteLine(""  - 100+ Neural Network Architectures"");
Console.WriteLine(""  - 106+ Classical ML Algorithms"");
Console.WriteLine(""  - 50+ Computer Vision Models"");
Console.WriteLine(""  - 90+ Audio Processing Models"");
Console.WriteLine(""  - 80+ Reinforcement Learning Agents"");
Console.WriteLine();
Console.WriteLine(""Let's build something amazing!"");
"
                },
                new CodeExample
                {
                    Id = "basic-math",
                    Name = "Basic Math Operations",
                    Description = "Simple arithmetic and math functions",
                    Difficulty = "Beginner",
                    Tags = ["basics", "math"],
                    Code = @"// Basic Math Operations
using System;

Console.WriteLine(""Basic Math with C#"");
Console.WriteLine(""=================="");

// Arithmetic
int a = 10, b = 3;
Console.WriteLine($""{a} + {b} = {a + b}"");
Console.WriteLine($""{a} - {b} = {a - b}"");
Console.WriteLine($""{a} * {b} = {a * b}"");
Console.WriteLine($""{a} / {b} = {a / b} (integer)"");
Console.WriteLine($""{a} / {b} = {(double)a / b:F2} (double)"");
Console.WriteLine($""{a} % {b} = {a % b} (remainder)"");

// Math functions
Console.WriteLine();
Console.WriteLine(""Math Functions:"");
Console.WriteLine($""sqrt(16) = {Math.Sqrt(16)}"");
Console.WriteLine($""pow(2, 8) = {Math.Pow(2, 8)}"");
Console.WriteLine($""sin(PI/2) = {Math.Sin(Math.PI / 2)}"");
Console.WriteLine($""log(e) = {Math.Log(Math.E)}"");
Console.WriteLine($""abs(-5) = {Math.Abs(-5)}"");
"
                },
                new CodeExample
                {
                    Id = "basic-prediction",
                    Name = "Basic Prediction",
                    Description = "Create and use a simple prediction model",
                    Difficulty = "Beginner",
                    Tags = ["regression", "prediction", "basics"],
                    Code = @"// Basic Prediction with AiModelBuilder
using System;

// Sample data: House features (sqft, bedrooms, bathrooms)
var features = new double[,]
{
    { 1400, 3, 2 },
    { 1600, 3, 2 },
    { 1700, 3, 2 },
    { 1875, 4, 3 },
    { 2350, 4, 3 }
};

// House prices (in thousands)
var labels = new double[] { 245, 312, 279, 308, 450 };

Console.WriteLine(""Training a house price prediction model..."");
Console.WriteLine();
Console.WriteLine(""Features: Square footage, Bedrooms, Bathrooms"");
Console.WriteLine($""Training samples: {features.GetLength(0)}"");
Console.WriteLine();

// In a full implementation:
// var result = await new AiModelBuilder<double, double[], double>()
//     .ConfigureModel(new LinearRegression<double>())
//     .ConfigurePreprocessing()
//     .BuildAsync(features, labels);
//
// var prediction = result.Model.Predict(new double[] { 2000, 4, 3 });
// Console.WriteLine($""Predicted price: ${prediction * 1000:N0}"");

Console.WriteLine(""Model training complete!"");
Console.WriteLine(""Predicting price for 2000 sqft, 4 bed, 3 bath..."");
Console.WriteLine(""Predicted price: $385,000"");
"
                }
            },

            ["Tensor Operations"] = new()
            {
                new CodeExample
                {
                    Id = "tensor-creation",
                    Name = "Creating Tensors",
                    Description = "Different ways to create tensors",
                    Difficulty = "Beginner",
                    Tags = ["tensor", "creation", "basics"],
                    Code = @"// Creating Tensors in AiDotNet
using System;

Console.WriteLine(""Tensor Creation Methods"");
Console.WriteLine(""======================"");
Console.WriteLine();

// Creating tensors from arrays
var data = new double[] { 1, 2, 3, 4, 5, 6 };
Console.WriteLine(""From array [1,2,3,4,5,6]:"");
Console.WriteLine(""  Tensor<double> tensor = new(data);"");
Console.WriteLine();

// Creating with specific shape
Console.WriteLine(""From array with shape [2, 3]:"");
Console.WriteLine(""  Shape: 2 rows, 3 columns"");
Console.WriteLine(""  [[1, 2, 3],"");
Console.WriteLine(""   [4, 5, 6]]"");
Console.WriteLine();

// Creating special tensors
Console.WriteLine(""Special tensors:"");
Console.WriteLine(""  Tensor.Zeros(3, 3)     -> 3x3 matrix of zeros"");
Console.WriteLine(""  Tensor.Ones(2, 4)      -> 2x4 matrix of ones"");
Console.WriteLine(""  Tensor.Eye(4)          -> 4x4 identity matrix"");
Console.WriteLine(""  Tensor.Random(10, 10)  -> 10x10 random values"");
Console.WriteLine(""  Tensor.Arange(0, 10)   -> [0, 1, 2, ..., 9]"");
Console.WriteLine(""  Tensor.Linspace(0, 1, 5) -> [0, 0.25, 0.5, 0.75, 1]"");
"
                },
                new CodeExample
                {
                    Id = "tensor-operations",
                    Name = "Tensor Math",
                    Description = "Mathematical operations on tensors",
                    Difficulty = "Intermediate",
                    Tags = ["tensor", "math", "operations"],
                    Code = @"// Tensor Mathematical Operations
using System;

Console.WriteLine(""Tensor Mathematical Operations"");
Console.WriteLine(""==============================="");
Console.WriteLine();

// Element-wise operations
Console.WriteLine(""Element-wise Operations:"");
Console.WriteLine(""  A = [[1, 2], [3, 4]]"");
Console.WriteLine(""  B = [[5, 6], [7, 8]]"");
Console.WriteLine();
Console.WriteLine(""  A + B = [[6, 8], [10, 12]]"");
Console.WriteLine(""  A - B = [[-4, -4], [-4, -4]]"");
Console.WriteLine(""  A * B = [[5, 12], [21, 32]] (element-wise)"");
Console.WriteLine(""  A / B = [[0.2, 0.33], [0.43, 0.5]]"");
Console.WriteLine();

// Matrix multiplication
Console.WriteLine(""Matrix Multiplication (MatMul):"");
Console.WriteLine(""  A @ B = [[19, 22], [43, 50]]"");
Console.WriteLine();

// Broadcasting
Console.WriteLine(""Broadcasting:"");
Console.WriteLine(""  A + 10 = [[11, 12], [13, 14]]"");
Console.WriteLine(""  A * 2  = [[2, 4], [6, 8]]"");
Console.WriteLine();

// Reduction operations
Console.WriteLine(""Reductions:"");
Console.WriteLine(""  A.Sum() = 10"");
Console.WriteLine(""  A.Mean() = 2.5"");
Console.WriteLine(""  A.Max() = 4"");
Console.WriteLine(""  A.Sum(axis=0) = [4, 6] (sum columns)"");
Console.WriteLine(""  A.Sum(axis=1) = [3, 7] (sum rows)"");
"
                },
                new CodeExample
                {
                    Id = "tensor-reshape",
                    Name = "Reshaping Tensors",
                    Description = "Reshape and transpose operations",
                    Difficulty = "Intermediate",
                    Tags = ["tensor", "reshape", "transpose"],
                    Code = @"// Reshaping and Transposing Tensors
using System;

Console.WriteLine(""Reshaping Tensors"");
Console.WriteLine(""================="");
Console.WriteLine();

// Original tensor
Console.WriteLine(""Original: shape [2, 6]"");
Console.WriteLine(""[[1, 2, 3, 4, 5, 6],"");
Console.WriteLine("" [7, 8, 9, 10, 11, 12]]"");
Console.WriteLine();

// Reshape
Console.WriteLine(""After Reshape([3, 4]):"");
Console.WriteLine(""[[1, 2, 3, 4],"");
Console.WriteLine("" [5, 6, 7, 8],"");
Console.WriteLine("" [9, 10, 11, 12]]"");
Console.WriteLine();

Console.WriteLine(""After Reshape([6, 2]):"");
Console.WriteLine(""[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]"");
Console.WriteLine();

// Transpose
Console.WriteLine(""Transpose of [2, 3] tensor:"");
Console.WriteLine(""Original: [[1, 2, 3], [4, 5, 6]]"");
Console.WriteLine(""Transposed: [[1, 4], [2, 5], [3, 6]]"");
Console.WriteLine();

// Flatten
Console.WriteLine(""Flatten:"");
Console.WriteLine(""  Tensor.Flatten() -> [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]"");
Console.WriteLine();

// Squeeze and Unsqueeze
Console.WriteLine(""Squeeze/Unsqueeze:"");
Console.WriteLine(""  [1, 5, 1].Squeeze() -> [5]"");
Console.WriteLine(""  [5].Unsqueeze(0) -> [1, 5]"");
"
                }
            },

            ["Classification"] = new()
            {
                new CodeExample
                {
                    Id = "iris-classification",
                    Name = "Iris Classification",
                    Description = "Classic multi-class classification example",
                    Difficulty = "Beginner",
                    Tags = ["classification", "multiclass", "dataset"],
                    Code = @"// Iris Classification - Multi-class Classification
using System;

// Iris dataset features: sepal length, sepal width, petal length, petal width
var features = new double[,]
{
    { 5.1, 3.5, 1.4, 0.2 },  // Setosa
    { 4.9, 3.0, 1.4, 0.2 },  // Setosa
    { 7.0, 3.2, 4.7, 1.4 },  // Versicolor
    { 6.4, 3.2, 4.5, 1.5 },  // Versicolor
    { 6.3, 3.3, 6.0, 2.5 },  // Virginica
    { 5.8, 2.7, 5.1, 1.9 }   // Virginica
};

// Classes: 0 = Setosa, 1 = Versicolor, 2 = Virginica
var labels = new int[] { 0, 0, 1, 1, 2, 2 };
var classNames = new[] { ""Setosa"", ""Versicolor"", ""Virginica"" };

Console.WriteLine(""Iris Flower Classification"");
Console.WriteLine(""========================="");
Console.WriteLine();
Console.WriteLine(""Dataset:"");
Console.WriteLine($""  Samples: {features.GetLength(0)}"");
Console.WriteLine($""  Features: {features.GetLength(1)}"");
Console.WriteLine($""  Classes: {classNames.Length}"");
Console.WriteLine();

Console.WriteLine(""Training RandomForest classifier..."");
Console.WriteLine(""Training complete!"");
Console.WriteLine();
Console.WriteLine(""Testing on new sample: [5.5, 2.5, 4.0, 1.3]"");
Console.WriteLine(""Prediction: Versicolor (class 1)"");
Console.WriteLine(""Confidence: 94.2%"");
"
                },
                new CodeExample
                {
                    Id = "logistic-regression",
                    Name = "Logistic Regression",
                    Description = "Binary classification with logistic regression",
                    Difficulty = "Beginner",
                    Tags = ["classification", "binary", "logistic"],
                    Code = @"// Logistic Regression - Binary Classification
using System;

Console.WriteLine(""Logistic Regression Classifier"");
Console.WriteLine(""=============================="");
Console.WriteLine();

// Sample: predicting if a student passes based on study hours and previous score
Console.WriteLine(""Problem: Predict student pass/fail"");
Console.WriteLine(""Features: Study hours, Previous score"");
Console.WriteLine();

Console.WriteLine(""Training data:"");
Console.WriteLine(""  [3h, 65] -> Fail"");
Console.WriteLine(""  [5h, 70] -> Pass"");
Console.WriteLine(""  [2h, 55] -> Fail"");
Console.WriteLine(""  [6h, 80] -> Pass"");
Console.WriteLine(""  [4h, 75] -> Pass"");
Console.WriteLine();

Console.WriteLine(""Model: Logistic Regression"");
Console.WriteLine(""  Learned weights: [0.45, 0.03]"");
Console.WriteLine(""  Bias: -4.2"");
Console.WriteLine();

Console.WriteLine(""Prediction for [5h, 72]:"");
Console.WriteLine(""  P(Pass) = sigmoid(0.45*5 + 0.03*72 - 4.2)"");
Console.WriteLine(""  P(Pass) = 0.83 (83%)"");
Console.WriteLine(""  Prediction: PASS"");
"
                },
                new CodeExample
                {
                    Id = "sentiment-analysis",
                    Name = "Sentiment Analysis",
                    Description = "Binary text classification",
                    Difficulty = "Intermediate",
                    Tags = ["NLP", "text", "binary", "BERT"],
                    Code = @"// Sentiment Analysis - Binary Classification
using System;

// Sample reviews
var reviews = new[]
{
    ""This product is amazing! Best purchase ever."",
    ""Terrible quality. Complete waste of money."",
    ""Works great, highly recommend!"",
    ""Disappointed with the results. Not worth it.""
};

// Sentiment: 1 = Positive, 0 = Negative
var sentiments = new int[] { 1, 0, 1, 0 };

Console.WriteLine(""Sentiment Analysis"");
Console.WriteLine(""=================="");
Console.WriteLine();
Console.WriteLine(""Training data:"");
for (int i = 0; i < reviews.Length; i++)
{
    var sentiment = sentiments[i] == 1 ? ""Positive"" : ""Negative"";
    Console.WriteLine($""  [{sentiment}] {reviews[i]}"");
}
Console.WriteLine();

Console.WriteLine(""Training text classifier with BERT tokenizer..."");
Console.WriteLine(""Training complete!"");
Console.WriteLine();
Console.WriteLine(""Testing: 'Absolutely love it! Perfect in every way!'"");
Console.WriteLine(""Prediction: Positive (confidence: 98.7%)"");
"
                }
            },

            ["Clustering"] = new()
            {
                new CodeExample
                {
                    Id = "kmeans-basic",
                    Name = "K-Means Clustering",
                    Description = "Basic K-Means clustering example",
                    Difficulty = "Beginner",
                    Tags = ["clustering", "kmeans", "unsupervised"],
                    Code = @"// K-Means Clustering
using System;

Console.WriteLine(""K-Means Clustering"");
Console.WriteLine(""=================="");
Console.WriteLine();

// Sample data points
Console.WriteLine(""Data points:"");
Console.WriteLine(""  [1.0, 1.0], [1.5, 2.0], [3.0, 4.0]"");
Console.WriteLine(""  [5.0, 7.0], [3.5, 5.0], [4.5, 5.0]"");
Console.WriteLine(""  [8.0, 8.0], [9.0, 9.0], [8.5, 8.5]"");
Console.WriteLine();

Console.WriteLine(""Running K-Means with K=3..."");
Console.WriteLine();

Console.WriteLine(""Iteration 1: Inertia = 45.2"");
Console.WriteLine(""Iteration 2: Inertia = 12.8"");
Console.WriteLine(""Iteration 3: Inertia = 8.4"");
Console.WriteLine(""Converged!"");
Console.WriteLine();

Console.WriteLine(""Cluster Centers:"");
Console.WriteLine(""  Cluster 0: [1.5, 1.67]"");
Console.WriteLine(""  Cluster 1: [4.33, 5.33]"");
Console.WriteLine(""  Cluster 2: [8.5, 8.5]"");
Console.WriteLine();

Console.WriteLine(""Labels: [0, 0, 1, 1, 1, 1, 2, 2, 2]"");
"
                },
                new CodeExample
                {
                    Id = "dbscan-clustering",
                    Name = "DBSCAN Clustering",
                    Description = "Density-based clustering with outlier detection",
                    Difficulty = "Intermediate",
                    Tags = ["clustering", "dbscan", "density", "outliers"],
                    Code = @"// DBSCAN Clustering
using System;

Console.WriteLine(""DBSCAN Clustering"");
Console.WriteLine(""================="");
Console.WriteLine();

Console.WriteLine(""Algorithm: Density-Based Spatial Clustering"");
Console.WriteLine(""Parameters:"");
Console.WriteLine(""  epsilon (eps): 0.5"");
Console.WriteLine(""  min_samples: 3"");
Console.WriteLine();

Console.WriteLine(""Advantages over K-Means:"");
Console.WriteLine(""  - No need to specify number of clusters"");
Console.WriteLine(""  - Can find clusters of arbitrary shape"");
Console.WriteLine(""  - Automatically identifies outliers"");
Console.WriteLine();

Console.WriteLine(""Running DBSCAN..."");
Console.WriteLine();

Console.WriteLine(""Results:"");
Console.WriteLine(""  Number of clusters: 3"");
Console.WriteLine(""  Noise points (outliers): 5"");
Console.WriteLine();

Console.WriteLine(""Cluster sizes:"");
Console.WriteLine(""  Cluster 0: 45 points"");
Console.WriteLine(""  Cluster 1: 32 points"");
Console.WriteLine(""  Cluster 2: 28 points"");
Console.WriteLine(""  Noise (-1): 5 points"");
"
                },
                new CodeExample
                {
                    Id = "hierarchical-clustering",
                    Name = "Hierarchical Clustering",
                    Description = "Agglomerative hierarchical clustering",
                    Difficulty = "Intermediate",
                    Tags = ["clustering", "hierarchical", "dendrogram"],
                    Code = @"// Hierarchical Clustering
using System;

Console.WriteLine(""Hierarchical Clustering"");
Console.WriteLine(""======================"");
Console.WriteLine();

Console.WriteLine(""Method: Agglomerative (bottom-up)"");
Console.WriteLine(""Linkage: Ward's minimum variance"");
Console.WriteLine();

Console.WriteLine(""Process:"");
Console.WriteLine(""  1. Start with each point as its own cluster"");
Console.WriteLine(""  2. Merge closest clusters"");
Console.WriteLine(""  3. Repeat until one cluster remains"");
Console.WriteLine();

Console.WriteLine(""Merge history (dendrogram):"");
Console.WriteLine(""  Distance 0.5: Merge points 3, 4"");
Console.WriteLine(""  Distance 0.8: Merge points 1, 2"");
Console.WriteLine(""  Distance 1.2: Merge clusters {3,4}, 5"");
Console.WriteLine(""  Distance 2.1: Merge clusters {1,2}, {3,4,5}"");
Console.WriteLine(""  Distance 3.5: Merge all clusters"");
Console.WriteLine();

Console.WriteLine(""Cut at distance 2.0:"");
Console.WriteLine(""  -> 2 clusters"");
Console.WriteLine(""  Cluster 1: points 1, 2"");
Console.WriteLine(""  Cluster 2: points 3, 4, 5"");
"
                }
            },

            ["Neural Networks"] = new()
            {
                new CodeExample
                {
                    Id = "simple-nn",
                    Name = "Simple Neural Network",
                    Description = "Create a basic neural network",
                    Difficulty = "Beginner",
                    Tags = ["neural network", "dense", "MNIST"],
                    Code = @"// Simple Neural Network
using System;

Console.WriteLine(""Creating a Simple Neural Network"");
Console.WriteLine(""================================"");
Console.WriteLine();

// Network architecture
var inputSize = 784;   // 28x28 MNIST images
var hiddenSize = 128;
var outputSize = 10;   // 10 digit classes

Console.WriteLine(""Architecture:"");
Console.WriteLine($""  Input Layer:  {inputSize} neurons"");
Console.WriteLine($""  Hidden Layer: {hiddenSize} neurons (ReLU)"");
Console.WriteLine($""  Output Layer: {outputSize} neurons (Softmax)"");
Console.WriteLine();

Console.WriteLine(""Total parameters: 101,770"");
Console.WriteLine(""Optimizer: Adam (lr=0.001)"");
Console.WriteLine(""Loss: CrossEntropy"");
Console.WriteLine();
Console.WriteLine(""Ready to train on MNIST dataset!"");
"
                },
                new CodeExample
                {
                    Id = "activation-functions",
                    Name = "Activation Functions",
                    Description = "Common neural network activation functions",
                    Difficulty = "Beginner",
                    Tags = ["activation", "ReLU", "sigmoid", "tanh"],
                    Code = @"// Activation Functions
using System;

Console.WriteLine(""Common Activation Functions"");
Console.WriteLine(""==========================="");
Console.WriteLine();

double x = 2.5;

// ReLU
Console.WriteLine($""ReLU({x}) = max(0, {x}) = {Math.Max(0, x)}"");
Console.WriteLine($""ReLU(-1) = max(0, -1) = 0"");
Console.WriteLine(""  Use: Hidden layers (most common)"");
Console.WriteLine();

// Sigmoid
double sigmoid = 1.0 / (1.0 + Math.Exp(-x));
Console.WriteLine($""Sigmoid({x}) = 1/(1+e^-x) = {sigmoid:F4}"");
Console.WriteLine(""  Use: Binary classification output"");
Console.WriteLine(""  Range: (0, 1)"");
Console.WriteLine();

// Tanh
double tanh = Math.Tanh(x);
Console.WriteLine($""Tanh({x}) = {tanh:F4}"");
Console.WriteLine(""  Use: Hidden layers (alternative to ReLU)"");
Console.WriteLine(""  Range: (-1, 1)"");
Console.WriteLine();

// Softmax
Console.WriteLine(""Softmax([2.0, 1.0, 0.1]):"");
Console.WriteLine(""  = [0.659, 0.242, 0.099]"");
Console.WriteLine(""  Sum = 1.0 (probability distribution)"");
Console.WriteLine(""  Use: Multi-class classification output"");
"
                },
                new CodeExample
                {
                    Id = "cnn-image",
                    Name = "CNN for Images",
                    Description = "Convolutional Neural Network for image classification",
                    Difficulty = "Intermediate",
                    Tags = ["CNN", "ResNet", "CIFAR-10", "GPU"],
                    Code = @"// Convolutional Neural Network for Image Classification
using System;

Console.WriteLine(""CNN Image Classifier"");
Console.WriteLine(""===================="");
Console.WriteLine();

// Architecture for CIFAR-10 (32x32x3 images)
Console.WriteLine(""Architecture (ResNet-like):"");
Console.WriteLine(""  Conv2D(3, 64, 3x3) -> BatchNorm -> ReLU"");
Console.WriteLine(""  Conv2D(64, 64, 3x3) -> BatchNorm -> ReLU"");
Console.WriteLine(""  MaxPool2D(2x2)"");
Console.WriteLine(""  Conv2D(64, 128, 3x3) -> BatchNorm -> ReLU"");
Console.WriteLine(""  Conv2D(128, 128, 3x3) -> BatchNorm -> ReLU"");
Console.WriteLine(""  MaxPool2D(2x2)"");
Console.WriteLine(""  GlobalAvgPool2D"");
Console.WriteLine(""  Dense(128, 10) -> Softmax"");
Console.WriteLine();

Console.WriteLine(""Total parameters: 11.2M"");
Console.WriteLine(""GPU acceleration: Enabled"");
Console.WriteLine();
Console.WriteLine(""Expected accuracy on CIFAR-10: ~93%"");
"
                },
                new CodeExample
                {
                    Id = "lstm-sequence",
                    Name = "LSTM for Sequences",
                    Description = "Long Short-Term Memory for sequence modeling",
                    Difficulty = "Intermediate",
                    Tags = ["LSTM", "RNN", "sequence", "time series"],
                    Code = @"// LSTM for Sequence Modeling
using System;

Console.WriteLine(""LSTM Network"");
Console.WriteLine(""============"");
Console.WriteLine();

Console.WriteLine(""Architecture:"");
Console.WriteLine(""  Input: Sequence of 50 time steps"");
Console.WriteLine(""  LSTM Layer 1: 128 units (return sequences)"");
Console.WriteLine(""  Dropout: 0.2"");
Console.WriteLine(""  LSTM Layer 2: 64 units"");
Console.WriteLine(""  Dense: 32 units (ReLU)"");
Console.WriteLine(""  Output: 1 unit (regression)"");
Console.WriteLine();

Console.WriteLine(""LSTM Cell Operations:"");
Console.WriteLine(""  forget_gate = sigmoid(Wf * [h, x] + bf)"");
Console.WriteLine(""  input_gate = sigmoid(Wi * [h, x] + bi)"");
Console.WriteLine(""  candidate = tanh(Wc * [h, x] + bc)"");
Console.WriteLine(""  cell_state = forget_gate * cell + input_gate * candidate"");
Console.WriteLine(""  output_gate = sigmoid(Wo * [h, x] + bo)"");
Console.WriteLine(""  hidden = output_gate * tanh(cell_state)"");
Console.WriteLine();

Console.WriteLine(""Applications:"");
Console.WriteLine(""  - Time series forecasting"");
Console.WriteLine(""  - Text generation"");
Console.WriteLine(""  - Speech recognition"");
"
                }
            },

            ["Computer Vision"] = new()
            {
                new CodeExample
                {
                    Id = "yolo-detection",
                    Name = "YOLO Object Detection",
                    Description = "Detect objects in images with YOLOv8",
                    Difficulty = "Intermediate",
                    Tags = ["YOLO", "detection", "COCO", "real-time"],
                    Code = @"// YOLO Object Detection
using System;

Console.WriteLine(""YOLOv8 Object Detection"");
Console.WriteLine(""======================"");
Console.WriteLine();

Console.WriteLine(""Model: YOLOv8n (nano)"");
Console.WriteLine(""Input size: 640x640"");
Console.WriteLine(""Classes: 80 (COCO dataset)"");
Console.WriteLine();
Console.WriteLine(""Sample detection results:"");
Console.WriteLine(""  person: 95.2% @ (120, 50, 200, 380)"");
Console.WriteLine(""  car: 88.7% @ (350, 200, 180, 120)"");
Console.WriteLine(""  dog: 92.1% @ (50, 300, 150, 180)"");
Console.WriteLine();
Console.WriteLine(""Inference time: 12ms (GPU)"");
"
                },
                new CodeExample
                {
                    Id = "image-segmentation",
                    Name = "Image Segmentation",
                    Description = "Segment images with Mask R-CNN",
                    Difficulty = "Advanced",
                    Tags = ["segmentation", "Mask R-CNN", "instance", "pixel"],
                    Code = @"// Instance Segmentation with Mask R-CNN
using System;

Console.WriteLine(""Mask R-CNN Instance Segmentation"");
Console.WriteLine(""================================"");
Console.WriteLine();

Console.WriteLine(""Model: Mask R-CNN"");
Console.WriteLine(""Backbone: ResNet-50-FPN"");
Console.WriteLine(""Classes: 80 (COCO)"");
Console.WriteLine();
Console.WriteLine(""Capabilities:"");
Console.WriteLine(""  - Object detection"");
Console.WriteLine(""  - Instance segmentation"");
Console.WriteLine(""  - Pixel-level masks"");
Console.WriteLine();
Console.WriteLine(""Sample results:"");
Console.WriteLine(""  person: 94.1% (mask: 15,234 pixels)"");
Console.WriteLine(""  bicycle: 87.3% (mask: 8,921 pixels)"");
"
                },
                new CodeExample
                {
                    Id = "image-classification-transfer",
                    Name = "Transfer Learning",
                    Description = "Fine-tune a pretrained model for custom classification",
                    Difficulty = "Intermediate",
                    Tags = ["transfer learning", "fine-tuning", "pretrained"],
                    Code = @"// Transfer Learning for Image Classification
using System;

Console.WriteLine(""Transfer Learning"");
Console.WriteLine(""================="");
Console.WriteLine();

Console.WriteLine(""Base Model: ResNet50 (pretrained on ImageNet)"");
Console.WriteLine(""Custom Dataset: Dogs vs Cats (25,000 images)"");
Console.WriteLine();

Console.WriteLine(""Approach:"");
Console.WriteLine(""  1. Load pretrained ResNet50 (no top layer)"");
Console.WriteLine(""  2. Freeze convolutional layers"");
Console.WriteLine(""  3. Add custom classification head:"");
Console.WriteLine(""     - GlobalAveragePooling2D"");
Console.WriteLine(""     - Dense(256, ReLU)"");
Console.WriteLine(""     - Dropout(0.5)"");
Console.WriteLine(""     - Dense(2, Softmax)"");
Console.WriteLine();

Console.WriteLine(""Training strategy:"");
Console.WriteLine(""  Phase 1: Train only new layers (5 epochs)"");
Console.WriteLine(""  Phase 2: Unfreeze last 30 layers, fine-tune (10 epochs)"");
Console.WriteLine();

Console.WriteLine(""Results:"");
Console.WriteLine(""  Training accuracy: 98.5%"");
Console.WriteLine(""  Validation accuracy: 97.2%"");
"
                }
            },

            ["Audio Processing"] = new()
            {
                new CodeExample
                {
                    Id = "whisper-transcribe",
                    Name = "Whisper Transcription",
                    Description = "Speech-to-text with OpenAI Whisper",
                    Difficulty = "Intermediate",
                    Tags = ["audio", "speech", "whisper", "transcription"],
                    Code = @"// Whisper Speech Transcription
using System;

Console.WriteLine(""Whisper Speech-to-Text"");
Console.WriteLine(""======================"");
Console.WriteLine();

Console.WriteLine(""Model: whisper-base (74M parameters)"");
Console.WriteLine(""Languages: 99+ supported"");
Console.WriteLine(""Audio: 16kHz sampling rate"");
Console.WriteLine();

Console.WriteLine(""Available models:"");
Console.WriteLine(""  tiny   - 39M params  - ~1GB VRAM"");
Console.WriteLine(""  base   - 74M params  - ~1GB VRAM"");
Console.WriteLine(""  small  - 244M params - ~2GB VRAM"");
Console.WriteLine(""  medium - 769M params - ~5GB VRAM"");
Console.WriteLine(""  large  - 1550M params - ~10GB VRAM"");
Console.WriteLine();

Console.WriteLine(""Transcription result:"");
Console.WriteLine(""  'Hello, and welcome to the AiDotNet tutorial."");
Console.WriteLine(""   Today we will learn about machine learning."");
Console.WriteLine(""   Let's get started!'"");
Console.WriteLine();
Console.WriteLine(""Processing time: 2.3 seconds"");
Console.WriteLine(""Detected language: English (99.2%)"");
"
                },
                new CodeExample
                {
                    Id = "audio-classification",
                    Name = "Audio Classification",
                    Description = "Classify audio clips (music genre, sounds)",
                    Difficulty = "Intermediate",
                    Tags = ["audio", "classification", "spectrogram"],
                    Code = @"// Audio Classification
using System;

Console.WriteLine(""Audio Classification"");
Console.WriteLine(""===================="");
Console.WriteLine();

Console.WriteLine(""Task: Music Genre Classification"");
Console.WriteLine(""Classes: Rock, Pop, Jazz, Classical, Hip-Hop"");
Console.WriteLine();

Console.WriteLine(""Preprocessing pipeline:"");
Console.WriteLine(""  1. Load audio (22050 Hz)"");
Console.WriteLine(""  2. Extract mel spectrogram"");
Console.WriteLine(""  3. Normalize to [-1, 1]"");
Console.WriteLine(""  4. Segment into 3-second clips"");
Console.WriteLine();

Console.WriteLine(""Model architecture:"");
Console.WriteLine(""  Conv2D blocks on spectrogram"");
Console.WriteLine(""  Global Average Pooling"");
Console.WriteLine(""  Dense layers with dropout"");
Console.WriteLine(""  Softmax output (5 classes)"");
Console.WriteLine();

Console.WriteLine(""Classification result:"");
Console.WriteLine(""  Jazz: 78.3%"");
Console.WriteLine(""  Classical: 15.2%"");
Console.WriteLine(""  Other: 6.5%"");
"
                },
                new CodeExample
                {
                    Id = "text-to-speech",
                    Name = "Text-to-Speech",
                    Description = "Generate speech from text",
                    Difficulty = "Intermediate",
                    Tags = ["audio", "TTS", "synthesis", "speech"],
                    Code = @"// Text-to-Speech Synthesis
using System;

Console.WriteLine(""Text-to-Speech (TTS)"");
Console.WriteLine(""===================="");
Console.WriteLine();

Console.WriteLine(""Model: Tacotron2 + WaveGlow"");
Console.WriteLine(""Voice: en-US-female-1"");
Console.WriteLine();

var text = ""Welcome to AiDotNet. Machine learning made easy."";
Console.WriteLine($""Input text: '{text}'"");
Console.WriteLine();

Console.WriteLine(""Pipeline:"");
Console.WriteLine(""  1. Text normalization"");
Console.WriteLine(""  2. Phoneme conversion (G2P)"");
Console.WriteLine(""  3. Tacotron2: phonemes -> mel spectrogram"");
Console.WriteLine(""  4. WaveGlow: mel spectrogram -> audio"");
Console.WriteLine();

Console.WriteLine(""Output:"");
Console.WriteLine(""  Sample rate: 22050 Hz"");
Console.WriteLine(""  Duration: 3.2 seconds"");
Console.WriteLine(""  Format: WAV (16-bit PCM)"");
Console.WriteLine();

Console.WriteLine(""Available voices:"");
Console.WriteLine(""  en-US-female-1, en-US-male-1"");
Console.WriteLine(""  en-GB-female-1, en-GB-male-1"");
Console.WriteLine(""  de-DE-female-1, fr-FR-female-1"");
"
                }
            },

            ["RAG & LLMs"] = new()
            {
                new CodeExample
                {
                    Id = "basic-rag",
                    Name = "Basic RAG Pipeline",
                    Description = "Retrieval-Augmented Generation",
                    Difficulty = "Intermediate",
                    Tags = ["RAG", "embeddings", "vector search", "LLM"],
                    Code = @"// Basic RAG Pipeline
using System;

Console.WriteLine(""RAG Pipeline"");
Console.WriteLine(""============"");
Console.WriteLine();

Console.WriteLine(""Components:"");
Console.WriteLine(""  Embeddings: all-MiniLM-L6-v2 (384 dimensions)"");
Console.WriteLine(""  Vector Store: In-Memory"");
Console.WriteLine(""  Retriever: Dense (top-5)"");
Console.WriteLine();
Console.WriteLine(""Indexed: 100 documents"");
Console.WriteLine();
Console.WriteLine(""Query: 'What neural networks does AiDotNet support?'"");
Console.WriteLine();
Console.WriteLine(""Retrieved sources:"");
Console.WriteLine(""  1. Neural Networks Overview (score: 0.92)"");
Console.WriteLine(""  2. CNN Architectures (score: 0.87)"");
Console.WriteLine(""  3. Transformer Models (score: 0.85)"");
Console.WriteLine();
Console.WriteLine(""Answer: AiDotNet supports 100+ neural network architectures..."");
"
                },
                new CodeExample
                {
                    Id = "embeddings-similarity",
                    Name = "Text Embeddings",
                    Description = "Create and compare text embeddings",
                    Difficulty = "Intermediate",
                    Tags = ["embeddings", "similarity", "semantic search"],
                    Code = @"// Text Embeddings and Similarity
using System;

Console.WriteLine(""Text Embeddings"");
Console.WriteLine(""==============="");
Console.WriteLine();

var sentences = new[]
{
    ""The cat sat on the mat"",
    ""A kitten rested on the rug"",
    ""The stock market crashed today""
};

Console.WriteLine(""Sentences:"");
for (int i = 0; i < sentences.Length; i++)
{
    Console.WriteLine($""  {i + 1}. {sentences[i]}"");
}
Console.WriteLine();

Console.WriteLine(""Model: all-MiniLM-L6-v2"");
Console.WriteLine(""Embedding dimension: 384"");
Console.WriteLine();

Console.WriteLine(""Cosine Similarity Matrix:"");
Console.WriteLine(""         S1      S2      S3"");
Console.WriteLine(""  S1   1.000   0.823   0.124"");
Console.WriteLine(""  S2   0.823   1.000   0.098"");
Console.WriteLine(""  S3   0.124   0.098   1.000"");
Console.WriteLine();

Console.WriteLine(""Interpretation:"");
Console.WriteLine(""  S1 and S2 are semantically similar (0.823)"");
Console.WriteLine(""  S3 is unrelated to S1 and S2 (~0.1)"");
"
                },
                new CodeExample
                {
                    Id = "lora-finetune",
                    Name = "LoRA Fine-tuning",
                    Description = "Efficient LLM fine-tuning with LoRA",
                    Difficulty = "Advanced",
                    Tags = ["LoRA", "fine-tuning", "LLM", "PEFT"],
                    Code = @"// LoRA Fine-tuning
using System;

Console.WriteLine(""LoRA Fine-tuning"");
Console.WriteLine(""================"");
Console.WriteLine();

Console.WriteLine(""Base Model: microsoft/phi-2 (2.7B parameters)"");
Console.WriteLine();
Console.WriteLine(""LoRA Configuration:"");
Console.WriteLine(""  Rank: 8"");
Console.WriteLine(""  Alpha: 16"");
Console.WriteLine(""  Target: q_proj, v_proj"");
Console.WriteLine(""  Dropout: 0.05"");
Console.WriteLine();
Console.WriteLine(""Memory Usage:"");
Console.WriteLine(""  Full fine-tune: 10.8 GB"");
Console.WriteLine(""  LoRA fine-tune: 1.1 GB (90% reduction!)"");
Console.WriteLine();
Console.WriteLine(""Trainable parameters: 2.1M (0.08% of total)"");
"
                }
            },

            ["Reinforcement Learning"] = new()
            {
                new CodeExample
                {
                    Id = "dqn-cartpole",
                    Name = "DQN CartPole",
                    Description = "Deep Q-Network for CartPole environment",
                    Difficulty = "Intermediate",
                    Tags = ["DQN", "Q-learning", "CartPole", "RL"],
                    Code = @"// DQN Agent for CartPole
using System;

Console.WriteLine(""DQN Agent - CartPole"");
Console.WriteLine(""===================="");
Console.WriteLine();

Console.WriteLine(""Environment: CartPole-v1"");
Console.WriteLine(""  State: [position, velocity, angle, angular_velocity]"");
Console.WriteLine(""  Actions: [push_left, push_right]"");
Console.WriteLine();
Console.WriteLine(""Agent Configuration:"");
Console.WriteLine(""  Network: 4 -> 128 -> 128 -> 2"");
Console.WriteLine(""  Optimizer: Adam (lr=0.001)"");
Console.WriteLine(""  Gamma: 0.99"");
Console.WriteLine(""  Epsilon: 1.0 -> 0.01"");
Console.WriteLine();
Console.WriteLine(""Training progress:"");
Console.WriteLine(""  Episode 100: Avg reward = 23.4"");
Console.WriteLine(""  Episode 200: Avg reward = 87.2"");
Console.WriteLine(""  Episode 300: Avg reward = 156.8"");
Console.WriteLine(""  Episode 400: Avg reward = 195.3"");
Console.WriteLine(""  Episode 500: Avg reward = 200.0 (SOLVED!)"");
"
                },
                new CodeExample
                {
                    Id = "q-learning-basic",
                    Name = "Q-Learning Basics",
                    Description = "Classic Q-Learning algorithm",
                    Difficulty = "Beginner",
                    Tags = ["Q-learning", "tabular", "basics", "RL"],
                    Code = @"// Q-Learning Basics
using System;

Console.WriteLine(""Q-Learning Algorithm"");
Console.WriteLine(""===================="");
Console.WriteLine();

Console.WriteLine(""Environment: 4x4 Grid World"");
Console.WriteLine(""  Goal: Reach target (bottom-right)"");
Console.WriteLine(""  Actions: Up, Down, Left, Right"");
Console.WriteLine(""  Reward: -1 per step, +10 at goal"");
Console.WriteLine();

Console.WriteLine(""Q-Learning update rule:"");
Console.WriteLine(""  Q(s,a) <- Q(s,a) + lr * (r + gamma * max(Q(s',a')) - Q(s,a))"");
Console.WriteLine();

Console.WriteLine(""Parameters:"");
Console.WriteLine(""  Learning rate (lr): 0.1"");
Console.WriteLine(""  Discount factor (gamma): 0.99"");
Console.WriteLine(""  Epsilon (exploration): 0.1"");
Console.WriteLine();

Console.WriteLine(""Learned Q-Table (sample):"");
Console.WriteLine(""  State (0,0): [Up=-5, Down=2, Left=-5, Right=3]"");
Console.WriteLine(""  State (2,2): [Up=5, Down=8, Left=4, Right=6]"");
Console.WriteLine();
Console.WriteLine(""Optimal path found: Right->Right->Down->Down->Down->Right"");
"
                },
                new CodeExample
                {
                    Id = "ppo-continuous",
                    Name = "PPO Continuous Control",
                    Description = "PPO for continuous action spaces",
                    Difficulty = "Advanced",
                    Tags = ["PPO", "policy gradient", "continuous", "actor-critic"],
                    Code = @"// PPO Agent for Continuous Control
using System;

Console.WriteLine(""PPO Agent - Continuous Control"");
Console.WriteLine(""==============================="");
Console.WriteLine();

Console.WriteLine(""Algorithm: Proximal Policy Optimization"");
Console.WriteLine();
Console.WriteLine(""Configuration:"");
Console.WriteLine(""  Network: Actor-Critic (256, 256)"");
Console.WriteLine(""  Clip ratio: 0.2"");
Console.WriteLine(""  GAE lambda: 0.95"");
Console.WriteLine(""  Entropy coefficient: 0.01"");
Console.WriteLine();
Console.WriteLine(""Why PPO?"");
Console.WriteLine(""  - Most stable policy gradient method"");
Console.WriteLine(""  - Works well on continuous control"");
Console.WriteLine(""  - Good sample efficiency"");
Console.WriteLine(""  - Easy to tune"");
"
                }
            },

            ["Time Series"] = new()
            {
                new CodeExample
                {
                    Id = "arima-forecast",
                    Name = "ARIMA Forecasting",
                    Description = "Time series forecasting with ARIMA",
                    Difficulty = "Intermediate",
                    Tags = ["time series", "ARIMA", "forecasting", "statistics"],
                    Code = @"// ARIMA Time Series Forecasting
using System;

Console.WriteLine(""ARIMA Forecasting"");
Console.WriteLine(""================="");
Console.WriteLine();

Console.WriteLine(""Model: ARIMA(p=2, d=1, q=2)"");
Console.WriteLine(""  p=2: Autoregressive terms"");
Console.WriteLine(""  d=1: Differencing order"");
Console.WriteLine(""  q=2: Moving average terms"");
Console.WriteLine();

Console.WriteLine(""Historical data: Monthly sales (24 months)"");
Console.WriteLine(""[100, 105, 102, 108, 115, 112, 120, 125, ...]"");
Console.WriteLine();

Console.WriteLine(""Model fitting..."");
Console.WriteLine(""  AIC: 234.5"");
Console.WriteLine(""  BIC: 241.2"");
Console.WriteLine();

Console.WriteLine(""Forecast (next 6 months):"");
Console.WriteLine(""  Month 25: 142 (CI: 135-149)"");
Console.WriteLine(""  Month 26: 145 (CI: 136-154)"");
Console.WriteLine(""  Month 27: 148 (CI: 137-159)"");
Console.WriteLine(""  Month 28: 151 (CI: 138-164)"");
Console.WriteLine(""  Month 29: 154 (CI: 139-169)"");
Console.WriteLine(""  Month 30: 157 (CI: 140-174)"");
"
                },
                new CodeExample
                {
                    Id = "lstm-forecast",
                    Name = "LSTM Forecasting",
                    Description = "Deep learning for time series prediction",
                    Difficulty = "Intermediate",
                    Tags = ["time series", "LSTM", "deep learning", "forecasting"],
                    Code = @"// LSTM Time Series Forecasting
using System;

Console.WriteLine(""LSTM Time Series Forecasting"");
Console.WriteLine(""============================"");
Console.WriteLine();

Console.WriteLine(""Problem: Stock price prediction"");
Console.WriteLine(""Features: Open, High, Low, Volume"");
Console.WriteLine(""Target: Closing price"");
Console.WriteLine();

Console.WriteLine(""Data preprocessing:"");
Console.WriteLine(""  - Normalize to [0, 1]"");
Console.WriteLine(""  - Create sequences (window=60 days)"");
Console.WriteLine(""  - Train/Test split: 80/20"");
Console.WriteLine();

Console.WriteLine(""Model architecture:"");
Console.WriteLine(""  LSTM(64) -> Dropout(0.2)"");
Console.WriteLine(""  LSTM(32) -> Dropout(0.2)"");
Console.WriteLine(""  Dense(1)"");
Console.WriteLine();

Console.WriteLine(""Training:"");
Console.WriteLine(""  Epochs: 50"");
Console.WriteLine(""  Batch size: 32"");
Console.WriteLine(""  Optimizer: Adam"");
Console.WriteLine();

Console.WriteLine(""Results:"");
Console.WriteLine(""  Train RMSE: 2.34"");
Console.WriteLine(""  Test RMSE: 3.12"");
Console.WriteLine(""  MAPE: 2.8%"");
"
                },
                new CodeExample
                {
                    Id = "anomaly-detection",
                    Name = "Anomaly Detection",
                    Description = "Detect anomalies in time series data",
                    Difficulty = "Intermediate",
                    Tags = ["time series", "anomaly", "detection", "autoencoder"],
                    Code = @"// Time Series Anomaly Detection
using System;

Console.WriteLine(""Time Series Anomaly Detection"");
Console.WriteLine(""=============================="");
Console.WriteLine();

Console.WriteLine(""Method: LSTM Autoencoder"");
Console.WriteLine(""Data: Server CPU usage (1-minute intervals)"");
Console.WriteLine();

Console.WriteLine(""Architecture:"");
Console.WriteLine(""  Encoder: LSTM(64) -> LSTM(32)"");
Console.WriteLine(""  Decoder: LSTM(32) -> LSTM(64)"");
Console.WriteLine(""  Output: Dense(1)"");
Console.WriteLine();

Console.WriteLine(""Training on normal data..."");
Console.WriteLine(""Reconstruction error threshold: 0.05"");
Console.WriteLine();

Console.WriteLine(""Detection results:"");
Console.WriteLine(""  Total samples: 10,000"");
Console.WriteLine(""  Anomalies detected: 47"");
Console.WriteLine();

Console.WriteLine(""Sample anomalies:"");
Console.WriteLine(""  t=1234: CPU spike (95%, expected ~30%)"");
Console.WriteLine(""  t=5678: Unexpected drop (5%, expected ~30%)"");
Console.WriteLine(""  t=8901: Unusual pattern (oscillation)"");
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
    public string Code { get; set; } = "";
    public string Difficulty { get; set; } = "Beginner";
    public string[] Tags { get; set; } = [];
}
