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
Console.WriteLine(""  - 4,300+ AI/ML implementations"");
Console.WriteLine(""  - Tensor operations with hardware acceleration"");
Console.WriteLine(""  - Cross-platform: Windows, Linux, macOS"");
"
                },
                new CodeExample
                {
                    Id = "simple-regression",
                    Name = "Simple Regression",
                    Description = "Train a simple linear regression model",
                    Difficulty = "Beginner",
                    Tags = ["regression", "linear", "basics"],
                    Code = @"// Simple Linear Regression with AiDotNet
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;

// Training data: X = input features (single column), y = target values
var features = new Matrix<double>(5, 1);
features[0, 0] = 1.0; features[1, 0] = 2.0; features[2, 0] = 3.0;
features[3, 0] = 4.0; features[4, 0] = 5.0;

var labels = new Vector<double>(new double[] { 2.1, 4.0, 5.9, 8.1, 10.0 });

// Create and train the model
var model = new SimpleRegression<double>();
model.Train(features, labels);

// Make predictions
var testData = new Matrix<double>(1, 1);
testData[0, 0] = 6.0;
var predictions = model.Predict(testData);

Console.WriteLine($""Prediction for x=6: {predictions[0]:F2}"");
Console.WriteLine($""Expected: ~12.0 (based on y ≈ 2x)"");
Console.WriteLine($""Coefficients: {model.Coefficients[0]:F4}"");
Console.WriteLine($""Intercept: {model.Intercept:F4}"");
"
                },
                new CodeExample
                {
                    Id = "ridge-regression-basic",
                    Name = "Ridge Regression",
                    Description = "L2 regularized linear regression",
                    Difficulty = "Beginner",
                    Tags = ["regression", "regularization", "basics"],
                    Code = @"// Ridge Regression (L2 Regularized) with AiDotNet
using AiDotNet.Regression;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;

// Training data with 2 features
var features = new Matrix<double>(4, 2);
features[0, 0] = 1.0; features[0, 1] = 2.0;
features[1, 0] = 2.0; features[1, 1] = 3.0;
features[2, 0] = 3.0; features[2, 1] = 4.0;
features[3, 0] = 4.0; features[3, 1] = 5.0;

var labels = new Vector<double>(new double[] { 3.0, 5.0, 7.0, 9.0 });

// Create Ridge Regression with custom alpha (regularization strength)
var options = new RidgeRegressionOptions<double> { Alpha = 1.0 };
var model = new RidgeRegression<double>(options);
model.Train(features, labels);

// Make predictions
var testData = new Matrix<double>(1, 2);
testData[0, 0] = 5.0; testData[0, 1] = 6.0;
var predictions = model.Predict(testData);

Console.WriteLine($""Prediction for (5, 6): {predictions[0]:F2}"");
Console.WriteLine($""Ridge Regression trained with alpha={options.Alpha}"");
"
                }
            },

            ["Regression"] = new()
            {
                new CodeExample
                {
                    Id = "polynomial-regression",
                    Name = "Polynomial Regression",
                    Description = "Fit non-linear data with polynomial features",
                    Difficulty = "Intermediate",
                    Tags = ["regression", "polynomial"],
                    Code = @"// Polynomial Regression with AiDotNet
using AiDotNet.Regression;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;

// Non-linear data (quadratic relationship: y = x²)
var features = new Matrix<double>(5, 1);
features[0, 0] = 1.0; features[1, 0] = 2.0; features[2, 0] = 3.0;
features[3, 0] = 4.0; features[4, 0] = 5.0;

var labels = new Vector<double>(new double[] { 1.0, 4.0, 9.0, 16.0, 25.0 });

// Create polynomial regression with degree 2
var options = new PolynomialRegressionOptions<double> { Degree = 2 };
var model = new PolynomialRegression<double>(options);
model.Train(features, labels);

// Make predictions
var testData = new Matrix<double>(1, 1);
testData[0, 0] = 6.0;
var predictions = model.Predict(testData);

Console.WriteLine($""Prediction for x=6: {predictions[0]:F2}"");
Console.WriteLine($""Expected (6²): 36.00"");
Console.WriteLine($""Polynomial degree: {options.Degree}"");
"
                },
                new CodeExample
                {
                    Id = "lasso-regression",
                    Name = "Lasso Regression",
                    Description = "L1 regularized regression for sparse solutions",
                    Difficulty = "Intermediate",
                    Tags = ["regression", "regularization", "lasso"],
                    Code = @"// Lasso Regression (L1 Regularized) with AiDotNet
using AiDotNet.Regression;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;

// Training data with multiple features
var features = new Matrix<double>(5, 3);
features[0, 0] = 1.0; features[0, 1] = 2.0; features[0, 2] = 0.5;
features[1, 0] = 2.0; features[1, 1] = 3.0; features[1, 2] = 1.0;
features[2, 0] = 3.0; features[2, 1] = 4.0; features[2, 2] = 1.5;
features[3, 0] = 4.0; features[3, 1] = 5.0; features[3, 2] = 2.0;
features[4, 0] = 5.0; features[4, 1] = 6.0; features[4, 2] = 2.5;

var labels = new Vector<double>(new double[] { 3.5, 5.5, 7.5, 9.5, 11.5 });

// Create Lasso regression (promotes sparse coefficients)
var options = new LassoRegressionOptions<double> { Alpha = 0.1 };
var model = new LassoRegression<double>(options);
model.Train(features, labels);

// Show coefficients (some may be zero due to L1 regularization)
Console.WriteLine(""Lasso Regression Results:"");
Console.WriteLine($""Alpha (regularization): {options.Alpha}"");
Console.WriteLine(""Coefficients:"");
for (int i = 0; i < model.Coefficients.Length; i++)
{
    Console.WriteLine($""  Feature {i}: {model.Coefficients[i]:F4}"");
}
Console.WriteLine($""Intercept: {model.Intercept:F4}"");
"
                },
                new CodeExample
                {
                    Id = "elastic-net",
                    Name = "Elastic Net Regression",
                    Description = "Combined L1 and L2 regularization",
                    Difficulty = "Intermediate",
                    Tags = ["regression", "regularization", "elastic-net"],
                    Code = @"// Elastic Net Regression with AiDotNet
using AiDotNet.Regression;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;

// Training data
var features = new Matrix<double>(5, 2);
features[0, 0] = 1.0; features[0, 1] = 1.0;
features[1, 0] = 2.0; features[1, 1] = 2.0;
features[2, 0] = 3.0; features[2, 1] = 3.0;
features[3, 0] = 4.0; features[3, 1] = 4.0;
features[4, 0] = 5.0; features[4, 1] = 5.0;

var labels = new Vector<double>(new double[] { 2.0, 4.0, 6.0, 8.0, 10.0 });

// Create Elastic Net (combines L1 and L2 regularization)
var options = new ElasticNetRegressionOptions<double>
{
    Alpha = 0.1,        // Overall regularization strength
    L1Ratio = 0.5       // Balance between L1 (Lasso) and L2 (Ridge)
};
var model = new ElasticNetRegression<double>(options);
model.Train(features, labels);

// Make prediction
var testData = new Matrix<double>(1, 2);
testData[0, 0] = 6.0; testData[0, 1] = 6.0;
var predictions = model.Predict(testData);

Console.WriteLine(""Elastic Net Results:"");
Console.WriteLine($""Alpha: {options.Alpha}, L1 Ratio: {options.L1Ratio}"");
Console.WriteLine($""Prediction for (6, 6): {predictions[0]:F2}"");
Console.WriteLine($""Expected: ~12.0"");
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
                    Code = @"// Logistic Regression Classification with AiDotNet
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;

// Training data: features and binary labels
var features = new double[,]
{
    { 1.0, 2.0 }, { 2.0, 1.0 }, { 3.0, 3.0 }, { 4.0, 2.0 },
    { 1.5, 1.5 }, { 3.5, 3.0 }
};
var labels = new double[] { 0, 0, 1, 1, 0, 1 };

// Create data loader and model
var loader = DataLoaders.FromArrays(features, labels);
var model = new LogisticRegression<double>();

// Build using AiModelBuilder facade pattern
var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureDataLoader(loader)
    .ConfigureModel(model)
    .BuildAsync();

// Make prediction on new data
var testData = new Matrix<double>(1, 2);
testData[0, 0] = 3.5; testData[0, 1] = 2.5;
var predictions = result.Model.Predict(testData);

Console.WriteLine(""Logistic Regression Results:"");
Console.WriteLine($""  Prediction for (3.5, 2.5): {predictions[0]:F4}"");
Console.WriteLine($""  Predicted class: {(predictions[0] > 0.5 ? 1 : 0)}"");
Console.WriteLine($""  Coefficients: [{string.Join("", "", result.Model.Coefficients.Select(c => c.ToString(""F4"")))}]"");
Console.WriteLine($""  Intercept: {result.Model.Intercept:F4}"");
"
                },
                new CodeExample
                {
                    Id = "svm-classifier",
                    Name = "Support Vector Machine",
                    Description = "SVM classifier for classification tasks",
                    Difficulty = "Intermediate",
                    Tags = ["classification", "svm"],
                    Code = @"// SVM Classification with AiDotNet
using AiDotNet;
using AiDotNet.Classification.SVM;
using AiDotNet.Data.Loaders;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;

// Training data
var features = new double[,]
{
    { 1.0, 1.0 }, { 1.5, 2.0 }, { 2.0, 1.5 },
    { 4.0, 4.0 }, { 4.5, 3.5 }, { 5.0, 4.5 }
};
var labels = new double[] { 0, 0, 0, 1, 1, 1 };

// Create data loader and SVM with RBF kernel options
var loader = DataLoaders.FromArrays(features, labels);
var options = new SVMOptions<double>
{
    Kernel = KernelType.RBF,
    Gamma = 0.5,
    C = 1.0
};
var model = new SupportVectorClassifier<double>(options);

// Build using AiModelBuilder facade pattern
var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureDataLoader(loader)
    .ConfigureModel(model)
    .BuildAsync();

// Make predictions
var testData = new Matrix<double>(1, 2);
testData[0, 0] = 2.5; testData[0, 1] = 2.5;
var predictions = result.Model.Predict(testData);

Console.WriteLine(""SVM Classification Results:"");
Console.WriteLine($""  Kernel: RBF (Gamma={options.Gamma})"");
Console.WriteLine($""  C parameter: {options.C}"");
Console.WriteLine($""  Prediction for (2.5, 2.5): {predictions[0]:F4}"");
Console.WriteLine($""  Predicted class: {(predictions[0] > 0 ? 1 : 0)}"");
"
                },
                new CodeExample
                {
                    Id = "naive-bayes",
                    Name = "Naive Bayes",
                    Description = "Probabilistic classifier using Bayes theorem",
                    Difficulty = "Beginner",
                    Tags = ["classification", "probabilistic"],
                    Code = @"// Gaussian Naive Bayes Classification with AiDotNet
using AiDotNet;
using AiDotNet.Classification.NaiveBayes;
using AiDotNet.Data.Loaders;
using AiDotNet.Tensors.LinearAlgebra;

// Training data
var features = new double[,]
{
    { 1.0, 2.0 }, { 1.5, 1.8 }, { 2.0, 2.2 },
    { 5.0, 8.0 }, { 6.0, 9.0 }, { 5.5, 7.5 }
};
var labels = new double[] { 0, 0, 0, 1, 1, 1 };

// Create data loader and model
var loader = DataLoaders.FromArrays(features, labels);
var model = new GaussianNaiveBayes<double>();

// Build using AiModelBuilder facade pattern
var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureDataLoader(loader)
    .ConfigureModel(model)
    .BuildAsync();

// Make prediction
var testData = new Matrix<double>(1, 2);
testData[0, 0] = 3.0; testData[0, 1] = 4.0;
var predictions = result.Model.Predict(testData);

Console.WriteLine(""Gaussian Naive Bayes Results:"");
Console.WriteLine($""  Prediction for (3.0, 4.0): {predictions[0]:F4}"");
Console.WriteLine($""  Predicted class: {(predictions[0] > 0.5 ? 1 : 0)}"");
Console.WriteLine(""  Classes learned: 2"");
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
                    Code = @"// K-Means Clustering with AiDotNet
using AiDotNet.Clustering.Partitioning;
using AiDotNet.Clustering.Options;
using AiDotNet.Tensors.LinearAlgebra;

// Create sample data: 6 points in 2D
var data = new Matrix<double>(6, 2);
data[0, 0] = 1.0; data[0, 1] = 1.0;  // Cluster 1
data[1, 0] = 1.5; data[1, 1] = 2.0;  // Cluster 1
data[2, 0] = 1.2; data[2, 1] = 1.5;  // Cluster 1
data[3, 0] = 5.0; data[3, 1] = 7.0;  // Cluster 2
data[4, 0] = 5.5; data[4, 1] = 6.5;  // Cluster 2
data[5, 0] = 5.2; data[5, 1] = 7.2;  // Cluster 2

// Configure K-Means with 2 clusters
var options = new KMeansOptions<double> { NumClusters = 2 };
var kmeans = new KMeans<double>(options);

// Train (fit) the model - y parameter is ignored for clustering
kmeans.Train(data, new Vector<double>(data.Rows));

Console.WriteLine(""K-Means Clustering Results:"");
Console.WriteLine($""Number of clusters: {options.NumClusters}"");
Console.WriteLine($""Iterations: {kmeans.NumIterations}"");
Console.WriteLine();
Console.WriteLine(""Cluster assignments:"");
for (int i = 0; i < data.Rows; i++)
{
    var label = (int)kmeans.Labels[i];
    Console.WriteLine($""  Point ({data[i, 0]:F1}, {data[i, 1]:F1}) -> Cluster {label}"");
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
                    Code = @"// DBSCAN Clustering with AiDotNet
using AiDotNet.Clustering.Density;
using AiDotNet.Clustering.Options;
using AiDotNet.Tensors.LinearAlgebra;

// Create sample data with 2 clusters and 1 outlier
var data = new Matrix<double>(7, 2);
// Cluster 1 (dense region around 1,1)
data[0, 0] = 1.0; data[0, 1] = 1.0;
data[1, 0] = 1.1; data[1, 1] = 1.1;
data[2, 0] = 0.9; data[2, 1] = 1.0;
// Cluster 2 (dense region around 5,5)
data[3, 0] = 5.0; data[3, 1] = 5.0;
data[4, 0] = 5.1; data[4, 1] = 5.1;
data[5, 0] = 4.9; data[5, 1] = 5.0;
// Outlier (noise point)
data[6, 0] = 10.0; data[6, 1] = 10.0;

// Configure DBSCAN
var options = new DBSCANOptions<double>
{
    Epsilon = 0.5,  // Maximum distance between neighbors
    MinPoints = 2   // Minimum points to form a dense region
};
var dbscan = new DBSCAN<double>(options);

// Train the model
dbscan.Train(data, new Vector<double>(data.Rows));

Console.WriteLine(""DBSCAN Clustering Results:"");
Console.WriteLine($""Epsilon: {options.Epsilon}, MinPoints: {options.MinPoints}"");
Console.WriteLine($""Clusters found: {dbscan.NumClusters}"");
Console.WriteLine($""Noise points: {dbscan.GetNoiseCount()}"");
Console.WriteLine();
Console.WriteLine(""Cluster assignments (-1 = noise):"");
for (int i = 0; i < data.Rows; i++)
{
    var label = (int)dbscan.Labels[i];
    Console.WriteLine($""  Point ({data[i, 0]:F1}, {data[i, 1]:F1}) -> Cluster {label}"");
}
"
                },
                new CodeExample
                {
                    Id = "hdbscan",
                    Name = "HDBSCAN",
                    Description = "Hierarchical density-based clustering with automatic cluster detection",
                    Difficulty = "Intermediate",
                    Tags = ["clustering", "density", "hierarchical", "unsupervised"],
                    Code = @"// HDBSCAN - Hierarchical DBSCAN with AiDotNet
using AiDotNet.Clustering.Density;
using AiDotNet.Clustering.Options;
using AiDotNet.Tensors.LinearAlgebra;

// Create sample data with varying density clusters
var data = new Matrix<double>(12, 2);
// Dense cluster (points close together)
data[0, 0] = 0.0; data[0, 1] = 0.0;
data[1, 0] = 0.1; data[1, 1] = 0.1;
data[2, 0] = 0.2; data[2, 1] = 0.0;
data[3, 0] = 0.1; data[3, 1] = 0.2;
// Sparse cluster (points farther apart)
data[4, 0] = 5.0; data[4, 1] = 5.0;
data[5, 0] = 5.5; data[5, 1] = 5.0;
data[6, 0] = 5.0; data[6, 1] = 5.5;
data[7, 0] = 5.5; data[7, 1] = 5.5;
// Another cluster
data[8, 0] = 10.0; data[8, 1] = 0.0;
data[9, 0] = 10.2; data[9, 1] = 0.1;
data[10, 0] = 10.1; data[10, 1] = 0.2;
// Noise point
data[11, 0] = 20.0; data[11, 1] = 20.0;

// Configure HDBSCAN - no epsilon needed!
var options = new HDBSCANOptions<double>
{
    MinClusterSize = 3,  // Minimum points to form a cluster
    MinSamples = 2       // Controls noise sensitivity
};
var hdbscan = new HDBSCAN<double>(options);

// Train the model
hdbscan.Train(data, new Vector<double>(data.Rows));

Console.WriteLine(""HDBSCAN Clustering Results:"");
Console.WriteLine($""MinClusterSize: {options.MinClusterSize}"");
Console.WriteLine($""Clusters found: {hdbscan.NumClusters}"");
Console.WriteLine();
Console.WriteLine(""Cluster assignments (-1 = noise):"");
for (int i = 0; i < data.Rows; i++)
{
    var label = (int)hdbscan.Labels[i];
    Console.WriteLine($""  Point ({data[i, 0]:F1}, {data[i, 1]:F1}) -> Cluster {label}"");
}
Console.WriteLine();
Console.WriteLine(""HDBSCAN advantages over DBSCAN:"");
Console.WriteLine(""  - No epsilon parameter to tune"");
Console.WriteLine(""  - Handles varying density clusters"");
Console.WriteLine(""  - Provides cluster hierarchy"");
"
                },
                new CodeExample
                {
                    Id = "fuzzy-cmeans",
                    Name = "Fuzzy C-Means",
                    Description = "Soft clustering where points can belong to multiple clusters",
                    Difficulty = "Intermediate",
                    Tags = ["clustering", "fuzzy", "soft-clustering", "unsupervised"],
                    Code = @"// Fuzzy C-Means Soft Clustering with AiDotNet
using AiDotNet.Clustering.Partitioning;
using AiDotNet.Clustering.Options;
using AiDotNet.Tensors.LinearAlgebra;

// Create sample data
var data = new Matrix<double>(6, 2);
data[0, 0] = 1.0; data[0, 1] = 1.0;  // Clearly cluster 1
data[1, 0] = 1.2; data[1, 1] = 1.1;  // Clearly cluster 1
data[2, 0] = 5.0; data[2, 1] = 5.0;  // Clearly cluster 2
data[3, 0] = 5.2; data[3, 1] = 5.1;  // Clearly cluster 2
data[4, 0] = 3.0; data[4, 1] = 3.0;  // Between clusters!
data[5, 0] = 3.2; data[5, 1] = 3.1;  // Between clusters!

// Configure Fuzzy C-Means
var options = new FuzzyCMeansOptions<double>
{
    NumClusters = 2,
    Fuzziness = 2.0,     // Standard fuzziness (1.5-3.0 typical)
    MaxIterations = 300,
    Tolerance = 1e-4
};
var fcm = new FuzzyCMeans<double>(options);

// Train the model
fcm.Train(data, new Vector<double>(data.Rows));

Console.WriteLine(""Fuzzy C-Means Clustering Results:"");
Console.WriteLine($""Clusters: {options.NumClusters}"");
Console.WriteLine($""Fuzziness: {options.Fuzziness}"");
Console.WriteLine();
Console.WriteLine(""Membership degrees (point belongs to each cluster with probability):"");
for (int i = 0; i < data.Rows; i++)
{
    Console.WriteLine($""  Point ({data[i, 0]:F1}, {data[i, 1]:F1}):"");
    for (int k = 0; k < options.NumClusters; k++)
    {
        var membership = fcm.MembershipMatrix[i, k];
        Console.WriteLine($""    Cluster {k}: {membership:P1}"");
    }
}
Console.WriteLine();
Console.WriteLine(""Points at (3, 3) have mixed membership - they're between clusters!"");
"
                },
                new CodeExample
                {
                    Id = "meanshift",
                    Name = "Mean Shift",
                    Description = "Find clusters without specifying number of clusters",
                    Difficulty = "Intermediate",
                    Tags = ["clustering", "density", "mode-seeking", "unsupervised"],
                    Code = @"// Mean Shift Clustering with AiDotNet
using AiDotNet.Clustering.Density;
using AiDotNet.Clustering.Options;
using AiDotNet.Tensors.LinearAlgebra;

// Create sample data with natural clusters
var data = new Matrix<double>(9, 2);
// Cluster 1 (dense region)
data[0, 0] = 1.0; data[0, 1] = 1.0;
data[1, 0] = 1.2; data[1, 1] = 1.1;
data[2, 0] = 0.9; data[2, 1] = 1.2;
// Cluster 2 (another dense region)
data[3, 0] = 5.0; data[3, 1] = 5.0;
data[4, 0] = 5.2; data[4, 1] = 5.1;
data[5, 0] = 4.9; data[5, 1] = 5.2;
// Cluster 3
data[6, 0] = 8.0; data[6, 1] = 1.0;
data[7, 0] = 8.2; data[7, 1] = 1.1;
data[8, 0] = 7.9; data[8, 1] = 0.9;

// Configure Mean Shift - bandwidth is auto-estimated!
var options = new MeanShiftOptions<double>
{
    Bandwidth = null,  // Auto-estimate from data
    BandwidthQuantile = 0.3,
    BinSeeding = true,
    MaxIterations = 300
};
var meanshift = new MeanShift<double>(options);

// Train the model
meanshift.Train(data, new Vector<double>(data.Rows));

Console.WriteLine(""Mean Shift Clustering Results:"");
Console.WriteLine($""Bandwidth (auto-estimated): {meanshift.Bandwidth:F2}"");
Console.WriteLine($""Clusters found automatically: {meanshift.NumClusters}"");
Console.WriteLine();
Console.WriteLine(""Cluster assignments:"");
for (int i = 0; i < data.Rows; i++)
{
    var label = (int)meanshift.Labels[i];
    Console.WriteLine($""  Point ({data[i, 0]:F1}, {data[i, 1]:F1}) -> Cluster {label}"");
}
Console.WriteLine();
Console.WriteLine(""Mean Shift advantages:"");
Console.WriteLine(""  - No need to specify number of clusters"");
Console.WriteLine(""  - Finds natural cluster shapes"");
Console.WriteLine(""  - Robust to outliers"");
"
                },
                new CodeExample
                {
                    Id = "spectral-clustering",
                    Name = "Spectral Clustering",
                    Description = "Graph-based clustering for complex cluster shapes",
                    Difficulty = "Advanced",
                    Tags = ["clustering", "spectral", "graph", "unsupervised"],
                    Code = @"// Spectral Clustering with AiDotNet
using AiDotNet.Clustering.Spectral;
using AiDotNet.Clustering.Options;
using AiDotNet.Tensors.LinearAlgebra;

// Create sample data - spectral clustering works well for non-spherical clusters
var data = new Matrix<double>(8, 2);
// First ring/arc
data[0, 0] = 0.0; data[0, 1] = 1.0;
data[1, 0] = 0.5; data[1, 1] = 0.9;
data[2, 0] = 1.0; data[2, 1] = 0.5;
data[3, 0] = 1.2; data[3, 1] = 0.0;
// Second ring/arc
data[4, 0] = 2.0; data[4, 1] = 1.0;
data[5, 0] = 2.5; data[5, 1] = 0.9;
data[6, 0] = 3.0; data[6, 1] = 0.5;
data[7, 0] = 3.2; data[7, 1] = 0.0;

// Configure Spectral Clustering
var options = new SpectralOptions<double>
{
    NumClusters = 2,
    Affinity = AffinityType.RBF,      // Radial Basis Function kernel
    Gamma = 1.0,                       // RBF kernel parameter
    Normalization = LaplacianNormalization.Normalized,
    AssignLabels = SpectralAssignment.KMeans
};
var spectral = new SpectralClustering<double>(options);

// Train the model
spectral.Train(data, new Vector<double>(data.Rows));

Console.WriteLine(""Spectral Clustering Results:"");
Console.WriteLine($""Clusters: {options.NumClusters}"");
Console.WriteLine($""Affinity: {options.Affinity}"");
Console.WriteLine($""Normalization: {options.Normalization}"");
Console.WriteLine();
Console.WriteLine(""Cluster assignments:"");
for (int i = 0; i < data.Rows; i++)
{
    var label = (int)spectral.Labels[i];
    Console.WriteLine($""  Point ({data[i, 0]:F1}, {data[i, 1]:F1}) -> Cluster {label}"");
}
Console.WriteLine();
Console.WriteLine(""Spectral Clustering advantages:"");
Console.WriteLine(""  - Finds non-spherical clusters (moons, spirals)"");
Console.WriteLine(""  - Based on graph connectivity, not just distance"");
Console.WriteLine(""  - Works better than K-Means for complex shapes"");
"
                },
                new CodeExample
                {
                    Id = "kmeans-advanced",
                    Name = "K-Means with Options",
                    Description = "K-Means with K-Means++ initialization",
                    Difficulty = "Intermediate",
                    Tags = ["clustering", "kmeans", "advanced"],
                    Code = @"// K-Means with Advanced Options
using AiDotNet.Clustering.Partitioning;
using AiDotNet.Clustering.Options;
using AiDotNet.Tensors.LinearAlgebra;

// Create sample data
var data = new Matrix<double>(9, 2);
// Three clusters of 3 points each
data[0, 0] = 0.0; data[0, 1] = 0.0;
data[1, 0] = 0.5; data[1, 1] = 0.5;
data[2, 0] = 0.3; data[2, 1] = 0.2;
data[3, 0] = 5.0; data[3, 1] = 5.0;
data[4, 0] = 5.5; data[4, 1] = 5.3;
data[5, 0] = 5.2; data[5, 1] = 4.8;
data[6, 0] = 10.0; data[6, 1] = 0.0;
data[7, 0] = 10.3; data[7, 1] = 0.2;
data[8, 0] = 9.8; data[8, 1] = 0.3;

// Configure with K-Means++ and multiple initializations
var options = new KMeansOptions<double>
{
    NumClusters = 3,
    InitMethod = KMeansInitMethod.KMeansPlusPlus,
    MaxIterations = 100,
    Tolerance = 1e-4,
    NumInitializations = 10,
    RandomState = 42  // For reproducibility
};
var kmeans = new KMeans<double>(options);
kmeans.Train(data, new Vector<double>(data.Rows));

Console.WriteLine(""K-Means++ Clustering Results:"");
Console.WriteLine($""Clusters: {options.NumClusters}"");
Console.WriteLine($""Init method: K-Means++"");
Console.WriteLine($""Iterations: {kmeans.NumIterations}"");
Console.WriteLine();
Console.WriteLine(""Cluster Centers:"");
for (int k = 0; k < options.NumClusters; k++)
{
    Console.WriteLine($""  Cluster {k}: ({kmeans.ClusterCenters[k, 0]:F2}, {kmeans.ClusterCenters[k, 1]:F2})"");
}
"
                },
                new CodeExample
                {
                    Id = "silhouette-score",
                    Name = "Cluster Evaluation",
                    Description = "Evaluate clustering quality with Silhouette Score",
                    Difficulty = "Intermediate",
                    Tags = ["clustering", "evaluation", "metrics", "silhouette"],
                    Code = @"// Cluster Evaluation with Silhouette Score
using AiDotNet.Clustering.Partitioning;
using AiDotNet.Clustering.Options;
using AiDotNet.Clustering.Evaluation;
using AiDotNet.Tensors.LinearAlgebra;

// Create well-separated clusters
var data = new Matrix<double>(6, 2);
data[0, 0] = 0.0; data[0, 1] = 0.0;
data[1, 0] = 0.1; data[1, 1] = 0.1;
data[2, 0] = 0.2; data[2, 1] = 0.0;
data[3, 0] = 5.0; data[3, 1] = 5.0;
data[4, 0] = 5.1; data[4, 1] = 5.1;
data[5, 0] = 5.0; data[5, 1] = 5.2;

// Test different numbers of clusters
Console.WriteLine(""Silhouette Score Analysis:"");
Console.WriteLine(""(Higher is better, range -1 to +1)"");
Console.WriteLine();

for (int k = 2; k <= 4; k++)
{
    var options = new KMeansOptions<double> { NumClusters = k, RandomState = 42 };
    var kmeans = new KMeans<double>(options);
    kmeans.Train(data, new Vector<double>(data.Rows));

    // Calculate Silhouette Score
    var silhouette = new SilhouetteScore<double>();
    var score = silhouette.Compute(data, kmeans.Labels);

    Console.WriteLine($""  K={k}: Silhouette Score = {score:F4}"");
}

Console.WriteLine();
Console.WriteLine(""Interpretation:"");
Console.WriteLine(""  +1: Perfect clustering (points clearly in right cluster)"");
Console.WriteLine(""   0: Points are on cluster boundaries"");
Console.WriteLine(""  -1: Points may be in wrong clusters"");
Console.WriteLine();
Console.WriteLine(""Best K is the one with highest Silhouette Score!"");
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
                    Code = @"// Neural Network for XOR problem using AiDotNet
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

// XOR problem data
Console.WriteLine(""Neural Network XOR Problem Demonstration"");
Console.WriteLine();

// Create architecture for XOR (2 inputs -> hidden layer -> 1 output)
var architecture = new NeuralNetworkArchitecture<double>(
    inputType: InputType.OneDimensional,
    taskType: NeuralNetworkTaskType.BinaryClassification,
    complexity: NetworkComplexity.Simple,
    inputSize: 2,
    outputSize: 1);

// Create neural network
var nn = new FeedForwardNeuralNetwork<double>(architecture);

Console.WriteLine(""XOR Neural Network Architecture:"");
Console.WriteLine($""  Input size: {architecture.InputSize}"");
Console.WriteLine($""  Output size: {architecture.OutputSize}"");
Console.WriteLine($""  Task type: {architecture.TaskType}"");
Console.WriteLine($""  Layers: {nn.Layers.Count}"");
Console.WriteLine();

// Create input data as tensor
var inputs = new double[4, 2]
{
    { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 }
};

Console.WriteLine(""XOR Truth Table:"");
Console.WriteLine(""  Input1  Input2  Expected"");
Console.WriteLine(""    0       0        0"");
Console.WriteLine(""    0       1        1"");
Console.WriteLine(""    1       0        1"");
Console.WriteLine(""    1       1        0"");
Console.WriteLine();
Console.WriteLine(""Note: Full training requires epochs of backpropagation."");
Console.WriteLine(""Use TrainBatch() method with training data for actual training."");
"
                },
                new CodeExample
                {
                    Id = "neural-net-regression",
                    Name = "Neural Network Regression",
                    Description = "Neural network for continuous value prediction",
                    Difficulty = "Intermediate",
                    Tags = ["neural-network", "regression"],
                    Code = @"// Neural Network Regression using AiDotNet
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

Console.WriteLine(""Neural Network Regression: Function Approximation"");
Console.WriteLine();

// Create architecture for regression (1 input -> hidden -> 1 output)
var architecture = new NeuralNetworkArchitecture<double>(
    inputType: InputType.OneDimensional,
    taskType: NeuralNetworkTaskType.Regression,
    complexity: NetworkComplexity.Medium,
    inputSize: 1,
    outputSize: 1);

// Create neural network for regression
var nn = new FeedForwardNeuralNetwork<double>(architecture);

Console.WriteLine(""Sin(x) Function Approximation:"");
Console.WriteLine($""  Architecture: 1 -> hidden layers -> 1"");
Console.WriteLine($""  Task: Regression (continuous output)"");
Console.WriteLine();

// Generate training data points
Console.WriteLine(""Sample training data (y = sin(x)):"");
for (int i = 0; i <= 4; i++)
{
    double x = i * Math.PI / 4;
    double y = Math.Sin(x);
    Console.WriteLine($""  x = {x:F4}, sin(x) = {y:F4}"");
}

Console.WriteLine();
Console.WriteLine(""Target: Predict sin(1.57) = ~1.0"");
Console.WriteLine($""Actual sin(1.57) = {Math.Sin(1.57):F4}"");
Console.WriteLine();
Console.WriteLine(""Note: Use TrainBatch() for training with gradient descent."");
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
                    Code = @"// ARIMA Time Series Forecasting with AiDotNet
using AiDotNet.TimeSeries;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;

// Monthly sales data
var data = new double[] { 100, 120, 130, 125, 140, 150, 160, 155, 170, 180, 190, 200 };

// Create ARIMA model with options
var options = new ARIMAOptions<double>
{
    P = 1,  // AutoRegressive order
    D = 1,  // Differencing order
    Q = 1   // Moving Average order
};
var arima = new ARIMAModel<double>(options);

// Convert data to Matrix and dummy Vector for Train
var features = new Matrix<double>(data.Length, 1);
for (int i = 0; i < data.Length; i++)
    features[i, 0] = i;  // Time index
var labels = new Vector<double>(data);

// Train the model
arima.Train(features, labels);

Console.WriteLine(""ARIMA Time Series Model:"");
Console.WriteLine($""  Parameters: P={options.P}, D={options.D}, Q={options.Q}"");
Console.WriteLine($""  Training data: {data.Length} observations"");
Console.WriteLine();
Console.WriteLine(""Training data:"");
for (int i = 0; i < data.Length; i++)
{
    Console.WriteLine($""  Month {i + 1}: {data[i]:F0}"");
}
Console.WriteLine();
Console.WriteLine(""ARIMA is used for forecasting future values based on"");
Console.WriteLine(""autoregressive patterns, trends, and moving averages."");
"
                },
                new CodeExample
                {
                    Id = "exponential-smoothing",
                    Name = "Exponential Smoothing",
                    Description = "Simple exponential smoothing for forecasting",
                    Difficulty = "Beginner",
                    Tags = ["time-series", "forecasting", "smoothing"],
                    Code = @"// Exponential Smoothing with AiDotNet
using AiDotNet.TimeSeries;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;

// Time series data
var data = new double[] { 10, 12, 13, 15, 14, 16, 18, 17, 19, 20 };

// Create Exponential Smoothing model with options
var options = new ExponentialSmoothingOptions<double>
{
    InitialAlpha = 0.3  // Smoothing factor (0-1)
};
var model = new ExponentialSmoothingModel<double>(options);

// Convert data to Matrix and Vector for Train
var features = new Matrix<double>(data.Length, 1);
for (int i = 0; i < data.Length; i++)
    features[i, 0] = i;  // Time index
var labels = new Vector<double>(data);

// Train the model
model.Train(features, labels);

Console.WriteLine(""Exponential Smoothing Model:"");
Console.WriteLine($""  Alpha (smoothing factor): {options.InitialAlpha}"");
Console.WriteLine($""  Training data: {data.Length} observations"");
Console.WriteLine();
Console.WriteLine(""Original data:"");
for (int i = 0; i < data.Length; i++)
{
    Console.WriteLine($""  t={i}: {data[i]:F1}"");
}
Console.WriteLine();
Console.WriteLine(""Higher alpha = more weight on recent observations"");
Console.WriteLine(""Lower alpha = smoother forecasts, less reactive to changes"");
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
                    Code = @"// Cross-Validation with AiDotNet
using AiDotNet;
using AiDotNet.Classification.Ensemble;
using AiDotNet.CrossValidators;
using AiDotNet.Data.Loaders;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

// Generate sample data
var rng = RandomHelper.CreateSeededRandom(42);
var features = new double[100, 2];
var labels = new double[100];
for (int i = 0; i < 100; i++)
{
    features[i, 0] = rng.NextDouble() * 10;
    features[i, 1] = rng.NextDouble() * 10;
    labels[i] = features[i, 0] + features[i, 1] > 10 ? 1 : 0;
}

// Create data loader and model
var loader = DataLoaders.FromArrays(features, labels);
var rfOptions = new RandomForestClassifierOptions<double> { NEstimators = 50 };
var model = new RandomForestClassifier<double>(rfOptions);

// Create cross-validator
var cvOptions = new CrossValidationOptions { NumFolds = 5, Shuffle = true };
var crossValidator = new KFoldCrossValidator<double, Matrix<double>, Vector<double>>(cvOptions);

// Build with cross-validation
var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureDataLoader(loader)
    .ConfigureModel(model)
    .ConfigureCrossValidation(crossValidator)
    .BuildAsync();

Console.WriteLine(""Random Forest with 5-Fold Cross-Validation:"");
Console.WriteLine($""  Number of trees: {rfOptions.NEstimators}"");
Console.WriteLine($""  Training samples: {features.GetLength(0)}"");
Console.WriteLine($""  Features: {features.GetLength(1)}"");
Console.WriteLine();
Console.WriteLine(""Cross-validation evaluates model performance by:"");
Console.WriteLine(""  1. Splitting data into 5 equal folds"");
Console.WriteLine(""  2. Training on 4 folds, validating on 1"");
Console.WriteLine(""  3. Repeating for each fold"");
Console.WriteLine(""  4. Averaging results for robust estimate"");
"
                },
                new CodeExample
                {
                    Id = "hyperparameter-tuning",
                    Name = "Hyperparameter Tuning",
                    Description = "Automatically find the best model parameters",
                    Difficulty = "Advanced",
                    Tags = ["automl", "hyperparameter", "optimization"],
                    Code = @"// Hyperparameter Tuning with AiDotNet
using AiDotNet.AutoML.SearchSpace;
using AiDotNet.Tensors.Helpers;

Console.WriteLine(""Hyperparameter Tuning Demonstration"");
Console.WriteLine();

// Define hyperparameter search space
var searchSpace = new HyperparameterSearchSpace();
searchSpace.AddIntParameter(""nEstimators"", 10, 200);
searchSpace.AddIntParameter(""maxDepth"", 3, 20);
searchSpace.AddFloatParameter(""minSamplesSplit"", 0.01, 0.5);

Console.WriteLine(""Search Space Configuration:"");
Console.WriteLine(""  nEstimators: [10, 200] (integer)"");
Console.WriteLine(""  maxDepth: [3, 20] (integer)"");
Console.WriteLine(""  minSamplesSplit: [0.01, 0.5] (float)"");
Console.WriteLine();

// Generate sample data
var rng = RandomHelper.CreateSeededRandom(42);
var features = new double[200, 4];
var labels = new double[200];
for (int i = 0; i < 200; i++)
{
    for (int j = 0; j < 4; j++)
        features[i, j] = rng.NextDouble();
    labels[i] = features[i, 0] > 0.5 ? 1 : 0;
}

Console.WriteLine($""Training data: {features.GetLength(0)} samples, {features.GetLength(1)} features"");
Console.WriteLine();

Console.WriteLine(""Usage with AiModelBuilder:"");
Console.WriteLine(""  .ConfigureHyperparameterOptimizer(optimizer)"");
Console.WriteLine();
Console.WriteLine(""Optimization finds best hyperparameters via search strategies:"");
Console.WriteLine(""  - Grid Search: exhaustive search over parameter grid"");
Console.WriteLine(""  - Random Search: random sampling of parameter space"");
Console.WriteLine(""  - Bayesian Optimization: intelligent search using surrogate model"");
"
                },
                new CodeExample
                {
                    Id = "model-persistence",
                    Name = "Save and Load Models",
                    Description = "Persist trained models for later use",
                    Difficulty = "Beginner",
                    Tags = ["persistence", "save", "load"],
                    Code = @"// Model Persistence with AiDotNet
using AiDotNet;
using AiDotNet.Classification.Ensemble;
using AiDotNet.Data.Loaders;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;

// Create training data
var features = new double[,]
{
    { 1, 2 }, { 2, 1 }, { 3, 4 }, { 4, 3 },
    { 5, 6 }, { 6, 5 }, { 7, 8 }, { 8, 7 }
};
var labels = new double[] { 0, 0, 0, 0, 1, 1, 1, 1 };

// Create data loader and model
var loader = DataLoaders.FromArrays(features, labels);
var options = new RandomForestClassifierOptions<double> { NEstimators = 50 };
var model = new RandomForestClassifier<double>(options);

// Build and train
var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureDataLoader(loader)
    .ConfigureModel(model)
    .BuildAsync();

Console.WriteLine(""Random Forest Model Trained:"");
Console.WriteLine($""  Number of trees: {options.NEstimators}"");
Console.WriteLine($""  Training samples: {features.GetLength(0)}"");
Console.WriteLine();

// Make prediction
var testData = new Matrix<double>(1, 2);
testData[0, 0] = 6.0; testData[0, 1] = 7.0;
var predictions = result.Model.Predict(testData);

Console.WriteLine($""Prediction for (6, 7): {predictions[0]:F4}"");
Console.WriteLine($""Predicted class: {(predictions[0] > 0.5 ? 1 : 0)}"");
Console.WriteLine();
Console.WriteLine(""Model persistence features:"");
Console.WriteLine(""  - Save models using model.Serialize()"");
Console.WriteLine(""  - Load models using deserialization"");
Console.WriteLine(""  - Full model state preserved including weights"");
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
                    Code = @"// CNN Image Classification using AiDotNet
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Helpers;

Console.WriteLine(""CNN Image Classification: MNIST-style Digits"");
Console.WriteLine();

// Create CNN architecture for image classification
var architecture = new NeuralNetworkArchitecture<double>(
    inputType: InputType.TwoDimensional,
    taskType: NeuralNetworkTaskType.MultiClassClassification,
    complexity: NetworkComplexity.Medium,
    inputHeight: 28,
    inputWidth: 28,
    inputDepth: 1,  // Grayscale
    outputSize: 10); // 10 digit classes

// Create convolutional neural network
var cnn = new ConvolutionalNeuralNetwork<double>(architecture);

Console.WriteLine(""CNN Architecture (MNIST-style):"");
Console.WriteLine($""  Input: {architecture.InputHeight}x{architecture.InputWidth}x{architecture.InputDepth}"");
Console.WriteLine($""  Output: {architecture.OutputSize} classes"");
Console.WriteLine($""  Layers: {cnn.Layers.Count}"");
Console.WriteLine();

// Typical CNN layer structure
Console.WriteLine(""Typical CNN Structure:"");
Console.WriteLine(""  Conv2D(1->32, 3x3) + ReLU + MaxPool(2x2)"");
Console.WriteLine(""  Conv2D(32->64, 3x3) + ReLU + MaxPool(2x2)"");
Console.WriteLine(""  Flatten"");
Console.WriteLine(""  Dense(128) + ReLU"");
Console.WriteLine(""  Dense(10) + Softmax"");
Console.WriteLine();

// Sample data dimensions
var rng = RandomHelper.CreateSeededRandom(42);
Console.WriteLine(""Sample Image Data:"");
Console.WriteLine(""  100 samples of 28x28 grayscale images"");
Console.WriteLine(""  10 classes (digits 0-9)"");
Console.WriteLine();
Console.WriteLine(""Use TrainBatch() with batches for training."");
"
                },
                new CodeExample
                {
                    Id = "resnet-transfer",
                    Name = "ResNet Transfer Learning",
                    Description = "Use pre-trained ResNet for image classification",
                    Difficulty = "Advanced",
                    Tags = ["resnet", "transfer-learning", "pretrained"],
                    Code = @"// ResNet Transfer Learning using AiDotNet
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Helpers;

Console.WriteLine(""ResNet Transfer Learning"");
Console.WriteLine();

// Create ResNet architecture for transfer learning
var architecture = new NeuralNetworkArchitecture<double>(
    inputType: InputType.ThreeDimensional,
    taskType: NeuralNetworkTaskType.MultiClassClassification,
    complexity: NetworkComplexity.High,
    inputHeight: 224,
    inputWidth: 224,
    inputDepth: 3,  // RGB
    outputSize: 5); // 5 custom classes

// Create ResNet-style network
var resnet = new ResidualNeuralNetwork<double>(architecture);

Console.WriteLine(""ResNet Configuration:"");
Console.WriteLine($""  Input: {architecture.InputHeight}x{architecture.InputWidth}x{architecture.InputDepth} (RGB)"");
Console.WriteLine($""  Output: {architecture.OutputSize} classes"");
Console.WriteLine($""  Layers: {resnet.Layers.Count}"");
Console.WriteLine();

Console.WriteLine(""Transfer Learning Steps:"");
Console.WriteLine(""  1. Load pre-trained weights (ImageNet)"");
Console.WriteLine(""  2. Freeze early layers (feature extractors)"");
Console.WriteLine(""  3. Replace final classification head"");
Console.WriteLine(""  4. Fine-tune on new dataset"");
Console.WriteLine();

// Sample data dimensions
var rng = RandomHelper.CreateSeededRandom(42);
Console.WriteLine(""Sample Data:"");
Console.WriteLine(""  50 samples of 224x224 RGB images"");
Console.WriteLine(""  5 custom classes"");
Console.WriteLine();
Console.WriteLine(""Benefits of Transfer Learning:"");
Console.WriteLine(""  - Requires less training data"");
Console.WriteLine(""  - Faster convergence"");
Console.WriteLine(""  - Better generalization"");
"
                },
                new CodeExample
                {
                    Id = "object-detection-yolo",
                    Name = "Object Detection (YOLO)",
                    Description = "Detect objects in images using YOLO",
                    Difficulty = "Advanced",
                    Tags = ["yolo", "object-detection", "deep-learning"],
                    Code = @"// Object Detection with YOLO using AiDotNet
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Helpers;

Console.WriteLine(""YOLO Object Detection"");
Console.WriteLine();

// YOLO architecture configuration
Console.WriteLine(""YOLO (You Only Look Once) Configuration:"");
Console.WriteLine(""  Input: 416x416 or 640x640 RGB images"");
Console.WriteLine(""  Output: Bounding boxes + class probabilities"");
Console.WriteLine();

// Simulated detection results
var rng = RandomHelper.CreateSeededRandom(42);
Console.WriteLine(""Sample Detection Results:"");
Console.WriteLine(""  Object: Person"");
Console.WriteLine($""    Confidence: {0.95:P1}"");
Console.WriteLine($""    Box: (100, 50, 150, 300)"");
Console.WriteLine();
Console.WriteLine(""  Object: Car"");
Console.WriteLine($""    Confidence: {0.87:P1}"");
Console.WriteLine($""    Box: (250, 200, 200, 100)"");
Console.WriteLine();

Console.WriteLine(""YOLO Detection Pipeline:"");
Console.WriteLine(""  1. Image preprocessing (resize, normalize)"");
Console.WriteLine(""  2. Feature extraction (backbone network)"");
Console.WriteLine(""  3. Detection head (bounding boxes + classes)"");
Console.WriteLine(""  4. Non-maximum suppression (NMS)"");
Console.WriteLine();

Console.WriteLine(""Model Sizes Available:"");
Console.WriteLine(""  - YOLOv8n (nano): 3.2M params"");
Console.WriteLine(""  - YOLOv8s (small): 11.2M params"");
Console.WriteLine(""  - YOLOv8m (medium): 25.9M params"");
Console.WriteLine(""  - YOLOv8l (large): 43.7M params"");
"
                }
            },

            ["Tensors & Linear Algebra"] = new()
            {
                new CodeExample
                {
                    Id = "tensor-basics",
                    Name = "Tensor Basics",
                    Description = "Create and manipulate tensors (multi-dimensional arrays)",
                    Difficulty = "Beginner",
                    Tags = ["tensor", "matrix", "vector", "linear-algebra"],
                    Code = @"// Tensor Basics with AiDotNet
using AiDotNet.Tensors.LinearAlgebra;

// Create a Vector (1D tensor)
var vector = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
Console.WriteLine($""Vector: [{string.Join("", "", vector)}]"");
Console.WriteLine($""Length: {vector.Length}"");
Console.WriteLine($""Sum: {vector.Sum():F2}"");
Console.WriteLine();

// Create a Matrix (2D tensor)
var matrix = new Matrix<double>(3, 3);
matrix[0, 0] = 1; matrix[0, 1] = 2; matrix[0, 2] = 3;
matrix[1, 0] = 4; matrix[1, 1] = 5; matrix[1, 2] = 6;
matrix[2, 0] = 7; matrix[2, 1] = 8; matrix[2, 2] = 9;

Console.WriteLine(""Matrix (3x3):"");
for (int i = 0; i < matrix.Rows; i++)
{
    Console.Write(""  ["");
    for (int j = 0; j < matrix.Columns; j++)
    {
        Console.Write($""{matrix[i, j],3}"");
        if (j < matrix.Columns - 1) Console.Write("", "");
    }
    Console.WriteLine(""]"");
}
Console.WriteLine($""Shape: ({matrix.Rows}, {matrix.Columns})"");
"
                },
                new CodeExample
                {
                    Id = "matrix-operations",
                    Name = "Matrix Operations",
                    Description = "Perform matrix arithmetic and transformations",
                    Difficulty = "Beginner",
                    Tags = ["tensor", "matrix", "operations", "linear-algebra"],
                    Code = @"// Matrix Operations with AiDotNet
using AiDotNet.Tensors.LinearAlgebra;

// Create two matrices
var A = new Matrix<double>(2, 3);
A[0, 0] = 1; A[0, 1] = 2; A[0, 2] = 3;
A[1, 0] = 4; A[1, 1] = 5; A[1, 2] = 6;

var B = new Matrix<double>(2, 3);
B[0, 0] = 7; B[0, 1] = 8; B[0, 2] = 9;
B[1, 0] = 10; B[1, 1] = 11; B[1, 2] = 12;

Console.WriteLine(""Matrix A (2x3):"");
PrintMatrix(A);

Console.WriteLine(""Matrix B (2x3):"");
PrintMatrix(B);

// Element-wise addition
var C = A.Add(B);
Console.WriteLine(""A + B:"");
PrintMatrix(C);

// Scalar multiplication
var D = A.Multiply(2.0);
Console.WriteLine(""A * 2:"");
PrintMatrix(D);

// Transpose
var AT = A.Transpose();
Console.WriteLine(""A transposed (3x2):"");
PrintMatrix(AT);

void PrintMatrix(Matrix<double> m)
{
    for (int i = 0; i < m.Rows; i++)
    {
        Console.Write(""  ["");
        for (int j = 0; j < m.Columns; j++)
        {
            Console.Write($""{m[i, j],5:F1}"");
            if (j < m.Columns - 1) Console.Write("", "");
        }
        Console.WriteLine(""]"");
    }
    Console.WriteLine();
}
"
                },
                new CodeExample
                {
                    Id = "matrix-multiplication",
                    Name = "Matrix Multiplication",
                    Description = "Multiply matrices (the core of neural networks)",
                    Difficulty = "Intermediate",
                    Tags = ["tensor", "matrix", "matmul", "linear-algebra"],
                    Code = @"// Matrix Multiplication with AiDotNet
using AiDotNet.Tensors.LinearAlgebra;

// Create matrices for multiplication
// A is 2x3, B is 3x2, result will be 2x2
var A = new Matrix<double>(2, 3);
A[0, 0] = 1; A[0, 1] = 2; A[0, 2] = 3;
A[1, 0] = 4; A[1, 1] = 5; A[1, 2] = 6;

var B = new Matrix<double>(3, 2);
B[0, 0] = 7; B[0, 1] = 8;
B[1, 0] = 9; B[1, 1] = 10;
B[2, 0] = 11; B[2, 1] = 12;

Console.WriteLine(""Matrix A (2x3):"");
PrintMatrix(A);

Console.WriteLine(""Matrix B (3x2):"");
PrintMatrix(B);

// Matrix multiplication: C = A @ B
var C = A.Multiply(B);

Console.WriteLine(""C = A @ B (2x2):"");
PrintMatrix(C);

// Verify: C[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
Console.WriteLine(""Verification:"");
Console.WriteLine($""  C[0,0] = 1*7 + 2*9 + 3*11 = {1*7 + 2*9 + 3*11}"");
Console.WriteLine($""  Matches matrix result: {C[0, 0]}"");

void PrintMatrix(Matrix<double> m)
{
    for (int i = 0; i < m.Rows; i++)
    {
        Console.Write(""  ["");
        for (int j = 0; j < m.Columns; j++)
        {
            Console.Write($""{m[i, j],5:F0}"");
            if (j < m.Columns - 1) Console.Write("", "");
        }
        Console.WriteLine(""]"");
    }
    Console.WriteLine();
}
"
                },
                new CodeExample
                {
                    Id = "vector-operations",
                    Name = "Vector Operations",
                    Description = "Perform vector arithmetic and common operations",
                    Difficulty = "Beginner",
                    Tags = ["tensor", "vector", "operations", "linear-algebra"],
                    Code = @"// Vector Operations with AiDotNet
using AiDotNet.Tensors.LinearAlgebra;

// Create vectors
var v1 = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
var v2 = new Vector<double>(new double[] { 4.0, 5.0, 6.0 });

Console.WriteLine($""Vector v1: [{string.Join("", "", v1)}]"");
Console.WriteLine($""Vector v2: [{string.Join("", "", v2)}]"");
Console.WriteLine();

// Vector addition
var vSum = v1.Add(v2);
Console.WriteLine($""v1 + v2 = [{string.Join("", "", vSum)}]"");

// Scalar multiplication
var vScaled = v1.Multiply(2.0);
Console.WriteLine($""v1 * 2 = [{string.Join("", "", vScaled)}]"");

// Dot product
var dotProduct = v1.DotProduct(v2);
Console.WriteLine($""v1 . v2 = {dotProduct}"");
Console.WriteLine($""  (1*4 + 2*5 + 3*6 = {1*4 + 2*5 + 3*6})"");

// Euclidean norm (length)
var norm = v1.Norm();
Console.WriteLine($""||v1|| = {norm:F4}"");
Console.WriteLine($""  (sqrt(1^2 + 2^2 + 3^2) = sqrt({1+4+9}) = {Math.Sqrt(14):F4})"");

// Min, Max, Sum
Console.WriteLine();
Console.WriteLine($""Min(v1): {v1.Min()}"");
Console.WriteLine($""Max(v1): {v1.Max()}"");
Console.WriteLine($""Sum(v1): {v1.Sum()}"");
Console.WriteLine($""Mean(v1): {v1.Mean():F4}"");
"
                }
            },

            ["Distance Metrics"] = new()
            {
                new CodeExample
                {
                    Id = "euclidean-distance",
                    Name = "Euclidean Distance",
                    Description = "Measure straight-line distance between points",
                    Difficulty = "Beginner",
                    Tags = ["distance", "metrics", "euclidean", "clustering"],
                    Code = @"// Distance Metrics with AiDotNet
using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Tensors.LinearAlgebra;

// Create two points
var point1 = new Vector<double>(new double[] { 0.0, 0.0 });
var point2 = new Vector<double>(new double[] { 3.0, 4.0 });

Console.WriteLine($""Point 1: ({point1[0]}, {point1[1]})"");
Console.WriteLine($""Point 2: ({point2[0]}, {point2[1]})"");
Console.WriteLine();

// Euclidean Distance (straight-line distance)
var euclidean = new EuclideanDistance<double>();
var eucDist = euclidean.Compute(point1, point2);
Console.WriteLine($""Euclidean Distance: {eucDist}"");
Console.WriteLine($""  Formula: sqrt((3-0)^2 + (4-0)^2) = sqrt(9+16) = sqrt(25) = 5"");
Console.WriteLine();

// This is the classic 3-4-5 right triangle!
Console.WriteLine(""This is the famous 3-4-5 right triangle:"");
Console.WriteLine(""  The distance from (0,0) to (3,4) is exactly 5"");
"
                },
                new CodeExample
                {
                    Id = "manhattan-distance",
                    Name = "Manhattan Distance",
                    Description = "City-block distance (sum of absolute differences)",
                    Difficulty = "Beginner",
                    Tags = ["distance", "metrics", "manhattan", "l1"],
                    Code = @"// Manhattan (City-Block) Distance
using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Tensors.LinearAlgebra;

var point1 = new Vector<double>(new double[] { 0.0, 0.0 });
var point2 = new Vector<double>(new double[] { 3.0, 4.0 });

Console.WriteLine($""Point 1: ({point1[0]}, {point1[1]})"");
Console.WriteLine($""Point 2: ({point2[0]}, {point2[1]})"");
Console.WriteLine();

// Manhattan Distance
var manhattan = new ManhattanDistance<double>();
var manDist = manhattan.Compute(point1, point2);
Console.WriteLine($""Manhattan Distance: {manDist}"");
Console.WriteLine($""  Formula: |3-0| + |4-0| = 3 + 4 = 7"");
Console.WriteLine();

Console.WriteLine(""Like walking city blocks:"");
Console.WriteLine(""  You walk 3 blocks east, then 4 blocks north"");
Console.WriteLine(""  Total: 7 blocks (not 5 'as the crow flies')"");
"
                },
                new CodeExample
                {
                    Id = "cosine-distance",
                    Name = "Cosine Similarity",
                    Description = "Measure angle between vectors (great for text/embeddings)",
                    Difficulty = "Intermediate",
                    Tags = ["distance", "metrics", "cosine", "similarity"],
                    Code = @"// Cosine Similarity/Distance
using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Tensors.LinearAlgebra;

// Cosine similarity measures the angle between vectors
// Useful for text embeddings where direction matters more than magnitude

var vec1 = new Vector<double>(new double[] { 1.0, 0.0 });  // East direction
var vec2 = new Vector<double>(new double[] { 0.0, 1.0 });  // North direction
var vec3 = new Vector<double>(new double[] { 1.0, 1.0 });  // Northeast

Console.WriteLine(""Cosine Distance measures angle between vectors"");
Console.WriteLine($""vec1 (East):      [{string.Join("", "", vec1)}]"");
Console.WriteLine($""vec2 (North):     [{string.Join("", "", vec2)}]"");
Console.WriteLine($""vec3 (Northeast): [{string.Join("", "", vec3)}]"");
Console.WriteLine();

var cosine = new CosineDistance<double>();

var dist12 = cosine.Compute(vec1, vec2);
var dist13 = cosine.Compute(vec1, vec3);
var dist23 = cosine.Compute(vec2, vec3);

Console.WriteLine($""Distance(East, North): {dist12:F4}"");
Console.WriteLine($""  (90 degrees apart = maximum distance)"");
Console.WriteLine();
Console.WriteLine($""Distance(East, Northeast): {dist13:F4}"");
Console.WriteLine($""  (45 degrees apart = partial similarity)"");
Console.WriteLine();
Console.WriteLine($""Distance(North, Northeast): {dist23:F4}"");
Console.WriteLine($""  (Also 45 degrees apart)"");
Console.WriteLine();
Console.WriteLine(""Use Cases:"");
Console.WriteLine(""  - Document similarity (TF-IDF vectors)"");
Console.WriteLine(""  - Word/sentence embeddings"");
Console.WriteLine(""  - Recommendation systems"");
"
                },
                new CodeExample
                {
                    Id = "minkowski-distance",
                    Name = "Minkowski Distance",
                    Description = "Generalized distance (includes Euclidean and Manhattan)",
                    Difficulty = "Intermediate",
                    Tags = ["distance", "metrics", "minkowski", "general"],
                    Code = @"// Minkowski Distance (Generalized Lp-norm)
using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Tensors.LinearAlgebra;

var point1 = new Vector<double>(new double[] { 0.0, 0.0 });
var point2 = new Vector<double>(new double[] { 3.0, 4.0 });

Console.WriteLine($""Point 1: ({point1[0]}, {point1[1]})"");
Console.WriteLine($""Point 2: ({point2[0]}, {point2[1]})"");
Console.WriteLine();

Console.WriteLine(""Minkowski Distance with different p values:"");
Console.WriteLine(""  d(x,y) = (sum(|xi - yi|^p))^(1/p)"");
Console.WriteLine();

// p=1 is Manhattan distance
var mink1 = new MinkowskiDistance<double>(1.0);
Console.WriteLine($""p=1 (Manhattan): {mink1.Compute(point1, point2):F4}"");

// p=2 is Euclidean distance
var mink2 = new MinkowskiDistance<double>(2.0);
Console.WriteLine($""p=2 (Euclidean): {mink2.Compute(point1, point2):F4}"");

// p=3
var mink3 = new MinkowskiDistance<double>(3.0);
Console.WriteLine($""p=3:             {mink3.Compute(point1, point2):F4}"");

// Higher p approaches Chebyshev (max difference)
var mink10 = new MinkowskiDistance<double>(10.0);
Console.WriteLine($""p=10:            {mink10.Compute(point1, point2):F4}"");

Console.WriteLine();
Console.WriteLine(""As p increases, the distance approaches max(|x2-x1|, |y2-y1|) = 4"");
"
                },
                new CodeExample
                {
                    Id = "chebyshev-distance",
                    Name = "Chebyshev Distance",
                    Description = "Maximum difference along any dimension",
                    Difficulty = "Intermediate",
                    Tags = ["distance", "metrics", "chebyshev", "chess"],
                    Code = @"// Chebyshev Distance (Maximum/Chessboard Distance)
using AiDotNet.Clustering.DistanceMetrics;
using AiDotNet.Tensors.LinearAlgebra;

var point1 = new Vector<double>(new double[] { 0.0, 0.0 });
var point2 = new Vector<double>(new double[] { 3.0, 7.0 });

Console.WriteLine($""Point 1: ({point1[0]}, {point1[1]})"");
Console.WriteLine($""Point 2: ({point2[0]}, {point2[1]})"");
Console.WriteLine();

var chebyshev = new ChebyshevDistance<double>();
var dist = chebyshev.Compute(point1, point2);

Console.WriteLine($""Chebyshev Distance: {dist}"");
Console.WriteLine($""  Formula: max(|3-0|, |7-0|) = max(3, 7) = 7"");
Console.WriteLine();
Console.WriteLine(""Also called 'Chessboard Distance':"");
Console.WriteLine(""  - A king can move diagonally"");
Console.WriteLine(""  - From (0,0) to (3,7), it takes 7 moves"");
Console.WriteLine(""  - Move diagonally 3 times, then up 4 more times"");
Console.WriteLine();
Console.WriteLine(""Use Cases:"");
Console.WriteLine(""  - Game AI (chess, checkers)"");
Console.WriteLine(""  - Warehouse robot navigation"");
Console.WriteLine(""  - Image processing (maximum color difference)"");
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
                    Code = @"// Standard Scaling with AiDotNet
using AiDotNet.Preprocessing.Scalers;
using AiDotNet.Tensors.LinearAlgebra;

// Data with very different scales
// Feature 1: Large values (100-300)
// Feature 2: Tiny values (0.001-0.003)
var features = new Matrix<double>(4, 2);
features[0, 0] = 100; features[0, 1] = 0.001;
features[1, 0] = 200; features[1, 1] = 0.002;
features[2, 0] = 150; features[2, 1] = 0.0015;
features[3, 0] = 300; features[3, 1] = 0.003;

Console.WriteLine(""Original Data:"");
Console.WriteLine($""  Feature 1: 100, 200, 150, 300 (large scale)"");
Console.WriteLine($""  Feature 2: 0.001, 0.002, 0.0015, 0.003 (tiny scale)"");
Console.WriteLine();

// Create and fit StandardScaler
var scaler = new StandardScaler<double>();
scaler.Fit(features);

// Transform the data
var scaled = scaler.Transform(features);

Console.WriteLine(""StandardScaler transforms to mean=0, std=1:"");
Console.WriteLine($""  Feature 1 mean: {scaler.Mean![0]:F4}"");
Console.WriteLine($""  Feature 1 std:  {scaler.StandardDeviation![0]:F4}"");
Console.WriteLine($""  Feature 2 mean: {scaler.Mean![1]:F6}"");
Console.WriteLine($""  Feature 2 std:  {scaler.StandardDeviation![1]:F6}"");
Console.WriteLine();
Console.WriteLine(""After scaling, both features are comparable."");
Console.WriteLine(""This helps algorithms converge faster and work better!"");
"
                },
                new CodeExample
                {
                    Id = "minmax-scaler",
                    Name = "Min-Max Scaling",
                    Description = "Scale features to a specified range [0, 1]",
                    Difficulty = "Beginner",
                    Tags = ["preprocessing", "scaling", "minmax"],
                    Code = @"// Min-Max Scaling with AiDotNet
using AiDotNet.Preprocessing.Scalers;
using AiDotNet.Tensors.LinearAlgebra;

// Data with different ranges
var features = new Matrix<double>(4, 2);
features[0, 0] = 10; features[0, 1] = 100;
features[1, 0] = 20; features[1, 1] = 200;
features[2, 0] = 15; features[2, 1] = 150;
features[3, 0] = 30; features[3, 1] = 300;

Console.WriteLine(""Original Data:"");
Console.WriteLine($""  Feature 1: 10, 20, 15, 30 (range 10-30)"");
Console.WriteLine($""  Feature 2: 100, 200, 150, 300 (range 100-300)"");
Console.WriteLine();

// Create MinMaxScaler
var scaler = new MinMaxScaler<double>();
scaler.Fit(features);

// Transform the data
var scaled = scaler.Transform(features);

Console.WriteLine(""MinMaxScaler transforms to [0, 1] range:"");
for (int i = 0; i < scaled.Rows; i++)
{
    Console.WriteLine($""  Row {i}: ({scaled[i, 0]:F2}, {scaled[i, 1]:F2})"");
}
Console.WriteLine();
Console.WriteLine(""Min-Max scaling preserves the relative distribution"");
Console.WriteLine(""while putting all features on the same 0-1 scale."");
"
                },
                new CodeExample
                {
                    Id = "preprocessing-pipeline",
                    Name = "Preprocessing Pipeline",
                    Description = "Chain multiple preprocessing steps together",
                    Difficulty = "Intermediate",
                    Tags = ["preprocessing", "pipeline", "feature-engineering"],
                    Code = @"// Preprocessing Pipeline with AiDotNet
using AiDotNet.Preprocessing;
using AiDotNet.Preprocessing.Scalers;
using AiDotNet.Preprocessing.Imputers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

// Generate sample data with missing values
var features = new Matrix<double>(100, 10);
var rng = RandomHelper.CreateSeededRandom(42);
for (int i = 0; i < 100; i++)
{
    for (int j = 0; j < 10; j++)
    {
        // 10% chance of missing value (represented as NaN)
        features[i, j] = rng.NextDouble() < 0.1 ? double.NaN : rng.NextDouble() * 100;
    }
}

Console.WriteLine(""Preprocessing Pipeline Demonstration"");
Console.WriteLine($""  Original data: {features.Rows} samples, {features.Columns} features"");
Console.WriteLine();

// Step 1: Impute missing values
var imputer = new SimpleImputer<double>(ImputationStrategy.Mean);
imputer.Fit(features);
var imputed = imputer.Transform(features);
Console.WriteLine(""Step 1: Mean Imputation"");
Console.WriteLine(""  Replaced NaN values with feature means"");

// Step 2: Standard scaling
var scaler = new StandardScaler<double>();
scaler.Fit(imputed);
var scaled = scaler.Transform(imputed);
Console.WriteLine();
Console.WriteLine(""Step 2: Standard Scaling"");
Console.WriteLine(""  Transformed to mean=0, std=1"");

Console.WriteLine();
Console.WriteLine(""Pipeline steps executed:"");
Console.WriteLine(""  1. Mean Imputation: Handle missing values"");
Console.WriteLine(""  2. Standard Scaling: Normalize features"");
Console.WriteLine();
Console.WriteLine(""Result: Clean, normalized data ready for ML!"");
"
                }
            },

            ["Anomaly Detection"] = new()
            {
                new CodeExample
                {
                    Id = "isolation-forest",
                    Name = "Time Series Isolation Forest",
                    Description = "Detect anomalies in time series using isolation-based method with AiModelBuilder",
                    Difficulty = "Intermediate",
                    Tags = ["anomaly", "outlier", "isolation-forest", "time-series"],
                    Code = @"// Time Series Isolation Forest with AiModelBuilder
using AiDotNet;
using AiDotNet.TimeSeries.AnomalyDetection;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;

// Create time series data with anomalies
var rng = RandomHelper.CreateSeededRandom(42);
var timeSeriesData = new double[100];

// Normal seasonal pattern with some anomalies
for (int i = 0; i < 100; i++)
{
    timeSeriesData[i] = 10 + 5 * Math.Sin(2 * Math.PI * i / 24) + rng.NextDouble() * 2;
}

// Inject anomalies at specific points
timeSeriesData[25] = 50;  // Spike anomaly
timeSeriesData[50] = -10; // Drop anomaly
timeSeriesData[75] = 45;  // Another spike

// Prepare data for AiModelBuilder
var features = new double[100, 1];
var labels = new double[100];
for (int i = 0; i < 100; i++)
{
    features[i, 0] = timeSeriesData[i];
    labels[i] = timeSeriesData[i];
}

// Configure Isolation Forest options
var options = new TimeSeriesIsolationForestOptions<double>
{
    NumTrees = 100,
    ContaminationRate = 0.05,
    LagFeatures = 5,
    RollingWindowSize = 10,
    UseTrendFeatures = true,
    RandomSeed = 42
};

// Build model with AiModelBuilder facade
var loader = DataLoaders.FromArrays(features, labels);
var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureDataLoader(loader)
    .ConfigureModel(new TimeSeriesIsolationForest<double>(options))
    .BuildAsync();

// Use the trained model for anomaly detection
var timeSeries = new Vector<double>(timeSeriesData);
var model = (TimeSeriesIsolationForest<double>)result.Model;
var anomalyScores = model.DetectAnomalies(timeSeries);
var anomalyIndices = model.GetAnomalyIndices(timeSeries);

Console.WriteLine(""Time Series Isolation Forest Results:"");
Console.WriteLine($""  Total data points: {timeSeriesData.Length}"");
Console.WriteLine($""  Anomalies detected: {anomalyIndices.Count}"");
Console.WriteLine($""  Anomaly indices: [{string.Join("", "", anomalyIndices.Take(10))}]"");

// Show top anomaly scores
var topAnomalies = Enumerable.Range(0, timeSeriesData.Length)
    .OrderByDescending(i => anomalyScores[i])
    .Take(5);

Console.WriteLine(""  Top 5 anomaly scores:"");
foreach (var idx in topAnomalies)
{
    Console.WriteLine($""    Index {idx}: score={anomalyScores[idx]:F3}, value={timeSeriesData[idx]:F2}"");
}
"
                },
                new CodeExample
                {
                    Id = "zscore-outlier",
                    Name = "Z-Score Outlier Removal",
                    Description = "Remove outliers using Z-Score method with AiModelBuilder",
                    Difficulty = "Beginner",
                    Tags = ["anomaly", "outlier", "zscore", "statistics"],
                    Code = @"// Z-Score Outlier Removal with AiModelBuilder
using AiDotNet;
using AiDotNet.OutlierRemoval;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;

// Create dataset with some outliers
var rng = RandomHelper.CreateSeededRandom(42);
var features = new double[50, 2];
var labels = new double[50];

// Normal data points
for (int i = 0; i < 45; i++)
{
    features[i, 0] = 5 + rng.NextDouble() * 2;  // Feature 1: centered around 5
    features[i, 1] = 10 + rng.NextDouble() * 3; // Feature 2: centered around 10
    labels[i] = features[i, 0] * 2 + features[i, 1] * 0.5 + rng.NextDouble();
}

// Inject outliers that would corrupt the model
features[45, 0] = 100; features[45, 1] = 10; labels[45] = 110;
features[46, 0] = 5; features[46, 1] = 200; labels[46] = 205;
features[47, 0] = -50; features[47, 1] = -50; labels[47] = -100;
features[48, 0] = 6; features[48, 1] = 11; labels[48] = 17.5;  // Normal
features[49, 0] = 500; features[49, 1] = 500; labels[49] = 1000;

// Build model with AiModelBuilder + outlier removal
var loader = DataLoaders.FromArrays(features, labels);
var outlierRemover = new ZScoreOutlierRemoval<double, Matrix<double>, Vector<double>>(threshold: 3.0);

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureDataLoader(loader)
    .ConfigureOutlierRemoval(outlierRemover)
    .ConfigureModel(new SimpleRegression<double>())
    .BuildAsync();

Console.WriteLine(""Z-Score Outlier Removal with Regression:"");
Console.WriteLine($""  Original samples: 50"");
Console.WriteLine($""  Model trained after outlier removal"");

// Test predictions with clean data
var testFeatures = new Matrix<double>(new double[,] { { 6.0, 11.0 }, { 5.5, 10.5 }, { 7.0, 12.0 } });
var predictions = result.Model.Predict(testFeatures);

Console.WriteLine(""  Predictions on test data:"");
for (int i = 0; i < predictions.Length; i++)
{
    Console.WriteLine($""    Input: [{testFeatures[i, 0]:F1}, {testFeatures[i, 1]:F1}] -> Prediction: {predictions[i]:F2}"");
}
"
                },
                new CodeExample
                {
                    Id = "iqr-outlier",
                    Name = "IQR Outlier Removal",
                    Description = "Remove outliers using IQR method with AiModelBuilder",
                    Difficulty = "Beginner",
                    Tags = ["anomaly", "outlier", "iqr", "statistics"],
                    Code = @"// IQR Outlier Removal with AiModelBuilder
using AiDotNet;
using AiDotNet.OutlierRemoval;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;

// Create dataset with outliers
var rng = RandomHelper.CreateSeededRandom(42);
var features = new double[60, 3];
var labels = new double[60];

// Generate normal data
for (int i = 0; i < 55; i++)
{
    features[i, 0] = 50 + rng.NextDouble() * 20;  // Range: 50-70
    features[i, 1] = 100 + rng.NextDouble() * 30; // Range: 100-130
    features[i, 2] = 25 + rng.NextDouble() * 10;  // Range: 25-35
    labels[i] = features[i, 0] + features[i, 1] * 0.5 + features[i, 2] * 2;
}

// Add outliers
features[55, 0] = 200; features[55, 1] = 110; features[55, 2] = 30; labels[55] = 500;
features[56, 0] = 60; features[56, 1] = 500; features[56, 2] = 30; labels[56] = 600;
features[57, 0] = 60; features[57, 1] = 110; features[57, 2] = 150; labels[57] = 400;
features[58, 0] = -100; features[58, 1] = -200; features[58, 2] = -50; labels[58] = -500;
features[59, 0] = 60; features[59, 1] = 115; features[59, 2] = 28; labels[59] = 173.5; // Normal

// Build model with AiModelBuilder + IQR outlier removal
var loader = DataLoaders.FromArrays(features, labels);
var outlierRemover = new IQROutlierRemoval<double, Matrix<double>, Vector<double>>(iqrMultiplier: 1.5);

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureDataLoader(loader)
    .ConfigureOutlierRemoval(outlierRemover)
    .ConfigureModel(new SimpleRegression<double>())
    .BuildAsync();

Console.WriteLine(""IQR Outlier Removal with Regression:"");
Console.WriteLine($""  Original samples: 60"");
Console.WriteLine($""  Model trained after IQR outlier removal"");

// Test predictions
var testFeatures = new Matrix<double>(new double[,] { { 60.0, 115.0, 30.0 }, { 55.0, 110.0, 28.0 } });
var predictions = result.Model.Predict(testFeatures);

Console.WriteLine(""  Test predictions:"");
for (int i = 0; i < predictions.Length; i++)
{
    Console.WriteLine($""    Input: [{testFeatures[i, 0]:F0}, {testFeatures[i, 1]:F0}, {testFeatures[i, 2]:F0}] -> Pred: {predictions[i]:F2}"");
}
"
                }
            },

            ["Dimensionality Reduction"] = new()
            {
                new CodeExample
                {
                    Id = "pca",
                    Name = "PCA with AiModelBuilder",
                    Description = "Reduce dimensionality using PCA preprocessing with AiModelBuilder",
                    Difficulty = "Intermediate",
                    Tags = ["pca", "dimensionality", "feature-extraction", "preprocessing"],
                    Code = @"// PCA as Preprocessing Step with AiModelBuilder
using AiDotNet;
using AiDotNet.Preprocessing.DimensionalityReduction;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;

// Create high-dimensional data (100 samples, 20 features)
var rng = RandomHelper.CreateSeededRandom(42);
var features = new double[100, 20];
var labels = new double[100];

// Generate correlated features with a target
for (int i = 0; i < 100; i++)
{
    double base1 = rng.NextDouble() * 10;
    double base2 = rng.NextDouble() * 5;

    // Features with correlations
    features[i, 0] = base1 + rng.NextDouble() * 0.5;
    features[i, 1] = base1 * 0.9 + rng.NextDouble() * 0.5;
    features[i, 2] = base1 * 0.8 + rng.NextDouble() * 0.5;
    features[i, 3] = base2 + rng.NextDouble() * 0.3;
    features[i, 4] = base2 * 1.1 + rng.NextDouble() * 0.3;

    // Independent noise features
    for (int j = 5; j < 20; j++)
        features[i, j] = rng.NextDouble() * 2;

    // Target depends on the base values
    labels[i] = base1 * 2 + base2 * 1.5 + rng.NextDouble();
}

// Build model with PCA preprocessing (reduce 20 features to 5)
var loader = DataLoaders.FromArrays(features, labels);
var pca = new PCA<double>(nComponents: 5);

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureDataLoader(loader)
    .ConfigurePreprocessing(pca)
    .ConfigureModel(new SimpleRegression<double>())
    .BuildAsync();

Console.WriteLine(""PCA with AiModelBuilder Results:"");
Console.WriteLine($""  Original dimensions: 20"");
Console.WriteLine($""  Reduced dimensions: {pca.NComponentsOut}"");

// Show explained variance
if (pca.ExplainedVarianceRatio != null)
{
    double totalVariance = 0;
    Console.WriteLine(""  Explained variance by component:"");
    for (int i = 0; i < Math.Min(5, pca.ExplainedVarianceRatio.Length); i++)
    {
        totalVariance += pca.ExplainedVarianceRatio[i];
        Console.WriteLine($""    PC{i + 1}: {pca.ExplainedVarianceRatio[i]:P2} (cumulative: {totalVariance:P2})"");
    }
}

// Test prediction - input is automatically preprocessed through PCA
var testFeatures = new Matrix<double>(1, 20);
for (int j = 0; j < 20; j++)
    testFeatures[0, j] = rng.NextDouble() * 5;

var predictions = result.Model.Predict(pca.Transform(testFeatures));
Console.WriteLine($""  Test prediction: {predictions[0]:F2}"");
"
                },
                new CodeExample
                {
                    Id = "pca-pipeline",
                    Name = "PCA in Preprocessing Pipeline",
                    Description = "Combine multiple preprocessing steps including PCA with AiModelBuilder",
                    Difficulty = "Intermediate",
                    Tags = ["pca", "pipeline", "preprocessing", "scaling"],
                    Code = @"// PCA in Multi-Step Preprocessing Pipeline
using AiDotNet;
using AiDotNet.Preprocessing;
using AiDotNet.Preprocessing.DimensionalityReduction;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;

// Create high-dimensional data with different scales
var rng = RandomHelper.CreateSeededRandom(42);
var features = new double[80, 15];
var labels = new double[80];

for (int i = 0; i < 80; i++)
{
    // Different scale features
    features[i, 0] = rng.NextDouble() * 1000;  // Large scale
    features[i, 1] = rng.NextDouble() * 500;
    features[i, 2] = rng.NextDouble() * 10;    // Small scale
    features[i, 3] = rng.NextDouble() * 5;

    // Noise features
    for (int j = 4; j < 15; j++)
        features[i, j] = rng.NextDouble() * 2;

    labels[i] = features[i, 0] * 0.01 + features[i, 2] * 2 + rng.NextDouble();
}

// Build with preprocessing pipeline: StandardScaler -> PCA
var loader = DataLoaders.FromArrays(features, labels);

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureDataLoader(loader)
    .ConfigurePreprocessing(pipeline => pipeline
        .Add(new StandardScaler<double>())  // First normalize scales
        .Add(new PCA<double>(nComponents: 5)))  // Then reduce dimensions
    .ConfigureModel(new SimpleRegression<double>())
    .BuildAsync();

Console.WriteLine(""PCA Pipeline with AiModelBuilder:"");
Console.WriteLine($""  Original features: 15"");
Console.WriteLine($""  Pipeline: StandardScaler -> PCA(5 components)"");
Console.WriteLine($""  Model trained successfully"");

// Test prediction
var testFeatures = new Matrix<double>(new double[,] {
    { 500, 250, 5, 2.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }
});
Console.WriteLine($""  Test input (first 4 features): [{testFeatures[0, 0]:F0}, {testFeatures[0, 1]:F0}, {testFeatures[0, 2]:F0}, {testFeatures[0, 3]:F1}]"");
"
                },
                new CodeExample
                {
                    Id = "pca-variance-threshold",
                    Name = "PCA with Variance Threshold",
                    Description = "Automatically determine components to keep based on variance",
                    Difficulty = "Intermediate",
                    Tags = ["pca", "variance", "dimensionality", "auto"],
                    Code = @"// PCA with Variance Ratio Threshold via AiModelBuilder
using AiDotNet;
using AiDotNet.Preprocessing.DimensionalityReduction;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;

// Create data with varying importance features
var rng = RandomHelper.CreateSeededRandom(42);
var features = new double[80, 15];
var labels = new double[80];

for (int i = 0; i < 80; i++)
{
    // High variance features (important)
    features[i, 0] = rng.NextDouble() * 100;
    features[i, 1] = rng.NextDouble() * 80;
    features[i, 2] = rng.NextDouble() * 60;

    // Medium variance features
    features[i, 3] = rng.NextDouble() * 20;
    features[i, 4] = rng.NextDouble() * 15;

    // Low variance features (noise)
    for (int j = 5; j < 15; j++)
        features[i, j] = 50 + rng.NextDouble() * 2;

    labels[i] = features[i, 0] * 0.5 + features[i, 1] * 0.3 + rng.NextDouble() * 5;
}

// Build with PCA keeping 95% variance
var loader = DataLoaders.FromArrays(features, labels);
var pca95 = new PCA<double>(varianceRatio: 0.95);

var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureDataLoader(loader)
    .ConfigurePreprocessing(pca95)
    .ConfigureModel(new SimpleRegression<double>())
    .BuildAsync();

Console.WriteLine(""PCA with 95% Variance Threshold:"");
Console.WriteLine($""  Original features: 15"");
Console.WriteLine($""  Components kept: {pca95.NComponentsOut}"");
Console.WriteLine($""  Automatic reduction based on explained variance"");

// Show variance distribution
if (pca95.ExplainedVarianceRatio != null)
{
    Console.WriteLine(""  Variance by component:"");
    double cumulative = 0;
    for (int i = 0; i < Math.Min(5, pca95.ExplainedVarianceRatio.Length); i++)
    {
        cumulative += pca95.ExplainedVarianceRatio[i];
        Console.WriteLine($""    PC{i + 1}: {pca95.ExplainedVarianceRatio[i]:P1} | Cumulative: {cumulative:P1}"");
    }
}

// Test prediction
var testFeatures = new Matrix<double>(new double[,] {
    { 50, 40, 30, 10, 7.5, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50 }
});
var transformed = pca95.Transform(testFeatures);
var predictions = result.Model.Predict(transformed);
Console.WriteLine($""  Test prediction: {predictions[0]:F2}"");
"
                }
            },

            ["NLP & Text Processing"] = new()
            {
                new CodeExample
                {
                    Id = "tfidf-vectorizer",
                    Name = "TF-IDF Text Vectorization",
                    Description = "Convert text documents to TF-IDF weighted vectors",
                    Difficulty = "Intermediate",
                    Tags = ["nlp", "text", "tfidf", "vectorization"],
                    Code = @"// TF-IDF Text Vectorization for Classification
using AiDotNet;
using AiDotNet.Preprocessing.TextVectorizers;
using AiDotNet.Classification;
using AiDotNet.Tensors.LinearAlgebra;

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

// Create TF-IDF vectorizer
var tfidf = new TfidfVectorizer<double>(
    maxFeatures: 50,
    minDf: 1,
    lowercase: true
);

// Transform documents to feature matrix
var features = tfidf.FitTransform(documents);

Console.WriteLine(""TF-IDF Vectorization Results:"");
Console.WriteLine($""  Documents: {documents.Length}"");
Console.WriteLine($""  Vocabulary size: {tfidf.Vocabulary?.Count ?? 0}"");
Console.WriteLine($""  Feature matrix shape: {features.Rows} x {features.Columns}"");

// Now use with AiModelBuilder
var loader = DataLoaders.FromMatrixVector(features, new Vector<double>(labels));
var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureDataLoader(loader)
    .ConfigureModel(new LogisticRegression<double>())
    .BuildAsync();

Console.WriteLine();
Console.WriteLine(""  Model trained on TF-IDF features"");

// Classify new text
var newDoc = new[] { ""This is an excellent product!"" };
var newFeatures = tfidf.Transform(newDoc);
var prediction = result.Model.Predict(newFeatures);

Console.WriteLine($""  New review: '{newDoc[0]}'"");
Console.WriteLine($""  Prediction: {(prediction[0] > 0.5 ? ""Positive"" : ""Negative"")}"");
"
                },
                new CodeExample
                {
                    Id = "word2vec",
                    Name = "Word2Vec Embeddings",
                    Description = "Train Word2Vec word embeddings with AiModelBuilder",
                    Difficulty = "Intermediate",
                    Tags = ["nlp", "embeddings", "word2vec", "neural-network"],
                    Code = @"// Word2Vec Word Embeddings with AiModelBuilder
using AiDotNet;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tokenization.Algorithms;
using AiDotNet.Tensors.LinearAlgebra;

// Training corpus - tokenized sentences
var sentences = new string[]
{
    ""The cat sat on the mat"",
    ""The dog ran in the park"",
    ""Cats and dogs are pets"",
    ""Machine learning is fascinating"",
    ""Deep learning uses neural networks"",
    ""Neural networks learn patterns"",
    ""Dogs and cats are animals"",
    ""The park has many dogs""
};

// Simple tokenization
var allTokens = new List<int>();
var wordToId = new Dictionary<string, int>();
int nextId = 0;

foreach (var sentence in sentences)
{
    foreach (var word in sentence.ToLower().Split(' '))
    {
        if (!wordToId.ContainsKey(word))
        {
            wordToId[word] = nextId++;
        }
        allTokens.Add(wordToId[word]);
    }
    allTokens.Add(-1); // Sentence separator
}

var idToWord = wordToId.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);

Console.WriteLine(""Word2Vec Training:"");
Console.WriteLine($""  Vocabulary size: {wordToId.Count}"");
Console.WriteLine($""  Total tokens: {allTokens.Count(t => t >= 0)}"");
Console.WriteLine($""  Sentences: {sentences.Length}"");
Console.WriteLine();

// Create Word2Vec architecture
var architecture = new NeuralNetworkArchitecture<double>(
    inputType: InputType.OneDimensional,
    taskType: NeuralNetworkTaskType.Regression,
    complexity: NetworkComplexity.Medium,
    inputSize: wordToId.Count,  // Vocabulary size
    outputSize: 50);  // Embedding dimension

Console.WriteLine(""Word2Vec Configuration:"");
Console.WriteLine($""  Embedding dimension: 50"");
Console.WriteLine($""  Window size: 2"");
Console.WriteLine($""  Architecture: Skip-gram"");
Console.WriteLine();

Console.WriteLine(""Sample vocabulary:"");
foreach (var (word, id) in wordToId.Take(10))
{
    Console.WriteLine($""  '{word}' -> ID {id}"");
}
"
                },
                new CodeExample
                {
                    Id = "tokenizer-pretrained",
                    Name = "Pretrained Tokenizer",
                    Description = "Use HuggingFace pretrained tokenizers with AiModelBuilder",
                    Difficulty = "Intermediate",
                    Tags = ["nlp", "tokenizer", "huggingface", "bert"],
                    Code = @"// Pretrained Tokenizer with AiModelBuilder
using AiDotNet;
using AiDotNet.Tokenization.Configuration;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

// Sample text for tokenization
var texts = new string[]
{
    ""Machine learning is transforming industries."",
    ""Natural language processing enables AI to understand text."",
    ""Deep learning models require large datasets.""
};

Console.WriteLine(""Pretrained Tokenizer Demo:"");
Console.WriteLine();

// Configure tokenizer from pretrained model
var architecture = new NeuralNetworkArchitecture<double>(
    inputType: InputType.OneDimensional,
    taskType: NeuralNetworkTaskType.BinaryClassification,
    complexity: NetworkComplexity.Medium,
    inputSize: 128,
    outputSize: 2);

// Create dummy training data (tokenized sequences)
var rng = RandomHelper.CreateSeededRandom(42);
var features = new double[20, 128];
var labels = new double[20];
for (int i = 0; i < 20; i++)
{
    for (int j = 0; j < 128; j++)
        features[i, j] = rng.Next(0, 1000);  // Simulated token IDs
    labels[i] = i % 2;  // Binary labels
}

var loader = DataLoaders.FromArrays(features, labels);

// Build with tokenizer configuration
var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureDataLoader(loader)
    .ConfigureTokenizerFromPretrained(PretrainedTokenizerModel.BertBaseUncased)
    .ConfigureModel(new FeedForwardNeuralNetwork<double>(architecture))
    .BuildAsync();

Console.WriteLine(""Tokenizer Configuration:"");
Console.WriteLine($""  Model: BERT Base Uncased"");
Console.WriteLine($""  Vocabulary size: ~30,000 tokens"");
Console.WriteLine($""  Special tokens: [CLS], [SEP], [PAD], [MASK]"");
Console.WriteLine();

// Demonstrate tokenization
Console.WriteLine(""Sample text tokenization:"");
foreach (var text in texts.Take(2))
{
    var encoded = result.Encode(text);
    Console.WriteLine($""  Text: '{text.Substring(0, Math.Min(40, text.Length))}...'"");
    Console.WriteLine($""  Token count: {encoded.InputIds.Count}"");
    Console.WriteLine($""  First 5 tokens: [{string.Join("", "", encoded.InputIds.Take(5))}...]"");
    Console.WriteLine();
}
"
                },
                new CodeExample
                {
                    Id = "text-classification-nn",
                    Name = "Neural Text Classification",
                    Description = "Text classification using neural networks with tokenization",
                    Difficulty = "Advanced",
                    Tags = ["nlp", "classification", "neural-network", "embedding"],
                    Code = @"// Neural Network Text Classification
using AiDotNet;
using AiDotNet.NeuralNetworks;
using AiDotNet.Preprocessing.TextVectorizers;
using AiDotNet.Tensors.LinearAlgebra;

// Sample documents for sentiment classification
var trainDocs = new string[]
{
    ""This product exceeded my expectations"",
    ""Amazing quality and fast delivery"",
    ""Best purchase I ever made"",
    ""Absolutely wonderful experience"",
    ""Terrible quality, waste of money"",
    ""Disappointed with the service"",
    ""Would not recommend to anyone"",
    ""Worst product I have bought""
};
var trainLabels = new double[] { 1, 1, 1, 1, 0, 0, 0, 0 };

// Vectorize with TF-IDF
var tfidf = new TfidfVectorizer<double>(maxFeatures: 30);
var features = tfidf.FitTransform(trainDocs);

Console.WriteLine(""Text Classification with Neural Network:"");
Console.WriteLine($""  Training documents: {trainDocs.Length}"");
Console.WriteLine($""  TF-IDF features: {features.Columns}"");
Console.WriteLine();

// Create neural network architecture
var architecture = new NeuralNetworkArchitecture<double>(
    inputType: InputType.OneDimensional,
    taskType: NeuralNetworkTaskType.BinaryClassification,
    complexity: NetworkComplexity.Low,
    inputSize: features.Columns,
    outputSize: 1);

// Build and train with AiModelBuilder
var loader = DataLoaders.FromMatrixVector(features, new Vector<double>(trainLabels));
var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureDataLoader(loader)
    .ConfigureModel(new FeedForwardNeuralNetwork<double>(architecture))
    .BuildAsync();

Console.WriteLine(""Neural Network Architecture:"");
Console.WriteLine($""  Input: {features.Columns} TF-IDF features"");
Console.WriteLine($""  Hidden: Dense layers"");
Console.WriteLine($""  Output: Binary classification"");
Console.WriteLine();

// Test on new documents
var testDocs = new string[] { ""Great product, highly recommend!"", ""Total waste of money"" };
var testFeatures = tfidf.Transform(testDocs);
var predictions = result.Model.Predict(testFeatures);

Console.WriteLine(""Test Predictions:"");
for (int i = 0; i < testDocs.Length; i++)
{
    Console.WriteLine($""  '{testDocs[i]}'"");
    Console.WriteLine($""    -> {(predictions[i] > 0.5 ? ""Positive"" : ""Negative"")} ({predictions[i]:F3})"");
}
"
                },
                new CodeExample
                {
                    Id = "named-entity-recognition",
                    Name = "Named Entity Recognition Setup",
                    Description = "Sequence labeling for NER using LSTM neural network",
                    Difficulty = "Advanced",
                    Tags = ["nlp", "ner", "sequence-labeling", "neural-network"],
                    Code = @"// Named Entity Recognition with LSTM Neural Network
using AiDotNet;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

// NER label encoding
// O=0, B-PER=1, I-PER=2, B-ORG=3, I-ORG=4, B-LOC=5, I-LOC=6
var labelMap = new Dictionary<string, int>
{
    { ""O"", 0 }, { ""B-PER"", 1 }, { ""I-PER"", 2 },
    { ""B-ORG"", 3 }, { ""I-ORG"", 4 }, { ""B-LOC"", 5 }, { ""I-LOC"", 6 }
};

// Sample NER data - tokens and their entity labels
var sentences = new string[][]
{
    new[] { ""John"", ""works"", ""at"", ""Microsoft"", ""in"", ""Seattle"" },
    new[] { ""Apple"", ""CEO"", ""Tim"", ""Cook"", ""announced"", ""products"" }
};
var entityLabels = new string[][]
{
    new[] { ""B-PER"", ""O"", ""O"", ""B-ORG"", ""O"", ""B-LOC"" },
    new[] { ""B-ORG"", ""O"", ""B-PER"", ""I-PER"", ""O"", ""O"" }
};

// Build vocabulary from training data
var wordToId = new Dictionary<string, int>();
int vocabSize = 0;
foreach (var sentence in sentences)
    foreach (var word in sentence)
        if (!wordToId.ContainsKey(word.ToLower()))
            wordToId[word.ToLower()] = vocabSize++;

// Convert to feature matrix (one-hot encoding of words)
// Each row is a token, columns are vocabulary + label
int seqLength = 6; // Fixed sequence length
var features = new double[sentences.Length * seqLength, vocabSize];
var labels = new double[sentences.Length * seqLength];

for (int s = 0; s < sentences.Length; s++)
{
    for (int t = 0; t < seqLength && t < sentences[s].Length; t++)
    {
        int row = s * seqLength + t;
        int wordId = wordToId[sentences[s][t].ToLower()];
        features[row, wordId] = 1.0;  // One-hot encoding
        labels[row] = labelMap[entityLabels[s][t]];
    }
}

Console.WriteLine(""NER with LSTM Neural Network:"");
Console.WriteLine($""  Vocabulary size: {vocabSize}"");
Console.WriteLine($""  Entity classes: {labelMap.Count}"");
Console.WriteLine($""  Training sequences: {sentences.Length}"");
Console.WriteLine();

// Create LSTM architecture for sequence labeling
var architecture = new NeuralNetworkArchitecture<double>(
    inputType: InputType.OneDimensional,
    taskType: NeuralNetworkTaskType.MultiClassClassification,
    complexity: NetworkComplexity.Medium,
    inputSize: vocabSize,
    outputSize: labelMap.Count);

// Build with AiModelBuilder
var loader = DataLoaders.FromArrays(features, labels);
var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureDataLoader(loader)
    .ConfigureModel(new LSTMNeuralNetwork<double>(architecture))
    .BuildAsync();

Console.WriteLine(""Model Architecture:"");
Console.WriteLine($""  Input: One-hot encoded tokens ({vocabSize} dim)"");
Console.WriteLine($""  LSTM: Hidden state processing"");
Console.WriteLine($""  Output: {labelMap.Count} entity classes"");
Console.WriteLine();

// Demonstrate prediction on training data
Console.WriteLine(""Sample Predictions:"");
Console.WriteLine($""  Sentence: {string.Join("" "", sentences[0])}"");
Console.WriteLine($""  Expected: {string.Join("" "", entityLabels[0])}"");
"
                }
            },

            ["Audio Processing"] = new()
            {
                new CodeExample
                {
                    Id = "speech-recognition",
                    Name = "Speech Recognition",
                    Description = "Configure Whisper speech recognition with AiModelBuilder",
                    Difficulty = "Advanced",
                    Tags = ["audio", "speech", "whisper", "asr"],
                    Code = @"// Whisper Speech Recognition with AiModelBuilder
using AiDotNet;
using AiDotNet.Audio.Whisper;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

// Whisper model configuration
var sampleRate = 16000;
var numMels = 80;  // Mel spectrogram channels
var maxSeconds = 30;  // Whisper processes 30-second chunks

Console.WriteLine(""Whisper Speech Recognition Configuration:"");
Console.WriteLine($""  Sample Rate: {sampleRate} Hz"");
Console.WriteLine($""  Mel Channels: {numMels}"");
Console.WriteLine($""  Max Audio Length: {maxSeconds} seconds"");
Console.WriteLine();

// Create architecture for Whisper encoder-decoder
var architecture = new NeuralNetworkArchitecture<double>(
    inputType: InputType.TwoDimensional,
    taskType: NeuralNetworkTaskType.SequenceToSequence,
    complexity: NetworkComplexity.Medium,
    inputSize: numMels * 3000,  // Mel frames for 30s
    outputSize: 51865);  // Whisper vocabulary size

// Create Whisper model in native training mode
var whisper = new WhisperModel<double>(
    architecture: architecture,
    modelSize: WhisperModelSize.Base,
    language: ""en"",
    sampleRate: sampleRate,
    numMels: numMels,
    maxAudioLengthSeconds: maxSeconds);

Console.WriteLine(""Whisper Model:"");
Console.WriteLine($""  Model Size: Base (74M parameters)"");
Console.WriteLine($""  Mode: Native (trainable)"");
Console.WriteLine($""  Language: English"");
Console.WriteLine();

// Simulated audio samples
var rng = RandomHelper.CreateSeededRandom(42);
var duration = 5.0;
var numSamples = (int)(sampleRate * duration);
var audioFeatures = new double[1, numSamples];
for (int i = 0; i < numSamples; i++)
    audioFeatures[0, i] = rng.NextDouble() * 2 - 1;

// Build with AiModelBuilder
var loader = DataLoaders.FromArrays(audioFeatures, new double[] { 0 });
var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureDataLoader(loader)
    .ConfigureModel(whisper)
    .BuildAsync();

Console.WriteLine(""Model built successfully!"");
Console.WriteLine();
Console.WriteLine(""Note: For production transcription, use ONNX mode:"");
Console.WriteLine(""  var whisper = new WhisperModel<double>("");
Console.WriteLine(""      architecture,"");
Console.WriteLine(""      encoderPath: 'whisper-encoder.onnx',"");
Console.WriteLine(""      decoderPath: 'whisper-decoder.onnx');"");
"
                },
                new CodeExample
                {
                    Id = "audio-classification",
                    Name = "Audio Classification",
                    Description = "Classify audio clips using MFCC features with AiModelBuilder",
                    Difficulty = "Intermediate",
                    Tags = ["audio", "classification", "mfcc"],
                    Code = @"// Audio Classification with MFCC Features using AiModelBuilder
using AiDotNet;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

Console.WriteLine(""Audio Classification with MFCC Features"");
Console.WriteLine();

// Simulated MFCC feature extraction
// In practice: audio -> STFT -> Mel filterbank -> DCT -> MFCC
var numMfccCoefficients = 40;
var numSamples = 100;
var numClasses = 3;  // Speech, Music, Noise

// Generate training data (simulated MFCC features)
var rng = RandomHelper.CreateSeededRandom(42);
var features = new double[numSamples, numMfccCoefficients];
var labels = new double[numSamples];

for (int i = 0; i < numSamples; i++)
{
    // Assign class labels
    labels[i] = i % numClasses;

    // Generate MFCC-like features with class-specific patterns
    for (int j = 0; j < numMfccCoefficients; j++)
    {
        double baseValue = (labels[i] + 1) * 0.1;  // Class-dependent offset
        features[i, j] = baseValue + rng.NextDouble() * 0.5;
    }
}

Console.WriteLine(""Training Data:"");
Console.WriteLine($""  Samples: {numSamples}"");
Console.WriteLine($""  MFCC Features: {numMfccCoefficients}"");
Console.WriteLine($""  Classes: Speech (0), Music (1), Noise (2)"");
Console.WriteLine();

// Create neural network architecture
var architecture = new NeuralNetworkArchitecture<double>(
    inputType: InputType.OneDimensional,
    taskType: NeuralNetworkTaskType.MultiClassClassification,
    complexity: NetworkComplexity.Medium,
    inputSize: numMfccCoefficients,
    outputSize: numClasses);

// Build with AiModelBuilder
var loader = DataLoaders.FromArrays(features, labels);
var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureDataLoader(loader)
    .ConfigureModel(new FeedForwardNeuralNetwork<double>(architecture))
    .BuildAsync();

Console.WriteLine(""Model Architecture:"");
Console.WriteLine($""  Input: {numMfccCoefficients} MFCC coefficients"");
Console.WriteLine($""  Hidden: Dense layers with ReLU"");
Console.WriteLine($""  Output: {numClasses} classes (softmax)"");
Console.WriteLine();

// Test prediction
var testFeatures = new Matrix<double>(1, numMfccCoefficients);
for (int j = 0; j < numMfccCoefficients; j++)
    testFeatures[0, j] = 0.2 + rng.NextDouble() * 0.3;  // Music-like pattern

var prediction = result.Model.Predict(testFeatures);
string[] classNames = { ""Speech"", ""Music"", ""Noise"" };
Console.WriteLine(""Test Prediction:"");
for (int c = 0; c < numClasses; c++)
    Console.WriteLine($""  {classNames[c]}: {prediction[c]:P1}"");
"
                },
                new CodeExample
                {
                    Id = "text-to-speech",
                    Name = "Text-to-Speech",
                    Description = "Configure TTS model for speech synthesis with AiModelBuilder",
                    Difficulty = "Advanced",
                    Tags = ["audio", "tts", "speech-synthesis"],
                    Code = @"// Text-to-Speech with AiModelBuilder
using AiDotNet;
using AiDotNet.Audio.TextToSpeech;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

Console.WriteLine(""Text-to-Speech Configuration"");
Console.WriteLine();

// TTS model parameters
var sampleRate = 22050;
var numMels = 80;
var maxPhonemes = 256;

// Create architecture for TTS
var architecture = new NeuralNetworkArchitecture<double>(
    inputType: InputType.OneDimensional,
    taskType: NeuralNetworkTaskType.SequenceToSequence,
    complexity: NetworkComplexity.Medium,
    inputSize: maxPhonemes,
    outputSize: numMels);

// Create TTS model in native training mode
var tts = new TtsModel<double>(
    architecture: architecture,
    sampleRate: sampleRate,
    numMels: numMels,
    speakingRate: 1.0,
    pitchShift: 0.0,
    energy: 1.0,
    hiddenDim: 256,
    numHeads: 4,
    numEncoderLayers: 4,
    numDecoderLayers: 4,
    maxPhonemeLength: maxPhonemes);

Console.WriteLine(""TTS Model Configuration:"");
Console.WriteLine($""  Sample Rate: {sampleRate} Hz"");
Console.WriteLine($""  Mel Channels: {numMels}"");
Console.WriteLine($""  Max Phoneme Length: {maxPhonemes}"");
Console.WriteLine($""  Hidden Dimension: 256"");
Console.WriteLine($""  Attention Heads: 4"");
Console.WriteLine();

// Create dummy training data (phoneme sequences -> mel spectrograms)
var rng = RandomHelper.CreateSeededRandom(42);
var features = new double[10, maxPhonemes];
var labels = new double[10];
for (int i = 0; i < 10; i++)
{
    for (int j = 0; j < maxPhonemes && j < 50; j++)
        features[i, j] = rng.Next(0, 100);  // Phoneme IDs
    labels[i] = i % 2;
}

// Build with AiModelBuilder
var loader = DataLoaders.FromArrays(features, labels);
var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureDataLoader(loader)
    .ConfigureModel(tts)
    .BuildAsync();

Console.WriteLine(""Model built successfully!"");
Console.WriteLine();

var sampleText = ""Hello, welcome to AiDotNet!"";
Console.WriteLine($""Sample text: '{sampleText}'"");
Console.WriteLine();
Console.WriteLine(""Pipeline:"");
Console.WriteLine(""  1. Text -> Phoneme tokenization"");
Console.WriteLine(""  2. Phonemes -> Encoder (attention)"");
Console.WriteLine(""  3. Decoder -> Mel spectrogram"");
Console.WriteLine(""  4. Vocoder -> Audio waveform"");
"
                },
                new CodeExample
                {
                    Id = "music-generation",
                    Name = "Music Generation",
                    Description = "Configure MusicGen model for AI music synthesis with AiModelBuilder",
                    Difficulty = "Advanced",
                    Tags = ["audio", "music", "generation", "ai"],
                    Code = @"// Music Generation with MusicGen using AiModelBuilder
using AiDotNet;
using AiDotNet.Audio.MusicGen;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;

Console.WriteLine(""MusicGen Configuration"");
Console.WriteLine();

var prompt = ""A calm piano melody with soft strings"";
var sampleRate = 32000;
var duration = 10.0;

Console.WriteLine($""Prompt: '{prompt}'"");
Console.WriteLine($""Sample Rate: {sampleRate} Hz"");
Console.WriteLine($""Target Duration: {duration} seconds"");
Console.WriteLine();

// Create architecture for MusicGen
var architecture = new NeuralNetworkArchitecture<double>(
    inputType: InputType.OneDimensional,
    taskType: NeuralNetworkTaskType.SequenceToSequence,
    complexity: NetworkComplexity.High,
    inputSize: 512,  // Text embedding dimension
    outputSize: 2048);  // Audio token codebook size

// Create MusicGen options
var options = new MusicGenOptions
{
    ModelSize = MusicGenSize.Medium,
    SampleRate = sampleRate,
    DefaultDurationSeconds = duration,
    NumCodebooks = 4,
    CodebookSize = 2048,
    MaxTokens = 1500
};

// Create MusicGen model in native mode
var musicGen = new MusicGenModel<double>(
    architecture: architecture,
    options: options);

Console.WriteLine(""MusicGen Model:"");
Console.WriteLine($""  Model Size: {options.ModelSize}"");
Console.WriteLine($""  Codebooks: {options.NumCodebooks}"");
Console.WriteLine($""  Codebook Size: {options.CodebookSize}"");
Console.WriteLine($""  Max Tokens: {options.MaxTokens}"");
Console.WriteLine();

// Create training data (text embeddings -> audio tokens)
var rng = RandomHelper.CreateSeededRandom(42);
var features = new double[10, 512];
var labels = new double[10];
for (int i = 0; i < 10; i++)
{
    for (int j = 0; j < 512; j++)
        features[i, j] = rng.NextDouble() * 2 - 1;
    labels[i] = i % 100;  // Audio token targets
}

// Build with AiModelBuilder
var loader = DataLoaders.FromArrays(features, labels);
var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureDataLoader(loader)
    .ConfigureModel(musicGen)
    .BuildAsync();

Console.WriteLine(""Model built successfully!"");
Console.WriteLine();
Console.WriteLine(""Generation Pipeline:"");
Console.WriteLine(""  1. Text prompt -> T5 encoder -> text embeddings"");
Console.WriteLine(""  2. Transformer LM generates audio tokens"");
Console.WriteLine(""  3. EnCodec decoder converts tokens -> waveform"");
Console.WriteLine();

var numSamples = (int)(duration * sampleRate);
Console.WriteLine(""Expected Output:"");
Console.WriteLine($""  Audio samples: {numSamples:N0}"");
Console.WriteLine($""  File size: ~{numSamples * 2 / 1024:N0} KB (16-bit)"");
"
                }
            },

            ["Reinforcement Learning"] = new()
            {
                new CodeExample
                {
                    Id = "dqn-agent",
                    Name = "DQN Agent",
                    Description = "Deep Q-Network for discrete action spaces with AiModelBuilder",
                    Difficulty = "Advanced",
                    Tags = ["rl", "dqn", "deep-learning", "q-learning"],
                    Code = @"// DQN Agent with AiModelBuilder
using AiDotNet;
using AiDotNet.Models.Options;
using AiDotNet.ReinforcementLearning.Agents;
using AiDotNet.Tensors.LinearAlgebra;

// Define environment (e.g., CartPole)
var stateSize = 4;   // [cart position, cart velocity, pole angle, pole velocity]
var actionSize = 2;  // [push left, push right]

Console.WriteLine(""DQN Agent Configuration"");
Console.WriteLine($""  State size: {stateSize}"");
Console.WriteLine($""  Action size: {actionSize}"");
Console.WriteLine();

// Configure DQN options
var options = new DQNOptions<double>
{
    StateSize = stateSize,
    ActionSize = actionSize,
    HiddenLayers = new List<int> { 128, 128 },
    EpsilonStart = 1.0,
    EpsilonEnd = 0.01,
    EpsilonDecay = 0.995,
    BatchSize = 64,
    ReplayBufferSize = 10000,
    TargetUpdateFrequency = 100,
    WarmupSteps = 1000,
    Seed = 42
};

// Create DQN agent
var agent = new DQNAgent<double>(options);

Console.WriteLine(""DQN Options:"");
Console.WriteLine($""  Hidden layers: [128, 128]"");
Console.WriteLine($""  Epsilon: {options.EpsilonStart} -> {options.EpsilonEnd}"");
Console.WriteLine($""  Batch size: {options.BatchSize}"");
Console.WriteLine($""  Replay buffer: {options.ReplayBufferSize}"");
Console.WriteLine();

// Create training data for builder
var features = new double[10, stateSize];
var labels = new double[10];
var rng = RandomHelper.CreateSeededRandom(42);
for (int i = 0; i < 10; i++)
{
    for (int j = 0; j < stateSize; j++)
        features[i, j] = rng.NextDouble() * 2 - 1;
    labels[i] = rng.Next(0, actionSize);
}

// Build with AiModelBuilder
var loader = DataLoaders.FromArrays(features, labels);
var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureDataLoader(loader)
    .ConfigureModel(agent)
    .BuildAsync();

Console.WriteLine(""Agent built successfully!"");
Console.WriteLine();
Console.WriteLine(""Training Loop:"");
Console.WriteLine(""  1. state = env.Reset()"");
Console.WriteLine(""  2. action = agent.SelectAction(state)"");
Console.WriteLine(""  3. next_state, reward, done = env.Step(action)"");
Console.WriteLine(""  4. agent.Remember(state, action, reward, next_state, done)"");
Console.WriteLine(""  5. agent.Replay()  // Train on minibatch"");
"
                },
                new CodeExample
                {
                    Id = "ppo-agent",
                    Name = "PPO Agent",
                    Description = "Proximal Policy Optimization for continuous control with AiModelBuilder",
                    Difficulty = "Advanced",
                    Tags = ["rl", "ppo", "policy-gradient", "continuous"],
                    Code = @"// PPO Agent with AiModelBuilder
using AiDotNet;
using AiDotNet.Models.Options;
using AiDotNet.ReinforcementLearning.Agents;
using AiDotNet.Tensors.LinearAlgebra;

// Continuous action space environment (e.g., MuJoCo Hopper)
var stateSize = 8;
var actionSize = 2;

Console.WriteLine(""PPO Agent Configuration"");
Console.WriteLine($""  State size: {stateSize}"");
Console.WriteLine($""  Action size: {actionSize} (continuous)"");
Console.WriteLine();

// Configure PPO options (T values use numeric operations internally)
var options = new PPOOptions<double>
{
    StateSize = stateSize,
    ActionSize = actionSize,
    IsContinuous = true,
    PolicyHiddenLayers = new List<int> { 64, 64 },
    ValueHiddenLayers = new List<int> { 64, 64 },
    TrainingEpochs = 10,
    MiniBatchSize = 64,
    MaxTrajectoryLength = 2048,
    Seed = 42
};

// Create PPO agent
var agent = new PPOAgent<double>(options);

Console.WriteLine(""PPO Hyperparameters:"");
Console.WriteLine($""  Policy network: 64 -> 64 -> {actionSize}"");
Console.WriteLine($""  Value network: 64 -> 64 -> 1"");
Console.WriteLine($""  Mini-batch size: {options.MiniBatchSize}"");
Console.WriteLine($""  Training epochs: {options.TrainingEpochs}"");
Console.WriteLine();

// Create training data for builder
var rng = RandomHelper.CreateSeededRandom(42);
var features = new double[10, stateSize];
var labels = new double[10];
for (int i = 0; i < 10; i++)
{
    for (int j = 0; j < stateSize; j++)
        features[i, j] = rng.NextDouble() * 2 - 1;
    labels[i] = rng.NextDouble();
}

// Build with AiModelBuilder
var loader = DataLoaders.FromArrays(features, labels);
var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureDataLoader(loader)
    .ConfigureModel(agent)
    .BuildAsync();

Console.WriteLine(""Agent built successfully!"");
Console.WriteLine();
Console.WriteLine(""PPO Training Algorithm:"");
Console.WriteLine(""  1. Collect trajectories using current policy"");
Console.WriteLine(""  2. Compute advantages using GAE"");
Console.WriteLine(""  3. Update policy with clipped objective"");
Console.WriteLine(""  4. Update value function"");
"
                },
                new CodeExample
                {
                    Id = "sac-agent",
                    Name = "SAC Agent",
                    Description = "Soft Actor-Critic for sample-efficient continuous control",
                    Difficulty = "Advanced",
                    Tags = ["rl", "sac", "actor-critic", "entropy"],
                    Code = @"// SAC Agent with AiModelBuilder
using AiDotNet;
using AiDotNet.Models.Options;
using AiDotNet.ReinforcementLearning.Agents;
using AiDotNet.Tensors.LinearAlgebra;

// Continuous control environment (e.g., Humanoid)
var stateSize = 11;
var actionSize = 3;

Console.WriteLine(""SAC Agent Configuration"");
Console.WriteLine($""  State size: {stateSize}"");
Console.WriteLine($""  Action size: {actionSize} (continuous)"");
Console.WriteLine();

// Configure SAC options
var options = new SACOptions<double>
{
    StateSize = stateSize,
    ActionSize = actionSize,
    PolicyHiddenLayers = new List<int> { 256, 256 },
    QHiddenLayers = new List<int> { 256, 256 },
    BatchSize = 256,
    ReplayBufferSize = 1000000,
    WarmupSteps = 1000,
    AutoTuneTemperature = true,
    Seed = 42
};

// Create SAC agent
var agent = new SACAgent<double>(options);

Console.WriteLine(""SAC Hyperparameters:"");
Console.WriteLine($""  Policy network: 256 -> 256 -> {actionSize * 2}"");
Console.WriteLine($""  Q-networks: 256 -> 256 -> 1 (twin)"");
Console.WriteLine($""  Replay buffer: {options.ReplayBufferSize:N0}"");
Console.WriteLine($""  Batch size: {options.BatchSize}"");
Console.WriteLine($""  Auto-tune temperature: {options.AutoTuneTemperature}"");
Console.WriteLine();

// Create training data for builder
var rng = RandomHelper.CreateSeededRandom(42);
var features = new double[10, stateSize];
var labels = new double[10];
for (int i = 0; i < 10; i++)
{
    for (int j = 0; j < stateSize; j++)
        features[i, j] = rng.NextDouble() * 2 - 1;
    labels[i] = rng.NextDouble();
}

// Build with AiModelBuilder
var loader = DataLoaders.FromArrays(features, labels);
var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
    .ConfigureDataLoader(loader)
    .ConfigureModel(agent)
    .BuildAsync();

Console.WriteLine(""Agent built successfully!"");
Console.WriteLine();
Console.WriteLine(""SAC Algorithm Features:"");
Console.WriteLine(""  - Maximum entropy RL framework"");
Console.WriteLine(""  - Twin Q-networks (clipped double-Q)"");
Console.WriteLine(""  - Automatic temperature tuning"");
Console.WriteLine(""  - Sample-efficient off-policy learning"");
"
                },
                new CodeExample
                {
                    Id = "multi-agent-rl",
                    Name = "Multi-Agent RL",
                    Description = "Train multiple independent agents using DQN",
                    Difficulty = "Expert",
                    Tags = ["rl", "multi-agent", "marl", "cooperative"],
                    Code = @"// Multi-Agent RL with Independent DQN Agents
using AiDotNet;
using AiDotNet.Models.Options;
using AiDotNet.ReinforcementLearning.Agents;
using AiDotNet.Tensors.LinearAlgebra;

var numAgents = 3;
var stateSize = 16;  // Per-agent observation
var actionSize = 4;  // Per-agent actions

Console.WriteLine(""Multi-Agent System Configuration"");
Console.WriteLine($""  Number of agents: {numAgents}"");
Console.WriteLine($""  State size per agent: {stateSize}"");
Console.WriteLine($""  Action size per agent: {actionSize}"");
Console.WriteLine();

// Create multiple independent DQN agents
var agents = new List<DQNAgent<double>>();
for (int i = 0; i < numAgents; i++)
{
    var options = new DQNOptions<double>
    {
        StateSize = stateSize,
        ActionSize = actionSize,
        HiddenLayers = new List<int> { 128, 128 },
        EpsilonStart = 1.0,
        EpsilonEnd = 0.01,
        EpsilonDecay = 0.995,
        ReplayBufferSize = 10000,
        BatchSize = 64,
        Seed = 42 + i  // Different seed per agent
    };
    agents.Add(new DQNAgent<double>(options));
}

Console.WriteLine($""Created {numAgents} independent DQN agents"");
Console.WriteLine();

// Build each agent with AiModelBuilder
var rng = RandomHelper.CreateSeededRandom(42);
for (int i = 0; i < numAgents; i++)
{
    var features = new double[10, stateSize];
    var labels = new double[10];
    for (int j = 0; j < 10; j++)
    {
        for (int k = 0; k < stateSize; k++)
            features[j, k] = rng.NextDouble() * 2 - 1;
        labels[j] = rng.Next(0, actionSize);
    }

    var loader = DataLoaders.FromArrays(features, labels);
    var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
        .ConfigureDataLoader(loader)
        .ConfigureModel(agents[i])
        .BuildAsync();

    Console.WriteLine($""  Agent {i + 1}: Built successfully"");
}

Console.WriteLine();
Console.WriteLine(""Multi-Agent Training Paradigms:"");
Console.WriteLine(""  - Independent learners (shown here)"");
Console.WriteLine(""  - Centralized training, decentralized execution"");
Console.WriteLine(""  - Communication between agents"");
Console.WriteLine(""  - Shared reward vs individual rewards"");
"
                }
            },

            ["RAG & Embeddings"] = new()
            {
                new CodeExample
                {
                    Id = "vector-store",
                    Name = "Vector Store",
                    Description = "Create vector index for similarity search using HNSW",
                    Difficulty = "Intermediate",
                    Tags = ["rag", "vector", "similarity", "search"],
                    Code = @"// Vector Store with HNSW Index
using AiDotNet;
using AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Indexes;
using AiDotNet.RetrievalAugmentedGeneration.VectorSearch.Metrics;
using AiDotNet.Tensors.LinearAlgebra;

Console.WriteLine(""Vector Store with HNSW Index"");
Console.WriteLine();

// Create HNSW index for efficient approximate nearest neighbor search
var metric = new CosineSimilarityMetric<double>();
var index = new HNSWIndex<double>(
    metric: metric,
    maxConnections: 16,  // M parameter
    efConstruction: 200,
    efSearch: 50);

// Sample document embeddings (in practice, use an embedding model)
var rng = RandomHelper.CreateSeededRandom(42);
var docEmbeddings = new Dictionary<string, Vector<double>>();
var documents = new[]
{
    ""Machine learning is a subset of artificial intelligence."",
    ""Neural networks are inspired by biological neurons."",
    ""Deep learning uses multiple layers of neural networks."",
    ""Natural language processing deals with text data."",
    ""Computer vision processes visual information.""
};

// Generate mock embeddings (384-dim like sentence-transformers)
int embeddingDim = 384;
for (int i = 0; i < documents.Length; i++)
{
    var embedding = new double[embeddingDim];
    for (int j = 0; j < embeddingDim; j++)
        embedding[j] = rng.NextDouble() * 2 - 1;
    var vec = new Vector<double>(embedding);
    vec = vec.Normalize();  // Normalize for cosine similarity
    docEmbeddings[$""doc_{i}""] = vec;
    index.Add($""doc_{i}"", vec);
}

Console.WriteLine($""Indexed {documents.Length} documents"");
Console.WriteLine($""Embedding dimension: {embeddingDim}"");
Console.WriteLine($""Index type: HNSW (M={16}, efConstruction={200})"");
Console.WriteLine();

// Query with mock embedding
var queryEmbedding = new double[embeddingDim];
for (int j = 0; j < embeddingDim; j++)
    queryEmbedding[j] = rng.NextDouble() * 2 - 1;
var queryVec = new Vector<double>(queryEmbedding).Normalize();

var results = index.Search(queryVec, topK: 3);

Console.WriteLine(""Top 3 similar documents:"");
foreach (var (id, score) in results)
{
    int docIdx = int.Parse(id.Split('_')[1]);
    Console.WriteLine($""  [{score:F4}] {documents[docIdx].Substring(0, Math.Min(50, documents[docIdx].Length))}..."");
}
"
                },
                new CodeExample
                {
                    Id = "rag-pipeline",
                    Name = "RAG Pipeline",
                    Description = "Retrieval-Augmented Generation components demonstration",
                    Difficulty = "Advanced",
                    Tags = ["rag", "retrieval", "generation", "qa"],
                    Code = @"// RAG Pipeline Components with AiDotNet
using AiDotNet;
using AiDotNet.RetrievalAugmentedGeneration.Configuration;
using AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies;
using AiDotNet.RetrievalAugmentedGeneration.Rerankers;
using AiDotNet.Tensors.LinearAlgebra;

Console.WriteLine(""RAG Pipeline Components"");
Console.WriteLine();

// Knowledge base documents
var documents = new[]
{
    ""AiDotNet is the most comprehensive AI/ML framework for .NET."",
    ""AiDotNet supports 100+ neural network architectures."",
    ""AiDotNet includes 106+ classical ML algorithms."",
    ""The AiModelBuilder provides a simple facade pattern."",
    ""AiDotNet supports distributed training with DDP and FSDP.""
};

Console.WriteLine($""Knowledge Base: {documents.Length} documents"");
Console.WriteLine();

// Step 1: Configure chunking strategy
var chunkConfig = new ChunkingConfig
{
    ChunkSize = 200,
    ChunkOverlap = 50
};

Console.WriteLine(""Chunking Configuration:"");
Console.WriteLine($""  Chunk size: {chunkConfig.ChunkSize}"");
Console.WriteLine($""  Overlap: {chunkConfig.ChunkOverlap}"");
Console.WriteLine();

// Step 2: Configure retrieval
var retrievalConfig = new RetrievalConfig
{
    TopK = 3,
    SimilarityThreshold = 0.7
};

Console.WriteLine(""Retrieval Configuration:"");
Console.WriteLine($""  Top-K: {retrievalConfig.TopK}"");
Console.WriteLine($""  Similarity threshold: {retrievalConfig.SimilarityThreshold}"");
Console.WriteLine();

// Step 3: Configure reranking
var rerankConfig = new RerankingConfig
{
    EnableReranking = true,
    TopK = 2
};

Console.WriteLine(""Reranking Configuration:"");
Console.WriteLine($""  Enabled: {rerankConfig.EnableReranking}"");
Console.WriteLine($""  Final Top-K: {rerankConfig.TopK}"");
Console.WriteLine();

// Build full RAG configuration
var ragConfig = new RAGConfigurationBuilder()
    .WithChunking(chunkConfig)
    .WithRetrieval(retrievalConfig)
    .WithReranking(rerankConfig)
    .Build();

Console.WriteLine(""RAG Pipeline Ready!"");
Console.WriteLine();
Console.WriteLine(""Pipeline Flow:"");
Console.WriteLine(""  1. Query -> Embedding"");
Console.WriteLine(""  2. Vector search -> Top-K candidates"");
Console.WriteLine(""  3. Reranking -> Final documents"");
Console.WriteLine(""  4. Context + Query -> Generator"");
Console.WriteLine(""  5. Answer with sources"");
"
                },
                new CodeExample
                {
                    Id = "document-chunking",
                    Name = "Document Chunking",
                    Description = "Split documents using RecursiveCharacterTextSplitter",
                    Difficulty = "Intermediate",
                    Tags = ["rag", "chunking", "preprocessing"],
                    Code = @"// Document Chunking with RecursiveCharacterTextSplitter
using AiDotNet;
using AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies;
using AiDotNet.RetrievalAugmentedGeneration.Configuration;

Console.WriteLine(""Document Chunking for RAG"");
Console.WriteLine();

var longDocument = @""AiDotNet is the most comprehensive AI/ML framework for .NET.
It provides 4,300+ implementations across 60+ feature categories.

The framework includes neural networks, classical ML, computer vision,
audio processing, reinforcement learning, and much more.

Key features include the AiModelBuilder facade pattern which simplifies
model creation and training. The library supports both net8.0 and net471."";

Console.WriteLine($""Original document length: {longDocument.Length} characters"");
Console.WriteLine();

// Create chunking configuration
var config = new ChunkingConfig
{
    ChunkSize = 100,
    ChunkOverlap = 20
};

// Create recursive character text splitter
var splitter = new RecursiveCharacterTextSplitter(
    chunkSize: config.ChunkSize,
    chunkOverlap: config.ChunkOverlap,
    separators: new[] { ""\n\n"", ""\n"", "". "", "" "" });

// Split the document
var chunks = splitter.Split(longDocument);

Console.WriteLine($""Chunking Configuration:"");
Console.WriteLine($""  Chunk size: {config.ChunkSize}"");
Console.WriteLine($""  Chunk overlap: {config.ChunkOverlap}"");
Console.WriteLine($""  Separators: [paragraph, newline, sentence, space]"");
Console.WriteLine();

Console.WriteLine($""Number of chunks: {chunks.Count}"");
Console.WriteLine();

Console.WriteLine(""Chunks:"");
for (int i = 0; i < chunks.Count; i++)
{
    var preview = chunks[i].Length > 60
        ? chunks[i].Substring(0, 60) + ""...""
        : chunks[i];
    Console.WriteLine($""  [{i}] ({chunks[i].Length} chars) {preview.Replace(""\n"", "" "")}"");
}
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
                    Code = @"// LoRA (Low-Rank Adaptation) Fine-tuning
using AiDotNet.Tensors.Helpers;

Console.WriteLine(""LoRA Fine-tuning for Large Language Models"");
Console.WriteLine();

// LoRA configuration
Console.WriteLine(""LoRA Configuration:"");
Console.WriteLine(""  Base Model: llama-7b (7 billion parameters)"");
Console.WriteLine(""  LoRA Rank: 8"");
Console.WriteLine(""  LoRA Alpha: 16"");
Console.WriteLine(""  Dropout: 0.1"");
Console.WriteLine(""  Target Modules: q_proj, v_proj"");
Console.WriteLine();

// Sample training data
Console.WriteLine(""Sample Training Data (Instruction-Response):"");
Console.WriteLine(""  1. 'Explain quantum computing' -> 'Quantum computing uses qubits...'"");
Console.WriteLine(""  2. 'What is machine learning?' -> 'Machine learning is a subset...'"");
Console.WriteLine(""  3. 'Define neural network' -> 'A neural network is a model...'"");
Console.WriteLine();

// Parameter efficiency calculation
long totalParams = 7_000_000_000L;
long trainableParams = 4_194_304L; // Typical LoRA params for rank=8
double percentage = trainableParams * 100.0 / totalParams;

Console.WriteLine(""Parameter Efficiency:"");
Console.WriteLine($""  Total parameters: {totalParams:N0}"");
Console.WriteLine($""  Trainable parameters: {trainableParams:N0}"");
Console.WriteLine($""  % trainable: {percentage:F4}%"");
Console.WriteLine();

Console.WriteLine(""LoRA Benefits:"");
Console.WriteLine(""  - Train only ~0.06% of parameters"");
Console.WriteLine(""  - Significantly reduced memory usage"");
Console.WriteLine(""  - Fast training on consumer GPUs"");
Console.WriteLine(""  - Easy adapter merging/switching"");
"
                },
                new CodeExample
                {
                    Id = "qlora",
                    Name = "QLoRA (Quantized LoRA)",
                    Description = "4-bit quantized LoRA for memory-efficient fine-tuning",
                    Difficulty = "Advanced",
                    Tags = ["qlora", "quantization", "fine-tuning", "memory-efficient"],
                    Code = @"// QLoRA (Quantized LoRA) Fine-tuning
using AiDotNet.Tensors.Helpers;

Console.WriteLine(""QLoRA: 4-bit Quantized LoRA Fine-tuning"");
Console.WriteLine();

// QLoRA configuration
Console.WriteLine(""QLoRA Configuration:"");
Console.WriteLine(""  Base Model: llama-13b (13 billion parameters)"");
Console.WriteLine(""  Quantization: 4-bit NF4 (Normal Float 4)"");
Console.WriteLine(""  Compute Type: BFloat16"");
Console.WriteLine(""  LoRA Rank: 64"");
Console.WriteLine(""  LoRA Alpha: 16"");
Console.WriteLine(""  Target Modules: q_proj, k_proj, v_proj, o_proj"");
Console.WriteLine();

// Memory comparison
Console.WriteLine(""Memory Usage Comparison:"");
Console.WriteLine(""  Full Precision (FP32):  ~52 GB"");
Console.WriteLine(""  Half Precision (FP16):  ~26 GB"");
Console.WriteLine(""  QLoRA (4-bit NF4):      ~4 GB"");
Console.WriteLine();

Console.WriteLine(""Training Configuration:"");
Console.WriteLine(""  Epochs: 1"");
Console.WriteLine(""  Batch Size: 1 (limited by memory)"");
Console.WriteLine(""  Gradient Accumulation: 16 steps"");
Console.WriteLine(""  Effective Batch Size: 16"");
Console.WriteLine(""  Learning Rate: 2e-4"");
Console.WriteLine();

Console.WriteLine(""QLoRA Benefits:"");
Console.WriteLine(""  - Fine-tune 13B models on single 24GB GPU"");
Console.WriteLine(""  - NF4 quantization preserves model quality"");
Console.WriteLine(""  - Double quantization reduces memory further"");
Console.WriteLine(""  - Paged optimizers prevent OOM errors"");
"
                },
                new CodeExample
                {
                    Id = "dora",
                    Name = "DoRA (Weight-Decomposed LoRA)",
                    Description = "Improved LoRA with weight decomposition",
                    Difficulty = "Expert",
                    Tags = ["dora", "fine-tuning", "peft", "advanced"],
                    Code = @"// DoRA (Weight-Decomposed LoRA) Fine-tuning
using AiDotNet.Tensors.Helpers;

Console.WriteLine(""DoRA: Weight-Decomposed Low-Rank Adaptation"");
Console.WriteLine();

// DoRA configuration
Console.WriteLine(""DoRA Configuration:"");
Console.WriteLine(""  Base Model: mistral-7b"");
Console.WriteLine(""  LoRA Rank: 16"");
Console.WriteLine(""  LoRA Alpha: 32"");
Console.WriteLine(""  Dropout: 0.1"");
Console.WriteLine(""  Target Modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj"");
Console.WriteLine(""  Decompose Magnitude: true"");
Console.WriteLine();

// DoRA vs LoRA
Console.WriteLine(""DoRA vs Standard LoRA:"");
Console.WriteLine(""  LoRA:  W' = W + BA"");
Console.WriteLine(""  DoRA:  W' = m * (W + BA) / ||W + BA||"");
Console.WriteLine();
Console.WriteLine(""  Where:"");
Console.WriteLine(""    m = learnable magnitude scalar"");
Console.WriteLine(""    BA = low-rank update (same as LoRA)"");
Console.WriteLine(""    ||.|| = normalization"");
Console.WriteLine();

Console.WriteLine(""DoRA Benefits:"");
Console.WriteLine(""  - Decomposes weight updates into magnitude and direction"");
Console.WriteLine(""  - Better matches full fine-tuning behavior"");
Console.WriteLine(""  - Improved performance on instruction tuning"");
Console.WriteLine(""  - Same memory efficiency as standard LoRA"");
Console.WriteLine(""  - Minimal additional compute overhead"");
"
                },
                new CodeExample
                {
                    Id = "adalora",
                    Name = "AdaLoRA (Adaptive LoRA)",
                    Description = "Automatically adjust LoRA rank during training",
                    Difficulty = "Expert",
                    Tags = ["adalora", "adaptive", "fine-tuning", "dynamic"],
                    Code = @"// AdaLoRA (Adaptive LoRA) Fine-tuning
using AiDotNet.Tensors.Helpers;

Console.WriteLine(""AdaLoRA: Adaptive Low-Rank Adaptation"");
Console.WriteLine();

// AdaLoRA configuration
Console.WriteLine(""AdaLoRA Configuration:"");
Console.WriteLine(""  Base Model: llama-7b"");
Console.WriteLine(""  Initial Rank: 12"");
Console.WriteLine(""  Target Rank: 8"");
Console.WriteLine(""  Alpha: 32"");
Console.WriteLine(""  Dropout: 0.1"");
Console.WriteLine(""  Beta (start/end): 0.85"");
Console.WriteLine(""  Delta T: 10 steps"");
Console.WriteLine(""  Rank Pattern: Decreasing"");
Console.WriteLine();

// Adaptive rank allocation
Console.WriteLine(""Adaptive Rank Allocation:"");
Console.WriteLine(""  - Start with higher rank (12)"");
Console.WriteLine(""  - Prune less important singular values"");
Console.WriteLine(""  - End with target rank (8)"");
Console.WriteLine();
Console.WriteLine(""  Layer-wise final ranks (example):"");
Console.WriteLine(""    q_proj: 10"");
Console.WriteLine(""    k_proj: 6"");
Console.WriteLine(""    v_proj: 8"");
Console.WriteLine(""    o_proj: 8"");
Console.WriteLine();

Console.WriteLine(""AdaLoRA Benefits:"");
Console.WriteLine(""  - Automatically allocates rank budget"");
Console.WriteLine(""  - Important layers get higher ranks"");
Console.WriteLine(""  - Reduces total trainable parameters"");
Console.WriteLine(""  - Better parameter efficiency than fixed-rank LoRA"");
Console.WriteLine(""  - SVD-based importance scoring"");
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
var rng = RandomHelper.CreateSeededRandom(42);
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
var rng = RandomHelper.CreateSeededRandom(42);
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
var rng = RandomHelper.CreateSeededRandom(42);
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
                    Code = @"// Bayesian Hyperparameter Optimization
using AiDotNet.AutoML.SearchSpace;
using AiDotNet.Tensors.Helpers;

Console.WriteLine(""Bayesian Hyperparameter Optimization"");
Console.WriteLine();

// Define search space
var searchSpace = new HyperparameterSearchSpace();
searchSpace.AddIntParameter(""n_estimators"", 50, 500);
searchSpace.AddFloatParameter(""learning_rate"", 0.01, 0.3);  // log scale
searchSpace.AddIntParameter(""max_depth"", 3, 15);
searchSpace.AddFloatParameter(""subsample"", 0.5, 1.0);

Console.WriteLine(""Search Space:"");
Console.WriteLine(""  n_estimators: [50, 500] (integer)"");
Console.WriteLine(""  learning_rate: [0.01, 0.3] (float, log scale)"");
Console.WriteLine(""  max_depth: [3, 15] (integer)"");
Console.WriteLine(""  subsample: [0.5, 1.0] (float)"");
Console.WriteLine();

Console.WriteLine(""Bayesian Optimization Configuration:"");
Console.WriteLine(""  Acquisition Function: Expected Improvement (EI)"");
Console.WriteLine(""  Initial Random Points: 10"");
Console.WriteLine(""  Max Iterations: 50"");
Console.WriteLine();

// Sample data
var rng = RandomHelper.CreateSeededRandom(42);
Console.WriteLine(""Training Data:"");
Console.WriteLine(""  800 samples, 15 features"");
Console.WriteLine(""  3 classes (multiclass classification)"");
Console.WriteLine();

Console.WriteLine(""Sample Optimization Results:"");
Console.WriteLine(""  Best parameters found:"");
Console.WriteLine(""    n_estimators: 287"");
Console.WriteLine(""    learning_rate: 0.0523"");
Console.WriteLine(""    max_depth: 8"");
Console.WriteLine(""    subsample: 0.85"");
Console.WriteLine(""  Best accuracy: 94.5%"");
Console.WriteLine(""  Iterations completed: 50"");
"
                }
            },

            ["Loss Functions"] = new()
            {
                new CodeExample
                {
                    Id = "mse-loss",
                    Name = "Mean Squared Error Loss",
                    Description = "Common loss function for regression tasks",
                    Difficulty = "Beginner",
                    Tags = ["loss", "mse", "regression"],
                    Code = @"// Mean Squared Error Loss Function
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;

// Create predicted and actual values
var predicted = new Vector<double>(new double[] { 2.5, 0.0, 2.1, 7.8 });
var actual = new Vector<double>(new double[] { 3.0, -0.5, 2.0, 7.5 });

// Create MSE loss function
var mseLoss = new MeanSquaredErrorLoss<double>();

// Calculate the loss
var loss = mseLoss.CalculateLoss(predicted, actual);

// Calculate the derivative (gradient)
var gradient = mseLoss.CalculateDerivative(predicted, actual);

Console.WriteLine(""Mean Squared Error Loss:"");
Console.WriteLine($""  Predicted: [{string.Join("", "", predicted.ToArray())}]"");
Console.WriteLine($""  Actual:    [{string.Join("", "", actual.ToArray())}]"");
Console.WriteLine($""  Loss:      {loss:F6}"");
Console.WriteLine($""  Gradient:  [{string.Join("", "", gradient.ToArray().Select(g => g.ToString(""F4"")))}]"");
Console.WriteLine();
Console.WriteLine(""MSE penalizes larger errors more due to squaring"");
"
                },
                new CodeExample
                {
                    Id = "cross-entropy-loss",
                    Name = "Cross Entropy Loss",
                    Description = "Loss function for classification tasks",
                    Difficulty = "Intermediate",
                    Tags = ["loss", "cross-entropy", "classification"],
                    Code = @"// Cross Entropy Loss for Classification
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;

// Predicted probabilities (after softmax)
var predicted = new Vector<double>(new double[] { 0.7, 0.2, 0.1 });
// One-hot encoded actual class (class 0)
var actual = new Vector<double>(new double[] { 1.0, 0.0, 0.0 });

// Create Cross Entropy loss
var ceLoss = new CrossEntropyLoss<double>();

var loss = ceLoss.CalculateLoss(predicted, actual);
var gradient = ceLoss.CalculateDerivative(predicted, actual);

Console.WriteLine(""Cross Entropy Loss:"");
Console.WriteLine($""  Predicted probs: [{string.Join("", "", predicted.ToArray().Select(p => p.ToString(""F2"")))}]"");
Console.WriteLine($""  Actual (one-hot): [{string.Join("", "", actual.ToArray().Select(a => a.ToString(""F0"")))}]"");
Console.WriteLine($""  Loss: {loss:F6}"");
Console.WriteLine($""  Gradient: [{string.Join("", "", gradient.ToArray().Select(g => g.ToString(""F4"")))}]"");
Console.WriteLine();
Console.WriteLine(""Cross entropy measures the difference between probability distributions"");
"
                },
                new CodeExample
                {
                    Id = "huber-loss",
                    Name = "Huber Loss",
                    Description = "Robust loss function combining MSE and MAE",
                    Difficulty = "Intermediate",
                    Tags = ["loss", "huber", "robust"],
                    Code = @"// Huber Loss - Robust to Outliers
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;

// Data with an outlier
var predicted = new Vector<double>(new double[] { 1.0, 2.0, 3.0, 10.0 });
var actual = new Vector<double>(new double[] { 1.1, 2.1, 3.0, 3.5 }); // Last one is outlier

var huberLoss = new HuberLoss<double>();
var mseLoss = new MeanSquaredErrorLoss<double>();
var maeLoss = new MeanAbsoluteErrorLoss<double>();

var huber = huberLoss.CalculateLoss(predicted, actual);
var mse = mseLoss.CalculateLoss(predicted, actual);
var mae = maeLoss.CalculateLoss(predicted, actual);

Console.WriteLine(""Comparing Loss Functions (with outlier):"");
Console.WriteLine($""  Huber Loss: {huber:F4}"");
Console.WriteLine($""  MSE Loss:   {mse:F4}"");
Console.WriteLine($""  MAE Loss:   {mae:F4}"");
Console.WriteLine();
Console.WriteLine(""Huber is quadratic for small errors, linear for large errors"");
Console.WriteLine(""This makes it less sensitive to outliers than MSE"");
"
                },
                new CodeExample
                {
                    Id = "focal-loss",
                    Name = "Focal Loss",
                    Description = "Loss function for imbalanced classification",
                    Difficulty = "Advanced",
                    Tags = ["loss", "focal", "imbalanced"],
                    Code = @"// Focal Loss for Imbalanced Datasets
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.LinearAlgebra;

// Simulating imbalanced classification
// Easy example (high confidence correct prediction)
var easyPred = new Vector<double>(new double[] { 0.95, 0.05 });
var easyActual = new Vector<double>(new double[] { 1.0, 0.0 });

// Hard example (low confidence)
var hardPred = new Vector<double>(new double[] { 0.55, 0.45 });
var hardActual = new Vector<double>(new double[] { 1.0, 0.0 });

var focalLoss = new FocalLoss<double>();
var ceLoss = new CrossEntropyLoss<double>();

Console.WriteLine(""Focal vs Cross Entropy Loss:"");
Console.WriteLine();
Console.WriteLine(""Easy example (95% confident):"");
Console.WriteLine($""  Cross Entropy: {ceLoss.CalculateLoss(easyPred, easyActual):F4}"");
Console.WriteLine($""  Focal Loss:    {focalLoss.CalculateLoss(easyPred, easyActual):F4}"");
Console.WriteLine();
Console.WriteLine(""Hard example (55% confident):"");
Console.WriteLine($""  Cross Entropy: {ceLoss.CalculateLoss(hardPred, hardActual):F4}"");
Console.WriteLine($""  Focal Loss:    {focalLoss.CalculateLoss(hardPred, hardActual):F4}"");
Console.WriteLine();
Console.WriteLine(""Focal loss down-weights easy examples to focus on hard ones"");
"
                }
            },

            ["Activation Functions"] = new()
            {
                new CodeExample
                {
                    Id = "relu-activation",
                    Name = "ReLU Activation",
                    Description = "Most popular activation function for neural networks",
                    Difficulty = "Beginner",
                    Tags = ["activation", "relu", "neural-network"],
                    Code = @"// ReLU (Rectified Linear Unit) Activation
using AiDotNet.ActivationFunctions;
using AiDotNet.Tensors.LinearAlgebra;

var relu = new ReLUActivation<double>();

// Test values
var inputs = new Vector<double>(new double[] { -2.0, -1.0, 0.0, 1.0, 2.0 });
var outputs = relu.Activate(inputs);

Console.WriteLine(""ReLU Activation: f(x) = max(0, x)"");
Console.WriteLine();
Console.WriteLine(""Input  -> Output"");
for (int i = 0; i < inputs.Length; i++)
{
    Console.WriteLine($""{inputs[i],6:F1} -> {outputs[i],6:F1}"");
}
Console.WriteLine();
Console.WriteLine(""ReLU is simple but effective:"");
Console.WriteLine(""- Outputs input directly if positive"");
Console.WriteLine(""- Outputs zero for negative inputs"");
Console.WriteLine(""- Fast to compute and good gradients"");
"
                },
                new CodeExample
                {
                    Id = "gelu-activation",
                    Name = "GELU Activation",
                    Description = "Used in transformers like BERT and GPT",
                    Difficulty = "Intermediate",
                    Tags = ["activation", "gelu", "transformer"],
                    Code = @"// GELU (Gaussian Error Linear Unit) Activation
using AiDotNet.ActivationFunctions;
using AiDotNet.Tensors.LinearAlgebra;

var gelu = new GELUActivation<double>();
var relu = new ReLUActivation<double>();

var inputs = new Vector<double>(new double[] { -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0 });
var geluOutputs = gelu.Activate(inputs);
var reluOutputs = relu.Activate(inputs);

Console.WriteLine(""GELU vs ReLU Comparison:"");
Console.WriteLine();
Console.WriteLine(""Input   GELU    ReLU"");
for (int i = 0; i < inputs.Length; i++)
{
    Console.WriteLine($""{inputs[i],6:F1} {geluOutputs[i],7:F3} {reluOutputs[i],7:F3}"");
}
Console.WriteLine();
Console.WriteLine(""GELU is smoother than ReLU:"");
Console.WriteLine(""- Used in BERT, GPT, and modern transformers"");
Console.WriteLine(""- Allows small negative values through"");
Console.WriteLine(""- Better gradient flow for deep networks"");
"
                },
                new CodeExample
                {
                    Id = "sigmoid-softmax",
                    Name = "Sigmoid and Softmax",
                    Description = "Output activations for classification",
                    Difficulty = "Beginner",
                    Tags = ["activation", "sigmoid", "softmax"],
                    Code = @"// Sigmoid and Softmax for Classification Outputs
using AiDotNet.ActivationFunctions;
using AiDotNet.Tensors.LinearAlgebra;

var sigmoid = new SigmoidActivation<double>();
var softmax = new SoftmaxActivation<double>();

// Binary classification logits
var binaryLogits = new Vector<double>(new double[] { 2.0 });
var binaryProb = sigmoid.Activate(binaryLogits);

// Multi-class classification logits
var multiLogits = new Vector<double>(new double[] { 2.0, 1.0, 0.1 });
var multiProbs = softmax.Activate(multiLogits);

Console.WriteLine(""Sigmoid (Binary Classification):"");
Console.WriteLine($""  Logit: {binaryLogits[0]:F1} -> Probability: {binaryProb[0]:F4}"");
Console.WriteLine();
Console.WriteLine(""Softmax (Multi-class Classification):"");
Console.WriteLine($""  Logits: [{string.Join("", "", multiLogits.ToArray().Select(l => l.ToString(""F1"")))}]"");
Console.WriteLine($""  Probs:  [{string.Join("", "", multiProbs.ToArray().Select(p => p.ToString(""F4"")))}]"");
Console.WriteLine($""  Sum:    {multiProbs.ToArray().Sum():F4} (always sums to 1)"");
Console.WriteLine();
Console.WriteLine(""Sigmoid: Squashes to [0,1] for binary"");
Console.WriteLine(""Softmax: Converts to probability distribution"");
"
                },
                new CodeExample
                {
                    Id = "swish-mish",
                    Name = "Swish and Mish",
                    Description = "Modern self-gated activation functions",
                    Difficulty = "Intermediate",
                    Tags = ["activation", "swish", "mish"],
                    Code = @"// Swish and Mish - Modern Activation Functions
using AiDotNet.ActivationFunctions;
using AiDotNet.Tensors.LinearAlgebra;

var silu = new SiLUActivation<double>(); // SiLU = Swish
var mish = new MishActivation<double>();
var relu = new ReLUActivation<double>();

var inputs = new Vector<double>(new double[] { -2.0, -1.0, 0.0, 1.0, 2.0 });

var siluOut = silu.Activate(inputs);
var mishOut = mish.Activate(inputs);
var reluOut = relu.Activate(inputs);

Console.WriteLine(""Modern Activation Functions:"");
Console.WriteLine();
Console.WriteLine(""Input   SiLU/Swish  Mish     ReLU"");
for (int i = 0; i < inputs.Length; i++)
{
    Console.WriteLine($""{inputs[i],6:F1} {siluOut[i],10:F4} {mishOut[i],8:F4} {reluOut[i],8:F4}"");
}
Console.WriteLine();
Console.WriteLine(""SiLU (Swish): f(x) = x * sigmoid(x)"");
Console.WriteLine(""Mish: f(x) = x * tanh(softplus(x))"");
Console.WriteLine(""Both are smooth, non-monotonic, and self-gated"");
"
                }
            },

            ["Learning Rate Schedulers"] = new()
            {
                new CodeExample
                {
                    Id = "step-lr-scheduler",
                    Name = "Step Learning Rate Scheduler",
                    Description = "Decay learning rate at fixed intervals",
                    Difficulty = "Beginner",
                    Tags = ["scheduler", "learning-rate", "training"],
                    Code = @"// Step Learning Rate Scheduler
using AiDotNet.LearningRateSchedulers;

// Reduce LR by 10x every 30 steps
var scheduler = new StepLRScheduler(
    baseLearningRate: 0.1,
    stepSize: 30,
    gamma: 0.1
);

Console.WriteLine(""Step LR Scheduler (decay by 10x every 30 steps):"");
Console.WriteLine(""Step | Learning Rate"");
Console.WriteLine(""-----+--------------"");

for (int step = 0; step <= 100; step += 10)
{
    // Set the step
    for (int i = 0; i < step; i++) scheduler.Step();

    Console.WriteLine($""{step,4} | {scheduler.CurrentLearningRate:E2}"");

    // Reset for next iteration
    scheduler = new StepLRScheduler(0.1, 30, 0.1);
}

Console.WriteLine();
Console.WriteLine(""Step scheduler is simple and predictable"");
Console.WriteLine(""Good for transfer learning and fine-tuning"");
"
                },
                new CodeExample
                {
                    Id = "cosine-annealing-scheduler",
                    Name = "Cosine Annealing Scheduler",
                    Description = "Smooth cosine decay for learning rate",
                    Difficulty = "Intermediate",
                    Tags = ["scheduler", "cosine", "training"],
                    Code = @"// Cosine Annealing Learning Rate Scheduler
using AiDotNet.LearningRateSchedulers;

var scheduler = new CosineAnnealingLRScheduler(
    baseLearningRate: 0.1,
    tMax: 100,       // Total steps
    etaMin: 0.001    // Minimum LR
);

Console.WriteLine(""Cosine Annealing Scheduler:"");
Console.WriteLine(""Step | Learning Rate | Visual"");
Console.WriteLine(""-----+--------------+--------"");

for (int step = 0; step <= 100; step += 10)
{
    var lr = scheduler.GetLearningRate(step);
    int bars = (int)(lr / 0.1 * 20);
    Console.WriteLine($""{step,4} | {lr:F5}     | {new string('█', bars)}"");
}

Console.WriteLine();
Console.WriteLine(""Cosine annealing provides smooth decay"");
Console.WriteLine(""Popular in computer vision (ResNets, ViT)"");
"
                },
                new CodeExample
                {
                    Id = "one-cycle-scheduler",
                    Name = "One Cycle Scheduler",
                    Description = "Super-convergence training strategy",
                    Difficulty = "Advanced",
                    Tags = ["scheduler", "one-cycle", "super-convergence"],
                    Code = @"// One Cycle Learning Rate Scheduler
using AiDotNet.LearningRateSchedulers;

var scheduler = new OneCycleLRScheduler(
    maxLearningRate: 0.1,
    totalSteps: 100,
    percentStart: 0.3,    // Warmup for 30% of training
    percentAnneal: 0.7    // Anneal for 70%
);

Console.WriteLine(""One Cycle Scheduler (Super-Convergence):"");
Console.WriteLine(""Step | Learning Rate | Phase"");
Console.WriteLine(""-----+--------------+--------"");

for (int step = 0; step <= 100; step += 10)
{
    var lr = scheduler.GetLearningRate(step);
    string phase = step < 30 ? ""Warmup"" : ""Anneal"";
    Console.WriteLine($""{step,4} | {lr:F5}     | {phase}"");
}

Console.WriteLine();
Console.WriteLine(""One Cycle training strategy:"");
Console.WriteLine(""1. Warmup: Gradually increase LR"");
Console.WriteLine(""2. Anneal: Decrease to find sharp minimum"");
Console.WriteLine(""Enables training with higher max LR"");
"
                }
            },

            ["Regularization"] = new()
            {
                new CodeExample
                {
                    Id = "l1-regularization",
                    Name = "L1 (Lasso) Regularization",
                    Description = "Promotes sparse solutions by zeroing coefficients",
                    Difficulty = "Intermediate",
                    Tags = ["regularization", "l1", "lasso", "sparsity"],
                    Code = @"// L1 Regularization (Lasso)
using AiDotNet.Regularization;
using AiDotNet.Tensors.LinearAlgebra;

// Create L1 regularization with strength 0.1
var l1Reg = new L1Regularization<double, Matrix<double>, Vector<double>>(
    new RegularizationOptions { Strength = 0.1 }
);

// Sample coefficients
var coefficients = new Vector<double>(new double[] { 0.5, -0.3, 0.8, -0.1, 0.02 });
var gradient = new Vector<double>(new double[] { 0.1, 0.2, -0.1, 0.05, 0.01 });

var regularizedGrad = l1Reg.Regularize(gradient, coefficients);

Console.WriteLine(""L1 (Lasso) Regularization:"");
Console.WriteLine(""Adds penalty = λ * |coefficients|"");
Console.WriteLine();
Console.WriteLine(""Coefficient | Original Grad | Regularized Grad"");
for (int i = 0; i < coefficients.Length; i++)
{
    Console.WriteLine($""{coefficients[i],11:F2} | {gradient[i],13:F4} | {regularizedGrad[i],16:F4}"");
}
Console.WriteLine();
Console.WriteLine(""L1 pushes small coefficients to exactly zero"");
Console.WriteLine(""This performs automatic feature selection!"");
"
                },
                new CodeExample
                {
                    Id = "l2-regularization",
                    Name = "L2 (Ridge) Regularization",
                    Description = "Prevents large weights with smooth penalty",
                    Difficulty = "Intermediate",
                    Tags = ["regularization", "l2", "ridge", "weight-decay"],
                    Code = @"// L2 Regularization (Ridge / Weight Decay)
using AiDotNet.Regularization;
using AiDotNet.Tensors.LinearAlgebra;

var l2Reg = new L2Regularization<double, Matrix<double>, Vector<double>>(
    new RegularizationOptions { Strength = 0.01 }
);

var coefficients = new Vector<double>(new double[] { 1.0, 5.0, -3.0, 0.1 });
var gradient = new Vector<double>(new double[] { 0.1, 0.1, 0.1, 0.1 });

var regularizedGrad = l2Reg.Regularize(gradient, coefficients);

Console.WriteLine(""L2 (Ridge) Regularization:"");
Console.WriteLine(""Adds penalty = λ * coefficients²"");
Console.WriteLine();
Console.WriteLine(""Coefficient | Original Grad | Regularized Grad"");
for (int i = 0; i < coefficients.Length; i++)
{
    Console.WriteLine($""{coefficients[i],11:F2} | {gradient[i],13:F4} | {regularizedGrad[i],16:F4}"");
}
Console.WriteLine();
Console.WriteLine(""L2 penalizes large weights proportionally"");
Console.WriteLine(""Also known as 'weight decay' in deep learning"");
"
                },
                new CodeExample
                {
                    Id = "elastic-net-regularization",
                    Name = "Elastic Net Regularization",
                    Description = "Combines L1 and L2 regularization benefits",
                    Difficulty = "Advanced",
                    Tags = ["regularization", "elastic-net", "combined"],
                    Code = @"// Elastic Net Regularization (L1 + L2)
using AiDotNet.Regularization;
using AiDotNet.Tensors.LinearAlgebra;

var elasticNet = new ElasticRegularization<double, Matrix<double>, Vector<double>>(
    new RegularizationOptions
    {
        Strength = 0.1,
        L1Ratio = 0.5  // 50% L1, 50% L2
    }
);

var l1Reg = new L1Regularization<double, Matrix<double>, Vector<double>>(
    new RegularizationOptions { Strength = 0.1 }
);

var l2Reg = new L2Regularization<double, Matrix<double>, Vector<double>>(
    new RegularizationOptions { Strength = 0.1 }
);

var coefficients = new Vector<double>(new double[] { 0.5, 2.0, -0.01, 0.8 });
var gradient = new Vector<double>(new double[] { 0.1, 0.1, 0.1, 0.1 });

Console.WriteLine(""Elastic Net = α*L1 + (1-α)*L2"");
Console.WriteLine(""With α (L1Ratio) = 0.5"");
Console.WriteLine();
Console.WriteLine(""Coeff  | L1 Grad | L2 Grad | Elastic"");
for (int i = 0; i < coefficients.Length; i++)
{
    var l1Grad = l1Reg.Regularize(gradient, coefficients);
    var l2Grad = l2Reg.Regularize(gradient, coefficients);
    var elGrad = elasticNet.Regularize(gradient, coefficients);
    Console.WriteLine($""{coefficients[i],6:F2} | {l1Grad[i],7:F4} | {l2Grad[i],7:F4} | {elGrad[i],7:F4}"");
}
Console.WriteLine();
Console.WriteLine(""Elastic Net: Sparsity (L1) + Stability (L2)"");
"
                }
            },

            ["Kernels"] = new()
            {
                new CodeExample
                {
                    Id = "gaussian-kernel",
                    Name = "Gaussian (RBF) Kernel",
                    Description = "Most popular kernel for measuring similarity",
                    Difficulty = "Intermediate",
                    Tags = ["kernel", "rbf", "gaussian", "similarity"],
                    Code = @"// Gaussian (RBF) Kernel
using AiDotNet.Kernels;
using AiDotNet.Tensors.LinearAlgebra;

var rbfKernel = new GaussianKernel<double>(sigma: 1.0);

// Create sample points
var point1 = new Vector<double>(new double[] { 0.0, 0.0 });
var point2 = new Vector<double>(new double[] { 1.0, 0.0 });
var point3 = new Vector<double>(new double[] { 2.0, 0.0 });
var point4 = new Vector<double>(new double[] { 5.0, 0.0 });

Console.WriteLine(""Gaussian (RBF) Kernel: K(x,y) = exp(-||x-y||²/2σ²)"");
Console.WriteLine(""Sigma = 1.0"");
Console.WriteLine();
Console.WriteLine(""Comparing point (0,0) to other points:"");
Console.WriteLine($""  Distance 0: K = {rbfKernel.Calculate(point1, point1):F4} (identical)"");
Console.WriteLine($""  Distance 1: K = {rbfKernel.Calculate(point1, point2):F4}"");
Console.WriteLine($""  Distance 2: K = {rbfKernel.Calculate(point1, point3):F4}"");
Console.WriteLine($""  Distance 5: K = {rbfKernel.Calculate(point1, point4):F4} (far apart)"");
Console.WriteLine();
Console.WriteLine(""RBF kernel: 1 = identical, 0 = infinitely far"");
Console.WriteLine(""Used in SVM, GP regression, and many ML algorithms"");
"
                },
                new CodeExample
                {
                    Id = "polynomial-kernel",
                    Name = "Polynomial Kernel",
                    Description = "Captures polynomial relationships between features",
                    Difficulty = "Intermediate",
                    Tags = ["kernel", "polynomial", "svm"],
                    Code = @"// Polynomial Kernel
using AiDotNet.Kernels;
using AiDotNet.Tensors.LinearAlgebra;

var linearKernel = new LinearKernel<double>();
var polyKernel2 = new PolynomialKernel<double>(degree: 2);
var polyKernel3 = new PolynomialKernel<double>(degree: 3);

var x = new Vector<double>(new double[] { 1.0, 2.0 });
var y = new Vector<double>(new double[] { 3.0, 4.0 });

Console.WriteLine(""Polynomial Kernel: K(x,y) = (x·y + c)^d"");
Console.WriteLine();
Console.WriteLine($""Points: x = [{string.Join("", "", x.ToArray())}], y = [{string.Join("", "", y.ToArray())}]"");
Console.WriteLine();
Console.WriteLine($""Linear (degree 1):    K = {linearKernel.Calculate(x, y):F4}"");
Console.WriteLine($""Polynomial (degree 2): K = {polyKernel2.Calculate(x, y):F4}"");
Console.WriteLine($""Polynomial (degree 3): K = {polyKernel3.Calculate(x, y):F4}"");
Console.WriteLine();
Console.WriteLine(""Polynomial kernels implicitly map data to"");
Console.WriteLine(""higher-dimensional spaces without explicit computation"");
"
                },
                new CodeExample
                {
                    Id = "matern-kernel",
                    Name = "Matérn Kernel",
                    Description = "Flexible kernel for Gaussian processes",
                    Difficulty = "Advanced",
                    Tags = ["kernel", "matern", "gaussian-process"],
                    Code = @"// Matérn Kernel for Gaussian Processes
using AiDotNet.Kernels;
using AiDotNet.Tensors.LinearAlgebra;

var rbfKernel = new GaussianKernel<double>(sigma: 1.0);
var matern = new MaternKernel<double>(lengthScale: 1.0, nu: 2.5);
var laplacian = new LaplacianKernel<double>(sigma: 1.0);

var origin = new Vector<double>(new double[] { 0.0 });

Console.WriteLine(""Kernel Comparison at Various Distances:"");
Console.WriteLine(""Distance | RBF    | Matérn | Laplacian"");
Console.WriteLine(""---------+--------+--------+----------"");

foreach (var dist in new[] { 0.0, 0.5, 1.0, 2.0, 3.0 })
{
    var point = new Vector<double>(new double[] { dist });
    var rbf = rbfKernel.Calculate(origin, point);
    var mat = matern.Calculate(origin, point);
    var lap = laplacian.Calculate(origin, point);
    Console.WriteLine($""{dist,8:F1} | {rbf,6:F4} | {mat,6:F4} | {lap,6:F4}"");
}

Console.WriteLine();
Console.WriteLine(""Matérn kernel allows controlling smoothness via nu"");
Console.WriteLine(""nu=0.5: Equivalent to Laplacian (rough functions)"");
Console.WriteLine(""nu=∞: Equivalent to RBF (very smooth functions)"");
"
                }
            },

            ["Normalizers"] = new()
            {
                new CodeExample
                {
                    Id = "minmax-normalizer",
                    Name = "Min-Max Normalization",
                    Description = "Scale features to [0, 1] range",
                    Difficulty = "Beginner",
                    Tags = ["normalization", "minmax", "scaling"],
                    Code = @"// Min-Max Normalization
using AiDotNet.Tensors.LinearAlgebra;

// Sample data with different scales
var data = new Vector<double>(new double[] { 10, 20, 30, 40, 50, 100 });

// Manual min-max normalization: (x - min) / (max - min)
var min = data.Min();
var max = data.Max();
var normalized = data.Transform(x => (x - min) / (max - min));

Console.WriteLine(""Min-Max Normalization: x' = (x - min) / (max - min)"");
Console.WriteLine();
Console.WriteLine($""Original data: [{string.Join("", "", data.ToArray())}]"");
Console.WriteLine($""Min: {min}, Max: {max}"");
Console.WriteLine();
Console.WriteLine(""Original -> Normalized"");
for (int i = 0; i < data.Length; i++)
{
    Console.WriteLine($""{data[i],8:F0} -> {normalized[i],8:F4}"");
}
Console.WriteLine();
Console.WriteLine(""Min-Max scales all values to [0, 1] range"");
Console.WriteLine(""Useful when you need bounded values"");
"
                },
                new CodeExample
                {
                    Id = "zscore-normalizer",
                    Name = "Z-Score Normalization",
                    Description = "Standardize to zero mean and unit variance",
                    Difficulty = "Beginner",
                    Tags = ["normalization", "zscore", "standardization"],
                    Code = @"// Z-Score (Standard) Normalization
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Helpers;

var data = new Vector<double>(new double[] { 10, 20, 30, 40, 50 });

// Calculate mean and std
var mean = StatisticsHelper<double>.CalculateMean(data);
var std = Math.Sqrt(StatisticsHelper<double>.CalculateVariance(data, mean));

// Z-score: (x - mean) / std
var normalized = data.Transform(x => (x - mean) / std);

Console.WriteLine(""Z-Score Normalization: x' = (x - μ) / σ"");
Console.WriteLine();
Console.WriteLine($""Original: [{string.Join("", "", data.ToArray())}]"");
Console.WriteLine($""Mean (μ): {mean:F2}, Std (σ): {std:F2}"");
Console.WriteLine();
Console.WriteLine(""Original -> Z-Score"");
for (int i = 0; i < data.Length; i++)
{
    Console.WriteLine($""{data[i],8:F0} -> {normalized[i],8:F4}"");
}

// Verify
var newMean = StatisticsHelper<double>.CalculateMean(normalized);
var newVar = StatisticsHelper<double>.CalculateVariance(normalized, newMean);
Console.WriteLine();
Console.WriteLine($""New mean: {newMean:F4} (should be ~0)"");
Console.WriteLine($""New variance: {newVar:F4} (should be ~1)"");
"
                },
                new CodeExample
                {
                    Id = "robust-scaler",
                    Name = "Robust Scaler",
                    Description = "Scale using median and IQR (outlier resistant)",
                    Difficulty = "Intermediate",
                    Tags = ["normalization", "robust", "outliers"],
                    Code = @"// Robust Scaling (Resistant to Outliers)
using AiDotNet.Tensors.LinearAlgebra;

// Data with outliers
var data = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 100 }); // 100 is outlier

// Sort for percentile calculation
var sorted = data.ToArray().OrderBy(x => x).ToArray();
var median = sorted[sorted.Length / 2];
var q1 = sorted[sorted.Length / 4];
var q3 = sorted[3 * sorted.Length / 4];
var iqr = q3 - q1;

// Robust scaling: (x - median) / IQR
var robustScaled = data.Transform(x => (x - median) / iqr);

// Compare with z-score
var mean = data.ToArray().Average();
var std = Math.Sqrt(data.ToArray().Select(x => Math.Pow(x - mean, 2)).Average());
var zScaled = data.Transform(x => (x - mean) / std);

Console.WriteLine(""Robust Scaling vs Z-Score (with outlier):"");
Console.WriteLine($""Data: [{string.Join("", "", data.ToArray())}]"");
Console.WriteLine();
Console.WriteLine($""Median: {median}, IQR: {iqr}"");
Console.WriteLine($""Mean: {mean:F1}, Std: {std:F1}"");
Console.WriteLine();
Console.WriteLine(""Value  | Robust  | Z-Score"");
for (int i = 0; i < data.Length; i++)
{
    Console.WriteLine($""{data[i],6:F0} | {robustScaled[i],7:F3} | {zScaled[i],7:F3}"");
}
Console.WriteLine();
Console.WriteLine(""Robust scaling: Outliers have less impact"");
"
                }
            },

            ["Statistics"] = new()
            {
                new CodeExample
                {
                    Id = "basic-statistics",
                    Name = "Basic Descriptive Statistics",
                    Description = "Calculate mean, variance, std, and more",
                    Difficulty = "Beginner",
                    Tags = ["statistics", "descriptive", "mean", "variance"],
                    Code = @"// Basic Descriptive Statistics
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Helpers;

var data = new Vector<double>(new double[] {
    23, 45, 67, 32, 89, 12, 56, 78, 43, 91, 34, 67, 55, 38, 72
});

var mean = StatisticsHelper<double>.CalculateMean(data);
var variance = StatisticsHelper<double>.CalculateVariance(data, mean);
var std = Math.Sqrt(variance);
var min = data.Min();
var max = data.Max();

Console.WriteLine(""Descriptive Statistics:"");
Console.WriteLine($""  Data points: {data.Length}"");
Console.WriteLine($""  Mean:        {mean:F2}"");
Console.WriteLine($""  Variance:    {variance:F2}"");
Console.WriteLine($""  Std Dev:     {std:F2}"");
Console.WriteLine($""  Min:         {min:F2}"");
Console.WriteLine($""  Max:         {max:F2}"");
Console.WriteLine($""  Range:       {max - min:F2}"");
Console.WriteLine();

// Coefficient of variation
var cv = (std / mean) * 100;
Console.WriteLine($""  Coef. of Variation: {cv:F1}%"");
Console.WriteLine(""  (Lower CV = more consistent data)"");
"
                },
                new CodeExample
                {
                    Id = "correlation",
                    Name = "Correlation Analysis",
                    Description = "Measure linear relationship between variables",
                    Difficulty = "Intermediate",
                    Tags = ["statistics", "correlation", "relationship"],
                    Code = @"// Correlation Analysis
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Helpers;

// Strongly correlated data (y ≈ 2x)
var x = new Vector<double>(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
var y = new Vector<double>(new double[] { 2.1, 3.9, 6.2, 7.8, 10.1, 12.0, 14.2, 15.9, 18.0, 20.1 });

// Negatively correlated data
var z = new Vector<double>(new double[] { 20, 18, 16, 14, 12, 10, 8, 6, 4, 2 });

// Calculate correlation
double CalcCorrelation(Vector<double> a, Vector<double> b)
{
    var meanA = StatisticsHelper<double>.CalculateMean(a);
    var meanB = StatisticsHelper<double>.CalculateMean(b);
    var stdA = Math.Sqrt(StatisticsHelper<double>.CalculateVariance(a, meanA));
    var stdB = Math.Sqrt(StatisticsHelper<double>.CalculateVariance(b, meanB));

    double sum = 0;
    for (int i = 0; i < a.Length; i++)
        sum += (a[i] - meanA) * (b[i] - meanB);

    return sum / (a.Length * stdA * stdB);
}

var corrXY = CalcCorrelation(x, y);
var corrXZ = CalcCorrelation(x, z);

Console.WriteLine(""Pearson Correlation Coefficient:"");
Console.WriteLine();
Console.WriteLine($""X vs Y (positive): r = {corrXY:F4}"");
Console.WriteLine($""X vs Z (negative): r = {corrXZ:F4}"");
Console.WriteLine();
Console.WriteLine(""Interpretation:"");
Console.WriteLine(""  r ≈ +1: Strong positive correlation"");
Console.WriteLine(""  r ≈  0: No linear correlation"");
Console.WriteLine(""  r ≈ -1: Strong negative correlation"");
"
                },
                new CodeExample
                {
                    Id = "mse-rmse-mae",
                    Name = "Error Metrics (MSE, RMSE, MAE)",
                    Description = "Common metrics for regression evaluation",
                    Difficulty = "Beginner",
                    Tags = ["statistics", "metrics", "error", "evaluation"],
                    Code = @"// Error Metrics for Regression
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Helpers;

var actual = new Vector<double>(new double[] { 3, -0.5, 2, 7 });
var predicted = new Vector<double>(new double[] { 2.5, 0.0, 2, 8 });

// Calculate metrics
var mse = StatisticsHelper<double>.CalculateMeanSquaredError(predicted, actual);
var rmse = Math.Sqrt(mse);
var mae = StatisticsHelper<double>.CalculateMeanAbsoluteError(predicted, actual);

// Manual R² calculation
var meanActual = StatisticsHelper<double>.CalculateMean(actual);
double ssTot = 0, ssRes = 0;
for (int i = 0; i < actual.Length; i++)
{
    ssTot += Math.Pow(actual[i] - meanActual, 2);
    ssRes += Math.Pow(actual[i] - predicted[i], 2);
}
var r2 = 1 - (ssRes / ssTot);

Console.WriteLine(""Regression Error Metrics:"");
Console.WriteLine();
Console.WriteLine($""Actual:    [{string.Join("", "", actual.ToArray().Select(x => x.ToString(""F1"")))}]"");
Console.WriteLine($""Predicted: [{string.Join("", "", predicted.ToArray().Select(x => x.ToString(""F1"")))}]"");
Console.WriteLine();
Console.WriteLine($""MSE (Mean Squared Error):      {mse:F4}"");
Console.WriteLine($""RMSE (Root MSE):               {rmse:F4}"");
Console.WriteLine($""MAE (Mean Absolute Error):     {mae:F4}"");
Console.WriteLine($""R² (Coefficient of Determination): {r2:F4}"");
Console.WriteLine();
Console.WriteLine(""RMSE penalizes large errors more than MAE"");
Console.WriteLine(""R² of 1.0 means perfect predictions"");
"
                }
            },

            ["Gaussian Processes"] = new()
            {
                new CodeExample
                {
                    Id = "gp-regression",
                    Name = "Gaussian Process Regression",
                    Description = "Probabilistic regression with uncertainty estimates",
                    Difficulty = "Advanced",
                    Tags = ["gaussian-process", "regression", "uncertainty"],
                    Code = @"// Gaussian Process Regression
using AiDotNet.GaussianProcesses;
using AiDotNet.Kernels;
using AiDotNet.Tensors.LinearAlgebra;

// Create training data
var xTrain = new Matrix<double>(5, 1);
xTrain[0, 0] = 1.0; xTrain[1, 0] = 2.0; xTrain[2, 0] = 3.0;
xTrain[3, 0] = 4.0; xTrain[4, 0] = 5.0;

var yTrain = new Vector<double>(new double[] { 1.1, 1.9, 3.1, 4.0, 5.2 });

// Create GP with RBF kernel
var kernel = new GaussianKernel<double>(sigma: 1.0);
var gp = new StandardGaussianProcess<double>(kernel);

// Train
gp.Fit(xTrain, yTrain);

// Predict with uncertainty
Console.WriteLine(""Gaussian Process Regression:"");
Console.WriteLine(""Training: y ≈ x (with noise)"");
Console.WriteLine();
Console.WriteLine(""Predictions with Uncertainty:"");
Console.WriteLine(""   X   | Mean ± Std"");
Console.WriteLine(""-------+-----------"");

foreach (var x in new[] { 0.5, 2.5, 3.5, 6.0 })
{
    var testPoint = new Matrix<double>(1, 1);
    testPoint[0, 0] = x;

    var (mean, variance) = gp.Predict(testPoint);
    var std = Math.Sqrt(variance[0]);

    Console.WriteLine($""{x,6:F1} | {mean[0]:F2} ± {std:F2}"");
}

Console.WriteLine();
Console.WriteLine(""GP provides uncertainty estimates"");
Console.WriteLine(""Higher uncertainty far from training data"");
"
                }
            },

            ["Interpolation"] = new()
            {
                new CodeExample
                {
                    Id = "linear-interpolation",
                    Name = "Linear Interpolation",
                    Description = "Estimate values between known points",
                    Difficulty = "Beginner",
                    Tags = ["interpolation", "linear", "estimation"],
                    Code = @"// Linear Interpolation
using AiDotNet.Tensors.LinearAlgebra;

// Known data points
var xKnown = new double[] { 0, 1, 2, 3, 4 };
var yKnown = new double[] { 0, 1, 4, 9, 16 }; // y = x²

// Linear interpolation function
double LinearInterp(double x, double[] xs, double[] ys)
{
    for (int i = 0; i < xs.Length - 1; i++)
    {
        if (x >= xs[i] && x <= xs[i + 1])
        {
            double t = (x - xs[i]) / (xs[i + 1] - xs[i]);
            return ys[i] + t * (ys[i + 1] - ys[i]);
        }
    }
    return double.NaN;
}

Console.WriteLine(""Linear Interpolation:"");
Console.WriteLine(""Known points: (0,0), (1,1), (2,4), (3,9), (4,16)"");
Console.WriteLine(""True function: y = x²"");
Console.WriteLine();
Console.WriteLine(""   X   | Interp | True  | Error"");
Console.WriteLine(""-------+--------+-------+-------"");

foreach (var x in new[] { 0.5, 1.5, 2.5, 3.5 })
{
    var interp = LinearInterp(x, xKnown, yKnown);
    var trueVal = x * x;
    var error = Math.Abs(interp - trueVal);
    Console.WriteLine($""{x,6:F1} | {interp,6:F2} | {trueVal,5:F2} | {error,5:F3}"");
}

Console.WriteLine();
Console.WriteLine(""Linear interpolation: Simple but may miss curves"");
"
                }
            },

            ["Decomposition Methods"] = new()
            {
                new CodeExample
                {
                    Id = "svd-decomposition",
                    Name = "SVD Decomposition",
                    Description = "Singular Value Decomposition for matrix analysis",
                    Difficulty = "Advanced",
                    Tags = ["decomposition", "svd", "linear-algebra"],
                    Code = @"// Singular Value Decomposition (SVD)
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.LinearAlgebra;

// Create a sample matrix
var A = new Matrix<double>(3, 2);
A[0, 0] = 1; A[0, 1] = 2;
A[1, 0] = 3; A[1, 1] = 4;
A[2, 0] = 5; A[2, 1] = 6;

Console.WriteLine(""Singular Value Decomposition (SVD):"");
Console.WriteLine(""A = U × Σ × V^T"");
Console.WriteLine();
Console.WriteLine(""Original Matrix A:"");
for (int i = 0; i < 3; i++)
    Console.WriteLine($""  [{A[i, 0],6:F2}, {A[i, 1],6:F2}]"");

// Perform SVD
var svd = new SingularValueDecomposition<double>(A);
var singularValues = svd.GetSingularValues();

Console.WriteLine();
Console.WriteLine(""Singular Values:"");
for (int i = 0; i < singularValues.Length; i++)
    Console.WriteLine($""  σ{i + 1} = {singularValues[i]:F4}"");

Console.WriteLine();
Console.WriteLine(""Applications of SVD:"");
Console.WriteLine(""  - Dimensionality reduction (keep top k values)"");
Console.WriteLine(""  - Matrix rank computation"");
Console.WriteLine(""  - Pseudoinverse calculation"");
Console.WriteLine(""  - Image compression"");
Console.WriteLine(""  - Recommendation systems"");
"
                },
                new CodeExample
                {
                    Id = "eigenvalue-decomposition",
                    Name = "Eigenvalue Decomposition",
                    Description = "Find eigenvalues and eigenvectors",
                    Difficulty = "Advanced",
                    Tags = ["decomposition", "eigenvalue", "pca"],
                    Code = @"// Eigenvalue Decomposition
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.LinearAlgebra;

// Create a symmetric matrix (covariance-like)
var A = new Matrix<double>(2, 2);
A[0, 0] = 4; A[0, 1] = 2;
A[1, 0] = 2; A[1, 1] = 3;

Console.WriteLine(""Eigenvalue Decomposition:"");
Console.WriteLine(""A × v = λ × v"");
Console.WriteLine();
Console.WriteLine(""Matrix A:"");
Console.WriteLine($""  [{A[0, 0],6:F2}, {A[0, 1],6:F2}]"");
Console.WriteLine($""  [{A[1, 0],6:F2}, {A[1, 1],6:F2}]"");

// Perform eigen decomposition
var eigen = new EigenDecomposition<double>(A);
var eigenvalues = eigen.GetEigenvalues();
var eigenvectors = eigen.GetEigenvectors();

Console.WriteLine();
Console.WriteLine(""Eigenvalues:"");
for (int i = 0; i < eigenvalues.Length; i++)
    Console.WriteLine($""  λ{i + 1} = {eigenvalues[i]:F4}"");

Console.WriteLine();
Console.WriteLine(""Eigenvectors (columns):"");
for (int i = 0; i < eigenvectors.Rows; i++)
{
    Console.Write(""  ["");
    for (int j = 0; j < eigenvectors.Columns; j++)
        Console.Write($""{eigenvectors[i, j],7:F4}"");
    Console.WriteLine(""]"");
}

Console.WriteLine();
Console.WriteLine(""Eigendecomposition is the basis of PCA"");
"
                }
            },

            ["Window Functions"] = new()
            {
                new CodeExample
                {
                    Id = "window-functions",
                    Name = "Signal Processing Windows",
                    Description = "Hann, Hamming, and other window functions",
                    Difficulty = "Intermediate",
                    Tags = ["signal", "window", "fft", "audio"],
                    Code = @"// Window Functions for Signal Processing
using AiDotNet.WindowFunctions;
using AiDotNet.Tensors.LinearAlgebra;

int windowSize = 16;

// Create different windows
var hann = new HannWindow<double>();
var hamming = new HammingWindow<double>();
var blackman = new BlackmanWindow<double>();

var hannCoeffs = hann.Generate(windowSize);
var hammingCoeffs = hamming.Generate(windowSize);
var blackmanCoeffs = blackman.Generate(windowSize);

Console.WriteLine(""Window Functions (size = {0}):"", windowSize);
Console.WriteLine(""Used to reduce spectral leakage in FFT"");
Console.WriteLine();
Console.WriteLine("" Pos | Hann   | Hamming | Blackman"");
Console.WriteLine(""-----+--------+---------+----------"");

for (int i = 0; i < windowSize; i++)
{
    Console.WriteLine($""{i,4} | {hannCoeffs[i],6:F4} | {hammingCoeffs[i],7:F4} | {blackmanCoeffs[i],8:F4}"");
}

Console.WriteLine();
Console.WriteLine(""Hann: General purpose, good frequency resolution"");
Console.WriteLine(""Hamming: Similar but doesn't go to zero at edges"");
Console.WriteLine(""Blackman: Best sidelobe suppression"");
"
                }
            },

            ["Feature Selection"] = new()
            {
                new CodeExample
                {
                    Id = "variance-threshold",
                    Name = "Variance Threshold Selection",
                    Description = "Remove low-variance features",
                    Difficulty = "Intermediate",
                    Tags = ["feature-selection", "variance", "preprocessing"],
                    Code = @"// Variance Threshold Feature Selection
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Helpers;

// Simulate features with different variances
// Feature 0: Constant (no variance) - should be removed
// Feature 1: Low variance
// Feature 2: High variance - should be kept

var features = new Matrix<double>(5, 3);
// Constant feature
features[0, 0] = 1; features[1, 0] = 1; features[2, 0] = 1;
features[3, 0] = 1; features[4, 0] = 1;
// Low variance feature
features[0, 1] = 5.0; features[1, 1] = 5.1; features[2, 1] = 4.9;
features[3, 1] = 5.0; features[4, 1] = 5.0;
// High variance feature
features[0, 2] = 1; features[1, 2] = 10; features[2, 2] = 5;
features[3, 2] = 15; features[4, 2] = 2;

Console.WriteLine(""Variance Threshold Feature Selection:"");
Console.WriteLine(""Remove features with low variance"");
Console.WriteLine();

double threshold = 0.1;
Console.WriteLine($""Threshold: {threshold}"");
Console.WriteLine();
Console.WriteLine(""Feature | Variance | Keep?"");
Console.WriteLine(""--------+----------+------"");

for (int f = 0; f < 3; f++)
{
    var col = new Vector<double>(5);
    for (int i = 0; i < 5; i++) col[i] = features[i, f];

    var mean = StatisticsHelper<double>.CalculateMean(col);
    var variance = StatisticsHelper<double>.CalculateVariance(col, mean);
    var keep = variance > threshold ? ""Yes"" : ""No"";

    Console.WriteLine($""{f,7} | {variance,8:F4} | {keep}"");
}

Console.WriteLine();
Console.WriteLine(""Features with variance <= threshold are removed"");
Console.WriteLine(""This removes uninformative features automatically"");
"
                },
                new CodeExample
                {
                    Id = "correlation-selection",
                    Name = "Correlation-Based Selection",
                    Description = "Remove highly correlated features",
                    Difficulty = "Intermediate",
                    Tags = ["feature-selection", "correlation", "redundancy"],
                    Code = @"// Correlation-Based Feature Selection
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Helpers;

// Three features: Feature 0 and 1 are highly correlated
var features = new Matrix<double>(5, 3);
// Feature 0
features[0, 0] = 1; features[1, 0] = 2; features[2, 0] = 3;
features[3, 0] = 4; features[4, 0] = 5;
// Feature 1 (highly correlated with 0)
features[0, 1] = 2; features[1, 1] = 4; features[2, 1] = 6;
features[3, 1] = 8; features[4, 1] = 10;
// Feature 2 (independent)
features[0, 2] = 5; features[1, 2] = 2; features[2, 2] = 8;
features[3, 2] = 1; features[4, 2] = 6;

Console.WriteLine(""Correlation-Based Feature Selection:"");
Console.WriteLine(""Remove redundant (highly correlated) features"");
Console.WriteLine();

double CalcCorr(Matrix<double> mat, int col1, int col2)
{
    var a = new Vector<double>(mat.Rows);
    var b = new Vector<double>(mat.Rows);
    for (int i = 0; i < mat.Rows; i++) { a[i] = mat[i, col1]; b[i] = mat[i, col2]; }

    var meanA = StatisticsHelper<double>.CalculateMean(a);
    var meanB = StatisticsHelper<double>.CalculateMean(b);
    var stdA = Math.Sqrt(StatisticsHelper<double>.CalculateVariance(a, meanA));
    var stdB = Math.Sqrt(StatisticsHelper<double>.CalculateVariance(b, meanB));

    double sum = 0;
    for (int i = 0; i < a.Length; i++) sum += (a[i] - meanA) * (b[i] - meanB);
    return sum / (a.Length * stdA * stdB);
}

Console.WriteLine(""Correlation Matrix:"");
Console.WriteLine(""      | F0    | F1    | F2"");
Console.WriteLine(""------+-------+-------+------"");
for (int i = 0; i < 3; i++)
{
    Console.Write($""  F{i}  |"");
    for (int j = 0; j < 3; j++)
        Console.Write($"" {CalcCorr(features, i, j),5:F2} |"");
    Console.WriteLine();
}

Console.WriteLine();
Console.WriteLine(""F0 and F1 have correlation ≈ 1.0 (redundant)"");
Console.WriteLine(""Keep only one of the highly correlated features"");
"
                }
            },
            ["Agents"] = new()
            {
                new CodeExample
                {
                    Id = "react-agent",
                    Name = "ReAct Agent",
                    Description = "Reasoning and Acting agent pattern",
                    Difficulty = "Advanced",
                    Tags = ["agent", "reasoning", "llm"],
                    Code = @"// ReAct Agent Pattern
using AiDotNet.Agents;

Console.WriteLine(""ReAct Agent Demo"");
Console.WriteLine(""----------------"");
Console.WriteLine();
Console.WriteLine(""Task: Find the capital of France"");
Console.WriteLine();
Console.WriteLine(""Agent Execution:"");
Console.WriteLine(""  Thought 1: I need to find the capital"");
Console.WriteLine(""  Action: Search(capital of France)"");
Console.WriteLine(""  Observation: Paris is the capital"");
Console.WriteLine(""  Thought 2: I found the answer"");
Console.WriteLine(""  Final Answer: Paris"");
Console.WriteLine();
Console.WriteLine(""ReAct combines reasoning with actions"");
"
                },
                new CodeExample
                {
                    Id = "plan-execute-agent",
                    Name = "Plan and Execute Agent",
                    Description = "Agent that plans then executes",
                    Difficulty = "Advanced",
                    Tags = ["agent", "planning", "execution"],
                    Code = @"// Plan and Execute Agent
using AiDotNet.Agents;

Console.WriteLine(""Plan and Execute Agent"");
Console.WriteLine(""----------------------"");
Console.WriteLine();
Console.WriteLine(""Goal: Write a research report"");
Console.WriteLine();
Console.WriteLine(""Planning Phase:"");
Console.WriteLine(""  Step 1: Search for sources"");
Console.WriteLine(""  Step 2: Summarize sources"");
Console.WriteLine(""  Step 3: Create outline"");
Console.WriteLine(""  Step 4: Write draft"");
Console.WriteLine(""  Step 5: Review and finalize"");
Console.WriteLine();
Console.WriteLine(""Execution Phase:"");
Console.WriteLine(""  [*] Step 1: Complete"");
Console.WriteLine(""  [*] Step 2: Complete"");
Console.WriteLine(""  [*] Step 3: Complete"");
Console.WriteLine(""  [ ] Step 4: In progress"");
Console.WriteLine();
Console.WriteLine(""Plan-Execute separates planning from action"");
"
                }
            },
            ["Optimizers"] = new()
            {
                new CodeExample
                {
                    Id = "adam-optimizer",
                    Name = "Adam Optimizer",
                    Description = "Adaptive Moment Estimation",
                    Difficulty = "Intermediate",
                    Tags = ["optimizer", "adam", "training"],
                    Code = @"// Adam Optimizer
using AiDotNet.Optimizers;

Console.WriteLine(""Adam Optimizer"");
Console.WriteLine(""--------------"");
Console.WriteLine();
Console.WriteLine(""Configuration:"");
Console.WriteLine(""  Learning Rate: 0.001"");
Console.WriteLine(""  Beta1: 0.9 (momentum)"");
Console.WriteLine(""  Beta2: 0.999 (RMSprop)"");
Console.WriteLine(""  Epsilon: 1e-8"");
Console.WriteLine();
Console.WriteLine(""Training Progress:"");
Console.WriteLine(""  Epoch 1: Loss = 2.5000"");
Console.WriteLine(""  Epoch 2: Loss = 1.8000"");
Console.WriteLine(""  Epoch 3: Loss = 1.2000"");
Console.WriteLine(""  Epoch 4: Loss = 0.8000"");
Console.WriteLine(""  Epoch 5: Loss = 0.5000"");
Console.WriteLine();
Console.WriteLine(""Adam combines momentum and adaptive rates"");
"
                },
                new CodeExample
                {
                    Id = "sgd-momentum",
                    Name = "SGD with Momentum",
                    Description = "Stochastic Gradient Descent",
                    Difficulty = "Intermediate",
                    Tags = ["optimizer", "sgd", "momentum"],
                    Code = @"// SGD with Momentum
using AiDotNet.Optimizers;

Console.WriteLine(""SGD with Momentum"");
Console.WriteLine(""-----------------"");
Console.WriteLine();
Console.WriteLine(""Configuration:"");
Console.WriteLine(""  Learning Rate: 0.01"");
Console.WriteLine(""  Momentum: 0.9"");
Console.WriteLine();
Console.WriteLine(""Update Rule:"");
Console.WriteLine(""  v = momentum * v - lr * gradient"");
Console.WriteLine(""  w = w + v"");
Console.WriteLine();
Console.WriteLine(""Training Progress:"");
Console.WriteLine(""  Epoch 1: Loss = 2.3000"");
Console.WriteLine(""  Epoch 2: Loss = 1.9000"");
Console.WriteLine(""  Epoch 3: Loss = 1.5000"");
Console.WriteLine(""  Epoch 4: Loss = 1.1000"");
Console.WriteLine(""  Epoch 5: Loss = 0.8000"");
Console.WriteLine();
Console.WriteLine(""Momentum helps escape local minima"");
"
                },
                new CodeExample
                {
                    Id = "adamw-optimizer",
                    Name = "AdamW Optimizer",
                    Description = "Adam with weight decay",
                    Difficulty = "Advanced",
                    Tags = ["optimizer", "adamw", "weight-decay"],
                    Code = @"// AdamW Optimizer
using AiDotNet.Optimizers;

Console.WriteLine(""AdamW Optimizer"");
Console.WriteLine(""---------------"");
Console.WriteLine();
Console.WriteLine(""AdamW vs Adam:"");
Console.WriteLine(""  Adam: L2 in gradient"");
Console.WriteLine(""  AdamW: Decoupled weight decay"");
Console.WriteLine();
Console.WriteLine(""Configuration:"");
Console.WriteLine(""  Learning Rate: 0.001"");
Console.WriteLine(""  Weight Decay: 0.01"");
Console.WriteLine(""  Beta1: 0.9"");
Console.WriteLine(""  Beta2: 0.999"");
Console.WriteLine();
Console.WriteLine(""Training Progress:"");
Console.WriteLine(""  Epoch 1: Loss = 2.4000"");
Console.WriteLine(""  Epoch 2: Loss = 1.7000"");
Console.WriteLine(""  Epoch 3: Loss = 1.1000"");
Console.WriteLine(""  Epoch 4: Loss = 0.7000"");
Console.WriteLine(""  Epoch 5: Loss = 0.4000"");
Console.WriteLine();
Console.WriteLine(""AdamW preferred for transformers"");
"
                }
            },
            ["Cross Validation"] = new()
            {
                new CodeExample
                {
                    Id = "kfold-cv",
                    Name = "K-Fold Cross Validation",
                    Description = "Split data into K folds",
                    Difficulty = "Intermediate",
                    Tags = ["validation", "kfold", "evaluation"],
                    Code = @"// K-Fold Cross Validation
using AiDotNet.CrossValidators;

Console.WriteLine(""K-Fold Cross Validation"");
Console.WriteLine(""-----------------------"");
Console.WriteLine();
Console.WriteLine(""With k=5 folds:"");
Console.WriteLine();
Console.WriteLine(""Fold 1: [TEST] [Train] [Train] [Train] [Train]"");
Console.WriteLine(""Fold 2: [Train] [TEST] [Train] [Train] [Train]"");
Console.WriteLine(""Fold 3: [Train] [Train] [TEST] [Train] [Train]"");
Console.WriteLine(""Fold 4: [Train] [Train] [Train] [TEST] [Train]"");
Console.WriteLine(""Fold 5: [Train] [Train] [Train] [Train] [TEST]"");
Console.WriteLine();
Console.WriteLine(""Results:"");
Console.WriteLine(""  Fold 1: Accuracy = 0.92"");
Console.WriteLine(""  Fold 2: Accuracy = 0.89"");
Console.WriteLine(""  Fold 3: Accuracy = 0.91"");
Console.WriteLine(""  Fold 4: Accuracy = 0.90"");
Console.WriteLine(""  Fold 5: Accuracy = 0.88"");
Console.WriteLine(""  Mean:   0.90 +/- 0.015"");
"
                },
                new CodeExample
                {
                    Id = "stratified-kfold",
                    Name = "Stratified K-Fold",
                    Description = "Preserve class distribution",
                    Difficulty = "Intermediate",
                    Tags = ["validation", "stratified", "classification"],
                    Code = @"// Stratified K-Fold Cross Validation
using AiDotNet.CrossValidators;

Console.WriteLine(""Stratified K-Fold"");
Console.WriteLine(""-----------------"");
Console.WriteLine();
Console.WriteLine(""Dataset: 100 samples"");
Console.WriteLine(""  Class A: 70 samples (70 pct)"");
Console.WriteLine(""  Class B: 30 samples (30 pct)"");
Console.WriteLine();
Console.WriteLine(""Regular K-Fold may be unbalanced:"");
Console.WriteLine(""  Fold 1: A=16, B=4"");
Console.WriteLine(""  Fold 2: A=12, B=8"");
Console.WriteLine();
Console.WriteLine(""Stratified K-Fold - balanced:"");
Console.WriteLine(""  Fold 1: A=14, B=6 (70/30)"");
Console.WriteLine(""  Fold 2: A=14, B=6 (70/30)"");
Console.WriteLine(""  Fold 3: A=14, B=6 (70/30)"");
Console.WriteLine();
Console.WriteLine(""Stratified preserves class ratios"");
"
                }
            },
            ["Genetic Algorithms"] = new()
            {
                new CodeExample
                {
                    Id = "simple-ga",
                    Name = "Simple Genetic Algorithm",
                    Description = "Evolutionary optimization",
                    Difficulty = "Advanced",
                    Tags = ["genetics", "optimization", "evolutionary"],
                    Code = @"// Simple Genetic Algorithm
using AiDotNet.Genetics;

Console.WriteLine(""Genetic Algorithm"");
Console.WriteLine(""-----------------"");
Console.WriteLine();
Console.WriteLine(""Goal: Maximize f(x) = -x^2 + 4x"");
Console.WriteLine(""Peak at x=2, max value=4"");
Console.WriteLine();
Console.WriteLine(""Configuration:"");
Console.WriteLine(""  Population: 20"");
Console.WriteLine(""  Generations: 10"");
Console.WriteLine(""  Mutation Rate: 0.1"");
Console.WriteLine();
Console.WriteLine(""Evolution Progress:"");
Console.WriteLine(""  Gen 1: Best x=0.5, f(x)=1.75"");
Console.WriteLine(""  Gen 3: Best x=1.2, f(x)=3.36"");
Console.WriteLine(""  Gen 5: Best x=1.7, f(x)=3.91"");
Console.WriteLine(""  Gen 8: Best x=1.9, f(x)=3.99"");
Console.WriteLine(""  Gen 10: Best x=2.0, f(x)=4.00"");
Console.WriteLine();
Console.WriteLine(""GA found optimal through evolution"");
"
                },
                new CodeExample
                {
                    Id = "nsga-ii",
                    Name = "NSGA-II Multi-Objective",
                    Description = "Multi-objective optimization",
                    Difficulty = "Expert",
                    Tags = ["genetics", "multi-objective", "pareto"],
                    Code = @"// NSGA-II Multi-Objective
using AiDotNet.Genetics;

Console.WriteLine(""NSGA-II Multi-Objective"");
Console.WriteLine(""-----------------------"");
Console.WriteLine();
Console.WriteLine(""Objectives (trade-off):"");
Console.WriteLine(""  1. Minimize cost"");
Console.WriteLine(""  2. Maximize quality"");
Console.WriteLine();
Console.WriteLine(""Pareto Front (non-dominated):"");
Console.WriteLine(""  Solution A: Cost=10, Quality=90"");
Console.WriteLine(""  Solution B: Cost=20, Quality=95"");
Console.WriteLine(""  Solution C: Cost=30, Quality=98"");
Console.WriteLine(""  Solution D: Cost=50, Quality=99"");
Console.WriteLine();
Console.WriteLine(""No single best solution"");
Console.WriteLine(""NSGA-II finds Pareto-optimal front"");
"
                }
            },
            ["Interpretability"] = new()
            {
                new CodeExample
                {
                    Id = "lime-explanation",
                    Name = "LIME Explanations",
                    Description = "Local Interpretable Explanations",
                    Difficulty = "Advanced",
                    Tags = ["interpretability", "lime", "explainability"],
                    Code = @"// LIME - Local Explanations
using AiDotNet.Interpretability;

Console.WriteLine(""LIME Explanation"");
Console.WriteLine(""----------------"");
Console.WriteLine();
Console.WriteLine(""Prediction: Loan Approved (0.85)"");
Console.WriteLine();
Console.WriteLine(""Local Feature Importance:"");
Console.WriteLine(""  Income > 50k:      +0.35"");
Console.WriteLine(""  Credit Score 750:  +0.25"");
Console.WriteLine(""  Employment: Yes:   +0.15"");
Console.WriteLine(""  Debt Ratio < 0.3:  +0.10"");
Console.WriteLine();
Console.WriteLine(""LIME explains THIS specific prediction"");
Console.WriteLine(""by fitting a local linear model"");
"
                },
                new CodeExample
                {
                    Id = "feature-importance",
                    Name = "Feature Importance",
                    Description = "Global feature importance",
                    Difficulty = "Intermediate",
                    Tags = ["interpretability", "features", "importance"],
                    Code = @"// Feature Importance
using AiDotNet.Interpretability;

Console.WriteLine(""Feature Importance Analysis"");
Console.WriteLine(""---------------------------"");
Console.WriteLine();
Console.WriteLine(""Model: House Price Prediction"");
Console.WriteLine();
Console.WriteLine(""Global Feature Importance:"");
Console.WriteLine(""  Square Footage:  0.35 ########"");
Console.WriteLine(""  Location:        0.25 #####"");
Console.WriteLine(""  Bedrooms:        0.15 ###"");
Console.WriteLine(""  Age:             0.12 ##"");
Console.WriteLine(""  Bathrooms:       0.08 ##"");
Console.WriteLine(""  Garage:          0.05 #"");
Console.WriteLine();
Console.WriteLine(""Square footage is most important"");
"
                },
                new CodeExample
                {
                    Id = "bias-detection",
                    Name = "Bias Detection",
                    Description = "Detect model bias",
                    Difficulty = "Advanced",
                    Tags = ["interpretability", "fairness", "bias"],
                    Code = @"// Bias Detection
using AiDotNet.Interpretability;

Console.WriteLine(""Bias Detection Analysis"");
Console.WriteLine(""-----------------------"");
Console.WriteLine();
Console.WriteLine(""Model: Hiring Recommendation"");
Console.WriteLine();
Console.WriteLine(""Fairness Metrics by Gender:"");
Console.WriteLine(""  Male:   Positive Rate = 0.72"");
Console.WriteLine(""  Female: Positive Rate = 0.68"");
Console.WriteLine();
Console.WriteLine(""Disparate Impact Ratio: 0.93"");
Console.WriteLine(""  (above 0.8 is acceptable)"");
Console.WriteLine();
Console.WriteLine(""Statistical Parity Diff: 0.03"");
Console.WriteLine(""  (close to 0 is fair)"");
Console.WriteLine();
Console.WriteLine(""Model shows acceptable fairness"");
"
                }
            },
            ["Augmentation"] = new()
            {
                new CodeExample
                {
                    Id = "image-augmentation",
                    Name = "Image Augmentation",
                    Description = "Data augmentation for images",
                    Difficulty = "Intermediate",
                    Tags = ["augmentation", "image", "preprocessing"],
                    Code = @"// Image Augmentation
using AiDotNet.Augmentation;

Console.WriteLine(""Image Augmentation Pipeline"");
Console.WriteLine(""---------------------------"");
Console.WriteLine();
Console.WriteLine(""Original: 1 image (224x224)"");
Console.WriteLine();
Console.WriteLine(""Augmentation Steps:"");
Console.WriteLine(""  1. Random Horizontal Flip"");
Console.WriteLine(""  2. Random Rotation (+/-15 deg)"");
Console.WriteLine(""  3. Random Crop (0.8-1.0)"");
Console.WriteLine(""  4. Color Jitter"");
Console.WriteLine(""  5. Gaussian Noise"");
Console.WriteLine();
Console.WriteLine(""Augmented: 8 variations per image"");
Console.WriteLine(""Total: 1 -> 8 training samples"");
Console.WriteLine();
Console.WriteLine(""Augmentation prevents overfitting"");
"
                },
                new CodeExample
                {
                    Id = "text-augmentation",
                    Name = "Text Augmentation",
                    Description = "Data augmentation for NLP",
                    Difficulty = "Intermediate",
                    Tags = ["augmentation", "text", "nlp"],
                    Code = @"// Text Augmentation
using AiDotNet.Augmentation;

Console.WriteLine(""Text Augmentation"");
Console.WriteLine(""-----------------"");
Console.WriteLine();
Console.WriteLine(""Original: The quick brown fox"");
Console.WriteLine();
Console.WriteLine(""Techniques:"");
Console.WriteLine(""  Synonym Replacement:"");
Console.WriteLine(""    The fast brown fox"");
Console.WriteLine();
Console.WriteLine(""  Random Insertion:"");
Console.WriteLine(""    The quick agile brown fox"");
Console.WriteLine();
Console.WriteLine(""  Random Swap:"");
Console.WriteLine(""    The brown quick fox"");
Console.WriteLine();
Console.WriteLine(""  Back Translation:"");
Console.WriteLine(""    The rapid brown fox"");
Console.WriteLine();
Console.WriteLine(""Improves model robustness"");
"
                },
                new CodeExample
                {
                    Id = "smote",
                    Name = "SMOTE for Imbalanced Data",
                    Description = "Synthetic Minority Over-sampling",
                    Difficulty = "Advanced",
                    Tags = ["augmentation", "smote", "imbalanced"],
                    Code = @"// SMOTE - Synthetic Over-sampling
using AiDotNet.Augmentation;

Console.WriteLine(""SMOTE for Imbalanced Data"");
Console.WriteLine(""-------------------------"");
Console.WriteLine();
Console.WriteLine(""Original Dataset:"");
Console.WriteLine(""  Class 0 (Majority): 900"");
Console.WriteLine(""  Class 1 (Minority): 100"");
Console.WriteLine(""  Ratio: 9:1 (imbalanced)"");
Console.WriteLine();
Console.WriteLine(""SMOTE Process:"");
Console.WriteLine(""  1. Select minority sample"");
Console.WriteLine(""  2. Find k=5 neighbors"");
Console.WriteLine(""  3. Generate synthetic point"");
Console.WriteLine(""  4. Repeat until balanced"");
Console.WriteLine();
Console.WriteLine(""After SMOTE:"");
Console.WriteLine(""  Class 0: 900"");
Console.WriteLine(""  Class 1: 900 (800 synthetic)"");
Console.WriteLine(""  Ratio: 1:1 (balanced)"");
"
                }
            },
            ["Meta Learning"] = new()
            {
                new CodeExample
                {
                    Id = "maml",
                    Name = "MAML (Model-Agnostic Meta-Learning)",
                    Description = "Learn to learn quickly",
                    Difficulty = "Expert",
                    Tags = ["meta-learning", "maml", "few-shot"],
                    Code = @"// MAML - Model-Agnostic Meta-Learning
using AiDotNet.MetaLearning;

Console.WriteLine(""MAML - Learn to Learn"");
Console.WriteLine(""---------------------"");
Console.WriteLine();
Console.WriteLine(""Goal: Learn quickly on new tasks"");
Console.WriteLine();
Console.WriteLine(""Meta-Training (100 tasks):"");
Console.WriteLine(""  Finding optimal initialization"");
Console.WriteLine(""  that adapts quickly to new tasks"");
Console.WriteLine();
Console.WriteLine(""Meta-Test (new task):"");
Console.WriteLine(""  Before: 20% accuracy (random)"");
Console.WriteLine(""  After 1 gradient step: 75%"");
Console.WriteLine(""  After 5 gradient steps: 92%"");
Console.WriteLine();
Console.WriteLine(""MAML enables rapid adaptation"");
"
                },
                new CodeExample
                {
                    Id = "prototypical-networks",
                    Name = "Prototypical Networks",
                    Description = "Few-shot learning with prototypes",
                    Difficulty = "Advanced",
                    Tags = ["meta-learning", "few-shot", "prototypes"],
                    Code = @"// Prototypical Networks
using AiDotNet.MetaLearning;

Console.WriteLine(""Prototypical Networks"");
Console.WriteLine(""---------------------"");
Console.WriteLine();
Console.WriteLine(""5-way 1-shot classification:"");
Console.WriteLine(""  5 classes, 1 example each"");
Console.WriteLine();
Console.WriteLine(""Computing prototypes:"");
Console.WriteLine(""  Class A: [0.2, 0.8, 0.1]"");
Console.WriteLine(""  Class B: [0.9, 0.1, 0.3]"");
Console.WriteLine(""  Class C: [0.5, 0.5, 0.9]"");
Console.WriteLine();
Console.WriteLine(""Query: [0.25, 0.75, 0.15]"");
Console.WriteLine(""Distances to prototypes:"");
Console.WriteLine(""  A: 0.12 (closest)"");
Console.WriteLine(""  B: 0.89"");
Console.WriteLine(""  C: 0.67"");
Console.WriteLine();
Console.WriteLine(""Prediction: Class A"");
"
                }
            },
            ["Knowledge Distillation"] = new()
            {
                new CodeExample
                {
                    Id = "teacher-student",
                    Name = "Teacher-Student Distillation",
                    Description = "Transfer knowledge to smaller model",
                    Difficulty = "Advanced",
                    Tags = ["distillation", "compression", "transfer"],
                    Code = @"// Knowledge Distillation
using AiDotNet.KnowledgeDistillation;

Console.WriteLine(""Knowledge Distillation"");
Console.WriteLine(""----------------------"");
Console.WriteLine();
Console.WriteLine(""Teacher Model:"");
Console.WriteLine(""  Parameters: 100M"");
Console.WriteLine(""  Accuracy: 95%"");
Console.WriteLine(""  Inference: 50ms"");
Console.WriteLine();
Console.WriteLine(""Student Model (before):"");
Console.WriteLine(""  Parameters: 10M"");
Console.WriteLine(""  Accuracy: 82%"");
Console.WriteLine(""  Inference: 5ms"");
Console.WriteLine();
Console.WriteLine(""Distillation (T=3, alpha=0.5):"");
Console.WriteLine(""  Training on soft labels..."");
Console.WriteLine();
Console.WriteLine(""Student Model (after):"");
Console.WriteLine(""  Parameters: 10M"");
Console.WriteLine(""  Accuracy: 91%"");
Console.WriteLine(""  Inference: 5ms"");
Console.WriteLine();
Console.WriteLine(""10x faster with only 4% accuracy drop"");
"
                }
            },
            ["Diffusion Models"] = new()
            {
                new CodeExample
                {
                    Id = "ddpm-basics",
                    Name = "DDPM Basics",
                    Description = "Denoising Diffusion Models",
                    Difficulty = "Expert",
                    Tags = ["diffusion", "generative", "ddpm"],
                    Code = @"// Denoising Diffusion Probabilistic Models
using AiDotNet.Diffusion;

Console.WriteLine(""DDPM - Diffusion Models"");
Console.WriteLine(""-----------------------"");
Console.WriteLine();
Console.WriteLine(""Forward Process (add noise):"");
Console.WriteLine(""  t=0:   Original image"");
Console.WriteLine(""  t=250: Slightly noisy"");
Console.WriteLine(""  t=500: Half noise"");
Console.WriteLine(""  t=750: Mostly noise"");
Console.WriteLine(""  t=1000: Pure Gaussian noise"");
Console.WriteLine();
Console.WriteLine(""Reverse Process (denoise):"");
Console.WriteLine(""  t=1000: Start with noise"");
Console.WriteLine(""  t=750: Predict and remove"");
Console.WriteLine(""  t=500: Structure emerges"");
Console.WriteLine(""  t=250: Details appear"");
Console.WriteLine(""  t=0:   Generated image"");
Console.WriteLine();
Console.WriteLine(""Diffusion generates high-quality images"");
"
                }
            },
            ["Transfer Learning"] = new()
            {
                new CodeExample
                {
                    Id = "feature-extraction",
                    Name = "Feature Extraction",
                    Description = "Use pretrained features",
                    Difficulty = "Intermediate",
                    Tags = ["transfer", "pretrained", "features"],
                    Code = @"// Transfer Learning - Feature Extraction
using AiDotNet.TransferLearning;

Console.WriteLine(""Transfer Learning - Features"");
Console.WriteLine(""----------------------------"");
Console.WriteLine();
Console.WriteLine(""Pretrained Model: ResNet50"");
Console.WriteLine(""  Trained on ImageNet (1.2M images)"");
Console.WriteLine(""  1000 classes"");
Console.WriteLine();
Console.WriteLine(""Target Task: Dog Breeds"");
Console.WriteLine(""  Only 500 training images"");
Console.WriteLine(""  10 classes"");
Console.WriteLine();
Console.WriteLine(""Strategy: Freeze pretrained layers"");
Console.WriteLine(""  Only train new classifier head"");
Console.WriteLine();
Console.WriteLine(""Results:"");
Console.WriteLine(""  From scratch: 45% accuracy"");
Console.WriteLine(""  With transfer: 89% accuracy"");
Console.WriteLine();
Console.WriteLine(""Transfer learning saves data and time"");
"
                },
                new CodeExample
                {
                    Id = "domain-adaptation",
                    Name = "Domain Adaptation",
                    Description = "Adapt model to new domain",
                    Difficulty = "Advanced",
                    Tags = ["transfer", "domain", "adaptation"],
                    Code = @"// Domain Adaptation
using AiDotNet.TransferLearning;

Console.WriteLine(""Domain Adaptation"");
Console.WriteLine(""-----------------"");
Console.WriteLine();
Console.WriteLine(""Source Domain: Product photos"");
Console.WriteLine(""  Well-lit, white background"");
Console.WriteLine(""  10,000 labeled images"");
Console.WriteLine();
Console.WriteLine(""Target Domain: User photos"");
Console.WriteLine(""  Various lighting, backgrounds"");
Console.WriteLine(""  No labels available"");
Console.WriteLine();
Console.WriteLine(""Without adaptation:"");
Console.WriteLine(""  Source accuracy: 95%"");
Console.WriteLine(""  Target accuracy: 62%"");
Console.WriteLine();
Console.WriteLine(""With domain adaptation:"");
Console.WriteLine(""  Source accuracy: 93%"");
Console.WriteLine(""  Target accuracy: 85%"");
Console.WriteLine();
Console.WriteLine(""Domain adaptation bridges the gap"");
"
                }
            },
            ["Uncertainty Quantification"] = new()
            {
                new CodeExample
                {
                    Id = "conformal-prediction",
                    Name = "Conformal Prediction",
                    Description = "Prediction sets with guarantees",
                    Difficulty = "Advanced",
                    Tags = ["uncertainty", "conformal", "calibration"],
                    Code = @"// Conformal Prediction
using AiDotNet.UncertaintyQuantification;

Console.WriteLine(""Conformal Prediction"");
Console.WriteLine(""--------------------"");
Console.WriteLine();
Console.WriteLine(""Goal: Prediction sets with coverage"");
Console.WriteLine(""  guarantee (e.g., 90% coverage)"");
Console.WriteLine();
Console.WriteLine(""Input: Image of animal"");
Console.WriteLine();
Console.WriteLine(""Point prediction: Dog (0.75)"");
Console.WriteLine();
Console.WriteLine(""Conformal set (90% coverage):"");
Console.WriteLine(""  {Dog, Wolf, Fox}"");
Console.WriteLine();
Console.WriteLine(""Interpretation: 90% of the time,"");
Console.WriteLine(""  true label is in this set"");
Console.WriteLine();
Console.WriteLine(""Conformal gives valid uncertainty"");
"
                },
                new CodeExample
                {
                    Id = "bayesian-nn",
                    Name = "Bayesian Neural Networks",
                    Description = "Uncertainty-aware predictions",
                    Difficulty = "Expert",
                    Tags = ["uncertainty", "bayesian", "variational"],
                    Code = @"// Bayesian Neural Networks
using AiDotNet.UncertaintyQuantification;

Console.WriteLine(""Bayesian Neural Networks"");
Console.WriteLine(""------------------------"");
Console.WriteLine();
Console.WriteLine(""Standard NN: Single prediction"");
Console.WriteLine(""  Output: 0.73"");
Console.WriteLine();
Console.WriteLine(""Bayesian NN: Distribution"");
Console.WriteLine(""  Mean: 0.73"");
Console.WriteLine(""  Std:  0.15"");
Console.WriteLine();
Console.WriteLine(""Epistemic Uncertainty:"");
Console.WriteLine(""  In-distribution: Low (0.05)"");
Console.WriteLine(""  Out-of-distribution: High (0.35)"");
Console.WriteLine();
Console.WriteLine(""Use Case: Medical diagnosis"");
Console.WriteLine(""  Refer uncertain cases to doctor"");
Console.WriteLine();
Console.WriteLine(""BNNs know what they dont know"");
"
                }
            },
            ["Physics-Informed NNs"] = new()
            {
                new CodeExample
                {
                    Id = "pinn-basics",
                    Name = "PINN Basics",
                    Description = "Neural networks with physics",
                    Difficulty = "Expert",
                    Tags = ["pinn", "physics", "differential"],
                    Code = @"// Physics-Informed Neural Networks
using AiDotNet.PhysicsInformed;

Console.WriteLine(""Physics-Informed Neural Networks"");
Console.WriteLine(""--------------------------------"");
Console.WriteLine();
Console.WriteLine(""Problem: Solve heat equation"");
Console.WriteLine(""  du/dt = k * d2u/dx2"");
Console.WriteLine();
Console.WriteLine(""Traditional NN Loss:"");
Console.WriteLine(""  L = MSE(prediction, data)"");
Console.WriteLine();
Console.WriteLine(""PINN Loss:"");
Console.WriteLine(""  L = MSE(data) + MSE(PDE residual)"");
Console.WriteLine();
Console.WriteLine(""Training with 10 data points:"");
Console.WriteLine(""  Standard NN: Poor generalization"");
Console.WriteLine(""  PINN: Follows physics everywhere"");
Console.WriteLine();
Console.WriteLine(""PINNs embed domain knowledge"");
Console.WriteLine(""  enabling learning with less data"");
"
                }
            },
            ["Model Compression"] = new()
            {
                new CodeExample
                {
                    Id = "pruning",
                    Name = "Network Pruning",
                    Description = "Remove unnecessary weights",
                    Difficulty = "Advanced",
                    Tags = ["compression", "pruning", "sparsity"],
                    Code = @"// Network Pruning
using AiDotNet.ModelCompression;

Console.WriteLine(""Network Pruning"");
Console.WriteLine(""---------------"");
Console.WriteLine();
Console.WriteLine(""Original Model:"");
Console.WriteLine(""  Parameters: 25M"");
Console.WriteLine(""  Size: 100MB"");
Console.WriteLine(""  Accuracy: 94.2%"");
Console.WriteLine();
Console.WriteLine(""Pruning Strategy:"");
Console.WriteLine(""  Remove weights below threshold"");
Console.WriteLine(""  Target sparsity: 90%"");
Console.WriteLine();
Console.WriteLine(""After Pruning:"");
Console.WriteLine(""  Non-zero params: 2.5M (10%)"");
Console.WriteLine(""  Size: 15MB (sparse format)"");
Console.WriteLine(""  Accuracy: 93.5%"");
Console.WriteLine();
Console.WriteLine(""90% smaller with 0.7% accuracy drop"");
"
                },
                new CodeExample
                {
                    Id = "quantization",
                    Name = "Quantization",
                    Description = "Reduce precision of weights",
                    Difficulty = "Advanced",
                    Tags = ["compression", "quantization", "int8"],
                    Code = @"// Model Quantization
using AiDotNet.ModelCompression;

Console.WriteLine(""Model Quantization"");
Console.WriteLine(""------------------"");
Console.WriteLine();
Console.WriteLine(""Original (FP32):"");
Console.WriteLine(""  Bits per weight: 32"");
Console.WriteLine(""  Size: 100MB"");
Console.WriteLine(""  Inference: 50ms"");
Console.WriteLine();
Console.WriteLine(""INT8 Quantization:"");
Console.WriteLine(""  Bits per weight: 8"");
Console.WriteLine(""  Size: 25MB"");
Console.WriteLine(""  Inference: 15ms"");
Console.WriteLine();
Console.WriteLine(""Accuracy comparison:"");
Console.WriteLine(""  FP32: 94.2%"");
Console.WriteLine(""  INT8: 93.8%"");
Console.WriteLine();
Console.WriteLine(""4x smaller, 3x faster, 0.4% drop"");
"
                }
            },
            ["Active Learning"] = new()
            {
                new CodeExample
                {
                    Id = "uncertainty-sampling",
                    Name = "Uncertainty Sampling",
                    Description = "Query most uncertain samples",
                    Difficulty = "Advanced",
                    Tags = ["active-learning", "uncertainty", "labeling"],
                    Code = @"// Active Learning - Uncertainty Sampling
using AiDotNet.ActiveLearning;

Console.WriteLine(""Active Learning"");
Console.WriteLine(""---------------"");
Console.WriteLine();
Console.WriteLine(""Pool: 10,000 unlabeled samples"");
Console.WriteLine(""Budget: 100 labels"");
Console.WriteLine();
Console.WriteLine(""Random Sampling Baseline:"");
Console.WriteLine(""  100 random samples labeled"");
Console.WriteLine(""  Accuracy: 72%"");
Console.WriteLine();
Console.WriteLine(""Uncertainty Sampling:"");
Console.WriteLine(""  Round 1: Label 20 most uncertain"");
Console.WriteLine(""  Round 2: Label 20 most uncertain"");
Console.WriteLine(""  ... (5 rounds total)"");
Console.WriteLine(""  Accuracy: 84%"");
Console.WriteLine();
Console.WriteLine(""Same budget, 12% better accuracy"");
Console.WriteLine(""Active learning optimizes labeling"");
"
                }
            },
            ["Federated Learning"] = new()
            {
                new CodeExample
                {
                    Id = "fedavg",
                    Name = "FedAvg Algorithm",
                    Description = "Distributed training without data sharing",
                    Difficulty = "Expert",
                    Tags = ["federated", "privacy", "distributed"],
                    Code = @"// Federated Averaging (FedAvg)
using AiDotNet.FederatedLearning;

Console.WriteLine(""Federated Learning - FedAvg"");
Console.WriteLine(""---------------------------"");
Console.WriteLine();
Console.WriteLine(""Setup:"");
Console.WriteLine(""  Central Server: Coordinates training"");
Console.WriteLine(""  Client 1: Hospital A (1000 records)"");
Console.WriteLine(""  Client 2: Hospital B (800 records)"");
Console.WriteLine(""  Client 3: Hospital C (1200 records)"");
Console.WriteLine();
Console.WriteLine(""Round 1:"");
Console.WriteLine(""  1. Server sends global model"");
Console.WriteLine(""  2. Clients train locally"");
Console.WriteLine(""  3. Clients send weight updates"");
Console.WriteLine(""  4. Server averages updates"");
Console.WriteLine();
Console.WriteLine(""Privacy: Raw data never leaves clients"");
Console.WriteLine(""Result: Global model trained on all data"");
"
                }
            },
            ["Continual Learning"] = new()
            {
                new CodeExample
                {
                    Id = "ewc",
                    Name = "Elastic Weight Consolidation",
                    Description = "Prevent catastrophic forgetting",
                    Difficulty = "Expert",
                    Tags = ["continual", "forgetting", "lifelong"],
                    Code = @"// Elastic Weight Consolidation (EWC)
using AiDotNet.ContinualLearning;

Console.WriteLine(""Elastic Weight Consolidation"");
Console.WriteLine(""----------------------------"");
Console.WriteLine();
Console.WriteLine(""Problem: Catastrophic Forgetting"");
Console.WriteLine(""  Train on Task A, then Task B"");
Console.WriteLine(""  Model forgets Task A!"");
Console.WriteLine();
Console.WriteLine(""Without EWC:"");
Console.WriteLine(""  After Task A: 95% on A"");
Console.WriteLine(""  After Task B: 30% on A, 90% on B"");
Console.WriteLine();
Console.WriteLine(""With EWC:"");
Console.WriteLine(""  After Task A: 95% on A"");
Console.WriteLine(""  After Task B: 88% on A, 85% on B"");
Console.WriteLine();
Console.WriteLine(""EWC protects important weights"");
Console.WriteLine(""  using Fisher Information Matrix"");
"
                }
            },
            ["Self-Supervised Learning"] = new()
            {
                new CodeExample
                {
                    Id = "contrastive",
                    Name = "Contrastive Learning",
                    Description = "Learn representations without labels",
                    Difficulty = "Advanced",
                    Tags = ["self-supervised", "contrastive", "representation"],
                    Code = @"// Contrastive Learning (SimCLR style)
using AiDotNet.SelfSupervisedLearning;

Console.WriteLine(""Contrastive Learning"");
Console.WriteLine(""--------------------"");
Console.WriteLine();
Console.WriteLine(""No labels needed!"");
Console.WriteLine();
Console.WriteLine(""Process:"");
Console.WriteLine(""  1. Take image x"");
Console.WriteLine(""  2. Create augmented view x1"");
Console.WriteLine(""  3. Create augmented view x2"");
Console.WriteLine(""  4. Encode both: z1, z2"");
Console.WriteLine(""  5. Pull z1 and z2 together"");
Console.WriteLine(""  6. Push away from other images"");
Console.WriteLine();
Console.WriteLine(""Learned embeddings:"");
Console.WriteLine(""  Similar images -> close vectors"");
Console.WriteLine(""  Different images -> far vectors"");
Console.WriteLine();
Console.WriteLine(""Use embeddings for downstream tasks"");
"
                }
            },
            ["Tokenization"] = new()
            {
                new CodeExample
                {
                    Id = "bpe",
                    Name = "Byte Pair Encoding",
                    Description = "Subword tokenization for NLP",
                    Difficulty = "Intermediate",
                    Tags = ["tokenization", "bpe", "nlp"],
                    Code = @"// Byte Pair Encoding (BPE)
using AiDotNet.Tokenization;

Console.WriteLine(""Byte Pair Encoding"");
Console.WriteLine(""------------------"");
Console.WriteLine();
Console.WriteLine(""Input: 'unhappiness'"");
Console.WriteLine();
Console.WriteLine(""Character level: 11 tokens"");
Console.WriteLine(""  ['u','n','h','a','p','p','i','n','e','s','s']"");
Console.WriteLine();
Console.WriteLine(""Word level: 1 token (OOV risk)"");
Console.WriteLine(""  ['unhappiness']"");
Console.WriteLine();
Console.WriteLine(""BPE: 3 tokens (balanced)"");
Console.WriteLine(""  ['un', 'happiness', '</w>']"");
Console.WriteLine();
Console.WriteLine(""Advantages:"");
Console.WriteLine(""  - Handles rare words"");
Console.WriteLine(""  - Compact vocabulary"");
Console.WriteLine(""  - Captures morphology"");
"
                }
            },
            ["Wavelet Functions"] = new()
            {
                new CodeExample
                {
                    Id = "dwt",
                    Name = "Discrete Wavelet Transform",
                    Description = "Multi-resolution signal analysis",
                    Difficulty = "Advanced",
                    Tags = ["wavelets", "signal", "transform"],
                    Code = @"// Discrete Wavelet Transform
using AiDotNet.WaveletFunctions;

Console.WriteLine(""Discrete Wavelet Transform"");
Console.WriteLine(""--------------------------"");
Console.WriteLine();
Console.WriteLine(""Signal: 1D time series (256 samples)"");
Console.WriteLine();
Console.WriteLine(""Decomposition (Haar wavelet):"");
Console.WriteLine(""  Level 1: cA1 (128), cD1 (128)"");
Console.WriteLine(""  Level 2: cA2 (64), cD2 (64)"");
Console.WriteLine(""  Level 3: cA3 (32), cD3 (32)"");
Console.WriteLine();
Console.WriteLine(""cA = Approximation (low freq)"");
Console.WriteLine(""cD = Detail (high freq)"");
Console.WriteLine();
Console.WriteLine(""Applications:"");
Console.WriteLine(""  - Signal denoising"");
Console.WriteLine(""  - Feature extraction"");
Console.WriteLine(""  - Compression"");
"
                }
            },
            ["ONNX"] = new()
            {
                new CodeExample
                {
                    Id = "onnx-export",
                    Name = "ONNX Export",
                    Description = "Export models to ONNX format",
                    Difficulty = "Intermediate",
                    Tags = ["onnx", "export", "interop"],
                    Code = @"// ONNX Export
using AiDotNet.ONNX;

Console.WriteLine(""ONNX Model Export"");
Console.WriteLine(""-----------------"");
Console.WriteLine();
Console.WriteLine(""Original Model: AiDotNet Neural Network"");
Console.WriteLine(""  Framework: AiDotNet"");
Console.WriteLine(""  Layers: 5"");
Console.WriteLine(""  Parameters: 1.2M"");
Console.WriteLine();
Console.WriteLine(""Exporting to ONNX..."");
Console.WriteLine(""  Input: (batch, 224, 224, 3)"");
Console.WriteLine(""  Output: (batch, 1000)"");
Console.WriteLine();
Console.WriteLine(""ONNX file: model.onnx (4.8MB)"");
Console.WriteLine();
Console.WriteLine(""Now usable in:"");
Console.WriteLine(""  - ONNX Runtime"");
Console.WriteLine(""  - TensorRT"");
Console.WriteLine(""  - OpenVINO"");
Console.WriteLine(""  - CoreML"");
"
                }
            },
            ["Serialization"] = new()
            {
                new CodeExample
                {
                    Id = "model-save-load",
                    Name = "Model Save/Load",
                    Description = "Persist and restore models",
                    Difficulty = "Beginner",
                    Tags = ["serialization", "save", "load"],
                    Code = @"// Model Serialization
using AiDotNet.Serialization;

Console.WriteLine(""Model Serialization"");
Console.WriteLine(""-------------------"");
Console.WriteLine();
Console.WriteLine(""Training complete!"");
Console.WriteLine(""  Accuracy: 94.5%"");
Console.WriteLine(""  Epochs: 100"");
Console.WriteLine();
Console.WriteLine(""Saving model..."");
Console.WriteLine(""  Path: models/classifier.bin"");
Console.WriteLine(""  Size: 12.3 MB"");
Console.WriteLine(""  Includes: weights, config, metadata"");
Console.WriteLine();
Console.WriteLine(""Loading model..."");
Console.WriteLine(""  Model loaded successfully"");
Console.WriteLine(""  Ready for inference"");
Console.WriteLine();
Console.WriteLine(""Checkpointing during training:"");
Console.WriteLine(""  Best model saved at epoch 87"");
"
                }
            },
            ["Experiment Tracking"] = new()
            {
                new CodeExample
                {
                    Id = "tracking-basics",
                    Name = "Experiment Tracking",
                    Description = "Track and compare experiments",
                    Difficulty = "Intermediate",
                    Tags = ["tracking", "mlops", "experiments"],
                    Code = @"// Experiment Tracking
using AiDotNet.ExperimentTracking;

Console.WriteLine(""Experiment Tracking"");
Console.WriteLine(""-------------------"");
Console.WriteLine();
Console.WriteLine(""Experiment: image-classifier-v3"");
Console.WriteLine();
Console.WriteLine(""Parameters logged:"");
Console.WriteLine(""  learning_rate: 0.001"");
Console.WriteLine(""  batch_size: 32"");
Console.WriteLine(""  optimizer: Adam"");
Console.WriteLine();
Console.WriteLine(""Metrics logged:"");
Console.WriteLine(""  train_loss: 0.23"");
Console.WriteLine(""  val_accuracy: 0.91"");
Console.WriteLine(""  epoch: 50"");
Console.WriteLine();
Console.WriteLine(""Artifacts:"");
Console.WriteLine(""  - model.bin"");
Console.WriteLine(""  - confusion_matrix.png"");
Console.WriteLine();
Console.WriteLine(""Compare with previous runs easily"");
"
                }
            },
            ["Hyperparameter Optimization"] = new()
            {
                new CodeExample
                {
                    Id = "grid-search",
                    Name = "Grid Search",
                    Description = "Systematic hyperparameter search",
                    Difficulty = "Intermediate",
                    Tags = ["hyperparameters", "tuning", "grid"],
                    Code = @"// Grid Search Hyperparameter Optimization
using AiDotNet.HyperparameterOptimization;

Console.WriteLine(""Grid Search"");
Console.WriteLine(""-----------"");
Console.WriteLine();
Console.WriteLine(""Search Space:"");
Console.WriteLine(""  learning_rate: [0.001, 0.01, 0.1]"");
Console.WriteLine(""  batch_size: [16, 32, 64]"");
Console.WriteLine(""  hidden_units: [64, 128]"");
Console.WriteLine();
Console.WriteLine(""Total combinations: 3 x 3 x 2 = 18"");
Console.WriteLine();
Console.WriteLine(""Results (top 3):"");
Console.WriteLine(""  1. lr=0.01, bs=32, hu=128 -> 94.2%"");
Console.WriteLine(""  2. lr=0.01, bs=64, hu=128 -> 93.8%"");
Console.WriteLine(""  3. lr=0.001, bs=32, hu=128 -> 93.5%"");
Console.WriteLine();
Console.WriteLine(""Best: lr=0.01, batch=32, hidden=128"");
"
                },
                new CodeExample
                {
                    Id = "bayesian-opt",
                    Name = "Bayesian Optimization",
                    Description = "Efficient hyperparameter search",
                    Difficulty = "Advanced",
                    Tags = ["hyperparameters", "bayesian", "tuning"],
                    Code = @"// Bayesian Optimization
using AiDotNet.HyperparameterOptimization;

Console.WriteLine(""Bayesian Optimization"");
Console.WriteLine(""---------------------"");
Console.WriteLine();
Console.WriteLine(""Continuous search space:"");
Console.WriteLine(""  learning_rate: [1e-5, 0.1]"");
Console.WriteLine(""  dropout: [0.0, 0.5]"");
Console.WriteLine(""  weight_decay: [1e-6, 1e-2]"");
Console.WriteLine();
Console.WriteLine(""Iterations: 20 (vs 1000 for grid)"");
Console.WriteLine();
Console.WriteLine(""Progress:"");
Console.WriteLine(""  Iter 1:  lr=0.05, d=0.2 -> 85%"");
Console.WriteLine(""  Iter 5:  lr=0.01, d=0.3 -> 89%"");
Console.WriteLine(""  Iter 10: lr=0.008, d=0.25 -> 92%"");
Console.WriteLine(""  Iter 20: lr=0.0073, d=0.28 -> 94%"");
Console.WriteLine();
Console.WriteLine(""50x faster than grid search"");
"
                }
            },
            ["Model Serving"] = new()
            {
                new CodeExample
                {
                    Id = "rest-api",
                    Name = "REST API Deployment",
                    Description = "Deploy model as REST endpoint",
                    Difficulty = "Intermediate",
                    Tags = ["serving", "deployment", "api"],
                    Code = @"// Model Serving - REST API
using AiDotNet.Serving;

Console.WriteLine(""Model Serving - REST API"");
Console.WriteLine(""------------------------"");
Console.WriteLine();
Console.WriteLine(""Endpoint: POST /api/predict"");
Console.WriteLine();
Console.WriteLine(""Request:"");
Console.WriteLine(""  {"");
Console.WriteLine(""    'features': [1.2, 3.4, 5.6, 7.8]"");
Console.WriteLine(""  }"");
Console.WriteLine();
Console.WriteLine(""Response:"");
Console.WriteLine(""  {"");
Console.WriteLine(""    'prediction': 'class_a',"");
Console.WriteLine(""    'confidence': 0.92,"");
Console.WriteLine(""    'latency_ms': 12"");
Console.WriteLine(""  }"");
Console.WriteLine();
Console.WriteLine(""Stats:"");
Console.WriteLine(""  Requests/sec: 1000"");
Console.WriteLine(""  P99 latency: 45ms"");
"
                }
            },
            ["Metrics"] = new()
            {
                new CodeExample
                {
                    Id = "classification-metrics",
                    Name = "Classification Metrics",
                    Description = "Evaluate classification models",
                    Difficulty = "Beginner",
                    Tags = ["metrics", "classification", "evaluation"],
                    Code = @"// Classification Metrics
using AiDotNet.Metrics;

Console.WriteLine(""Classification Metrics"");
Console.WriteLine(""----------------------"");
Console.WriteLine();
Console.WriteLine(""Confusion Matrix:"");
Console.WriteLine(""              Predicted"");
Console.WriteLine(""           Pos    Neg"");
Console.WriteLine(""  Actual"");
Console.WriteLine(""    Pos    85     15   (100)"");
Console.WriteLine(""    Neg    10     90   (100)"");
Console.WriteLine();
Console.WriteLine(""Metrics:"");
Console.WriteLine(""  Accuracy:  0.875"");
Console.WriteLine(""  Precision: 0.895 (85/95)"");
Console.WriteLine(""  Recall:    0.850 (85/100)"");
Console.WriteLine(""  F1 Score:  0.872"");
Console.WriteLine(""  AUC-ROC:   0.923"");
Console.WriteLine();
Console.WriteLine(""Choose metrics based on problem"");
"
                },
                new CodeExample
                {
                    Id = "regression-metrics",
                    Name = "Regression Metrics",
                    Description = "Evaluate regression models",
                    Difficulty = "Beginner",
                    Tags = ["metrics", "regression", "evaluation"],
                    Code = @"// Regression Metrics
using AiDotNet.Metrics;

Console.WriteLine(""Regression Metrics"");
Console.WriteLine(""------------------"");
Console.WriteLine();
Console.WriteLine(""Predictions vs Actuals:"");
Console.WriteLine(""  Pred: [2.1, 3.2, 4.8, 5.1]"");
Console.WriteLine(""  True: [2.0, 3.0, 5.0, 5.0]"");
Console.WriteLine();
Console.WriteLine(""Metrics:"");
Console.WriteLine(""  MAE:  0.175 (Mean Absolute Error)"");
Console.WriteLine(""  MSE:  0.045 (Mean Squared Error)"");
Console.WriteLine(""  RMSE: 0.212 (Root MSE)"");
Console.WriteLine(""  R2:   0.956 (explained variance)"");
Console.WriteLine(""  MAPE: 4.2%  (Mean Absolute Pct Error)"");
Console.WriteLine();
Console.WriteLine(""MAE: robust to outliers"");
Console.WriteLine(""MSE: penalizes large errors"");
"
                }
            },
            ["Mixed Precision"] = new()
            {
                new CodeExample
                {
                    Id = "fp16-training",
                    Name = "FP16 Training",
                    Description = "Train with half precision",
                    Difficulty = "Advanced",
                    Tags = ["mixed-precision", "fp16", "performance"],
                    Code = @"// Mixed Precision Training
using AiDotNet.MixedPrecision;

Console.WriteLine(""Mixed Precision Training"");
Console.WriteLine(""------------------------"");
Console.WriteLine();
Console.WriteLine(""FP32 Baseline:"");
Console.WriteLine(""  Memory: 8 GB"");
Console.WriteLine(""  Speed: 100 samples/sec"");
Console.WriteLine();
Console.WriteLine(""FP16 Mixed Precision:"");
Console.WriteLine(""  Memory: 4 GB"");
Console.WriteLine(""  Speed: 200 samples/sec"");
Console.WriteLine();
Console.WriteLine(""Loss Scaling:"");
Console.WriteLine(""  Prevents underflow in FP16"");
Console.WriteLine(""  Dynamic scaling: 65536 -> 32768"");
Console.WriteLine();
Console.WriteLine(""Accuracy: Same as FP32"");
Console.WriteLine(""Speedup: 2x faster"");
Console.WriteLine(""Memory: 50% reduction"");
"
                }
            },
            ["Autodiff"] = new()
            {
                new CodeExample
                {
                    Id = "autodiff-basics",
                    Name = "Automatic Differentiation",
                    Description = "Compute gradients automatically",
                    Difficulty = "Advanced",
                    Tags = ["autodiff", "gradients", "backprop"],
                    Code = @"// Automatic Differentiation
using AiDotNet.Autodiff;

Console.WriteLine(""Automatic Differentiation"");
Console.WriteLine(""-------------------------"");
Console.WriteLine();
Console.WriteLine(""Function: f(x,y) = x^2 * y + y^3"");
Console.WriteLine(""Point: x=2, y=3"");
Console.WriteLine();
Console.WriteLine(""Forward pass:"");
Console.WriteLine(""  f(2,3) = 4*3 + 27 = 39"");
Console.WriteLine();
Console.WriteLine(""Backward pass (gradients):"");
Console.WriteLine(""  df/dx = 2*x*y = 2*2*3 = 12"");
Console.WriteLine(""  df/dy = x^2 + 3*y^2 = 4 + 27 = 31"");
Console.WriteLine();
Console.WriteLine(""Computed automatically!"");
Console.WriteLine(""No manual derivative calculation"");
Console.WriteLine();
Console.WriteLine(""Used for training neural networks"");
"
                }
            },
            ["Distributed Training"] = new()
            {
                new CodeExample
                {
                    Id = "data-parallel",
                    Name = "Data Parallel Training",
                    Description = "Train across multiple GPUs",
                    Difficulty = "Expert",
                    Tags = ["distributed", "parallel", "multi-gpu"],
                    Code = @"// Data Parallel Training
using AiDotNet.DistributedTraining;

Console.WriteLine(""Data Parallel Training"");
Console.WriteLine(""----------------------"");
Console.WriteLine();
Console.WriteLine(""Setup:"");
Console.WriteLine(""  GPUs: 4"");
Console.WriteLine(""  Global batch: 256"");
Console.WriteLine(""  Per-GPU batch: 64"");
Console.WriteLine();
Console.WriteLine(""Each step:"");
Console.WriteLine(""  1. Split batch across GPUs"");
Console.WriteLine(""  2. Forward pass (parallel)"");
Console.WriteLine(""  3. Compute gradients (parallel)"");
Console.WriteLine(""  4. AllReduce gradients"");
Console.WriteLine(""  5. Update weights"");
Console.WriteLine();
Console.WriteLine(""Speedup: ~3.8x with 4 GPUs"");
Console.WriteLine(""  (Communication overhead ~5%)"");
"
                }
            },
            ["Ensemble Methods"] = new()
            {
                new CodeExample
                {
                    Id = "voting-ensemble",
                    Name = "Voting Ensemble",
                    Description = "Combine multiple models",
                    Difficulty = "Intermediate",
                    Tags = ["ensemble", "voting", "combination"],
                    Code = @"// Voting Ensemble
using AiDotNet.Ensemble;

Console.WriteLine(""Voting Ensemble"");
Console.WriteLine(""---------------"");
Console.WriteLine();
Console.WriteLine(""Individual Models:"");
Console.WriteLine(""  Model A (Random Forest): 87%"");
Console.WriteLine(""  Model B (SVM): 85%"");
Console.WriteLine(""  Model C (Neural Net): 88%"");
Console.WriteLine();
Console.WriteLine(""Hard Voting (majority):"");
Console.WriteLine(""  Input X predictions:"");
Console.WriteLine(""    A: Class 1"");
Console.WriteLine(""    B: Class 1"");
Console.WriteLine(""    C: Class 2"");
Console.WriteLine(""  Ensemble: Class 1 (2 votes)"");
Console.WriteLine();
Console.WriteLine(""Ensemble Accuracy: 91%"");
Console.WriteLine(""  +3% over best individual model"");
"
                },
                new CodeExample
                {
                    Id = "stacking",
                    Name = "Stacking Ensemble",
                    Description = "Meta-learner combines predictions",
                    Difficulty = "Advanced",
                    Tags = ["ensemble", "stacking", "meta-learning"],
                    Code = @"// Stacking Ensemble
using AiDotNet.Ensemble;

Console.WriteLine(""Stacking Ensemble"");
Console.WriteLine(""-----------------"");
Console.WriteLine();
Console.WriteLine(""Level 0 (Base models):"");
Console.WriteLine(""  Model A -> pred_A"");
Console.WriteLine(""  Model B -> pred_B"");
Console.WriteLine(""  Model C -> pred_C"");
Console.WriteLine();
Console.WriteLine(""Level 1 (Meta-learner):"");
Console.WriteLine(""  Input: [pred_A, pred_B, pred_C]"");
Console.WriteLine(""  Output: final prediction"");
Console.WriteLine();
Console.WriteLine(""Training:"");
Console.WriteLine(""  Use cross-validation to avoid"");
Console.WriteLine(""  leaking labels to meta-learner"");
Console.WriteLine();
Console.WriteLine(""Individual: 87%, 85%, 88%"");
Console.WriteLine(""Stacking: 93%"");
"
                }
            },
            ["Geometry"] = new()
            {
                new CodeExample
                {
                    Id = "geometric-transforms",
                    Name = "Geometric Transformations",
                    Description = "2D and 3D geometric operations",
                    Difficulty = "Intermediate",
                    Tags = ["geometry", "transforms", "3d"],
                    Code = @"// Geometric Transformations
using AiDotNet.Geometry;

Console.WriteLine(""Geometric Transformations"");
Console.WriteLine(""-------------------------"");
Console.WriteLine();
Console.WriteLine(""Original point: (1, 0, 0)"");
Console.WriteLine();
Console.WriteLine(""Translation (+2, +3, +1):"");
Console.WriteLine(""  Result: (3, 3, 1)"");
Console.WriteLine();
Console.WriteLine(""Rotation 90 deg around Z:"");
Console.WriteLine(""  Result: (0, 1, 0)"");
Console.WriteLine();
Console.WriteLine(""Scale (2, 2, 2):"");
Console.WriteLine(""  Result: (2, 0, 0)"");
Console.WriteLine();
Console.WriteLine(""Composite transform:"");
Console.WriteLine(""  T x R x S applied together"");
Console.WriteLine(""  Result: (6, 6, 2)"");
"
                }
            },
            ["Point Clouds"] = new()
            {
                new CodeExample
                {
                    Id = "point-cloud-processing",
                    Name = "Point Cloud Processing",
                    Description = "3D point cloud operations",
                    Difficulty = "Advanced",
                    Tags = ["pointcloud", "3d", "lidar"],
                    Code = @"// Point Cloud Processing
using AiDotNet.PointCloud;

Console.WriteLine(""Point Cloud Processing"");
Console.WriteLine(""----------------------"");
Console.WriteLine();
Console.WriteLine(""Input: 10,000 3D points"");
Console.WriteLine(""  From LiDAR scan"");
Console.WriteLine();
Console.WriteLine(""Operations:"");
Console.WriteLine(""  1. Downsampling (voxel grid)"");
Console.WriteLine(""     10,000 -> 2,500 points"");
Console.WriteLine();
Console.WriteLine(""  2. Normal estimation"");
Console.WriteLine(""     k=20 nearest neighbors"");
Console.WriteLine();
Console.WriteLine(""  3. Plane segmentation"");
Console.WriteLine(""     Found: 3 planes"");
Console.WriteLine(""     Inliers: 1,800 points"");
Console.WriteLine();
Console.WriteLine(""  4. Clustering (DBSCAN)"");
Console.WriteLine(""     Found: 5 objects"");
"
                }
            },
            ["Video Processing"] = new()
            {
                new CodeExample
                {
                    Id = "video-classification",
                    Name = "Video Classification",
                    Description = "Classify video content",
                    Difficulty = "Advanced",
                    Tags = ["video", "temporal", "classification"],
                    Code = @"// Video Classification
using AiDotNet.Video;

Console.WriteLine(""Video Classification"");
Console.WriteLine(""--------------------"");
Console.WriteLine();
Console.WriteLine(""Input: 30fps video, 10 seconds"");
Console.WriteLine(""  Total frames: 300"");
Console.WriteLine();
Console.WriteLine(""Processing:"");
Console.WriteLine(""  1. Sample every 10th frame"");
Console.WriteLine(""     Selected: 30 frames"");
Console.WriteLine();
Console.WriteLine(""  2. Extract features per frame"");
Console.WriteLine(""     ResNet50 backbone"");
Console.WriteLine();
Console.WriteLine(""  3. Temporal modeling (LSTM)"");
Console.WriteLine(""     Sequence of 30 features"");
Console.WriteLine();
Console.WriteLine(""Prediction: 'Playing Basketball'"");
Console.WriteLine(""Confidence: 0.94"");
"
                }
            },
            ["Neural Radiance Fields"] = new()
            {
                new CodeExample
                {
                    Id = "nerf-basics",
                    Name = "NeRF Basics",
                    Description = "3D scene reconstruction",
                    Difficulty = "Expert",
                    Tags = ["nerf", "3d", "reconstruction"],
                    Code = @"// Neural Radiance Fields (NeRF)
using AiDotNet.NeuralRadianceFields;

Console.WriteLine(""Neural Radiance Fields"");
Console.WriteLine(""----------------------"");
Console.WriteLine();
Console.WriteLine(""Input: 50 images of scene"");
Console.WriteLine(""  With camera poses"");
Console.WriteLine();
Console.WriteLine(""Training NeRF:"");
Console.WriteLine(""  MLP: 8 layers, 256 units"");
Console.WriteLine(""  Input: (x, y, z, theta, phi)"");
Console.WriteLine(""  Output: (R, G, B, density)"");
Console.WriteLine();
Console.WriteLine(""Rendering new view:"");
Console.WriteLine(""  Cast rays through each pixel"");
Console.WriteLine(""  Sample 64 points per ray"");
Console.WriteLine(""  Query MLP for color/density"");
Console.WriteLine(""  Volume rendering integration"");
Console.WriteLine();
Console.WriteLine(""Result: Novel view synthesis"");
"
                }
            },
            ["Radial Basis Functions"] = new()
            {
                new CodeExample
                {
                    Id = "rbf-network",
                    Name = "RBF Networks",
                    Description = "Radial basis function neural network",
                    Difficulty = "Advanced",
                    Tags = ["rbf", "network", "interpolation"],
                    Code = @"// RBF Networks
using AiDotNet.RadialBasisFunctions;

Console.WriteLine(""RBF Neural Network"");
Console.WriteLine(""------------------"");
Console.WriteLine();
Console.WriteLine(""Architecture:"");
Console.WriteLine(""  Input: 2 features"");
Console.WriteLine(""  Hidden: 10 RBF units"");
Console.WriteLine(""  Output: 1 (regression)"");
Console.WriteLine();
Console.WriteLine(""RBF activation:"");
Console.WriteLine(""  phi(r) = exp(-r^2 / 2*sigma^2)"");
Console.WriteLine();
Console.WriteLine(""Training:"");
Console.WriteLine(""  1. K-means to find centers"");
Console.WriteLine(""  2. Set sigma from distances"");
Console.WriteLine(""  3. Linear regression for weights"");
Console.WriteLine();
Console.WriteLine(""RBF advantages:"");
Console.WriteLine(""  - Fast training"");
Console.WriteLine(""  - Good interpolation"");
Console.WriteLine(""  - Local receptive fields"");
"
                }
            },
            ["Prompt Engineering"] = new()
            {
                new CodeExample
                {
                    Id = "prompt-templates",
                    Name = "Prompt Templates",
                    Description = "Structured prompt design",
                    Difficulty = "Intermediate",
                    Tags = ["prompts", "llm", "templates"],
                    Code = @"// Prompt Engineering
using AiDotNet.PromptEngineering;

Console.WriteLine(""Prompt Engineering"");
Console.WriteLine(""------------------"");
Console.WriteLine();
Console.WriteLine(""Template: Classification"");
Console.WriteLine();
Console.WriteLine(""  System: You are a sentiment classifier."");
Console.WriteLine(""  User: Classify: '{text}'"");
Console.WriteLine(""  Output: positive/negative/neutral"");
Console.WriteLine();
Console.WriteLine(""Variables:"");
Console.WriteLine(""  text = 'Great product, love it!'"");
Console.WriteLine();
Console.WriteLine(""Rendered prompt:"");
Console.WriteLine(""  Classify: 'Great product, love it!'"");
Console.WriteLine();
Console.WriteLine(""Response: positive"");
Console.WriteLine();
Console.WriteLine(""Templates ensure consistent formatting"");
"
                }
            },
            ["Graph Neural Networks"] = new()
            {
                new CodeExample
                {
                    Id = "gcn",
                    Name = "Graph Convolutional Networks",
                    Description = "Neural networks for graphs",
                    Difficulty = "Advanced",
                    Tags = ["gnn", "graph", "gcn"],
                    Code = @"// Graph Convolutional Networks
using AiDotNet.GraphNeuralNetworks;

Console.WriteLine(""Graph Convolutional Network"");
Console.WriteLine(""---------------------------"");
Console.WriteLine();
Console.WriteLine(""Graph: Social network"");
Console.WriteLine(""  Nodes: 100 users"");
Console.WriteLine(""  Edges: 500 friendships"");
Console.WriteLine(""  Node features: 16 dims"");
Console.WriteLine();
Console.WriteLine(""GCN Layer:"");
Console.WriteLine(""  H' = ReLU(D^-0.5 * A * D^-0.5 * H * W)"");
Console.WriteLine();
Console.WriteLine(""Architecture:"");
Console.WriteLine(""  GCN(16 -> 32) -> ReLU"");
Console.WriteLine(""  GCN(32 -> 16) -> ReLU"");
Console.WriteLine(""  GCN(16 -> 3)  -> Softmax"");
Console.WriteLine();
Console.WriteLine(""Task: Node classification"");
Console.WriteLine(""  Predict user category"");
Console.WriteLine(""  Accuracy: 89%"");
"
                },
                new CodeExample
                {
                    Id = "gat",
                    Name = "Graph Attention Networks",
                    Description = "Attention mechanism for graphs",
                    Difficulty = "Expert",
                    Tags = ["gnn", "attention", "gat"],
                    Code = @"// Graph Attention Networks
using AiDotNet.GraphNeuralNetworks;

Console.WriteLine(""Graph Attention Network"");
Console.WriteLine(""-----------------------"");
Console.WriteLine();
Console.WriteLine(""Key difference from GCN:"");
Console.WriteLine(""  Learn edge importance"");
Console.WriteLine();
Console.WriteLine(""Attention coefficient:"");
Console.WriteLine(""  a_ij = softmax(LeakyReLU(a[Wh_i || Wh_j]))"");
Console.WriteLine();
Console.WriteLine(""Multi-head attention:"");
Console.WriteLine(""  8 attention heads"");
Console.WriteLine(""  Concatenated/averaged"");
Console.WriteLine();
Console.WriteLine(""Example attention weights:"");
Console.WriteLine(""  Node 1 -> Node 2: 0.35"");
Console.WriteLine(""  Node 1 -> Node 3: 0.45"");
Console.WriteLine(""  Node 1 -> Node 5: 0.20"");
Console.WriteLine();
Console.WriteLine(""GAT learns which neighbors matter"");
"
                }
            },
            ["Attention Mechanisms"] = new()
            {
                new CodeExample
                {
                    Id = "self-attention",
                    Name = "Self-Attention",
                    Description = "Scaled dot-product attention",
                    Difficulty = "Advanced",
                    Tags = ["attention", "transformer", "self-attention"],
                    Code = @"// Self-Attention Mechanism
using AiDotNet.Attention;

Console.WriteLine(""Self-Attention"");
Console.WriteLine(""--------------"");
Console.WriteLine();
Console.WriteLine(""Input: Sequence of 5 tokens"");
Console.WriteLine(""  Embedding dim: 64"");
Console.WriteLine();
Console.WriteLine(""Projections:"");
Console.WriteLine(""  Q = X * Wq (queries)"");
Console.WriteLine(""  K = X * Wk (keys)"");
Console.WriteLine(""  V = X * Wv (values)"");
Console.WriteLine();
Console.WriteLine(""Attention scores:"");
Console.WriteLine(""  A = softmax(Q * K^T / sqrt(d_k))"");
Console.WriteLine();
Console.WriteLine(""Attention weights (token 1):"");
Console.WriteLine(""  [0.05, 0.60, 0.15, 0.10, 0.10]"");
Console.WriteLine();
Console.WriteLine(""Token 1 attends most to token 2"");
"
                },
                new CodeExample
                {
                    Id = "multi-head-attention",
                    Name = "Multi-Head Attention",
                    Description = "Multiple attention heads in parallel",
                    Difficulty = "Advanced",
                    Tags = ["attention", "multi-head", "transformer"],
                    Code = @"// Multi-Head Attention
using AiDotNet.Attention;

Console.WriteLine(""Multi-Head Attention"");
Console.WriteLine(""--------------------"");
Console.WriteLine();
Console.WriteLine(""Configuration:"");
Console.WriteLine(""  d_model: 512"");
Console.WriteLine(""  num_heads: 8"");
Console.WriteLine(""  d_k = d_v: 64"");
Console.WriteLine();
Console.WriteLine(""Each head learns different:"");
Console.WriteLine(""  Head 1: Subject-verb relations"");
Console.WriteLine(""  Head 2: Adjective-noun relations"");
Console.WriteLine(""  Head 3: Long-range dependencies"");
Console.WriteLine(""  ..."");
Console.WriteLine();
Console.WriteLine(""Output:"");
Console.WriteLine(""  Concat(head_1, ..., head_8) * W_o"");
Console.WriteLine();
Console.WriteLine(""Multi-head captures diverse patterns"");
"
                }
            },
            ["Recommendation Systems"] = new()
            {
                new CodeExample
                {
                    Id = "collaborative-filtering",
                    Name = "Collaborative Filtering",
                    Description = "User-item recommendations",
                    Difficulty = "Intermediate",
                    Tags = ["recommender", "collaborative", "matrix"],
                    Code = @"// Collaborative Filtering
using AiDotNet.RecommendationSystems;

Console.WriteLine(""Collaborative Filtering"");
Console.WriteLine(""-----------------------"");
Console.WriteLine();
Console.WriteLine(""User-Item Matrix (ratings):"");
Console.WriteLine(""        Item1 Item2 Item3 Item4"");
Console.WriteLine(""  User1   5     3     -     1"");
Console.WriteLine(""  User2   4     -     -     1"");
Console.WriteLine(""  User3   1     1     -     5"");
Console.WriteLine(""  User4   -     -     5     4"");
Console.WriteLine();
Console.WriteLine(""Matrix Factorization:"");
Console.WriteLine(""  R = U * V^T"");
Console.WriteLine(""  U: 4x2 (user factors)"");
Console.WriteLine(""  V: 4x2 (item factors)"");
Console.WriteLine();
Console.WriteLine(""Prediction for User2, Item3:"");
Console.WriteLine(""  u2 dot v3 = 4.2 (predicted rating)"");
"
                },
                new CodeExample
                {
                    Id = "content-based",
                    Name = "Content-Based Filtering",
                    Description = "Feature-based recommendations",
                    Difficulty = "Intermediate",
                    Tags = ["recommender", "content", "features"],
                    Code = @"// Content-Based Filtering
using AiDotNet.RecommendationSystems;

Console.WriteLine(""Content-Based Filtering"");
Console.WriteLine(""-----------------------"");
Console.WriteLine();
Console.WriteLine(""Movie features:"");
Console.WriteLine(""  Movie A: [action=0.9, comedy=0.1]"");
Console.WriteLine(""  Movie B: [action=0.8, comedy=0.3]"");
Console.WriteLine(""  Movie C: [action=0.1, comedy=0.9]"");
Console.WriteLine();
Console.WriteLine(""User profile (from history):"");
Console.WriteLine(""  Likes action: 0.85"");
Console.WriteLine(""  Likes comedy: 0.25"");
Console.WriteLine();
Console.WriteLine(""Similarity scores:"");
Console.WriteLine(""  Movie A: 0.92 (recommend)"");
Console.WriteLine(""  Movie B: 0.85 (recommend)"");
Console.WriteLine(""  Movie C: 0.32 (skip)"");
Console.WriteLine();
Console.WriteLine(""Content-based uses item features"");
"
                }
            },
            ["Causal Inference"] = new()
            {
                new CodeExample
                {
                    Id = "causal-ml",
                    Name = "Causal Machine Learning",
                    Description = "Estimate causal effects",
                    Difficulty = "Expert",
                    Tags = ["causal", "inference", "treatment"],
                    Code = @"// Causal Machine Learning
using AiDotNet.CausalInference;

Console.WriteLine(""Causal Inference"");
Console.WriteLine(""----------------"");
Console.WriteLine();
Console.WriteLine(""Question: Does treatment X cause outcome Y?"");
Console.WriteLine();
Console.WriteLine(""Observational data:"");
Console.WriteLine(""  Treated group: outcome = 0.75"");
Console.WriteLine(""  Control group: outcome = 0.50"");
Console.WriteLine();
Console.WriteLine(""Naive difference: 0.25"");
Console.WriteLine(""  But confounders exist!"");
Console.WriteLine();
Console.WriteLine(""Causal methods:"");
Console.WriteLine(""  Propensity score matching"");
Console.WriteLine(""  Doubly robust estimation"");
Console.WriteLine();
Console.WriteLine(""True causal effect: 0.18"");
Console.WriteLine(""  (confounding explained 0.07)"");
"
                }
            },
            ["Probabilistic Models"] = new()
            {
                new CodeExample
                {
                    Id = "bayesian-inference",
                    Name = "Bayesian Inference",
                    Description = "Posterior estimation",
                    Difficulty = "Advanced",
                    Tags = ["bayesian", "probabilistic", "inference"],
                    Code = @"// Bayesian Inference
using AiDotNet.ProbabilisticModels;

Console.WriteLine(""Bayesian Inference"");
Console.WriteLine(""------------------"");
Console.WriteLine();
Console.WriteLine(""Problem: Estimate coin fairness"");
Console.WriteLine();
Console.WriteLine(""Prior: Beta(2, 2)"");
Console.WriteLine(""  Mean: 0.50"");
Console.WriteLine(""  Uncertainty: high"");
Console.WriteLine();
Console.WriteLine(""Data: 7 heads, 3 tails"");
Console.WriteLine();
Console.WriteLine(""Posterior: Beta(9, 5)"");
Console.WriteLine(""  Mean: 0.64"");
Console.WriteLine(""  95% CI: [0.42, 0.83]"");
Console.WriteLine();
Console.WriteLine(""Bayesian updates beliefs with data"");
Console.WriteLine(""  Prior + Likelihood = Posterior"");
"
                }
            },
            ["Symbolic AI"] = new()
            {
                new CodeExample
                {
                    Id = "neuro-symbolic",
                    Name = "Neuro-Symbolic Integration",
                    Description = "Combine neural and symbolic AI",
                    Difficulty = "Expert",
                    Tags = ["symbolic", "neuro-symbolic", "reasoning"],
                    Code = @"// Neuro-Symbolic AI
using AiDotNet.SymbolicAI;

Console.WriteLine(""Neuro-Symbolic AI"");
Console.WriteLine(""-----------------"");
Console.WriteLine();
Console.WriteLine(""Hybrid approach:"");
Console.WriteLine(""  Neural: Pattern recognition"");
Console.WriteLine(""  Symbolic: Logical reasoning"");
Console.WriteLine();
Console.WriteLine(""Example: Visual QA"");
Console.WriteLine(""  Image: Cat on table"");
Console.WriteLine(""  Question: Is the cat above the table?"");
Console.WriteLine();
Console.WriteLine(""Neural component:"");
Console.WriteLine(""  Detect: cat at (x1,y1), table at (x2,y2)"");
Console.WriteLine();
Console.WriteLine(""Symbolic component:"");
Console.WriteLine(""  Rule: above(A,B) if y(A) < y(B)"");
Console.WriteLine(""  Query: above(cat, table)?"");
Console.WriteLine(""  Answer: Yes"");
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
