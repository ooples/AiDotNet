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
using AiDotNet.Models.Options;
using AiDotNet.ReinforcementLearning.Agents.PPO;

// Continuous action space environment
var stateSize = 8;
var actionSize = 2;

// Configure PPO options
var options = new PPOOptions<double>
{
    StateSize = stateSize,
    ActionSize = actionSize,
    IsContinuous = true,
    PolicyHiddenLayers = new List<int> { 64, 64 },
    ValueHiddenLayers = new List<int> { 64, 64 },
    PolicyLearningRate = 0.0003,
    ValueLearningRate = 0.001,
    DiscountFactor = 0.99,
    GaeLambda = 0.95,
    ClipEpsilon = 0.2,
    EntropyCoefficient = 0.01,
    TrainingEpochs = 10,
    MiniBatchSize = 64
};

var result = await new AiModelBuilder<double, double[], double[]>()
    .ConfigureModel(new PPOAgent<double>(options))
    .BuildAsync();

Console.WriteLine(""PPO Agent Created:"");
Console.WriteLine($""  State size: {stateSize}"");
Console.WriteLine($""  Action size: {actionSize} (continuous)"");
Console.WriteLine($""  Clip epsilon: {options.ClipEpsilon}"");
Console.WriteLine($""  GAE lambda: {options.GaeLambda}"");

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
