using AiDotNet;
using AiDotNet.Classification.Ensemble;
using AiDotNet.Data.Loaders;
using AiDotNet.Models.Options;
using AiDotNet.Preprocessing.Scalers;
using AiDotNet.Tensors.LinearAlgebra;

Console.WriteLine("=== AiDotNet Basic Classification ===");
Console.WriteLine("Classifying Iris flowers using Random Forest\n");

// Iris dataset (simplified - in production, load from file or use a data loader)
// Features: sepal_length, sepal_width, petal_length, petal_width
// Labels: 0=Setosa, 1=Versicolor, 2=Virginica
var (features, labels) = LoadIrisDataset();

Console.WriteLine($"Loaded {features.Length} samples with {features[0].Length} features\n");

// Split into train/test sets (80/20)
var splitIndex = (int)(features.Length * 0.8);
var trainFeatures = features.Take(splitIndex).ToArray();
var trainLabels = labels.Take(splitIndex).ToArray();
var testFeatures = features.Skip(splitIndex).ToArray();
var testLabels = labels.Skip(splitIndex).ToArray();

Console.WriteLine($"Training set: {trainFeatures.Length} samples");
Console.WriteLine($"Test set: {testFeatures.Length} samples\n");

// Build and train the classifier using the facade pattern
Console.WriteLine("Building Random Forest classifier with AiModelBuilder...");
Console.WriteLine("  - 100 decision trees");
Console.WriteLine("  - StandardScaler preprocessing\n");

try
{
    // Use the AiModelBuilder facade to configure and train
    var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
        .ConfigureModel(new RandomForestClassifier<double>(
            new RandomForestClassifierOptions<double>
            {
                NEstimators = 100,
                MinSamplesSplit = 2
            }))
        .ConfigurePreprocessing(pipeline => pipeline
            .Add(new StandardScaler<double>()))
        .ConfigureDataLoader(DataLoaders.FromArrays(trainFeatures, trainLabels));

    Console.WriteLine("Training...\n");

    var result = await builder.BuildAsync();

    // Evaluate on test set using the result object directly (facade pattern)
    Console.WriteLine("\nFinal Model Evaluation:");
    Console.WriteLine("─────────────────────────────────────");

    // Batch-predict the whole test matrix, then read per-row class scores.
    var predVector = result.Predict(ToMatrix(testFeatures));
    int correct = 0;
    for (int i = 0; i < testFeatures.Length; i++)
    {
        if (Math.Abs(predVector[i] - testLabels[i]) < 0.5)
            correct++;
    }

    double testAccuracy = (double)correct / testFeatures.Length;
    Console.WriteLine($"  Test Accuracy: {testAccuracy:P2}");

    // Show some predictions using the result object
    Console.WriteLine("\nSample Predictions:");
    Console.WriteLine("─────────────────────────────────────");
    string[] speciesNames = { "Setosa", "Versicolor", "Virginica" };

    for (int i = 0; i < Math.Min(5, testFeatures.Length); i++)
    {
        int predictedClass = Math.Clamp((int)Math.Round(predVector[i]), 0, 2);
        int actualClass = (int)testLabels[i];

        string status = predictedClass == actualClass ? "✓" : "✗";
        Console.WriteLine($"  Sample {i + 1}: Predicted={speciesNames[predictedClass]}, Actual={speciesNames[actualClass]} {status}");
    }
}
catch (Exception ex)
{
    Console.WriteLine($"Note: Full training requires complete model implementation.");
    Console.WriteLine($"This sample demonstrates the facade pattern API for classification.");
    Console.WriteLine($"\nError details: {ex.Message}");
}

Console.WriteLine("\n=== Sample Complete ===");

// Helper function to generate Iris-like dataset
static (double[][] features, double[] labels) LoadIrisDataset()
{
    // Simplified Iris dataset (representative samples)
    // In production, load from CSV or use a data loading library
    var random = new Random(42);
    var features = new List<double[]>();
    var labels = new List<double>();

    // Setosa (class 0) - small petals
    for (int i = 0; i < 50; i++)
    {
        features.Add(new double[]
        {
            5.0 + random.NextDouble() * 0.8,  // sepal_length
            3.4 + random.NextDouble() * 0.4,  // sepal_width
            1.4 + random.NextDouble() * 0.3,  // petal_length
            0.2 + random.NextDouble() * 0.2   // petal_width
        });
        labels.Add(0);
    }

    // Versicolor (class 1) - medium petals
    for (int i = 0; i < 50; i++)
    {
        features.Add(new double[]
        {
            5.9 + random.NextDouble() * 0.9,
            2.8 + random.NextDouble() * 0.4,
            4.2 + random.NextDouble() * 0.6,
            1.3 + random.NextDouble() * 0.3
        });
        labels.Add(1);
    }

    // Virginica (class 2) - large petals
    for (int i = 0; i < 50; i++)
    {
        features.Add(new double[]
        {
            6.6 + random.NextDouble() * 0.8,
            3.0 + random.NextDouble() * 0.4,
            5.5 + random.NextDouble() * 0.6,
            2.0 + random.NextDouble() * 0.4
        });
        labels.Add(2);
    }

    // Shuffle the dataset
    var combined = features.Zip(labels, (f, l) => (f, l)).OrderBy(_ => random.Next()).ToList();

    return (combined.Select(x => x.f).ToArray(), combined.Select(x => x.l).ToArray());
}

// Pack a jagged feature array into the dense Matrix the model's Predict expects.
static Matrix<double> ToMatrix(double[][] rows)
{
    int r = rows.Length, c = rows[0].Length;
    var m = new Matrix<double>(r, c);
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            m[i, j] = rows[i][j];
    return m;
}
