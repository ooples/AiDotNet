using AiDotNet;
using AiDotNet.Classification;
using AiDotNet.CrossValidation;
using AiDotNet.Preprocessing.Scalers;

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

// Build and train the classifier
Console.WriteLine("Building Random Forest classifier...");
Console.WriteLine("  - 100 decision trees");
Console.WriteLine("  - StandardScaler preprocessing");
Console.WriteLine("  - 5-fold cross-validation\n");

try
{
    var builder = new PredictionModelBuilder<double, double[], double>()
        .ConfigureModel(new RandomForestClassifier<double, double[], double>(
            nEstimators: 100,
            maxDepth: 10,
            minSamplesSplit: 2))
        .ConfigurePreprocessing(pipeline => pipeline
            .Add(new StandardScaler<double>()))
        .ConfigureCrossValidation(new KFoldCrossValidator<double, double[], double>(k: 5));

    Console.WriteLine("Training with cross-validation...\n");

    var result = await builder.BuildAsync(trainFeatures, trainLabels);

    // Display cross-validation results
    if (result.CrossValidationResult != null)
    {
        Console.WriteLine("Cross-Validation Results:");
        Console.WriteLine("─────────────────────────────────────");

        var cvResult = result.CrossValidationResult;
        for (int i = 0; i < cvResult.FoldScores.Count; i++)
        {
            Console.WriteLine($"  Fold {i + 1}: Accuracy = {cvResult.FoldScores[i]:P2}");
        }

        Console.WriteLine($"\n  Mean Accuracy: {cvResult.MeanScore:P2} (+/- {cvResult.StandardDeviation:P2})");
    }

    // Evaluate on test set
    Console.WriteLine("\nFinal Model Evaluation:");
    Console.WriteLine("─────────────────────────────────────");

    int correct = 0;
    for (int i = 0; i < testFeatures.Length; i++)
    {
        var prediction = result.Model!.Predict(testFeatures[i]);
        if (Math.Abs(prediction - testLabels[i]) < 0.5)
            correct++;
    }

    double testAccuracy = (double)correct / testFeatures.Length;
    Console.WriteLine($"  Test Accuracy: {testAccuracy:P2}");

    // Show some predictions
    Console.WriteLine("\nSample Predictions:");
    Console.WriteLine("─────────────────────────────────────");
    string[] speciesNames = { "Setosa", "Versicolor", "Virginica" };

    for (int i = 0; i < Math.Min(5, testFeatures.Length); i++)
    {
        var prediction = result.Model!.Predict(testFeatures[i]);
        int predictedClass = (int)Math.Round(prediction);
        int actualClass = (int)testLabels[i];

        string status = predictedClass == actualClass ? "✓" : "✗";
        Console.WriteLine($"  Sample {i + 1}: Predicted={speciesNames[predictedClass]}, Actual={speciesNames[actualClass]} {status}");
    }
}
catch (Exception ex)
{
    Console.WriteLine($"Note: Full training requires complete model implementation.");
    Console.WriteLine($"This sample demonstrates the API pattern for classification.");
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
