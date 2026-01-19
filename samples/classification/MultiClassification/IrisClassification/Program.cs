using AiDotNet;
using AiDotNet.Classification.Ensemble;
using AiDotNet.Classification.SVM;
using AiDotNet.CrossValidation;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.Preprocessing.Scalers;

Console.WriteLine("=== AiDotNet Iris Classification ===");
Console.WriteLine("Multi-class classification comparing multiple classifiers using AiModelBuilder\n");

// Load the classic Iris dataset
var (features, labels) = LoadIrisDataset();
var speciesNames = new[] { "Setosa", "Versicolor", "Virginica" };

Console.WriteLine("Dataset: Iris flower classification");
Console.WriteLine("  - 150 samples, 4 features, 3 classes");
Console.WriteLine("  - Features: sepal_length, sepal_width, petal_length, petal_width");
Console.WriteLine($"  - Classes: {string.Join(", ", speciesNames)}\n");

// Display class distribution
var classDistribution = labels.GroupBy(l => (int)l).OrderBy(g => g.Key);
Console.WriteLine("Class distribution:");
foreach (var group in classDistribution)
{
    Console.WriteLine($"  - {speciesNames[group.Key]}: {group.Count()} samples");
}
Console.WriteLine();

// Split into train/test sets (80/20)
var random = new Random(42);
var indices = Enumerable.Range(0, features.Length).OrderBy(_ => random.Next()).ToArray();
var splitIndex = (int)(indices.Length * 0.8);

var trainIndices = indices.Take(splitIndex).ToArray();
var testIndices = indices.Skip(splitIndex).ToArray();

var trainFeatures = trainIndices.Select(i => features[i]).ToArray();
var trainLabels = trainIndices.Select(i => labels[i]).ToArray();
var testFeatures = testIndices.Select(i => features[i]).ToArray();
var testLabels = testIndices.Select(i => labels[i]).ToArray();

Console.WriteLine($"Training set: {trainFeatures.Length} samples");
Console.WriteLine($"Test set: {testFeatures.Length} samples\n");

// Define classifiers to compare - each will be used with AiModelBuilder
var classifierConfigs = new Dictionary<string, IFullModel<double, double[], double>>
{
    ["Random Forest"] = new RandomForestClassifier<double, double[], double>(
        nEstimators: 100,
        maxDepth: 10,
        minSamplesSplit: 2),
    ["Gradient Boosting"] = new GradientBoostingClassifier<double, double[], double>(
        nEstimators: 100,
        learningRate: 0.1,
        maxDepth: 3),
    ["SVM (RBF)"] = new SupportVectorClassifier<double, double[], double>(
        c: 1.0,
        kernel: KernelType.RBF,
        gamma: 0.1)
};

Console.WriteLine("╔════════════════════════════════════════════════════════════════╗");
Console.WriteLine("║     CLASSIFIER COMPARISON USING PREDICTIONMODELBUILDER         ║");
Console.WriteLine("╚════════════════════════════════════════════════════════════════╝\n");

var results = new Dictionary<string, (AiModelResult<double, double[], double>? result, double testAcc)>();

foreach (var (name, classifier) in classifierConfigs)
{
    Console.WriteLine($"Training {name} with AiModelBuilder...");

    try
    {
        // Use the AiModelBuilder facade to train each classifier
        var builder = new AiModelBuilder<double, double[], double>()
            .ConfigureModel(classifier)
            .ConfigurePreprocessing(pipeline => pipeline
                .Add(new StandardScaler<double>()))
            .ConfigureCrossValidation(new KFoldCrossValidator<double, double[], double>(k: 5));

        var result = await builder.BuildAsync(trainFeatures, trainLabels);

        // Display cross-validation results from the result object
        if (result.CrossValidationResult != null)
        {
            var cv = result.CrossValidationResult;
            Console.WriteLine($"  CV Accuracy: {cv.MeanScore:P2} (+/- {cv.StandardDeviation:P2})");
        }

        // Evaluate on test set using result.Predict() - the facade pattern
        int correct = 0;
        for (int i = 0; i < testFeatures.Length; i++)
        {
            // Use result.Predict() directly - NOT result.Model.Predict()
            var prediction = result.Predict(testFeatures[i]);
            if (Math.Abs(Math.Round(prediction) - testLabels[i]) < 0.5)
                correct++;
        }
        double testAccuracy = (double)correct / testFeatures.Length;

        results[name] = (result, testAccuracy);
        Console.WriteLine($"  Test Accuracy: {testAccuracy:P2}\n");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"  Error: {ex.Message}\n");
        results[name] = (null, 0);
    }
}

// Summary comparison table
Console.WriteLine("╔════════════════════════════════════════════════════════════════╗");
Console.WriteLine("║                     RESULTS SUMMARY                            ║");
Console.WriteLine("╚════════════════════════════════════════════════════════════════╝\n");

Console.WriteLine("Summary Statistics:");
Console.WriteLine(new string('-', 55));
Console.WriteLine($"{"Classifier",-20} {"Mean CV Acc",-15} {"Std Dev",-12} {"Test Acc",-12}");
Console.WriteLine(new string('-', 55));

foreach (var (name, (result, testAcc)) in results.Where(r => r.Value.result != null).OrderByDescending(r => r.Value.testAcc))
{
    var cv = result!.CrossValidationResult;
    Console.WriteLine($"{name,-20} {cv?.MeanScore ?? 0:P2,-15} {cv?.StandardDeviation ?? 0:P2,-12} {testAcc:P2,-12}");
}

// Best model selection
var bestEntry = results.Where(r => r.Value.result != null).OrderByDescending(r => r.Value.testAcc).FirstOrDefault();
if (bestEntry.Value.result != null)
{
    Console.WriteLine($"\nBest performing model: {bestEntry.Key} (Test Accuracy: {bestEntry.Value.testAcc:P2})");

    // Detailed predictions for best model
    Console.WriteLine("\n╔════════════════════════════════════════════════════════════════╗");
    Console.WriteLine($"║     SAMPLE PREDICTIONS ({bestEntry.Key.ToUpper(),-15})                    ║");
    Console.WriteLine("╚════════════════════════════════════════════════════════════════╝\n");

    Console.WriteLine("Sample Predictions:");
    Console.WriteLine(new string('-', 60));

    var bestResult = bestEntry.Value.result;
    for (int i = 0; i < Math.Min(10, testFeatures.Length); i++)
    {
        // Use result.Predict() - the facade pattern
        var prediction = bestResult.Predict(testFeatures[i]);
        int predicted = (int)Math.Round(Math.Clamp(prediction, 0, 2));
        int actual = (int)testLabels[i];
        string status = predicted == actual ? "[correct]" : "[WRONG]";

        Console.WriteLine($"  Sample {i + 1,2}: Predicted={speciesNames[predicted],-12} Actual={speciesNames[actual],-12} {status}");
    }

    // Show how to save and load the model through the result object
    Console.WriteLine("\n// Model Persistence Example (using the result object):");
    Console.WriteLine("// await bestResult.SaveModelAsync(\"iris_model.json\");");
    Console.WriteLine("// var loadedResult = await AiModelResult<double, double[], double>.LoadModelAsync(\"iris_model.json\");");
    Console.WriteLine("// var prediction = loadedResult.Predict(newSample);");
}

Console.WriteLine("\n=== Sample Complete ===");

// Helper function to load the Iris dataset
static (double[][] features, double[] labels) LoadIrisDataset()
{
    var random = new Random(42);
    var features = new List<double[]>();
    var labels = new List<double>();

    // Setosa (class 0) - distinctive small petals
    double[][] setosaBase =
    [
        [5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2],
        [4.6, 3.1, 1.5, 0.2], [5.0, 3.6, 1.4, 0.2], [5.4, 3.9, 1.7, 0.4],
        [4.6, 3.4, 1.4, 0.3], [5.0, 3.4, 1.5, 0.2], [4.4, 2.9, 1.4, 0.2],
        [4.9, 3.1, 1.5, 0.1], [5.4, 3.7, 1.5, 0.2], [4.8, 3.4, 1.6, 0.2],
        [4.8, 3.0, 1.4, 0.1], [4.3, 3.0, 1.1, 0.1], [5.8, 4.0, 1.2, 0.2],
        [5.7, 4.4, 1.5, 0.4], [5.4, 3.9, 1.3, 0.4], [5.1, 3.5, 1.4, 0.3]
    ];

    foreach (var row in setosaBase)
    {
        features.Add(row);
        labels.Add(0);

        for (int i = 0; i < 2; i++)
        {
            features.Add(row.Select(v => v + (random.NextDouble() - 0.5) * 0.2).ToArray());
            labels.Add(0);
        }
    }

    // Versicolor (class 1) - medium sized
    double[][] versicolorBase =
    [
        [7.0, 3.2, 4.7, 1.4], [6.4, 3.2, 4.5, 1.5], [6.9, 3.1, 4.9, 1.5],
        [5.5, 2.3, 4.0, 1.3], [6.5, 2.8, 4.6, 1.5], [5.7, 2.8, 4.5, 1.3],
        [6.3, 3.3, 4.7, 1.6], [4.9, 2.4, 3.3, 1.0], [6.6, 2.9, 4.6, 1.3],
        [5.2, 2.7, 3.9, 1.4], [5.0, 2.0, 3.5, 1.0], [5.9, 3.0, 4.2, 1.5],
        [6.0, 2.2, 4.0, 1.0], [6.1, 2.9, 4.7, 1.4], [5.6, 2.9, 3.6, 1.3],
        [6.7, 3.1, 4.4, 1.4], [5.6, 3.0, 4.5, 1.5], [5.8, 2.7, 4.1, 1.0]
    ];

    foreach (var row in versicolorBase)
    {
        features.Add(row);
        labels.Add(1);

        for (int i = 0; i < 2; i++)
        {
            features.Add(row.Select(v => v + (random.NextDouble() - 0.5) * 0.2).ToArray());
            labels.Add(1);
        }
    }

    // Virginica (class 2) - largest
    double[][] virginicaBase =
    [
        [6.3, 3.3, 6.0, 2.5], [5.8, 2.7, 5.1, 1.9], [7.1, 3.0, 5.9, 2.1],
        [6.3, 2.9, 5.6, 1.8], [6.5, 3.0, 5.8, 2.2], [7.6, 3.0, 6.6, 2.1],
        [4.9, 2.5, 4.5, 1.7], [7.3, 2.9, 6.3, 1.8], [6.7, 2.5, 5.8, 1.8],
        [7.2, 3.6, 6.1, 2.5], [6.5, 3.2, 5.1, 2.0], [6.4, 2.7, 5.3, 1.9],
        [6.8, 3.0, 5.5, 2.1], [5.7, 2.5, 5.0, 2.0], [5.8, 2.8, 5.1, 2.4],
        [6.4, 3.2, 5.3, 2.3], [6.5, 3.0, 5.5, 1.8], [7.7, 3.8, 6.7, 2.2]
    ];

    foreach (var row in virginicaBase)
    {
        features.Add(row);
        labels.Add(2);

        for (int i = 0; i < 2; i++)
        {
            features.Add(row.Select(v => v + (random.NextDouble() - 0.5) * 0.2).ToArray());
            labels.Add(2);
        }
    }

    // Shuffle the dataset
    var combined = features.Zip(labels, (f, l) => (f, l)).OrderBy(_ => random.Next()).ToList();

    return (combined.Select(x => x.f).ToArray(), combined.Select(x => x.l).ToArray());
}
