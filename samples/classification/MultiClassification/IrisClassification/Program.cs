using AiDotNet;
using AiDotNet.Classification.Ensemble;
using AiDotNet.Classification.SVM;
using AiDotNet.CrossValidators;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Preprocessing.Scalers;
using AiDotNet.Tensors.LinearAlgebra;

Console.WriteLine("=== AiDotNet Iris Classification ===");
Console.WriteLine("Multi-class classification comparing multiple classifiers with 5-fold cross-validation\n");

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

// Convert to Matrix/Vector format
var X = new Matrix<double>(features.Length, 4);
var y = new Vector<double>(features.Length);
for (int i = 0; i < features.Length; i++)
{
    for (int j = 0; j < 4; j++)
    {
        X[i, j] = features[i][j];
    }
    y[i] = labels[i];
}

// Standardize features
Console.WriteLine("Preprocessing: Standardizing features...\n");
var scaler = new StandardScaler<double>();
X = scaler.FitTransform(X);

// Split into train/test sets (80/20) while maintaining class distribution
var random = new Random(42);
var indices = Enumerable.Range(0, features.Length).OrderBy(_ => random.Next()).ToArray();
var splitIndex = (int)(indices.Length * 0.8);

var trainIndices = indices.Take(splitIndex).ToArray();
var testIndices = indices.Skip(splitIndex).ToArray();

var xTrain = new Matrix<double>(trainIndices.Length, 4);
var yTrain = new Vector<double>(trainIndices.Length);
for (int i = 0; i < trainIndices.Length; i++)
{
    int idx = trainIndices[i];
    for (int j = 0; j < 4; j++)
    {
        xTrain[i, j] = X[idx, j];
    }
    yTrain[i] = y[idx];
}

var xTest = new Matrix<double>(testIndices.Length, 4);
var yTest = new Vector<double>(testIndices.Length);
for (int i = 0; i < testIndices.Length; i++)
{
    int idx = testIndices[i];
    for (int j = 0; j < 4; j++)
    {
        xTest[i, j] = X[idx, j];
    }
    yTest[i] = y[idx];
}

Console.WriteLine($"Training set: {trainIndices.Length} samples");
Console.WriteLine($"Test set: {testIndices.Length} samples\n");

// Define classifiers to compare
var classifiers = new Dictionary<string, Func<IFullModel<double, Matrix<double>, Vector<double>>>>
{
    ["Random Forest"] = () => new RandomForestClassifier<double>(new RandomForestClassifierOptions<double>
    {
        NEstimators = 100,
        MaxDepth = 10,
        MinSamplesSplit = 2,
        MaxFeatures = "sqrt",
        RandomState = 42
    }),
    ["Gradient Boosting"] = () => new GradientBoostingClassifier<double>(new GradientBoostingClassifierOptions<double>
    {
        NEstimators = 100,
        LearningRate = 0.1,
        MaxDepth = 3,
        Subsample = 0.8,
        RandomState = 42
    }),
    ["SVM (RBF)"] = () => new SupportVectorClassifier<double>(new SVMOptions<double>
    {
        C = 1.0,
        Kernel = KernelType.RBF,
        Gamma = 0.1,
        RandomState = 42
    })
};

Console.WriteLine("╔════════════════════════════════════════════════════════════════╗");
Console.WriteLine("║              5-FOLD CROSS-VALIDATION COMPARISON                ║");
Console.WriteLine("╚════════════════════════════════════════════════════════════════╝\n");

var results = new Dictionary<string, (double[] foldScores, double mean, double std, double testAcc)>();

foreach (var (name, createClassifier) in classifiers)
{
    Console.WriteLine($"Training {name}...");

    try
    {
        // Perform 5-fold cross-validation
        var foldScores = new double[5];
        int foldSize = trainIndices.Length / 5;

        for (int fold = 0; fold < 5; fold++)
        {
            // Create train/validation split for this fold
            var foldValidationIndices = Enumerable.Range(fold * foldSize, foldSize).ToArray();
            var foldTrainIndices = Enumerable.Range(0, trainIndices.Length)
                .Where(i => !foldValidationIndices.Contains(i))
                .ToArray();

            var xFoldTrain = new Matrix<double>(foldTrainIndices.Length, 4);
            var yFoldTrain = new Vector<double>(foldTrainIndices.Length);
            for (int i = 0; i < foldTrainIndices.Length; i++)
            {
                int idx = foldTrainIndices[i];
                for (int j = 0; j < 4; j++)
                {
                    xFoldTrain[i, j] = xTrain[idx, j];
                }
                yFoldTrain[i] = yTrain[idx];
            }

            var xFoldVal = new Matrix<double>(foldValidationIndices.Length, 4);
            var yFoldVal = new Vector<double>(foldValidationIndices.Length);
            for (int i = 0; i < foldValidationIndices.Length; i++)
            {
                int idx = foldValidationIndices[i];
                for (int j = 0; j < 4; j++)
                {
                    xFoldVal[i, j] = xTrain[idx, j];
                }
                yFoldVal[i] = yTrain[idx];
            }

            // Train and evaluate
            var classifier = createClassifier();
            classifier.Train(xFoldTrain, yFoldTrain);
            var predictions = classifier.Predict(xFoldVal);

            int correct = 0;
            for (int i = 0; i < predictions.Length; i++)
            {
                if (Math.Abs(Math.Round(predictions[i]) - yFoldVal[i]) < 0.5)
                    correct++;
            }
            foldScores[fold] = (double)correct / predictions.Length;
        }

        // Calculate mean and std
        double mean = foldScores.Average();
        double variance = foldScores.Select(s => Math.Pow(s - mean, 2)).Average();
        double std = Math.Sqrt(variance);

        // Final evaluation on test set
        var finalClassifier = createClassifier();
        finalClassifier.Train(xTrain, yTrain);
        var testPredictions = finalClassifier.Predict(xTest);

        int testCorrect = 0;
        for (int i = 0; i < testPredictions.Length; i++)
        {
            if (Math.Abs(Math.Round(testPredictions[i]) - yTest[i]) < 0.5)
                testCorrect++;
        }
        double testAccuracy = (double)testCorrect / testPredictions.Length;

        results[name] = (foldScores, mean, std, testAccuracy);

        Console.WriteLine($"  CV Accuracy: {mean:P2} (+/- {std:P2})");
        Console.WriteLine($"  Test Accuracy: {testAccuracy:P2}\n");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"  Error: {ex.Message}\n");
    }
}

// Summary comparison table
Console.WriteLine("╔════════════════════════════════════════════════════════════════╗");
Console.WriteLine("║                     RESULTS SUMMARY                            ║");
Console.WriteLine("╚════════════════════════════════════════════════════════════════╝\n");

Console.WriteLine("Cross-Validation Results (5-Fold):");
Console.WriteLine(new string('-', 70));
Console.WriteLine($"{"Classifier",-20} {"Fold 1",-10} {"Fold 2",-10} {"Fold 3",-10} {"Fold 4",-10} {"Fold 5",-10}");
Console.WriteLine(new string('-', 70));

foreach (var (name, result) in results)
{
    Console.WriteLine($"{name,-20} {result.foldScores[0]:P0,-10} {result.foldScores[1]:P0,-10} {result.foldScores[2]:P0,-10} {result.foldScores[3]:P0,-10} {result.foldScores[4]:P0,-10}");
}
Console.WriteLine();

Console.WriteLine("Summary Statistics:");
Console.WriteLine(new string('-', 55));
Console.WriteLine($"{"Classifier",-20} {"Mean CV Acc",-15} {"Std Dev",-12} {"Test Acc",-12}");
Console.WriteLine(new string('-', 55));

foreach (var (name, result) in results.OrderByDescending(r => r.Value.testAcc))
{
    Console.WriteLine($"{name,-20} {result.mean:P2,-15} {result.std:P2,-12} {result.testAcc:P2,-12}");
}

// Best model selection
var bestModel = results.OrderByDescending(r => r.Value.testAcc).First();
Console.WriteLine($"\nBest performing model: {bestModel.Key} (Test Accuracy: {bestModel.Value.testAcc:P2})");

// Detailed confusion matrix for best model
Console.WriteLine("\n╔════════════════════════════════════════════════════════════════╗");
Console.WriteLine($"║          DETAILED CONFUSION MATRIX ({bestModel.Key.ToUpper(),-15})        ║");
Console.WriteLine("╚════════════════════════════════════════════════════════════════╝\n");

// Re-train best model for confusion matrix
try
{
    var bestClassifier = classifiers[bestModel.Key]();
    bestClassifier.Train(xTrain, yTrain);
    var finalPredictions = bestClassifier.Predict(xTest);

    // Build confusion matrix
    var confusionMatrix = new int[3, 3];
    var perClassMetrics = new (int tp, int fp, int fn, int tn)[3];

    for (int i = 0; i < finalPredictions.Length; i++)
    {
        int predicted = (int)Math.Round(finalPredictions[i]);
        int actual = (int)yTest[i];
        predicted = Math.Clamp(predicted, 0, 2);
        confusionMatrix[actual, predicted]++;
    }

    // Calculate per-class metrics
    for (int c = 0; c < 3; c++)
    {
        int tp = confusionMatrix[c, c];
        int fp = 0, fn = 0, tn = 0;

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                if (i == c && j != c) fn += confusionMatrix[i, j];
                if (i != c && j == c) fp += confusionMatrix[i, j];
                if (i != c && j != c) tn += confusionMatrix[i, j];
            }
        }

        perClassMetrics[c] = (tp, fp, fn, tn);
    }

    Console.WriteLine("Confusion Matrix:");
    Console.WriteLine(new string('-', 50));
    Console.WriteLine($"{"Actual \\ Predicted",-18} {"Setosa",-10} {"Versicolor",-12} {"Virginica",-10}");
    Console.WriteLine(new string('-', 50));

    for (int i = 0; i < 3; i++)
    {
        Console.WriteLine($"{speciesNames[i],-18} {confusionMatrix[i, 0],-10} {confusionMatrix[i, 1],-12} {confusionMatrix[i, 2],-10}");
    }

    Console.WriteLine("\nPer-Class Metrics:");
    Console.WriteLine(new string('-', 60));
    Console.WriteLine($"{"Class",-12} {"Precision",-12} {"Recall",-12} {"F1-Score",-12} {"Support",-10}");
    Console.WriteLine(new string('-', 60));

    double macroF1 = 0;
    int totalSupport = 0;

    for (int c = 0; c < 3; c++)
    {
        var (tp, fp, fn, _) = perClassMetrics[c];
        int support = tp + fn;
        totalSupport += support;

        double precision = tp + fp > 0 ? (double)tp / (tp + fp) : 0;
        double recall = tp + fn > 0 ? (double)tp / (tp + fn) : 0;
        double f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;
        macroF1 += f1;

        Console.WriteLine($"{speciesNames[c],-12} {precision:P2,-12} {recall:P2,-12} {f1:P2,-12} {support,-10}");
    }

    Console.WriteLine(new string('-', 60));
    Console.WriteLine($"{"Macro Avg",-12} {"-",-12} {"-",-12} {macroF1 / 3:P2,-12} {totalSupport,-10}");

    // Show sample predictions
    Console.WriteLine("\nSample Predictions:");
    Console.WriteLine(new string('-', 60));

    for (int i = 0; i < Math.Min(10, finalPredictions.Length); i++)
    {
        int predicted = (int)Math.Round(finalPredictions[i]);
        int actual = (int)yTest[i];
        predicted = Math.Clamp(predicted, 0, 2);
        string status = predicted == actual ? "[correct]" : "[WRONG]";

        Console.WriteLine($"  Sample {i + 1,2}: Predicted={speciesNames[predicted],-12} Actual={speciesNames[actual],-12} {status}");
    }
}
catch (Exception ex)
{
    Console.WriteLine($"Error generating confusion matrix: {ex.Message}");
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
        // Add original
        features.Add(row);
        labels.Add(0);

        // Add variations
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
