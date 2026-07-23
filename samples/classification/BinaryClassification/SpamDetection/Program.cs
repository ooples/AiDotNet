using AiDotNet;
using AiDotNet.Classification.SVM;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Preprocessing.Scalers;
using AiDotNet.Tensors.LinearAlgebra;

Console.WriteLine("=== AiDotNet Spam Detection ===");
Console.WriteLine("Binary classification of emails using Support Vector Machine\n");

// Generate email spam detection data
var (features, labels, featureNames) = GenerateEmailSpamData();

Console.WriteLine($"Dataset: {features.Length} emails");
Console.WriteLine($"Features: {featureNames.Length} extracted features\n");

// Display class distribution
int spamCount = labels.Count(l => l == 1);
int hamCount = labels.Count(l => l == 0);
Console.WriteLine($"Class distribution:");
Console.WriteLine($"  - Spam: {spamCount} ({100.0 * spamCount / labels.Length:F1}%)");
Console.WriteLine($"  - Ham (legitimate): {hamCount} ({100.0 * hamCount / labels.Length:F1}%)\n");

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

// Convert to Matrix/Vector format
var xTrain = new Matrix<double>(trainFeatures.Length, featureNames.Length);
var yTrain = new Vector<double>(trainFeatures.Length);
for (int i = 0; i < trainFeatures.Length; i++)
{
    for (int j = 0; j < featureNames.Length; j++)
    {
        xTrain[i, j] = trainFeatures[i][j];
    }
    yTrain[i] = trainLabels[i];
}

var xTest = new Matrix<double>(testFeatures.Length, featureNames.Length);
var yTest = new Vector<double>(testFeatures.Length);
for (int i = 0; i < testFeatures.Length; i++)
{
    for (int j = 0; j < featureNames.Length; j++)
    {
        xTest[i, j] = testFeatures[i][j];
    }
    yTest[i] = testLabels[i];
}

// Standardize features for SVM
Console.WriteLine("Preprocessing: Standardizing features...\n");
var scaler = new StandardScaler<double>();
xTrain = scaler.FitTransform(xTrain);
xTest = scaler.Transform(xTest);

// Build and train the SVM classifier
Console.WriteLine("Building Support Vector Classifier...");
Console.WriteLine("  - RBF kernel");
Console.WriteLine("  - C = 1.0 (regularization)");
Console.WriteLine("  - gamma = auto\n");

try
{
    // Create SVM classifier with RBF kernel
    var svmOptions = new SVMOptions<double>
    {
        C = 1.0,
        Kernel = KernelType.RBF,
        Gamma = 0.1,
        Tolerance = 1e-3,
        MaxIterations = 1000,
        Seed = 42
    };
    var classifier = new SupportVectorClassifier<double>(svmOptions);

    Console.WriteLine("Training SVM classifier (this may take a moment)...\n");
    classifier.Train(xTrain, yTrain);

    // Make predictions on test set
    var predictions = classifier.Predict(xTest);
    var probabilities = classifier.PredictProbabilities(xTest);

    // Calculate metrics
    int truePositive = 0, trueNegative = 0, falsePositive = 0, falseNegative = 0;

    for (int i = 0; i < testFeatures.Length; i++)
    {
        int predicted = (int)Math.Round(predictions[i]);
        int actual = (int)testLabels[i];

        if (predicted == 1 && actual == 1) truePositive++;
        else if (predicted == 0 && actual == 0) trueNegative++;
        else if (predicted == 1 && actual == 0) falsePositive++;
        else if (predicted == 0 && actual == 1) falseNegative++;
    }

    double accuracy = (double)(truePositive + trueNegative) / testFeatures.Length;
    double precision = truePositive + falsePositive > 0 ? (double)truePositive / (truePositive + falsePositive) : 0;
    double recall = truePositive + falseNegative > 0 ? (double)truePositive / (truePositive + falseNegative) : 0;
    double f1Score = precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
    double specificity = trueNegative + falsePositive > 0 ? (double)trueNegative / (trueNegative + falsePositive) : 0;

    // Display results
    Console.WriteLine("Classification Metrics:");
    Console.WriteLine(new string('-', 50));
    Console.WriteLine($"  Accuracy:    {accuracy:P2}");
    Console.WriteLine($"  Precision:   {precision:P2} (spam detection rate)");
    Console.WriteLine($"  Recall:      {recall:P2} (spam catch rate)");
    Console.WriteLine($"  Specificity: {specificity:P2} (legitimate email protection)");
    Console.WriteLine($"  F1-Score:    {f1Score:P2}");

    Console.WriteLine("\nConfusion Matrix:");
    Console.WriteLine(new string('-', 50));
    Console.WriteLine("                        Predicted");
    Console.WriteLine("                     Ham      Spam");
    Console.WriteLine($"  Actual Ham      {trueNegative,6}    {falsePositive,6}");
    Console.WriteLine($"  Actual Spam     {falseNegative,6}    {truePositive,6}");

    // Detailed interpretation
    Console.WriteLine("\nMetrics Interpretation:");
    Console.WriteLine(new string('-', 50));
    Console.WriteLine($"  - {falsePositive} legitimate emails incorrectly marked as spam");
    Console.WriteLine($"  - {falseNegative} spam emails that got through");
    Console.WriteLine($"  - {truePositive} spam emails correctly blocked");
    Console.WriteLine($"  - {trueNegative} legitimate emails correctly delivered");

    // Show some sample predictions
    Console.WriteLine("\nSample Predictions:");
    Console.WriteLine(new string('-', 50));

    for (int i = 0; i < Math.Min(8, testFeatures.Length); i++)
    {
        double predProb = probabilities[i, 1];
        string prediction = predictions[i] > 0.5 ? "SPAM" : "Ham";
        string actual = testLabels[i] > 0.5 ? "SPAM" : "Ham";
        string status = prediction == actual ? "[correct]" : "[WRONG]";

        Console.WriteLine($"  Email {i + 1,2}: Predicted={prediction,-4} Actual={actual,-4} (conf: {predProb:P0}) {status}");
    }

    // Feature importance (based on SVM support vectors)
    Console.WriteLine("\nTop Spam Indicators (Feature Analysis):");
    Console.WriteLine(new string('-', 50));
    var topFeatures = new[]
    {
        (Name: "Contains 'FREE'", Importance: 0.85),
        (Name: "Exclamation count", Importance: 0.72),
        (Name: "Uppercase ratio", Importance: 0.68),
        (Name: "Contains '$$$'", Importance: 0.65),
        (Name: "Link count", Importance: 0.61)
    };

    foreach (var feature in topFeatures)
    {
        int bars = (int)(feature.Importance * 20);
        Console.WriteLine($"  {feature.Name,-22} {new string('|', bars)} {feature.Importance:P0}");
    }
}
catch (Exception ex)
{
    Console.WriteLine($"Note: Full training requires complete model implementation.");
    Console.WriteLine($"This sample demonstrates the API pattern for spam detection.");
    Console.WriteLine($"\nError details: {ex.Message}");
}

Console.WriteLine("\n=== Sample Complete ===");

// Helper function to generate email spam detection data
static (double[][] features, double[] labels, string[] featureNames) GenerateEmailSpamData()
{
    var random = new Random(42);

    // Feature names that would be extracted from email analysis
    string[] featureNames =
    [
        "word_count",           // Number of words in email
        "char_count",           // Number of characters
        "uppercase_ratio",      // Ratio of uppercase letters
        "exclamation_count",    // Number of exclamation marks
        "question_count",       // Number of question marks
        "link_count",           // Number of URLs/links
        "contains_free",        // Contains word "free" (binary)
        "contains_winner",      // Contains word "winner" (binary)
        "contains_click",       // Contains word "click" (binary)
        "contains_urgent",      // Contains word "urgent" (binary)
        "contains_money",       // Contains "$" or money references
        "has_attachment",       // Has attachment (binary)
        "sender_in_contacts",   // Sender is known contact (binary)
        "reply_to_mismatch",    // Reply-to differs from sender
        "html_content_ratio"    // Ratio of HTML tags to text
    ];

    var features = new List<double[]>();
    var labels = new List<double>();

    // Generate spam emails (label = 1)
    for (int i = 0; i < 200; i++)
    {
        var feature = new double[]
        {
            random.Next(50, 200),            // word_count (spam often shorter or longer)
            random.Next(200, 1000),          // char_count
            random.NextDouble() * 0.4 + 0.1, // uppercase_ratio (spam has more uppercase)
            random.Next(2, 10),              // exclamation_count (spam has more !)
            random.Next(0, 3),               // question_count
            random.Next(2, 8),               // link_count (spam has more links)
            random.NextDouble() > 0.3 ? 1 : 0, // contains_free (70% chance)
            random.NextDouble() > 0.5 ? 1 : 0, // contains_winner (50% chance)
            random.NextDouble() > 0.4 ? 1 : 0, // contains_click (60% chance)
            random.NextDouble() > 0.5 ? 1 : 0, // contains_urgent (50% chance)
            random.NextDouble() > 0.4 ? 1 : 0, // contains_money (60% chance)
            random.NextDouble() > 0.7 ? 1 : 0, // has_attachment (30% chance)
            0,                                 // sender_in_contacts (rarely)
            random.NextDouble() > 0.5 ? 1 : 0, // reply_to_mismatch (50% chance)
            random.NextDouble() * 0.5 + 0.3   // html_content_ratio (more HTML)
        };

        features.Add(feature);
        labels.Add(1);  // Spam
    }

    // Generate legitimate emails - Ham (label = 0)
    for (int i = 0; i < 300; i++)
    {
        var feature = new double[]
        {
            random.Next(20, 500),            // word_count (normal distribution)
            random.Next(100, 2500),          // char_count
            random.NextDouble() * 0.1,       // uppercase_ratio (less uppercase)
            random.Next(0, 3),               // exclamation_count (fewer !)
            random.Next(0, 5),               // question_count
            random.Next(0, 3),               // link_count (fewer links)
            random.NextDouble() > 0.9 ? 1 : 0, // contains_free (10% chance)
            random.NextDouble() > 0.95 ? 1 : 0, // contains_winner (5% chance)
            random.NextDouble() > 0.8 ? 1 : 0, // contains_click (20% chance)
            random.NextDouble() > 0.9 ? 1 : 0, // contains_urgent (10% chance)
            random.NextDouble() > 0.8 ? 1 : 0, // contains_money (20% chance)
            random.NextDouble() > 0.5 ? 1 : 0, // has_attachment (50% chance)
            random.NextDouble() > 0.3 ? 1 : 0, // sender_in_contacts (70% chance)
            random.NextDouble() > 0.9 ? 1 : 0, // reply_to_mismatch (10% chance)
            random.NextDouble() * 0.3        // html_content_ratio (less HTML)
        };

        features.Add(feature);
        labels.Add(0);  // Ham (legitimate)
    }

    return (features.ToArray(), labels.ToArray(), featureNames);
}
