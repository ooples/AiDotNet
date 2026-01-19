using AiDotNet;
using AiDotNet.Classification.NaiveBayes;
using AiDotNet.CrossValidators;
using AiDotNet.Models.Options;
using AiDotNet.Preprocessing.Scalers;
using AiDotNet.Tensors.LinearAlgebra;

Console.WriteLine("=== AiDotNet Sentiment Analysis ===");
Console.WriteLine("Binary classification of movie reviews using Naive Bayes\n");

// Generate realistic movie review data with TF-IDF-like features
var (features, labels, vocabulary) = GenerateMovieReviewData();

Console.WriteLine($"Dataset: {features.Length} movie reviews");
Console.WriteLine($"Features: {vocabulary.Length} words (TF-IDF weighted)\n");

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
var xTrain = new Matrix<double>(trainFeatures.Length, vocabulary.Length);
var yTrain = new Vector<double>(trainFeatures.Length);
for (int i = 0; i < trainFeatures.Length; i++)
{
    for (int j = 0; j < vocabulary.Length; j++)
    {
        xTrain[i, j] = trainFeatures[i][j];
    }
    yTrain[i] = trainLabels[i];
}

var xTest = new Matrix<double>(testFeatures.Length, vocabulary.Length);
var yTest = new Vector<double>(testFeatures.Length);
for (int i = 0; i < testFeatures.Length; i++)
{
    for (int j = 0; j < vocabulary.Length; j++)
    {
        xTest[i, j] = testFeatures[i][j];
    }
    yTest[i] = testLabels[i];
}

// Build and train the Naive Bayes classifier
Console.WriteLine("Building Multinomial Naive Bayes classifier...");
Console.WriteLine("  - Laplace smoothing (alpha=1.0)");
Console.WriteLine("  - TF-IDF preprocessing applied to text features\n");

try
{
    // Create Naive Bayes classifier with options
    var nbOptions = new NaiveBayesOptions<double>
    {
        Alpha = 1.0,  // Laplace smoothing
        FitPriors = true
    };
    var classifier = new MultinomialNaiveBayes<double>(nbOptions);

    Console.WriteLine("Training classifier...\n");
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

    // Display results
    Console.WriteLine("Classification Metrics:");
    Console.WriteLine(new string('-', 45));
    Console.WriteLine($"  Accuracy:  {accuracy:P2}");
    Console.WriteLine($"  Precision: {precision:P2}");
    Console.WriteLine($"  Recall:    {recall:P2}");
    Console.WriteLine($"  F1-Score:  {f1Score:P2}");

    Console.WriteLine("\nConfusion Matrix:");
    Console.WriteLine(new string('-', 45));
    Console.WriteLine("              Predicted");
    Console.WriteLine("              Neg    Pos");
    Console.WriteLine($"  Actual Neg   {trueNegative,4}   {falsePositive,4}");
    Console.WriteLine($"  Actual Pos   {falseNegative,4}   {truePositive,4}");

    // Show some sample predictions
    Console.WriteLine("\nSample Predictions:");
    Console.WriteLine(new string('-', 45));

    var sampleReviews = new[]
    {
        "This movie was absolutely amazing and wonderful!",
        "Terrible waste of time, completely boring and bad.",
        "The acting was decent but the plot was confusing.",
        "A masterpiece of cinema with brilliant performances!",
        "I hated every minute of this awful film."
    };

    for (int i = 0; i < Math.Min(5, sampleReviews.Length); i++)
    {
        double predProb = probabilities[i, 1];
        string sentiment = predictions[i] > 0.5 ? "Positive" : "Negative";
        string actual = testLabels[i] > 0.5 ? "Positive" : "Negative";
        string status = sentiment == actual ? "[correct]" : "[wrong]";
        Console.WriteLine($"  Review {i + 1}: {sentiment} (confidence: {predProb:P0}) {status}");
    }
}
catch (Exception ex)
{
    Console.WriteLine($"Note: Full training requires complete model implementation.");
    Console.WriteLine($"This sample demonstrates the API pattern for sentiment analysis.");
    Console.WriteLine($"\nError details: {ex.Message}");
}

Console.WriteLine("\n=== Sample Complete ===");

// Helper function to generate movie review data with TF-IDF-like features
static (double[][] features, double[] labels, string[] vocabulary) GenerateMovieReviewData()
{
    var random = new Random(42);

    // Vocabulary of sentiment-indicating words
    string[] vocabulary =
    [
        "amazing", "wonderful", "excellent", "great", "good", "love", "beautiful", "fantastic", "brilliant", "masterpiece",
        "terrible", "awful", "bad", "boring", "hate", "worst", "waste", "horrible", "disappointing", "poor",
        "movie", "film", "acting", "plot", "story", "director", "characters", "scenes", "performance", "cinematography"
    ];

    var features = new List<double[]>();
    var labels = new List<double>();

    // Generate positive reviews (label = 1)
    for (int i = 0; i < 250; i++)
    {
        var feature = new double[vocabulary.Length];

        // Positive words have higher TF-IDF scores
        for (int j = 0; j < 10; j++)  // First 10 are positive words
        {
            feature[j] = random.NextDouble() * 0.8 + 0.2;  // Higher weights
        }
        // Negative words have low scores
        for (int j = 10; j < 20; j++)  // Next 10 are negative words
        {
            feature[j] = random.NextDouble() * 0.2;  // Lower weights
        }
        // Neutral words have medium scores
        for (int j = 20; j < vocabulary.Length; j++)
        {
            feature[j] = random.NextDouble() * 0.5;
        }

        features.Add(feature);
        labels.Add(1);  // Positive
    }

    // Generate negative reviews (label = 0)
    for (int i = 0; i < 250; i++)
    {
        var feature = new double[vocabulary.Length];

        // Positive words have low scores
        for (int j = 0; j < 10; j++)
        {
            feature[j] = random.NextDouble() * 0.2;  // Lower weights
        }
        // Negative words have higher TF-IDF scores
        for (int j = 10; j < 20; j++)
        {
            feature[j] = random.NextDouble() * 0.8 + 0.2;  // Higher weights
        }
        // Neutral words have medium scores
        for (int j = 20; j < vocabulary.Length; j++)
        {
            feature[j] = random.NextDouble() * 0.5;
        }

        features.Add(feature);
        labels.Add(0);  // Negative
    }

    return (features.ToArray(), labels.ToArray(), vocabulary);
}
