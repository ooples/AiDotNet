using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Models.Options;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;

Console.WriteLine("=== AiDotNet Basic Regression ===");
Console.WriteLine("Predicting house prices using Gradient Boosting\n");

// Generate synthetic house price data
var (trainFeatures, trainPrices, testFeatures, testPrices) = GenerateHousePriceData();

Console.WriteLine($"Training set: {trainFeatures.Length} samples");
Console.WriteLine($"Test set: {testFeatures.Length} samples");
Console.WriteLine("\nFeatures: sqft, bedrooms, bathrooms, age, location_score\n");

// Build and train the regression model
Console.WriteLine("Building Gradient Boosting Regression model...");
Console.WriteLine("  - 100 estimators");
Console.WriteLine("  - Max depth: 5");
Console.WriteLine("  - Learning rate: 0.1\n");

try
{
    var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
        .ConfigureModel(new GradientBoostingRegression<double>(
            new GradientBoostingRegressionOptions
            {
                NumberOfTrees = 100,
                MaxDepth = 5,
                LearningRate = 0.1
            }))
        .ConfigureDataLoader(DataLoaders.FromArrays(trainFeatures, trainPrices));

    Console.WriteLine("Training...\n");

    var result = await builder.BuildAsync();

    // Evaluate on the held-out test set through the facade's rich metrics — no hand-rolled math.
    // GetDataSetStats runs the model and returns a full DataSetStats with error + prediction metrics.
    var testStats = result.GetDataSetStats(ToMatrix(testFeatures), new Vector<double>(testPrices));

    var predVector = testStats.Predicted;
    var predictions = new double[testFeatures.Length];
    for (int i = 0; i < testFeatures.Length; i++)
        predictions[i] = predVector[i];

    Console.WriteLine("Model Evaluation (held-out test set):");
    Console.WriteLine("─────────────────────────────────────");
    Console.WriteLine($"  R² Score:    {testStats.PredictionStats.R2:F4}");
    Console.WriteLine($"  Adjusted R²: {testStats.PredictionStats.AdjustedR2:F4}");
    Console.WriteLine($"  MAE:         ${testStats.ErrorStats.MAE:N0}");
    Console.WriteLine($"  RMSE:        ${testStats.ErrorStats.RMSE:N0}");
    Console.WriteLine($"  MAPE:        {testStats.ErrorStats.MAPE:F2}%");

    // The facade also auto-evaluates its own internal train/validation split and caches it on
    // result.Evaluation — read it directly, no need to re-pass any data.
    Console.WriteLine($"  Training R² (result.Evaluation): {result.Evaluation.TrainingSet.PredictionStats.R2:F4}");

    // Show sample predictions
    Console.WriteLine("\nSample Predictions:");
    Console.WriteLine("─────────────────────────────────────");

    for (int i = 0; i < Math.Min(5, testFeatures.Length); i++)
    {
        double predicted = predictions[i];
        double actual = testPrices[i];
        double errorPct = Math.Abs((predicted - actual) / actual) * 100;

        Console.WriteLine($"  House {i + 1}: Predicted=${predicted:N0}, Actual=${actual:N0} (Error: {errorPct:F1}%)");
    }

    // Feature importance (simplified - actual would come from model)
    Console.WriteLine("\nFeature Importance (approximate):");
    Console.WriteLine("─────────────────────────────────────");
    Console.WriteLine("  1. Square footage: High");
    Console.WriteLine("  2. Location score: High");
    Console.WriteLine("  3. Bedrooms: Medium");
    Console.WriteLine("  4. Bathrooms: Medium");
    Console.WriteLine("  5. Age: Low");
}
catch (Exception ex)
{
    Console.WriteLine($"Note: Full training requires complete model implementation.");
    Console.WriteLine($"This sample demonstrates the API pattern for regression.");
    Console.WriteLine($"\nError details: {ex.Message}");
}

Console.WriteLine("\n=== Sample Complete ===");

// Generate synthetic house price data
static (double[][] trainFeatures, double[] trainPrices, double[][] testFeatures, double[] testPrices) GenerateHousePriceData()
{
    var random = new Random(42);
    var allFeatures = new List<double[]>();
    var allPrices = new List<double>();

    for (int i = 0; i < 600; i++)
    {
        // Features: sqft, bedrooms, bathrooms, age, location_score
        double sqft = 1000 + random.NextDouble() * 3000;
        double bedrooms = Math.Floor(1 + random.NextDouble() * 5);
        double bathrooms = Math.Floor(1 + random.NextDouble() * 4);
        double age = random.NextDouble() * 50;
        double locationScore = random.NextDouble() * 10;

        allFeatures.Add(new double[] { sqft, bedrooms, bathrooms, age, locationScore });

        // Price formula with some noise
        double basePrice = 100000;
        double price = basePrice
            + sqft * 150                    // $150 per sqft
            + bedrooms * 20000              // $20k per bedroom
            + bathrooms * 15000             // $15k per bathroom
            - age * 1000                    // -$1k per year of age
            + locationScore * 30000         // $30k per location point
            + (random.NextDouble() - 0.5) * 50000;  // Random noise

        allPrices.Add(Math.Max(50000, price));  // Minimum $50k
    }

    // Shuffle and split
    var indices = Enumerable.Range(0, 600).OrderBy(_ => random.Next()).ToArray();
    var shuffledFeatures = indices.Select(i => allFeatures[i]).ToArray();
    var shuffledPrices = indices.Select(i => allPrices[i]).ToArray();

    return (
        shuffledFeatures.Take(500).ToArray(),
        shuffledPrices.Take(500).ToArray(),
        shuffledFeatures.Skip(500).ToArray(),
        shuffledPrices.Skip(500).ToArray()
    );
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
