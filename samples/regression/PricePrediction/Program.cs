// AiDotNet — House Price Prediction with Hyperparameter Optimization
//
// Regression + hyperparameter search through the AiModelBuilder facade:
// ConfigureHyperparameterOptimizer runs an optimizer over a search space during
// BuildAsync and returns the best AiModelResult, which you predict through.

using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.HyperparameterOptimization;   // RandomSearchOptimizer
using AiDotNet.Models;                        // HyperparameterSearchSpace
using AiDotNet.Models.Options;               // GradientBoostingRegressionOptions
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;

Console.WriteLine("=== AiDotNet House Price Prediction ===");
Console.WriteLine("Gradient Boosting + hyperparameter search via the facade\n");

var (trainFeatures, trainPrices, testFeatures, testPrices) = GenerateHousePriceData();
Console.WriteLine($"Training set: {trainFeatures.Length} samples, Test set: {testFeatures.Length} samples");
Console.WriteLine("Features: sqft, bedrooms, bathrooms, age, location_score\n");

// ── Hyperparameter optimizer + search space ────────────────────────────────
var hpo = new RandomSearchOptimizer<double, Matrix<double>, Vector<double>>(maximize: false, seed: 42);
var searchSpace = new HyperparameterSearchSpace();
searchSpace.AddContinuous("learning_rate", 0.01, 0.3);

Console.WriteLine("Searching hyperparameters through AiModelBuilder.ConfigureHyperparameterOptimizer ...");
try
{
    var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
        .ConfigureModel(new GradientBoostingRegression<double>(
            new GradientBoostingRegressionOptions { NumberOfTrees = 50, MaxDepth = 3, LearningRate = 0.1 }))
        .ConfigureDataLoader(DataLoaders.FromArrays(trainFeatures, trainPrices))
        .ConfigureHyperparameterOptimizer(hpo, searchSpace, nTrials: 4)
        .BuildAsync();

    Console.WriteLine("  Search complete.\n");

    // Evaluate the best model on the held-out test set, through the result object.
    var predictions = result.Predict(ToMatrix(testFeatures));
    double sumSq = 0, sumAbs = 0, mean = testPrices.Average(), totSq = 0;
    for (int i = 0; i < testFeatures.Length; i++)
    {
        double err = predictions[i] - testPrices[i];
        sumSq += err * err;
        sumAbs += Math.Abs(err);
        totSq += Math.Pow(testPrices[i] - mean, 2);
    }
    double rmse = Math.Sqrt(sumSq / testFeatures.Length);
    double mae = sumAbs / testFeatures.Length;
    double r2 = 1 - (sumSq / totSq);

    Console.WriteLine("Best Model Evaluation:");
    Console.WriteLine("--------------------------------------");
    Console.WriteLine($"  R2 Score: {r2:F4}");
    Console.WriteLine($"  MAE:  ${mae:N0}");
    Console.WriteLine($"  RMSE: ${rmse:N0}\n");

    Console.WriteLine("Sample Predictions:");
    for (int i = 0; i < Math.Min(5, testFeatures.Length); i++)
    {
        double errPct = Math.Abs((predictions[i] - testPrices[i]) / testPrices[i]) * 100;
        Console.WriteLine($"  House {i + 1}: Predicted=${predictions[i]:N0}, Actual=${testPrices[i]:N0} (Error: {errPct:F1}%)");
    }
}
catch (Exception ex)
{
    Console.WriteLine($"  Hyperparameter search reported: {ex.Message}");
}

Console.WriteLine("\n=== Sample Complete ===");

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

static (double[][] trainFeatures, double[] trainPrices, double[][] testFeatures, double[] testPrices) GenerateHousePriceData()
{
    var random = new Random(42);
    var allFeatures = new List<double[]>();
    var allPrices = new List<double>();
    for (int i = 0; i < 600; i++)
    {
        double sqft = 1000 + random.NextDouble() * 3000;
        double bedrooms = Math.Floor(1 + random.NextDouble() * 5);
        double bathrooms = Math.Floor(1 + random.NextDouble() * 4);
        double age = random.NextDouble() * 50;
        double locationScore = random.NextDouble() * 10;
        allFeatures.Add(new[] { sqft, bedrooms, bathrooms, age, locationScore });
        double price = 100000 + sqft * 150 + bedrooms * 20000 + bathrooms * 15000
            - age * 1000 + locationScore * 30000 + (random.NextDouble() - 0.5) * 50000;
        allPrices.Add(Math.Max(50000, price));
    }
    var idx = Enumerable.Range(0, 600).OrderBy(_ => random.Next()).ToArray();
    var f = idx.Select(i => allFeatures[i]).ToArray();
    var p = idx.Select(i => allPrices[i]).ToArray();
    return (f.Take(500).ToArray(), p.Take(500).ToArray(), f.Skip(500).ToArray(), p.Skip(500).ToArray());
}
