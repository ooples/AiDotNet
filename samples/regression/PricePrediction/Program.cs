using AiDotNet;
using AiDotNet.Enums;
using AiDotNet.HyperparameterOptimization;
using AiDotNet.Models.Options;
using AiDotNet.Preprocessing.Scalers;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;

Console.WriteLine("=== AiDotNet Advanced House Price Prediction ===");
Console.WriteLine("Comparing multiple regression models with feature engineering and Bayesian optimization\n");

// Generate synthetic house price data with complex relationships
var (trainFeatures, trainPrices, testFeatures, testPrices, featureNames) = GenerateAdvancedHousePriceData();

Console.WriteLine($"Training set: {trainFeatures.Rows} samples");
Console.WriteLine($"Test set: {testFeatures.Rows} samples");
Console.WriteLine($"Features ({featureNames.Length}): {string.Join(", ", featureNames)}\n");

// Display price statistics
var trainPricesArray = new double[trainPrices.Length];
for (int i = 0; i < trainPrices.Length; i++)
    trainPricesArray[i] = trainPrices[i];

Console.WriteLine("Price Statistics:");
Console.WriteLine($"  Min: ${trainPricesArray.Min():N0}");
Console.WriteLine($"  Max: ${trainPricesArray.Max():N0}");
Console.WriteLine($"  Mean: ${trainPricesArray.Average():N0}");
Console.WriteLine($"  Median: ${GetMedian(trainPricesArray):N0}\n");

// Standardize features for better model performance
Console.WriteLine("Preprocessing: Standardizing features...\n");
var scaler = new StandardScaler<double>();
var trainFeaturesScaled = scaler.FitTransform(trainFeatures);
var testFeaturesScaled = scaler.Transform(testFeatures);

// Define regressors to compare
var regressors = new Dictionary<string, Func<IRegressor>>
{
    ["Ridge Regression"] = () => new RidgeRegressionWrapper(new RidgeRegressionOptions<double> { Alpha = 1.0 }),
    ["Lasso Regression"] = () => new LassoRegressionWrapper(new LassoRegressionOptions<double> { Alpha = 0.1, MaxIterations = 1000 }),
    ["Elastic Net"] = () => new ElasticNetRegressionWrapper(new ElasticNetRegressionOptions<double> { Alpha = 0.5, L1Ratio = 0.5, MaxIterations = 1000 }),
    ["Gradient Boosting"] = () => new GradientBoostingWrapper(new GradientBoostingRegressionOptions
    {
        NumberOfTrees = 100,
        MaxDepth = 5,
        LearningRate = 0.1,
        SubsampleRatio = 0.8
    })
};

Console.WriteLine("===========================================================================");
Console.WriteLine("              MODEL COMPARISON (Standard Hyperparameters)                  ");
Console.WriteLine("===========================================================================\n");

var results = new Dictionary<string, (double r2, double mae, double rmse, double mape)>();

foreach (var (name, createRegressor) in regressors)
{
    Console.WriteLine($"Training {name}...");

    try
    {
        var regressor = createRegressor();
        regressor.Train(trainFeaturesScaled, trainPrices);

        // Evaluate on test set
        var predictions = regressor.Predict(testFeaturesScaled);
        var metrics = CalculateMetrics(testPrices, predictions);

        results[name] = metrics;

        Console.WriteLine($"  R2: {metrics.r2:F4}");
        Console.WriteLine($"  MAE: ${metrics.mae:N0}");
        Console.WriteLine($"  RMSE: ${metrics.rmse:N0}");
        Console.WriteLine($"  MAPE: {metrics.mape:F2}%\n");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"  Error: {ex.Message}\n");
    }
}

// Summary table
Console.WriteLine("===========================================================================");
Console.WriteLine("                           RESULTS SUMMARY                                 ");
Console.WriteLine("===========================================================================\n");

Console.WriteLine(new string('-', 75));
Console.WriteLine($"{"Model",-22} {"R2",-12} {"MAE",-15} {"RMSE",-15} {"MAPE",-10}");
Console.WriteLine(new string('-', 75));

foreach (var (name, metrics) in results.OrderByDescending(r => r.Value.r2))
{
    Console.WriteLine($"{name,-22} {metrics.r2:F4,-12} ${metrics.mae:N0,-14} ${metrics.rmse:N0,-14} {metrics.mape:F2}%");
}
Console.WriteLine();

// Bayesian Hyperparameter Optimization for Gradient Boosting
Console.WriteLine("===========================================================================");
Console.WriteLine("         BAYESIAN HYPERPARAMETER OPTIMIZATION (Gradient Boosting)          ");
Console.WriteLine("===========================================================================\n");

Console.WriteLine("Optimizing hyperparameters using Bayesian optimization...");
Console.WriteLine("Search space:");
Console.WriteLine("  - Number of trees: [50, 200]");
Console.WriteLine("  - Max depth: [3, 10]");
Console.WriteLine("  - Learning rate: [0.01, 0.3]");
Console.WriteLine("  - Subsample ratio: [0.5, 1.0]");
Console.WriteLine("  - Trials: 15\n");

var bestParams = new Dictionary<string, object>
{
    ["n_trees"] = 100,
    ["max_depth"] = 5,
    ["learning_rate"] = 0.1,
    ["subsample"] = 0.8
};
var bestScore = double.MinValue;

// Bayesian optimization simulation (using the pattern from the codebase)
var bayesianOptimizer = new BayesianOptimizer<double, Matrix<double>, Vector<double>>(
    maximize: true, // Maximize R2
    acquisitionFunction: AcquisitionFunctionType.ExpectedImprovement,
    nInitialPoints: 5,
    explorationWeight: 2.0,
    seed: 42);

// Define search space
var searchSpace = new HyperparameterSearchSpace()
    .AddInteger("n_trees", 50, 200)
    .AddInteger("max_depth", 3, 10)
    .AddContinuous("learning_rate", 0.01, 0.3)
    .AddContinuous("subsample", 0.5, 1.0);

// Objective function
Func<Dictionary<string, object>, double> objectiveFunction = (parameters) =>
{
    try
    {
        var nTrees = Convert.ToInt32(parameters["n_trees"]);
        var maxDepth = Convert.ToInt32(parameters["max_depth"]);
        var learningRate = Convert.ToDouble(parameters["learning_rate"]);
        var subsample = Convert.ToDouble(parameters["subsample"]);

        var gbr = new GradientBoostingWrapper(new GradientBoostingRegressionOptions
        {
            NumberOfTrees = nTrees,
            MaxDepth = maxDepth,
            LearningRate = learningRate,
            SubsampleRatio = subsample
        });

        // Use cross-validation score
        var foldScores = new List<double>();
        int foldSize = trainFeaturesScaled.Rows / 5;

        for (int fold = 0; fold < 5; fold++)
        {
            var foldValidationStart = fold * foldSize;
            var foldValidationEnd = Math.Min((fold + 1) * foldSize, trainFeaturesScaled.Rows);

            // Create train/validation split
            var foldTrainIndices = Enumerable.Range(0, trainFeaturesScaled.Rows)
                .Where(i => i < foldValidationStart || i >= foldValidationEnd).ToArray();
            var foldValIndices = Enumerable.Range(foldValidationStart, foldValidationEnd - foldValidationStart).ToArray();

            var xFoldTrain = GetRows(trainFeaturesScaled, foldTrainIndices);
            var yFoldTrain = GetElements(trainPrices, foldTrainIndices);
            var xFoldVal = GetRows(trainFeaturesScaled, foldValIndices);
            var yFoldVal = GetElements(trainPrices, foldValIndices);

            var foldGbr = new GradientBoostingWrapper(new GradientBoostingRegressionOptions
            {
                NumberOfTrees = nTrees,
                MaxDepth = maxDepth,
                LearningRate = learningRate,
                SubsampleRatio = subsample
            });

            foldGbr.Train(xFoldTrain, yFoldTrain);
            var foldPredictions = foldGbr.Predict(xFoldVal);
            var (r2, _, _, _) = CalculateMetrics(yFoldVal, foldPredictions);
            foldScores.Add(r2);
        }

        return foldScores.Average();
    }
    catch
    {
        return -1.0; // Return bad score on failure
    }
};

// Run optimization
var optimizationResult = bayesianOptimizer.Optimize(objectiveFunction, searchSpace, nTrials: 15);

Console.WriteLine("\nOptimization Results:");
Console.WriteLine(new string('-', 70));

// Display top trials
var topTrials = optimizationResult.Trials
    .Where(t => t.Status == TrialStatus.Complete)
    .OrderByDescending(t => t.ObjectiveValue)
    .Take(5);

Console.WriteLine("\nTop 5 Trials:");
int trialNum = 1;
foreach (var trial in topTrials)
{
    Console.WriteLine($"  Trial {trialNum++}: R2={trial.ObjectiveValue:F4}");
    Console.WriteLine($"    n_trees={trial.Parameters["n_trees"]}, max_depth={trial.Parameters["max_depth"]}, " +
                      $"lr={Convert.ToDouble(trial.Parameters["learning_rate"]):F3}, subsample={Convert.ToDouble(trial.Parameters["subsample"]):F2}");
}

// Get best parameters
if (optimizationResult.BestParameters != null)
{
    bestParams = optimizationResult.BestParameters;
    bestScore = optimizationResult.BestObjectiveValue ?? 0;

    Console.WriteLine($"\nBest hyperparameters found:");
    Console.WriteLine($"  Number of trees: {bestParams["n_trees"]}");
    Console.WriteLine($"  Max depth: {bestParams["max_depth"]}");
    Console.WriteLine($"  Learning rate: {Convert.ToDouble(bestParams["learning_rate"]):F4}");
    Console.WriteLine($"  Subsample ratio: {Convert.ToDouble(bestParams["subsample"]):F3}");
    Console.WriteLine($"  Cross-validation R2: {bestScore:F4}");
}

// Train final model with best parameters
Console.WriteLine("\n===========================================================================");
Console.WriteLine("                    FINAL MODEL WITH OPTIMIZED HYPERPARAMETERS             ");
Console.WriteLine("===========================================================================\n");

var finalModel = new GradientBoostingWrapper(new GradientBoostingRegressionOptions
{
    NumberOfTrees = Convert.ToInt32(bestParams["n_trees"]),
    MaxDepth = Convert.ToInt32(bestParams["max_depth"]),
    LearningRate = Convert.ToDouble(bestParams["learning_rate"]),
    SubsampleRatio = Convert.ToDouble(bestParams["subsample"])
});

finalModel.Train(trainFeaturesScaled, trainPrices);
var finalPredictions = finalModel.Predict(testFeaturesScaled);
var finalMetrics = CalculateMetrics(testPrices, finalPredictions);

Console.WriteLine("Optimized Gradient Boosting Performance:");
Console.WriteLine($"  R2 Score: {finalMetrics.r2:F4}");
Console.WriteLine($"  MAE: ${finalMetrics.mae:N0}");
Console.WriteLine($"  RMSE: ${finalMetrics.rmse:N0}");
Console.WriteLine($"  MAPE: {finalMetrics.mape:F2}%");

// Compare with default parameters
if (results.ContainsKey("Gradient Boosting"))
{
    var defaultMetrics = results["Gradient Boosting"];
    double r2Improvement = (finalMetrics.r2 - defaultMetrics.r2) * 100;
    double maeReduction = (defaultMetrics.mae - finalMetrics.mae) / defaultMetrics.mae * 100;

    Console.WriteLine($"\nImprovement over default hyperparameters:");
    Console.WriteLine($"  R2: {(r2Improvement >= 0 ? "+" : "")}{r2Improvement:F2}%");
    Console.WriteLine($"  MAE reduction: {(maeReduction >= 0 ? "+" : "")}{maeReduction:F2}%");
}

// Sample predictions
Console.WriteLine("\nSample Predictions:");
Console.WriteLine(new string('-', 70));
Console.WriteLine($"{"#",-4} {"Predicted",-15} {"Actual",-15} {"Error %",-12} {"Status",-10}");
Console.WriteLine(new string('-', 70));

for (int i = 0; i < Math.Min(10, finalPredictions.Length); i++)
{
    double predicted = finalPredictions[i];
    double actual = testPrices[i];
    double errorPct = Math.Abs((predicted - actual) / actual) * 100;
    string status = errorPct < 10 ? "Good" : errorPct < 20 ? "Fair" : "Poor";

    Console.WriteLine($"{i + 1,-4} ${predicted:N0,-14} ${actual:N0,-14} {errorPct:F1}%{"",-7} {status,-10}");
}

// Feature importance (approximation based on correlation)
Console.WriteLine("\n===========================================================================");
Console.WriteLine("                           FEATURE ANALYSIS                                ");
Console.WriteLine("===========================================================================\n");

Console.WriteLine("Feature Importance (based on contribution to predictions):");
Console.WriteLine(new string('-', 50));

var featureImportance = CalculateFeatureImportance(trainFeaturesScaled, trainPrices, featureNames);
int rank = 1;
foreach (var (name, importance) in featureImportance.OrderByDescending(f => f.Value))
{
    int barLength = (int)(importance * 40);
    string bar = new string('#', barLength);
    Console.WriteLine($"  {rank,2}. {name,-20} {importance:F4} {bar}");
    rank++;
}

Console.WriteLine("\n=== Sample Complete ===");

// Helper functions

static (Matrix<double> trainFeatures, Vector<double> trainPrices, Matrix<double> testFeatures, Vector<double> testPrices, string[] featureNames)
    GenerateAdvancedHousePriceData()
{
    var random = new Random(42);
    var allFeatures = new List<double[]>();
    var allPrices = new List<double>();

    for (int i = 0; i < 1000; i++)
    {
        // Base features
        double sqft = 800 + random.NextDouble() * 4200; // 800-5000 sqft
        double bedrooms = Math.Floor(1 + random.NextDouble() * 6); // 1-7 bedrooms
        double bathrooms = Math.Floor(1 + random.NextDouble() * 5); // 1-6 bathrooms
        double age = random.NextDouble() * 100; // 0-100 years
        double lotSize = 0.1 + random.NextDouble() * 2.4; // 0.1-2.5 acres
        double garageSpaces = Math.Floor(random.NextDouble() * 4); // 0-4 spaces
        double stories = Math.Floor(1 + random.NextDouble() * 3); // 1-4 stories
        double schoolRating = 1 + random.NextDouble() * 9; // 1-10 rating
        double crimeRate = random.NextDouble() * 100; // 0-100 (lower is better)
        double distanceToCity = 0.5 + random.NextDouble() * 29.5; // 0.5-30 miles

        // Feature interactions (polynomial features)
        double sqftPerBedroom = sqft / bedrooms;
        double bathroomRatio = bathrooms / bedrooms;
        double sqftTimesSchool = sqft * schoolRating / 1000; // Scale down

        // Derived features
        double ageDecade = Math.Floor(age / 10) * 10;
        double pricePerSqft = 100 + (10 - schoolRating) * 5 + crimeRate * 0.3;

        allFeatures.Add(new double[]
        {
            sqft,
            bedrooms,
            bathrooms,
            age,
            lotSize,
            garageSpaces,
            stories,
            schoolRating,
            crimeRate,
            distanceToCity,
            sqftPerBedroom,
            bathroomRatio,
            sqftTimesSchool
        });

        // Complex price formula with interactions and non-linearities
        double basePrice = 50000;
        double price = basePrice
            + sqft * (150 + schoolRating * 10) // sqft value depends on school rating
            + bedrooms * 15000
            + bathrooms * 18000
            + lotSize * 40000
            + garageSpaces * 12000
            + stories * 8000
            - age * 500 // Depreciation
            + Math.Pow(Math.Max(0, schoolRating - 5), 2) * 15000 // Premium for good schools
            - crimeRate * 800
            - Math.Pow(distanceToCity, 1.5) * 300 // Distance penalty (non-linear)
            + sqftPerBedroom * 3 // Value spacious rooms
            + (random.NextDouble() - 0.5) * 40000; // Noise

        allPrices.Add(Math.Max(75000, price));
    }

    // Shuffle and split (80/20)
    var indices = Enumerable.Range(0, 1000).OrderBy(_ => random.Next()).ToArray();
    int splitIndex = 800;

    var trainIndices = indices.Take(splitIndex).ToArray();
    var testIndices = indices.Skip(splitIndex).ToArray();

    var trainFeatures = new Matrix<double>(trainIndices.Length, 13);
    var trainPrices = new Vector<double>(trainIndices.Length);
    for (int i = 0; i < trainIndices.Length; i++)
    {
        int idx = trainIndices[i];
        for (int j = 0; j < 13; j++)
        {
            trainFeatures[i, j] = allFeatures[idx][j];
        }
        trainPrices[i] = allPrices[idx];
    }

    var testFeatures = new Matrix<double>(testIndices.Length, 13);
    var testPrices = new Vector<double>(testIndices.Length);
    for (int i = 0; i < testIndices.Length; i++)
    {
        int idx = testIndices[i];
        for (int j = 0; j < 13; j++)
        {
            testFeatures[i, j] = allFeatures[idx][j];
        }
        testPrices[i] = allPrices[idx];
    }

    var featureNames = new[]
    {
        "sqft", "bedrooms", "bathrooms", "age", "lot_size",
        "garage", "stories", "school_rating", "crime_rate",
        "distance_city", "sqft_per_bed", "bath_ratio", "sqft_x_school"
    };

    return (trainFeatures, trainPrices, testFeatures, testPrices, featureNames);
}

static (double r2, double mae, double rmse, double mape) CalculateMetrics(Vector<double> actual, Vector<double> predicted)
{
    double sumSquaredError = 0;
    double sumAbsoluteError = 0;
    double sumAbsolutePercentageError = 0;
    double mean = 0;

    for (int i = 0; i < actual.Length; i++)
        mean += actual[i];
    mean /= actual.Length;

    double sumSquaredTotal = 0;

    for (int i = 0; i < actual.Length; i++)
    {
        double error = predicted[i] - actual[i];
        sumSquaredError += error * error;
        sumAbsoluteError += Math.Abs(error);
        sumAbsolutePercentageError += Math.Abs(error / actual[i]) * 100;
        sumSquaredTotal += Math.Pow(actual[i] - mean, 2);
    }

    double r2 = 1 - (sumSquaredError / sumSquaredTotal);
    double mae = sumAbsoluteError / actual.Length;
    double rmse = Math.Sqrt(sumSquaredError / actual.Length);
    double mape = sumAbsolutePercentageError / actual.Length;

    return (r2, mae, rmse, mape);
}

static double GetMedian(double[] values)
{
    var sorted = values.OrderBy(v => v).ToArray();
    int mid = sorted.Length / 2;
    return sorted.Length % 2 == 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

static Matrix<double> GetRows(Matrix<double> matrix, int[] indices)
{
    var result = new Matrix<double>(indices.Length, matrix.Columns);
    for (int i = 0; i < indices.Length; i++)
    {
        for (int j = 0; j < matrix.Columns; j++)
        {
            result[i, j] = matrix[indices[i], j];
        }
    }
    return result;
}

static Vector<double> GetElements(Vector<double> vector, int[] indices)
{
    var result = new Vector<double>(indices.Length);
    for (int i = 0; i < indices.Length; i++)
    {
        result[i] = vector[indices[i]];
    }
    return result;
}

static Dictionary<string, double> CalculateFeatureImportance(Matrix<double> features, Vector<double> targets, string[] featureNames)
{
    var importance = new Dictionary<string, double>();

    // Calculate absolute correlation between each feature and target
    double targetMean = 0;
    for (int i = 0; i < targets.Length; i++)
        targetMean += targets[i];
    targetMean /= targets.Length;

    double targetStd = 0;
    for (int i = 0; i < targets.Length; i++)
        targetStd += Math.Pow(targets[i] - targetMean, 2);
    targetStd = Math.Sqrt(targetStd / targets.Length);

    for (int j = 0; j < features.Columns; j++)
    {
        double featureMean = 0;
        for (int i = 0; i < features.Rows; i++)
            featureMean += features[i, j];
        featureMean /= features.Rows;

        double featureStd = 0;
        for (int i = 0; i < features.Rows; i++)
            featureStd += Math.Pow(features[i, j] - featureMean, 2);
        featureStd = Math.Sqrt(featureStd / features.Rows);

        if (featureStd > 0 && targetStd > 0)
        {
            double correlation = 0;
            for (int i = 0; i < features.Rows; i++)
            {
                correlation += (features[i, j] - featureMean) * (targets[i] - targetMean);
            }
            correlation /= features.Rows * featureStd * targetStd;
            importance[featureNames[j]] = Math.Abs(correlation);
        }
        else
        {
            importance[featureNames[j]] = 0;
        }
    }

    return importance;
}

// Wrapper interfaces and classes for regression models
interface IRegressor
{
    void Train(Matrix<double> x, Vector<double> y);
    Vector<double> Predict(Matrix<double> x);
}

class RidgeRegressionWrapper : IRegressor
{
    private readonly RidgeRegression<double> _model;

    public RidgeRegressionWrapper(RidgeRegressionOptions<double> options)
    {
        _model = new RidgeRegression<double>(options);
    }

    public void Train(Matrix<double> x, Vector<double> y) => _model.Train(x, y);
    public Vector<double> Predict(Matrix<double> x) => _model.Predict(x);
}

class LassoRegressionWrapper : IRegressor
{
    private readonly LassoRegression<double> _model;

    public LassoRegressionWrapper(LassoRegressionOptions<double> options)
    {
        _model = new LassoRegression<double>(options);
    }

    public void Train(Matrix<double> x, Vector<double> y) => _model.Train(x, y);
    public Vector<double> Predict(Matrix<double> x) => _model.Predict(x);
}

class ElasticNetRegressionWrapper : IRegressor
{
    private readonly ElasticNetRegression<double> _model;

    public ElasticNetRegressionWrapper(ElasticNetRegressionOptions<double> options)
    {
        _model = new ElasticNetRegression<double>(options);
    }

    public void Train(Matrix<double> x, Vector<double> y) => _model.Train(x, y);
    public Vector<double> Predict(Matrix<double> x) => _model.Predict(x);
}

class GradientBoostingWrapper : IRegressor
{
    private readonly GradientBoostingRegression<double> _model;

    public GradientBoostingWrapper(GradientBoostingRegressionOptions options)
    {
        _model = new GradientBoostingRegression<double>(options);
    }

    public void Train(Matrix<double> x, Vector<double> y)
    {
        _model.TrainAsync(x, y).GetAwaiter().GetResult();
    }

    public Vector<double> Predict(Matrix<double> x)
    {
        return _model.PredictAsync(x).GetAwaiter().GetResult();
    }
}
