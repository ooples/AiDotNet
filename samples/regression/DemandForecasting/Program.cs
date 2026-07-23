using AiDotNet.Models.Options;
using AiDotNet.Preprocessing.Scalers;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;

Console.WriteLine("=== AiDotNet Product Demand Forecasting ===");
Console.WriteLine("Time-aware regression with regularization and prediction intervals\n");

// Generate synthetic demand data with time-based patterns
var (trainData, testData, featureNames) = GenerateDemandData();

Console.WriteLine($"Training set: {trainData.Features.Rows} samples (2 years of daily data)");
Console.WriteLine($"Test set: {testData.Features.Rows} samples (6 months of daily data)");
Console.WriteLine($"Features ({featureNames.Length}): {string.Join(", ", featureNames)}\n");

// Display demand statistics
var trainDemandArray = new double[trainData.Targets.Length];
for (int i = 0; i < trainData.Targets.Length; i++)
    trainDemandArray[i] = trainData.Targets[i];

Console.WriteLine("Demand Statistics (Training):");
Console.WriteLine($"  Min: {trainDemandArray.Min():N0} units");
Console.WriteLine($"  Max: {trainDemandArray.Max():N0} units");
Console.WriteLine($"  Mean: {trainDemandArray.Average():N0} units");
Console.WriteLine($"  Std Dev: {CalculateStdDev(trainDemandArray):N0} units\n");

// Standardize features
Console.WriteLine("Preprocessing: Standardizing features...\n");
var scaler = new StandardScaler<double>();
var trainFeaturesScaled = scaler.FitTransform(trainData.Features);
var testFeaturesScaled = scaler.Transform(testData.Features);

// Define regularization models to compare
Console.WriteLine("===========================================================================");
Console.WriteLine("         REGULARIZATION COMPARISON (Ridge vs Lasso vs ElasticNet)          ");
Console.WriteLine("===========================================================================\n");

Console.WriteLine("Regularization helps prevent overfitting by penalizing large coefficients:\n");
Console.WriteLine("  - Ridge (L2): Shrinks all coefficients proportionally");
Console.WriteLine("  - Lasso (L1): Can zero out unimportant features (feature selection)");
Console.WriteLine("  - Elastic Net: Combines L1 and L2 for best of both worlds\n");

// Test different regularization strengths
var alphaValues = new[] { 0.01, 0.1, 1.0, 10.0 };
var results = new List<(string model, double alpha, double r2, double mae, double rmse, int selectedFeatures)>();

Console.WriteLine("Testing different regularization strengths (alpha)...\n");

// Ridge Regression with different alpha values
Console.WriteLine("Ridge Regression:");
foreach (var alpha in alphaValues)
{
    var ridge = new RidgeRegression<double>(new RidgeRegressionOptions<double> { Alpha = alpha });
    ridge.Train(trainFeaturesScaled, trainData.Targets);
    var predictions = ridge.Predict(testFeaturesScaled);
    var metrics = CalculateMetrics(testData.Targets, predictions);

    results.Add(("Ridge", alpha, metrics.r2, metrics.mae, metrics.rmse, featureNames.Length));
    Console.WriteLine($"  alpha={alpha,-6:F2} R2={metrics.r2:F4} MAE={metrics.mae:N0} RMSE={metrics.rmse:N0}");
}
Console.WriteLine();

// Lasso Regression with different alpha values
Console.WriteLine("Lasso Regression:");
foreach (var alpha in alphaValues)
{
    var lasso = new LassoRegression<double>(new LassoRegressionOptions<double>
    {
        Alpha = alpha,
        MaxIterations = 1000,
        Tolerance = 1e-6
    });
    lasso.Train(trainFeaturesScaled, trainData.Targets);
    var predictions = lasso.Predict(testFeaturesScaled);
    var metrics = CalculateMetrics(testData.Targets, predictions);

    int selectedFeatures = lasso.NumberOfSelectedFeatures;
    results.Add(("Lasso", alpha, metrics.r2, metrics.mae, metrics.rmse, selectedFeatures));
    Console.WriteLine($"  alpha={alpha,-6:F2} R2={metrics.r2:F4} MAE={metrics.mae:N0} RMSE={metrics.rmse:N0} Selected={selectedFeatures}/{featureNames.Length}");
}
Console.WriteLine();

// Elastic Net with different alpha and l1_ratio values
Console.WriteLine("Elastic Net Regression (L1 ratio = 0.5):");
foreach (var alpha in alphaValues)
{
    var elasticNet = new ElasticNetRegression<double>(new ElasticNetRegressionOptions<double>
    {
        Alpha = alpha,
        L1Ratio = 0.5,
        MaxIterations = 1000,
        Tolerance = 1e-6
    });
    elasticNet.Train(trainFeaturesScaled, trainData.Targets);
    var predictions = elasticNet.Predict(testFeaturesScaled);
    var metrics = CalculateMetrics(testData.Targets, predictions);

    int selectedFeatures = elasticNet.NumberOfSelectedFeatures;
    results.Add(("ElasticNet", alpha, metrics.r2, metrics.mae, metrics.rmse, selectedFeatures));
    Console.WriteLine($"  alpha={alpha,-6:F2} R2={metrics.r2:F4} MAE={metrics.mae:N0} RMSE={metrics.rmse:N0} Selected={selectedFeatures}/{featureNames.Length}");
}
Console.WriteLine();

// Find best models
var bestRidge = results.Where(r => r.model == "Ridge").OrderByDescending(r => r.r2).First();
var bestLasso = results.Where(r => r.model == "Lasso").OrderByDescending(r => r.r2).First();
var bestElasticNet = results.Where(r => r.model == "ElasticNet").OrderByDescending(r => r.r2).First();

Console.WriteLine("===========================================================================");
Console.WriteLine("                           BEST MODELS SUMMARY                             ");
Console.WriteLine("===========================================================================\n");

Console.WriteLine(new string('-', 80));
Console.WriteLine($"{"Model",-15} {"Alpha",-8} {"R2",-10} {"MAE",-12} {"RMSE",-12} {"Features",-10}");
Console.WriteLine(new string('-', 80));
Console.WriteLine($"{"Ridge",-15} {bestRidge.alpha,-8:F2} {bestRidge.r2,-10:F4} {bestRidge.mae,-12:N0} {bestRidge.rmse,-12:N0} {bestRidge.selectedFeatures,-10}");
Console.WriteLine($"{"Lasso",-15} {bestLasso.alpha,-8:F2} {bestLasso.r2,-10:F4} {bestLasso.mae,-12:N0} {bestLasso.rmse,-12:N0} {bestLasso.selectedFeatures,-10}");
Console.WriteLine($"{"ElasticNet",-15} {bestElasticNet.alpha,-8:F2} {bestElasticNet.r2,-10:F4} {bestElasticNet.mae,-12:N0} {bestElasticNet.rmse,-12:N0} {bestElasticNet.selectedFeatures,-10}");
Console.WriteLine();

// Use best Lasso model to show feature selection
Console.WriteLine("===========================================================================");
Console.WriteLine("                    LASSO FEATURE SELECTION ANALYSIS                       ");
Console.WriteLine("===========================================================================\n");

var bestLassoModel = new LassoRegression<double>(new LassoRegressionOptions<double>
{
    Alpha = bestLasso.alpha,
    MaxIterations = 1000,
    Tolerance = 1e-6
});
bestLassoModel.Train(trainFeaturesScaled, trainData.Targets);

Console.WriteLine($"Features selected by Lasso (alpha={bestLasso.alpha}):");
Console.WriteLine(new string('-', 60));

var coefficients = bestLassoModel.Coefficients;
var featureCoeffs = new List<(string name, double coef)>();
for (int i = 0; i < coefficients.Length && i < featureNames.Length; i++)
{
    featureCoeffs.Add((featureNames[i], coefficients[i]));
}

Console.WriteLine("\nSelected Features (non-zero coefficients):");
int rank = 1;
foreach (var (name, coef) in featureCoeffs.Where(f => Math.Abs(f.coef) > 1e-6).OrderByDescending(f => Math.Abs(f.coef)))
{
    string direction = coef > 0 ? "+" : "-";
    Console.WriteLine($"  {rank,2}. {name,-20} {direction}{Math.Abs(coef):F4}");
    rank++;
}

Console.WriteLine("\nEliminated Features (zero coefficients):");
foreach (var (name, coef) in featureCoeffs.Where(f => Math.Abs(f.coef) < 1e-6))
{
    Console.WriteLine($"      {name,-20} (eliminated)");
}

// Prediction Intervals
Console.WriteLine("\n===========================================================================");
Console.WriteLine("                         PREDICTION INTERVALS                              ");
Console.WriteLine("===========================================================================\n");

Console.WriteLine("Calculating prediction intervals using residual-based approach...\n");

// Use Elastic Net for predictions with intervals
var finalModel = new ElasticNetRegression<double>(new ElasticNetRegressionOptions<double>
{
    Alpha = bestElasticNet.alpha,
    L1Ratio = 0.5,
    MaxIterations = 1000
});
finalModel.Train(trainFeaturesScaled, trainData.Targets);

// Calculate residuals on training data for interval estimation
var trainPredictions = finalModel.Predict(trainFeaturesScaled);
var residuals = new double[trainData.Targets.Length];
for (int i = 0; i < residuals.Length; i++)
{
    residuals[i] = trainData.Targets[i] - trainPredictions[i];
}

// Calculate standard deviation of residuals for prediction intervals
double residualMean = residuals.Average();
double residualStd = Math.Sqrt(residuals.Select(r => Math.Pow(r - residualMean, 2)).Average());

// Z-scores for different confidence levels
double z95 = 1.96;  // 95% confidence
double z80 = 1.28;  // 80% confidence

Console.WriteLine($"Residual Standard Deviation: {residualStd:N0} units\n");

// Make predictions on test set with intervals
var testPredictions = finalModel.Predict(testFeaturesScaled);

Console.WriteLine("Demand Forecast with Prediction Intervals:");
Console.WriteLine(new string('-', 95));
Console.WriteLine($"{"Day",-6} {"Predicted",-12} {"80% Lower",-12} {"80% Upper",-12} {"95% Lower",-12} {"95% Upper",-12} {"Actual",-12} {"Status",-10}");
Console.WriteLine(new string('-', 95));

int correctPredictions80 = 0;
int correctPredictions95 = 0;

for (int i = 0; i < Math.Min(20, testPredictions.Length); i++)
{
    double pred = testPredictions[i];
    double actual = testData.Targets[i];

    double lower80 = pred - z80 * residualStd;
    double upper80 = pred + z80 * residualStd;
    double lower95 = pred - z95 * residualStd;
    double upper95 = pred + z95 * residualStd;

    bool within80 = actual >= lower80 && actual <= upper80;
    bool within95 = actual >= lower95 && actual <= upper95;

    if (within95) correctPredictions95++;
    if (within80) correctPredictions80++;

    string status = within80 ? "In 80%" : within95 ? "In 95%" : "Outside";

    Console.WriteLine($"{i + 1,-6} {pred,-12:N0} {lower80,-12:N0} {upper80,-12:N0} {lower95,-12:N0} {upper95,-12:N0} {actual,-12:N0} {status,-10}");
}

Console.WriteLine(new string('-', 95));
Console.WriteLine($"\nInterval Coverage (first 20 days):");
Console.WriteLine($"  80% interval: {(correctPredictions80 / 20.0) * 100:F1}% coverage");
Console.WriteLine($"  95% interval: {(correctPredictions95 / 20.0) * 100:F1}% coverage");

// Time-based analysis
Console.WriteLine("\n===========================================================================");
Console.WriteLine("                         SEASONALITY ANALYSIS                              ");
Console.WriteLine("===========================================================================\n");

Console.WriteLine("Analyzing demand patterns by time components...\n");

// Monthly aggregation
Console.WriteLine("Average Demand by Month:");
Console.WriteLine(new string('-', 50));
var monthlyDemand = new Dictionary<int, List<double>>();
for (int i = 0; i < trainData.Features.Rows; i++)
{
    int month = (int)trainData.Features[i, 0]; // month feature is first
    if (!monthlyDemand.ContainsKey(month))
        monthlyDemand[month] = new List<double>();
    monthlyDemand[month].Add(trainData.Targets[i]);
}

string[] monthNames = { "", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec" };
foreach (var (month, demands) in monthlyDemand.OrderBy(m => m.Key))
{
    double avg = demands.Average();
    int barLength = (int)(avg / 10);
    string bar = new string('#', Math.Min(barLength, 50));
    Console.WriteLine($"  {monthNames[month],-4} {avg,8:N0} {bar}");
}

// Day of week analysis
Console.WriteLine("\nAverage Demand by Day of Week:");
Console.WriteLine(new string('-', 50));
var weekdayDemand = new Dictionary<int, List<double>>();
for (int i = 0; i < trainData.Features.Rows; i++)
{
    int dayOfWeek = (int)trainData.Features[i, 1]; // day_of_week feature is second
    if (!weekdayDemand.ContainsKey(dayOfWeek))
        weekdayDemand[dayOfWeek] = new List<double>();
    weekdayDemand[dayOfWeek].Add(trainData.Targets[i]);
}

string[] dayNames = { "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun" };
foreach (var (day, demands) in weekdayDemand.OrderBy(d => d.Key))
{
    double avg = demands.Average();
    int barLength = (int)(avg / 8);
    string bar = new string('#', Math.Min(barLength, 50));
    Console.WriteLine($"  {dayNames[day],-4} {avg,8:N0} {bar}");
}

// Final summary
Console.WriteLine("\n===========================================================================");
Console.WriteLine("                              FINAL SUMMARY                                ");
Console.WriteLine("===========================================================================\n");

var finalMetrics = CalculateMetrics(testData.Targets, testPredictions);
Console.WriteLine($"Best Model: Elastic Net (alpha={bestElasticNet.alpha}, L1 ratio=0.5)");
Console.WriteLine($"  - R2 Score: {finalMetrics.r2:F4} ({finalMetrics.r2 * 100:F1}% variance explained)");
Console.WriteLine($"  - MAE: {finalMetrics.mae:N0} units (average error)");
Console.WriteLine($"  - RMSE: {finalMetrics.rmse:N0} units");
Console.WriteLine($"  - Features used: {bestElasticNet.selectedFeatures}/{featureNames.Length}");
Console.WriteLine($"\nKey Insights:");
Console.WriteLine($"  - Regularization helps prevent overfitting to noise in time series data");
Console.WriteLine($"  - Lasso eliminates irrelevant features, improving interpretability");
Console.WriteLine($"  - Elastic Net balances feature selection with coefficient shrinkage");
Console.WriteLine($"  - Prediction intervals quantify forecast uncertainty");

Console.WriteLine("\n=== Sample Complete ===");

// Helper functions

static ((Matrix<double> Features, Vector<double> Targets) trainData,
        (Matrix<double> Features, Vector<double> Targets) testData,
        string[] featureNames)
    GenerateDemandData()
{
    var random = new Random(42);
    var allFeatures = new List<double[]>();
    var allTargets = new List<double>();

    // Generate 2.5 years of daily data
    var startDate = new DateTime(2022, 1, 1);
    var endDate = new DateTime(2024, 6, 30);

    for (var date = startDate; date <= endDate; date = date.AddDays(1))
    {
        int month = date.Month;
        int dayOfWeek = (int)date.DayOfWeek == 0 ? 6 : (int)date.DayOfWeek - 1; // 0=Mon, 6=Sun
        int dayOfMonth = date.Day;
        int quarter = (date.Month - 1) / 3 + 1;
        int weekOfYear = (date.DayOfYear - 1) / 7 + 1;
        int isWeekend = dayOfWeek >= 5 ? 1 : 0;
        int year = date.Year - 2022; // 0, 1, 2

        // Cyclical encoding for month and day of week
        double monthSin = Math.Sin(2 * Math.PI * month / 12);
        double monthCos = Math.Cos(2 * Math.PI * month / 12);
        double dowSin = Math.Sin(2 * Math.PI * dayOfWeek / 7);
        double dowCos = Math.Cos(2 * Math.PI * dayOfWeek / 7);

        // Lag features (simulated - in practice these would be actual lagged values)
        double lag1 = 250 + random.NextDouble() * 50; // Previous day demand
        double lag7 = 240 + random.NextDouble() * 60; // Same day last week
        double lag30 = 245 + random.NextDouble() * 55; // Same day last month

        // Rolling averages (simulated)
        double ma7 = 245 + random.NextDouble() * 30;  // 7-day moving average
        double ma30 = 248 + random.NextDouble() * 20; // 30-day moving average

        // External factors
        double temperature = 50 + 30 * Math.Sin(2 * Math.PI * (month - 3) / 12) + random.NextDouble() * 10;
        double isHoliday = (month == 12 && dayOfMonth >= 20) || (month == 1 && dayOfMonth <= 5) ? 1 : 0;
        double promotion = random.NextDouble() < 0.1 ? 1 : 0; // 10% chance of promotion

        allFeatures.Add(new double[]
        {
            month, dayOfWeek, dayOfMonth, quarter, weekOfYear, isWeekend, year,
            monthSin, monthCos, dowSin, dowCos,
            lag1, lag7, lag30, ma7, ma30,
            temperature, isHoliday, promotion
        });

        // Generate demand with realistic patterns
        double baseDemand = 250;

        // Trend component (slight growth over time)
        double trend = year * 10;

        // Seasonal component (monthly)
        double seasonality = 30 * Math.Sin(2 * Math.PI * (month - 3) / 12); // Peak in summer

        // Weekly component
        double weeklyEffect = 0;
        if (dayOfWeek == 5) weeklyEffect = 40; // Saturday boost
        else if (dayOfWeek == 6) weeklyEffect = 30; // Sunday boost
        else if (dayOfWeek == 0) weeklyEffect = -10; // Monday dip

        // Holiday effect
        double holidayEffect = isHoliday * 80;

        // Promotion effect
        double promotionEffect = promotion * 60;

        // Temperature effect (moderate temperatures are best)
        double tempEffect = -0.3 * Math.Pow(temperature - 70, 2) / 100;

        // Random noise
        double noise = (random.NextDouble() - 0.5) * 40;

        double demand = baseDemand + trend + seasonality + weeklyEffect + holidayEffect + promotionEffect + tempEffect + noise;
        demand = Math.Max(100, demand); // Minimum 100 units

        allTargets.Add(demand);
    }

    // Split: first 2 years for training, last 6 months for testing
    int totalDays = allFeatures.Count;
    int trainDays = (int)(totalDays * 0.8); // ~2 years

    var trainFeatures = new Matrix<double>(trainDays, 19);
    var trainTargets = new Vector<double>(trainDays);
    for (int i = 0; i < trainDays; i++)
    {
        for (int j = 0; j < 19; j++)
        {
            trainFeatures[i, j] = allFeatures[i][j];
        }
        trainTargets[i] = allTargets[i];
    }

    var testFeatures = new Matrix<double>(totalDays - trainDays, 19);
    var testTargets = new Vector<double>(totalDays - trainDays);
    for (int i = trainDays; i < totalDays; i++)
    {
        for (int j = 0; j < 19; j++)
        {
            testFeatures[i - trainDays, j] = allFeatures[i][j];
        }
        testTargets[i - trainDays] = allTargets[i];
    }

    var featureNames = new[]
    {
        "month", "day_of_week", "day_of_month", "quarter", "week_of_year",
        "is_weekend", "year", "month_sin", "month_cos", "dow_sin", "dow_cos",
        "lag_1d", "lag_7d", "lag_30d", "ma_7d", "ma_30d",
        "temperature", "is_holiday", "is_promotion"
    };

    return ((trainFeatures, trainTargets), (testFeatures, testTargets), featureNames);
}

static (double r2, double mae, double rmse) CalculateMetrics(Vector<double> actual, Vector<double> predicted)
{
    double sumSquaredError = 0;
    double sumAbsoluteError = 0;
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
        sumSquaredTotal += Math.Pow(actual[i] - mean, 2);
    }

    double r2 = 1 - (sumSquaredError / sumSquaredTotal);
    double mae = sumAbsoluteError / actual.Length;
    double rmse = Math.Sqrt(sumSquaredError / actual.Length);

    return (r2, mae, rmse);
}

static double CalculateStdDev(double[] values)
{
    double mean = values.Average();
    double sumSquares = values.Sum(v => Math.Pow(v - mean, 2));
    return Math.Sqrt(sumSquares / values.Length);
}
