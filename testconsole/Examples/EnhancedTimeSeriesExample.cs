using System.Globalization;
using AiDotNet.Data.Loaders;
using AiDotNet.DataProcessor;
using AiDotNet.FeatureSelectors;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.Normalizers;
using AiDotNet.Optimizers;
using AiDotNet.OutlierRemoval;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TimeSeries;

namespace AiDotNet.Examples;

/// <summary>
/// Enhanced real-world example demonstrating time series forecasting for energy demand.
/// </summary>
public class EnhancedTimeSeriesExample
{
    public async Task RunExample()
    {
        Console.WriteLine("Enhanced Time Series Example - Energy Demand Forecasting");
        Console.WriteLine("======================================================\n");
        Console.WriteLine("This example demonstrates using time series models to predict energy demand.");
        Console.WriteLine("We'll analyze historical energy consumption data and forecast future demand.\n");

        try
        {
            // 1. Generate synthetic energy consumption data
            Console.WriteLine("Generating synthetic energy consumption data...");
            var (timestamps, demand, temperature, dayType) = GenerateEnergyData(730);  // 2 years of daily data

            Console.WriteLine($"Generated {demand.Length} days of energy consumption data");

            // 2. Explore the data
            Console.WriteLine("\nExploring the data:");

            // Calculate summary statistics
            double minDemand = demand.Min();
            double maxDemand = demand.Max();
            double avgDemand = demand.Average();

            double minTemp = temperature.Min();
            double maxTemp = temperature.Max();
            double avgTemp = temperature.Average();

            Console.WriteLine($"Energy Demand (MWh):");
            Console.WriteLine($"- Minimum: {minDemand:F1}");
            Console.WriteLine($"- Maximum: {maxDemand:F1}");
            Console.WriteLine($"- Average: {avgDemand:F1}");

            Console.WriteLine($"\nTemperature (°C):");
            Console.WriteLine($"- Minimum: {minTemp:F1}");
            Console.WriteLine($"- Maximum: {maxTemp:F1}");
            Console.WriteLine($"- Average: {avgTemp:F1}");

            // Count day types
            int weekdays = dayType.Count(d => d == 0);
            int weekends = dayType.Count(d => d == 1);
            int holidays = dayType.Count(d => d == 2);

            Console.WriteLine($"\nDay Types:");
            Console.WriteLine($"- Weekdays: {weekdays}");
            Console.WriteLine($"- Weekends: {weekends}");
            Console.WriteLine($"- Holidays: {holidays}");

            // 3. Prepare data for modeling
            Console.WriteLine("\nPreparing data for modeling...");

            // Convert to matrix format
            // Features: timestamp, temperature, day type
            var featuresMatrix = new double[demand.Length, 3];
            for (int i = 0; i < demand.Length; i++)
            {
                featuresMatrix[i, 0] = timestamps[i];
                featuresMatrix[i, 1] = temperature[i];
                featuresMatrix[i, 2] = dayType[i];
            }

            // Create feature matrix and target vector
            var features = new Matrix<double>(featuresMatrix);
            var target = new Vector<double>(demand);

            // 4. Configure data preprocessing with normalization
            Console.WriteLine("\nSetting up data preprocessing...");

            // Create normalizer
            var normalizer = new NoNormalizer<double, Matrix<double>, Vector<double>>();

            // Create feature selector and outlier removal
            var featureSelector = new NoFeatureSelector<double, Matrix<double>>();
            var outlierRemoval = new NoOutlierRemoval<double, Matrix<double>, Vector<double>>();

            // Create data preprocessor
            var dataPreprocessorOptions = new DataProcessorOptions
            {
                TestingSplitPercentage = 0.2,
                ValidationSplitPercentage = 0.0, // No validation set needed for this example
                RandomSeed = 42
            };
            var dataPreprocessor = new DefaultDataPreprocessor<double, Matrix<double>, Vector<double>>(
                normalizer, featureSelector, outlierRemoval, dataPreprocessorOptions);

            // 5. Preprocess data - normalize and split into training and test sets
            Console.WriteLine("Normalizing and splitting data...");
            var (normalizedFeatures, normalizedTarget, normInfo) = dataPreprocessor.PreprocessData(features, target);
            var dataSplit = dataPreprocessor.SplitData(normalizedFeatures, normalizedTarget);

            int trainSize = dataSplit.XTrain.Rows;
            int testSize = dataSplit.XTest.Rows;

            Console.WriteLine($"Data split into training ({trainSize} days) and test ({testSize} days) sets");

            // 6. Create model builder
            var modelBuilder = new PredictionModelBuilder<double, Matrix<double>, Vector<double>>();

            // 7. Train multiple time series models
            Console.WriteLine("\nTraining time series models...");

            // Train Prophet-like model
            Console.WriteLine("\n1. Training Prophet-like model...");
            var prophetModel = await TrainProphetModel(dataSplit.XTrain, dataSplit.yTrain);

            // Train ARIMA model
            Console.WriteLine("\n2. Training ARIMA model...");
            var arimaModel = await TrainArimaModel(dataSplit.XTrain, dataSplit.yTrain);

            // Train Exponential Smoothing model
            Console.WriteLine("\n3. Training Exponential Smoothing model...");
            var esModel = await TrainExponentialSmoothingModel(dataSplit.XTrain, dataSplit.yTrain);

            // 8. Evaluate models on test set
            Console.WriteLine("\nEvaluating models on test set:");

            // Evaluate Prophet model
            var prophetMetrics = EvaluateModel(prophetModel, dataSplit.XTest, dataSplit.yTest);
            Console.WriteLine("\nProphet-like model performance:");
            PrintMetrics(prophetMetrics);

            // Evaluate ARIMA model
            var arimaMetrics = EvaluateModel(arimaModel, dataSplit.XTest, dataSplit.yTest);
            Console.WriteLine("\nARIMA model performance:");
            PrintMetrics(arimaMetrics);

            // Evaluate Exponential Smoothing model
            var esMetrics = EvaluateModel(esModel, dataSplit.XTest, dataSplit.yTest);
            Console.WriteLine("\nExponential Smoothing model performance:");
            PrintMetrics(esMetrics);

            // 9. Choose the best model
            var bestMetric = new[] { prophetMetrics.mape, arimaMetrics.mape, esMetrics.mape }.Min();
            var bestModel = bestMetric == prophetMetrics.mape ? prophetModel :
                           bestMetric == arimaMetrics.mape ? arimaModel : esModel;
            var bestModelName = bestMetric == prophetMetrics.mape ? "Prophet-like" :
                               bestMetric == arimaMetrics.mape ? "ARIMA" : "Exponential Smoothing";

            Console.WriteLine($"\nBest model based on MAPE: {bestModelName}");

            // 10. Generate future dates and features
            Console.WriteLine("\nGenerating features for future forecasting...");

            // Generate dates for the next 30 days
            var lastDate = DateTimeOffset.FromUnixTimeSeconds((long)timestamps.Last());
            var futureDates = new double[30];
            var futureTemperature = new double[30];
            var futureDayType = new double[30];

            // Create future feature matrix
            var futureFeatures = new Matrix<double>(30, 3);

            // Fill future features
            for (int i = 0; i < 30; i++)
            {
                // Add one day to last date
                var nextDate = lastDate.AddDays(i + 1);
                futureDates[i] = nextDate.ToUnixTimeSeconds();

                // Generate synthetic temperature (simulate seasonal pattern)
                double baseTemp = avgTemp + 5 * Math.Sin(2 * Math.PI * (nextDate.DayOfYear / 365.0));
                futureTemperature[i] = baseTemp + random.NextDouble() * 4 - 2;  // Add some noise

                // Determine day type
                int dayOfWeek = (int)nextDate.DayOfWeek;
                if (dayOfWeek == 0 || dayOfWeek == 6)
                {
                    futureDayType[i] = 1;  // Weekend
                }
                else
                {
                    futureDayType[i] = 0;  // Weekday
                }

                // Special days (holidays)
                if ((nextDate.Month == 1 && nextDate.Day == 1) ||    // New Year's
                    (nextDate.Month == 12 && nextDate.Day == 25))    // Christmas
                {
                    futureDayType[i] = 2;  // Holiday
                }

                // Set features
                futureFeatures[i, 0] = futureDates[i];
                futureFeatures[i, 1] = futureTemperature[i];
                futureFeatures[i, 2] = futureDayType[i];
            }

            // 11. Normalize future features using the same normalization info
            var (normalizedFutureFeatures, _) = normalizer.NormalizeInput(futureFeatures);

            // 12. Make forecasts with the best model
            Console.WriteLine($"\nForecasting energy demand for the next 30 days using {bestModelName} model...");
            var normalizedForecast = modelBuilder.Predict(normalizedFutureFeatures, bestModel);

            // 13. Denormalize the forecast to get actual values
            var forecast = normalizer.Denormalize(normalizedForecast, normInfo.YParams);

            // 14. Visualize and analyze the forecast
            Console.WriteLine("\nForecast Results:");
            Console.WriteLine("Date                  | Temperature | Day Type  | Forecast (MWh)");
            Console.WriteLine("----------------------|-------------|-----------|---------------");

            for (int i = 0; i < 30; i++)
            {
                var date = DateTimeOffset.FromUnixTimeSeconds((long)futureDates[i]);
                string dateStr = date.ToString("yyyy-MM-dd (ddd)");
                string dayTypeStr = futureDayType[i] == 0 ? "Weekday" :
                                   futureDayType[i] == 1 ? "Weekend" : "Holiday";

                Console.WriteLine($"{dateStr} | {futureTemperature[i],10:F1}°C | {dayTypeStr,-9} | {forecast[i],13:F1}");
            }

            // 15. Identify peak demand days
            Console.WriteLine("\nPeak Demand Days (Top 5):");

            var peakIndices = Enumerable.Range(0, forecast.Length)
                                       .OrderByDescending(i => forecast[i])
                                       .Take(5)
                                       .ToArray();

            for (int i = 0; i < peakIndices.Length; i++)
            {
                int idx = peakIndices[i];
                var date = DateTimeOffset.FromUnixTimeSeconds((long)futureDates[idx]);
                string dateStr = date.ToString("yyyy-MM-dd (ddd)");
                string dayTypeStr = futureDayType[idx] == 0 ? "Weekday" :
                                  futureDayType[idx] == 1 ? "Weekend" : "Holiday";

                Console.WriteLine($"{i + 1}. {dateStr} - {forecast[idx]:F1} MWh (Temperature: {futureTemperature[idx]:F1}°C, {dayTypeStr})");
            }

            // 16. Analyze temperature impact on demand
            Console.WriteLine("\nAnalyzing temperature impact on demand...");

            // Group temperatures into bins
            int[] temperatureBins = { -10, 0, 10, 20, 30, 40 };
            double[] avgDemandByTemp = new double[temperatureBins.Length - 1];
            int[] countsByTemp = new int[temperatureBins.Length - 1];

            for (int i = 0; i < demand.Length; i++)
            {
                for (int j = 0; j < temperatureBins.Length - 1; j++)
                {
                    if (temperature[i] >= temperatureBins[j] && temperature[i] < temperatureBins[j + 1])
                    {
                        avgDemandByTemp[j] += demand[i];
                        countsByTemp[j]++;
                        break;
                    }
                }
            }

            // Calculate average demand by temperature
            for (int j = 0; j < avgDemandByTemp.Length; j++)
            {
                if (countsByTemp[j] > 0)
                {
                    avgDemandByTemp[j] /= countsByTemp[j];
                }
            }

            Console.WriteLine("\nAverage Demand by Temperature Range:");
            for (int j = 0; j < avgDemandByTemp.Length; j++)
            {
                if (countsByTemp[j] > 0)
                {
                    Console.WriteLine($"{temperatureBins[j],3}°C to {temperatureBins[j + 1],3}°C: {avgDemandByTemp[j]:F1} MWh (based on {countsByTemp[j]} days)");
                }
            }

            // 17. Save the best model
            string modelPath = "energy_forecast_model.bin";
            modelBuilder.SaveModel(bestModel, modelPath);
            Console.WriteLine($"\nBest model saved to {modelPath}");

            Console.WriteLine("\nEnhanced time series example completed!");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }
    }

    // Random number generator
    private static readonly Random random = RandomHelper.CreateSeededRandom(123);

    // Helper method to generate synthetic energy data
    private (double[] Timestamps, double[] Demand, double[] Temperature, double[] DayType) GenerateEnergyData(int numDays)
    {
        // Initialize arrays
        double[] timestamps = new double[numDays];
        double[] demand = new double[numDays];
        double[] temperature = new double[numDays];
        double[] dayType = new double[numDays];

        // Set start date (2 years ago)
        DateTimeOffset startDate = DateTimeOffset.Now.AddDays(-numDays);
        startDate = new DateTimeOffset(startDate.Year, startDate.Month, startDate.Day, 0, 0, 0, startDate.Offset);

        for (int i = 0; i < numDays; i++)
        {
            // Current date
            DateTimeOffset currentDate = startDate.AddDays(i);
            timestamps[i] = currentDate.ToUnixTimeSeconds();

            // Day of week (0-6, where 0 is Sunday)
            int dayOfWeek = (int)currentDate.DayOfWeek;

            // Day of year (0-364)
            int dayOfYear = currentDate.DayOfYear - 1;

            // Set day type
            if (dayOfWeek == 0 || dayOfWeek == 6)
            {
                dayType[i] = 1;  // Weekend
            }
            else
            {
                dayType[i] = 0;  // Weekday
            }

            // Special days (holidays)
            if ((currentDate.Month == 1 && currentDate.Day == 1) ||    // New Year's
                (currentDate.Month == 12 && currentDate.Day == 25))    // Christmas
            {
                dayType[i] = 2;  // Holiday
            }

            // Generate temperature with seasonal pattern
            double baseTemp = 15 + 15 * Math.Sin(2 * Math.PI * (dayOfYear / 365.0));
            temperature[i] = baseTemp + random.NextDouble() * 6 - 3;  // Add some noise

            // Base demand
            double baseDemand = 5000;

            // Weekend effect
            double weekendEffect = dayType[i] == 1 ? 0.8 : 1.0;

            // Holiday effect
            double holidayEffect = dayType[i] == 2 ? 0.7 : 1.0;

            // Temperature effect (U-shaped: high demand at low and high temperatures)
            double tempEffect = 1.0 + 0.3 * Math.Pow((temperature[i] - 20) / 20, 2);

            // Yearly seasonal pattern
            double yearlyPattern = 1.0 + 0.1 * Math.Sin(2 * Math.PI * (dayOfYear / 365.0));

            // Weekly pattern
            double weeklyPattern = 1.0 + 0.05 * Math.Sin(2 * Math.PI * (dayOfWeek / 7.0));

            // Trend component (slight increase over time)
            double trend = 1.0 + 0.0001 * i;

            // Random noise
            double noise = 1.0 + 0.05 * (random.NextDouble() * 2 - 1);

            // Combine all effects
            demand[i] = baseDemand * weekendEffect * holidayEffect * tempEffect * yearlyPattern * weeklyPattern * trend * noise;
        }

        return (timestamps, demand, temperature, dayType);
    }

    // Helper method to train a Prophet-like model
    private Task<PredictionModelResult<double, Matrix<double>, Vector<double>>> TrainProphetModel(
        Matrix<double> features, Vector<double> target)
    {
        var modelBuilder = new PredictionModelBuilder<double, Matrix<double>, Vector<double>>();

        // Configure Adam optimizer
        var adamOptions = new AdamOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            LearningRate = 0.01,
            MaxIterations = 1000,
            Tolerance = 1e-6
        };
        var optimizer = new AdamOptimizer<double, Matrix<double>, Vector<double>>(null, adamOptions);

        // Configure Prophet model options
        var prophetOptions = new ProphetOptions<double, Matrix<double>, Vector<double>>
        {
            SeasonalPeriods = new List<int> { 7, 365 },  // Daily and yearly seasonality
            ChangePointPriorScale = 0.05,
            ForecastHorizon = 30
        };

        // Build and return the model
        return modelBuilder
            .ConfigureOptimizer(optimizer)
            .ConfigureModel(new ProphetModel<double, Matrix<double>, Vector<double>>(prophetOptions))
            .ConfigureDataLoader(new InMemoryDataLoader<double, Matrix<double>, Vector<double>>(features, target))
            .BuildAsync();
    }

    // Helper method to train an ARIMA model
    private Task<PredictionModelResult<double, Matrix<double>, Vector<double>>> TrainArimaModel(
        Matrix<double> features, Vector<double> target)
    {
        var modelBuilder = new PredictionModelBuilder<double, Matrix<double>, Vector<double>>();

        // Configure Adam optimizer
        var adamOptions = new AdamOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            LearningRate = 0.01,
            MaxIterations = 1000,
            Tolerance = 1e-6
        };
        var optimizer = new AdamOptimizer<double, Matrix<double>, Vector<double>>(null, adamOptions);

        // Configure ARIMA model options
        var arimaOptions = new ARIMAOptions<double>
        {
            P = 3,  // AR order
            D = 1,  // Differencing
            Q = 2,  // MA order
        };

        // Build and return the model
        return modelBuilder
            .ConfigureOptimizer(optimizer)
            .ConfigureModel(new ARIMAModel<double>(arimaOptions))
            .ConfigureDataLoader(new InMemoryDataLoader<double, Matrix<double>, Vector<double>>(features, target))
            .BuildAsync();
    }

    // Helper method to train an Exponential Smoothing model
    private Task<PredictionModelResult<double, Matrix<double>, Vector<double>>> TrainExponentialSmoothingModel(
        Matrix<double> features, Vector<double> target)
    {
        var modelBuilder = new PredictionModelBuilder<double, Matrix<double>, Vector<double>>();

        // Configure Adam optimizer
        var adamOptions = new AdamOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            LearningRate = 0.01,
            MaxIterations = 1000,
            Tolerance = 1e-6
        };
        var optimizer = new AdamOptimizer<double, Matrix<double>, Vector<double>>(null, adamOptions);

        // Configure Exponential Smoothing model options
        var esOptions = new ExponentialSmoothingOptions<double>
        {
            InitialAlpha = 0.2,  // Level smoothing
            InitialBeta = 0.1,   // Trend smoothing
            InitialGamma = 0.3,  // Seasonal smoothing
            SeasonalPeriod = 7,  // Weekly seasonality
        };

        // Build and return the model
        return modelBuilder
            .ConfigureOptimizer(optimizer)
            .ConfigureModel(new ExponentialSmoothingModel<double>(esOptions))
            .ConfigureDataLoader(new InMemoryDataLoader<double, Matrix<double>, Vector<double>>(features, target))
            .BuildAsync();
    }

    // Helper method to evaluate a model
    private (double rmse, double mape, double mae) EvaluateModel(
        PredictionModelResult<double, Matrix<double>, Vector<double>> model,
        Matrix<double> features,
        Vector<double> targets)
    {
        var modelBuilder = new PredictionModelBuilder<double, Matrix<double>, Vector<double>>();
        var predictions = modelBuilder.Predict(features, model);

        // Calculate RMSE (Root Mean Squared Error)
        double mse = 0;
        for (int i = 0; i < targets.Length; i++)
        {
            double error = targets[i] - predictions[i];
            mse += error * error;
        }
        mse /= targets.Length;
        double rmse = Math.Sqrt(mse);

        // Calculate MAPE (Mean Absolute Percentage Error)
        double mape = 0;
        for (int i = 0; i < targets.Length; i++)
        {
            double pct_error = Math.Abs((targets[i] - predictions[i]) / targets[i]);
            mape += pct_error;
        }
        mape = (mape / targets.Length) * 100;  // Convert to percentage

        // Calculate MAE (Mean Absolute Error)
        double mae = 0;
        for (int i = 0; i < targets.Length; i++)
        {
            mae += Math.Abs(targets[i] - predictions[i]);
        }
        mae /= targets.Length;

        return (rmse, mape, mae);
    }

    // Helper method to print evaluation metrics
    private void PrintMetrics((double rmse, double mape, double mae) metrics)
    {
        Console.WriteLine($"- Root Mean Squared Error (RMSE): {metrics.rmse:F2} MWh");
        Console.WriteLine($"- Mean Absolute Percentage Error (MAPE): {metrics.mape:F2}%");
        Console.WriteLine($"- Mean Absolute Error (MAE): {metrics.mae:F2} MWh");
    }
}
