using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.DataProcessor;
using AiDotNet.Enums;
using AiDotNet.FeatureSelectors;
using AiDotNet.FitnessCalculators;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Normalizers;
using AiDotNet.Optimizers;
using AiDotNet.AnomalyDetection;
using AiDotNet.Regression;
using AiDotNet.Regularization;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

public class EnhancedRegressionExample
{
    public async Task RunExample()
    {
        Console.WriteLine("Enhanced Regression Example - Real Estate Analysis");
        Console.WriteLine("================================================\n");
        Console.WriteLine("This example demonstrates a more sophisticated real estate analysis model");
        Console.WriteLine("that handles feature engineering, data preprocessing, and model cross-validation.\n");

        try
        {
            // 1. Load synthetic real estate data
            var (features, prices, featureNames, zipCodes) = LoadRealEstateData();

            Console.WriteLine("Real estate dataset loaded with the following features:");
            for (int i = 0; i < featureNames.Length; i++)
            {
                Console.WriteLine($"- {featureNames[i]}");
            }

            Console.WriteLine($"\nTotal properties: {features.Rows}");
            Console.WriteLine($"Number of features: {features.Columns}");

            // 2. Feature engineering: Add polynomial features for square footage
            int sqFtFeatureIndex = Array.IndexOf(featureNames, "SquareFeet");
            if (sqFtFeatureIndex >= 0)
            {
                // Add squared feature for training set
                var enhancedFeatures = new Matrix<double>(features.Rows, features.Columns + 1);
                for (int i = 0; i < features.Rows; i++)
                {
                    for (int j = 0; j < features.Columns; j++)
                    {
                        enhancedFeatures[i, j] = features[i, j];
                    }
                    // Add squared term for square footage
                    enhancedFeatures[i, features.Columns] =
                        features[i, sqFtFeatureIndex] * features[i, sqFtFeatureIndex];
                }

                // Update feature matrix and feature names
                features = enhancedFeatures;

                var enhancedFeatureNames = new string[featureNames.Length + 1];
                Array.Copy(featureNames, enhancedFeatureNames, featureNames.Length);
                enhancedFeatureNames[featureNames.Length] = "SquareFeet_Squared";
                featureNames = enhancedFeatureNames;

                Console.WriteLine("\nFeature engineering: Added polynomial feature for square footage");
                Console.WriteLine($"New feature count: {featureNames.Length}");
            }

            // 3. Configure data preprocessing options
            var normalizer = new NoNormalizer<double, Matrix<double>, Vector<double>>();
            var featureSelector = new NoFeatureSelector<double, Matrix<double>>();
            var outlierRemoval = new NoOutlierRemoval<double, Matrix<double>, Vector<double>>();
            var dataPreprocessor = new DefaultDataPreprocessor<double, Matrix<double>, Vector<double>>(
                normalizer, featureSelector, outlierRemoval);

            // 4. Train multiple types of regression models
            Console.WriteLine("\nTraining multiple regression models...");

            // 5. Create a model builder
            var modelBuilder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();

            // Linear regression model
            Console.WriteLine("\n1. Training Multiple Linear Regression model...");
            var linearModel = await modelBuilder
                .ConfigureDataPreprocessor(dataPreprocessor)
                .ConfigureOptimizer(new AdamOptimizer<double, Matrix<double>, Vector<double>>(null, new AdamOptimizerOptions<double, Matrix<double>, Vector<double>>
                {
                    InitialLearningRate = 0.01,
                    MaxIterations = 2000,
                    Tolerance = 1e-6,
                    UseAdaptiveLearningRate = true
                }))
                .ConfigureModel(new MultipleRegression<double>(new RegressionOptions<double>
                {
                    UseIntercept = true
                }))
                .ConfigureFitnessCalculator(new RSquaredFitnessCalculator<double, Matrix<double>, Vector<double>>())
                .ConfigureDataLoader(new InMemoryDataLoader<double, Matrix<double>, Vector<double>>(features, prices))
                .BuildAsync();

            // Ridge regression model (with L2 regularization)
            Console.WriteLine("\n2. Training Ridge Regression model (with regularization)...");
            double alpha = 1.0;
            var ridgeModel = await modelBuilder
                .ConfigureDataPreprocessor(dataPreprocessor)
                .ConfigureRegularization(new L2Regularization<double, Matrix<double>, Vector<double>>(new RegularizationOptions
                {
                    Type = RegularizationType.L2,
                    Strength = alpha
                }))
                .ConfigureOptimizer(new AdamOptimizer<double, Matrix<double>, Vector<double>>(null, new AdamOptimizerOptions<double, Matrix<double>, Vector<double>>
                {
                    InitialLearningRate = 0.01,
                    MaxIterations = 2000,
                    Tolerance = 1e-6,
                    UseAdaptiveLearningRate = true
                }))
                .ConfigureModel(new MultipleRegression<double>(new RegressionOptions<double>
                {
                    UseIntercept = true
                }))
                .ConfigureFitnessCalculator(new RSquaredFitnessCalculator<double, Matrix<double>, Vector<double>>())
                .ConfigureDataLoader(new InMemoryDataLoader<double, Matrix<double>, Vector<double>>(features, prices))
                .BuildAsync();

            // 6. Evaluate models on test set
            Console.WriteLine("\nEvaluating models on test set:");

            // Split data into training and test sets (80/20 split)
            int trainingSize = (int)(features.Rows * 0.8);
            int testSize = features.Rows - trainingSize;

            var trainingFeatures = new Matrix<double>(trainingSize, features.Columns);
            var trainingPrices = new Vector<double>(trainingSize);

            var testFeatures = new Matrix<double>(testSize, features.Columns);
            var testPrices = new Vector<double>(testSize);

            // Simple random split (you can use your framework's DataPreprocessor instead)
            Random random = RandomHelper.CreateSeededRandom(42);
            var indices = Enumerable.Range(0, features.Rows).OrderBy(_ => random.Next()).ToArray();

            for (int i = 0; i < trainingSize; i++)
            {
                int idx = indices[i];
                for (int j = 0; j < features.Columns; j++)
                {
                    trainingFeatures[i, j] = features[idx, j];
                }
                trainingPrices[i] = prices[idx];
            }

            for (int i = 0; i < testSize; i++)
            {
                int idx = indices[trainingSize + i];
                for (int j = 0; j < features.Columns; j++)
                {
                    testFeatures[i, j] = features[idx, j];
                }
                testPrices[i] = prices[idx];
            }

            // Evaluate linear regression
            var linearPredictions = linearModel.Predict(testFeatures);
            var linearMetrics = EvaluateModel(linearPredictions, testPrices);
            Console.WriteLine("\nMultiple Linear Regression performance:");
            PrintMetrics(linearMetrics);

            // Evaluate ridge regression
            var ridgePredictions = ridgeModel.Predict(testFeatures);
            var ridgeMetrics = EvaluateModel(ridgePredictions, testPrices);
            Console.WriteLine("\nRidge Regression performance:");
            PrintMetrics(ridgeMetrics);

            // 7. Choose the best model based on validation metrics
            var bestModel = ridgeMetrics.r2 > linearMetrics.r2 ? ridgeModel : linearModel;
            var bestModelName = ridgeMetrics.r2 > linearMetrics.r2 ? "Ridge Regression" : "Multiple Linear Regression";

            Console.WriteLine($"\nBest model based on R² score: {bestModelName}");

            // 8. Analyze feature importance (for the best model)
            if (bestModel.Model != null)
            {
                if (bestModel.Model.GetType().GetProperty("Coefficients") != null)
                {
                    var coefficients = bestModel.Model.GetType().GetProperty("Coefficients")?.GetValue(bestModel.Model);

                    if (coefficients is Vector<double> coefVector)
                    {
                        Console.WriteLine("\nFeature Importance Analysis:");
                        Console.WriteLine("---------------------------");

                        // Create list of features and their importance
                        var featureImportance = new List<(string Feature, double Importance)>();
                        for (int i = 0; i < featureNames.Length; i++)
                        {
                            featureImportance.Add((featureNames[i], Math.Abs(coefVector[i])));
                        }

                        // Sort by importance (descending)
                        featureImportance.Sort((a, b) => b.Importance.CompareTo(a.Importance));

                        // Print feature importance
                        for (int i = 0; i < featureImportance.Count; i++)
                        {
                            Console.WriteLine($"{i + 1}. {featureImportance[i].Feature}: {featureImportance[i].Importance:F4}");
                        }
                    }
                }
            }

            // 9. Save the best model
            string modelPath = "real_estate_analysis_model.bin";
            modelBuilder.SaveModel(bestModel, modelPath);
            Console.WriteLine($"\nBest model saved to {modelPath}");

            // 10. Make predictions on new properties
            Console.WriteLine("\nMaking predictions for sample properties using the best model:");

            // Create new property examples to predict
            var newProperties = CreateSampleProperties();
            Console.WriteLine($"\nPredicting prices for {newProperties.Length} sample properties...");

            for (int i = 0; i < newProperties.Length; i++)
            {
                var property = newProperties[i];

                // Create matrix with single row
                var propertyMatrix = new Matrix<double>(1, featureNames.Length);

                for (int j = 0; j < property.Features.Length; j++)
                {
                    propertyMatrix[0, j] = property.Features[j];
                }

                // Add engineered feature (square footage squared)
                if (sqFtFeatureIndex >= 0 && property.Features.Length < featureNames.Length)
                {
                    propertyMatrix[0, featureNames.Length - 1] =
                        property.Features[sqFtFeatureIndex] * property.Features[sqFtFeatureIndex];
                }

                // Make prediction
                var predictedPrice = bestModel.Predict(propertyMatrix);

                // Print property details and prediction
                Console.WriteLine($"\nProperty {i + 1}:");
                Console.WriteLine($"- Location: {property.Description}");
                Console.WriteLine($"- Square Feet: {property.Features[sqFtFeatureIndex]:N0}");
                Console.WriteLine($"- Bedrooms: {property.Features[Array.IndexOf(featureNames, "Bedrooms")]:N0}");
                Console.WriteLine($"- Bathrooms: {property.Features[Array.IndexOf(featureNames, "Bathrooms")]:N1}");
                Console.WriteLine($"- Property Age: {property.Features[Array.IndexOf(featureNames, "PropertyAge")]:N0} years");
                Console.WriteLine($"- Predicted Price: ${predictedPrice[0]:N0}");
            }

            Console.WriteLine("\nEnhanced regression example completed!");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }
    }

    // Helper method to evaluate model performance
    private (double mse, double mae, double r2) EvaluateModel(
        Vector<double> predictions,
        Vector<double> testTargets)
    {
        // Calculate Mean Squared Error
        double mse = 0;
        for (int i = 0; i < testTargets.Length; i++)
        {
            double error = testTargets[i] - predictions[i];
            mse += error * error;
        }
        mse /= testTargets.Length;

        // Calculate Mean Absolute Error
        double mae = 0;
        for (int i = 0; i < testTargets.Length; i++)
        {
            mae += Math.Abs(testTargets[i] - predictions[i]);
        }
        mae /= testTargets.Length;

        // Calculate R-squared
        double meanTarget = 0;
        for (int i = 0; i < testTargets.Length; i++)
        {
            meanTarget += testTargets[i];
        }
        meanTarget /= testTargets.Length;

        double totalSumSquares = 0;
        for (int i = 0; i < testTargets.Length; i++)
        {
            double diff = testTargets[i] - meanTarget;
            totalSumSquares += diff * diff;
        }

        double residualSumSquares = 0;
        for (int i = 0; i < testTargets.Length; i++)
        {
            double error = testTargets[i] - predictions[i];
            residualSumSquares += error * error;
        }

        double r2 = 1 - (residualSumSquares / totalSumSquares);

        return (mse, mae, r2);
    }

    // Helper method to print evaluation metrics
    private void PrintMetrics((double mse, double mae, double r2) metrics)
    {
        Console.WriteLine($"- Mean Squared Error (MSE): {metrics.mse:N2}");
        Console.WriteLine($"- Mean Absolute Error (MAE): {metrics.mae:N2}");
        Console.WriteLine($"- R² Score: {metrics.r2:N4}");
    }

    // Helper method to load synthetic real estate data
    private (Matrix<double> Features, Vector<double> Prices, string[] FeatureNames, int[] ZipCodes) LoadRealEstateData()
    {
        // Generate synthetic dataset for real estate properties
        Random random = RandomHelper.CreateSeededRandom(123);  // For reproducibility

        // Define features names
        string[] featureNames = { "SquareFeet", "Bedrooms", "Bathrooms", "PropertyAge", "GarageSize",
                                 "HasBasement", "HasPool", "DistanceToCenter", "SchoolRating" };

        // Define number of samples
        int numSamples = 500;

        // Create feature matrix and target vector
        var features = new Matrix<double>(numSamples, featureNames.Length);
        var prices = new Vector<double>(numSamples);
        var zipCodes = new int[numSamples];

        // Create synthetic zip codes (5 different areas)
        int[] zipCodeValues = { 98101, 98102, 98103, 98104, 98105 };

        // Generate data
        for (int i = 0; i < numSamples; i++)
        {
            // Assign a zip code
            int zipCodeIndex = random.Next(zipCodeValues.Length);
            zipCodes[i] = zipCodeValues[zipCodeIndex];

            // Generate base price for the zip code (different areas have different base prices)
            double basePrice = zipCodeIndex switch
            {
                0 => 500000,  // High-end downtown area
                1 => 400000,  // Nice suburban area
                2 => 350000,  // Mixed residential area
                3 => 300000,  // Older residential area
                4 => 450000,  // Newer development area
                _ => 350000   // Default
            };

            // Square feet (varying by area)
            double meanSqFt = zipCodeIndex switch
            {
                0 => 1800,   // Smaller downtown units
                1 => 2200,   // Larger suburban homes
                2 => 1900,   // Mixed sizes
                3 => 1700,   // Older, typically smaller homes
                4 => 2400,   // Newer, typically larger homes
                _ => 2000    // Default
            };
            features[i, 0] = Math.Max(800, meanSqFt + random.NextDouble() * 1000 - 500);

            // Bedrooms (correlated with square feet)
            features[i, 1] = Math.Max(1, Math.Min(6, Math.Round(features[i, 0] / 750)));

            // Bathrooms (correlated with bedrooms)
            features[i, 2] = Math.Max(1, Math.Min(5, Math.Round(features[i, 1] * 0.75) + random.Next(0, 2)));

            // Property age (varying by area)
            double meanAge = zipCodeIndex switch
            {
                0 => 30,    // Older downtown buildings
                1 => 20,    // Established suburbs
                2 => 35,    // Older mixed area
                3 => 50,    // Oldest area
                4 => 8,     // Newest development
                _ => 25     // Default
            };
            features[i, 3] = Math.Max(0, meanAge + random.NextDouble() * 20 - 10);

            // Garage size (number of cars)
            features[i, 4] = random.Next(0, 4);

            // Has basement (boolean 0 or 1)
            features[i, 5] = random.NextDouble() > 0.5 ? 1 : 0;

            // Has pool (boolean 0 or 1, more likely in certain areas)
            features[i, 6] = random.NextDouble() > (zipCodeIndex == 1 || zipCodeIndex == 4 ? 0.8 : 0.9) ? 1 : 0;

            // Distance to city center (miles, varying by area)
            double meanDistance = zipCodeIndex switch
            {
                0 => 1,     // Downtown
                1 => 8,     // Near suburbs
                2 => 12,    // Further out
                3 => 15,    // Even further
                4 => 10,    // Planned community
                _ => 10     // Default
            };
            features[i, 7] = Math.Max(0.1, meanDistance + random.NextDouble() * 6 - 3);

            // School rating (1-10)
            double meanRating = zipCodeIndex switch
            {
                0 => 7.5,   // Good downtown schools
                1 => 8.5,   // Excellent suburban schools
                2 => 6.5,   // Average schools
                3 => 5.5,   // Below average schools
                4 => 9.0,   // New excellent schools
                _ => 7.0    // Default
            };
            features[i, 8] = Math.Max(1, Math.Min(10, meanRating + random.NextDouble() * 2 - 1));

            // Calculate price based on features with some noise
            prices[i] = CalculatePrice(features, i, basePrice, random);
        }

        return (features, prices, featureNames, zipCodes);
    }

    // Helper method to calculate synthetic house price based on features
    private double CalculatePrice(Matrix<double> features, int index, double basePrice, Random random)
    {
        double price = basePrice;

        // Square feet has major impact
        price += features[index, 0] * 150;

        // Each bedroom adds value
        price += features[index, 1] * 15000;

        // Each bathroom adds value
        price += features[index, 2] * 25000;

        // Older properties are worth less (2% depreciation per year)
        price *= Math.Pow(0.98, features[index, 3]);

        // Garage adds value
        price += features[index, 4] * 12000;

        // Basement adds value
        if (features[index, 5] > 0.5) price += 30000;

        // Pool adds value
        if (features[index, 6] > 0.5) price += 45000;

        // Distance to center decreases value
        price *= Math.Pow(0.97, features[index, 7]);

        // School rating increases value
        price *= 1 + (features[index, 8] - 5) * 0.03;

        // Add random noise (±10%)
        price *= 0.9 + random.NextDouble() * 0.2;

        return price;
    }

    // Structure to represent a property for prediction
    private struct Property
    {
        public double[] Features;
        public string Description;
    }

    // Helper method to create sample properties for prediction
    private Property[] CreateSampleProperties()
    {
        return new Property[]
        {
            new Property
            {
                Features = new double[] { 2200, 3, 2.5, 5, 2, 1, 0, 12, 8.5 },
                Description = "Modern suburban home in school district 98102"
            },
            new Property
            {
                Features = new double[] { 1500, 2, 1, 30, 1, 0, 0, 3, 7.2 },
                Description = "Downtown condo in 98101"
            },
            new Property
            {
                Features = new double[] { 3500, 5, 4, 15, 3, 1, 1, 15, 9.0 },
                Description = "Luxury home in 98105 with pool"
            }
        };
    }
}
