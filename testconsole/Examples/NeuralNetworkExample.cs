using AiDotNet;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Normalizers;
using AiDotNet.Optimizers;

namespace AiDotNetTestConsole.Examples;

/// <summary>
/// Demonstrates how to use neural networks in AiDotNet for classification and regression tasks.
/// </summary>
public class NeuralNetworkExample
{
    public void RunExample()
    {
        Console.WriteLine("Neural Network Example");
        Console.WriteLine("=====================\n");

        // First, let's demonstrate a simple classification example (Iris flower classification)
        RunIrisClassificationExample();

        Console.WriteLine("\nPress any key to continue to the regression example...");
        Console.ReadKey();
        Console.Clear();

        // Next, let's demonstrate a regression example (housing price prediction)
        RunHousingRegressionExample();
    }

    /// <summary>
    /// Example of using a neural network for a classification task (Iris flower classification).
    /// </summary>
    private void RunIrisClassificationExample()
    {
        Console.WriteLine("Example 1: Iris Flower Classification");
        Console.WriteLine("------------------------------------\n");
        Console.WriteLine("This example demonstrates how to use a neural network for classification.");
        Console.WriteLine("We'll classify iris flowers into three species based on four features.\n");

        try
        {
            // 1. Create sample data (Iris dataset - simplified version)
            // Features: sepal length, sepal width, petal length, petal width
            var irisData = new double[,]
            {
                // Setosa examples
                { 5.1, 3.5, 1.4, 0.2 },
                { 4.9, 3.0, 1.4, 0.2 },
                { 4.7, 3.2, 1.3, 0.2 },
                { 4.6, 3.1, 1.5, 0.2 },
                
                // Versicolor examples
                { 7.0, 3.2, 4.7, 1.4 },
                { 6.4, 3.2, 4.5, 1.5 },
                { 6.9, 3.1, 4.9, 1.5 },
                { 5.5, 2.3, 4.0, 1.3 },
                
                // Virginica examples
                { 6.3, 3.3, 6.0, 2.5 },
                { 5.8, 2.7, 5.1, 1.9 },
                { 7.1, 3.0, 5.9, 2.1 },
                { 6.3, 2.9, 5.6, 1.8 }
            };

            // Convert to tensors
            var features = new Tensor<double>(new int[] { 12, 4 }); // 12 samples, 4 features

            // Fill the tensor with data
            for (int i = 0; i < 12; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    features[i, j] = irisData[i, j];
                }
            }

            // Labels: 0 = Setosa, 1 = Versicolor, 2 = Virginica (already one-hot encoded)
            var irisLabels = new double[,]
            {
                { 1, 0, 0 }, // Setosa
                { 1, 0, 0 },
                { 1, 0, 0 },
                { 1, 0, 0 },

                { 0, 1, 0 }, // Versicolor
                { 0, 1, 0 },
                { 0, 1, 0 },
                { 0, 1, 0 },

                { 0, 0, 1 }, // Virginica
                { 0, 0, 1 },
                { 0, 0, 1 },
                { 0, 0, 1 }
            };

            var labels = new Tensor<double>(new int[] { 12, 3 });

            for (int i = 0; i < 12; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    labels[i, j] = irisLabels[i, j];
                }
            }

            // 2. Normalize the input features using the standard normalizer
            Console.WriteLine("Normalizing input features...");
            var normalizer = new DecimalNormalizer<double, Tensor<double>, Tensor<double>>();
            var (normalizedFeatures, featureParams) = normalizer.NormalizeInput(features);

            Console.WriteLine("Data prepared. Starting model training...");

            // 3. Create neural network architecture for classification
            var architecture = new NeuralNetworkArchitecture<double>(
                complexity: NetworkComplexity.Medium,
                isMultiClass: true
            );

            // 4. Create the model and model builder
            var neuralNetworkModel = new NeuralNetwork<double>(architecture);
            var modelBuilder = new PredictionModelBuilder<double, Tensor<double>, Tensor<double>>();

            // 5. Configure Adam optimizer for training
            var adamOptions = new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>
            {
                LearningRate = 0.01,
                MaxIterations = 1000,
                Beta1 = 0.9,
                Beta2 = 0.999,
                Epsilon = 1e-8
            };

            var optimizer = new AdamOptimizer<double, Tensor<double>, Tensor<double>>(neuralNetworkModel, adamOptions);

            // 6. Build the neural network model
            Console.WriteLine("Training the network...");
            var trainedModel = modelBuilder
                .ConfigureOptimizer(optimizer)
                .Build(normalizedFeatures, labels);  // Note: using normalized features but original labels

            Console.WriteLine("Model trained successfully!");

            // 7. Make predictions with the trained model using normalized features
            Console.WriteLine("\nMaking predictions with the trained model...");
            var predictions = modelBuilder.Predict(normalizedFeatures, trainedModel);

            // 8. Display results
            Console.WriteLine("\nPrediction Results:");
            Console.WriteLine("-------------------");
            Console.WriteLine("Sample | Features                      | Actual Class | Predicted Class | Confidence");
            Console.WriteLine("-------|-------------------------------|--------------|-----------------|----------");

            string[] classNames = { "Setosa", "Versicolor", "Virginica" };

            for (int i = 0; i < features.Shape[0]; i++)
            {
                // Get the actual class (the index of the maximum value in the label)
                int actualClass = 0;
                for (int j = 1; j < labels.Shape[1]; j++)
                {
                    if (labels[i, j] > labels[i, actualClass])
                    {
                        actualClass = j;
                    }
                }

                // Get the predicted class (the index of the maximum value in the prediction)
                int predictedClass = 0;
                double maxProb = predictions[i, 0];
                for (int j = 1; j < predictions.Shape[1]; j++)
                {
                    if (predictions[i, j] > maxProb)
                    {
                        maxProb = predictions[i, j];
                        predictedClass = j;
                    }
                }

                // Format the features for display (using original features for display)
                string featuresStr = $"{features[i, 0]:F1}, {features[i, 1]:F1}, " +
                                    $"{features[i, 2]:F1}, {features[i, 3]:F1}";

                Console.WriteLine($"{i + 1,7} | {featuresStr,29} | {classNames[actualClass],-12} | {classNames[predictedClass],-15} | {maxProb:P2}");
            }

            // 9. Save the trained model along with normalization parameters
            Console.WriteLine("\nSaving the trained model...");
            string modelPath = "iris_classification_model.bin";
            modelBuilder.SaveModel(trainedModel, modelPath);

            // Save normalization parameters separately
            string normParamsPath = "iris_normalization_params.bin";
            SaveNormalizationParameters(featureParams, normParamsPath);

            Console.WriteLine($"Model saved to {modelPath}");
            Console.WriteLine($"Normalization parameters saved to {normParamsPath}");

            // 10. Load the model and normalization parameters
            Console.WriteLine("\nLoading the saved model and normalization parameters...");
            var loadedModel = modelBuilder.LoadModel(modelPath);
            var loadedFeatureParams = LoadFeatureNormalizationParameters<double>(normParamsPath);
            Console.WriteLine("Model and parameters loaded successfully");

            // 11. Use the loaded model to make a prediction on a new sample
            Console.WriteLine("\nMaking a prediction with the loaded model on a new sample...");

            // Create a new sample (Setosa)
            var newSampleRaw = new Tensor<double>(new int[] { 1, 4 });  // 1 sample, 4 features
            newSampleRaw[0, 0] = 5.0;
            newSampleRaw[0, 1] = 3.3;
            newSampleRaw[0, 2] = 1.4;
            newSampleRaw[0, 3] = 0.2;

            // Normalize the new sample using the saved parameters
            var newSample = new Tensor<double>(newSampleRaw.Shape);

            // Apply normalization manually using the saved parameters
            for (int j = 0; j < newSampleRaw.Shape[1]; j++)
            {
                // Apply the normalization formula using the saved parameters
                double value = newSampleRaw[0, j];
                double normalizedValue = (value - loadedFeatureParams[j].Mean) / loadedFeatureParams[j].StdDev;
                newSample[0, j] = normalizedValue;
            }

            var newPrediction = modelBuilder.Predict(newSample, loadedModel);

            // Get the predicted class
            int newPredictedClass = 0;
            double newMaxProb = newPrediction[0, 0];
            for (int j = 1; j < newPrediction.Shape[1]; j++)
            {
                if (newPrediction[0, j] > newMaxProb)
                {
                    newMaxProb = newPrediction[0, j];
                    newPredictedClass = j;
                }
            }

            Console.WriteLine($"New sample features: {newSampleRaw[0, 0]:F1}, {newSampleRaw[0, 1]:F1}, " +
                             $"{newSampleRaw[0, 2]:F1}, {newSampleRaw[0, 3]:F1}");
            Console.WriteLine($"Predicted class: {classNames[newPredictedClass]} with {newMaxProb:P2} confidence");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }
    }

    /// <summary>
    /// Example of using a neural network for a regression task (housing price prediction).
    /// </summary>
    private void RunHousingRegressionExample()
    {
        Console.WriteLine("Example 2: Housing Price Regression");
        Console.WriteLine("----------------------------------\n");
        Console.WriteLine("This example demonstrates how to use a neural network for regression.");
        Console.WriteLine("We'll predict house prices based on features like size, bedrooms, etc.\n");

        try
        {
            // 1. Create sample data (simplified housing dataset)
            // Features: size (sq ft), bedrooms, bathrooms, age (years), distance to city center (miles)
            var housingData = new double[,]
            {
                { 1400, 3, 2, 15, 5 },    // $235,000
                { 1800, 4, 2, 10, 7 },    // $285,000
                { 1200, 2, 1, 20, 3 },    // $195,000
                { 2200, 4, 3, 5, 10 },    // $330,000
                { 1600, 3, 2, 12, 4 },    // $255,000
                { 2500, 5, 3, 8, 12 },    // $365,000
                { 1100, 2, 1, 25, 2 },    // $180,000
                { 1900, 3, 2, 7, 8 },     // $295,000
                { 3000, 5, 4, 2, 15 },    // $420,000
                { 1300, 3, 1, 18, 6 }     // $210,000
            };

            // Target: house prices (in thousands of dollars)
            var housePrices = new double[]
            {
                235,
                285,
                195,
                330,
                255,
                365,
                180,
                295,
                420,
                210
            };

            // Convert to tensors
            var features = new Tensor<double>(new int[] { 10, 5 }); // 10 samples, 5 features
            var targets = new Tensor<double>(new int[] { 10, 1 });  // 10 samples, 1 target value

            // Fill the tensors with data
            for (int i = 0; i < 10; i++)
            {
                for (int j = 0; j < 5; j++)
                {
                    features[i, j] = housingData[i, j];
                }
                targets[i, 0] = housePrices[i];
            }

            // 2. Normalize the data (important for neural networks)
            Console.WriteLine("Normalizing the data...");
            var normalizer = new DecimalNormalizer<double, Tensor<double>, Tensor<double>>();

            // Normalize features and get normalization parameters
            var (normalizedFeatures, featureParams) = normalizer.NormalizeInput(features);

            // Normalize targets and get normalization parameters
            var (normalizedTargets, targetParams) = normalizer.NormalizeOutput(targets);

            // 3. Create neural network architecture for regression
            Console.WriteLine("Creating neural network architecture...");
            var architecture = new NeuralNetworkArchitecture<double>(
                taskType: NeuralNetworkTaskType.Regression,
                complexity: NetworkComplexity.Medium
            );

            // 4. Create the model and model builder
            var neuralNetworkModel = new NeuralNetwork<double>(architecture);
            var modelBuilder = new PredictionModelBuilder<double, Tensor<double>, Tensor<double>>();

            // 5. Configure Adam optimizer for training
            var adamOptions = new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>
            {
                LearningRate = 0.01,
                MaxIterations = 2000,
                Beta1 = 0.9,
                Beta2 = 0.999,
                Epsilon = 1e-8
            };

            var optimizer = new AdamOptimizer<double, Tensor<double>, Tensor<double>>(neuralNetworkModel, adamOptions);

            // 6. Build the neural network model
            Console.WriteLine("Training the network...");
            var trainedModel = modelBuilder
                .ConfigureOptimizer(optimizer)
                .Build(normalizedFeatures, normalizedTargets);

            Console.WriteLine("Model trained successfully!");

            // 7. Make predictions with the trained model
            Console.WriteLine("\nMaking predictions with the trained model...");
            var normalizedPredictions = modelBuilder.Predict(normalizedFeatures, trainedModel);

            // Denormalize the predictions
            var predictions = normalizer.Denormalize(normalizedPredictions, targetParams);

            // 8. Display results
            Console.WriteLine("\nPrediction Results:");
            Console.WriteLine("-------------------");
            Console.WriteLine("House | Features                                  | Actual Price | Predicted Price | Error");
            Console.WriteLine("------|-------------------------------------------|--------------|-----------------|-------");

            for (int i = 0; i < features.Shape[0]; i++)
            {
                // Format the features for display
                string featuresStr = $"{features[i, 0]:F0} sqft, {features[i, 1]:F0}bd, " +
                                   $"{features[i, 2]:F0}ba, {features[i, 3]:F0}yr, " +
                                   $"{features[i, 4]:F0}mi";

                double actualPrice = targets[i, 0];
                double pricePrediction = predictions[i, 0];
                double error = Math.Abs(actualPrice - pricePrediction);
                double errorPercent = error / actualPrice * 100;

                Console.WriteLine($"{i + 1,6} | {featuresStr,41} | ${actualPrice,12:F0} | ${pricePrediction,15:F0} | {errorPercent,5:F1}%");
            }

            // 9. Calculate and display model metrics
            double mse = 0;
            double mae = 0;
            double maxError = 0;

            for (int i = 0; i < targets.Shape[0]; i++)
            {
                double error = targets[i, 0] - predictions[i, 0];
                mse += error * error;
                mae += Math.Abs(error);
                maxError = Math.Max(maxError, Math.Abs(error));
            }

            mse /= targets.Shape[0];
            mae /= targets.Shape[0];

            Console.WriteLine("\nModel Metrics:");
            Console.WriteLine($"Mean Squared Error (MSE): {mse:F2}");
            Console.WriteLine($"Mean Absolute Error (MAE): {mae:F2}");
            Console.WriteLine($"Root Mean Squared Error (RMSE): {Math.Sqrt(mse):F2}");
            Console.WriteLine($"Maximum Error: {maxError:F2}");

            // 10. Save the trained model and normalization parameters
            Console.WriteLine("\nSaving the trained model and normalization parameters...");
            string modelPath = "housing_regression_model.bin";
            modelBuilder.SaveModel(trainedModel, modelPath);

            string featureParamsPath = "housing_feature_params.bin";
            string targetParamsPath = "housing_target_params.bin";

            SaveNormalizationParameters(featureParams, featureParamsPath);
            SaveNormalizationParameters(targetParams, targetParamsPath);

            Console.WriteLine($"Model saved to {modelPath}");
            Console.WriteLine($"Normalization parameters saved to {featureParamsPath} and {targetParamsPath}");

            // 11. Load the model and normalization parameters
            Console.WriteLine("\nLoading the saved model and normalization parameters...");
            var loadedModel = modelBuilder.LoadModel(modelPath);
            var loadedFeatureParams = LoadFeatureNormalizationParameters<double>(featureParamsPath);
            var loadedTargetParams = LoadTargetNormalizationParameters<double>(targetParamsPath);
            Console.WriteLine("Model and parameters loaded successfully");

            // 12. Make a prediction on a new house
            Console.WriteLine("\nPredicting price for a new house...");

            // New house: 1750 sqft, 3 bedrooms, 2 bathrooms, 5 years old, 6 miles from city center
            var newHouse = new Tensor<double>(new int[] { 1, 5 });
            newHouse[0, 0] = 1750;
            newHouse[0, 1] = 3;
            newHouse[0, 2] = 2;
            newHouse[0, 3] = 5;
            newHouse[0, 4] = 6;

            // Normalize the new house data manually using loaded parameters
            var normalizedNewHouse = new Tensor<double>(newHouse.Shape);
            for (int j = 0; j < newHouse.Shape[1]; j++)
            {
                // Apply the normalization formula using the saved parameters
                normalizedNewHouse[0, j] = (newHouse[0, j] - loadedFeatureParams[j].Mean) / loadedFeatureParams[j].StdDev;
            }

            // Make prediction
            var normalizedNewPrediction = modelBuilder.Predict(normalizedNewHouse, loadedModel);

            // Denormalize the prediction using the loaded target parameters
            var denormalizedPrediction = normalizer.Denormalize(normalizedNewPrediction, loadedTargetParams);
            double predictedPrice = denormalizedPrediction[0, 0];

            Console.WriteLine("New house features:");
            Console.WriteLine($"- Size: 1750 sq ft");
            Console.WriteLine($"- Bedrooms: 3");
            Console.WriteLine($"- Bathrooms: 2");
            Console.WriteLine($"- Age: 5 years");
            Console.WriteLine($"- Distance to city center: 6 miles");
            Console.WriteLine($"\nPredicted price: ${predictedPrice:F0}");

            Console.WriteLine("\nNeural network regression example completed!");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }
    }

    /// <summary>
    /// Helper method to save normalization parameters to a file.
    /// </summary>
    private void SaveNormalizationParameters<T>(List<NormalizationParameters<T>> parameters, string filePath)
    {
        using (var stream = new FileStream(filePath, FileMode.Create))
        using (var writer = new BinaryWriter(stream))
        {
            // Write count
            writer.Write(parameters.Count);

            // Write each parameter set
            foreach (var param in parameters)
            {
                // Serialize each parameter set (implementation depends on your NormalizationParameters class)
                writer.Write(param.Mean.ToString());
                writer.Write(param.StdDev.ToString());
                // Add any other parameters your class contains
            }
        }
    }

    /// <summary>
    /// Helper method to save a single normalization parameter to a file.
    /// </summary>
    private void SaveNormalizationParameters<T>(NormalizationParameters<T> parameters, string filePath)
    {
        using (var stream = new FileStream(filePath, FileMode.Create))
        using (var writer = new BinaryWriter(stream))
        {
            // Serialize the parameter set
            writer.Write(parameters.Mean.ToString());
            writer.Write(parameters.StdDev.ToString());
            // Add any other parameters your class contains
        }
    }

    /// <summary>
    /// Helper method to load a list of normalization parameters from a file.
    /// </summary>
    private List<NormalizationParameters<T>> LoadFeatureNormalizationParameters<T>(string filePath)
    {
        var parameters = new List<NormalizationParameters<T>>();

        using (var stream = new FileStream(filePath, FileMode.Open))
        using (var reader = new BinaryReader(stream))
        {
            // Read count of parameters
            int count = reader.ReadInt32();

            // Read each parameter set
            for (int i = 0; i < count; i++)
            {
                var mean = Convert.ChangeType(reader.ReadString(), typeof(T));
                var stdDev = Convert.ChangeType(reader.ReadString(), typeof(T));

                parameters.Add(new NormalizationParameters<T>
                {
                    Mean = (T)mean,
                    StdDev = (T)stdDev
                    // Read any other parameters your class contains
                });
            }
        }

        return parameters;
    }

    /// <summary>
    /// Helper method to load a single normalization parameter from a file.
    /// </summary>
    private NormalizationParameters<T> LoadTargetNormalizationParameters<T>(string filePath)
    {
        using (var stream = new FileStream(filePath, FileMode.Open))
        using (var reader = new BinaryReader(stream))
        {
            var mean = Convert.ChangeType(reader.ReadString(), typeof(T));
            var stdDev = Convert.ChangeType(reader.ReadString(), typeof(T));

            return new NormalizationParameters<T>
            {
                Mean = (T)mean,
                StdDev = (T)stdDev
                // Read any other parameters your class contains
            };
        }
    }
}