using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;

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

        // Labels: 0 = Setosa, 1 = Versicolor, 2 = Virginica
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

        // Convert to tensors
        // Create tensors with proper dimensions
        var features = new Tensor<double>(new int[] { 12, 4 }); // 12 samples, 4 features
        var labels = new Tensor<double>(new int[] { 12, 3 });   // 12 samples, 3 classes

        // Fill the tensors with data
        for (int i = 0; i < 12; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                features[new int[] { i, j }] = irisData[i, j];
            }

            for (int j = 0; j < 3; j++)
            {
                labels[new int[] { i, j }] = irisLabels[i, j];
            }
        }

        // 2. Create neural network architecture for classification
        Console.WriteLine("Creating neural network architecture...");
        var architecture = new NeuralNetworkArchitecture<double>(
            inputFeatures: 4,    // 4 features
            numClasses: 3,       // 3 classes (Setosa, Versicolor, Virginica)
            complexity: NetworkComplexity.Medium
        );

        // 3. Create and initialize the neural network
        Console.WriteLine("Initializing neural network...");
        var neuralNetwork = new NeuralNetwork<double>(architecture);

        // 4. Train the network
        Console.WriteLine("Training the network...");
        // In a real application, you would use more epochs and a proper training loop
        // with validation data, but we'll keep it simple for this example
        for (int epoch = 0; epoch < 1000; epoch++)
        {
            neuralNetwork.Train(features, labels);
            double loss = neuralNetwork.GetLastLoss();

            if (epoch % 100 == 0)
            {
                Console.WriteLine($"Epoch {epoch}: Loss = {loss:F4}");
            }
        }

        // 5. Make predictions with the trained network
        Console.WriteLine("\nMaking predictions with the trained network...");
        var predictions = neuralNetwork.Predict(features);

        // 6. Display results
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
                if (labels[new int[] { i, j }] > labels[new int[] { i, actualClass }])
                {
                    actualClass = j;
                }
            }

            // Get the predicted class (the index of the maximum value in the prediction)
            int predictedClass = 0;
            double maxProb = predictions[new int[] { i, 0 }];
            for (int j = 1; j < predictions.Shape[1]; j++)
            {
                if (predictions[new int[] { i, j }] > maxProb)
                {
                    maxProb = predictions[new int[] { i, j }];
                    predictedClass = j;
                }
            }

            // Format the features for display
            string featuresStr = $"{features[new int[] { i, 0 }]:F1}, {features[new int[] { i, 1 }]:F1}, " +
                                $"{features[new int[] { i, 2 }]:F1}, {features[new int[] { i, 3 }]:F1}";

            Console.WriteLine($"{i + 1,7} | {featuresStr,29} | {classNames[actualClass],-12} | {classNames[predictedClass],-15} | {maxProb:P2}");
        }

        // 7. Save the trained model
        Console.WriteLine("\nSaving the trained model...");
        byte[] serializedModel = neuralNetwork.Serialize();
        Console.WriteLine($"Model saved ({serializedModel.Length} bytes)");

        // 8. Load the model (in a real application, this would be in a different session)
        Console.WriteLine("\nLoading the saved model...");
        var loadedNetwork = new NeuralNetwork<double>(architecture);
        loadedNetwork.Deserialize(serializedModel);
        Console.WriteLine("Model loaded successfully");

        // 9. Use the loaded model to make a prediction on a new sample
        Console.WriteLine("\nMaking a prediction with the loaded model on a new sample...");

        // Create a new tensor for the sample
        var newSample = new Tensor<double>(new int[] { 1, 4 });  // 1 sample, 4 features
        newSample[new int[] { 0, 0 }] = 5.0;
        newSample[new int[] { 0, 1 }] = 3.3;
        newSample[new int[] { 0, 2 }] = 1.4;
        newSample[new int[] { 0, 3 }] = 0.2;

        var newPrediction = loadedNetwork.Predict(newSample);

        // Get the predicted class
        int newPredictedClass = 0;
        double newMaxProb = newPrediction[new int[] { 0, 0 }];
        for (int j = 1; j < newPrediction.Shape[1]; j++)
        {
            if (newPrediction[new int[] { 0, j }] > newMaxProb)
            {
                newMaxProb = newPrediction[new int[] { 0, j }];
                newPredictedClass = j;
            }
        }

        Console.WriteLine($"New sample features: {newSample[new int[] { 0, 0 }]:F1}, {newSample[new int[] { 0, 1 }]:F1}, " +
                         $"{newSample[new int[] { 0, 2 }]:F1}, {newSample[new int[] { 0, 3 }]:F1}");
        Console.WriteLine($"Predicted class: {classNames[newPredictedClass]} with {newMaxProb:P2} confidence");
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
        var housePrices = new double[,]
        {
            { 235 },
            { 285 },
            { 195 },
            { 330 },
            { 255 },
            { 365 },
            { 180 },
            { 295 },
            { 420 },
            { 210 }
        };

        // Convert to tensors
        // Create tensors with proper dimensions
        var features = new Tensor<double>(new int[] { 10, 5 }); // 10 samples, 5 features
        var targets = new Tensor<double>(new int[] { 10, 1 });  // 10 samples, 1 target value

        // Fill the tensors with data
        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                features[new int[] { i, j }] = housingData[i, j];
            }

            targets[new int[] { i, 0 }] = housePrices[i, 0];
        }

        // 2. Normalize the data (important for regression)
        Console.WriteLine("Normalizing the data...");

        // Calculate mean and standard deviation for each feature
        var featureMeans = new double[features.Shape[1]];
        var featureStds = new double[features.Shape[1]];

        for (int j = 0; j < features.Shape[1]; j++)
        {
            double sum = 0;
            for (int i = 0; i < features.Shape[0]; i++)
            {
                sum += features[new int[] { i, j }];
            }
            featureMeans[j] = sum / features.Shape[0];

            double sumSquaredDiff = 0;
            for (int i = 0; i < features.Shape[0]; i++)
            {
                double diff = features[new int[] { i, j }] - featureMeans[j];
                sumSquaredDiff += diff * diff;
            }
            featureStds[j] = Math.Sqrt(sumSquaredDiff / features.Shape[0]);

            // Avoid division by zero
            if (featureStds[j] == 0)
            {
                featureStds[j] = 1;
            }
        }

        // Normalize features
        var normalizedFeatures = new Tensor<double>(new int[] { features.Shape[0], features.Shape[1] });
        for (int i = 0; i < features.Shape[0]; i++)
        {
            for (int j = 0; j < features.Shape[1]; j++)
            {
                normalizedFeatures[new int[] { i, j }] = (features[new int[] { i, j }] - featureMeans[j]) / featureStds[j];
            }
        }

        // Calculate mean and standard deviation for target
        double targetMean = 0;
        for (int i = 0; i < targets.Shape[0]; i++)
        {
            targetMean += targets[new int[] { i, 0 }];
        }
        targetMean /= targets.Shape[0];

        double targetSumSquaredDiff = 0;
        for (int i = 0; i < targets.Shape[0]; i++)
        {
            double diff = targets[new int[] { i, 0 }] - targetMean;
            targetSumSquaredDiff += diff * diff;
        }
        double targetStd = Math.Sqrt(targetSumSquaredDiff / targets.Shape[0]);

        // Normalize targets
        var normalizedTargets = new Tensor<double>(new int[] { targets.Shape[0], 1 });
        for (int i = 0; i < targets.Shape[0]; i++)
        {
            normalizedTargets[new int[] { i, 0 }] = (targets[new int[] { i, 0 }] - targetMean) / targetStd;
        }

        // 3. Create neural network architecture for regression
        Console.WriteLine("Creating neural network architecture...");
        var architecture = new NeuralNetworkArchitecture<double>(
            inputFeatures: 5,    // 5 features
            outputSize: 1,       // 1 output (price)
            complexity: NetworkComplexity.Medium
        );

        // 4. Create and initialize the neural network
        Console.WriteLine("Initializing neural network...");
        var neuralNetwork = new NeuralNetwork<double>(architecture);

        // 5. Train the network
        Console.WriteLine("Training the network...");
        double bestLoss = double.MaxValue;

        for (int epoch = 0; epoch < 2000; epoch++)
        {
            neuralNetwork.Train(normalizedFeatures, normalizedTargets);
            double loss = neuralNetwork.GetLastLoss();

            if (epoch % 200 == 0)
            {
                Console.WriteLine($"Epoch {epoch}: Loss = {loss:F4}");

                if (loss < bestLoss)
                {
                    bestLoss = loss;
                }
            }
        }

        // 6. Make predictions with the trained network
        Console.WriteLine("\nMaking predictions with the trained network...");
        var normalizedPredictions = neuralNetwork.Predict(normalizedFeatures);

        // Denormalize predictions
        var predictions = new Tensor<double>(new int[] { normalizedPredictions.Shape[0], 1 });
        for (int i = 0; i < normalizedPredictions.Shape[0]; i++)
        {
            predictions[new int[] { i, 0 }] = normalizedPredictions[new int[] { i, 0 }] * targetStd + targetMean;
        }

        // 7. Display results
        Console.WriteLine("\nPrediction Results:");
        Console.WriteLine("-------------------");
        Console.WriteLine("House | Features                                  | Actual Price | Predicted Price | Error");
        Console.WriteLine("------|-------------------------------------------|--------------|-----------------|-------");

        for (int i = 0; i < features.Shape[0]; i++)
        {
            // Format the features for display
            string featuresStr = $"{features[new int[] { i, 0 }]:F0} sqft, {features[new int[] { i, 1 }]:F0}bd, " +
                               $"{features[new int[] { i, 2 }]:F0}ba, {features[new int[] { i, 3 }]:F0}yr, " +
                               $"{features[new int[] { i, 4 }]:F0}mi";

            double actualPrice = targets[new int[] { i, 0 }];
            double predictedPrice = predictions[new int[] { i, 0 }];
            double error = Math.Abs(actualPrice - predictedPrice);
            double errorPercent = error / actualPrice * 100;

            Console.WriteLine($"{i + 1,6} | {featuresStr,41} | ${actualPrice,12:F0} | ${predictedPrice,15:F0} | {errorPercent,5:F1}%");
        }

        // 8. Calculate and display model metrics
        double mse = 0;
        double mae = 0;
        double maxError = 0;

        for (int i = 0; i < targets.Shape[0]; i++)
        {
            double error = targets[new int[] { i, 0 }] - predictions[new int[] { i, 0 }];
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

        // 9. Save the trained model
        Console.WriteLine("\nSaving the trained model...");
        byte[] serializedModel = neuralNetwork.Serialize();
        Console.WriteLine($"Model saved ({serializedModel.Length} bytes)");

        // 10. Make a prediction on a new house
        Console.WriteLine("\nPredicting price for a new house...");

        // New house: 1750 sqft, 3 bedrooms, 2 bathrooms, 5 years old, 6 miles from city center
        var newHouse = new Tensor<double>(new int[] { 1, 5 });
        newHouse[new int[] { 0, 0 }] = 1750;
        newHouse[new int[] { 0, 1 }] = 3;
        newHouse[new int[] { 0, 2 }] = 2;
        newHouse[new int[] { 0, 3 }] = 5;
        newHouse[new int[] { 0, 4 }] = 6;

        // Normalize the new house data using the same means and standard deviations
        var normalizedNewHouse = new Tensor<double>(new int[] { 1, 5 });
        for (int j = 0; j < newHouse.Shape[1]; j++)
        {
            normalizedNewHouse[new int[] { 0, j }] = (newHouse[new int[] { 0, j }] - featureMeans[j]) / featureStds[j];
        }

        // Make prediction
        var normalizedNewPrediction = neuralNetwork.Predict(normalizedNewHouse);

        // Denormalize the prediction
        double newPredictedPrice = normalizedNewPrediction[new int[] { 0, 0 }] * targetStd + targetMean;

        Console.WriteLine("New house features:");
        Console.WriteLine($"- Size: 1750 sq ft");
        Console.WriteLine($"- Bedrooms: 3");
        Console.WriteLine($"- Bathrooms: 2");
        Console.WriteLine($"- Age: 5 years");
        Console.WriteLine($"- Distance to city center: 6 miles");
        Console.WriteLine($"\nPredicted price: ${newPredictedPrice:F0}");

        Console.WriteLine("\nNeural network regression example completed!");
    }
}
