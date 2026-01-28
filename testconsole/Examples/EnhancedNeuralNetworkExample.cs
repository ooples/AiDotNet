using AiDotNet.ActivationFunctions;
using AiDotNet.DataProcessor;
using AiDotNet.Enums;
using AiDotNet.FeatureSelectors;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Normalizers;
using AiDotNet.AnomalyDetection;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNetTestConsole.Examples;

/// <summary>
/// Enhanced real-world example demonstrating neural networks for customer churn prediction.
/// </summary>
public class EnhancedNeuralNetworkExample
{
    // Customer Churn Prediction constants
    private const int NUMBER_OF_FEATURES = 10;
    private const int NUMBER_OF_CLASSES = 2; // Churn or Not Churn

    public void RunExample()
    {
        Console.WriteLine("Enhanced Neural Network Example - Customer Churn Prediction");
        Console.WriteLine("==========================================================\n");
        Console.WriteLine("This example demonstrates using neural networks to predict customer churn.");
        Console.WriteLine("We'll analyze customer data to predict which customers are likely to cancel their services.\n");

        try
        {
            // 1. Generate synthetic customer data
            Console.WriteLine("Generating synthetic customer data...");
            var (customerData, churnLabels, featureNames) = GenerateCustomerData(1000);

            Console.WriteLine($"Generated data for {customerData.Length} customers with the following features:");
            foreach (var feature in featureNames)
            {
                Console.WriteLine($"- {feature}");
            }

            // 2. Analyze class distribution
            int churnCount = 0;
            int nonChurnCount = 0;
            for (int i = 0; i < churnLabels.Length; i++)
            {
                if (churnLabels[i] == 1)
                {
                    churnCount++;
                }
                else
                {
                    nonChurnCount++;
                }
            }

            Console.WriteLine($"\nClass distribution:");
            Console.WriteLine($"- Churned customers: {churnCount} ({(double)churnCount / churnLabels.Length:P1})");
            Console.WriteLine($"- Retained customers: {nonChurnCount} ({(double)nonChurnCount / churnLabels.Length:P1})");

            // 3. Convert to matrix format for normalization
            Console.WriteLine("\nPreparing data for preprocessing...");

            // Create feature matrix
            var features = new Matrix<double>(customerData.Length, NUMBER_OF_FEATURES);
            for (int i = 0; i < customerData.Length; i++)
            {
                for (int j = 0; j < NUMBER_OF_FEATURES; j++)
                {
                    features[i, j] = customerData[i][j];
                }
            }

            // 4. Setup normalization
            Console.WriteLine("Setting up data normalization...");

            var normalizer = new NoNormalizer<double, Matrix<double>, Vector<double>>();

            // Use a dummy vector just to satisfy the interface (we don't need to normalize labels)
            var dummyVector = new Vector<double>(customerData.Length);

            // Normalize the features
            var (normalizedFeatures, normInfo) = normalizer.NormalizeInput(features);

            // 5. Split data into training, validation, and test sets (70%/15%/15%)
            Console.WriteLine("\nSplitting data into training, validation, and test sets...");

            // Shuffle data
            Random random = RandomHelper.CreateSeededRandom(42);
            var indices = Enumerable.Range(0, customerData.Length).ToArray();
            for (int i = 0; i < indices.Length; i++)
            {
                int j = random.Next(i, indices.Length);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }

            int trainSize = (int)(customerData.Length * 0.7);
            int valSize = (int)(customerData.Length * 0.15);
            int testSize = customerData.Length - trainSize - valSize;

            // Create tensor arrays
            var trainFeatures = new double[trainSize, NUMBER_OF_FEATURES];
            var trainLabels = new double[trainSize, NUMBER_OF_CLASSES];

            var valFeatures = new double[valSize, NUMBER_OF_FEATURES];
            var valLabels = new double[valSize, NUMBER_OF_CLASSES];

            var testFeatures = new double[testSize, NUMBER_OF_FEATURES];
            var testLabels = new double[testSize, NUMBER_OF_CLASSES];

            // Assign data to sets using normalized features
            for (int i = 0; i < trainSize; i++)
            {
                for (int j = 0; j < NUMBER_OF_FEATURES; j++)
                {
                    trainFeatures[i, j] = normalizedFeatures[indices[i], j];
                }

                trainLabels[i, 0] = churnLabels[indices[i]] == 0 ? 1 : 0; // Not churned
                trainLabels[i, 1] = churnLabels[indices[i]] == 1 ? 1 : 0; // Churned
            }

            for (int i = 0; i < valSize; i++)
            {
                for (int j = 0; j < NUMBER_OF_FEATURES; j++)
                {
                    valFeatures[i, j] = normalizedFeatures[indices[trainSize + i], j];
                }

                valLabels[i, 0] = churnLabels[indices[trainSize + i]] == 0 ? 1 : 0;
                valLabels[i, 1] = churnLabels[indices[trainSize + i]] == 1 ? 1 : 0;
            }

            for (int i = 0; i < testSize; i++)
            {
                for (int j = 0; j < NUMBER_OF_FEATURES; j++)
                {
                    testFeatures[i, j] = normalizedFeatures[indices[trainSize + valSize + i], j];
                }

                testLabels[i, 0] = churnLabels[indices[trainSize + valSize + i]] == 0 ? 1 : 0;
                testLabels[i, 1] = churnLabels[indices[trainSize + valSize + i]] == 1 ? 1 : 0;
            }

            Console.WriteLine($"Data split into:");
            Console.WriteLine($"- Training set: {trainSize} customers");
            Console.WriteLine($"- Validation set: {valSize} customers");
            Console.WriteLine($"- Test set: {testSize} customers");

            // 6. Create tensors
            Console.WriteLine("Converting data to tensors...");

            var trainFeaturesTensor = new Tensor<double>(new int[] { trainSize, NUMBER_OF_FEATURES });
            var trainLabelsTensor = new Tensor<double>(new int[] { trainSize, NUMBER_OF_CLASSES });

            var valFeaturesTensor = new Tensor<double>(new int[] { valSize, NUMBER_OF_FEATURES });
            var valLabelsTensor = new Tensor<double>(new int[] { valSize, NUMBER_OF_CLASSES });

            var testFeaturesTensor = new Tensor<double>(new int[] { testSize, NUMBER_OF_FEATURES });
            var testLabelsTensor = new Tensor<double>(new int[] { testSize, NUMBER_OF_CLASSES });

            // Fill the tensors with data
            for (int i = 0; i < trainSize; i++)
            {
                for (int j = 0; j < NUMBER_OF_FEATURES; j++)
                {
                    trainFeaturesTensor[new int[] { i, j }] = trainFeatures[i, j];
                }
                for (int j = 0; j < NUMBER_OF_CLASSES; j++)
                {
                    trainLabelsTensor[new int[] { i, j }] = trainLabels[i, j];
                }
            }

            for (int i = 0; i < valSize; i++)
            {
                for (int j = 0; j < NUMBER_OF_FEATURES; j++)
                {
                    valFeaturesTensor[new int[] { i, j }] = valFeatures[i, j];
                }
                for (int j = 0; j < NUMBER_OF_CLASSES; j++)
                {
                    valLabelsTensor[new int[] { i, j }] = valLabels[i, j];
                }
            }

            for (int i = 0; i < testSize; i++)
            {
                for (int j = 0; j < NUMBER_OF_FEATURES; j++)
                {
                    testFeaturesTensor[new int[] { i, j }] = testFeatures[i, j];
                }
                for (int j = 0; j < NUMBER_OF_CLASSES; j++)
                {
                    testLabelsTensor[new int[] { i, j }] = testLabels[i, j];
                }
            }

            // 7. Create neural network architectures
            Console.WriteLine("\nCreating neural network architectures...");

            // Standard architecture
            var standardArchitecture = new NeuralNetworkArchitecture<double>(
                inputFeatures: NUMBER_OF_FEATURES,
                numClasses: NUMBER_OF_CLASSES,
                complexity: NetworkComplexity.Medium
            );

            var deepArchitecture = new NeuralNetworkArchitecture<double>(
                InputType.TwoDimensional,  // For tabular data, each row is a sample
                NeuralNetworkTaskType.MultiClassClassification,
                NetworkComplexity.VeryDeep,
                inputHeight: 0,            // Will be determined by the number of samples
                inputWidth: NUMBER_OF_FEATURES,
                outputSize: NUMBER_OF_CLASSES,
                layers: CreateCustomLayers(NUMBER_OF_FEATURES, NUMBER_OF_CLASSES)
            );

            // 8. Initialize neural networks
            Console.WriteLine("Initializing neural networks...");

            var standardNetwork = new NeuralNetwork<double>(standardArchitecture);
            var deepNetwork = new NeuralNetwork<double>(deepArchitecture);

            // 9. Train the networks with early stopping
            Console.WriteLine("\nTraining the standard neural network with early stopping...");
            TrainNetworkWithEarlyStopping(
                standardNetwork,
                trainFeaturesTensor,
                trainLabelsTensor,
                valFeaturesTensor,
                valLabelsTensor,
                maxEpochs: 500,
                patience: 20,
                "Standard Network"
            );

            Console.WriteLine("\nTraining the deep neural network with early stopping...");
            TrainNetworkWithEarlyStopping(
                deepNetwork,
                trainFeaturesTensor,
                trainLabelsTensor,
                valFeaturesTensor,
                valLabelsTensor,
                maxEpochs: 500,
                patience: 20,
                "Deep Network"
            );

            // 10. Evaluate networks on test set
            Console.WriteLine("\nEvaluating neural networks on test set:");

            Console.WriteLine("\nStandard Neural Network performance:");
            EvaluateModel(standardNetwork, testFeaturesTensor, testLabelsTensor);

            Console.WriteLine("\nDeep Neural Network performance:");
            EvaluateModel(deepNetwork, testFeaturesTensor, testLabelsTensor);

            // 11. Analyze which features are most predictive of churn
            Console.WriteLine("\nAnalyzing feature importance...");
            AnalyzeFeatureImportance(deepNetwork, testFeaturesTensor, testLabelsTensor, featureNames);

            // 12. Save the better model
            Console.WriteLine("\nSaving the deep neural network...");
            byte[] serializedModel = deepNetwork.Serialize();
            File.WriteAllBytes("customer_churn_model.bin", serializedModel);
            Console.WriteLine($"Model saved (size: {serializedModel.Length:N0} bytes)");

            // 13. Create a churn prediction for a new customer
            Console.WriteLine("\nPredicting churn for sample customers:");

            var sampleCustomers = CreateSampleCustomers();

            for (int i = 0; i < sampleCustomers.Length; i++)
            {
                // Create a tensor for the customer
                var customerTensor = new Tensor<double>(new int[] { 1, NUMBER_OF_FEATURES });

                // Normalize features using stored normalization info
                for (int j = 0; j < NUMBER_OF_FEATURES; j++)
                {
                    // Apply the same normalization used during training
                    double normalizedValue = (sampleCustomers[i].Features[j] - normInfo[j].Mean) / normInfo[j].StdDev;
                    customerTensor[new int[] { 0, j }] = normalizedValue;
                }

                // Make prediction
                var prediction = deepNetwork.Predict(customerTensor);

                // Get churn probability
                double churnProbability = prediction[new int[] { 0, 1 }];

                // Print customer details and prediction
                Console.WriteLine($"\nCustomer {i + 1}:");
                Console.WriteLine($"- Description: {sampleCustomers[i].Description}");
                Console.WriteLine($"- Contract Length: {sampleCustomers[i].Features[0]} months");
                Console.WriteLine($"- Monthly Charge: ${sampleCustomers[i].Features[1]:F2}");
                Console.WriteLine($"- Total Spend: ${sampleCustomers[i].Features[2]:F2}");
                Console.WriteLine($"- Tenure: {sampleCustomers[i].Features[3]} months");
                Console.WriteLine($"- Age: {sampleCustomers[i].Features[4]} years");
                Console.WriteLine($"- Service Calls: {sampleCustomers[i].Features[5]}");
                Console.WriteLine($"- Churn Probability: {churnProbability:P1}");

                // Classify risk
                string riskLevel;
                if (churnProbability < 0.3) riskLevel = "Low Risk";
                else if (churnProbability < 0.7) riskLevel = "Medium Risk";
                else riskLevel = "High Risk";

                Console.WriteLine($"- Risk Level: {riskLevel}");

                // Suggest retention strategies based on customer profile
                if (churnProbability > 0.5)
                {
                    Console.WriteLine("- Recommended Retention Strategy:");

                    // Different strategies based on customer features
                    if (sampleCustomers[i].Features[0] < 6)  // Short contract
                    {
                        Console.WriteLine("  * Offer a discount for upgrading to 12-month contract");
                    }

                    if (sampleCustomers[i].Features[1] > 80)  // High monthly charge
                    {
                        Console.WriteLine("  * Provide a 3-month promotional rate of 15% off");
                    }

                    if (sampleCustomers[i].Features[5] > 2)  // Multiple service calls
                    {
                        Console.WriteLine("  * Schedule a customer success call to address service issues");
                    }

                    if (sampleCustomers[i].Features[3] < 12)  // Low tenure
                    {
                        Console.WriteLine("  * Send a customer satisfaction survey with incentive");
                    }
                }
            }

            Console.WriteLine("\nEnhanced neural network example completed!");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }
    }

    // Helper method to create a custom layer configuration
    private List<ILayer<double>> CreateCustomLayers(int inputFeatures, int numClasses)
    {
        // Create a list of layers with the custom architecture: [input -> 32 -> 64 -> 32 -> 16 -> output]
        var layers = new List<ILayer<double>>
        {
            // First hidden layer: inputFeatures -> 32
            new DenseLayer<double>(inputFeatures, 32, new ReLUActivation<double>() as IActivationFunction<double>),

            // Second hidden layer: 32 -> 64
            new DenseLayer<double>(32, 64, new ReLUActivation<double>() as IActivationFunction<double>),

            // Third hidden layer: 64 -> 32
            new DenseLayer<double>(64, 32, new ReLUActivation<double>() as IActivationFunction<double>),

            // Fourth hidden layer: 32 -> 16
            new DenseLayer<double>(16, 16, new ReLUActivation<double>() as IActivationFunction<double>),

            // Output layer: 16 -> numClasses (with Softmax for classification)
            new DenseLayer<double>(16, numClasses, new SoftmaxActivation<double>() as IActivationFunction<double>)
        };

        return layers;
    }

    // Helper method to train a neural network with early stopping
    private void TrainNetworkWithEarlyStopping(
        NeuralNetwork<double> network,
        Tensor<double> trainFeatures,
        Tensor<double> trainLabels,
        Tensor<double> valFeatures,
        Tensor<double> valLabels,
        int maxEpochs,
        int patience,
        string networkName)
    {
        double bestValLoss = double.MaxValue;
        int epochsSinceImprovement = 0;
        int bestEpoch = 0;

        for (int epoch = 0; epoch < maxEpochs; epoch++)
        {
            // Train for one epoch
            network.Train(trainFeatures, trainLabels);
            double trainLoss = network.GetLastLoss();

            // Evaluate on validation set
            var valPredictions = network.Predict(valFeatures);
            double valLoss = CalculateLoss(valPredictions, valLabels);

            // Print progress
            if (epoch % 10 == 0 || epoch == maxEpochs - 1 || epochsSinceImprovement >= patience)
            {
                Console.WriteLine($"Epoch {epoch}: Training Loss = {trainLoss:F4}, Validation Loss = {valLoss:F4}");
            }

            // Check if validation loss improved
            if (valLoss < bestValLoss)
            {
                bestValLoss = valLoss;
                epochsSinceImprovement = 0;
                bestEpoch = epoch;
            }
            else
            {
                epochsSinceImprovement++;

                // Early stopping
                if (epochsSinceImprovement >= patience)
                {
                    Console.WriteLine($"Early stopping at epoch {epoch}. Best validation loss at epoch {bestEpoch}.");
                    break;
                }
            }
        }

        Console.WriteLine($"{networkName} training completed with best validation loss: {bestValLoss:F4}");
    }

    // Helper method to calculate loss (cross-entropy)
    private double CalculateLoss(Tensor<double> predictions, Tensor<double> labels)
    {
        double loss = 0;
        int count = 0;

        for (int i = 0; i < predictions.Shape[0]; i++)
        {
            for (int j = 0; j < predictions.Shape[1]; j++)
            {
                double p = Math.Max(1e-15, Math.Min(1 - 1e-15, predictions[new int[] { i, j }]));
                double y = labels[new int[] { i, j }];
                loss -= (y * Math.Log(p) + (1 - y) * Math.Log(1 - p));
                count++;
            }
        }

        return loss / count;
    }

    // Helper method to evaluate model performance
    private void EvaluateModel(NeuralNetwork<double> network, Tensor<double> features, Tensor<double> labels)
    {
        // Make predictions
        var predictions = network.Predict(features);

        // Calculate metrics
        int totalSamples = features.Shape[0];
        int correctPredictions = 0;
        int truePositives = 0;
        int falsePositives = 0;
        int trueNegatives = 0;
        int falseNegatives = 0;

        for (int i = 0; i < totalSamples; i++)
        {
            // Get predicted and actual class
            int predictedClass = predictions[new int[] { i, 0 }] > predictions[new int[] { i, 1 }] ? 0 : 1;
            int actualClass = labels[new int[] { i, 0 }] > labels[new int[] { i, 1 }] ? 0 : 1;

            // Update metrics
            if (predictedClass == actualClass)
            {
                correctPredictions++;
            }

            if (predictedClass == 1 && actualClass == 1)
            {
                truePositives++;
            }
            else if (predictedClass == 1 && actualClass == 0)
            {
                falsePositives++;
            }
            else if (predictedClass == 0 && actualClass == 0)
            {
                trueNegatives++;
            }
            else // predictedClass == 0 && actualClass == 1
            {
                falseNegatives++;
            }
        }

        // Calculate more metrics
        double accuracy = (double)correctPredictions / totalSamples;
        double precision = truePositives > 0 ? (double)truePositives / (truePositives + falsePositives) : 0;
        double recall = truePositives > 0 ? (double)truePositives / (truePositives + falseNegatives) : 0;
        double f1Score = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0;

        // Print metrics
        Console.WriteLine($"- Accuracy: {accuracy:P2}");
        Console.WriteLine($"- Precision: {precision:P2}");
        Console.WriteLine($"- Recall: {recall:P2}");
        Console.WriteLine($"- F1 Score: {f1Score:P2}");
        Console.WriteLine($"- Confusion Matrix:");
        Console.WriteLine($"  * True Positives: {truePositives}");
        Console.WriteLine($"  * False Positives: {falsePositives}");
        Console.WriteLine($"  * True Negatives: {trueNegatives}");
        Console.WriteLine($"  * False Negatives: {falseNegatives}");
    }

    // Helper method to analyze feature importance
    private void AnalyzeFeatureImportance(
        NeuralNetwork<double> network,
        Tensor<double> features,
        Tensor<double> labels,
        string[] featureNames)
    {
        // Calculate baseline predictions
        var baselinePredictions = network.Predict(features);

        // Store feature importance scores
        var featureImportance = new Dictionary<string, double>();

        // For each feature, calculate the impact of permuting it
        for (int j = 0; j < NUMBER_OF_FEATURES; j++)
        {
            // Create a copy of the features
            var permutedFeatures = new Tensor<double>(features.Shape);
            for (int i = 0; i < features.Shape[0]; i++)
            {
                for (int k = 0; k < features.Shape[1]; k++)
                {
                    permutedFeatures[new int[] { i, k }] = features[new int[] { i, k }];
                }
            }

            // Permute the current feature
            var random = RandomHelper.CreateSeededRandom(42);
            var indices = Enumerable.Range(0, features.Shape[0]).ToArray();
            for (int i = 0; i < indices.Length; i++)
            {
                int r = random.Next(i, indices.Length);
                (indices[i], indices[r]) = (indices[r], indices[i]);
            }

            for (int i = 0; i < features.Shape[0]; i++)
            {
                permutedFeatures[new int[] { i, j }] = features[new int[] { indices[i], j }];
            }

            // Calculate predictions with permuted feature
            var permutedPredictions = network.Predict(permutedFeatures);

            // Calculate the difference in accuracy
            int baselineCorrect = 0;
            int permutedCorrect = 0;

            for (int i = 0; i < features.Shape[0]; i++)
            {
                int baselinePredicted = baselinePredictions[new int[] { i, 0 }] > baselinePredictions[new int[] { i, 1 }] ? 0 : 1;
                int permutedPredicted = permutedPredictions[new int[] { i, 0 }] > permutedPredictions[new int[] { i, 1 }] ? 0 : 1;
                int actual = labels[new int[] { i, 0 }] > labels[new int[] { i, 1 }] ? 0 : 1;

                if (baselinePredicted == actual)
                {
                    baselineCorrect++;
                }

                if (permutedPredicted == actual)
                {
                    permutedCorrect++;
                }
            }

            double baselineAccuracy = (double)baselineCorrect / features.Shape[0];
            double permutedAccuracy = (double)permutedCorrect / features.Shape[0];

            // Feature importance is the decrease in accuracy when the feature is permuted
            double importance = baselineAccuracy - permutedAccuracy;
            featureImportance[featureNames[j]] = importance;
        }

        // Sort features by importance
        var sortedFeatures = featureImportance.OrderByDescending(x => x.Value).ToList();

        // Print feature importance
        Console.WriteLine("\nFeature Importance (based on permutation importance):");
        for (int i = 0; i < sortedFeatures.Count; i++)
        {
            Console.WriteLine($"{i + 1}. {sortedFeatures[i].Key}: {sortedFeatures[i].Value:F4}");
        }
    }

    // Helper method to generate synthetic customer data
    private (double[][] CustomerData, int[] ChurnLabels, string[] FeatureNames) GenerateCustomerData(int numSamples)
    {
        Random random = RandomHelper.CreateSeededRandom(42);

        // Define feature names
        string[] featureNames =
        {
            "ContractLength",      // Length of contract in months
            "MonthlyCharge",       // Monthly charges in dollars
            "TotalSpend",          // Total amount spent with company
            "Tenure",              // How long they've been a customer in months
            "Age",                 // Customer age
            "ServiceCalls",        // Number of service calls in past year
            "HasMultipleLines",    // Whether they have multiple phone lines
            "HasOnlineBackup",     // Whether they have online backup service
            "HasInternetService",  // Whether they have internet service
            "HasPaperlessBilling"  // Whether they use paperless billing
        };

        // Generate customer data
        double[][] customerData = new double[numSamples][];
        int[] churnLabels = new int[numSamples];

        for (int i = 0; i < numSamples; i++)
        {
            customerData[i] = new double[NUMBER_OF_FEATURES];

            // Generate feature values
            // Contract length (1, 12, 24, or 36 months)
            int[] contractOptions = { 1, 12, 24, 36 };
            customerData[i][0] = contractOptions[random.Next(contractOptions.Length)];

            // Monthly charge ($30-$120)
            customerData[i][1] = 30 + random.NextDouble() * 90;

            // Tenure (1-60 months)
            customerData[i][3] = random.Next(1, 61);

            // Total spend
            customerData[i][2] = customerData[i][1] * customerData[i][3] * (1 - random.NextDouble() * 0.1);

            // Age (18-80)
            customerData[i][4] = random.Next(18, 81);

            // Service calls (0-10)
            customerData[i][5] = random.Next(0, 11);

            // Boolean features (0 or 1)
            customerData[i][6] = random.NextDouble() > 0.7 ? 1 : 0;
            customerData[i][7] = random.NextDouble() > 0.5 ? 1 : 0;
            customerData[i][8] = random.NextDouble() > 0.2 ? 1 : 0;
            customerData[i][9] = random.NextDouble() > 0.4 ? 1 : 0;

            // Determine churn status based on features
            double churnProbability = 0.1;  // Base probability

            // Contract length strongly affects churn (shorter contracts have higher churn)
            if (customerData[i][0] == 1) churnProbability += 0.3;
            else if (customerData[i][0] == 12) churnProbability += 0.1;

            // Higher monthly charges increase churn
            churnProbability += (customerData[i][1] - 30) / 300;

            // Longer tenure decreases churn
            churnProbability -= customerData[i][3] / 200;

            // More service calls increase churn
            churnProbability += customerData[i][5] * 0.03;

            // Age slightly affects churn (younger customers churn more)
            churnProbability += (80 - customerData[i][4]) / 400;

            // Digital engagement decreases churn
            if (customerData[i][7] == 1) churnProbability -= 0.05;  // Online backup
            if (customerData[i][9] == 1) churnProbability -= 0.05;  // Paperless billing

            // Ensure probability is between 0 and 1
            churnProbability = Math.Max(0, Math.Min(1, churnProbability));

            // Assign churn label
            churnLabels[i] = random.NextDouble() < churnProbability ? 1 : 0;
        }

        return (customerData, churnLabels, featureNames);
    }

    // Structure to represent a customer for prediction
    private struct Customer
    {
        public double[] Features;
        public string Description;
    }

    // Helper method to create sample customers for prediction
    private Customer[] CreateSampleCustomers()
    {
        return new Customer[]
        {
            new Customer
            {
                Features = new double[] { 1, 99.95, 199.90, 2, 27, 3, 0, 0, 1, 1 },
                Description = "Young customer with high monthly charge and month-to-month contract"
            },
            new Customer
            {
                Features = new double[] { 24, 59.95, 1375.00, 24, 45, 1, 1, 1, 1, 1 },
                Description = "Mid-age customer with medium charge and 2-year contract"
            },
            new Customer
            {
                Features = new double[] { 12, 79.95, 850.00, 11, 32, 4, 1, 0, 1, 0 },
                Description = "Customer with multiple service calls and no paperless billing"
            }
        };
    }
}
