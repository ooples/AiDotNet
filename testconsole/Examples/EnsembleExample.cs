using System;
using System.Linq;
using AiDotNet;
using AiDotNet.Enums;
using AiDotNet.Extensions;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Regression;
using AiDotNet.NeuralNetworks;
using AiDotNet.Ensemble.Strategies;
using AiDotNet.ActivationFunctions;
using AiDotNet.Optimizers;
using AiDotNet.Factories;

namespace TestConsoleApp.Examples;

public static class EnsembleExample
{
    public static void Run()
    {
        Console.WriteLine("=== Ensemble Model Example ===\n");
        
        // Generate synthetic dataset for regression
        Console.WriteLine("Generating synthetic regression dataset...");
        var dataSize = 1000;
        var random = new Random(42);
        var inputs = new Matrix<double>(dataSize, 1);
        var outputs = new Vector<double>(dataSize);
        
        for (int i = 0; i < dataSize; i++)
        {
            double x = (double)i / dataSize * 10 - 5; // Range from -5 to 5
            inputs[i, 0] = x;
            outputs[i] = Math.Sin(x) + 0.1 * random.NextDouble(); // Sine wave with noise
        }
        
        // Split into training and testing sets
        var trainSize = (int)(dataSize * 0.8);
        var testSize = dataSize - trainSize;
        
        // Create training data
        var trainInputs = inputs.GetSubMatrix(0, 0, trainSize, 1);
        var trainOutputs = new Vector<double>(trainSize);
        for (int i = 0; i < trainSize; i++)
        {
            trainOutputs[i] = outputs[i];
        }
        
        // Create testing data
        var testInputs = inputs.GetSubMatrix(trainSize, 0, testSize, 1);
        var testOutputs = new Vector<double>(testSize);
        for (int i = 0; i < testSize; i++)
        {
            testOutputs[i] = outputs[trainSize + i];
        }
        
        Console.WriteLine($"Training samples: {trainSize}");
        Console.WriteLine($"Testing samples: {testSize}\n");
        
        // Create individual models
        Console.WriteLine("Creating individual models for the ensemble...\n");
        
        // Model 1: Simple Neural Network
        Console.WriteLine("1. Creating Neural Network model...");
        var nnOptions = new NeuralNetworkRegressionOptions<double, Matrix<double>, Vector<double>>
        {
            LayerSizes = new System.Collections.Generic.List<int> { 1, 10, 5, 1 },
            HiddenActivationFunction = new TanhActivation<double>(),
            OutputActivationFunction = new IdentityActivation<double>(),
            Epochs = 50,
            BatchSize = 32,
            LearningRate = 0.01
        };
        // The optimizer will be created based on the model's needs
        var nnModel = new NeuralNetworkRegression<double>(nnOptions);
        
        // Model 2: Polynomial Regression
        Console.WriteLine("2. Creating Polynomial Regression model...");
        var polyOptions = new PolynomialRegressionOptions<double>
        {
            Degree = 3
        };
        var polyModel = new PolynomialRegression<double>(polyOptions);
        
        // Model 3: Random Forest
        Console.WriteLine("3. Creating Random Forest model...");
        var rfOptions = new RandomForestRegressionOptions
        {
            NumberOfTrees = 50,
            MaxDepth = 5
        };
        var rfModel = new RandomForestRegression<double>(rfOptions);
        
        // Train individual models
        Console.WriteLine("\nTraining individual models...");
        
        Console.WriteLine("Training Neural Network...");
        nnModel.Train(trainInputs, trainOutputs);
        
        Console.WriteLine("Training Polynomial Regression...");
        polyModel.Train(trainInputs, trainOutputs);
        
        Console.WriteLine("Training Random Forest...");
        rfModel.Train(trainInputs, trainOutputs);
        
        // Evaluate individual models
        Console.WriteLine("\nEvaluating individual models on test set:");
        EvaluateModel("Neural Network", nnModel, testInputs, testOutputs);
        EvaluateModel("Polynomial Regression", polyModel, testInputs, testOutputs);
        EvaluateModel("Random Forest", rfModel, testInputs, testOutputs);
        
        // Create ensemble models with different strategies
        Console.WriteLine("\n=== Creating Ensemble Models ===\n");
        
        // 1. Voting Ensemble (Average)
        Console.WriteLine("1. Creating Voting Ensemble (Average)...");
        var votingOptions = new VotingEnsembleOptions<double>
        {
            VotingType = VotingType.Soft,
            Strategy = EnsembleStrategy.Average,
            MinModels = 2,
            MaxModels = 10
        };
        var votingEnsemble = new VotingEnsemble<double, Matrix<double>, Vector<double>>(votingOptions);
        
        votingEnsemble.AddModel(nnModel, 1.0);
        votingEnsemble.AddModel(polyModel, 1.0);
        votingEnsemble.AddModel(rfModel, 1.0);
        
        // 2. Weighted Voting Ensemble
        Console.WriteLine("2. Creating Weighted Voting Ensemble...");
        var weightedOptions = new VotingEnsembleOptions<double>
        {
            VotingType = VotingType.Weighted,
            Strategy = EnsembleStrategy.WeightedAverage,
            MinModels = 2,
            MaxModels = 10
        };
        var weightedEnsemble = new VotingEnsemble<double, Matrix<double>, Vector<double>>(weightedOptions);
        
        // Assign weights based on individual model performance (in practice, use validation set)
        weightedEnsemble.AddModel(nnModel, 0.5);    // Higher weight for NN
        weightedEnsemble.AddModel(polyModel, 0.3);
        weightedEnsemble.AddModel(rfModel, 0.2);
        
        // 3. Voting Ensemble with different strategies
        Console.WriteLine("3. Creating Voting Ensemble with Median strategy...");
        var medianOptions = new VotingEnsembleOptions<double>
        {
            VotingType = VotingType.Soft,
            Strategy = EnsembleStrategy.Median,
            MinModels = 2,
            MaxModels = 10
        };
        var medianEnsemble = new VotingEnsemble<double, Matrix<double>, Vector<double>>(medianOptions);
        
        medianEnsemble.AddModel(nnModel, 1.0);
        medianEnsemble.AddModel(polyModel, 1.0);
        medianEnsemble.AddModel(rfModel, 1.0);
        
        // Evaluate ensemble models
        Console.WriteLine("\n=== Evaluating Ensemble Models ===");
        EvaluateModel("Voting Ensemble (Average)", votingEnsemble, testInputs, testOutputs);
        EvaluateModel("Weighted Voting Ensemble", weightedEnsemble, testInputs, testOutputs);
        EvaluateModel("Voting Ensemble (Median)", medianEnsemble, testInputs, testOutputs);
        
        // Demonstrate individual predictions
        Console.WriteLine("\n=== Sample Predictions ===");
        var sampleInput = new Matrix<double>(1, 1);
        sampleInput[0, 0] = 0.0; // x = 0, expected: sin(0) = 0
        
        Console.WriteLine($"\nInput: x = {sampleInput[0, 0]}");
        Console.WriteLine($"Expected: {Math.Sin(sampleInput[0, 0]):F4}");
        Console.WriteLine("\nIndividual model predictions:");
        
        var nnPred = nnModel.Predict(sampleInput);
        var polyPred = polyModel.Predict(sampleInput);
        var rfPred = rfModel.Predict(sampleInput);
        
        Console.WriteLine($"  Neural Network: {nnPred[0]:F4}");
        Console.WriteLine($"  Polynomial Regression: {polyPred[0]:F4}");
        Console.WriteLine($"  Random Forest: {rfPred[0]:F4}");
        
        Console.WriteLine("\nEnsemble predictions:");
        var votingPred = votingEnsemble.Predict(sampleInput);
        var weightedPred = weightedEnsemble.Predict(sampleInput);
        var medianPred = medianEnsemble.Predict(sampleInput);
        
        Console.WriteLine($"  Voting (Average): {votingPred[0]:F4}");
        Console.WriteLine($"  Weighted Voting: {weightedPred[0]:F4}");
        Console.WriteLine($"  Voting (Median): {medianPred[0]:F4}");
        
        // Show individual predictions from ensemble
        Console.WriteLine("\n=== Individual Predictions from Ensemble ===");
        var individualPreds = votingEnsemble.GetIndividualPredictions(sampleInput);
        Console.WriteLine("Predictions from each model in the ensemble:");
        for (int i = 0; i < individualPreds.Count; i++)
        {
            Console.WriteLine($"  Model {i + 1}: {individualPreds[i][0]:F4}");
        }
        
        // Demonstrate dynamic weight adjustment
        Console.WriteLine("\n=== Dynamic Weight Adjustment ===");
        Console.WriteLine("Original weights in weighted ensemble:");
        var weights = weightedEnsemble.ModelWeights;
        for (int i = 0; i < weights.Length; i++)
        {
            Console.WriteLine($"  Model {i + 1}: {weights[i]:F4}");
        }
        
        Console.WriteLine("\nUpdating weights...");
        var newWeights = new Vector<double>(new[] { 0.2, 0.3, 0.5 }); // Favor Random Forest
        weightedEnsemble.UpdateWeights(newWeights);
        
        Console.WriteLine("New weights:");
        weights = weightedEnsemble.ModelWeights;
        for (int i = 0; i < weights.Length; i++)
        {
            Console.WriteLine($"  Model {i + 1}: {weights[i]:F4}");
        }
        
        var newPred = weightedEnsemble.Predict(sampleInput);
        Console.WriteLine($"\nPrediction with new weights: {newPred[0]:F4}");
        
        Console.WriteLine("\n=== Ensemble Example Completed ===");
    }
    
    private static void EvaluateModel(string modelName, IFullModel<double, Matrix<double>, Vector<double>> model, 
        Matrix<double> testInputs, Vector<double> testOutputs)
    {
        double mse = 0;
        double mae = 0;
        
        // Make batch prediction
        var predictions = model.Predict(testInputs);
        
        for (int i = 0; i < testOutputs.Length; i++)
        {
            double error = testOutputs[i] - predictions[i];
            mse += error * error;
            mae += Math.Abs(error);
        }
        
        mse /= testOutputs.Length;
        mae /= testOutputs.Length;
        double rmse = Math.Sqrt(mse);
        
        Console.WriteLine($"{modelName,-30} - MSE: {mse:F6}, RMSE: {rmse:F6}, MAE: {mae:F6}");
    }
}