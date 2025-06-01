using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.TransferLearning;
using AiDotNet.TransferLearning.Algorithms;

namespace AiDotNet.Examples;

/// <summary>
/// Example demonstrating transfer learning capabilities in AiDotNet.
/// </summary>
public class TransferLearningExample
{
    public static void Main(string[] args)
    {
        Console.WriteLine("=== AiDotNet Transfer Learning Example ===\n");
        
        // Example 1: Feature Extraction Transfer
        Console.WriteLine("1. Feature Extraction Transfer Learning");
        Console.WriteLine("---------------------------------------");
        RunFeatureExtractionExample();
        
        Console.WriteLine("\n2. Progressive Unfreezing Example");
        Console.WriteLine("---------------------------------");
        RunProgressiveUnfreezingExample();
        
        Console.WriteLine("\n3. Domain Adaptation Example");
        Console.WriteLine("----------------------------");
        RunDomainAdaptationExample();
        
        Console.WriteLine("\n=== Example Complete ===");
        
        if (args.Length == 0)
        {
            Console.WriteLine("\nPress any key to return to the main menu...");
            Console.ReadKey();
        }
    }
    
    private static void RunFeatureExtractionExample()
    {
        // Assume we have a pre-trained model on ImageNet (simulated)
        var sourceModel = CreatePretrainedModel();
        
        // Create a new model for our specific task (e.g., classifying medical images)
        var targetArchitecture = new NeuralNetworkArchitecture<double>(
            NetworkComplexity.Medium,
            NeuralNetworkTaskType.MultiClassClassification
        );
        var targetModel = new TransferNeuralNetwork<double>(targetArchitecture);
        
        // Transfer using feature extraction - freeze all but last 2 layers
        var transferOptions = new TransferLearningOptions<double>
        {
            LayersToFreeze = Enumerable.Range(0, 5).ToList(), // Freeze first 5 layers
            ResetFinalLayers = true,
            TransferredLayerLearningRateScale = 0.1  // Lower learning rate for transferred layers
        };
        
        targetModel.TransferFrom(sourceModel, TransferLearningStrategy.FeatureExtraction, transferOptions);
        
        Console.WriteLine($"Transferred model with {transferOptions.LayersToFreeze.Count} layers to freeze");
        Console.WriteLine($"Frozen layers: {string.Join(", ", targetModel.GetFrozenLayers())}");
        
        // Generate some synthetic data for our new task
        var random = new Random(42);
        var trainingData = GenerateSyntheticData(100, 224 * 224 * 3, 5, random);
        
        // Fine-tune only the unfrozen layers
        var fineTuningOptions = new FineTuningOptions<double>
        {
            InitialLearningRate = 0.001,
            Epochs = 5,
            BatchSize = 16,
            Optimizer = OptimizerType.Adam
        };
        
        Console.WriteLine("\nFine-tuning on new task...");
        targetModel.FineTune(
            trainingData.Select(d => d.Input).ToArray(),
            trainingData.Select(d => d.Output).ToArray(),
            fineTuningOptions
        );
        
        // Test the model
        var testData = GenerateSyntheticData(10, 224 * 224 * 3, 5, random);
        Console.WriteLine("\nTesting transferred model:");
        
        int correct = 0;
        foreach (var (input, expectedOutput) in testData)
        {
            var prediction = targetModel.Predict(input);
            if (GetClass(prediction) == GetClass(expectedOutput))
                correct++;
        }
        
        Console.WriteLine($"Accuracy: {correct}/{testData.Length} ({100.0 * correct / testData.Length:F1}%)");
    }
    
    private static void RunProgressiveUnfreezingExample()
    {
        var sourceModel = CreatePretrainedModel();
        var targetArchitecture = new NeuralNetworkArchitecture<double>(
            NetworkComplexity.Deep,
            NeuralNetworkTaskType.MultiClassClassification
        );
        var targetModel = new TransferNeuralNetwork<double>(targetArchitecture);
        
        // Transfer with progressive unfreezing strategy
        var transferOptions = new TransferLearningOptions<double>
        {
            UseDiscriminativeLearningRates = true
        };
        
        targetModel.TransferFrom(sourceModel, TransferLearningStrategy.ProgressiveUnfreezing, transferOptions);
        
        // Set discriminative learning rates (lower for early layers)
        var layerLearningRates = new Dictionary<int, double>();
        var layerCount = 7; // Estimated layer count for complex architecture
        for (int i = 0; i < layerCount; i++)
        {
            // Earlier layers get smaller learning rates
            layerLearningRates[i] = 0.0001 * Math.Pow(10, (double)i / layerCount);
        }
        targetModel.SetLayerLearningRates(layerLearningRates);
        
        Console.WriteLine("Set discriminative learning rates:");
        foreach (var kvp in layerLearningRates)
        {
            Console.WriteLine($"  Layer {kvp.Key}: {kvp.Value:E3}");
        }
        
        // Generate training data
        var random = new Random(42);
        var trainingData = GenerateSyntheticData(200, 224 * 224 * 3, 10, random);
        
        // Fine-tune with progressive unfreezing
        var fineTuningOptions = new FineTuningOptions<double>
        {
            InitialLearningRate = 0.001,
            Epochs = 10,
            BatchSize = 32,
            UnfreezeGradually = true,
            EpochsPerUnfreeze = 2,
            WarmupEpochs = 1
        };
        
        Console.WriteLine("\nFine-tuning with progressive unfreezing...");
        targetModel.FineTune(
            trainingData.Select(d => d.Input).ToArray(),
            trainingData.Select(d => d.Output).ToArray(),
            fineTuningOptions
        );
        
        var transferInfo = targetModel.GetTransferInfo();
        Console.WriteLine($"\nTransfer completed:");
        Console.WriteLine($"  Strategy: {transferInfo.TransferStrategy}");
        Console.WriteLine($"  Layers transferred: {transferInfo.LayersTransferred}");
        Console.WriteLine($"  Date: {transferInfo.TransferDate:yyyy-MM-dd HH:mm:ss}");
    }
    
    private static void RunDomainAdaptationExample()
    {
        // Simulate transfer between different domains (e.g., synthetic to real images)
        var sourceModel = CreatePretrainedModel();
        var targetArchitecture = new NeuralNetworkArchitecture<double>(
            NetworkComplexity.Simple,
            NeuralNetworkTaskType.MultiClassClassification
        );
        var targetModel = new TransferNeuralNetwork<double>(targetArchitecture);
        
        // Transfer with domain adaptation
        var transferOptions = new TransferLearningOptions<double>
        {
            DomainAdaptationStrength = 0.1
        };
        
        targetModel.TransferFrom(sourceModel, TransferLearningStrategy.DomainAdaptation, transferOptions);
        
        // Generate source domain data (e.g., synthetic images)
        var random = new Random(42);
        var sourceData = GenerateSyntheticData(100, 224 * 224 * 3, 5, random, domainShift: 0.0);
        
        // Generate target domain data (e.g., real images with domain shift)
        var targetData = GenerateSyntheticData(100, 224 * 224 * 3, 5, random, domainShift: 0.5);
        
        Console.WriteLine("Adapting model to target domain using MMD...");
        
        // Perform domain adaptation
        targetModel.AdaptDomain(
            sourceData.Select(d => d.Input).ToArray(),
            targetData.Select(d => d.Input).ToArray(),
            DomainAdaptationMethod.MMD
        );
        
        // Calculate transferability score
        var transferabilityScore = targetModel.GetTransferabilityScore(
            targetData.Select(d => d.Input).Take(20).ToArray(),
            targetData.Select(d => d.Output).Take(20).ToArray()
        );
        
        Console.WriteLine($"Transferability score: {transferabilityScore:F3}");
        
        // Fine-tune on target domain
        var fineTuningOptions = new FineTuningOptions<double>
        {
            InitialLearningRate = 0.0005,
            Epochs = 5,
            BatchSize = 16
        };
        
        Console.WriteLine("\nFine-tuning on target domain...");
        targetModel.FineTune(
            targetData.Select(d => d.Input).ToArray(),
            targetData.Select(d => d.Output).ToArray(),
            fineTuningOptions
        );
        
        Console.WriteLine("Domain adaptation complete!");
    }
    
    // Helper methods
    private static IFullModel<double, Tensor<double>, Tensor<double>> CreatePretrainedModel()
    {
        // Create a simulated pre-trained model
        var architecture = new NeuralNetworkArchitecture<double>(
            NetworkComplexity.Deep,
            NeuralNetworkTaskType.MultiClassClassification
        );
        var model = new FeedForwardNeuralNetwork<double>(architecture);
        
        // Simulate pre-training by setting some weights
        var parameters = model.GetParameters();
        var random = new Random(123);
        for (int i = 0; i < parameters.Length; i++)
        {
            parameters[i] = (random.NextDouble() - 0.5) * 0.1;
        }
        model.SetParameters(parameters);
        
        return model as IFullModel<double, Tensor<double>, Tensor<double>> ?? throw new InvalidOperationException("Failed to create model");
    }
    
    private static (Tensor<double> Input, Tensor<double> Output)[] GenerateSyntheticData(
        int count, int inputSize, int numClasses, Random random, double domainShift = 0.0)
    {
        var data = new (Tensor<double>, Tensor<double>)[count];
        
        for (int i = 0; i < count; i++)
        {
            // Generate random input with optional domain shift
            var input = new double[inputSize];
            for (int j = 0; j < inputSize; j++)
            {
                input[j] = random.NextDouble() + domainShift * (random.NextDouble() - 0.5);
            }
            
            // Generate one-hot encoded output
            var output = new double[numClasses];
            output[random.Next(numClasses)] = 1.0;
            
            // Create tensors with proper shape
            data[i] = (new Tensor<double>(new[] { inputSize }, new Vector<double>(input)), 
                      new Tensor<double>(new[] { numClasses }, new Vector<double>(output)));
        }
        
        return data;
    }
    
    private static int GetClass(Tensor<double> output)
    {
        int maxIndex = 0;
        double maxValue = output[0];
        
        // Access tensor elements directly using indexer
        for (int i = 1; i < output.Shape[0]; i++)
        {
            if (output[i] > maxValue)
            {
                maxValue = output[i];
                maxIndex = i;
            }
        }
        
        return maxIndex;
    }
}