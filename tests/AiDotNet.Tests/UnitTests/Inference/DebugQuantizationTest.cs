using System;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Inference;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Inference;

public class DebugQuantizationTest
{
    [Fact]
    public void Debug_WeightOnlyQuantization()
    {
        Console.WriteLine("DEBUG: Starting Debug_WeightOnlyQuantization");
        var model = CreateTinyDenseModel();

        var input = new AiDotNet.Tensors.LinearAlgebra.Tensor<float>(new[] { 1, 4 });
        for (int i = 0; i < input.Length; i++)
        {
            input[i] = 0.1f * (i + 1);
        }
        Console.WriteLine($"DEBUG: Input: [{string.Join(", ", input.ToArray())}]");

        var baseline = model.Predict(input);
        Console.WriteLine($"DEBUG: Baseline Prediction: [{string.Join(", ", baseline.ToArray())}]");

        var config = new InferenceOptimizationConfig
        {
            EnableKVCache = false,
            EnableFlashAttention = false,
            EnableWeightOnlyQuantization = true
        };

        var optimizer = new InferenceOptimizer<float>(config);
        var (optimized, anyApplied) = optimizer.OptimizeForInference(model, cloneModel: true);

        Console.WriteLine($"DEBUG: Optimizer Applied: {anyApplied}");
        if (anyApplied)
        {
            Console.WriteLine($"DEBUG: Optimized Layers: {string.Join(", ", optimized.Layers.Select(l => l.GetType().Name))}");
        }

        var y = optimized.Predict(input);
        Console.WriteLine($"DEBUG: Quantized Prediction: [{string.Join(", ", y.ToArray())}]");

        for (int i = 0; i < y.Length; i++)
        {
            float diff = Math.Abs(baseline[i] - y[i]);
            Console.WriteLine($"DEBUG: Index {i}: Baseline={baseline[i]}, Quantized={y[i]}, Diff={diff}");
            Assert.True(diff < 1e-1f, $"Mismatch at {i}: {baseline[i]} vs {y[i]}");
        }
    }

    private static NeuralNetworkBase<float> CreateTinyDenseModel()
    {
        const int inSize = 4;
        const int outSize = 3;

        var layers = new System.Collections.Generic.List<AiDotNet.Interfaces.ILayer<float>>
        {
            new InputLayer<float>(inSize),
            new DenseLayer<float>(inSize, outSize, activationFunction: new AiDotNet.ActivationFunctions.IdentityActivation<float>())
        };

        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: inSize,
            outputSize: outSize,
            layers: layers);

        var model = new NeuralNetwork<float>(architecture);

        var p = model.GetParameters();
        var deterministic = new float[p.Length];
        for (int i = 0; i < deterministic.Length; i++)
        {
            deterministic[i] = ((i % 13) - 6) / 6.0f;
        }
        Console.WriteLine($"DEBUG: Model Parameters: [{string.Join(", ", deterministic)}]");
        model.UpdateParameters(new AiDotNet.Tensors.LinearAlgebra.Vector<float>(deterministic));

        return model;
    }
}
