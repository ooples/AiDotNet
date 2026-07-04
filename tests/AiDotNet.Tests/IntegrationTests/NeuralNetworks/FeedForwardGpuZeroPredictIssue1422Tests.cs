// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.ActivationFunctions;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// #1422: a <see cref="FeedForwardNeuralNetwork{T}"/> with an IDENTITY output head trained on
/// <c>y = 3x + 1.5</c> converged (loss ~0.1) but every GPU inference call returned all zeros — the
/// same linear/None output-head class as #1629. This trains the exact repro on real GPU hardware and
/// asserts inference is non-zero and tracks the target. Skips when no GPU backend is available.
/// </summary>
public class FeedForwardGpuZeroPredictIssue1422Tests
{
    private const int ReproSeed = 1422;

    private readonly ITestOutputHelper _output;
    public FeedForwardGpuZeroPredictIssue1422Tests(ITestOutputHelper output) => _output = output;

    [Fact(Timeout = 240000)]
    public async Task IdentityHead_GpuInference_IsNonZero_AfterTraining()
    {
        await Task.Yield();
        DirectGpuTensorEngine? gpu = null;
        try
        {
            gpu = new DirectGpuTensorEngine();
        }
        catch (Exception ex)
        {
            _output.WriteLine($"GPU engine init failed; skipping #1422 GPU repro. {ex}");
        }

        if (gpu is null || !gpu.SupportsGpu)
        {
            _output.WriteLine("No GPU backend available — skipping #1422 GPU repro.");
            return;
        }

        var previous = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = gpu;

            // y = 3x + 1.5 — 40 samples
            var featureData = new double[40, 1];
            var labelData = new double[40, 1];
            for (int i = 0; i < 40; i++)
            {
                featureData[i, 0] = i * 0.25;
                labelData[i, 0] = 3.0 * featureData[i, 0] + 1.5;
            }
            var featureTensor = Tensor<double>.FromMatrix(new Matrix<double>(featureData));
            var labelTensor = Tensor<double>.FromMatrix(new Matrix<double>(labelData));

            var layers = new List<ILayer<double>>
            {
                new DenseLayer<double>(outputSize: 8, activationFunction: new ReLUActivation<double>())
                {
                    RandomSeed = ReproSeed + 1,
                },
                new DenseLayer<double>(outputSize: 8, activationFunction: new ReLUActivation<double>())
                {
                    RandomSeed = ReproSeed + 2,
                },
                new DenseLayer<double>(outputSize: 1, activationFunction: new IdentityActivation<double>())
                {
                    RandomSeed = ReproSeed + 3,
                },
            };
            var arch = new NeuralNetworkArchitecture<double>(
                inputType: InputType.OneDimensional,
                taskType: NeuralNetworkTaskType.Regression,
                complexity: NetworkComplexity.Simple,
                inputSize: 1,
                layers: layers)
            {
                RandomSeed = ReproSeed,
            };
            var nn = new FeedForwardNeuralNetwork<double>(arch);

            double firstLoss = double.NaN, lastLoss = double.NaN;
            for (int e = 0; e < 300; e++)
            {
                nn.Train(featureTensor, labelTensor);
                double loss = Convert.ToDouble(nn.GetLastLoss());
                if (e == 0) firstLoss = loss;
                lastLoss = loss;
            }
            _output.WriteLine($"loss {firstLoss:F3} -> {lastLoss:F4}");
            Assert.True(lastLoss < firstLoss * 0.5, $"model did not train (loss {firstLoss} -> {lastLoss})");

            // Inference — the bug returned literal 0 for every input.
            var testData = new double[3, 1] { { 2.0 }, { 5.0 }, { 8.0 } };
            double[] expected = { 7.5, 16.5, 25.5 };
            var testTensor = Tensor<double>.FromMatrix(new Matrix<double>(testData));

            foreach (var pred in new[] { nn.Forward(testTensor), nn.Predict(testTensor) })
            {
                AssertNonZeroAndTracksLine(pred, expected, "direct GPU inference");
            }

            var facadeResult = await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
                .ConfigureDataLoader(DataLoaders.FromTensors(featureTensor, labelTensor))
                .ConfigureModel(nn)
                .ConfigureGpuAcceleration()
                .BuildAsync();
            AssertNonZeroAndTracksLine(facadeResult.Predict(testTensor), expected, "AiModelBuilder result.Predict");
        }
        finally
        {
            AiDotNetEngine.Current = previous;
            gpu.Dispose();
        }
    }

    private static void AssertNonZeroAndTracksLine(Tensor<double> prediction, double[] expected, string path)
    {
        var flat = prediction.ToArray();
        bool allZero = true;
        for (int i = 0; i < flat.Length; i++)
        {
            if (flat[i] != 0.0)
            {
                allZero = false;
                break;
            }
        }

        Assert.False(allZero, $"{path} returned all zeros after training (#1422).");
        for (int i = 0; i < expected.Length; i++)
        {
            double rel = Math.Abs(flat[i] - expected[i]) / Math.Abs(expected[i]);
            Assert.True(rel < 0.15, $"{path} pred[{i}]={flat[i]:F3} vs expected {expected[i]} (rel {rel:F3}) — a converged y=3x+1.5 head must track the line.");
        }
    }
}
