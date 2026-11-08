// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.CreditAssignment;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// A training step under a local-objective credit rule on the GPU engine. The rule's gradient computer returned its
/// output loss and prediction past the tape that made them, and the GPU engine releases a finished tape's
/// intermediates, so Train threw "GPU intermediate of a finished GradientTape" when it read the loss (every
/// <c>--credit tied</c> run of an HRE language model failed at its first step).
/// </summary>
[Collection("TrainingDiagnosticsSequential")]
public class GpuLocalCreditTrainStepTests
{
    private const int Dim = 8, Width = 16, Classes = 3, Batch = 24;

    [SkippableFact]
    public void A_local_credit_train_step_reads_its_loss_on_the_gpu()
    {
        using var gpu = new DirectGpuTensorEngine();
        if (Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS") == "1")
            Assert.True(gpu.IsGpuAvailable, "The required GPU backend failed to initialize.");
        Skip.If(!gpu.IsGpuAvailable, "No DirectGpuTensorEngine available.");
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu;
        try
        {
            var layers = new List<ILayer<float>>
            {
                new FullyConnectedLayer<float>(Dim, Width, new TanhActivation<float>()),
                new FullyConnectedLayer<float>(Width, Width, new TanhActivation<float>()),
                new FullyConnectedLayer<float>(Width, Classes, new SoftmaxActivation<float>()),
            };
            var arch = new NeuralNetworkArchitecture<float>(
                inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.MultiClassClassification,
                complexity: NetworkComplexity.Simple, inputSize: Dim, outputSize: Classes, layers: layers) { RandomSeed = 3 };
            var net = new NeuralNetwork<float>(arch);
            net.SetCreditRule(new TiedReadoutLocalErrorCreditRule<float>(readoutLayerCount: 1));

            var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(7);
            var x = new Tensor<float>(new[] { Batch, Dim });
            var y = new Tensor<float>(new[] { Batch, Classes });
            for (int n = 0; n < Batch; n++)
            {
                for (int d = 0; d < Dim; d++) x[n * Dim + d] = (float)(rng.NextDouble() * 2 - 1);
                y[n * Classes + n % Classes] = 1f;
            }

            net.Train(x, y);

            float loss = net.GetLastLoss();
            Assert.True(float.IsFinite(loss) && loss > 0f, $"loss {loss}");
        }
        finally { AiDotNetEngine.Current = previous; }
    }
}
