// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// NeuralNetwork.Train on the GPU engine takes the compiled fused step: parameters are moved to the device once and
/// a device-side Adam updates them in place. After every successful fused step the network flushed its weight caches
/// as if the HOST had just been updated, which dropped the device update and detached the buffers the compiled plan
/// keeps writing. The plan's later updates then landed in orphaned buffers, so the weights changed on step 1 and
/// never again, and the loss stayed flat. (Measured on a 1024-3x1024-10 MLP: GPU loss 4.60 -> 4.76 over five steps
/// while the CPU engine went 4.60 -> 0.35; the weights were bit-identical from step 2 on.)
/// </summary>
[Collection("TrainingDiagnosticsSequential")]
public class GpuFusedTrainKeepsLearningTests
{
    private const int Inputs = 16, Hidden = 32, Outputs = 4, Batch = 32, Steps = 12;

    private static NeuralNetwork<float> BuildNetwork()
    {
        var layers = new List<ILayer<float>>
        {
            new DenseLayer<float>(Hidden, (IActivationFunction<float>)new ReLUActivation<float>()),
            new DenseLayer<float>(Outputs, (IActivationFunction<float>?)null),
        };
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputSize: Inputs, outputSize: Outputs, layers: layers);
        // A plain Adam: the configuration the compiled fused step accepts (no adaptive rate or betas).
        var options = new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
        {
            InitialLearningRate = 1e-2, UseAdaptiveLearningRate = false, UseAdaptiveBetas = false,
        };
        return new NeuralNetwork<float>(architecture, new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null, options));
    }

    private static (Tensor<float> X, Tensor<float> Y) LearnableData()
    {
        // y = A x: a target the network can fit, so a working optimizer must drive the loss well down.
        var rng = new Random(20260926);
        var x = new Tensor<float>(new[] { Batch, Inputs });
        var y = new Tensor<float>(new[] { Batch, Outputs });
        var a = new float[Outputs, Inputs];
        for (int o = 0; o < Outputs; o++) for (int i = 0; i < Inputs; i++) a[o, i] = (float)(rng.NextDouble() - 0.5);
        for (int r = 0; r < Batch; r++)
        {
            for (int i = 0; i < Inputs; i++) x[r, i] = (float)(rng.NextDouble() * 2 - 1);
            for (int o = 0; o < Outputs; o++)
            {
                float s = 0;
                for (int i = 0; i < Inputs; i++) s += a[o, i] * x[r, i];
                y[r, o] = s;
            }
        }
        return (x, y);
    }

    [SkippableFact]
    public void Every_fused_gpu_step_moves_the_weights_and_the_loss_falls()
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); } catch { /* no backend */ }
        var previousEngine = AiDotNetEngine.Current;
        var previousLevel = TrainingDiagnosticsConfig.Level;
        var previousSink = TrainingDiagnosticsConfig.Sink;
        try
        {
            Skip.If(gpu is null || !gpu.SupportsGpu || !gpu.IsGpuAvailable, "No DirectGpuTensorEngine available.");
            AiDotNetEngine.Current = gpu!;
            int fusedHits = 0;
            TrainingDiagnosticsConfig.Level = TrainingDiagnosticLevel.PerStep;
            TrainingDiagnosticsConfig.Sink = evt => { if (evt is FusedOptimizerPathEvent { Hit: true }) fusedHits++; };

            var net = BuildNetwork();
            net.SetTrainingMode(true);
            var (x, y) = LearnableData();

            var losses = new List<double>();
            var previousWeights = net.GetParameters().ToArray();
            for (int step = 0; step < Steps; step++)
            {
                net.Train(x, y);
                losses.Add(Convert.ToDouble(net.GetLastLoss()));
                var weights = net.GetParameters().ToArray();
                Assert.True(!weights.SequenceEqual(previousWeights),
                    $"step {step}: the weights did not change — the update landed somewhere the model never reads.");
                previousWeights = weights;
            }

            // The point of the test is the fused path; if it stopped engaging, this test would pass vacuously.
            Assert.True(fusedHits >= Steps - 1, $"the fused GPU step engaged on only {fusedHits} of {Steps} steps.");
            Assert.True(losses[^1] < 0.5 * losses[0],
                $"loss did not fall: {string.Join(" -> ", losses.Select(l => l.ToString("G4")))}");
        }
        finally
        {
            TrainingDiagnosticsConfig.Level = previousLevel;
            TrainingDiagnosticsConfig.Sink = previousSink;
            AiDotNetEngine.Current = previousEngine;
            gpu?.Dispose();
        }
    }
}
