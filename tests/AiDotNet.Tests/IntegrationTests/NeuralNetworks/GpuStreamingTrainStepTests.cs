// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// The streaming (optimizer-in-backward) training step on the GPU engine against the same step on the CPU engine.
/// The GPU step runs inside a step-wide <see cref="TensorScope"/>, measures its clip norm on the device and publishes
/// its gradients from reused snapshot buffers; none of that ran on a GPU in any test. Several steps are taken so the
/// scope (opened from the second step on) and the buffer reuse are both exercised, and the weights, the loss and the
/// published gradient surface must match the CPU step.
/// </summary>
[Collection("TrainingDiagnosticsSequential")]
public class GpuStreamingTrainStepTests
{
    public enum Clip { None, Exact, FastApprox }

    private const int Steps = 4;

    private static (float[] Parameters, float[] Gradients, float Loss) Run(Clip clip, Vector<float>? start, out Vector<float> used)
    {
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 4,
            outputSize: 2);
        arch.RandomSeed = 7;
        var sgd = new StochasticGradientDescentOptimizer<float, Tensor<float>, Tensor<float>>(null,
            new StochasticGradientDescentOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                InitialLearningRate = 0.01,
                UseAdaptiveLearningRate = false,
                EnableGradientClipping = false,
            });
        using var net = new FeedForwardNeuralNetwork<float>(arch, sgd,
            options: new FeedForwardNeuralNetworkOptions { MaxGradNorm = clip == Clip.None ? 0.0 : 0.05 });
        net.StreamingTraining = StreamingTrainingMode.ForceOn;
        net.FastApproxGradClip = clip == Clip.FastApprox;
        used = start ?? net.GetParameters();
        net.UpdateParameters(used);

        var x = new Tensor<float>(new[] { 3, 4 });
        var y = new Tensor<float>(new[] { 3, 2 });
        for (int i = 0; i < x.Length; i++) x[i] = ((i % 5) - 2) * 0.6f;
        for (int i = 0; i < y.Length; i++) y[i] = ((i % 3) - 1) * 2.0f;
        for (int step = 0; step < Steps; step++) net.Train(x, y);

        return (net.GetParameters().ToArray(), net.GetParameterGradients().ToArray(), net.GetLastLoss());
    }

    [SkippableTheory]
    [InlineData(Clip.None)]
    [InlineData(Clip.Exact)]
    [InlineData(Clip.FastApprox)]
    public void Streaming_steps_on_the_gpu_match_the_cpu(Clip clip)
    {
        var cpu = Run(clip, null, out var start);

        using var gpu = new DirectGpuTensorEngine();
        if (Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS") == "1")
            Assert.True(gpu.IsGpuAvailable, "The required GPU backend failed to initialize.");
        Skip.If(!gpu.IsGpuAvailable, "No DirectGpuTensorEngine available.");
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu;
        (float[] Parameters, float[] Gradients, float Loss) device;
        try
        {
            device = Run(clip, start, out _);
        }
        finally { AiDotNetEngine.Current = previous; }

        Assert.True(!float.IsNaN(device.Loss) && !float.IsInfinity(device.Loss), $"loss {device.Loss}");
        Assert.Equal(cpu.Loss, device.Loss, 3);

        Assert.Equal(cpu.Parameters.Length, device.Parameters.Length);
        double moved = 0.0;
        for (int i = 0; i < cpu.Parameters.Length; i++)
        {
            moved = Math.Max(moved, Math.Abs(cpu.Parameters[i] - start[i]));
            Assert.True(Math.Abs(cpu.Parameters[i] - device.Parameters[i]) <= 1e-4,
                $"parameter {i}: cpu {cpu.Parameters[i]}, gpu {device.Parameters[i]}");
        }
        Assert.True(moved > 1e-5, "the CPU steps did not move the weights, so the comparison is vacuous");

        Assert.Equal(cpu.Gradients.Length, device.Gradients.Length);
        double gradientScale = 0.0;
        for (int i = 0; i < cpu.Gradients.Length; i++)
        {
            gradientScale = Math.Max(gradientScale, Math.Abs(cpu.Gradients[i]));
            Assert.True(Math.Abs(cpu.Gradients[i] - device.Gradients[i]) <= 1e-3 * Math.Max(1.0, Math.Abs(cpu.Gradients[i])),
                $"published gradient {i}: cpu {cpu.Gradients[i]}, gpu {device.Gradients[i]}");
        }
        Assert.True(gradientScale > 1e-6, "no gradient was published");
    }
}
