// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.ActivationFunctions;
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
/// The eager tape step (TrainWithTape, the path a model takes when the compiled fused step does not apply) on the GPU
/// engine must follow the CPU engine's trajectory. Its parameters live in one ParameterBuffer, every parameter a view of
/// a flat array. Three bugs made the GPU step train on wrong values: the device weight cache served every view the
/// buffer cached for the whole flat array (the first parameter's data), the post-step invalidation dropped the device
/// optimizer's update, and the device optimizer was off unless an environment variable opted in. (Measured on a
/// 27M-parameter LM: GPU loss 10.91 -> 9.55 over five steps while the CPU engine went 10.91 -> 5.81.)
/// </summary>
[Collection("TrainingDiagnosticsSequential")]
public class GpuEagerTapeStepMatchesCpuTests
{
    private const int Inputs = 16, Hidden = 32, Outputs = 4, Batch = 32, Steps = 6;

    /// <summary>A network that always takes the eager tape step.</summary>
    private sealed class EagerNetwork : NeuralNetwork<float>
    {
        public EagerNetwork(NeuralNetworkArchitecture<float> architecture, IGradientBasedOptimizer<float, Tensor<float>, Tensor<float>> optimizer)
            : base(architecture, optimizer) { }

        protected override bool SupportsFusedCompiledTraining => false;
    }

    private static EagerNetwork BuildNetwork()
    {
        var layers = new List<ILayer<float>>
        {
            new DenseLayer<float>(Hidden, (IActivationFunction<float>)new ReLUActivation<float>()),
            // A norm gain read through the device weight path, at a nonzero offset of the flat parameter array.
            new LayerNormalizationLayer<float>(Hidden),
            new DenseLayer<float>(Outputs, (IActivationFunction<float>?)null),
        };
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputSize: Inputs, outputSize: Outputs, layers: layers);
        var options = new AdamWOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-2 };
        return new EagerNetwork(architecture, new AdamWOptimizer<float, Tensor<float>, Tensor<float>>(null, options));
    }

    private static (Tensor<float> X, Tensor<float> Y) Data()
    {
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(20261102);
        var x = new Tensor<float>(new[] { Batch, Inputs });
        var y = new Tensor<float>(new[] { Batch, Outputs });
        for (int r = 0; r < Batch; r++)
        {
            for (int i = 0; i < Inputs; i++) x[r, i] = (float)(rng.NextDouble() * 2 - 1);
            for (int o = 0; o < Outputs; o++) y[r, o] = (float)(rng.NextDouble() * 2 - 1);
        }
        return (x, y);
    }

    /// <summary>
    /// The losses of <see cref="Steps"/> training steps; on a GPU engine also checks after each step that every
    /// parameter is device-authoritative, which only the device optimizer leaves it (a host update writes the host
    /// copy). The losses alone cannot tell the two apart: both apply the same AdamW rule.
    /// </summary>
    private static List<double> Trajectory(IEngine engine, float[] initialParameters)
    {
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            var net = BuildNetwork();
            var (x, y) = Data();
            net.SetTrainingMode(true);
            net.Predict(x);                                   // materialize the lazy layers
            net.SetTrainingMode(true);
            net.UpdateParameters(new Vector<float>((float[])initialParameters.Clone()));
            var losses = new List<double>();
            for (int s = 0; s < Steps; s++)
            {
                net.Train(x, y);
                if (engine is DirectGpuTensorEngine gpu)
                {
                    var parameters = AiDotNet.Training.TapeTrainingStep<float>.CollectParameters(net.Layers);
                    Assert.NotEmpty(parameters);
                    for (int p = 0; p < parameters.Count; p++)
                        Assert.True(gpu.IsDeviceAuthoritative(parameters[p]),
                            $"step {s}: parameter {p} was not updated on the device (the GPU optimizer step did not run)");
                }
                losses.Add(Convert.ToDouble(net.GetLastLoss()));
            }
            return losses;
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }

    [SkippableFact]
    public void Gpu_eager_step_follows_the_cpu_trajectory()
    {
        // The constructor does not throw on a host without a GPU backend (it reports IsGpuAvailable = false), so any
        // exception from it is a failure, not a skip.
        var gpu = new DirectGpuTensorEngine();
        try
        {
            Skip.If(!gpu.SupportsGpu || !gpu.IsGpuAvailable, "No DirectGpuTensorEngine available.");

            // One shared initialization for both engines.
            float[] init;
            var previous = AiDotNetEngine.Current;
            AiDotNetEngine.Current = new CpuEngine();
            try
            {
                var seedNet = BuildNetwork();
                seedNet.Predict(Data().X);
                init = seedNet.GetParameters().ToArray();
            }
            finally
            {
                AiDotNetEngine.Current = previous;
            }

            var cpu = Trajectory(new CpuEngine(), init);
            var gpuLosses = Trajectory(gpu, init);
            string Show(List<double> l) => string.Join(" -> ", l.Select(v => v.ToString("G5")));
            Assert.True(cpu[^1] < 0.9 * cpu[0], $"the CPU reference did not learn: {Show(cpu)}");
            for (int s = 0; s < Steps; s++)
                Assert.True(Math.Abs(gpuLosses[s] - cpu[s]) <= 2e-3 * Math.Max(1.0, Math.Abs(cpu[s])),
                    $"step {s}: GPU {Show(gpuLosses)} vs CPU {Show(cpu)}");
        }
        finally
        {
            gpu.Dispose();
        }
    }
}
