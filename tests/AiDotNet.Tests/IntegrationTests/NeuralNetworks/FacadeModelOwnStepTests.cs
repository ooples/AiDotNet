// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.ActivationFunctions;
using AiDotNet.Data.Loaders;
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
/// The facade trains a neural network batch by batch through the network's OWN training step with the configured
/// optimizer (GradientBasedOptimizerBase.TryModelOwnStep) instead of the flat round trip - tape gradients scattered
/// into a host vector, clipped on the host, copied back into host tensors, updated and re-uploaded. Both paths apply
/// the same optimizer Step, so they must train to the same parameters; the model step only removes the round trip.
/// </summary>
[Collection("TrainingDiagnosticsSequential")]
public class FacadeModelOwnStepTests
{
    // 96 samples leave 67 in the facade's training split: batches of 32, 32 and 3, so every epoch ends on a short
    // batch - a batch-shape change the compiled step has to survive without restarting the optimizer.
    private const int Inputs = 12, Hidden = 24, Outputs = 3, Samples = 96, Batch = 32, Epochs = 3;

    private static (Tensor<float> X, Tensor<float> Y) Data()
    {
        var rng = new Random(20260927);
        var x = new Tensor<float>(new[] { Samples, Inputs });
        var y = new Tensor<float>(new[] { Samples, Outputs });
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int r = 0; r < Samples; r++)
            for (int o = 0; o < Outputs; o++)
            {
                float s = 0;
                for (int i = 0; i < Inputs; i++) s += x[r, i] * (float)Math.Sin(0.7 * (i + 1) * (o + 1));
                y[r, o] = s / Inputs;
            }
        return (x, y);
    }

    // Both runs start from the SAME weights: the first network's materialized initial parameters are loaded into every
    // later one (there is no global seed; DenseLayer initializes lazily on the first forward).
    private static Vector<float>? _initialParameters;

    private static NeuralNetwork<float> Network(Tensor<float> sample)
    {
        var layers = new List<ILayer<float>>
        {
            new DenseLayer<float>(Hidden, (IActivationFunction<float>)new ReLUActivation<float>()),
            new DenseLayer<float>(Outputs, (IActivationFunction<float>?)null),
        };
        var network = new NeuralNetwork<float>(new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputSize: Inputs, outputSize: Outputs, layers: layers));
        network.Predict(sample);   // materialize the lazily initialized weights
        if (_initialParameters is null) _initialParameters = network.GetParameters();
        else network.SetParameters(_initialParameters);
        return network;
    }

    private static async Task<(float[] Parameters, double Mse, int ModelSteps, string? Declined, bool Fused)> TrainAsync(bool modelStep)
    {
        var previous = Environment.GetEnvironmentVariable("AIDOTNET_FACADE_MODEL_STEP");
        Environment.SetEnvironmentVariable("AIDOTNET_FACADE_MODEL_STEP", modelStep ? null : "0");
        try
        {
            var (x, y) = Data();
            var network = Network(x);
            var optimizer = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(network,
                new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
                {
                    MaxIterations = Epochs, BatchSize = Batch, UseEarlyStopping = false, InitialLearningRate = 1e-2,
                    // Explicit, so both paths apply it: the model's own step applies only a regularization the
                    // caller chose (the implicit default L2 is the flat path's alone), and this compares the paths
                    // under the same optimizer - including the regularization reaching the fused step.
                    Regularization = new AiDotNet.Regularization.L2Regularization<float, Tensor<float>, Tensor<float>>(),
                });
            var result = await new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
                .ConfigureModel(network)
                .ConfigureOptimizer(optimizer)
                .ConfigureDataLoader(DataLoaders.FromTensors(x, y))
                .BuildAsync();
            var predicted = result.Predict(x);
            double mse = 0;
            for (int i = 0; i < y.Length; i++) { double d = predicted[i] - y[i]; mse += d * d; }
            bool fused = network.FusedSession.IsCommitted;
            return (network.GetParameters().ToArray(), mse / y.Length, optimizer.ModelOwnStepCount, optimizer.LastModelStepDeclineReason, fused);
        }
        finally
        {
            Environment.SetEnvironmentVariable("AIDOTNET_FACADE_MODEL_STEP", previous);
        }
    }

    [Fact(Timeout = 300000)]
    public async Task The_model_step_trains_to_the_same_parameters_as_the_flat_path()
    {
        // Yield first so the xUnit timeout covers the synchronous training work below.
        await Task.Yield();
        var previousEngine = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.ResetToCpu();
            var flat = await TrainAsync(modelStep: false);
            var model = await TrainAsync(modelStep: true);
            // Without this the comparison could pass vacuously: a model step that silently declined would leave both
            // runs on the flat path.
            const int batchesPerEpoch = 3;   // 67 training rows in batches of 32: 32, 32, 3
            int batches = Epochs * batchesPerEpoch;
            Assert.Equal(0, flat.ModelSteps);
            Assert.True(model.ModelSteps == batches, $"model step ran {model.ModelSteps}/{batches} batches; declined: {model.Declined}");
            // The model step must run the compiled fused step (optimizer state inside the plan), not the eager tape:
            // that is the path whose optimizer state has to survive the short batch.
            Assert.True(model.Fused, "the model step never committed to the fused compiled path");

            Assert.Equal(flat.Parameters.Length, model.Parameters.Length);
            double diff = 0, norm = 0;
            for (int i = 0; i < flat.Parameters.Length; i++)
            {
                double d = flat.Parameters[i] - model.Parameters[i];
                diff += d * d; norm += (double)flat.Parameters[i] * flat.Parameters[i];
            }
            Assert.True(Math.Sqrt(diff) <= 1e-4 * Math.Sqrt(norm),
                $"model step diverged from the flat path: |dtheta| = {Math.Sqrt(diff):G4}, |theta| = {Math.Sqrt(norm):G4}");
            Assert.True(Math.Abs(flat.Mse - model.Mse) <= 1e-4 * (1 + flat.Mse), $"mse flat={flat.Mse} model={model.Mse}");
        }
        finally
        {
            AiDotNetEngine.Current = previousEngine;
        }
    }
}
