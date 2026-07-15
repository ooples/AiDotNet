using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet;
using AiDotNet.ActivationFunctions;
using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Training;

/// <summary>
/// Locks in the facade's per-epoch training contract for neural networks, so it stays at parity with
/// the TimeSeries family.
/// </summary>
/// <remarks>
/// Neural networks reach training through the optimizer path, which already drives the facade's
/// per-epoch seam (OptimizerBase.SetEpochProgressCallback -> InvokeTrainingEpoch). TimeSeries models
/// instead own their epoch loop inside Train(), which the facade calls once — so their callbacks fired
/// a single synthetic epoch after training had finished, and no veto could take effect. These tests
/// assert the neural side genuinely works, so that the two families cannot silently diverge again.
/// </remarks>
public class TrainingContractParityTests
{
    private static NeuralNetwork<double> BuildMlp(int inputSize, int outputSize)
    {
        var layers = new List<ILayer<double>>
        {
            new DenseLayer<double>(8, (IActivationFunction<double>)new ReLUActivation<double>()),
            new DenseLayer<double>(outputSize),
        };
        var architecture = new NeuralNetworkArchitecture<double>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.Regression,
            inputSize: inputSize,
            outputSize: outputSize,
            layers: layers);
        return new NeuralNetwork<double>(architecture);
    }

    [Fact(Timeout = 120000)]
    public async Task NeuralNetwork_ConfigureTrainingCallback_ObservesRealEpochs()
    {
        const int rows = 48, cols = 4;
        var x = new Tensor<double>([rows, cols]);
        var y = new Tensor<double>([rows, 1]);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++) x[i, j] = Math.Sin((i + j) * 0.2);
            y[i, 0] = Math.Sin((i + cols) * 0.2);
        }

        var adam = new AdamOptimizer<double, Tensor<double>, Tensor<double>>(
            null,
            new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>
            {
                InitialLearningRate = 0.01,
                MaxIterations = 12,
                BatchSize = 16,
            });

        var observed = new List<int>();
        await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureModel(BuildMlp(cols, 1))
            .ConfigureOptimizer(adam)
            .ConfigureDataLoader(new InMemoryDataLoader<double, Tensor<double>, Tensor<double>>(x, y))
            .ConfigureTrainingCallback(p => { observed.Add(p.Epoch); return true; })
            .BuildAsync();

        Assert.True(
            observed.Count > 1,
            $"neural network: facade observed {observed.Count} epoch(s) â€” same single-epoch defect the TimeSeries family had");
    }

    [Fact(Timeout = 120000)]
    public async Task NeuralNetwork_CallbackReturningFalse_StopsTrainingEarly()
    {
        const int rows = 48, cols = 4;
        var x = new Tensor<double>([rows, cols]);
        var y = new Tensor<double>([rows, 1]);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++) x[i, j] = Math.Sin((i + j) * 0.2);
            y[i, 0] = Math.Sin((i + cols) * 0.2);
        }

        var adam = new AdamOptimizer<double, Tensor<double>, Tensor<double>>(
            null,
            new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>
            {
                InitialLearningRate = 0.01,
                MaxIterations = 60,
                BatchSize = 16,
            });

        int observed = 0;
        await new AiModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureModel(BuildMlp(cols, 1))
            .ConfigureOptimizer(adam)
            .ConfigureDataLoader(new InMemoryDataLoader<double, Tensor<double>, Tensor<double>>(x, y))
            .ConfigureTrainingCallback(_ => { observed++; return observed < 4; })
            .BuildAsync();

        Assert.Equal(4, observed);
    }
}
