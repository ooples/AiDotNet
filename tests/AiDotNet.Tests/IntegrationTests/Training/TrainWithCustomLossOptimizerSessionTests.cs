using System;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Training;

/// <summary>
/// A network stepped through TrainWithCustomLoss by two optimizers in turn (two losses over a shared encoder) keeps
/// each optimizer's compiled plan and moments: with one session for the network, every switch dropped the other
/// optimizer's plan and restarted its Adam moments, which the eager tape never does.
/// </summary>
[Collection("FusedOptimizerGlobalState")]
public class TrainWithCustomLossOptimizerSessionTests
{
    private const double RelativeTolerance = 1e-5;
    private const int Features = 4, Hidden = 6, Batch = 8;

    private static Tensor<double> Seeded(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t.SetFlat(i, rng.NextDouble() - 0.5);
        return t;
    }

    private static AdamOptimizer<double, Tensor<double>, Tensor<double>> Adam()
        => new(null, new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>
        {
            InitialLearningRate = 0.01,
            EnableGradientClipping = false,
        });

    private static void AssertClose(double expected, double actual, string what)
        => Assert.True(Math.Abs(expected - actual) <= RelativeTolerance * Math.Max(1.0, Math.Abs(expected)),
            $"{what}: compiled {actual}, eager {expected}");

    [Fact]
    public void TrainWithCustomLoss_alternating_two_optimizers_matches_eager()
    {
        static NeuralNetwork<double> Net()
        {
            var architecture = new NeuralNetworkArchitecture<double>(
                inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
                complexity: NetworkComplexity.Simple, inputSize: Features, outputSize: 1);
            var network = new NeuralNetwork<double>(architecture);
            network.AddLayer(LayerType.Dense, Hidden, ActivationFunction.Tanh);
            network.AddLayer(LayerType.Dense, 1, ActivationFunction.Linear);
            return network;
        }

        var original = TensorCodecOptions.Current;
        try
        {
            Vector<double> Run(bool compile, Vector<double> start)
            {
                TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = compile });
                CompiledTapeTrainingStep<double>.Invalidate();
                using var network = Net();
                network.Predict(new Tensor<double>(new[] { 1, Features }));
                network.SetParameters(start);
                var optimizers = new[] { Adam(), Adam() };
                var engine = AiDotNetEngine.Current;
                for (int step = 0; step < 8; step++)
                {
                    var input = Seeded(new[] { Batch, Features }, 300 + step);
                    var target = Seeded(new[] { Batch, 1 }, 400 + step);
                    network.TrainWithCustomLoss(input, output =>
                    {
                        var diff = engine.TensorSubtract(output, target);
                        return engine.ReduceMean(engine.TensorMultiply(diff, diff), new[] { 0, 1 }, keepDims: false);
                    }, optimizers[step % 2]);
                }
                return network.GetParameters();
            }

            Vector<double> start;
            using (var seed = Net())
            {
                seed.Predict(new Tensor<double>(new[] { 1, Features }));
                start = seed.GetParameters();
            }

            var eager = Run(compile: false, start);
            var compiled = Run(compile: true, start);
            for (int i = 0; i < eager.Length; i++)
                AssertClose(eager[i], compiled[i], $"parameter {i}");
        }
        finally
        {
            TensorCodecOptions.SetCurrent(original);
            CompiledTapeTrainingStep<double>.Invalidate();
        }
    }
}
