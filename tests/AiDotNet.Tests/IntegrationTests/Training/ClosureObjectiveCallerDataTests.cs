using System;
using System.Collections.Generic;
using System.Linq;
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
/// A closure objective usually reads tensors it did not create: the caller's pre-built batches, reused every epoch.
/// A compiled plan keeps the tensors of the step it was traced from, so replaying it on another batch must not write
/// that batch into the caller's first one, and a batch that comes round again must train on its own data.
/// </summary>
[Collection("FusedOptimizerGlobalState")]
public class ClosureObjectiveCallerDataTests
{
    private const double RelativeTolerance = 1e-5;
    private const int Features = 4, Hidden = 6, Batch = 8, BatchCount = 3, Epochs = 3;

    private static Tensor<double> Seeded(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t.SetFlat(i, rng.NextDouble() - 0.5);
        return t;
    }

    private static Tensor<double>[] Batches(int columns, int seed)
        => Enumerable.Range(0, BatchCount).Select(b => Seeded(new[] { Batch, columns }, seed + b)).ToArray();

    private static AdamOptimizer<double, Tensor<double>, Tensor<double>> Adam()
        => new(null, new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>
        {
            InitialLearningRate = 0.01,
            EnableGradientClipping = false,
        });

    private static void AssertUnchanged(Tensor<double>[] expected, Tensor<double>[] actual, string what)
    {
        for (int b = 0; b < expected.Length; b++)
            for (int i = 0; i < expected[b].Length; i++)
                Assert.True(expected[b].GetFlat(i) == actual[b].GetFlat(i),
                    $"{what} batch {b}[{i}] was {expected[b].GetFlat(i)} and is now {actual[b].GetFlat(i)}: " +
                    "training overwrote the caller's data");
    }

    private static void AssertClose(double expected, double actual, string what)
        => Assert.True(Math.Abs(expected - actual) <= RelativeTolerance * Math.Max(1.0, Math.Abs(expected)),
            $"{what}: compiled {actual}, eager {expected}");

    [Fact]
    public void Objective_step_leaves_the_callers_reused_batches_intact_and_matches_eager()
    {
        var original = TensorCodecOptions.Current;
        try
        {
            (Tensor<double>[] Weights, int Fused) Run(bool compile, Tensor<double>[] xs, Tensor<double>[] ys)
            {
                TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = compile });
                CompiledTapeTrainingStep<double>.Invalidate();
                var engine = AiDotNetEngine.Current;
                var weights = new[] { Seeded(new[] { Features, Hidden }, 1), Seeded(new[] { Hidden, 1 }, 2) };
                var optimizer = Adam();
                int step = 0;
                Func<Tensor<double>> objective = () =>
                {
                    var x = xs[step % BatchCount];
                    var y = ys[step % BatchCount];
                    var predicted = engine.TensorMatMul(engine.Tanh(engine.TensorMatMul(x, weights[0])), weights[1]);
                    var diff = engine.TensorSubtract(predicted, y);
                    return engine.ReduceMean(engine.TensorMultiply(diff, diff), new[] { 0, 1 }, keepDims: false);
                };
                for (step = 0; step < BatchCount * Epochs; step++)
                    TapeTrainingStepper<double>.EagerObjectiveStep(() => weights, objective, optimizer);
                return (weights, TapeTrainingStepper<double>.ForOwner(optimizer).FusedStepCount);
            }

            var pristineX = Batches(Features, 100);
            var pristineY = Batches(1, 200);

            var eager = Run(compile: false, Batches(Features, 100), Batches(1, 200));
            var xs = Batches(Features, 100);
            var ys = Batches(1, 200);
            var compiled = Run(compile: true, xs, ys);

            Assert.Equal(BatchCount * Epochs, compiled.Fused);
            AssertUnchanged(pristineX, xs, "input");
            AssertUnchanged(pristineY, ys, "target");
            for (int p = 0; p < eager.Weights.Length; p++)
                for (int i = 0; i < eager.Weights[p].Length; i++)
                    AssertClose(eager.Weights[p].GetFlat(i), compiled.Weights[p].GetFlat(i), $"parameter {p}[{i}]");
        }
        finally
        {
            TensorCodecOptions.SetCurrent(original);
            CompiledTapeTrainingStep<double>.Invalidate();
        }
    }

    [Fact]
    public void TrainWithCustomLoss_leaves_the_callers_reused_batches_intact_and_matches_eager()
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
            (Vector<double> Parameters, int Fused) Run(bool compile, Vector<double> start, Tensor<double>[] xs, Tensor<double>[] ys)
            {
                TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = compile });
                CompiledTapeTrainingStep<double>.Invalidate();
                using var network = Net();
                network.Predict(new Tensor<double>(new[] { 1, Features }));
                network.SetParameters(start);
                var optimizer = Adam();
                var engine = AiDotNetEngine.Current;
                for (int step = 0; step < BatchCount * Epochs; step++)
                {
                    var target = ys[step % BatchCount];
                    network.TrainWithCustomLoss(xs[step % BatchCount], output =>
                    {
                        var diff = engine.TensorSubtract(output, target);
                        return engine.ReduceMean(engine.TensorMultiply(diff, diff), new[] { 0, 1 }, keepDims: false);
                    }, optimizer);
                }
                return (network.GetParameters(), network.CustomFusedStepCount);
            }

            Vector<double> start;
            using (var seed = Net())
            {
                seed.Predict(new Tensor<double>(new[] { 1, Features }));
                start = seed.GetParameters();
            }

            var pristineX = Batches(Features, 300);
            var pristineY = Batches(1, 400);

            var eager = Run(compile: false, start, Batches(Features, 300), Batches(1, 400));
            var xs = Batches(Features, 300);
            var ys = Batches(1, 400);
            var compiled = Run(compile: true, start, xs, ys);

            Assert.True(compiled.Fused > 0, "TrainWithCustomLoss never ran on a compiled plan");
            AssertUnchanged(pristineX, xs, "input");
            AssertUnchanged(pristineY, ys, "target");
            for (int i = 0; i < eager.Parameters.Length; i++)
                AssertClose(eager.Parameters[i], compiled.Parameters[i], $"parameter {i}");
        }
        finally
        {
            TensorCodecOptions.SetCurrent(original);
            CompiledTapeTrainingStep<double>.Invalidate();
        }
    }
}
