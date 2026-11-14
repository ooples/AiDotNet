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
/// Closure objectives (a loss that builds its own per-step data inside itself) train on compiled plans through the
/// shared entry points (<see cref="TapeTrainingStepper{T}.EagerObjectiveStep(Func{IReadOnlyList{Tensor{T}}}, Func{Tensor{T}}, AiDotNet.Interfaces.IGradientBasedOptimizer{T, Tensor{T}, Tensor{T}}, Action{IReadOnlyDictionary{Tensor{T}, Tensor{T}}}?, double, Action?, bool)"/>,
/// the SGD variant and <c>NeuralNetworkBase.TrainWithCustomLoss</c>). Every step's data is new, so a plan that replayed
/// an earlier step's data, or a captured scalar, would leave the weights away from the eager run's.
/// </summary>
[Collection("FusedOptimizerGlobalState")]
public class ClosureObjectiveFusedTrainingTests
{
    // The fused plan takes Adam's beta1, beta2 and epsilon as float, so double training drifts from the eager step by
    // a few 1e-8 of the parameter scale per step; a step on stale data moves weights by far more.
    private const double RelativeTolerance = 1e-5;
    private const int Features = 4, Hidden = 6;

    private static Tensor<double> Seeded(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t.SetFlat(i, rng.NextDouble() - 0.5);
        return t;
    }

    private static Tensor<double>[] InitialWeights()
        => new[] { Seeded(new[] { Features, Hidden }, 1), Seeded(new[] { Hidden, 1 }, 2) };

    private static Tensor<double>[] Clone(Tensor<double>[] source)
        => source.Select(t => t.Clone()).ToArray();

    /// <summary>
    /// mean((tanh(x W1) W2 - y)^2) * scale on data built INSIDE the closure from the step index: new tensor objects,
    /// new values, and a per-step scalar every call.
    /// </summary>
    private static Func<Tensor<double>> Objective(Tensor<double>[] weights, Func<int> step, Func<int, int> batchOf)
    {
        var engine = AiDotNetEngine.Current;
        return () =>
        {
            int t = step();
            int batch = batchOf(t);
            var x = Seeded(new[] { batch, Features }, 100 + t);
            var y = Seeded(new[] { batch, 1 }, 200 + t);
            double scale = 1.0 + 0.25 * t;   // a per-step coefficient, like a diffusion timestep's weight
            var h = engine.Tanh(engine.TensorMatMul(x, weights[0]));
            var diff = engine.TensorSubtract(engine.TensorMatMul(h, weights[1]), y);
            var mse = engine.ReduceMean(engine.TensorMultiply(diff, diff), new[] { 0, 1 }, keepDims: false);
            return engine.TensorMultiplyScalar(mse, scale);
        };
    }

    private static AdamOptimizer<double, Tensor<double>, Tensor<double>> Adam()
        => new(null, new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>
        {
            InitialLearningRate = 0.01,
            EnableGradientClipping = false,
        });

    private sealed record RunResult(Tensor<double>[] Weights, int FusedSteps, int PlanCount, bool Dynamic);

    private static RunResult Run(bool compile, int steps, Func<int, int> batchOf)
    {
        var original = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = compile });
            CompiledTapeTrainingStep<double>.Invalidate();
            var weights = Clone(InitialWeights());
            var optimizer = Adam();
            int step = 0;
            var objective = Objective(weights, () => step, batchOf);
            for (step = 0; step < steps; step++)
                TapeTrainingStepper<double>.EagerObjectiveStep(() => weights, objective, optimizer);
            var stepper = TapeTrainingStepper<double>.ForOwner(optimizer);
            return new RunResult(weights, stepper.FusedStepCount,
                CompiledTapeTrainingStep<double>.LastClosurePlanCount, CompiledTapeTrainingStep<double>.LastClosureIsDynamic);
        }
        finally
        {
            TensorCodecOptions.SetCurrent(original);
            CompiledTapeTrainingStep<double>.Invalidate();
        }
    }

    private static void AssertSameWeights(Tensor<double>[] expected, Tensor<double>[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int p = 0; p < expected.Length; p++)
            for (int i = 0; i < expected[p].Length; i++)
            {
                double e = expected[p].GetFlat(i), a = actual[p].GetFlat(i);
                Assert.True(Math.Abs(e - a) <= RelativeTolerance * Math.Max(1.0, Math.Abs(e)),
                    $"parameter {p}[{i}]: compiled {a}, eager {e}");
            }
    }

    [Fact]
    public void Closure_objective_trains_fused_and_matches_eager()
    {
        var eager = Run(compile: false, steps: 6, batchOf: _ => 8);
        var compiled = Run(compile: true, steps: 6, batchOf: _ => 8);

        Assert.Equal(0, eager.FusedSteps);
        Assert.Equal(6, compiled.FusedSteps);
        Assert.Equal(1, compiled.PlanCount);
        AssertSameWeights(eager.Weights, compiled.Weights);
    }

    [Fact]
    public void Partial_last_batch_compiles_a_second_signature()
    {
        int[] batches = { 8, 8, 5, 8 };
        var eager = Run(compile: false, steps: batches.Length, batchOf: t => batches[t]);
        var compiled = Run(compile: true, steps: batches.Length, batchOf: t => batches[t]);

        Assert.Equal(batches.Length, compiled.FusedSteps);
        Assert.Equal(2, compiled.PlanCount);
        Assert.False(compiled.Dynamic);
        AssertSameWeights(eager.Weights, compiled.Weights);
    }

    [Fact]
    public void Dynamic_objective_goes_uncached_and_caps_the_cache()
    {
        // A new batch size every step: every probe misses the cache.
        var eager = Run(compile: false, steps: 12, batchOf: t => t + 1);
        var compiled = Run(compile: true, steps: 12, batchOf: t => t + 1);

        Assert.True(compiled.Dynamic);
        Assert.True(compiled.PlanCount <= ClosureObjectiveState<double>.MaxPlans, $"{compiled.PlanCount} plans cached");
        // Still compiled, each step from its own trace, so the optimizer's moments never restart.
        Assert.Equal(12, compiled.FusedSteps);
        AssertSameWeights(eager.Weights, compiled.Weights);
    }

    [Fact]
    public void Sgd_objective_with_an_owner_trains_fused_and_matches_plain_sgd()
    {
        var original = TensorCodecOptions.Current;
        try
        {
            Tensor<double>[] RunSgd(bool compile, object owner)
            {
                TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = compile });
                CompiledTapeTrainingStep<double>.Invalidate();
                var weights = Clone(InitialWeights());
                int step = 0;
                var objective = Objective(weights, () => step, _ => 8);
                for (step = 0; step < 5; step++)
                    TapeTrainingStepper<double>.EagerSgdObjectiveStep(() => weights, objective, 0.05,
                        owner: compile ? owner : null);
                return weights;
            }

            var compiledOwner = new object();
            var eager = RunSgd(compile: false, owner: new object());
            var compiled = RunSgd(compile: true, owner: compiledOwner);

            Assert.Equal(5, TapeTrainingStepper<double>.ForOwner(compiledOwner).FusedStepCount);
            AssertSameWeights(eager, compiled);
        }
        finally
        {
            TensorCodecOptions.SetCurrent(original);
            CompiledTapeTrainingStep<double>.Invalidate();
        }
    }

    [Fact]
    public void Host_read_inside_the_objective_never_replays_stale()
    {
        // The objective reads its loss on the host every other step (logging): those probes abort and the step
        // compiles from its own trace; the weights still match eager exactly.
        var original = TensorCodecOptions.Current;
        try
        {
            Tensor<double>[] RunLogged(bool compile, out int fused)
            {
                TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = compile });
                CompiledTapeTrainingStep<double>.Invalidate();
                var weights = Clone(InitialWeights());
                var optimizer = Adam();
                int step = 0;
                var inner = Objective(weights, () => step, _ => 8);
                double logged = 0;
                Func<Tensor<double>> objective = () =>
                {
                    var loss = inner();
                    if (step % 2 == 1) logged += loss.ToArray()[0];
                    return loss;
                };
                for (step = 0; step < 6; step++)
                    TapeTrainingStepper<double>.EagerObjectiveStep(() => weights, objective, optimizer);
                fused = TapeTrainingStepper<double>.ForOwner(optimizer).FusedStepCount;
                Assert.False(double.IsNaN(logged));
                return weights;
            }

            var eager = RunLogged(compile: false, out _);
            var compiled = RunLogged(compile: true, out int fusedSteps);

            Assert.Equal(6, fusedSteps);
            AssertSameWeights(eager, compiled);
        }
        finally
        {
            TensorCodecOptions.SetCurrent(original);
            CompiledTapeTrainingStep<double>.Invalidate();
        }
    }

    [Fact]
    public void TrainWithCustomLoss_runs_compiled_on_fresh_targets_and_matches_eager()
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
            (Vector<double> Parameters, int Fused) RunNet(bool compile, Vector<double>? start)
            {
                TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = compile });
                CompiledTapeTrainingStep<double>.Invalidate();
                using var network = Net();
                network.Predict(new Tensor<double>(new[] { 1, Features }));   // shape the lazy layers
                if (start is not null) network.SetParameters(start);
                var optimizer = Adam();
                var engine = AiDotNetEngine.Current;
                for (int step = 0; step < 5; step++)
                {
                    var input = Seeded(new[] { 8, Features }, 300 + step);
                    var target = Seeded(new[] { 8, 1 }, 400 + step);   // a per-step tensor the loss captures
                    double weight = 1.0 + step;                          // a per-step scalar the loss captures
                    network.TrainWithCustomLoss(input, output =>
                    {
                        var diff = engine.TensorSubtract(output, target);
                        var mse = engine.ReduceMean(engine.TensorMultiply(diff, diff), new[] { 0, 1 }, keepDims: false);
                        return engine.TensorMultiplyScalar(mse, weight);
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

            var eager = RunNet(compile: false, start);
            var compiled = RunNet(compile: true, start);

            Assert.True(compiled.Fused > 0, "TrainWithCustomLoss never ran on a compiled plan");
            Assert.Equal(eager.Parameters.Length, compiled.Parameters.Length);
            for (int i = 0; i < eager.Parameters.Length; i++)
            {
                double e = eager.Parameters[i], a = compiled.Parameters[i];
                Assert.True(Math.Abs(e - a) <= RelativeTolerance * Math.Max(1.0, Math.Abs(e)),
                    $"parameter {i}: compiled {a}, eager {e}");
            }
        }
        finally
        {
            TensorCodecOptions.SetCurrent(original);
            CompiledTapeTrainingStep<double>.Invalidate();
        }
    }
}
