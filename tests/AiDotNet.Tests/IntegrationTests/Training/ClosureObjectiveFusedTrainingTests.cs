using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Models;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
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
    /// mean((tanh(x W1) W2 - y)^2) * scale on data built INSIDE the closure from the step index: new tensor objects
    /// and new values every call. With <paramref name="noise"/> the prediction also gets per-step noise drawn through
    /// <see cref="CompiledStepRandom{T}"/>, and with <paramref name="dropout"/> the hidden layer goes through a seeded
    /// dropout layer: randomness owned by the model rather than by the step index. With
    /// <paramref name="perStepScale"/> the scale is a host scalar that changes every step; a plan freezes a host
    /// scalar, so such a step can never replay and compiles from its own trace.
    /// </summary>
    private static Func<Tensor<double>> Objective(Tensor<double>[] weights, Func<int> step, Func<int, int> batchOf,
        Random? noise = null, DropoutLayer<double>? dropout = null, bool perStepScale = false)
    {
        var engine = AiDotNetEngine.Current;
        return () =>
        {
            int t = step();
            int batch = batchOf(t);
            var x = Seeded(new[] { batch, Features }, 100 + t);
            var y = Seeded(new[] { batch, 1 }, 200 + t);
            double scale = perStepScale ? 1.0 + 0.25 * t : 1.25;   // per step: like a diffusion timestep's weight
            var h = engine.Tanh(engine.TensorMatMul(x, weights[0]));
            if (dropout is not null)
                h = dropout.Forward(h);
            var predicted = engine.TensorMatMul(h, weights[1]);
            if (noise is not null)
                predicted = engine.TensorAdd(predicted,
                    engine.TensorMultiplyScalar(CompiledStepRandom<double>.StandardNormal(new[] { batch, 1 }, noise), 0.1));
            var diff = engine.TensorSubtract(predicted, y);
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

    private sealed record RunResult(Tensor<double>[] Weights, int FusedSteps, int PlanCount, bool Dynamic,
        int GeneratorNext = 0);

    private static RunResult Run(bool compile, int steps, Func<int, int> batchOf, int? noiseSeed = null,
        int? dropoutSeed = null, bool perStepScale = false, Func<int, double>? clipOf = null)
    {
        var original = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = compile });
            CompiledTapeTrainingStep<double>.Invalidate();
            var weights = Clone(InitialWeights());
            var optimizer = Adam();
            int step = 0;
            var noise = noiseSeed is { } n ? new Random(n) : null;
            using var dropout = dropoutSeed is { } d ? new DropoutLayer<double>(0.5) { RandomSeed = d } : null;
            dropout?.SetTrainingMode(true);
            var objective = Objective(weights, () => step, batchOf, noise, dropout, perStepScale);
            for (step = 0; step < steps; step++)
                TapeTrainingStepper<double>.EagerObjectiveStep(() => weights, objective, optimizer,
                    modelGradientClip: clipOf?.Invoke(step) ?? 0.0);
            var stepper = TapeTrainingStepper<double>.ForOwner(optimizer);
            return new RunResult(weights, stepper.FusedStepCount,
                CompiledTapeTrainingStep<double>.LastClosurePlanCount, CompiledTapeTrainingStep<double>.LastClosureIsDynamic,
                noise?.Next() ?? 0);
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
    public void Clip_threshold_that_changes_between_steps_is_applied_as_eager_applies_it()
    {
        // The plan is configured unclipped on step 0; the threshold of every later call must still reach it.
        static double Clip(int step) => step == 0 ? 0.0 : step < 4 ? 1e-3 : 0.0;
        var eager = Run(compile: false, steps: 6, batchOf: _ => 8, clipOf: Clip);
        var compiled = Run(compile: true, steps: 6, batchOf: _ => 8, clipOf: Clip);

        Assert.Equal(6, compiled.FusedSteps);
        AssertSameWeights(eager.Weights, compiled.Weights);
    }

    [Fact]
    public void Per_step_host_scalar_never_replays_and_matches_eager()
    {
        // The scalar is frozen into the plan's ops, so every step's signature is new: no step may replay an earlier
        // step's plan (and coefficient). The objective goes dynamic and each step compiles from its own trace.
        var eager = Run(compile: false, steps: 6, batchOf: _ => 8, perStepScale: true);
        var compiled = Run(compile: true, steps: 6, batchOf: _ => 8, perStepScale: true);

        Assert.Equal(6, compiled.FusedSteps);
        Assert.True(compiled.Dynamic, "a per-step host scalar must stop the objective replaying cached plans");
        AssertSameWeights(eager.Weights, compiled.Weights);
    }

    [Fact]
    public void Model_random_draws_advance_the_generator_once_per_step_as_eager_does()
    {
        // Step 2's batch of 5 misses the cached plan, so that step is probed and then traced again: the trace must
        // reuse the probe's draws. The other steps replay, where the probe's draws are the step's draws and nothing
        // may redraw them. Either mistake moves the generator past the eager run's and changes the noise.
        int[] batches = { 8, 8, 5, 8, 8 };
        var eager = Run(compile: false, steps: batches.Length, batchOf: t => batches[t], noiseSeed: 7);
        var compiled = Run(compile: true, steps: batches.Length, batchOf: t => batches[t], noiseSeed: 7);

        Assert.Equal(batches.Length, compiled.FusedSteps);
        Assert.Equal(eager.GeneratorNext, compiled.GeneratorNext);
        AssertSameWeights(eager.Weights, compiled.Weights);
    }

    [Fact]
    public void Seeded_dropout_masks_match_eager_through_replays_and_retraces()
    {
        // Dropout derives each mask from its seed and a per-forward counter. The stepper's warm-up must not advance
        // the counter, a replay must not refresh the mask its probe drew, and the retrace of step 2 must reuse the
        // probe's counter value.
        int[] batches = { 8, 8, 5, 8, 8 };
        var eager = Run(compile: false, steps: batches.Length, batchOf: t => batches[t], dropoutSeed: 11);
        var compiled = Run(compile: true, steps: batches.Length, batchOf: t => batches[t], dropoutSeed: 11);

        Assert.Equal(batches.Length, compiled.FusedSteps);
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

    private sealed record LoggedRun(Tensor<double>[] Weights, int FusedSteps, int OutsidePlanSteps, int Compiles,
        int PlanCount, bool Dynamic, int GeneratorNext, Tensor<double>[][] Trajectory, double[] Losses, int[] PlanSteps, double[] Rates);

    private static FusedTrainingStepRequest<double> ClosureRequest(Tensor<double>[] weights,
        AdamOptimizer<double, Tensor<double>, Tensor<double>> optimizer, Func<Tensor<double>> objective,
        double clip = 0, bool declaredBreak = false, bool allowBreak = false,
        Action<IReadOnlyDictionary<Tensor<double>, Tensor<double>>>? onGradients = null)
    {
        var placeholder = new Tensor<double>(new[] { 1 });
        return new FusedTrainingStepRequest<double>
        {
            Layers = Array.Empty<AiDotNet.Interfaces.ITrainableLayer<double>>(),
            Input = placeholder,
            Target = placeholder,
            Forward = static input => input,
            ComputeLoss = (_, _) => objective(),
            ClosureObjective = objective,
            Optimizer = optimizer,
            ExtraParameters = weights,
            ModelGradientClip = clip,
            GraphBreakReason = declaredBreak ? "This step logs its current loss on the host." : null,
            ClosureGraphBreakPolicy = allowBreak ? ClosureGraphBreakPolicy.UseConfiguredOptimizer : ClosureGraphBreakPolicy.Disallow,
            OnGradients = onGradients,
        };
    }

    /// <summary>
    /// The objective of <see cref="Run"/>, reading its own loss on the host (logging) on the steps
    /// <paramref name="readsOn"/> picks: no plan can replay such a step.
    /// </summary>
    private static LoggedRun RunLogged(bool compile, int steps, Func<int, bool> readsOn, Func<int, int>? batchOf = null,
        int? noiseSeed = null, Func<int, double>? clipOf = null, bool scheduledL2 = false)
    {
        var original = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = compile });
            CompiledTapeTrainingStep<double>.Invalidate();
            var weights = Clone(InitialWeights());
            var optimizer = scheduledL2
                ? new AdamOptimizer<double, Tensor<double>, Tensor<double>>(null,
                    new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>
                    {
                        InitialLearningRate = 0.01, EnableGradientClipping = false,
                        LearningRateScheduler = new AiDotNet.LearningRateSchedulers.StepLRScheduler(0.01, 2, 0.5),
                        SchedulerStepMode = AiDotNet.LearningRateSchedulers.SchedulerStepMode.StepPerBatch,
                        Regularization = new AiDotNet.Regularization.L2Regularization<double, Tensor<double>, Tensor<double>>(
                            new RegularizationOptions { Strength = 0.03 }),
                    })
                : Adam();
            int step = 0;
            var noise = noiseSeed is { } n ? new Random(n) : null;
            var inner = Objective(weights, () => step, batchOf ?? (_ => 8), noise);
            double logged = 0;
            Func<Tensor<double>> objective = () =>
            {
                var loss = inner();
                if (readsOn(step)) logged += loss.ToArray()[0];
                return loss;
            };
            var stepper = TapeTrainingStepper<double>.ForOwner(optimizer);
            var trajectory = new List<Tensor<double>[]>();
            var losses = new List<double>();
            var rates = new List<double>();
            var planSteps = new List<int>();
            for (step = 0; step < steps; step++)
            {
                int previousPlanStep = -1;
                bool hostRead = readsOn(step);
                losses.Add(stepper.Step(ClosureRequest(weights, optimizer, objective, clipOf?.Invoke(step) ?? 0,
                    declaredBreak: hostRead, allowBreak: hostRead, onGradients: _ =>
                    {
                        if (CompiledTapeTrainingStep<double>.TryGetPlanOptimizerStep(out int count))
                            previousPlanStep = count;
                    })));
                if (compile)
                {
                    Assert.Equal(hostRead, stepper.LastStepHybrid);
                    Assert.Equal(!hostRead, stepper.LastStepFused);
                }
                trajectory.Add(Clone(weights));
                rates.Add(optimizer.GetCurrentLearningRate());
                planSteps.Add(previousPlanStep);
            }
            Assert.False(double.IsNaN(logged));
            return new LoggedRun(weights, stepper.FusedStepCount, stepper.OutsidePlanStepCount,
                compile ? CompiledTapeTrainingStep<double>.LastClosureCompileCount : 0,
                CompiledTapeTrainingStep<double>.LastClosurePlanCount, CompiledTapeTrainingStep<double>.LastClosureIsDynamic,
                noise?.Next() ?? 0, trajectory.ToArray(), losses.ToArray(), planSteps.ToArray(), rates.ToArray());
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
        // Logging steps are explicitly declared; those steps use the eager tape and the same plan's optimizer.
        // The intervening compiled steps resume without restarting its moments.
        static bool Read(int step) => step >= 2 && step % 2 == 0;
        static double Clip(int step) => step switch { 2 => 1e-3, 3 => 2e-3, 6 => 3e-3, _ => 0 };
        var eager = RunLogged(compile: false, steps: 7, readsOn: Read, noiseSeed: 19, clipOf: Clip);
        var compiled = RunLogged(compile: true, steps: 7, readsOn: Read, noiseSeed: 19, clipOf: Clip);

        Assert.Equal(4, compiled.FusedSteps);
        Assert.Equal(3, compiled.OutsidePlanSteps);
        Assert.Equal(1, compiled.Compiles);
        Assert.False(compiled.Dynamic, "one-off host reads must not stop the objective replaying its plan");
        AssertSameWeights(eager.Weights, compiled.Weights);
        Assert.Equal(eager.GeneratorNext, compiled.GeneratorNext);
        for (int step = 0; step < compiled.Trajectory.Length; step++)
        {
            AssertSameWeights(eager.Trajectory[step], compiled.Trajectory[step]);
            Assert.True(Math.Abs(eager.Losses[step] - compiled.Losses[step]) <= RelativeTolerance);
            if (step > 0) Assert.Equal(step, compiled.PlanSteps[step]);
        }
    }

    [Fact]
    public void Host_read_on_every_step_from_some_step_on_runs_outside_the_plan_and_replays_its_draws()
    {
        // From step 2 on every explicitly declared step reads on the host, without probing or compiling it.
        // Its noise generator must still advance once per step, as eagerly.
        var eager = RunLogged(compile: false, steps: 8, readsOn: s => s >= 2, noiseSeed: 7);
        var compiled = RunLogged(compile: true, steps: 8, readsOn: s => s >= 2, noiseSeed: 7);

        Assert.Equal(2, compiled.FusedSteps);
        Assert.Equal(6, compiled.OutsidePlanSteps);
        Assert.Equal(1, compiled.Compiles);
        Assert.Equal(eager.GeneratorNext, compiled.GeneratorNext);
        AssertSameWeights(eager.Weights, compiled.Weights);
    }

    [Fact]
    public void Declared_hybrid_preserves_scheduled_Adam_L2_and_clip_to_zero_through_compiled_resume()
    {
        static bool Read(int step) => step is 2 or 4;
        static double Clip(int step) => step switch { 2 => 1e-3, 3 => 2e-3, _ => 0 };
        var eager = RunLogged(false, 7, Read, noiseSeed: 23, clipOf: Clip, scheduledL2: true);
        var compiled = RunLogged(true, 7, Read, noiseSeed: 23, clipOf: Clip, scheduledL2: true);
        Assert.Equal(5, compiled.FusedSteps);
        Assert.Equal(2, compiled.OutsidePlanSteps);
        Assert.Equal(eager.GeneratorNext, compiled.GeneratorNext);
        Assert.Equal(eager.Rates, compiled.Rates);
        Assert.True(compiled.Rates[6] < compiled.Rates[0]);
        for (int step = 0; step < compiled.Trajectory.Length; step++)
        {
            AssertSameWeights(eager.Trajectory[step], compiled.Trajectory[step]);
            Assert.True(Math.Abs(eager.Losses[step] - compiled.Losses[step]) <= RelativeTolerance);
            if (step > 0) Assert.Equal(step, compiled.PlanSteps[step]);
        }
    }


    [Fact]
    public void A_lasting_change_of_structure_compiles_once_and_replays_the_new_plan()
    {
        // From step 2 on the batch is 5 rows for good: a matched probe with a new signature, which a plan can replay.
        // It compiles once and every later step replays it; nothing runs outside the plans.
        int[] batches = { 8, 8, 5, 5, 5, 5, 5, 5 };
        var eager = RunLogged(compile: false, steps: batches.Length, readsOn: _ => false, batchOf: t => batches[t]);
        var compiled = RunLogged(compile: true, steps: batches.Length, readsOn: _ => false, batchOf: t => batches[t]);

        Assert.Equal(batches.Length, compiled.FusedSteps);
        Assert.Equal(0, compiled.OutsidePlanSteps);
        Assert.Equal(2, compiled.Compiles);
        Assert.Equal(2, compiled.PlanCount);
        Assert.False(compiled.Dynamic);
        AssertSameWeights(eager.Weights, compiled.Weights);
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

    [Theory]
    [InlineData("learning-rate")]
    [InlineData("beta")]
    [InlineData("l2")]
    [InlineData("scheduler")]
    [InlineData("parameter-identity")]
    public void Declared_hybrid_refuses_configuration_drift_before_forward_or_optimizer_mutation(string drift)
    {
        var original = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = true });
            var weights = Clone(InitialWeights());
            var options = new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>
            {
                InitialLearningRate = 0.01, EnableGradientClipping = false,
            };
            var optimizer = new AdamOptimizer<double, Tensor<double>, Tensor<double>>(null, options);
            var owner = TapeTrainingStepper<double>.ForOwner(optimizer);
            int step = 0, evaluations = 0;
            var inner = Objective(weights, () => step, _ => 8);
            Tensor<double> ObjectiveWithCounter() { evaluations++; return inner(); }
            for (; step < 2; step++) owner.Step(ClosureRequest(weights, optimizer, ObjectiveWithCounter));
            Assert.Equal(2, owner.FusedStepCount);
            Assert.True(owner.Session.IsCommitted);
            switch (drift)
            {
                case "learning-rate":
                    typeof(GradientBasedOptimizerBase<double, Tensor<double>, Tensor<double>>)
                        .GetMethod("SetLearningRate", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!
                        .Invoke(optimizer, new object[] { 0.02 });
                    break;
                case "beta": options.Beta1 = 0.8; break;
                case "l2":
                    optimizer.SetRegularization(new AiDotNet.Regularization.L2Regularization<double, Tensor<double>, Tensor<double>>(
                        new RegularizationOptions { Strength = 0.1 }));
                    break;
                case "scheduler":
                    optimizer.SetLearningRateScheduler(new AiDotNet.LearningRateSchedulers.ConstantLRScheduler(0.01));
                    break;
                case "parameter-identity": weights[1] = weights[1].Clone(); break;
            }
            var before = Clone(weights);
            byte[] state = optimizer.Serialize();
            int evaluationsBefore = evaluations;
            Assert.Throws<InvalidOperationException>(() => owner.Step(ClosureRequest(weights, optimizer,
                ObjectiveWithCounter, clip: 1e-3, declaredBreak: true, allowBreak: true)));
            Assert.Equal(evaluationsBefore, evaluations);
            for (int p = 0; p < weights.Length; p++) Assert.Equal(before[p].ToArray(), weights[p].ToArray());
            Assert.Equal(state, optimizer.Serialize());
            Assert.Equal(2, owner.FusedStepCount);
            Assert.Equal(0, owner.HybridStepCount);
        }
        finally
        {
            TensorCodecOptions.SetCurrent(original);
            CompiledTapeTrainingStep<double>.Invalidate();
        }
    }

    private sealed class UnknownReplayState
    {
        public double Scale = 1;
    }

    [Theory]
    [InlineData("host-read")]
    [InlineData("unknown-state")]
    [InlineData("unexpected-exception")]
    [InlineData("declared-without-opt-in")]
    [InlineData("policy-without-reason")]
    public void Unexpected_closure_breaks_remain_visible_and_never_apply_hybrid_updates(string failure)
    {
        var original = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = true });
            var weights = Clone(InitialWeights());
            var optimizer = Adam();
            var owner = TapeTrainingStepper<double>.ForOwner(optimizer);
            int step = 0;
            var inner = Objective(weights, () => step, _ => 8);
            var unknown = new UnknownReplayState();
            Tensor<double> ObjectiveWithBreak()
            {
                var loss = inner();
                if (step < 2) return loss;
                if (failure == "unexpected-exception") throw new InvalidOperationException("deliberate unexpected objective failure");
                if (failure == "unknown-state" && AiDotNet.Tensors.Engines.Compilation.GraphMode.Current is { } scope)
                    return scope.RecordUnary(AiDotNet.Tensors.Engines.Compilation.LazyNodeType.Custom, "UnknownReplayState",
                        loss, loss.Shape.ToArray(), (engine, output) =>
                            AiDotNet.Tensors.Engines.DirectGpuTensorEngine.CopyResultInto(engine,
                                engine.TensorMultiplyScalar(loss, unknown.Scale), output));
                if (failure is "host-read" or "policy-without-reason" or "declared-without-opt-in") _ = loss.ToArray()[0];
                return loss;
            }
            for (; step < 2; step++) owner.Step(ClosureRequest(weights, optimizer, ObjectiveWithBreak));
            Assert.Equal(2, owner.FusedStepCount);
            var before = Clone(weights);
            byte[] state = optimizer.Serialize();
            var error = Record.Exception(() => owner.Step(ClosureRequest(weights, optimizer, ObjectiveWithBreak,
                declaredBreak: failure == "declared-without-opt-in",
                allowBreak: failure == "policy-without-reason")));
            Assert.NotNull(error);
            Assert.IsAssignableFrom<InvalidOperationException>(error);
            if (failure == "unexpected-exception") Assert.Contains("deliberate unexpected objective failure", error!.ToString());
            for (int p = 0; p < weights.Length; p++) Assert.Equal(before[p].ToArray(), weights[p].ToArray());
            Assert.Equal(state, optimizer.Serialize());
            Assert.Equal(2, owner.FusedStepCount);
            Assert.Equal(0, owner.HybridStepCount);
        }
        finally
        {
            TensorCodecOptions.SetCurrent(original);
            CompiledTapeTrainingStep<double>.Invalidate();
        }
    }
}
