using System;
using System.Collections.Generic;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Training;

/// <summary>
/// The shared training step every base class trains through: per-step random draws in compiled plans, graph breaks,
/// and the replay-agreement check.
/// </summary>
public class SharedTrainingStepTests
{
    private const int Features = 4;
    private const int Outputs = 3;
    private const int Batch = 5;

    [Fact]
    public void RecordedDrawIsRefilledInPlaceForItsPlan()
    {
        var plan = new object();
        Tensor<float> draw;
        using (var recording = CompiledStepRandom<float>.BeginRecording())
        {
            draw = CompiledStepRandom<float>.StandardNormal(new[] { 4, 4 }, new Random(7));
            CompiledStepRandom<float>.Attach(plan, recording);
        }

        Assert.Equal(1, CompiledStepRandom<float>.DrawCount(plan));
        var before = draw.ToArray();
        CompiledStepRandom<float>.RedrawFor(plan);
        var after = draw.ToArray();

        Assert.Equal(before.Length, after.Length);
        Assert.NotEqual(before, after);
    }

    [Fact]
    public void DrawOutsideARecordingIsNotRetained()
    {
        var plan = new object();
        CompiledStepRandom<float>.StandardNormal(new[] { 3 }, new Random(1));
        using (var recording = CompiledStepRandom<float>.BeginRecording())
            CompiledStepRandom<float>.Attach(plan, recording);

        Assert.Equal(0, CompiledStepRandom<float>.DrawCount(plan));
    }

    /// <summary>
    /// A forward that adds a fresh noise draw each step. The fused plan must redraw it on every replay, so its loss
    /// trajectory equals the eager one consuming the same generator; a plan replaying its traced draw diverges from
    /// step 2 on.
    /// </summary>
    [Fact]
    public void FusedPlanRedrawsPerStepNoiseLikeTheEagerStep()
    {
        var (fusedLayer, eagerLayer) = TwinLayers();
        var fusedRandom = new Random(123);
        var eagerRandom = new Random(123);
        var fusedOptimizer = Sgd();
        var eagerOptimizer = Sgd();
        var stepper = new TapeTrainingStepper<float>(new object());
        var data = new Random(5);

        for (int step = 0; step < 4; step++)
        {
            var input = RandomTensor(data, Batch, Features);
            var target = RandomTensor(data, Batch, Outputs);

            float fused = stepper.Step(NoisyRequest(fusedLayer, input, target, fusedRandom, fusedOptimizer));
            Assert.True(stepper.LastStepFused, $"step {step}: fused path did not engage ({stepper.Session.LastMissReason}).");
            float eager = TapeTrainingStepper<float>.EagerStep(NoisyRequest(eagerLayer, input, target, eagerRandom, eagerOptimizer));

            Assert.True(
                Math.Abs(fused - eager) <= 1e-4f * Math.Max(1f, Math.Abs(eager)),
                $"step {step}: fused loss {fused} differs from eager loss {eager}; the plan is not redrawing its noise.");
        }
    }

    [Fact]
    public void GraphBreakKeepsTheStepEager()
    {
        var (layer, _) = TwinLayers();
        var stepper = new TapeTrainingStepper<float>(new object());
        var data = new Random(9);
        var before = layer.GetParameters().ToArray();

        stepper.Step(new FusedTrainingStepRequest<float>
        {
            Layers = new ITrainableLayer<float>[] { layer },
            Input = RandomTensor(data, Batch, Features),
            Target = RandomTensor(data, Batch, Outputs),
            Forward = layer.Forward,
            ComputeLoss = MeanSquaredError,
            Optimizer = Sgd(),
            GraphBreakReason = "test: data-dependent forward",
        });

        Assert.False(stepper.LastStepFused);
        Assert.Contains("graph break", stepper.Session.LastMissReason ?? string.Empty);
        Assert.NotEqual(before, layer.GetParameters().ToArray());
    }

    /// <summary>
    /// A forward that reads its input on the host bakes the first batch's value into the trace. With the agreement
    /// check on, the first replay on new data is caught, its update undone, and the model stays eager.
    /// </summary>
    [Fact]
    public void ReplayDisagreementFallsBackToEager()
    {
        var (layer, _) = TwinLayers();
        var stepper = new TapeTrainingStepper<float>(new object());
        var optimizer = Sgd();
        var data = new Random(11);
        var engine = AiDotNetEngine.Current;

        Tensor<float> HostScaledForward(Tensor<float> x)
        {
            float scale = 1f + 10f * Math.Abs(x[0]);
            return engine.TensorMultiplyScalar(layer.Forward(x), scale);
        }

        FusedTrainingStepRequest<float> Request(Tensor<float> input, Tensor<float> target) => new()
        {
            Layers = new ITrainableLayer<float>[] { layer },
            Input = input,
            Target = target,
            Forward = HostScaledForward,
            ComputeLoss = MeanSquaredError,
            Optimizer = optimizer,
            VerifyReplayAgreement = true,
        };

        stepper.Step(Request(RandomTensor(data, Batch, Features), RandomTensor(data, Batch, Outputs)));
        Assert.True(stepper.LastStepFused, $"first step did not run fused ({stepper.Session.LastMissReason}).");

        var second = RandomTensor(data, Batch, Features);
        second[0] = 3f;
        stepper.Step(Request(second, RandomTensor(data, Batch, Outputs)));

        Assert.False(stepper.LastStepFused);
        Assert.True(stepper.Session.IsDisabled);
        Assert.Contains("disagrees", stepper.Session.LastMissReason ?? string.Empty);
    }

    /// <summary>
    /// A GAN steps two parameter groups (critic, generator) with two optimizers through one owner. Each group must keep
    /// its own fused plan and optimizer moments: Adam trajectories on the fused path equal two eager Adam trajectories.
    /// With one plan per owner, every switch of optimizer dropped the other group's plan and restarted its moments.
    /// </summary>
    [Fact]
    public void TwoParameterGroupsOfOneOwnerKeepTheirOwnPlansAndMoments()
    {
        var (fusedA, eagerA) = TwinLayers();
        var (fusedB, eagerB) = TwinLayers();
        var fusedAdamA = Adam();
        var fusedAdamB = Adam();
        var eagerAdamA = Adam();
        var eagerAdamB = Adam();
        var owner = new object();
        var data = new Random(21);

        for (int step = 0; step < 4; step++)
        {
            foreach (var (fused, eager, fusedAdam, eagerAdam, name) in new[]
            {
                (fusedA, eagerA, fusedAdamA, eagerAdamA, "A"),
                (fusedB, eagerB, fusedAdamB, eagerAdamB, "B"),
            })
            {
                var input = RandomTensor(data, Batch, Features);
                var target = RandomTensor(data, Batch, Outputs);
                var stepper = TapeTrainingStepper<float>.ForOwner(owner);
                float fusedLoss = stepper.Step(PlainRequest(fused, input, target, fusedAdam));
                float eagerLoss = TapeTrainingStepper<float>.EagerStep(PlainRequest(eager, input, target, eagerAdam));
                Assert.True(
                    Math.Abs(fusedLoss - eagerLoss) <= 1e-4f * Math.Max(1f, Math.Abs(eagerLoss)),
                    $"step {step}, group {name}: fused loss {fusedLoss} differs from eager loss {eagerLoss}; "
                    + "the group's plan or optimizer moments did not survive the other group's step.");
            }
        }

        var finalA = fusedA.GetParameters().ToArray();
        var referenceA = eagerA.GetParameters().ToArray();
        for (int i = 0; i < finalA.Length; i++)
            Assert.True(Math.Abs(finalA[i] - referenceA[i]) <= 1e-4f, $"group A parameter {i}: {finalA[i]} vs {referenceA[i]}.");
    }

    /// <summary>
    /// The shared accumulation step (one tape per micro-batch, one update from the mean gradient) must equal one
    /// update from a single tape over the mean of the same micro-objectives.
    /// </summary>
    [Fact]
    public void AccumulatedStepEqualsOneUpdateFromTheMeanObjective()
    {
        var (accumulated, reference) = TwinLayers();
        var data = new Random(31);
        var inputs = new Tensor<float>[3];
        var targets = new Tensor<float>[3];
        for (int i = 0; i < inputs.Length; i++)
        {
            inputs[i] = RandomTensor(data, Batch, Features);
            targets[i] = RandomTensor(data, Batch, Outputs);
        }

        float accumulatedLoss = TapeTrainingStepper<float>.EagerAccumulatedObjectiveStep(
            accumulated.GetTrainableParameters(),
            inputs.Length,
            i => MeanSquaredError(accumulated.Forward(inputs[i]), targets[i]),
            Sgd());

        var engine = AiDotNetEngine.Current;
        float referenceLoss = TapeTrainingStepper<float>.EagerObjectiveStep(
            reference.GetTrainableParameters(),
            () =>
            {
                Tensor<float>? total = null;
                for (int i = 0; i < inputs.Length; i++)
                {
                    var loss = MeanSquaredError(reference.Forward(inputs[i]), targets[i]);
                    total = total is null ? loss : engine.TensorAdd(total, loss);
                }
                return engine.TensorMultiplyScalar(total ?? new Tensor<float>(new[] { 1 }), 1f / inputs.Length);
            },
            Sgd());

        Assert.True(Math.Abs(accumulatedLoss - referenceLoss) <= 1e-5f, $"loss {accumulatedLoss} vs {referenceLoss}.");
        var after = accumulated.GetParameters().ToArray();
        var expected = reference.GetParameters().ToArray();
        for (int i = 0; i < after.Length; i++)
            Assert.True(Math.Abs(after[i] - expected[i]) <= 1e-5f, $"parameter {i}: {after[i]} vs {expected[i]}.");
    }

    /// <summary>
    /// A linear critic D(x) = w . x has input gradient w at every interpolate, so the WGAN-GP penalty is exactly
    /// (||w|| - 1)^2 whatever the batches and the interpolation weights.
    /// </summary>
    [Fact]
    public void GradientPenaltyOfALinearCriticIsTheSquaredNormDeviation()
    {
        var engine = AiDotNetEngine.Current;
        var weights = new Tensor<float>(new[] { Features, 1 });
        for (int i = 0; i < Features; i++) weights[i] = 1f; // ||w|| = 2 for four features
        var data = new Random(41);
        var real = RandomTensor(data, Batch, Features);
        var fake = RandomTensor(data, Batch, Features);

        var penalty = WassersteinCriticStep<float>.GradientPenalty(
            engine, real, fake, x => engine.TensorMatMul(x, weights), new Random(3));

        double norm = Math.Sqrt(Features);
        Assert.Equal((norm - 1.0) * (norm - 1.0), penalty[0], 3);
    }

    /// <summary>
    /// The critic step replayed by one owner must follow the same trajectory as one traced fresh on every step: the
    /// penalty is recomputed from each step's batch and redraws its interpolation weights. The copies this step
    /// replaced captured the first batch in the penalty, so their replays penalized it for the whole run.
    /// </summary>
    [Fact]
    public void CriticStepReplayMatchesAFreshTraceOnEveryBatch()
    {
        var (replayed, fresh) = CriticTwins();
        var replayedRandom = new Random(42);
        var freshRandom = new Random(42);
        var replayedOptimizer = Sgd();
        var replayOwner = new object();
        WganGpFusedStep<float>? replayedPrimitive = null;
        var data = new Random(43);

        for (int step = 0; step < 4; step++)
        {
            var real = RandomTensor(data, Batch, Features);
            var fake = RandomTensor(data, Batch, Features);
            float replayedLoss = WassersteinCriticStep<float>.Step(
                replayOwner, new ILayer<float>[] { replayed }, real, fake, replayed.Forward, 10.0,
                replayedOptimizer, replayedRandom, ref replayedPrimitive);

            // A fresh owner and a fresh SGD each step: always a first trace, never a stale replay (SGD has no state).
            WganGpFusedStep<float>? freshPrimitive = null;
            float freshLoss = WassersteinCriticStep<float>.Step(
                new object(), new ILayer<float>[] { fresh }, real, fake, fresh.Forward, 10.0,
                Sgd(), freshRandom, ref freshPrimitive);

            Assert.False(float.IsNaN(replayedLoss) || float.IsInfinity(replayedLoss), $"step {step}: critic loss {replayedLoss}.");
            Assert.True(
                Math.Abs(replayedLoss - freshLoss) <= 1e-3f * Math.Max(1f, Math.Abs(freshLoss)),
                $"step {step}: replayed critic loss {replayedLoss} differs from a fresh trace {freshLoss}.");
        }
    }

    /// <summary>
    /// With no noise and a clip norm no example reaches, the shared DP-SGD step must reduce to one update from the
    /// mean per-example gradient: the accumulation step's result.
    /// </summary>
    [Fact]
    public void DpSgdStepWithoutNoiseOrActiveClipEqualsTheMeanGradientStep()
    {
        var (privateLayer, reference) = TwinLayers();
        var data = new Random(51);
        var inputs = new Tensor<float>[4];
        var targets = new Tensor<float>[4];
        for (int i = 0; i < inputs.Length; i++)
        {
            inputs[i] = RandomTensor(data, 1, Features);
            targets[i] = RandomTensor(data, 1, Outputs);
        }

        float privateLoss = DpSgdTrainingStep<float>.Step(
            privateLayer.GetTrainableParameters(),
            inputs.Length,
            i => new[] { inputs[i], targets[i] },
            slots => privateLayer.Forward(slots[0]),
            (output, slots) => MeanSquaredError(output, slots[1]),
            clipNorm: 1e6,
            noiseMultiplier: 0.0,
            random: new Random(1),
            optimizer: Sgd());

        float referenceLoss = TapeTrainingStepper<float>.EagerAccumulatedObjectiveStep(
            reference.GetTrainableParameters(),
            inputs.Length,
            i => MeanSquaredError(reference.Forward(inputs[i]), targets[i]),
            Sgd());

        Assert.True(Math.Abs(privateLoss - referenceLoss) <= 1e-5f, $"loss {privateLoss} vs {referenceLoss}.");
        var after = privateLayer.GetParameters().ToArray();
        var expected = reference.GetParameters().ToArray();
        for (int i = 0; i < after.Length; i++)
            Assert.True(Math.Abs(after[i] - expected[i]) <= 1e-5f, $"parameter {i}: {after[i]} vs {expected[i]}.");
    }
    private static (DenseLayer<float> First, DenseLayer<float> Second) CriticTwins()
    {
        var first = new DenseLayer<float>(1);
        var second = new DenseLayer<float>(1);
        var warm = new Tensor<float>(new[] { 2 * Batch, Features });
        first.Forward(warm);
        second.Forward(warm);
        second.SetParameters(first.GetParameters());
        return (first, second);
    }

    private static FusedTrainingStepRequest<float> PlainRequest(
        DenseLayer<float> layer, Tensor<float> input, Tensor<float> target,
        IGradientBasedOptimizer<float, Tensor<float>, Tensor<float>> optimizer)
        => new()
        {
            Layers = new ITrainableLayer<float>[] { layer },
            Input = input,
            Target = target,
            Forward = layer.Forward,
            ComputeLoss = MeanSquaredError,
            Optimizer = optimizer,
        };

    private static AdamOptimizer<float, Tensor<float>, Tensor<float>> Adam()
        => new(model: null, options: new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
        {
            InitialLearningRate = 0.01,
            UseAdaptiveLearningRate = false,
            UseAdaptiveBetas = false,
        });
    private static FusedTrainingStepRequest<float> NoisyRequest(
        DenseLayer<float> layer, Tensor<float> input, Tensor<float> target, Random random,
        IGradientBasedOptimizer<float, Tensor<float>, Tensor<float>> optimizer)
    {
        var engine = AiDotNetEngine.Current;
        return new FusedTrainingStepRequest<float>
        {
            Layers = new ITrainableLayer<float>[] { layer },
            Input = input,
            Target = target,
            Forward = x =>
            {
                var output = layer.Forward(x);
                return engine.TensorAdd(output, CompiledStepRandom<float>.StandardNormal(output._shape, random));
            },
            ComputeLoss = MeanSquaredError,
            Optimizer = optimizer,
        };
    }

    private static (DenseLayer<float> First, DenseLayer<float> Second) TwinLayers()
    {
        var first = new DenseLayer<float>(Outputs);
        var second = new DenseLayer<float>(Outputs);
        var warm = new Tensor<float>(new[] { Batch, Features });
        first.Forward(warm);
        second.Forward(warm);
        second.SetParameters(first.GetParameters());
        return (first, second);
    }

    private static StochasticGradientDescentOptimizer<float, Tensor<float>, Tensor<float>> Sgd()
        => new(model: null, options: new StochasticGradientDescentOptimizerOptions<float, Tensor<float>, Tensor<float>>
        {
            InitialLearningRate = 0.05,
            UseAdaptiveLearningRate = false,
        });

    private static Tensor<float> MeanSquaredError(Tensor<float> predicted, Tensor<float> target)
    {
        var engine = AiDotNetEngine.Current;
        var difference = engine.TensorSubtract(predicted, target);
        return engine.TensorMultiplyScalar(
            engine.ReduceSum(engine.TensorMultiply(difference, difference), null), 1f / difference.Length);
    }

    private static Tensor<float> RandomTensor(Random random, int rows, int columns)
    {
        var tensor = new Tensor<float>(new[] { rows, columns });
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = (float)(random.NextDouble() * 2.0 - 1.0);
        return tensor;
    }
}