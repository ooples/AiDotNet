using System;
using System.Collections.Generic;
using System.Linq;
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
            draw = CompiledStepRandom<float>.StandardNormal(new[] { 4, 4 }, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(7));
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
        CompiledStepRandom<float>.StandardNormal(new[] { 3 }, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(1));
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
        var fusedRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(123);
        var eagerRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(123);
        var fusedOptimizer = Sgd();
        var eagerOptimizer = Sgd();
        var stepper = new TapeTrainingStepper<float>(new object());
        var data = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(5);

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
        var data = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(9);
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
        var data = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(11);
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
    /// A plan whose traced step already computes something other than the eager forward. The agreement check used to
    /// look only at the first replay, so the traced step's update went into the weights before anything was compared
    /// (CLAP's first compiled step gathered through indices the trace had not computed yet, and its replays were
    /// right, so nothing was ever caught). With the traced step checked, that step is undone and runs eagerly, and
    /// nothing of it is left behind: the forward has step state (a batch norm's running statistics, which the traced
    /// step advanced and the eager step would advance a second time) and the optimizer has a position (Adam's step and
    /// a per-batch schedule), and after every step all of it equals a twin that only ever ran eager.
    /// </summary>
    [Fact]
    public void TracedStepDisagreementIsUndoneBeforeItTrains()
    {
        var model = NormTwin.Create();
        var twin = model.Copy();
        var stepper = new TapeTrainingStepper<float>(new object());
        var optimizer = ScheduledAdam();
        var twinOptimizer = ScheduledAdam();
        var data = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(11);
        var engine = AiDotNetEngine.Current;

        Tensor<float> WrongWhenTraced(Tensor<float> x)
        {
            var output = model.Forward(x);
            return AiDotNet.Tensors.Engines.Autodiff.GradientRecording.IsTrainingTrace
                ? engine.TensorMultiplyScalar(output, 3f)
                : output;
        }

        for (int step = 0; step < 3; step++)
        {
            var input = RandomTensor(data, Batch, Features);
            var target = RandomTensor(data, Batch, Outputs);
            float loss = stepper.Step(new FusedTrainingStepRequest<float>
            {
                Layers = model.Layers,
                Input = input,
                Target = target,
                Forward = WrongWhenTraced,
                ComputeLoss = MeanSquaredError,
                Optimizer = optimizer,
                VerifyReplayAgreement = true,
            });

            Assert.False(stepper.LastStepFused);
            Assert.True(stepper.Session.IsDisabled);
            if (step == 0) Assert.Contains("traced step", stepper.Session.LastMissReason ?? string.Empty);

            float eagerLoss = TapeTrainingStepper<float>.EagerStep(new FusedTrainingStepRequest<float>
            {
                Layers = twin.Layers,
                Input = input,
                Target = target,
                Forward = twin.Forward,
                ComputeLoss = MeanSquaredError,
                Optimizer = twinOptimizer,
            });
            Assert.Equal(eagerLoss, loss);
            AssertSameState(twin, twinOptimizer, model, optimizer, step);
        }
    }

    /// <summary>
    /// The other undo: the traced step agrees and trains, the first replay on new data does not (the forward read its
    /// input on the host). The session then puts the model back before the traced step, redoes that step on the eager
    /// tape and leaves the replay's batch to the caller's eager step. The traced step and the replay both advanced the
    /// running statistics and the optimizer's position; after the undo, and on the steps after it, everything equals a
    /// twin that only ever ran eager.
    /// </summary>
    [Fact]
    public void FirstReplayDisagreementLeavesNoStepStateBehind()
    {
        var model = NormTwin.Create();
        var twin = model.Copy();
        var stepper = new TapeTrainingStepper<float>(new object());
        var optimizer = ScheduledAdam();
        var twinOptimizer = ScheduledAdam();
        var data = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(11);
        var engine = AiDotNetEngine.Current;

        Func<Tensor<float>, Tensor<float>> HostScaled(NormTwin of) => x =>
        {
            float scale = 1f + 10f * Math.Abs(x[0]);
            return engine.TensorMultiplyScalar(of.Forward(x), scale);
        };

        for (int step = 0; step < 4; step++)
        {
            var input = RandomTensor(data, Batch, Features);
            input[0] = step == 0 ? 0.25f : 3f + step;
            var target = RandomTensor(data, Batch, Outputs);
            stepper.Step(new FusedTrainingStepRequest<float>
            {
                Layers = model.Layers,
                Input = input,
                Target = target,
                Forward = HostScaled(model),
                ComputeLoss = MeanSquaredError,
                Optimizer = optimizer,
                VerifyReplayAgreement = true,
            });
            if (step == 0)
                Assert.True(stepper.LastStepFused, $"the traced step did not run fused ({stepper.Session.LastMissReason}).");
            else
                Assert.False(stepper.LastStepFused);
            if (step == 1) Assert.Contains("disagrees", stepper.Session.LastMissReason ?? string.Empty);

            TapeTrainingStepper<float>.EagerStep(new FusedTrainingStepRequest<float>
            {
                Layers = twin.Layers,
                Input = input,
                Target = target,
                Forward = HostScaled(twin),
                ComputeLoss = MeanSquaredError,
                Optimizer = twinOptimizer,
            });
            // After the traced step the model is ahead by the plan's own rounding only; from the undo on it is the
            // eager twin exactly.
            if (step > 0) AssertSameState(twin, twinOptimizer, model, optimizer, step);
        }
    }

    /// <summary>
    /// A declined plan's steps drew from the forward's generator. The eager steps that take their place reuse those
    /// draws, so the generator ends where an eager-only twin's does and the weights are that twin's: on the traced-step
    /// decline (the noise is in the loss, which the check's draw-free forward cannot reproduce) and on the first-replay
    /// undo (the noise is drawn but does not reach the loss; the host-read scale makes the replay disagree).
    /// </summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void UndoneStepsDoNotConsumeTheGeneratorTwice(bool declineOnTrace)
    {
        var (layer, eagerLayer) = TwinLayers();
        var stepper = new TapeTrainingStepper<float>(new object());
        var optimizer = Sgd();
        var twinOptimizer = Sgd();
        var data = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(11);
        var random = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(23);
        var twinRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(23);
        var engine = AiDotNetEngine.Current;

        Func<Tensor<float>, Tensor<float>> Noisy(DenseLayer<float> of, Random generator) => x =>
        {
            var output = of.Forward(x);
            var noise = CompiledStepRandom<float>.StandardNormal(output._shape, generator);
            if (declineOnTrace) return engine.TensorAdd(output, noise);
            float scale = 1f + 10f * Math.Abs(x[0]);
            return engine.TensorAdd(engine.TensorMultiplyScalar(output, scale), engine.TensorMultiplyScalar(noise, 0f));
        };

        for (int step = 0; step < 3; step++)
        {
            var input = RandomTensor(data, Batch, Features);
            input[0] = step == 0 ? 0.25f : 3f + step;
            var target = RandomTensor(data, Batch, Outputs);
            float loss = stepper.Step(new FusedTrainingStepRequest<float>
            {
                Layers = new ITrainableLayer<float>[] { layer },
                Input = input,
                Target = target,
                Forward = Noisy(layer, random),
                ComputeLoss = MeanSquaredError,
                Optimizer = optimizer,
                VerifyReplayAgreement = true,
            });
            Assert.Equal(!declineOnTrace && step == 0, stepper.LastStepFused);

            float eagerLoss = TapeTrainingStepper<float>.EagerStep(new FusedTrainingStepRequest<float>
            {
                Layers = new ITrainableLayer<float>[] { eagerLayer },
                Input = input,
                Target = target,
                Forward = Noisy(eagerLayer, twinRandom),
                ComputeLoss = MeanSquaredError,
                Optimizer = twinOptimizer,
            });
            if (declineOnTrace || step > 0)
            {
                Assert.Equal(eagerLoss, loss);
                Assert.Equal(eagerLayer.GetParameters().ToArray(), layer.GetParameters().ToArray());
            }
        }

        Assert.True(stepper.Session.IsDisabled);
        Assert.Equal(twinRandom.NextDouble(), random.NextDouble());
    }

    /// <summary>
    /// A vector quantizer's EMA codebooks are step state too: a training forward moves them, with no optimizer
    /// involved. A traced step that is declined moved them once, and the eager step of the same batch would move them
    /// again from there. After the undo every codebook buffer equals an eager-only twin's.
    /// </summary>
    [Fact]
    public void TracedStepDisagreementRestoresAnEmaCodebook()
    {
        const int dim = Outputs, frames = Batch;
        var (dense, twinDense) = TwinLayers();
        var quantizer = new AiDotNet.Audio.Codecs.ResidualVectorQuantizerLayer<float>(dim, quantizers: 2, bins: 4,
            kmeansIterations: 0, deadCodeThreshold: 0);
        var twinQuantizer = new AiDotNet.Audio.Codecs.ResidualVectorQuantizerLayer<float>(dim, quantizers: 2, bins: 4,
            kmeansIterations: 0, deadCodeThreshold: 0);
        CopyBuffers(quantizer, twinQuantizer);
        quantizer.SetTrainingMode(true);
        twinQuantizer.SetTrainingMode(true);
        var stepper = new TapeTrainingStepper<float>(new object());
        var optimizer = Sgd();
        var twinOptimizer = Sgd();
        var data = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(11);
        var engine = AiDotNetEngine.Current;

        Func<Tensor<float>, Tensor<float>> Quantized(DenseLayer<float> of,
            AiDotNet.Audio.Codecs.ResidualVectorQuantizerLayer<float> codebooks, bool wrongWhenTraced) => x =>
        {
            var rows = engine.Reshape(engine.TensorTranspose(of.Forward(x)), new[] { 1, dim, frames });
            var output = codebooks.Forward(rows);
            return wrongWhenTraced && AiDotNet.Tensors.Engines.Autodiff.GradientRecording.IsTrainingTrace
                ? engine.TensorMultiplyScalar(output, 3f)
                : output;
        };

        var before = Buffers(quantizer);
        for (int step = 0; step < 2; step++)
        {
            var input = RandomTensor(data, frames, Features);
            var target = new Tensor<float>(new[] { 1, dim, frames });
            for (int i = 0; i < target.Length; i++) target[i] = (float)(data.NextDouble() * 2.0 - 1.0);
            float loss = stepper.Step(new FusedTrainingStepRequest<float>
            {
                Layers = new ITrainableLayer<float>[] { dense, quantizer },
                Input = input,
                Target = target,
                Forward = Quantized(dense, quantizer, wrongWhenTraced: true),
                ComputeLoss = MeanSquaredError,
                Optimizer = optimizer,
                VerifyReplayAgreement = true,
            });
            Assert.False(stepper.LastStepFused);
            if (step == 0) Assert.Contains("traced step", stepper.Session.LastMissReason ?? string.Empty);

            float eagerLoss = TapeTrainingStepper<float>.EagerStep(new FusedTrainingStepRequest<float>
            {
                Layers = new ITrainableLayer<float>[] { twinDense, twinQuantizer },
                Input = input,
                Target = target,
                Forward = Quantized(twinDense, twinQuantizer, wrongWhenTraced: false),
                ComputeLoss = MeanSquaredError,
                Optimizer = twinOptimizer,
            });
            Assert.Equal(eagerLoss, loss);
            Assert.Equal(Buffers(twinQuantizer), Buffers(quantizer));
            Assert.Equal(twinDense.GetParameters().ToArray(), dense.GetParameters().ToArray());
        }

        Assert.NotEqual(before, Buffers(quantizer));
    }

    private static float[] Buffers(LayerBase<float> layer)
        => layer.GetRegisteredBuffers().SelectMany(buffer => buffer.Tensor.ToArray()).ToArray();

    private static void CopyBuffers(LayerBase<float> from, LayerBase<float> to)
    {
        var source = from.GetRegisteredBuffers();
        var destination = to.GetRegisteredBuffers();
        Assert.Equal(source.Count, destination.Count);
        for (int i = 0; i < source.Count; i++)
            source[i].Tensor.AsSpan().CopyTo(destination[i].Tensor.AsWritableSpan());
    }

    private static void AssertSameState(NormTwin expected, AdamOptimizer<float, Tensor<float>, Tensor<float>> expectedOptimizer,
        NormTwin actual, AdamOptimizer<float, Tensor<float>, Tensor<float>> actualOptimizer, int step)
    {
        Assert.Equal(expected.Norm.GetRunningMean().ToArray(), actual.Norm.GetRunningMean().ToArray());
        Assert.Equal(expected.Norm.GetRunningVariance().ToArray(), actual.Norm.GetRunningVariance().ToArray());
        Assert.Equal(expected.Norm.GetParameters().ToArray(), actual.Norm.GetParameters().ToArray());
        Assert.Equal(expected.Dense.GetParameters().ToArray(), actual.Dense.GetParameters().ToArray());
        Assert.True(expectedOptimizer.CurrentStep == actualOptimizer.CurrentStep,
            $"step {step}: the optimizer is at step {actualOptimizer.CurrentStep}, the eager twin at {expectedOptimizer.CurrentStep}.");
        Assert.Equal(step + 1, actualOptimizer.CurrentStep);
        Assert.Equal(expectedOptimizer.GetCurrentLearningRate(), actualOptimizer.GetCurrentLearningRate());
    }

    // A batch norm into a dense layer: trained weights plus running statistics that every training forward advances.
    private sealed class NormTwin
    {
        private NormTwin(BatchNormalizationLayer<float> norm, DenseLayer<float> dense)
        {
            Norm = norm;
            Dense = dense;
            Layers = new ITrainableLayer<float>[] { norm, dense };
        }

        public BatchNormalizationLayer<float> Norm { get; }

        public DenseLayer<float> Dense { get; }

        public ITrainableLayer<float>[] Layers { get; }

        public Tensor<float> Forward(Tensor<float> x) => Dense.Forward(Norm.Forward(x));

        public static NormTwin Create()
        {
            var twin = new NormTwin(new BatchNormalizationLayer<float>(), new DenseLayer<float>(Outputs));
            twin.Norm.SetTrainingMode(false);
            twin.Forward(new Tensor<float>(new[] { Batch, Features }));
            twin.Norm.SetTrainingMode(true);
            return twin;
        }

        public NormTwin Copy()
        {
            var copy = Create();
            copy.Norm.SetParameters(Norm.GetParameters());
            copy.Dense.SetParameters(Dense.GetParameters());
            CopyBuffers(Norm, copy.Norm);
            return copy;
        }
    }

    // Adam with a per-batch schedule: an optimizer whose position (step count, learning rate) a step advances.
    private static AdamOptimizer<float, Tensor<float>, Tensor<float>> ScheduledAdam()
        => new(model: null, options: new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
        {
            InitialLearningRate = 0.01,
            UseAdaptiveLearningRate = false,
            UseAdaptiveBetas = false,
            LearningRateScheduler = new AiDotNet.LearningRateSchedulers.ExponentialLRScheduler(0.01, gamma: 0.5),
            SchedulerStepMode = AiDotNet.LearningRateSchedulers.SchedulerStepMode.StepPerBatch,
        });

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
        var data = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(21);

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
        var data = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(31);
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
        var data = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(41);
        var real = RandomTensor(data, Batch, Features);
        var fake = RandomTensor(data, Batch, Features);

        var penalty = WassersteinCriticStep<float>.GradientPenalty(
            engine, real, fake, x => engine.TensorMatMul(x, weights), AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(3));

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
        var replayedRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(42);
        var freshRandom = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(42);
        var replayedOptimizer = Sgd();
        var replayOwner = new object();
        WganGpFusedStep<float>? replayedPrimitive = null;
        var data = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(43);

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
        var data = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(51);
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
            random: AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(1),
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

    /// <summary>
    /// Replacing a layer's weight tensor in the same slot and shape must rebind the compiled plan before
    /// its next update, preserving optimizer state. The new tensor must train immediately, and the old
    /// tensor must remain unchanged throughout the later persistence rechecks.
    /// </summary>
    [Fact]
    public void ReplacedWeightTensorIsTrainedAfterTheNextRecheck()
    {
        var layer = new SwappableDenseLayer(Outputs);
        layer.Forward(new Tensor<float>(new[] { Batch, Features }));
        var stepper = new TapeTrainingStepper<float>(new object());
        var optimizer = Adam();
        var data = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(31);
        int interval = FusedTrainingSession<float>.PersistenceRecheckInterval;

        for (int step = 0; step < interval + 4; step++)
            stepper.Step(PlainRequest(layer, RandomTensor(data, Batch, Features), RandomTensor(data, Batch, Outputs), optimizer));
        var originalWeight = layer.CurrentWeights;
        Assert.True(stepper.LastStepFused, $"the fused path did not engage ({stepper.Session.LastMissReason}).");

        var replacement = new Tensor<float>(layer.CurrentWeights.ToArray(), layer.CurrentWeights.Shape.ToArray());
        var oldWeightAtReplacement = originalWeight.ToArray();
        layer.ReplaceWeights(replacement);
        var atReplacement = replacement.ToArray();

        for (int step = 0; step < 2 * interval + 2; step++)
            stepper.Step(PlainRequest(layer, RandomTensor(data, Batch, Features), RandomTensor(data, Batch, Outputs), optimizer));

        Assert.Equal(oldWeightAtReplacement, originalWeight.ToArray());
        Assert.Same(replacement, layer.CurrentWeights);
        Assert.NotEqual(atReplacement, replacement.ToArray());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Same_layer_slot_replacement_preserves_Adam_moments_step_and_schedule(bool scheduled)
    {
        var layer = new SwappableDenseLayer(Outputs);
        var reference = new DenseLayer<float>(Outputs);
        var warm = new Tensor<float>(new[] { Batch, Features });
        layer.Forward(warm);
        reference.Forward(warm);
        reference.SetParameters(layer.GetParameters());
        AdamOptimizer<float, Tensor<float>, Tensor<float>> Optimizer() => new(null,
            new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                InitialLearningRate = 0.01,
                EnableGradientClipping = false,
                UseAdaptiveLearningRate = false,
                UseAdaptiveBetas = false,
                LearningRateScheduler = scheduled
                    ? new AiDotNet.LearningRateSchedulers.StepLRScheduler(0.01, 2, 0.7) : null,
                SchedulerStepMode = AiDotNet.LearningRateSchedulers.SchedulerStepMode.StepPerBatch,
            });
        var optimizer = Optimizer();
        var referenceOptimizer = Optimizer();
        var stepper = new TapeTrainingStepper<float>(new object());
        var parameters = layer.GetTrainableParameters().ToArray();
        var expected = parameters.Select(p => p.ToArray().Select(x => (double)x).ToArray()).ToArray();
        var firstMoment = expected.Select(p => new double[p.Length]).ToArray();
        var secondMoment = expected.Select(p => new double[p.Length]).ToArray();
        var data = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(107);
        Tensor<float>? oldWeight = null;
        float[]? oldWeightSnapshot = null;
        Tensor<float>? replacement = null;
        for (int step = 0; step < 7; step++)
        {
            if (step == 3)
            {
                oldWeight = layer.CurrentWeights;
                oldWeightSnapshot = oldWeight.ToArray();
                replacement = new Tensor<float>(oldWeightSnapshot.ToArray(), oldWeight.Shape.ToArray());
                layer.ReplaceWeights(replacement);
                Assert.Same(replacement, layer.CurrentWeights);
                Assert.NotSame(oldWeight, replacement);
            }
            parameters = layer.GetTrainableParameters().ToArray();
            var input = RandomTensor(data, Batch, Features);
            var target = RandomTensor(data, Batch, Outputs);
            float[][]? gradients = null;
            int previousOptimizerStep = -1;
            double rate = optimizer.GetCurrentLearningRate();
            float loss = stepper.Step(new FusedTrainingStepRequest<float>
            {
                Layers = new ITrainableLayer<float>[] { layer },
                Input = input, Target = target, Forward = layer.Forward,
                ComputeLoss = MeanSquaredError, Optimizer = optimizer,
                OnGradients = values =>
                {
                    gradients = parameters.Select(p => values[p].ToArray()).ToArray();
                    if (CompiledTapeTrainingStep<float>.TryGetPlanOptimizerStep(out int count))
                        previousOptimizerStep = count;
                },
            });
            Assert.True(stepper.LastStepFused, $"step {step}: {stepper.Session.LastMissReason}");
            Assert.False(stepper.LastStepHybrid);
            if (step > 0) Assert.Equal(step, previousOptimizerStep);
            float eagerLoss = TapeTrainingStepper<float>.EagerStep(PlainRequest(reference, input, target, referenceOptimizer));
            Assert.True(Math.Abs(loss - eagerLoss) < 1e-4f);
            Assert.Equal(referenceOptimizer.GetCurrentLearningRate(), optimizer.GetCurrentLearningRate());
            Assert.NotNull(gradients);
            for (int slot = 0; slot < parameters.Length; slot++)
            {
                var actual = parameters[slot].ToArray();
                // Independent Adam recurrence: the accumulated moments are deliberately not reset at the swap.
                double beta1 = (float)0.9, beta2 = (float)0.999, epsilon = (float)1e-8;
                for (int index = 0; index < actual.Length; index++)
                {
                    double gradient = gradients![slot][index];
                    firstMoment[slot][index] = beta1 * firstMoment[slot][index] + (1 - beta1) * gradient;
                    secondMoment[slot][index] = beta2 * secondMoment[slot][index] + (1 - beta2) * gradient * gradient;
                    double correctedMean = firstMoment[slot][index] / (1 - Math.Pow(beta1, step + 1));
                    double correctedVariance = secondMoment[slot][index] / (1 - Math.Pow(beta2, step + 1));
                    expected[slot][index] -= rate * correctedMean / (Math.Sqrt(correctedVariance) + epsilon);
                    Assert.True(Math.Abs(expected[slot][index] - actual[index]) < 1e-4,
                        $"step {step}, slot {slot}[{index}]: expected {expected[slot][index]}, got {actual[index]}");
                }
            }
            var eagerParameters = reference.GetParameters().ToArray();
            var fusedParameters = layer.GetParameters().ToArray();
            for (int index = 0; index < eagerParameters.Length; index++)
                Assert.True(Math.Abs(eagerParameters[index] - fusedParameters[index]) < 1e-4f);
            if (oldWeight is not null) Assert.Equal(oldWeightSnapshot, oldWeight.ToArray());
        }
        Assert.Same(replacement, layer.CurrentWeights);
        Assert.NotEqual(oldWeightSnapshot, replacement!.ToArray());
        Assert.Equal(7, stepper.FusedStepCount);
        Assert.Equal(0, stepper.HybridStepCount);
    }

    [Fact]
    public void Weight_replacement_does_not_authorize_optimizer_configuration_drift()
    {
        var layer = new SwappableDenseLayer(Outputs);
        layer.Forward(new Tensor<float>(new[] { Batch, Features }));
        var options = new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
        {
            InitialLearningRate = 0.01, EnableGradientClipping = false,
            UseAdaptiveLearningRate = false, UseAdaptiveBetas = false,
        };
        var optimizer = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null, options);
        var stepper = new TapeTrainingStepper<float>(new object());
        var data = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(109);
        for (int step = 0; step < 3; step++)
            stepper.Step(PlainRequest(layer, RandomTensor(data, Batch, Features), RandomTensor(data, Batch, Outputs), optimizer));
        var oldWeight = layer.CurrentWeights;
        var replacement = new Tensor<float>(oldWeight.ToArray(), oldWeight.Shape.ToArray());
        layer.ReplaceWeights(replacement);
        options.Beta1 = 0.8;
        var values = layer.GetParameters().ToArray();
        var oldValues = oldWeight.ToArray();
        var state = optimizer.Serialize();
        Assert.Throws<InvalidOperationException>(() => stepper.Step(PlainRequest(layer,
            RandomTensor(data, Batch, Features), RandomTensor(data, Batch, Outputs), optimizer)));
        Assert.Same(replacement, layer.CurrentWeights);
        Assert.Equal(values, layer.GetParameters().ToArray());
        Assert.Equal(oldValues, oldWeight.ToArray());
        Assert.Equal(state, optimizer.Serialize());
        Assert.Equal(3, stepper.FusedStepCount);
        Assert.Equal(0, stepper.HybridStepCount);
    }

    // Exposes DenseLayer's protected weight replacement: SetWeights binds a NEW tensor object, which is how a layer's
    // live parameters can stop being the tensors a compiled plan trains.
    private sealed class SwappableDenseLayer : DenseLayer<float>
    {
        public SwappableDenseLayer(int outputSize) : base(outputSize) { }
        public Tensor<float> CurrentWeights => GetWeights();
        public void ReplaceWeights(Tensor<float> weights) => SetWeights(weights);
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