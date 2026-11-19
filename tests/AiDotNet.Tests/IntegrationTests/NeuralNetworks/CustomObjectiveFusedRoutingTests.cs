using System;
using System.Linq;
using System.Reflection;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.IntegrationTests.NeuralNetworks;

/// <summary>
/// TrainWithCustomObjective and TrainWithCustomLoss on the fused session. A model's own objective (a transducer loss,
/// a masked reconstruction) used to run on the eager tape always: 258 of the 941 model families trained that way.
/// These objectives build their per-step data inside themselves (a captured batch, a drawn count, a scale), which a
/// plan traced once would replay forever. So every replay is guarded by the objective's signature
/// (ClosureObjectiveGuard): the plan compiled for that exact signature steps, refilled with this step's data; a new
/// signature gets its own plan, which continues the optimizer; an objective no plan can replay trains eagerly, with
/// the traced step redone so the run matches one that was eager from the start.
/// </summary>
[Collection("FusedTrainingSerial")]
public class CustomObjectiveFusedRoutingTests
{
    private const int Steps = 6;

    public enum Objective { Arguments, CapturedTarget, CapturedScale, HostRead, HostReadOnStep3, StepNoise, StepIndices, CapturedTargetHostRead }

    private sealed class ObjectiveNetwork : FeedForwardNeuralNetwork<float>
    {
        internal ObjectiveNetwork(AdamOptimizer<float, Tensor<float>, Tensor<float>> optimizer) : base(
            new NeuralNetworkArchitecture<float>(inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
                complexity: NetworkComplexity.Simple, inputSize: 6, outputSize: 2),
            optimizer, new MeanSquaredErrorLoss<float>())
        {
        }

        // Everything per step reaches the objective through its arguments.
        internal float ArgumentObjectiveStep(Tensor<float> x, Tensor<float> y)
            => TrainWithCustomObjective(x, y, (input, target) =>
                new MeanSquaredErrorLoss<float>().ComputeTapeLoss(ForwardForTraining(input), target));

        // A second objective on the same optimizer: same shapes, a different loss.
        internal float SecondObjectiveStep(Tensor<float> x, Tensor<float> y)
            => TrainWithCustomObjective(x, y, (input, target) =>
                new MeanAbsoluteErrorLoss<float>().ComputeTapeLoss(ForwardForTraining(input), target));

        // The target is captured by the closure, not passed: a tensor a plan would bake at its trace.
        internal float CapturedTargetStep(Tensor<float> x, Tensor<float> capturedTarget)
            => TrainWithCustomLoss(x, prediction => new MeanSquaredErrorLoss<float>().ComputeTapeLoss(prediction, capturedTarget));

        // A host scalar drawn per step and captured (as SoundStream's quantizer count is): two signatures.
        internal float CapturedScaleStep(Tensor<float> x, Tensor<float> y, float scale)
            => TrainWithCustomLoss(x, prediction => Engine.TensorMultiplyScalar(
                new MeanSquaredErrorLoss<float>().ComputeTapeLoss(prediction, y), scale));

        // Reads a value of its own forward on the host: no plan can replay it.
        internal float HostReadStep(Tensor<float> x, Tensor<float> y)
            => TrainWithCustomLoss(x, prediction =>
            {
                float first = prediction[0];
                var loss = new MeanSquaredErrorLoss<float>().ComputeTapeLoss(prediction, y);
                return Engine.TensorMultiplyScalar(loss, first > 0 ? 1.5f : 0.5f);
            });

        private readonly Random _noise = new(7);

        // Draws fresh noise every step through CompiledStepRandom (as a VAE's reparameterization does).
        internal float NoisyTargetStep(Tensor<float> x, Tensor<float> y)
            => TrainWithCustomLoss(x, prediction =>
            {
                var noise = AiDotNet.Training.CompiledStepRandom<float>.StandardNormal(y.Shape.ToArray(), _noise);
                var noisy = Engine.TensorAdd(y, Engine.TensorMultiplyScalar(noise, 0.5f));
                return new MeanSquaredErrorLoss<float>().ComputeTapeLoss(prediction, noisy);
            });

        // Reads a captured, already-computed data tensor on the host and builds a constant from it (as a vocoder's
        // teacher-forcing inputs are built from its target audio): the constant differs every step.
        internal float CapturedTargetHostReadStep(Tensor<float> x, Tensor<float> capturedTarget)
            => TrainWithCustomLoss(x, prediction =>
            {
                var shifted = new Tensor<float>(capturedTarget.Shape.ToArray());
                var source = capturedTarget.ToArray();
                var span = shifted.AsWritableSpan();
                for (int i = 0; i < span.Length; i++) span[i] = source[i] > 0 ? 1f : -1f;
                return new MeanSquaredErrorLoss<float>().ComputeTapeLoss(prediction, shifted);
            });

        private readonly Random _rows = new(11);

        // Draws per-step row indices, then noise whose scale depends on that draw (as wav2vec 2.0's distractors depend on its
        // mask): the target rows are resampled with replacement every step, and the replay must refill both in order.
        internal float ResampledTargetStep(Tensor<float> x, Tensor<float> y)
            => TrainWithCustomObjective(x, y, (input, target) =>
            {
                int rows = target.Shape[0];
                int lastFirst = 0;
                var picks = AiDotNet.Training.CompiledStepRandom<float>.DrawIndices(new[] { rows }, t =>
                {
                    var span = t.AsWritableSpan();
                    for (int i = 0; i < span.Length; i++) span[i] = _rows.Next(rows);
                    lastFirst = span[0];
                });
                var scale = AiDotNet.Training.CompiledStepRandom<float>.Draw(new[] { 1 }, t => t.AsWritableSpan()[0] = 0.25f * (1 + lastFirst));
                var resampled = Engine.TensorIndexSelect(target, picks, 0);
                var scaled = Engine.TensorMultiply(resampled, Engine.TensorBroadcastTo(Engine.Reshape(scale, new[] { 1, 1 }), resampled._shape));
                return new MeanSquaredErrorLoss<float>().ComputeTapeLoss(ForwardForTraining(input), scaled);
            });

        // Reads its own forward on the host only when asked (as Paraformer's sampler does on some steps).
        internal float SometimesHostReadStep(Tensor<float> x, Tensor<float> y, bool read)
            => TrainWithCustomLoss(x, prediction =>
            {
                var loss = new MeanSquaredErrorLoss<float>().ComputeTapeLoss(prediction, y);
                if (!read) return loss;
                float first = prediction[0];
                return Engine.TensorMultiplyScalar(loss, first > 0 ? 1.5f : 0.5f);
            });

        // The declared-objective route CLAP and TensorModelTrainer use: a one-time replay check against eager.
        internal float VerifiedCapturedTargetStep(Tensor<float> x, Tensor<float> capturedTarget)
            => (float)s_trainCustomTapeStep.Invoke(this, new object?[]
            {
                x, x, new Func<Tensor<float>, Tensor<float>>(ForwardForTraining),
                new Func<Tensor<float>, Tensor<float>, Tensor<float>>((prediction, _) =>
                    new MeanSquaredErrorLoss<float>().ComputeTapeLoss(prediction, capturedTarget)),
                null, null, true, true, null, null,
            })!;

        private static readonly MethodInfo s_trainCustomTapeStep = typeof(NeuralNetworkBase<float>).GetMethod(
            "TrainCustomTapeStep", BindingFlags.Instance | BindingFlags.NonPublic, null, new[]
            {
                typeof(Tensor<float>), typeof(Tensor<float>), typeof(Func<Tensor<float>, Tensor<float>>),
                typeof(Func<Tensor<float>, Tensor<float>, Tensor<float>>),
                typeof(IGradientBasedOptimizer<float, Tensor<float>, Tensor<float>>),
                typeof(Func<System.Collections.Generic.IReadOnlyList<Tensor<float>>>), typeof(bool), typeof(bool),
                typeof(MethodInfo), typeof(Func<float>),
            }, null) ?? throw new InvalidOperationException("TrainCustomTapeStep's private overload was not found.");
    }

    private static Tensor<float> Batch(int step, int rows, int cols, double scale)
    {
        var t = new Tensor<float>(new[] { rows, cols });
        for (int i = 0; i < t.Length; i++) t[i] = (float)(Math.Sin(0.7 * i + 1.3 * step) * scale);
        return t;
    }

    private static AdamOptimizer<float, Tensor<float>, Tensor<float>> NewOptimizer(bool warmup = false)
        => new(null, new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
        {
            InitialLearningRate = 0.01,
            UseAdaptiveLearningRate = false,
            LearningRateScheduler = warmup ? new AiDotNet.LearningRateSchedulers.LinearWarmupScheduler(0.01, 4, 20) : null,
            SchedulerStepMode = AiDotNet.LearningRateSchedulers.SchedulerStepMode.StepPerBatch,
        });

    private static (Vector<float> Parameters, ObjectiveNetwork Model) Train(bool compiled, Objective objective, Vector<float>? init,
        bool verified = false, double targetScale = 0.5, bool reuseBuffers = false, bool warmup = false)
    {
        var model = new ObjectiveNetwork(NewOptimizer(warmup));
        if (init is not null) model.UpdateParameters(init);
        var options = AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current;
        bool saved = options.EnableCompilation;
        try
        {
            options.EnableCompilation = compiled;
            Tensor<float>? sharedX = null, sharedY = null;
            for (int s = 0; s < Steps; s++)
            {
                var x = Batch(s, 4, 6, 1.0);
                var y = Batch(s + 100, 4, 2, targetScale);   // a different target every step
                if (reuseBuffers)
                {
                    // One input and one target tensor for the whole run, refilled in place, as a data loader does.
                    sharedX ??= x;
                    sharedY ??= y;
                    x.AsSpan().CopyTo(sharedX.AsWritableSpan());
                    y.AsSpan().CopyTo(sharedY.AsWritableSpan());
                    x = sharedX;
                    y = sharedY;
                }
                if (verified) model.VerifiedCapturedTargetStep(x, y);
                else switch (objective)
                {
                    case Objective.Arguments: model.ArgumentObjectiveStep(x, y); break;
                    case Objective.CapturedTarget: model.CapturedTargetStep(x, y); break;
                    case Objective.CapturedScale: model.CapturedScaleStep(x, y, s % 3 == 0 ? 2f : 0.5f); break;
                    case Objective.HostRead: model.HostReadStep(x, y); break;
                    case Objective.HostReadOnStep3: model.SometimesHostReadStep(x, y, read: s == 3); break;
                    case Objective.StepNoise: model.NoisyTargetStep(x, y); break;
                    case Objective.StepIndices: model.ResampledTargetStep(x, y); break;
                    case Objective.CapturedTargetHostRead: model.CapturedTargetHostReadStep(x, y); break;
                }
            }
            return (model.GetParameters(), model);
        }
        finally
        {
            options.EnableCompilation = saved;
        }
    }

    private static double MaxAbsDifference(Vector<float> a, Vector<float> b)
    {
        double max = 0;
        for (int i = 0; i < a.Length; i++) max = Math.Max(max, Math.Abs((double)a[i] - b[i]));
        return max;
    }

    private static (ObjectiveNetwork Fused, double Divergence) FusedVersusEager(Objective objective, bool verified = false,
        double targetScale = 0.5, bool reuseBuffers = false, bool warmup = false, double initScale = 1.0)
    {
        var init = Train(compiled: false, objective, init: null, verified, targetScale, reuseBuffers, warmup).Model.GetParameters();
        for (int i = 0; i < init.Length; i++) init[i] *= (float)initScale;
        var eager = Train(compiled: false, objective, init, verified, targetScale, reuseBuffers, warmup);
        var fused = Train(compiled: true, objective, init, verified, targetScale, reuseBuffers, warmup);
        Assert.Equal(0, eager.Model.CustomFusedStepCount);   // compilation off: every step eager
        Assert.True(MaxAbsDifference(eager.Parameters, init) > 1e-3 * initScale, "training did not move the parameters.");
        return (fused.Model, MaxAbsDifference(fused.Parameters, eager.Parameters));
    }

    [Theory(Timeout = 120000)]
    [InlineData(Objective.Arguments)]
    [InlineData(Objective.CapturedTarget)]
    [InlineData(Objective.CapturedScale)]
    [InlineData(Objective.StepNoise)]
    [InlineData(Objective.StepIndices)]
    public async System.Threading.Tasks.Task AGuardedObjective_TrainsFused_AndMatchesEager(Objective objective)
    {
        await System.Threading.Tasks.Task.CompletedTask;
        var (fused, divergence) = FusedVersusEager(objective);
        Assert.True(fused.ObjectiveSessionCommitted, $"{objective}: the custom objective did not commit a fused plan: "
            + (fused.LastObjectiveMissReason ?? "no reason recorded"));
        Assert.Equal(Steps, fused.CustomFusedStepCount);
        Assert.True(divergence < 1e-4, $"{objective}: fused diverged from eager by {divergence:E3}");
    }

    // A constant built on the host from a captured data tensor changes with the data: whether the step fuses or not,
    // the parameters must follow the eager tape.
    [Fact(Timeout = 120000)]
    public async System.Threading.Tasks.Task AConstantBuiltFromCapturedData_FollowsTheData_FusedOrNot()
    {
        await System.Threading.Tasks.Task.CompletedTask;
        var (fused, divergence) = FusedVersusEager(Objective.CapturedTargetHostRead);
        Assert.True(divergence < 1e-4, $"a constant built from captured data diverged from eager by {divergence:E3} "
            + "(fused steps " + fused.CustomFusedStepCount + ", " + (fused.LastObjectiveMissReason ?? "no reason") + ")");
    }

    [Fact(Timeout = 120000)]
    public async System.Threading.Tasks.Task AnObjectiveThatReadsItsOwnValues_TrainsEagerly_AndMatchesEager()
    {
        await System.Threading.Tasks.Task.CompletedTask;
        var (fused, divergence) = FusedVersusEager(Objective.HostRead);
        Assert.False(fused.ObjectiveSessionCommitted, "a host-reading objective was replayed from a compiled plan.");
        Assert.True(divergence < 1e-4, $"a host-reading objective diverged from eager by {divergence:E3} "
            + "(" + (fused.LastObjectiveMissReason ?? "no reason") + ")");
    }

    [Fact(Timeout = 120000)]
    public async System.Threading.Tasks.Task AStepNoPlanCanReplay_AfterFusedSteps_RunsThroughThePlansOptimizer()
    {
        await System.Threading.Tasks.Task.CompletedTask;
        // Steps 0-2 commit and replay the plan; step 3 reads its forward on the host, which no plan can replay. The
        // plan's moments cannot move to the eager optimizer, so that step used to throw. The eager tape now takes its
        // gradients and the plan's own optimizer applies them, and steps 4-5 replay the plan again: the run matches
        // pure eager.
        var (fused, divergence) = FusedVersusEager(Objective.HostReadOnStep3);
        Assert.True(fused.ObjectiveSessionCommitted, "the objective did not stay on its fused plan: "
            + (fused.LastObjectiveMissReason ?? "no reason recorded"));
        Assert.Equal(1, fused.CustomOutsidePlanStepCount);
        Assert.Equal(Steps - 1, fused.CustomFusedStepCount);
        Assert.Equal(0, fused.CustomEagerStepCount);
        Assert.True(divergence < 1e-4, $"the run with a step outside the plan diverged from eager by {divergence:E3}");
    }

    [Fact(Timeout = 120000)]
    public async System.Threading.Tasks.Task AReplayCheckThatFails_RedoesTheTracedStep_AndMatchesEager()
    {
        await System.Threading.Tasks.Task.CompletedTask;
        var (fused, divergence) = FusedVersusEager(Objective.CapturedTarget, verified: true);

        // The plan baked the first step's target; its first replay on a new target disagreed with the eager forward,
        // so both fused updates were undone and the traced step redone eagerly, with the optimizer back at its
        // pre-trace position: without that redo this run ended 6e-2 from pure eager (Adam's first-step moments lived
        // in the dropped plan).
        Assert.False(fused.ObjectiveSessionCommitted, "a closure-captured target was replayed from the trace.");
        string reason = fused.LastObjectiveMissReason ?? "";
        Assert.Contains("disagrees with the eager forward", reason);
        Assert.Contains("traced step was redone on the eager tape", reason);
        Assert.True(divergence < 1e-4, $"a baked-target run diverged from eager by {divergence:E3}");
    }

    [Fact(Timeout = 120000)]
    public async System.Threading.Tasks.Task AReplayCheckThatFails_IsCaughtWhenTheLossIsSmall()
    {
        await System.Threading.Tasks.Task.CompletedTask;
        // Targets of magnitude 1e-2 and a network that starts near zero output give losses around 1e-4. The replay
        // check compared losses with an absolute floor of 1e-3, so every such disagreement passed, the plan that had
        // frozen the first step's target was committed, and the run trained on that one target throughout.
        var (fused, divergence) = FusedVersusEager(Objective.CapturedTarget, verified: true, targetScale: 0.01, initScale: 0.01);
        Assert.False(fused.ObjectiveSessionCommitted, "a closure-captured target was replayed from the trace.");
        Assert.Contains("disagrees with the eager forward", fused.LastObjectiveMissReason ?? "");
        Assert.True(divergence < 1e-4, $"a baked-target run with small losses diverged from eager by {divergence:E3}");
    }

    [Fact(Timeout = 120000)]
    public async System.Threading.Tasks.Task ACapturedTarget_TrainsFused_WhenTheLossIsSmall()
    {
        await System.Threading.Tasks.Task.CompletedTask;
        // The guard compares the objective's signature and refills its data, so the loss magnitude does not enter.
        var (fused, divergence) = FusedVersusEager(Objective.CapturedTarget, targetScale: 0.01, initScale: 0.01);
        Assert.True(fused.ObjectiveSessionCommitted, "the custom objective did not commit a fused plan: "
            + (fused.LastObjectiveMissReason ?? "no reason recorded"));
        Assert.True(divergence < 1e-4, $"a captured-target run with small losses diverged from eager by {divergence:E3}");
    }

    [Theory(Timeout = 120000)]
    [InlineData(false)]
    [InlineData(true)]
    public async System.Threading.Tasks.Task ACapturedTarget_InBuffersRefilledInPlace_StillMatchesEager(bool verified)
    {
        await System.Threading.Tasks.Task.CompletedTask;
        // The caller keeps one input and one target tensor and refills them every step: whether the plan reads the
        // captured tensor live or the run goes back to the eager tape, every step must train on that step's data.
        var (_, divergence) = FusedVersusEager(Objective.CapturedTarget, verified, reuseBuffers: true);
        Assert.True(divergence < 1e-4, $"a captured-target run on refilled buffers diverged from eager by {divergence:E3}");
    }

    [Fact(Timeout = 120000)]
    public async System.Threading.Tasks.Task AnObjectiveOnAWarmupSchedule_TrainsFused_AndMatchesEager()
    {
        await System.Threading.Tasks.Task.CompletedTask;
        // The rate changes on every one of these steps (a four-step linear warm-up, then decay): the fused custom
        // objective has to follow the schedule step for step, as the eager one does.
        var (fused, divergence) = FusedVersusEager(Objective.Arguments, warmup: true);
        Assert.True(fused.ObjectiveSessionCommitted, "the custom objective did not commit a fused plan: "
            + (fused.LastObjectiveMissReason ?? "no reason recorded"));
        Assert.True(divergence < 1e-4, $"a fused custom objective on a warm-up schedule diverged from eager by {divergence:E3}");
    }

    private static (Vector<float> Parameters, ObjectiveNetwork Model) TrainTwoObjectives(bool compiled, Vector<float>? init,
        bool switchObjective = true)
    {
        var model = new ObjectiveNetwork(NewOptimizer());
        if (init is not null) model.UpdateParameters(init);
        var options = AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current;
        bool saved = options.EnableCompilation;
        try
        {
            options.EnableCompilation = compiled;
            for (int s = 0; s < Steps; s++)
            {
                var x = Batch(s, 4, 6, 1.0);
                var y = Batch(s + 100, 4, 2, 0.5);
                // The first objective long enough to trace and replay; then the second.
                if (s < 3 || !switchObjective) model.ArgumentObjectiveStep(x, y);
                else model.SecondObjectiveStep(x, y);
            }
            return (model.GetParameters(), model);
        }
        finally
        {
            options.EnableCompilation = saved;
        }
    }

    [Fact(Timeout = 120000)]
    public async System.Threading.Tasks.Task ASecondObjectiveOnTheSameOptimizer_GetsItsOwnPlan_AndMatchesEager()
    {
        await System.Threading.Tasks.Task.CompletedTask;
        // Both objectives share the optimizer, so they share its session. The plan traced for the first objective (mean
        // squared error) used to be replayed for the second (mean absolute error): the last three steps trained the
        // first objective again. The second objective signs differently, so it gets its own plan, which continues the
        // optimizer from the first.
        var init = TrainTwoObjectives(compiled: false, init: null).Model.GetParameters();
        var eager = TrainTwoObjectives(compiled: false, init);
        var switched = TrainTwoObjectives(compiled: true, init);
        var firstOnly = TrainTwoObjectives(compiled: true, init, switchObjective: false);

        Assert.True(switched.Model.ObjectiveSessionCommitted, "the two objectives did not train fused: "
            + (switched.Model.LastObjectiveMissReason ?? "no reason recorded"));
        Assert.Equal(Steps, switched.Model.CustomFusedStepCount);
        double divergence = MaxAbsDifference(switched.Parameters, eager.Parameters);
        Assert.True(divergence < 1e-4, $"two objectives on one optimizer diverged from eager by {divergence:E3}");
        double difference = MaxAbsDifference(switched.Parameters, firstOnly.Parameters);
        Assert.True(difference > 1e-3, $"the second objective's steps trained the first objective (difference {difference:E3}).");
    }
}
