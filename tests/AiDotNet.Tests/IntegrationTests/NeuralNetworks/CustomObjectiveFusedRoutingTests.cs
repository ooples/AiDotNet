using System;
using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.IntegrationTests.NeuralNetworks;

/// <summary>
/// TrainWithCustomObjective and TrainWithCustomLoss on the fused session. A model's own objective (a transducer
/// loss, a masked reconstruction) used to run on the eager tape always: 258 of the 941 model families trained that
/// way. These objectives were written for the eager tape, and many read per-step host data a compiled plan would
/// freeze at its trace. So the fused attempt checks its first replay on new data against the eager forward; on
/// disagreement it undoes that update and stays eager.
/// </summary>
[Collection("FusedTrainingSerial")]
public class CustomObjectiveFusedRoutingTests
{
    private const int Steps = 6;

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

        // The target is captured by the closure, not passed: a host value a plan would bake at its trace.
        internal float CapturedTargetStep(Tensor<float> x, Tensor<float> capturedTarget)
            => TrainWithCustomLoss(x, prediction => new MeanSquaredErrorLoss<float>().ComputeTapeLoss(prediction, capturedTarget));
    }

    private static Tensor<float> Batch(int step, int rows, int cols, double scale)
    {
        var t = new Tensor<float>(new[] { rows, cols });
        for (int i = 0; i < t.Length; i++) t[i] = (float)(Math.Sin(0.7 * i + 1.3 * step) * scale);
        return t;
    }

    private static (Vector<float> Parameters, ObjectiveNetwork Model) Train(bool compiled, bool captured, Vector<float>? init)
        => Train(compiled, captured, init, targetScale: 0.5);

    private static (Vector<float> Parameters, ObjectiveNetwork Model) Train(bool compiled, bool captured, Vector<float>? init,
        double targetScale, bool reuseBuffers = false, bool warmup = false)
    {
        var optimizer = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null,
            new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                InitialLearningRate = 0.01,
                UseAdaptiveLearningRate = false,
                LearningRateScheduler = warmup ? new AiDotNet.LearningRateSchedulers.LinearWarmupScheduler(0.01, 4, 20) : null,
                SchedulerStepMode = AiDotNet.LearningRateSchedulers.SchedulerStepMode.StepPerBatch,
            });
        var model = new ObjectiveNetwork(optimizer);
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
                if (captured) model.CapturedTargetStep(x, y);
                else model.ArgumentObjectiveStep(x, y);
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

    [Fact(Timeout = 120000)]
    public async System.Threading.Tasks.Task AnObjectiveThatReadsItsArguments_TrainsFused_AndMatchesEager()
    {
        await System.Threading.Tasks.Task.CompletedTask;
        var init = Train(compiled: false, captured: false, init: null).Model.GetParameters();
        var eager = Train(compiled: false, captured: false, init);
        var fused = Train(compiled: true, captured: false, init);

        Assert.True(fused.Model.ObjectiveSessionCommitted, "the custom objective did not commit a fused plan: "
            + (fused.Model.LastObjectiveMissReason ?? "no reason recorded"));
        Assert.True(fused.Model.CustomFusedStepCount >= Steps - 1, $"fused steps {fused.Model.CustomFusedStepCount} of {Steps}");
        Assert.Equal(0, eager.Model.CustomFusedStepCount);   // compilation off: every step eager
        Assert.True(MaxAbsDifference(eager.Parameters, init) > 1e-3, "training did not move the parameters.");
        double divergence = MaxAbsDifference(fused.Parameters, eager.Parameters);
        Assert.True(divergence < 1e-4, $"fused custom objective diverged from eager by {divergence:E3}");
    }

    [Fact(Timeout = 120000)]
    public async System.Threading.Tasks.Task AnObjectiveThatCapturesHostData_IsCaughtByTheReplayCheck_AndMatchesEager()
    {
        await System.Threading.Tasks.Task.CompletedTask;
        var init = Train(compiled: false, captured: true, init: null).Model.GetParameters();
        var eager = Train(compiled: false, captured: true, init);
        var fused = Train(compiled: true, captured: true, init);

        // The plan baked the first step's target; its first replay on a new target disagreed with the eager forward,
        // so that update was undone and the objective stayed eager.
        Assert.False(fused.Model.ObjectiveSessionCommitted, "a closure-captured target was replayed from the trace.");
        Assert.Contains("disagrees with the eager forward", fused.Model.LastObjectiveMissReason ?? "");
        // Both fused updates (the trace and its replay) were undone and the traced step redone eagerly, with the
        // optimizer back at its pre-trace position: without that redo this run ended 6e-2 from pure eager (Adam's
        // first-step moments lived in the dropped plan).
        Assert.Contains("traced step was redone on the eager tape", fused.Model.LastObjectiveMissReason ?? "");
        double divergence = MaxAbsDifference(fused.Parameters, eager.Parameters);
        Assert.True(divergence < 1e-4, $"a baked-target run diverged from eager by {divergence:E3}");
    }

    [Fact(Timeout = 120000)]
    public async System.Threading.Tasks.Task ACapturedTarget_IsCaughtWhenTheLossIsSmall()
    {
        await System.Threading.Tasks.Task.CompletedTask;
        // Targets of magnitude 1e-2 and a network that starts near zero output give losses around 1e-4. The replay
        // check compared losses with an absolute floor of 1e-3, so every such disagreement passed, the plan that had
        // frozen the first step's target was committed, and the run trained on that one target throughout.
        var init = Train(compiled: false, captured: true, init: null, targetScale: 0.01).Model.GetParameters();
        for (int i = 0; i < init.Length; i++) init[i] *= 0.01f;
        var eager = Train(compiled: false, captured: true, init, targetScale: 0.01);
        var fused = Train(compiled: true, captured: true, init, targetScale: 0.01);

        Assert.False(fused.Model.ObjectiveSessionCommitted, "a closure-captured target was replayed from the trace.");
        Assert.Contains("disagrees with the eager forward", fused.Model.LastObjectiveMissReason ?? "");
        double divergence = MaxAbsDifference(fused.Parameters, eager.Parameters);
        Assert.True(divergence < 1e-4, $"a baked-target run with small losses diverged from eager by {divergence:E3}");
    }

    [Fact(Timeout = 120000)]
    public async System.Threading.Tasks.Task ACapturedTarget_InBuffersRefilledInPlace_StillMatchesEager()
    {
        await System.Threading.Tasks.Task.CompletedTask;
        // The caller keeps one input and one target tensor and refills them every step. Whether the plan reads the
        // captured tensor live or the replay check sends the run back to the eager tape, every step must train on
        // that step's data.
        var init = Train(compiled: false, captured: true, init: null, targetScale: 0.5, reuseBuffers: true).Model.GetParameters();
        var eager = Train(compiled: false, captured: true, init, targetScale: 0.5, reuseBuffers: true);
        var fused = Train(compiled: true, captured: true, init, targetScale: 0.5, reuseBuffers: true);

        double divergence = MaxAbsDifference(fused.Parameters, eager.Parameters);
        Assert.True(divergence < 1e-4, $"a captured-target run on refilled buffers diverged from eager by {divergence:E3}");
    }

    [Fact(Timeout = 120000)]
    public async System.Threading.Tasks.Task AnObjectiveOnAWarmupSchedule_TrainsFused_AndMatchesEager()
    {
        await System.Threading.Tasks.Task.CompletedTask;
        // The rate changes on every one of these steps (a four-step linear warm-up, then decay): the fused custom
        // objective has to follow the schedule step for step, as the eager one does.
        var init = Train(compiled: false, captured: false, init: null, targetScale: 0.5, warmup: true).Model.GetParameters();
        var eager = Train(compiled: false, captured: false, init, targetScale: 0.5, warmup: true);
        var fused = Train(compiled: true, captured: false, init, targetScale: 0.5, warmup: true);

        Assert.True(fused.Model.ObjectiveSessionCommitted, "the custom objective did not commit a fused plan: "
            + (fused.Model.LastObjectiveMissReason ?? "no reason recorded"));
        double divergence = MaxAbsDifference(fused.Parameters, eager.Parameters);
        Assert.True(divergence < 1e-4, $"a fused custom objective on a warm-up schedule diverged from eager by {divergence:E3}");
    }

    private static (Vector<float> Parameters, ObjectiveNetwork Model) TrainTwoObjectives(bool compiled, Vector<float>? init,
        bool switchObjective = true)
    {
        var optimizer = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null,
            new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 0.01, UseAdaptiveLearningRate = false });
        var model = new ObjectiveNetwork(optimizer);
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
                // The first objective long enough to trace, replay and pass its replay check; then the second.
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
    public async System.Threading.Tasks.Task ASecondObjectiveOnTheSameOptimizer_IsNotReplayedFromTheFirstObjectivesPlan()
    {
        await System.Threading.Tasks.Task.CompletedTask;
        // Both objectives share the optimizer, so they share its session and its plan. The plan traced for the first
        // objective (mean squared error) was replayed for the second (mean absolute error): the last three steps
        // trained the first objective again, and the run ended exactly where six steps of the first objective end.
        var init = TrainTwoObjectives(compiled: false, init: null).Model.GetParameters();
        var switched = TrainTwoObjectives(compiled: true, init);
        var firstOnly = TrainTwoObjectives(compiled: true, init, switchObjective: false);

        Assert.True(firstOnly.Model.ObjectiveSessionCommitted, "the first objective alone did not commit a fused plan.");
        // The second objective's first step re-runs the replay check, which the first objective's plan fails; the
        // plan is dropped and the second objective trains on the eager tape (with fresh optimizer moments, as any
        // dropped committed plan does, so this run is not compared to an all-eager one).
        Assert.False(switched.Model.ObjectiveSessionCommitted, "the first objective's plan was kept for the second objective.");
        Assert.Contains("disagrees with the eager forward", switched.Model.LastObjectiveMissReason ?? "");
        double difference = MaxAbsDifference(switched.Parameters, firstOnly.Parameters);
        Assert.True(difference > 1e-3, $"the second objective's steps trained the first objective (difference {difference:E3}).");
    }
}
