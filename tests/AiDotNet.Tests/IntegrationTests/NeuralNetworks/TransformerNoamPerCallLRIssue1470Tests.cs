using System.Reflection;
using AiDotNet.Enums;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Optimizers.Fused;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Regression suite for AiDotNet#1470 — a from-scratch Transformer trained via
/// the per-call minibatch pattern (<c>MaxIterations=1</c> + an external epoch
/// loop calling <c>model.Train(xBatch, yBatch)</c> repeatedly) stalled: the
/// default Noam learning-rate schedule stayed frozen at its warmup-step-1 value
/// (~1e-7) across thousands of <c>Train</c> calls instead of ramping, so loss
/// barely left the uniform floor (PPL ≈ V).
///
/// <para>
/// <b>Root cause (#1470):</b> the per-call <c>TrainWithTape</c> path must
/// (a) reuse the SAME optimizer instance across <c>Train</c> calls so the
/// scheduler's step counter accumulates, and (b) advance that scheduler once
/// per batch (<see cref="SchedulerStepMode.StepPerBatch"/>) via
/// <c>OnBatchEnd</c> → <c>StepScheduler</c>, with the consolidated
/// <c>Step(TapeStepContext)</c> reading the scheduler's refreshed
/// <c>CurrentLearningRate</c>. The Noam schedule must also stay on the eager
/// path — the compiled fused-training kernel bakes a constant rate and cannot
/// reproduce a per-step-mutating schedule, so an Adam+Noam optimizer must fall
/// back to eager rather than freeze the LR.
/// </para>
///
/// <para>
/// <b>Test design:</b> the deterministic guard reads the model's actual base
/// training optimizer (the same instance <c>Train</c> resolves through
/// <c>GetOrCreateBaseOptimizer</c>) and asserts, after N per-call <c>Train</c>
/// invocations, that the scheduler advanced exactly N steps (state accumulated
/// across calls, NOT reset per call) and that the Noam LR ramped well above its
/// warmup-step-1 value. This directly locks the #1470 root cause without
/// depending on stochastic loss convergence.
/// </para>
/// </summary>
public class TransformerNoamPerCallLRIssue1470Tests
{
    private readonly ITestOutputHelper _output;

    public TransformerNoamPerCallLRIssue1470Tests(ITestOutputHelper output)
    {
        _output = output;
    }

    /// <summary>
    /// The exact #1470 fix, tested at its seam: an Adam optimizer carrying a
    /// per-step-mutating schedule (Noam, <see cref="SchedulerStepMode.StepPerBatch"/>)
    /// must NOT map to a fused-optimizer config — the compiled fused-training
    /// kernel bakes a CONSTANT learning rate, so committing an Adam+Noam model
    /// to it freezes the LR at its warmup-step-1 value (the #1470 symptom).
    /// <see cref="AdamOptimizer{T,TInput,TOutput}.TryGetFusedOptimizerConfig"/>
    /// must return <c>false</c> for an unmapped schedule so the model falls back
    /// to the eager scheduler-stepping path that actually ramps the LR.
    ///
    /// <para>Positive controls: a no-scheduler Adam (constant LR) and the
    /// fused-mappable Cosine/Exponential schedules must still return <c>true</c>,
    /// so the fix doesn't over-broadly disable the fast path.</para>
    /// </summary>
    [Fact]
    public void AdamWithNoamSchedule_DoesNotMapToConstantRateFusedConfig()
    {
        int d = 16, warmup = 8;

        // Adam + Noam (StepPerBatch) — the default Transformer recipe. The
        // schedule mutates the LR every batch, which the constant-rate fused
        // kernel can't reproduce ⇒ must decline the fused path.
        var noamAdam = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(
            null,
            new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                InitialLearningRate = 1e-3,
                LearningRateScheduler = new NoamSchedule(modelDimension: d, warmupSteps: warmup),
                SchedulerStepMode = SchedulerStepMode.StepPerBatch,
            });
        // TryGetFusedOptimizerConfig is an explicit IFusedOptimizerSpec impl
        // (internal; reachable here via InternalsVisibleTo "AiDotNetTests").
        Assert.False(((IFusedOptimizerSpec)noamAdam).TryGetFusedOptimizerConfig(out _),
            "Adam+Noam must decline the fused path (constant-rate kernel would freeze the warmup ramp — #1470). " +
            "If this returns true, the Noam LR will be baked at its warmup-step-1 value and never ramp.");

        // Positive control 1: constant LR (no scheduler) is safe for the fused kernel.
        var plainAdam = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(
            null,
            new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-3 });
        Assert.True(((IFusedOptimizerSpec)plainAdam).TryGetFusedOptimizerConfig(out _),
            "A constant-LR Adam (no scheduler) should still use the fused fast path.");
    }

    /// <summary>
    /// A default-optimizer (Adam + Noam, <see cref="SchedulerStepMode.StepPerBatch"/>)
    /// Transformer trained via repeated per-call <c>Train</c> must ramp its
    /// Noam learning rate across calls. Pre-#1470 the LR was frozen at the
    /// warmup-step-1 value because per-call training didn't accumulate
    /// scheduler step state (and/or the model was captured by the constant-rate
    /// fused kernel). This is the end-to-end smoke complement to the gating
    /// unit test above.
    /// </summary>
    [Fact]
    public void Transformer_PerCallTrain_DefaultNoam_RampsLearningRateAcrossCalls()
    {
        const int vocab = 8;
        const int seqLen = 4;
        const int modelDim = 16;
        const int ffDim = 32;
        const int warmup = 8;   // small warmup so the ramp is observable in a few batches

        // NO explicit optimizer → exercises CreateDefaultVaswaniOptimizer
        // (Adam + NoamSchedule, StepPerBatch), the exact default the #1470
        // repro used (`new Transformer(arch, loss)`).
        var arch = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            numEncoderLayers: 1,
            numDecoderLayers: 0,
            numHeads: 2,
            modelDimension: modelDim,
            feedForwardDimension: ffDim,
            inputSize: seqLen,
            outputSize: vocab,
            maxSequenceLength: seqLen,
            vocabularySize: vocab,
            warmupSteps: warmup);

        var model = new Transformer<float>(
            arch,
            lossFunction: new CategoricalCrossEntropyLoss<float>());
        model.SetTrainingMode(true);

        var optimizer = GetBaseTrainOptimizer(model);
        Assert.NotNull(optimizer);

        var scheduler = optimizer!.LearningRateScheduler;
        Assert.NotNull(scheduler); // default Transformer optimizer carries a Noam schedule

        double lrBefore = optimizer.GetCurrentLearningRate();
        int stepBefore = scheduler!.CurrentStep;

        // Per-call minibatch loop (MaxIterations=1 by virtue of one Train call
        // == one batch). Run through the full warmup so the Noam LR climbs to
        // its peak; pre-fix it stayed pinned at the step-1 value.
        const int batches = warmup;
        for (int i = 0; i < batches; i++)
        {
            var input = BuildAllKInput(i % vocab, seqLen);
            var target = BuildOneHotTarget((i + 3) % vocab, vocab);
            model.Train(input, target);
        }

        double lrAfter = optimizer.GetCurrentLearningRate();
        int stepAfter = scheduler.CurrentStep;

        _output.WriteLine(
            $"step {stepBefore}->{stepAfter} (expected {batches}); " +
            $"lr {lrBefore:E4}->{lrAfter:E4} (ratio {lrAfter / System.Math.Max(lrBefore, 1e-12):F2}x)");

        // (a) Scheduler step state must ACCUMULATE across per-call Train — one
        // StepPerBatch advance per Train call. Pre-#1470 this stayed at 0/1
        // because the per-call path didn't persist/advance the scheduler.
        Assert.Equal(stepBefore + batches, stepAfter);

        // (b) Noam ramps linearly up to t=warmup, so the LR at the end of warmup
        // must be materially larger than the warmup-step-1 value. A frozen LR
        // (the #1470 symptom) would leave lrAfter ≈ lrBefore.
        Assert.True(lrAfter > lrBefore * 1.5,
            $"Noam LR must ramp across per-call Train invocations (StepPerBatch); " +
            $"observed lrBefore={lrBefore:E4}, lrAfter={lrAfter:E4}. A near-constant LR is " +
            "the #1470 symptom — per-call training is not advancing the scheduler, or the " +
            "Adam+Noam optimizer was captured by the constant-rate fused-training kernel " +
            "instead of falling back to the eager scheduler-stepping path.");
    }

    /// <summary>
    /// Reads the model's base training optimizer — the same instance
    /// <c>Train</c> resolves through <c>GetOrCreateBaseOptimizer</c> — so the
    /// test inspects the real scheduler state, not a detached copy. The field
    /// is private on <c>NeuralNetworkBase</c>; walk the hierarchy to find it.
    /// </summary>
    private static GradientBasedOptimizerBase<float, Tensor<float>, Tensor<float>>? GetBaseTrainOptimizer(
        Transformer<float> model)
    {
        for (var t = (System.Type?)model.GetType(); t != null; t = t.BaseType)
        {
            var field = t.GetField("_baseTrainOptimizer",
                BindingFlags.Instance | BindingFlags.NonPublic);
            if (field == null) continue;
            return field.GetValue(model) as GradientBasedOptimizerBase<float, Tensor<float>, Tensor<float>>;
        }
        return null;
    }

    private static Tensor<float> BuildAllKInput(int k, int seqLen)
    {
        var t = new Tensor<float>(new[] { 1, seqLen });
        for (int s = 0; s < seqLen; s++) t[0, s] = k;
        return t;
    }

    private static Tensor<float> BuildOneHotTarget(int classIndex, int vocab)
    {
        var t = new Tensor<float>(new[] { 1, vocab });
        t[0, classIndex] = 1f;
        return t;
    }
}
