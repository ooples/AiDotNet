using System;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Optimizers.Fused;
using AiDotNet.Training;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Regression + integration suite for AiDotNet#1470 — a from-scratch Transformer
/// trained via the per-call minibatch pattern (<c>MaxIterations=1</c> + an
/// external epoch loop calling <c>model.Train(xBatch, yBatch)</c> repeatedly)
/// stalled: the default Noam learning-rate schedule stayed frozen at its
/// warmup-step-1 value (~1e-7) instead of ramping, so loss never left the
/// uniform floor (PPL ≈ V).
///
/// <para>
/// <b>Root cause:</b> the compiled fused-training kernel takes <c>lr</c> as a
/// per-step scalar (it evaluates <c>LrSchedule.GetLr(step)</c> every optimizer
/// step — the same model as PyTorch <c>fused=True</c>). The fused path
/// supported Cosine/Exponential/OneCycle/LinearWarmupCosine but NOT Noam, so an
/// Adam+Noam Transformer either fell back to the slower eager tape or (pre-gate)
/// was committed to a constant-rate fused config that froze the warmup ramp.
/// </para>
///
/// <para>
/// <b>Fix:</b> AiDotNet.Tensors 0.88.0 (PR #504) adds
/// <c>LrSchedule.Noam(modelDim, warmup, factor)</c>, and
/// <c>GradientBasedOptimizerBase.TryGetFusedLrSchedule</c> now maps
/// <see cref="NoamSchedule"/> to it. So Adam+Noam runs on the fused fast path
/// with a correct, per-step warmup ramp — bit-identical to the eager schedule
/// (both use <c>t = step</c>, 1-based) — instead of being forced to eager or
/// frozen.
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
    /// The fix seam: an Adam optimizer carrying a Noam schedule must now MAP to
    /// a fused-optimizer config (no eager fallback), and the mapped schedule
    /// must ramp — warmup-end LR far above the warmup-step-1 LR. Pre-fix this
    /// either returned false (forced eager) or carried a constant rate (#1470
    /// freeze). Positive control: a no-scheduler Adam maps with a null (constant)
    /// schedule and still takes the fused path.
    /// </summary>
    [Fact]
    public void AdamWithNoamSchedule_MapsToFusedConfig_WithRampingSchedule()
    {
        const int d = 512, warmup = 4000;

        var noamAdam = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(
            null,
            new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                InitialLearningRate = 1e-3,
                Beta2 = 0.98,
                Epsilon = 1e-9,
                LearningRateScheduler = new NoamSchedule(modelDimension: d, warmupSteps: warmup),
                SchedulerStepMode = SchedulerStepMode.StepPerBatch,
            });

        // TryGetFusedOptimizerConfig is an explicit IFusedOptimizerSpec impl
        // (internal; reachable via InternalsVisibleTo "AiDotNetTests").
        Assert.True(((IFusedOptimizerSpec)noamAdam).TryGetFusedOptimizerConfig(out var cfg),
            "Adam+Noam must now MAP to a fused-optimizer config — the fused kernel evaluates " +
            "the schedule per step, so there's no reason to force the slower eager path (#1470).");

        var schedule = cfg.Schedule;
        Assert.NotNull(schedule); // Noam is a real per-step schedule, not constant

        // The schedule must ramp during warmup: lr(warmup) is the peak, far above
        // lr(1). A constant/frozen schedule (the #1470 symptom) would make these equal.
        double lrStart = schedule!.GetLr(1);
        double lrPeak = schedule.GetLr(warmup);
        _output.WriteLine($"fused Noam schedule: GetLr(1)={lrStart:E4}, GetLr({warmup})={lrPeak:E4}, ratio={lrPeak / lrStart:F1}x");
        Assert.True(lrPeak > lrStart * 10,
            $"Fused Noam schedule must ramp during warmup (peak {lrPeak:E4} ≫ start {lrStart:E4}); " +
            "an unramped schedule is the #1470 freeze.");

        // Matches the paper formula (and therefore the eager NoamSchedule, which
        // uses the same t = step convention) at the warmup peak.
        double expectedPeak = Math.Pow(d, -0.5) * Math.Pow(warmup, -0.5);
        Assert.Equal(expectedPeak, lrPeak, 9);

        // Positive control: constant-LR Adam still uses the fused path (null schedule).
        var plainAdam = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(
            null,
            new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 1e-3 });
        Assert.True(((IFusedOptimizerSpec)plainAdam).TryGetFusedOptimizerConfig(out var plainCfg),
            "A constant-LR Adam (no scheduler) should still use the fused fast path.");
        Assert.Null(plainCfg.Schedule); // constant rate → no per-step schedule
    }

    /// <summary>
    /// The fused Noam schedule must produce the SAME per-step LR sequence as the
    /// eager <see cref="NoamSchedule"/> — otherwise the fused fast path would
    /// train differently from the eager reference. Both map <c>t = step</c>
    /// (1-based) to <c>lr(t=N)</c> on batch N.
    /// </summary>
    [Fact]
    public void FusedNoamSchedule_MatchesEagerNoamSchedule_StepForStep()
    {
        const int d = 256, warmup = 100;
        var eager = new NoamSchedule(modelDimension: d, warmupSteps: warmup);

        var adam = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(
            null,
            new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                LearningRateScheduler = new NoamSchedule(modelDimension: d, warmupSteps: warmup),
                SchedulerStepMode = SchedulerStepMode.StepPerBatch,
            });
        Assert.True(((IFusedOptimizerSpec)adam).TryGetFusedOptimizerConfig(out var cfg));
        var fused = cfg.Schedule!;

        // Eager batch N uses lr(t=N): the ctor sets lr(t=1) for batch 1, and each
        // OnBatchEnd Step() advances to the next. Replay that sequence and compare
        // to the fused schedule's GetLr(N).
        double eagerLr = eager.CurrentLearningRate;     // batch 1 (t=1), set in ctor
        for (int n = 1; n <= 3 * warmup; n++)
        {
            Assert.Equal(eagerLr, fused.GetLr(n), 9);
            eagerLr = eager.Step();                     // advance to batch n+1's lr
        }
    }

    /// <summary>
    /// End-to-end: a default-optimizer (Adam + Noam) Transformer trained via the
    /// per-call minibatch pattern (<c>MaxIterations=1</c> + an external epoch loop
    /// calling <c>model.Train</c>) must (1) actually ENGAGE the fused-compiled
    /// training path — proving Noam no longer forces the slow eager fallback —
    /// and (2) break through the uniform-softmax floor (loss &lt; ln(V)),
    /// proving the Noam LR ramped instead of freezing (#1470).
    ///
    /// <para>
    /// <b>Methodology note (why a real minibatch):</b> each <c>Train</c> call
    /// presents ALL <c>vocab</c> examples as one <c>[vocab, seqLen]</c> minibatch,
    /// NOT one example per call. Presenting the 8 maps one-at-a-time (batch size 1)
    /// is degenerate online SGD: each single example is driven to p≈1 and the next
    /// call overwrites it (catastrophic interference), so the loss can never settle
    /// below ln(V) no matter how correctly the Noam LR ramps — the convergence
    /// proxy would then test the optimizer-interference property, not #1470. The
    /// minibatch lets the averaged gradient fit all maps jointly, so convergence
    /// cleanly isolates the thing under test: Noam ramping on the fused path.
    /// (Verified: single-example overfit reaches p=1.0 stably; batch-1 cycling
    /// diverges and worsens with training; the minibatch reaches avgNll≈0, 8/8.)
    /// </para>
    /// </summary>
    [Fact]
    public async Task Transformer_PerCallTrain_DefaultNoam_EngagesFusedPath_AndConverges()
    {
        await Task.Yield();
        const int vocab = 8;
        const int seqLen = 4;
        const int totalEpochs = 800;
        const int modelDim = 32;
        const int ffDim = 64;
        const int warmup = 64;   // small so the ramp clears warmup within the budget

        // No explicit optimizer → CreateDefaultVaswaniOptimizer (Adam + Noam,
        // StepPerBatch) — the exact default the #1470 repro used.
        var arch = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            numEncoderLayers: 2,
            numDecoderLayers: 0,
            numHeads: 4,
            modelDimension: modelDim,
            feedForwardDimension: ffDim,
            inputSize: seqLen,
            outputSize: vocab,
            maxSequenceLength: seqLen,
            vocabularySize: vocab,
            warmupSteps: warmup,
            // This is a CONVERGENCE test for the fused Noam LR ramp (#1470), not a dropout test.
            // Two reasons to pin these: (1) dropoutRate:0 — the production default (0.1) is a
            // regularizer that fights the memorization this test measures, and the train-mode-
            // (dropped) vs eval-mode-(full) forward gap then caps eval accuracy on a short budget;
            // (2) randomSeed:42 — without a seed the dropout mask stream uses the non-reproducible
            // ThreadSafeRandom, so the test's convergence became ORDER-DEPENDENT (passed alone at
            // avgNll≈1.36, intermittently failed when run after other tests advanced the shared
            // stream). Both make the Noam-ramp signal deterministic and isolated.
            dropoutRate: 0.0,
            randomSeed: 42);

        var model = new Transformer<float>(
            arch,
            lossFunction: new CategoricalCrossEntropyLoss<float>());
        model.SetTrainingMode(true);

        CompiledTapeTrainingStep<float>.ResetFusedStepCount();

        // One minibatch of all `vocab` maps: input [vocab, seqLen], target
        // [vocab, vocab] (one-hot). The deterministic byte-LM task maps the
        // constant sequence [k,k,..] -> class (k+3) % vocab.
        var batchInput = new Tensor<float>(new[] { vocab, seqLen });
        var batchTarget = new Tensor<float>(new[] { vocab, vocab });
        for (int k = 0; k < vocab; k++)
        {
            for (int s = 0; s < seqLen; s++) batchInput[k, s] = k;
            batchTarget[k, (k + 3) % vocab] = 1f;
        }

        for (int epoch = 0; epoch < totalEpochs; epoch++)
        {
            model.Train(batchInput, batchTarget);
        }

        long fusedSteps = CompiledTapeTrainingStep<float>.GetFusedStepCount();
        model.SetTrainingMode(false);

        // (1) The fused-compiled path must have actually run. Pre-fix, Adam+Noam
        // was declined by TryGetFusedLrSchedule and every step fell back to the
        // eager tape (fusedSteps == 0). The fix maps Noam → LrSchedule.Noam so
        // the fast path engages.
        _output.WriteLine($"fused steps engaged: {fusedSteps} / {totalEpochs}");
        Assert.True(fusedSteps > 0,
            "Adam+Noam Transformer must engage the fused-compiled training path now that " +
            "LrSchedule.Noam exists — fusedSteps==0 means it's still forced onto the eager tape (#1470).");

        // (2) Convergence below the uniform floor proves the Noam LR ramped on
        // the fused path. Pre-fix the LR was frozen at warmup-step-1 and loss
        // stayed at exactly ln(V) (PPL = V).
        double totalNll = 0;
        int correct = 0;
        for (int k = 0; k < vocab; k++)
        {
            var input = BuildAllKInput(k, seqLen);
            var pred = model.Predict(input);
            int expectedNext = (k + 3) % vocab;
            float pTarget = pred.Length == vocab ? pred[expectedNext] : pred[0, expectedNext];
            totalNll += -Math.Log(Math.Max((double)pTarget, 1e-9));

            int argmax = 0;
            float maxVal = float.MinValue;
            for (int v = 0; v < vocab; v++)
            {
                float val = pred.Length == vocab ? pred[v] : pred[0, v];
                if (val > maxVal) { maxVal = val; argmax = v; }
            }
            if (argmax == expectedNext) correct++;
        }
        double avgNll = totalNll / vocab;
        double lnV = Math.Log(vocab);
        _output.WriteLine($"avgNll={avgNll:F3}  ln(V)={lnV:F3}  PPL={Math.Exp(avgNll):F2}  top-1={correct}/{vocab}");

        Assert.True(avgNll < lnV * 0.95,
            $"Default-Noam Transformer should train below ln(V)*0.95={lnV * 0.95:F3} on a deterministic " +
            $"byte-LM task; observed avgNll={avgNll:F3}. A value at/near ln(V) means the Noam LR is frozen " +
            "at warmup-step-1 — the #1470 regression.");
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
