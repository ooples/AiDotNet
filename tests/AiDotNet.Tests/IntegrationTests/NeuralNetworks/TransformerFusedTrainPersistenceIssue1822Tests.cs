using System;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Training;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Regression guard for ooples/AiDotNet#1822 — a token/embedding
/// <see cref="Transformer{T}"/> (SequenceClassification, 2-D integer input)
/// trained with <c>model.Train(input, expectedOutput)</c> must actually LEARN,
/// and must do so on the <b>fused compiled fast path</b> (the default for
/// float + Adam/SGD) — NOT by silently falling back to the eager tape.
///
/// <para>
/// #1822 was reported as a "silent no-op": the fused compiled step supposedly
/// captured parameter data by copy and never wrote the update back to the
/// live parameter tensors. The root-cause investigation (see PR discussion)
/// showed the fused path in fact <b>aliases the live backing arrays</b>
/// (<c>CompiledTrainingPlan.ConfigureOptimizerFloat</c> registers each
/// parameter's <c>GetLiveBackingArrayAllowingPaddingOrNull()</c> as the
/// optimizer's in-place update target, and the traced forward records the same
/// live <c>EmbeddingLayer._embeddingTensor</c> instance), so
/// <c>plan.Step()</c> mutates the model's own parameters in place. The prior
/// "detect-no-persistence-then-sticky-disable-fused" guard was therefore
/// removed: it degraded every embedding model to the slower eager path to work
/// around a defect that does not exist. This test pins the real contract —
/// fused stays engaged AND persists — so any future regression that decouples
/// the compiled plan from the live tensors fails loudly here instead of being
/// silently masked by an eager fallback.
/// </para>
/// </summary>
public class TransformerFusedTrainPersistenceIssue1822Tests
{
    private readonly ITestOutputHelper _output;

    public TransformerFusedTrainPersistenceIssue1822Tests(ITestOutputHelper output)
    {
        _output = output;
    }

    private const int V = 16;
    private const int Ctx = 6;
    private const int N = 64;
    private const int Steps = 120;

    private static TransformerArchitecture<float> MakeArch() => new(
        inputType: InputType.TwoDimensional,
        taskType: NeuralNetworkTaskType.SequenceClassification,
        numEncoderLayers: 1,
        numDecoderLayers: 0,
        numHeads: 4,
        modelDimension: 32,
        feedForwardDimension: 64,
        inputSize: Ctx,
        outputSize: V,
        maxSequenceLength: Ctx,
        vocabularySize: V,
        randomSeed: 123);

    // Deterministic learnable batch: target token = (last context token + 1) mod V.
    private static (Tensor<float> x, Tensor<float> y) MakeData()
    {
        var rng = new Random(7);
        var x = new Tensor<float>(new[] { N, Ctx });
        var y = new Tensor<float>(new[] { N, V });
        for (int i = 0; i < N; i++)
        {
            int last = 0;
            for (int s = 0; s < Ctx; s++) { int t = rng.Next(V); x[i, s] = t; last = t; }
            y[i, (last + 1) % V] = 1f;
        }
        return (x, y);
    }

    private static double ParamL1(Vector<float> p)
    {
        double s = 0;
        for (int i = 0; i < p.Length; i++) s += Math.Abs((double)p[i]);
        return s;
    }

    // Cross-entropy from the model's ACTUAL predictions (the SequenceClassification
    // head applies softmax, so rows are probabilities) — reflects what the model
    // has really learned, read straight off the model's live parameters (not any
    // fused-plan-internal buffer).
    private static double PredictNll(Transformer<float> model, Tensor<float> x, Tensor<float> y)
    {
        var pred = model.Predict(x);
        int rows = x.Shape[0];
        double tot = 0;
        for (int i = 0; i < rows; i++)
            for (int v = 0; v < V; v++)
                if (y[i, v] > 0.5f)
                    tot += -Math.Log(Math.Max(1e-9, (double)pred[i, v]));
        return tot / rows;
    }

    private static Transformer<float> MakeAdamModel() => new Transformer<float>(
        MakeArch(),
        lossFunction: new CategoricalCrossEntropyLoss<float>(),
        optimizer: new AdamOptimizer<float, Tensor<float>, Tensor<float>>(
            null,
            new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                InitialLearningRate = 3e-3,
                LearningRateScheduler = new AiDotNet.LearningRateSchedulers.ConstantLRScheduler(3e-3),
                UseAdaptiveLearningRate = false,
            }));

    private static double L2Sq(Vector<float> p)
    {
        double s = 0; for (int i = 0; i < p.Length; i++) { double v = (double)p[i]; s += v * v; } return s;
    }

    /// <summary>
    /// #1822 all-zero-init edge case (the scenario the withdrawn band-aid's
    /// <c>checksumBefore == 0</c>-guard removal targeted). Force EVERY trainable parameter to
    /// exactly zero (so a sum-of-squares checksum is 0.0), then train on the FUSED path. Even
    /// from this degenerate start the fused step must move params AWAY from zero on the first
    /// step (checksumAfter &gt; 0) and reduce the held-out loss below ln(V) — i.e. the fused
    /// "no-op at all-zero init" does not occur; the compiled optimizer writes to the live
    /// tensors regardless of their starting value.
    /// <para>
    /// Note: this all-zero start is SYNTHETIC (forced via UpdateParameters). Shipped layers
    /// random-init their weight matrices (only biases / some norms start at zero), so a real
    /// trainable Transformer never has a 0.0 whole-model checksum — but we pin the behaviour
    /// here anyway so the fused path is proven correct at the boundary.
    /// </para>
    /// </summary>
    [Fact]
    public void Transformer_AllZeroInit_FusedPath_MovesOffZero_AndLearns()
    {
        var (x, y) = MakeData();
        double lnV = Math.Log(V);

        CompiledTapeTrainingStep<float>.Invalidate();
        CompiledTapeTrainingStep<float>.ResetFusedStepCount();
        // Current => (_current ?? new Default()) hands back a throwaway instance, so mutating
        // Current.EnableCompilation is a no-op; SetCurrent installs the fused (compiled) path.
        AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.SetCurrent(
            new AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions { EnableCompilation = true });
        try
        {
            var model = MakeAdamModel();
            model.SetTrainingMode(false); _ = model.Predict(x); model.SetTrainingMode(true);
            int nParams = model.GetParameters().Length;
            model.UpdateParameters(new Vector<float>(nParams)); // force ALL trainable params to 0

            double checksumBefore = L2Sq(model.GetParameters());
            double nllBefore = PredictNll(model, x, y);

            long fusedBefore = CompiledTapeTrainingStep<float>.GetFusedStepCount();
            for (int s = 0; s < 10; s++) model.Train(x, y);
            long fusedSteps = CompiledTapeTrainingStep<float>.GetFusedStepCount() - fusedBefore;

            model.SetTrainingMode(false);
            double checksumAfter = L2Sq(model.GetParameters());
            double nllAfter = PredictNll(model, x, y);
            _output.WriteLine($"all-zero-init: fusedSteps={fusedSteps}/10 checksum {checksumBefore:E3}->{checksumAfter:E3} NLL {nllBefore:F4}->{nllAfter:F4} (lnV={lnV:F4})");

            Assert.Equal(0.0, checksumBefore, 12);        // genuinely all-zero start
            Assert.Equal(10, fusedSteps);                 // fused path used for every step
            Assert.True(checksumAfter > 1e-3,             // params moved OFF zero (persisted)
                $"fused step did not move all-zero-init params off zero (checksumAfter={checksumAfter:E3}).");
            Assert.True(nllAfter < lnV - 0.05,            // and it learned
                $"NLL {nllAfter:F4} did not drop below ln(V)={lnV:F4} from all-zero init.");
        }
        finally
        {
            AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.SetCurrent(null);
        }
    }

    /// <summary>
    /// The #1822 contract: with the DEFAULT (fused-eligible) training path, batched
    /// <c>model.Train(x, y)</c> must (1) run on the FUSED compiled path — proven by
    /// the per-thread fused-step counter equalling the number of <c>Train</c> calls,
    /// so we know it did NOT fall back to eager; (2) actually change the model's LIVE
    /// parameters; and (3) drive the model's own predictions well below the
    /// uniform-prior loss ln(V). A silent-no-op fused step would satisfy none of these.
    /// </summary>
    [Fact]
    public void Transformer_BatchedTrain_UsesFusedPath_PersistsUpdates_AndLearns()
    {
        var model = new Transformer<float>(
            MakeArch(),
            lossFunction: new CategoricalCrossEntropyLoss<float>(),
            // Constant-LR Adam so the assertion isolates parameter PERSISTENCE, not
            // the default Noam-warmup ramp (a separate, documented small-LR concern).
            // The fused fast path still engages for this plain Adam — this is exactly
            // the path #1822 claimed was a no-op.
            optimizer: new AdamOptimizer<float, Tensor<float>, Tensor<float>>(
                null,
                new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
                {
                    InitialLearningRate = 3e-3,
                    LearningRateScheduler = new AiDotNet.LearningRateSchedulers.ConstantLRScheduler(3e-3),
                    UseAdaptiveLearningRate = false,
                }));

        var (x, y) = MakeData();

        // Start from a clean per-thread compiled-training state so the fused-step
        // counter reflects only this test's Train calls.
        CompiledTapeTrainingStep<float>.Invalidate();
        CompiledTapeTrainingStep<float>.ResetFusedStepCount();

        // Warm up lazy weight materialisation so the baseline snapshot is the real
        // initialised weights, not the pre-Forward placeholders.
        model.SetTrainingMode(false);
        _ = model.Predict(x);
        model.SetTrainingMode(true);

        double lnV = Math.Log(V);
        var pBefore = model.GetParameters();
        double l1Before = ParamL1(pBefore);
        double nllBefore = PredictNll(model, x, y);

        long fusedBefore = CompiledTapeTrainingStep<float>.GetFusedStepCount();
        for (int step = 0; step < Steps; step++)
            model.Train(x, y);
        long fusedAfter = CompiledTapeTrainingStep<float>.GetFusedStepCount();

        model.SetTrainingMode(false);
        var pAfter = model.GetParameters();
        double l1After = ParamL1(pAfter);
        double nllAfter = PredictNll(model, x, y);

        double maxAbsDelta = 0;
        Assert.Equal(pBefore.Length, pAfter.Length);
        for (int i = 0; i < pBefore.Length; i++)
            maxAbsDelta = Math.Max(maxAbsDelta, Math.Abs((double)pAfter[i] - (double)pBefore[i]));

        long fusedSteps = fusedAfter - fusedBefore;
        _output.WriteLine($"fusedStepsUsed={fusedSteps}/{Steps}  (fused-path-used = {fusedSteps == Steps})");
        _output.WriteLine($"paramL1 {l1Before:F4} -> {l1After:F4}  maxAbsDelta={maxAbsDelta:E3}");
        _output.WriteLine($"PredictNll {nllBefore:F4} -> {nllAfter:F4}  (lnV={lnV:F4})");

        // (1) Every Train() call must have run on the FUSED compiled path — not the
        //     eager fallback. This is the assertion the removed production band-aid
        //     could only have violated (by sticky-disabling fused); we require the
        //     opposite — fused engaged for all Steps.
        Assert.Equal(Steps, fusedSteps);

        // (2) Train() must persist a weight update to the model's LIVE parameters.
        Assert.True(maxAbsDelta > 1e-4,
            $"Train() did not change the model parameters (maxAbsDelta={maxAbsDelta:E3}) — " +
            "a fused compiled step must write its update back to the live tensors (#1822).");

        // (3) The model's own predictions (read off the live params) must improve
        //     well past the uniform prior — i.e. the persisted updates actually learn.
        Assert.True(nllAfter < lnV - 0.25,
            $"Predict loss {nllAfter:F4} did not drop meaningfully below ln(V)={lnV:F4} — the model did not learn (#1822).");
        Assert.True(nllAfter < nllBefore - 0.25,
            $"Predict loss did not improve over the untrained baseline ({nllBefore:F4} -> {nllAfter:F4}) (#1822).");
    }
}
