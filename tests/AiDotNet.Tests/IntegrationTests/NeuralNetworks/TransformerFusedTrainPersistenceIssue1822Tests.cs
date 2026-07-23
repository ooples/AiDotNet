using System;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Regression suite for ooples/AiDotNet#1822 — <c>Transformer&lt;T&gt;.Train(input,
/// expectedOutput)</c> was a SILENT NO-OP: it reported a decreasing loss yet
/// applied no weight update, so <see cref="NeuralNetworkBase{T}.GetParameters"/>,
/// serialization, and the eager Predict path all saw an untrained model.
///
/// <para>
/// <b>Root cause:</b> for a token/embedding Transformer (SequenceClassification,
/// 2-D integer input), the fused compiled-training fast path's specialized
/// forward captured parameter DATA by copy rather than aliasing the live
/// backing arrays. <c>plan.Step()</c> trained an internal buffer (so
/// <c>GetLastLoss</c> fell), but never wrote the update back to the network's
/// live parameter tensors — every layer's parameters stayed byte-for-byte at
/// their initial values. The eager tape path updates the live tensors
/// correctly; a plain <see cref="FeedForwardNeuralNetwork{T}"/> persists on the
/// fused path (see FusedOptimizerParityTests), so the defect was specific to the
/// embedding-graph shape.
/// </para>
///
/// <para>
/// <b>Fix:</b> <see cref="NeuralNetworkBase{T}"/> now verifies, on the first
/// fused step for a model, that the step actually moved the live parameters. If
/// a fused step reports a real loss but persists nothing, the fused path is
/// sticky-disabled for that run and training falls through to the correct eager
/// tape — so <c>model.Train(x, y)</c> always learns, whatever path it takes.
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
    // has really learned, independent of any fused-plan-internal buffer.
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

    /// <summary>
    /// The canonical #1822 guard: with the DEFAULT (fused-eligible) training path,
    /// batched <c>model.Train(x, y)</c> must (a) actually change the model's
    /// parameters and (b) drive the model's own predictions below the uniform-prior
    /// loss ln(V). Before the fix both failed — parameters were frozen at init and
    /// Predict stayed at ln(V).
    /// </summary>
    [Fact]
    public void Transformer_BatchedTrain_PersistsUpdates_AndLearns()
    {
        var model = new Transformer<float>(
            MakeArch(),
            lossFunction: new CategoricalCrossEntropyLoss<float>(),
            // Constant-LR Adam so the assertion isolates parameter PERSISTENCE,
            // not the default Noam-warmup ramp (which is a separate, documented
            // small-LR concern). The fused fast path still engages for this
            // plain Adam — this is exactly the path #1822 found broken.
            optimizer: new AdamOptimizer<float, Tensor<float>, Tensor<float>>(
                null,
                new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
                {
                    InitialLearningRate = 3e-3,
                    LearningRateScheduler = new AiDotNet.LearningRateSchedulers.ConstantLRScheduler(3e-3),
                    UseAdaptiveLearningRate = false,
                }));

        var (x, y) = MakeData();

        // Warm up lazy weight materialisation so the baseline snapshot is the real
        // initialised weights, not the pre-Forward placeholders.
        model.SetTrainingMode(false);
        _ = model.Predict(x);
        model.SetTrainingMode(true);

        double lnV = Math.Log(V);
        var pBefore = model.GetParameters();
        double l1Before = ParamL1(pBefore);
        double nllBefore = PredictNll(model, x, y);

        for (int step = 0; step < 120; step++)
            model.Train(x, y);

        model.SetTrainingMode(false);
        var pAfter = model.GetParameters();
        double l1After = ParamL1(pAfter);
        double nllAfter = PredictNll(model, x, y);

        // Per-element parameter movement (not just an aggregate that could coincide).
        double maxAbsDelta = 0;
        Assert.Equal(pBefore.Length, pAfter.Length);
        for (int i = 0; i < pBefore.Length; i++)
            maxAbsDelta = Math.Max(maxAbsDelta, Math.Abs((double)pAfter[i] - (double)pBefore[i]));

        _output.WriteLine($"paramL1 {l1Before:F4} -> {l1After:F4}  maxAbsDelta={maxAbsDelta:E3}");
        _output.WriteLine($"PredictNll {nllBefore:F4} -> {nllAfter:F4}  (lnV={lnV:F4})");

        // (a) Train() must persist a weight update to the model's live parameters.
        Assert.True(maxAbsDelta > 1e-4,
            $"Train() did not change the model parameters (maxAbsDelta={maxAbsDelta:E3}) — " +
            "the fused compiled step reported a loss but never wrote back to the live tensors (#1822).");

        // (b) The model's own predictions must actually improve past the uniform prior.
        Assert.True(nllAfter < lnV - 0.25,
            $"Predict loss {nllAfter:F4} did not drop meaningfully below ln(V)={lnV:F4} — the model did not learn (#1822).");
        Assert.True(nllAfter < nllBefore - 0.25,
            $"Predict loss did not improve over the untrained baseline ({nllBefore:F4} -> {nllAfter:F4}) (#1822).");
    }
}
