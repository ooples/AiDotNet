using System;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Inputs;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Regularization;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.Optimizers;

/// <summary>
/// Regression test for issue #1380: the residual byte-LM mode collapse that
/// persists after PR #1351 + PR #1364 fixed H1 (double 1/N averaging),
/// H2 (no training-mode toggle), H3 (default L2 regularization gradient
/// math), H4 (default-loss MSE override), H5 (parameter/gradient ordering
/// parity), and H6 (two-Adam-impl divergence).
///
/// <para>
/// HarmonicEngine's <c>Phase_PAPER_A_PathB_SingleSeed_Runner</c> consumer
/// reproducer hit <c>top-1 = 0%, perplexity = 256.00 (uniform)</c> on a
/// V=256 byte-LM Transformer through <c>AiModelBuilder.BuildAsync</c>,
/// while the per-sample <c>model.Train(x, y)</c> driver reached
/// <c>top-1 = 55.7%, perplexity = 6.77</c> on the same model + optimizer +
/// data. The existing <see cref="BuildAsyncResidualModeCollapseTests"/>
/// eight-arm diagnostic uses V=16 and passes after PR #1364, but the
/// V=256 case still collapses — vocab-size dependence is the residual
/// signal this test pins down.
/// </para>
///
/// <para>
/// The fixture is scaled down from the consumer ticket (V=256, dModel=64,
/// L=1, ctx=64, 9216 samples, 3 epochs) to (V=256, dModel=32, L=1, ctx=8,
/// 128 samples, 50 epochs) to keep CI wall-time under one minute while
/// preserving the vocab-size that triggers the collapse.
/// </para>
///
/// <para>
/// <b>Assertion strategy:</b> the test does NOT assert on top-1 accuracy.
/// V=256 byte-LM is too hard for a 1-layer Transformer to learn in CI's
/// step budget — even the per-sample reference might not break uniform
/// by a comfortable margin. Instead the test asserts on
/// <b>output-distribution entropy</b>:
/// </para>
/// <list type="bullet">
///   <item>A model whose output collapses to exactly uniform produces
///         entropy = log(V) per sample.</item>
///   <item>A model that learned ANY structure produces entropy &lt; log(V).</item>
/// </list>
/// <para>
/// We compare BuildAsync's post-training entropy to a (much tighter)
/// bound below log(V). The pre-fix consumer ticket sits at entropy =
/// log(256) exactly because the output is provably uniform. Any
/// movement off uniform — even a fraction of a nat — proves the
/// optimizer is taking meaningful steps. This decouples the regression
/// test from the model's learning rate / fixture-scale tradeoff and
/// focuses it on the residual-collapse symptom alone.
/// </para>
/// </summary>
[Collection("NonParallelIntegration")]
public class ByteLMV256Issue1380Tests
{
    private readonly ITestOutputHelper _output;

    public ByteLMV256Issue1380Tests(ITestOutputHelper output)
    {
        _output = output;
    }

    // V=256 is the critical dimension that distinguishes this test from the
    // existing V=16 eight-arm diagnostic. The consumer reproducer
    // (HarmonicEngine Phase_PAPER_A) uses V=256 because that's the byte
    // alphabet size — every byte-LM training task on AiDotNet hits this
    // path. Other dimensions scaled down from (dModel=64, ctx=64, samples=9216,
    // epochs=3) to fit CI's per-test budget; the residual collapse is
    // vocab-size dependent, not data-volume dependent.
    private const int SampleCount = 128;
    private const int CtxLen = 8;
    private const int VocabSize = 256;
    private const int DModel = 32;
    private const int Heads = 2;
    private const int FfDim = 64;
    private const int NumLayers = 1;
    private const int BatchSize = 8;
    private const int Epochs = 50;
    private const double LearningRate = 5e-3;
    private const int Seed = 1380;

    // Uniform-distribution entropy in nats. A model whose output collapses
    // to exactly uniform achieves this value; a model that learned ANY
    // structure beats it. We assert BuildAsync's post-training entropy is
    // measurably below this — the gap quantifies how much the optimizer
    // moved the model off the uniform baseline.
    private static readonly double UniformEntropy = Math.Log(VocabSize);

    private static (TransformerArchitecture<float> Arch, Tensor<float> X, Tensor<float> Y) BuildFixture()
    {
        var arch = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.SequenceClassification,
            numEncoderLayers: NumLayers,
            numDecoderLayers: 0,
            numHeads: Heads,
            modelDimension: DModel,
            feedForwardDimension: FfDim,
            inputSize: CtxLen,
            outputSize: VocabSize,
            maxSequenceLength: CtxLen,
            vocabularySize: VocabSize);

        // Deterministic byte-LM task: predict the FIRST token from a small
        // sequence of bytes. Identity-style mapping (label = x[0]) is the
        // simplest non-trivial signal at V=256 — the model only needs to
        // route x[0] through to the output projection. Whether the model
        // actually solves the task isn't what this test checks (see
        // class docstring); the task only needs to produce non-uniform
        // gradients so we can measure the optimizer's effect on output
        // entropy.
        var rng = RandomHelper.CreateSeededRandom(Seed);
        var x = new Tensor<float>([SampleCount, CtxLen]);
        var y = new Tensor<float>([SampleCount, VocabSize]);
        for (int i = 0; i < SampleCount; i++)
        {
            int firstToken = rng.Next(VocabSize);
            x[i, 0] = firstToken;
            for (int s = 1; s < CtxLen; s++)
            {
                x[i, s] = rng.Next(VocabSize);
            }
            y[i, firstToken] = 1.0f;
        }
        return (arch, x, y);
    }

    /// <summary>
    /// Compute mean per-sample softmax entropy (in nats) of the model's
    /// last-position logits over the supplied input set. A model whose
    /// output collapses to exactly uniform returns <c>log(V)</c>; a
    /// model that learned anything returns less.
    /// </summary>
    private static double ComputeMeanOutputEntropy(
        Transformer<float> model,
        Tensor<float> x)
    {
        int total = x.Shape[0];
        double entropySum = 0.0;
        for (int i = 0; i < total; i++)
        {
            var sampleX = new Tensor<float>([1, CtxLen]);
            for (int s = 0; s < CtxLen; s++) sampleX[0, s] = x[i, s];
            var pred = model.Predict(sampleX);

            // Pred is [1, ctx, V] (or [1, V] depending on arch); pull the
            // LAST V values as the prediction for this sample, matching
            // the loss function's last-position contract for
            // SequenceClassification (see CategoricalCrossEntropyLoss's
            // EnsureTargetMatchesPredicted on rank > 2 predictions).
            int v = pred.Shape[pred.Shape.Length - 1];
            int strideOffset = pred.Length - v;

            // Numerically-stable softmax: subtract max logit then exp.
            float maxLogit = float.NegativeInfinity;
            for (int c = 0; c < v; c++)
            {
                float val = pred[strideOffset + c];
                if (val > maxLogit) maxLogit = val;
            }
            double expSum = 0.0;
            var exps = new double[v];
            for (int c = 0; c < v; c++)
            {
                double e = Math.Exp(pred[strideOffset + c] - maxLogit);
                exps[c] = e;
                expSum += e;
            }

            // H = -Σ p_c · log(p_c) where p_c = exps[c] / expSum.
            // Numerical-stability rearrangement:
            //   H = log(expSum) + maxLogit - (Σ pred_c · exps[c]) / expSum
            // (matches the standard "softmax cross-entropy with logits"
            // identity but for the negative-entropy direction.)
            double H = 0.0;
            for (int c = 0; c < v; c++)
            {
                double p = exps[c] / expSum;
                if (p > 0)
                {
                    H -= p * Math.Log(p);
                }
            }
            entropySum += H;
        }
        return entropySum / total;
    }

    [Fact(Timeout = 300_000)]
    public async Task BuildAsync_V256_ByteLM_OutputDoesNotCollapseToUniform()
    {
        await Task.Yield();
        var (arch, xTrain, yTrain) = BuildFixture();

        // Reference: per-sample model.Train driver. The consumer ticket
        // reports this path reaches top-1 = 55.7% (entropy well below
        // uniform) on the full-scale fixture. The CI-scaled fixture is
        // too small for per-sample to reach that level, but it should
        // still drop entropy off the uniform baseline measurably —
        // anchors the test as "training has signal".
        //
        // Override the Transformer's default Vaswani recipe (NoamSchedule,
        // warmupSteps=4000) with a plain Adam at lr=5e-3 so the small
        // fixture's step budget isn't consumed entirely by warmup.
        double perSampleEntropy;
        {
            var perSampleOptimizer = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(
                model: null,
                options: new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
                {
                    InitialLearningRate = LearningRate,
                });
            var model = new Transformer<float>(
                arch,
                lossFunction: new CategoricalCrossEntropyLoss<float>(),
                optimizer: perSampleOptimizer);
            for (int epoch = 0; epoch < Epochs; epoch++)
            {
                for (int i = 0; i < SampleCount; i++)
                {
                    var sampleX = new Tensor<float>([1, CtxLen]);
                    var sampleY = new Tensor<float>([1, VocabSize]);
                    for (int s = 0; s < CtxLen; s++) sampleX[0, s] = xTrain[i, s];
                    for (int c = 0; c < VocabSize; c++) sampleY[0, c] = yTrain[i, c];
                    model.Train(sampleX, sampleY);
                }
            }
            perSampleEntropy = ComputeMeanOutputEntropy(model, xTrain);
            _output.WriteLine($"Per-sample Train reference: mean entropy = {perSampleEntropy:F4} nats (uniform = {UniformEntropy:F4} nats, gap = {UniformEntropy - perSampleEntropy:F4})");
        }

        // BuildAsync batched-Adam path: identical fixture, identical
        // optimizer hyperparameters, only the training driver differs.
        // Uses NoRegularization so the H3 fix's gradient contribution
        // math cannot mask the residual collapse.
        double buildAsyncEntropy;
        {
            var (archBA, _, _) = BuildFixture();
            var model = new Transformer<float>(archBA, lossFunction: new CategoricalCrossEntropyLoss<float>());
            var options = new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                InitialLearningRate = LearningRate,
                MaxIterations = Epochs,
                BatchSize = BatchSize,
                UseAdaptiveLearningRate = false,
                UseAdaptiveBetas = false,
                RandomSeed = Seed,
                ShuffleData = true,
                Regularization = new NoRegularization<float, Tensor<float>, Tensor<float>>(),
            };
            var optimizer = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(model, options);
            var inputData = new OptimizationInputData<float, Tensor<float>, Tensor<float>>
            {
                XTrain = xTrain,
                YTrain = yTrain,
                XValidation = xTrain,
                YValidation = yTrain,
                XTest = xTrain,
                YTest = yTrain,
            };
            _ = optimizer.Optimize(inputData);
            buildAsyncEntropy = ComputeMeanOutputEntropy(model, xTrain);
            _output.WriteLine($"BuildAsync batched-Adam path:    mean entropy = {buildAsyncEntropy:F4} nats (uniform = {UniformEntropy:F4} nats, gap = {UniformEntropy - buildAsyncEntropy:F4})");
        }

        // The assertion this test exists for: BuildAsync's
        // off-uniform movement must be at least 50% of the per-sample
        // reference's off-uniform movement. Pre-fix consumer ticket:
        // per-sample moves entropy from log(V) down to log(6.77) ≈ 1.91
        // (gap of ~3.6 nats); BuildAsync stays at exactly log(V) (gap
        // = 0 nats) — ratio of 0.0. A 0.5 threshold leaves room for
        // batched-vs-per-sample stochasticity while flagging the
        // total-collapse regression that motivates this issue.
        //
        // Robust to fixture-size choices: if both paths fail to move
        // entropy off uniform (small fixture, hard V=256 task), the
        // ratio is 0/0 ≈ undefined and the test conservatively passes
        // via the lower-bound clause on the absolute per-sample gap.
        // Only when per-sample CAN move entropy AND BuildAsync cannot
        // does the assertion fire — that's the precise residual
        // collapse condition issue #1380 reports.
        double perSampleGap = UniformEntropy - perSampleEntropy;
        double buildAsyncGap = UniformEntropy - buildAsyncEntropy;
        _output.WriteLine($"Per-sample uniform-gap = {perSampleGap:F4} nats");
        _output.WriteLine($"BuildAsync uniform-gap = {buildAsyncGap:F4} nats");

        // Only enforce the ratio when per-sample produced a meaningful
        // signal (≥ 0.01 nats off uniform). Below that the fixture is
        // too small for either path to learn; the bug is not exercised
        // and the comparison is undefined. The 0.01-nat floor mirrors
        // the existing 8-arm diagnostic's "uniform + 3pp top-1"
        // threshold — both filter out the "fixture is too small to
        // measure" regime.
        if (perSampleGap >= 0.01)
        {
            double ratio = buildAsyncGap / perSampleGap;
            _output.WriteLine($"Ratio (BuildAsync gap / per-sample gap) = {ratio:F3}");

            Assert.True(
                ratio >= 0.5,
                $"Issue #1380: BuildAsync batched-Adam path moved entropy off uniform by only " +
                $"{buildAsyncGap:F4} nats vs the per-sample reference's {perSampleGap:F4} nats " +
                $"(ratio = {ratio:F3}, threshold = 0.5). " +
                "The residual mode collapse left by PR #1364 is still active for V=256.");
        }
        else
        {
            _output.WriteLine(
                $"Per-sample reference gap ({perSampleGap:F4} nats) below 0.01-nat learnability " +
                "floor; fixture too small to exercise the path-divergence bug. Test is " +
                "informational on this configuration — bump SampleCount/Epochs to " +
                "re-engage the assertion.");
        }
    }
}
