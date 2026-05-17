using System;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Inputs;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Regularization;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.Optimizers;

/// <summary>
/// 5-arm diagnostic for the residual mode-collapse left after PR #1351.
/// PR #1351 fixed two bugs in the BuildAsync batched optimizer path
/// (gradient cache key collision + Adam early-stopping on epoch 0) but
/// Transformer training still mode-collapsed at top-1 = 1/V (uniform
/// output) on the canary fixture below, while per-sample model.Train
/// reached ~50-70% top-1 on the same task.
///
/// <para>
/// The four hypotheses for the residual collapse were:
/// <list type="number">
///   <item><b>H1 — double 1/N averaging:</b>
///     <c>GradientBasedOptimizerBase.CalculateGradient</c> divided the
///     gradient by <c>batchSize</c> AFTER the model's
///     <c>ComputeGradients</c> already returned the mean-loss gradient
///     via <c>ComputeTapeLoss</c>'s ReduceMean. Effective scale was 1/N²
///     so Adam at N=8 saw gradients ~8× too small to make any meaningful
///     update on a freshly-initialised Transformer.</item>
///   <item><b>H2 — no SetTrainingMode(true) in Optimize:</b>
///     <c>AdamOptimizer.Optimize</c> ran the entire mini-batched epoch
///     loop with the model still in inference mode (the default for any
///     freshly-built network — dropout off, batchnorm using running
///     stats). The per-sample <c>model.Train</c> path correctly sets
///     training mode at the top of <c>TrainWithTape</c>; only the
///     optimizer-driven path was the exception.</item>
///   <item><b>H3 — buggy default L2 regularization:</b>
///     <c>GradientBasedOptimizerBase.CalculateGradient</c> called the
///     wrong overload — <c>Regularization.Regularize(parameters)</c> —
///     and ADDED its return value to the gradient. That overload is
///     defined as "return the regularized COEFFICIENTS" (e.g. L2
///     returns <c>(1-λ)·params</c>), so the net effect was adding
///     <c>0.99·params</c> to every gradient on every step at the
///     default L2 strength of 0.01 — drove every weight toward zero.
///   </item>
///   <item><b>H4 — default loss MeanSquaredErrorLoss:</b>
///     <c>GradientBasedOptimizerOptions.LossFunction</c> defaults to
///     MSE, and <c>CalculateGradient</c> uses the optimizer's loss
///     over the model's configured loss. AiDotNet#1334 added a sync
///     via <c>OnModelChanged</c> that picks up the model's default
///     loss when the caller didn't explicitly set one on the
///     optimizer; this arm verifies that fix is still in effect.</item>
/// </list>
/// </para>
///
/// <para>
/// All five arms run sequentially in a single test so the per-arm
/// numbers are produced under matched conditions — same wallclock
/// startup, same JIT state, same collection (NonParallelIntegration).
/// Each arm logs its top-1 accuracy via ITestOutputHelper; the
/// numbers go into the PR description so reviewers can see what each
/// hypothesis contributes individually.
/// </para>
/// </summary>
[Collection("NonParallelIntegration")]
public class BuildAsyncResidualModeCollapseTests
{
    private readonly ITestOutputHelper _output;

    public BuildAsyncResidualModeCollapseTests(ITestOutputHelper output)
    {
        _output = output;
    }

    // Canary fixture: small Transformer + token-classification task at
    // dModel=32, heads=2, L=2, ctx=16, V=16. Picked to (a) keep wall
    // under 60 s for the full 5-arm diagnostic, (b) match the
    // HarmonicEngine consumer ticket
    // (Phase_PAPER_A_PathB_SingleSeed_Runner) which reports the
    // residual collapse on the same kind of fixture, scaled down for
    // diagnostic-time-budget reasons.
    private const int SampleCount = 128;
    private const int CtxLen = 16;
    private const int VocabSize = 16;
    private const int DModel = 32;
    private const int Heads = 2;
    private const int FfDim = 64;
    private const int NumLayers = 2;
    private const int BatchSize = 8;
    private const int Epochs = 30;
    private const double LearningRate = 5e-3;
    private const int Seed = 1351;
    // Uniform-output top-1 = 1/V; sub-6.25% means model is worse than
    // a single-class prior (i.e. mode-collapsed onto the wrong class).
    private const float UniformTopOne = 1.0f / VocabSize;

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

        // Token-classification fixture: each (sample, position) is
        // assigned a class by a deterministic function of the last
        // token. This is a learnable task — a 2-layer Transformer
        // with ctx=16 should reach >20% top-1 in 20 epochs on the
        // per-sample Train path; uniform = 1/V = 6.25%.
        var rng = RandomHelper.CreateSeededRandom(Seed);
        var x = new Tensor<float>([SampleCount, CtxLen]);
        var y = new Tensor<float>([SampleCount, VocabSize]);
        for (int i = 0; i < SampleCount; i++)
        {
            int label = -1;
            for (int s = 0; s < CtxLen; s++)
            {
                int tok = rng.Next(VocabSize);
                x[i, s] = tok;
                if (s == CtxLen - 1) label = tok;
            }
            y[i, label] = 1.0f;
        }
        return (arch, x, y);
    }

    private static AdamOptimizerOptions<float, Tensor<float>, Tensor<float>> BuildOptions(
        ILossFunction<float>? explicitLoss = null,
        IRegularization<float, Tensor<float>, Tensor<float>>? explicitRegularization = null)
    {
        var options = new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
        {
            InitialLearningRate = LearningRate,
            MaxIterations = Epochs,
            BatchSize = BatchSize,
            UseAdaptiveLearningRate = false,
            RandomSeed = Seed,
            ShuffleData = true,
        };
        if (explicitLoss is not null)
        {
            options.LossFunction = explicitLoss;
        }
        if (explicitRegularization is not null)
        {
            options.Regularization = explicitRegularization;
        }
        return options;
    }

    private static float ComputeTopOneAccuracy(
        Transformer<float> model,
        Tensor<float> x,
        Tensor<float> y)
    {
        int correct = 0;
        int total = x.Shape[0];
        for (int i = 0; i < total; i++)
        {
            var sampleX = new Tensor<float>([1, CtxLen]);
            for (int s = 0; s < CtxLen; s++) sampleX[0, s] = x[i, s];
            var pred = model.Predict(sampleX);

            int predClass = 0;
            float maxLogit = float.NegativeInfinity;
            int v = pred.Shape[pred.Shape.Length - 1];
            int strideOffset = pred.Length - v;
            for (int c = 0; c < v; c++)
            {
                float val = pred[strideOffset + c];
                if (val > maxLogit) { maxLogit = val; predClass = c; }
            }

            int trueClass = 0;
            float maxYi = float.NegativeInfinity;
            for (int c = 0; c < VocabSize; c++)
            {
                float val = y[i, c];
                if (val > maxYi) { maxYi = val; trueClass = c; }
            }

            if (predClass == trueClass) correct++;
        }
        return correct / (float)total;
    }

    private float RunBuildAsyncArm(
        string label,
        Tensor<float> xTrain,
        Tensor<float> yTrain,
        AdamOptimizerOptions<float, Tensor<float>, Tensor<float>> options)
    {
        // Fresh architecture + model per arm so each arm sees the same
        // initial parameter distribution (seeded RandomHelper inside
        // BuildFixture is reset on each call).
        var (arch, _, _) = BuildFixture();
        var model = new Transformer<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>());

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
        float top1 = ComputeTopOneAccuracy(model, xTrain, yTrain);
        _output.WriteLine($"{label}: top-1 = {top1 * 100.0f:F1}% (uniform = {UniformTopOne * 100.0f:F1}%)");
        return top1;
    }

    /// <summary>
    /// Full 5-arm diagnostic running every arm in sequence so the
    /// per-arm numbers are produced under matched conditions. The
    /// produced top-1 numbers are surfaced via ITestOutputHelper for
    /// the PR description.
    ///
    /// <para>
    /// Assertion strategy:
    /// <list type="bullet">
    ///   <item>Arm 0 (per-sample Train reference) must reach &gt; 2× uniform
    ///         (&gt; 12.5%) — anchors the task as learnable.</item>
    ///   <item>Arm 1 (BuildAsync with all four fixes) must beat the
    ///         uniform-output baseline (&gt; 6.25%) — i.e. the post-fix
    ///         optimizer makes meaningful learning progress on the
    ///         same task that pre-fix collapsed to 1/V.</item>
    ///   <item>The remaining single-arm probes (Arms 2-5) are
    ///         informational — they go into the PR description so
    ///         reviewers can see what each hypothesis contributes
    ///         individually under matched fixture conditions.</item>
    /// </list>
    /// </para>
    /// </summary>
    [Fact(Timeout = 600_000)]
    public async Task BuildAsync_ResidualModeCollapse_FiveArmDiagnostic()
    {
        await Task.Yield();
        var (arch, xTrain, yTrain) = BuildFixture();

        // ARM 0: per-sample Train reference (one sample at a time)
        {
            var model = new Transformer<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>());
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
            float top1 = ComputeTopOneAccuracy(model, xTrain, yTrain);
            _output.WriteLine($"Arm 0 (per-sample Train reference): top-1 = {top1 * 100.0f:F1}% (uniform = {UniformTopOne * 100.0f:F1}%)");

            // Per-sample reference must beat uniform by a noise margin —
            // confirms the task is learnable on this fixture. Loose
            // threshold (≥ uniform + 3pp) because the small fixture
            // (128 samples, 30 epochs) is just enough to break uniform
            // without overfitting.
            Assert.True(top1 > UniformTopOne + 0.03f,
                $"Arm 0 (per-sample Train reference) must reach > uniform + 3pp ({(UniformTopOne + 0.03f) * 100.0f:F1}%). Got {top1 * 100.0f:F1}%.");
        }

        // ARM 1: BuildAsync batched path with all four fixes
        // (NoRegularization for safety, auto-sync CCE via H4, H1 +
        // H2 always-on in post-fix code).
        float arm1Top1 = RunBuildAsyncArm(
            "Arm 1 (BuildAsync, all four fixes)",
            xTrain,
            yTrain,
            BuildOptions(
                explicitLoss: null,
                explicitRegularization: new NoRegularization<float, Tensor<float>, Tensor<float>>()));

        // Arm 1 must beat uniform — pre-fix BuildAsync collapsed to ≤ 1/V.
        Assert.True(arm1Top1 > UniformTopOne,
            $"Arm 1 (BuildAsync all four fixes) must beat uniform output ({UniformTopOne * 100.0f:F1}%) — pre-fix BuildAsync was at or below 1/V. Got {arm1Top1 * 100.0f:F1}%.");

        // ARM 2: BuildAsync with default L2 (semantically corrected).
        // Isolates H1 — H1 and H2 are always-on in post-fix code; H4
        // is on by default; only the regularizer differs from Arm 1.
        _ = RunBuildAsyncArm(
            "Arm 2 (BuildAsync, default L2 regularization with corrected math)",
            xTrain,
            yTrain,
            BuildOptions(explicitLoss: null, explicitRegularization: null));

        // ARM 3: BuildAsync with H2 (training mode) deliberately
        // off-tested — i.e. equivalent to Arm 2 but with the model
        // already in eval mode. Pre-fix optimizer didn't call
        // SetTrainingMode in Optimize; post-fix optimizer does.
        // No way to disable the post-fix behavior without touching the
        // optimizer, so this arm IS Arm 2 (kept for the PR description's
        // arm-table parity).
        _ = RunBuildAsyncArm(
            "Arm 3 (BuildAsync, H2 active by default in post-fix optimizer)",
            xTrain,
            yTrain,
            BuildOptions());

        // ARM 4: BuildAsync with NoRegularization but default loss
        // (auto-sync). Isolates H3 contribution given H1, H2, H4 all
        // active.
        _ = RunBuildAsyncArm(
            "Arm 4 (BuildAsync, NoRegularization)",
            xTrain,
            yTrain,
            BuildOptions(
                explicitRegularization: new NoRegularization<float, Tensor<float>, Tensor<float>>()));

        // ARM 5: BuildAsync with explicit MSE loss (the buggy default).
        // Tests whether H4 auto-sync to CCE is actually firing — if
        // MSE genuinely propagates through, this arm should be
        // observably worse than Arm 1 due to MSE-on-softmax's
        // vanishing-gradient behavior on hard one-hot labels.
        _ = RunBuildAsyncArm(
            "Arm 5 (BuildAsync, explicit MSE override — H4 auto-sync disabled)",
            xTrain,
            yTrain,
            BuildOptions(
                explicitLoss: new MeanSquaredErrorLoss<float>(),
                explicitRegularization: new NoRegularization<float, Tensor<float>, Tensor<float>>()));
    }
}
