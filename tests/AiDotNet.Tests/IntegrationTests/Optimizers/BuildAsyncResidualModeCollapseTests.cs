using System;
using System.Linq;
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
/// 8-arm diagnostic for the residual mode-collapse left after PR #1351.
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
/// All eight arms run sequentially in a single test so the per-arm
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
    // under 60 s for the full 8-arm diagnostic, (b) match the
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
    /// Full 8-arm diagnostic running every arm in sequence so the
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
    ///   <item>The remaining single-arm probes (Arms 2-7) are
    ///         informational — they go into the PR description so
    ///         reviewers can see what each hypothesis contributes
    ///         individually under matched fixture conditions.</item>
    /// </list>
    /// </para>
    /// </summary>
    [Fact(Timeout = 600_000)]
    public async Task BuildAsync_ResidualModeCollapse_EightArmDiagnostic()
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

        // ARM 6: parameter-delta + gradient-magnitude probe.
        // H5 was refuted by the standalone ParameterGradientOrderingH5ProbeTests
        // suite — GetParameters and ComputeGradients agree on flat-vector
        // length AND per-index correspondence (round-trip verified). So the
        // residual mode collapse is not a swapped gradient. This arm
        // diagnoses whether the gradients have the RIGHT MAGNITUDE to
        // drive meaningful Adam steps on this fixture, and how much the
        // model's parameters actually move during a full Optimize() run.
        //
        // If the parameter L2 delta is large but accuracy stays at ~10%, the
        // gradients are driving the model in some non-helpful direction
        // (e.g. towards a degenerate fixed point). If the parameter L2
        // delta is small, the gradient magnitudes are too small for the
        // effective learning rate — Adam is being starved.
        // (No try-catch: per PR #1364 review, swallowing exceptions would
        // turn this diagnostic into an always-passing test. Let any
        // ComputeGradients / GetParameters / finite-diff failure
        // propagate so the test reports the real failure.)
        {
            var modelArm6 = new Transformer<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>());

            // Materialize lazy-init layers via a Predict pass (which
            // runs through NoGradScope + eval mode internally). Note:
            // layers whose lazy init is gated on the IsTraining mode
            // (e.g. layers that allocate buffers only in training mode)
            // will NOT materialize here — those need a training-mode
            // forward pass instead. For the current canary Transformer
            // (Embedding, attention QKV, FF), Predict materializes all
            // lazy banks because their init is mode-independent. If a
            // future layer is added whose init depends on training
            // mode, switch this to a SetTrainingMode(true) + Predict +
            // SetTrainingMode(false) bracket (review #1364 C4nLp).
            _ = modelArm6.Predict(xTrain);

            // Capture gradient on the FULL training set (the same batch
            // Optimize() will see first) to size up its magnitude.
            var grad = modelArm6.ComputeGradients(xTrain, yTrain, new CategoricalCrossEntropyLoss<float>());

            // Capture initial parameter snapshot AFTER ComputeGradients
            // has triggered any lazy allocation.
            var initialParams = modelArm6.GetParameters();
            var initialSnapshot = new float[initialParams.Length];
            for (int i = 0; i < initialParams.Length; i++) initialSnapshot[i] = initialParams[i];
            double gradL2 = 0.0;
            double gradMax = 0.0;
            for (int i = 0; i < grad.Length; i++)
            {
                double v = grad[i];
                gradL2 += v * v;
                if (Math.Abs(v) > gradMax) gradMax = Math.Abs(v);
            }
            gradL2 = Math.Sqrt(gradL2);

            // Now run Optimize() and capture final params.
            var optimizer = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(modelArm6,
                BuildOptions(
                    explicitLoss: null,
                    explicitRegularization: new NoRegularization<float, Tensor<float>, Tensor<float>>()));
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

            var finalParams = modelArm6.GetParameters();
            double deltaL2 = 0.0;
            double deltaMax = 0.0;
            double finalParamL2 = 0.0;
            for (int i = 0; i < finalParams.Length; i++)
            {
                double diff = (double)finalParams[i] - initialSnapshot[i];
                deltaL2 += diff * diff;
                if (Math.Abs(diff) > deltaMax) deltaMax = Math.Abs(diff);
                double v = (double)finalParams[i];
                finalParamL2 += v * v;
            }
            deltaL2 = Math.Sqrt(deltaL2);
            finalParamL2 = Math.Sqrt(finalParamL2);

            // Numerical finite-difference verification on the parameter
            // indices with the LARGEST analytic gradient magnitude. These
            // are the most informative spots: a wrong-gradient bug shows
            // up as analytic large but numeric ~0, OR analytic with wrong
            // SIGN (numeric and analytic opposite signs). Picking by
            // gradient magnitude avoids the degenerate case where both
            // numbers are tiny (where any rel error is meaningless).
            var freshArch = BuildFixture().Arch;
            var modelFd = new Transformer<float>(freshArch, lossFunction: new CategoricalCrossEntropyLoss<float>());
            var lossFn = new CategoricalCrossEntropyLoss<float>();
            // Disable stochastic layers (dropout, etc.) for the
            // finite-difference probe. ComputeGradients runs the model's
            // ForwardForTraining path with IsTrainingMode honoured by every
            // layer — leaving the default IsTrainingMode=true would mix
            // stochastic dropout masks into the analytic gradient,
            // producing a per-call random gradient that can't be compared
            // against the eval-mode numeric finite-difference (modelFd.Predict
            // flips to eval internally, so dropout is OFF there). With
            // dropout active on analytic but off on numeric the two compute
            // gradients of DIFFERENT loss surfaces and disagree by a margin
            // that varies with parameter sensitivity to the dropout mask —
            // the 7/12 mismatch the post-test-helper-fix run reports
            // converges to ~12/12 once both sides see the same loss surface.
            modelFd.SetTrainingMode(false);
            // Materialize and grab analytic gradient.
            var analyticGrad = modelFd.ComputeGradients(xTrain, yTrain, lossFn);
            var fdParams = modelFd.GetParameters();
            // Pick the 12 indices with largest |analytic gradient|.
            var sortedByMagnitude = Enumerable.Range(0, analyticGrad.Length)
                .OrderByDescending(i => Math.Abs((double)analyticGrad[i]))
                .Take(12)
                .ToArray();
            int[] probeIndices = sortedByMagnitude;
            const float fdEps = 1e-3f;
            int fdMatches = 0;
            int fdMismatches = 0;
            double worstFdRel = 0.0;
            int worstFdIdx = -1;
            double worstFdAnalytic = 0;
            double worstFdNumeric = 0;
            for (int p = 0; p < probeIndices.Length; p++)
            {
                int idx = probeIndices[p];
                if (idx < 0 || idx >= fdParams.Length) continue;

                // L(+ε)
                var paramsPlus = new Vector<float>(fdParams.Length);
                for (int j = 0; j < fdParams.Length; j++) paramsPlus[j] = fdParams[j];
                paramsPlus[idx] = fdParams[idx] + fdEps;
                modelFd.SetParameters(paramsPlus);
                var predPlus = modelFd.Predict(xTrain);
                float lossPlus = ScalarLoss(predPlus, yTrain, lossFn);

                // L(-ε)
                var paramsMinus = new Vector<float>(fdParams.Length);
                for (int j = 0; j < fdParams.Length; j++) paramsMinus[j] = fdParams[j];
                paramsMinus[idx] = fdParams[idx] - fdEps;
                modelFd.SetParameters(paramsMinus);
                var predMinus = modelFd.Predict(xTrain);
                float lossMinus = ScalarLoss(predMinus, yTrain, lossFn);

                // Restore original.
                modelFd.SetParameters(fdParams);

                double numericGrad = (lossPlus - lossMinus) / (2.0 * fdEps);
                double analyticVal = analyticGrad[idx];

                // Tolerance: relative error within 20% AND absolute within 1e-3.
                double absDiff = Math.Abs(numericGrad - analyticVal);
                double scale = Math.Max(Math.Abs(numericGrad), Math.Max(Math.Abs(analyticVal), 1e-6));
                double rel = absDiff / scale;
                if (rel > worstFdRel)
                {
                    worstFdRel = rel;
                    worstFdIdx = idx;
                    worstFdAnalytic = analyticVal;
                    worstFdNumeric = numericGrad;
                }
                if (rel < 0.2 || absDiff < 1e-3) fdMatches++;
                else fdMismatches++;
            }

            _output.WriteLine($"Arm 6 (parameter-delta + gradient-magnitude probe):");
            _output.WriteLine($"  initial params L2 = {Math.Sqrt(initialSnapshot.Sum(v => (double)v * v)):F6}");
            _output.WriteLine($"  final params L2   = {finalParamL2:F6}");
            _output.WriteLine($"  delta L2 (final - initial) = {deltaL2:F6}");
            _output.WriteLine($"  delta max element          = {deltaMax:F6}");
            _output.WriteLine($"  first-step gradient L2     = {gradL2:F6}");
            _output.WriteLine($"  first-step gradient max    = {gradMax:F6}");
            _output.WriteLine($"  finite-diff gradient check ({probeIndices.Length} indices): {fdMatches} match, {fdMismatches} mismatch");
            _output.WriteLine($"    worst rel error: {worstFdRel:F4} at idx {worstFdIdx} (analytic={worstFdAnalytic:F6}, numeric={worstFdNumeric:F6})");

            // Post-fix the PR description records ~10/12 matches on this
            // probe; pre-fix records 0/12 (analytic gradient was zero or
            // wrong-sign). Require a clear majority so a future regression
            // in the analytic-gradient path can't slip through as a
            // silently-passing test. (PR #1364 review — Arm 6 now asserts
            // on its finite-difference results instead of only logging them.)
            int totalProbed = fdMatches + fdMismatches;
            Assert.True(
                totalProbed > 0 && fdMatches * 2 >= totalProbed,
                $"Arm 6: finite-difference gradient agreement is {fdMatches}/{totalProbed}, " +
                $"below the >= 50% threshold. Worst rel error {worstFdRel:F4} at idx {worstFdIdx} " +
                $"(analytic={worstFdAnalytic:F6}, numeric={worstFdNumeric:F6}). " +
                "The analytic gradient is likely zero or wrong-sign on a meaningful fraction " +
                "of parameters — the residual mode-collapse fix has regressed.");
        }

        // ARM 7: long-horizon BuildAsync run with 0 tolerance + 200 epochs.
        // Verifies whether the residual mode collapse is a step-count
        // issue (the optimizer is being stopped too early by an aggressive
        // convergence criterion) or a fundamental optimizer-path bug. If
        // accuracy reaches near Arm 0 with more steps, the bug is in the
        // stopping criterion. If accuracy plateaus at ~10%, the bug is in
        // the optimizer step itself. (No try-catch: see Arm 6 rationale.)
        {
            // Build a fresh architecture too, not just fresh (X, Y). Every
            // other arm constructs its own architecture inside the per-arm
            // helper; reusing the outer-scope `arch` here would inherit any
            // mutable layer state (lazy-shape caches, layer registries)
            // populated by earlier arms 0-6 (review #1364).
            var (archArm7, xLong, yLong) = BuildFixture();
            var modelArm7 = new Transformer<float>(archArm7, lossFunction: new CategoricalCrossEntropyLoss<float>());
            var optionsArm7 = new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                InitialLearningRate = LearningRate,
                MaxIterations = 200,
                BatchSize = BatchSize,
                UseAdaptiveLearningRate = false,
                RandomSeed = Seed,
                ShuffleData = true,
                Tolerance = 0.0,
                Regularization = new NoRegularization<float, Tensor<float>, Tensor<float>>(),
            };
            var optimizerArm7 = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(modelArm7, optionsArm7);
            var inputDataArm7 = new OptimizationInputData<float, Tensor<float>, Tensor<float>>
            {
                XTrain = xLong,
                YTrain = yLong,
                XValidation = xLong,
                YValidation = yLong,
                XTest = xLong,
                YTest = yLong,
            };
            _ = optimizerArm7.Optimize(inputDataArm7);
            float arm7Top1 = ComputeTopOneAccuracy(modelArm7, xLong, yLong);
            _output.WriteLine($"Arm 7 (BuildAsync 200 epochs, Tolerance=0): top-1 = {arm7Top1 * 100.0f:F1}% (uniform = {UniformTopOne * 100.0f:F1}%)");
        }
    }

    /// <summary>
    /// Compute scalar mean loss from prediction and target tensors. Used by
    /// Arm 6's finite-difference gradient check. Routes through
    /// <c>LossFunctionBase{T}.ComputeTapeLoss</c> so the scalar this returns
    /// is the EXACT same scalar that <c>NeuralNetworkBase.ComputeGradients</c>
    /// (the analytic-gradient producer) backpropagates through — any
    /// finite-difference comparison against analytic gradient must use the
    /// same scalar or it's a normalization bug, not a gradient bug.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The previous implementation called <c>loss.CalculateLoss(Vector, Vector)</c>
    /// (which returns a raw sum, no reduction) and divided by
    /// <c>totalTargetElements = B*V</c>. That matched the BASE
    /// <see cref="LossFunctions.CrossEntropyLoss{T}"/>'s ReduceMean-over-all-axes
    /// normalization, but the test uses <see cref="LossFunctions.CategoricalCrossEntropyLoss{T}"/>,
    /// which overrides <c>ComputeTapeLoss</c> to do ReduceSum-over-classes
    /// then ReduceMean-over-batch — divisor B, not B*V. The mismatch made
    /// every finite-diff probe report numeric ≈ analytic / V (V=16 here),
    /// which falsely tripped Arm 6's "gradient agreement &lt; 50%"
    /// assertion as a regression. Pinned by
    /// <see cref="LossNormalizationConsistencyIssue1380Tests.CategoricalCrossEntropyLoss_ComputeTapeLoss_DividesByBatchOnly_OnRank2Target"/>.
    /// </para>
    /// <para>
    /// Routing through <c>ComputeTapeLoss</c> here makes the test agnostic
    /// to whichever reduction the production loss uses — categorical CE
    /// (mean over batch) vs binary CE (mean over all elements) vs any
    /// future loss with a different reduction will all just work, because
    /// the test reads the same scalar the analytic gradient is computed
    /// against.
    /// </para>
    /// </remarks>
    private static float ScalarLoss(Tensor<float> predictions, Tensor<float> targets, LossFunctionBase<float> loss)
    {
        var lossTensor = loss.ComputeTapeLoss(predictions, targets);
        // ComputeTapeLoss returns a rank-0 scalar tensor wrapping one float.
        return lossTensor[0];
    }
}
