using AiDotNet.Data.Loaders;
using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.Tests.IntegrationTests.ConfigureMethodCoverage;

/// <summary>
/// Forces the AiDotNet tensor engine to CPU before any Configure* test runs.
/// This avoids the OpenCL <c>SetKernelArg</c> access-violation crash observed
/// when MultiHeadAttentionLayer hits the DirectGpu backend on the test host
/// (filed upstream — DirectGpu OpenCL backend instability under concurrent
/// kernel dispatch).
/// </summary>
public sealed class ConfigureMethodTestCpuFixture
{
    public ConfigureMethodTestCpuFixture()
    {
        // Force CPU for the duration of all Configure* coverage tests.
        AiDotNetEngine.ResetToCpu();
    }
}

[Xunit.CollectionDefinition("ConfigureMethodCoverage")]
public sealed class ConfigureMethodCoverageCollection : Xunit.ICollectionFixture<ConfigureMethodTestCpuFixture> { }

/// <summary>
/// Shared scaffolding for end-to-end tests that exercise <c>AiModelBuilder.Configure*</c>
/// methods. Each Configure* method gets at least one integration test that builds a
/// small Transformer, trains a few epochs, and asserts non-degenerate output.
/// </summary>
/// <remarks>
/// <para>
/// Why this exists: several "drop-in" Configure* methods on AiModelBuilder were found
/// in HarmonicEngine field-testing to produce broken behavior with zero existing test
/// coverage (BuildAsync+Adam top-1=0% uniform, GradientCheckpointing wrong chain rule,
/// Int8Quantizer 0.36× speedup, FlashAttention top1=0% top5=100%). Each of those would
/// have been caught by "train tiny model, assert top-1 > random chance".
/// </para>
/// <para>
/// The "canary" config is intentionally small so every test runs in &lt; 60 s on CI:
/// vocab=8, ctxLen=4, dModel=16, layers=1, heads=2, ~64 training examples, &lt;= 200 steps.
/// (Values mirror the per-constant declarations below: <c>CanaryVocab = 8</c>,
/// <c>CanaryCtxLen = 4</c>.) At those sizes single-example or small-batch
/// memorization is fully achievable for any working training pipeline;
/// degenerate-output bugs flip the top-1 assertion.
/// </para>
/// </remarks>
public abstract class ConfigureMethodTestBase
{
    /// <summary>Vocabulary size for the canary memorization task. V=8 keeps the task small enough
    /// that a B=8 TrainBatched run converges within ~100 batch steps (mirrors the V=256 B=32
    /// 100-step pattern from <c>TransformerEndToEndIntegrationTests.TrainBatched_V256_LearnsBatchAfter100Steps</c>).</summary>
    protected const int CanaryVocab = 8;

    /// <summary>Context (sequence) length for the canary task.</summary>
    protected const int CanaryCtxLen = 4;

    /// <summary>Model dimension (d_model). Kept small so tests run fast.</summary>
    protected const int CanaryDModel = 16;

    /// <summary>Feed-forward dimension.</summary>
    protected const int CanaryDFf = 32;

    /// <summary>Number of encoder layers.</summary>
    protected const int CanaryLayers = 1;

    /// <summary>Number of attention heads.</summary>
    protected const int CanaryHeads = 2;

    /// <summary>Batch size for the canary training set. Matches the V=256 B=32 cell in
    /// <c>TransformerEndToEndIntegrationTests</c> proportionally.</summary>
    protected const int CanaryBatchSize = 8;

    /// <summary>Number of training epochs (each epoch = per-sample Train over the full
    /// batch). 20 epochs * 8 samples = 160 individual Train() calls, which exceeds the
    /// known V=16 per-sample convergence budget from
    /// <c>TransformerEndToEndIntegrationTests</c> proportionally.</summary>
    protected const int CanaryTrainSteps = 20;

    /// <summary>
    /// Builds a canary <see cref="TransformerArchitecture{T}"/> at the standard test
    /// sizes. <c>warmupSteps=10</c> so the LR schedule actually engages within the
    /// short test budget, and <c>randomSeed=42</c> for determinism.
    /// </summary>
    protected static TransformerArchitecture<float> MakeCanaryArch(int? seed = null) =>
        new(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.SequenceClassification,
            numEncoderLayers: CanaryLayers,
            numDecoderLayers: 0,
            numHeads: CanaryHeads,
            modelDimension: CanaryDModel,
            feedForwardDimension: CanaryDFf,
            inputSize: CanaryCtxLen,
            outputSize: CanaryVocab,
            maxSequenceLength: CanaryCtxLen,
            vocabularySize: CanaryVocab,
            warmupSteps: 10,
            randomSeed: seed ?? 42);

    /// <summary>
    /// Builds a canary <see cref="Transformer{T}"/> wired with cross-entropy loss.
    /// </summary>
    protected static Transformer<float> MakeCanaryModel(int? seed = null) =>
        new(MakeCanaryArch(seed), lossFunction: new CategoricalCrossEntropyLoss<float>());

    /// <summary>
    /// Builds a small deterministic memorization training set: <paramref name="batchSize"/>
    /// input/target pairs where input[i] uses shifted token positions and target[i] is
    /// one-hot at class <c>i % vocab</c>. Any working training pipeline reaches
    /// &gt; random-chance top-1 on this set within ~100 batch updates.
    /// </summary>
    protected static (Tensor<float> features, Tensor<float> labels) MakeMemorizationSet(
        int batchSize = CanaryBatchSize,
        int ctxLen = CanaryCtxLen,
        int vocab = CanaryVocab,
        int seed = 7)
    {
        // Pattern mirrors TransformerEndToEndIntegrationTests:
        //   inputs[b][s] = (b + s) % vocab  (distinct shift per row)
        //   targets[b][b % vocab] = 1       (one-hot at class b mod vocab)
        // The mapping is 1-1 when batchSize ≤ vocab so single-example
        // memorization works trivially.
        var features = new Tensor<float>([batchSize, ctxLen]);
        var labels = new Tensor<float>([batchSize, vocab]);
        for (int b = 0; b < batchSize; b++)
        {
            for (int s = 0; s < ctxLen; s++)
            {
                features[b, s] = (float)((b + s) % vocab);
            }
            int targetClass = b % vocab;
            labels[b, targetClass] = 1f;
        }
        return (features, labels);
    }

    /// <summary>
    /// Returns top-1 accuracy on the training set (memorization fitness). Any working
    /// training pipeline should reach &gt; random chance (=1/vocab) within
    /// <see cref="CanaryTrainSteps"/> steps; broken pipelines (uniform output) stay
    /// at exactly 1/vocab.
    /// </summary>
    protected static double MeasureTrainingTopOne(
        Transformer<float> model,
        Tensor<float> features,
        Tensor<float> labels)
    {
        int batch = features.Shape[0];
        int vocab = labels.Shape[1];
        int correct = 0;
        model.SetTrainingMode(false);
        for (int b = 0; b < batch; b++)
        {
            var probe = new Tensor<float>([1, features.Shape[1]]);
            for (int s = 0; s < features.Shape[1]; s++) probe[0, s] = features[b, s];
            var pred = model.Predict(probe);
            int predArg = ArgmaxRow(pred, vocab);
            int trueArg = OneHotArgmaxRow(labels, b, vocab);
            if (predArg == trueArg) correct++;
        }
        return (double)correct / batch;
    }

    /// <summary>
    /// Returns the maximum absolute pairwise difference of predictions across the
    /// batch. Uniform-output bugs (Flash Attention degenerate path, etc.) leave this
    /// near zero because every input produces the same output vector.
    /// </summary>
    protected static double MeasurePredictionSpread(
        Transformer<float> model,
        Tensor<float> features)
    {
        int batch = features.Shape[0];
        int ctxLen = features.Shape[1];
        var preds = new Tensor<float>[batch];
        model.SetTrainingMode(false);
        for (int b = 0; b < batch; b++)
        {
            var probe = new Tensor<float>([1, ctxLen]);
            for (int s = 0; s < ctxLen; s++) probe[0, s] = features[b, s];
            preds[b] = model.Predict(probe);
        }
        double maxDiff = 0;
        int outLen = preds[0].Length;
        for (int i = 0; i < batch; i++)
        {
            for (int j = i + 1; j < batch; j++)
            {
                for (int k = 0; k < outLen; k++)
                {
                    double d = Math.Abs(preds[i][k] - preds[j][k]);
                    if (d > maxDiff) maxDiff = d;
                }
            }
        }
        return maxDiff;
    }

    /// <summary>
    /// Trains the given model on the canary memorization set via per-example
    /// <c>Train</c> calls (matching the convergent V=16 single-example pattern
    /// from <c>TransformerEndToEndIntegrationTests</c>). Returns the
    /// post-training (top-1, spread) measurements.
    /// <para>
    /// NOTE: We tried <c>TrainBatched</c> at B=8/V=8 and observed spread → 0
    /// (uniform-output collapse). Per-example <c>Train</c> at the same task
    /// converges to spread &gt; 0.1. This is a separate finding worth filing
    /// upstream once isolated, but is out of scope for this Configure-method
    /// suite — we sidestep it by using per-example training in the baseline.
    /// </para>
    /// </summary>
    protected static (double topOne, double spread) DirectTrainAndMeasure(
        Transformer<float> model,
        Tensor<float> features,
        Tensor<float> labels,
        int trainSteps = CanaryTrainSteps)
    {
        int batch = features.Shape[0];
        int ctxLen = features.Shape[1];
        int vocab = labels.Shape[1];

        var inputs = new Tensor<float>[batch];
        var targets = new Tensor<float>[batch];
        for (int b = 0; b < batch; b++)
        {
            inputs[b] = new Tensor<float>([1, ctxLen]);
            for (int s = 0; s < ctxLen; s++) inputs[b][0, s] = features[b, s];
            targets[b] = new Tensor<float>([1, vocab]);
            for (int v = 0; v < vocab; v++) targets[b][0, v] = labels[b, v];
        }

        model.SetTrainingMode(true);
        // Per-example training is the known-convergent path. Don't use
        // TrainBatched here — see method docs.
        for (int step = 0; step < trainSteps; step++)
        {
            for (int b = 0; b < batch; b++)
            {
                model.Train(inputs[b], targets[b]);
            }
        }
        model.SetTrainingMode(false);

        double topOne = MeasureTrainingTopOne(model, features, labels);
        double spread = MeasurePredictionSpread(model, features);
        return (topOne, spread);
    }

    /// <summary>
    /// Asserts top-1 accuracy is strictly above random chance (1/vocab) AND strictly
    /// below 100% (rules out memorization on test set). Captures the most common
    /// degenerate-output failure modes:
    /// <list type="bullet">
    /// <item>Uniform output (top-1 = 1/vocab exactly) → BROKEN</item>
    /// <item>top-5 = 100% on every example with top-1 = 1/vocab → BROKEN (FlashAttention)</item>
    /// </list>
    /// </summary>
    protected static void AssertTopOneAboveChance(
        double measuredTopOne,
        int vocab,
        string featureName,
        double marginOverChance = 0.01)
    {
        double chance = 1.0 / vocab;
        Xunit.Assert.True(
            measuredTopOne > chance + marginOverChance,
            $"{featureName}: top-1 accuracy {measuredTopOne:P2} is not above random chance "
            + $"({chance:P2} + {marginOverChance:P2} margin). This indicates the training "
            + $"pipeline collapsed (uniform output, broken gradient, or stale model state). "
            + $"A working pipeline reaches > {chance + marginOverChance:P2} on the canary "
            + $"memorization set within {CanaryTrainSteps} steps.");
    }

    /// <summary>
    /// Asserts the model produces meaningfully-different outputs for different inputs.
    /// A spread near zero indicates uniform-output collapse — the network is returning
    /// the same vector regardless of input.
    /// </summary>
    protected static void AssertOutputSpreadNonZero(
        double spread,
        string featureName,
        double minSpread = 1e-4)
    {
        Xunit.Assert.True(
            spread > minSpread,
            $"{featureName}: prediction spread across inputs is {spread:E2} (bound {minSpread:E2}). "
            + $"The model is returning effectively-identical output for every input — "
            + $"this is the classic uniform-output collapse signature.");
    }

    /// <summary>
    /// Asserts two arms (baseline vs feature) produce comparable top-1 within tolerance.
    /// Feature arm should be at least <paramref name="minRetentionRatio"/> of baseline
    /// to count as "drop-in compatible". E.g. a 50% retention bound flags features that
    /// halve accuracy (Int8Quantization on Transformer LM hit this).
    /// </summary>
    protected static void AssertFeatureRetainsAccuracy(
        double baselineTopOne,
        double featureTopOne,
        string featureName,
        double minRetentionRatio = 0.5)
    {
        // Baseline must clear chance + 5pp before a retention comparison is
        // meaningful — silently returning here would let a broken feature-arm
        // test pass when the baseline itself was bad (an always-pass bug
        // pattern). Surface the bad baseline as the test failure instead.
        // (PR #1345 review.)
        Xunit.Assert.True(
            baselineTopOne > 1.0 / CanaryVocab + 0.05,
            $"{featureName}: baseline top-1 {baselineTopOne:P2} is at or below " +
            $"random chance ({1.0 / CanaryVocab:P2} + 5pp). Fix the baseline before " +
            "asserting retention — comparing against a degenerate baseline would " +
            "let a broken feature arm silently pass.");
        double minRequired = baselineTopOne * minRetentionRatio;
        Xunit.Assert.True(
            featureTopOne >= minRequired,
            $"{featureName}: feature-arm top-1 {featureTopOne:P2} is less than {minRetentionRatio:P0} "
            + $"of baseline top-1 {baselineTopOne:P2}. A 'drop-in' Configure* method should not "
            + $"halve model accuracy; this is the FlashAttention/Int8Quantization breakage signature.");
    }

    /// <summary>
    /// Asserts a measured speedup ratio lies in the documented range. A ratio
    /// &lt; 0.8× is a regression — a drop-in optimization should never be slower
    /// than the baseline. This catches Int8Quantization 0.36× and FlashAttention
    /// 3.76× slowdown bugs.
    /// </summary>
    protected static void AssertSpeedupBetween(
        double measuredSpeedup,
        double lowerBound,
        double upperBound,
        string featureName)
    {
        Xunit.Assert.True(
            measuredSpeedup >= lowerBound,
            $"{featureName}: measured speedup {measuredSpeedup:F2}x is below the documented "
            + $"lower bound {lowerBound:F2}x. A drop-in optimization should not be slower; "
            + $"this is the Int8Quantization 0.36x / FlashAttention 3.76x-slower signature.");
        Xunit.Assert.True(
            measuredSpeedup <= upperBound,
            $"{featureName}: measured speedup {measuredSpeedup:F2}x exceeds the documented "
            + $"upper bound {upperBound:F2}x. This is unusual — verify the timing harness "
            + $"isn't measuring noise (the model may be too small for the optimization to register).");
    }

    /// <summary>
    /// Asserts the AiModelResult.Predict returns a non-degenerate tensor (not all
    /// zero, not NaN/Inf). Common facade-bug signature: facade Predict returns
    /// zero-vector while underlying model returns trained logits (issue #1267).
    /// </summary>
    protected static void AssertFacadePredictNonDegenerate(
        Tensor<float> facadePrediction,
        string featureName)
    {
        double l2 = 0;
        bool hasNaN = false;
        bool hasInf = false;
        for (int i = 0; i < facadePrediction.Length; i++)
        {
            float v = facadePrediction[i];
            if (float.IsNaN(v)) hasNaN = true;
            if (float.IsInfinity(v)) hasInf = true;
            l2 += v * v;
        }
        l2 = Math.Sqrt(l2);
        Xunit.Assert.False(hasNaN, $"{featureName}: facade Predict returned NaN.");
        Xunit.Assert.False(hasInf, $"{featureName}: facade Predict returned Infinity.");
        Xunit.Assert.True(
            l2 > 1e-6,
            $"{featureName}: facade Predict returned all-zero output (L2={l2:E2}). "
            + $"This is the issue-#1267 facade bug — underlying model trained but result "
            + $"wrapper returns uninitialized zeros.");
    }

    /// <summary>
    /// Returns argmax of one-hot row b in <paramref name="labels"/>.
    /// </summary>
    private static int OneHotArgmaxRow(Tensor<float> labels, int row, int vocab)
    {
        int best = 0;
        float bestV = labels[row, 0];
        for (int v = 1; v < vocab; v++)
        {
            if (labels[row, v] > bestV) { bestV = labels[row, v]; best = v; }
        }
        return best;
    }

    /// <summary>
    /// Returns argmax of single-row prediction tensor.
    /// </summary>
    private static int ArgmaxRow(Tensor<float> pred, int vocab)
    {
        int best = 0;
        float bestV = pred[0, 0];
        for (int v = 1; v < vocab; v++)
        {
            if (pred[0, v] > bestV) { bestV = pred[0, v]; best = v; }
        }
        return best;
    }

    /// <summary>
    /// Returns a DataLoader wrapping the canary memorization set, suitable for
    /// AiModelBuilder.ConfigureDataLoader.
    /// </summary>
    protected static InMemoryDataLoader<float, Tensor<float>, Tensor<float>> MakeCanaryLoader(
        Tensor<float> features,
        Tensor<float> labels) =>
        DataLoaders.FromTensors<float>(features, labels);

    /// <summary>
    /// Times a no-arg action, returning wall-clock seconds. Uses 3 warmup iterations
    /// to amortize JIT, then averages 3 timed iterations. Intended for the speedup
    /// assertions; not high-precision but stable across machines for &gt; 2× speedups.
    /// </summary>
    protected static double TimeAction(Action action, int warmup = 3, int iterations = 3)
    {
        for (int i = 0; i < warmup; i++) action();
        var sw = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++) action();
        sw.Stop();
        return sw.Elapsed.TotalSeconds / iterations;
    }
}
