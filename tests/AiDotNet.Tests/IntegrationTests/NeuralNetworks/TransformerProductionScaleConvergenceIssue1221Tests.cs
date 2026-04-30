using System.Linq;
using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Regression suite for issue #1221 — Transformer training reports correct
/// loss magnitude but produces uniform output at production scale (top-1=0%,
/// PPL=|V|=eval NLL exactly ln(V)). Issue #1208 verified the V=8 / Adam LR=0.01
/// reproducer works on master, but #1221 reports the SAME failure mode at:
///
///   - V=8 / Adam LR=0.001 / 500 iters (the #1208 V3 variant the original
///     fix did not test directly — only V=8 / Adam LR=0.01 / 200 iters made
///     it into <see cref="TransformerEmbeddingGradientFlowIssue1208Tests"/>)
///
///   - V=256 / d=64 / L=2 / 1MB Shakespeare byte-LM (production-grade — the
///     reported smoking gun: top-1 = 0.00% over 500 eval positions, eval
///     NLL = ln(256) exactly, PPL = 256.00)
///
/// Both configs ran the existing fix path (RestoreOriginalParameters' "skip
/// swap-back when stable" optimisation + Tensors 0.58.2's tape-aware
/// TensorEmbeddingLookup) and both still failed. The hypothesis from #1221
/// is that something in the gradient flow OR in the eval-time forward path
/// regresses at production scale. These tests bring the repro configs into
/// the test suite so future investigations have a binary signal.
///
/// <para>
/// <b>Test design:</b> Each test asserts the exact "uniform output" failure
/// mode from the issue — pairwise L2 distance over per-input logit vectors.
/// Pre-fix every pair has distance exactly 0 (identical outputs); post-fix
/// healthy runs see distances in the 0.1–1.0 range. The signal is robust to
/// argmax flicker and weight-init seed luck because we measure logit
/// dispersion directly rather than counting argmax matches.
/// </para>
///
/// <para>
/// <b>Tractability:</b> The full 1MB / 519s / 188k-step config from #1221
/// would dominate CI wall time. We scale down to 64 distinct sequences
/// × 30 epochs (~2k Adam steps) — large enough to expose the
/// "gradient signal not flowing at scale" failure mode reported in the issue
/// while staying under a 60-second test budget. The bug, if present,
/// manifests as ZERO logit dispersion (PPL = exactly |V|) regardless of
/// iteration count, so a 60s test catches it just as cleanly as a 519s test.
/// </para>
/// </summary>
public class TransformerProductionScaleConvergenceIssue1221Tests
{
    private readonly ITestOutputHelper _output;

    public TransformerProductionScaleConvergenceIssue1221Tests(ITestOutputHelper output)
    {
        _output = output;
    }

    private static Transformer<float> BuildTransformer(
        int vocab,
        int modelDim,
        int feedForwardDim,
        int seqLen,
        int numEncoderLayers,
        int numHeads,
        double learningRate)
    {
        var arch = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.SequenceClassification,
            numEncoderLayers: numEncoderLayers,
            numDecoderLayers: 0,
            numHeads: numHeads,
            modelDimension: modelDim,
            feedForwardDimension: feedForwardDim,
            inputSize: seqLen,
            outputSize: vocab,
            maxSequenceLength: seqLen,
            vocabularySize: vocab);

        var optOptions = new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
        {
            InitialLearningRate = learningRate,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8
        };
        var optimizer = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null, optOptions);

        return new Transformer<float>(
            arch,
            lossFunction: new CategoricalCrossEntropyLoss<float>(),
            optimizer: optimizer);
    }

    private static Tensor<float> BuildIdentityInput(int classIndex, int seqLen)
    {
        var t = new Tensor<float>([1, seqLen]);
        for (int s = 0; s < seqLen; s++) t[0, s] = classIndex;
        return t;
    }

    private static Tensor<float> BuildOneHotTarget(int classIndex, int vocab)
    {
        var t = new Tensor<float>([1, vocab]);
        t[0, classIndex] = 1f;
        return t;
    }

    private static double L2Distance(float[] a, float[] b)
    {
        double s = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double d = a[i] - b[i];
            s += d * d;
        }
        return Math.Sqrt(s);
    }

    /// <summary>
    /// Mirrors the #1208 V3 variant that the original fix never tested directly:
    /// Adam LR=0.001 (vs the 0.01 used in TransformerEmbeddingGradientFlowIssue1208Tests).
    /// User reports this config still fails on AiDotNet v0.166.0 / Tensors 0.59.0
    /// with eval acc = 12% = 1/V random and uniform-output logits across all 8
    /// distinct token ids.
    ///
    /// If this test fails, the LR=0.01 path that #1208 verified is masking a
    /// gradient-flow bug that surfaces only at lower learning rates — meaning
    /// the existing fix is incomplete. If this test passes, the V=8 part of
    /// #1221's repro is config-specific to the user's environment and the
    /// remaining concern is the production-scale (V=256) test below.
    /// </summary>
    [Fact]
    public async Task Transformer_V8_Adam_LR0001_500iters_DifferentiatesInputs()
    {
        await Task.Yield();
        // Match #1208 V3 variant exactly: Adam LR=0.001, 500 iters, V=8.
        var model = BuildTransformer(
            vocab: 8, modelDim: 16, feedForwardDim: 32, seqLen: 4,
            numEncoderLayers: 1, numHeads: 2, learningRate: 0.001);
        model.SetTrainingMode(true);

        const int totalIters = 500;
        for (int iter = 0; iter < totalIters; iter++)
        {
            int k = iter % 8;
            model.Train(BuildIdentityInput(k, seqLen: 4), BuildOneHotTarget(k, vocab: 8));
        }

        model.SetTrainingMode(false);
        var logits = new float[8][];
        for (int k = 0; k < 8; k++)
        {
            var pred = model.Predict(BuildIdentityInput(k, seqLen: 4));
            logits[k] = new float[pred.Length];
            for (int j = 0; j < pred.Length; j++) logits[k][j] = pred[j];
        }

        double maxPairwise = 0.0;
        for (int i = 0; i < 8; i++)
        {
            for (int j = i + 1; j < 8; j++)
            {
                double d = L2Distance(logits[i], logits[j]);
                if (d > maxPairwise) maxPairwise = d;
            }
        }

        _output.WriteLine($"V=8 / Adam LR=0.001 / 500 iters: max pairwise L2 = {maxPairwise:E3}");

        Assert.True(maxPairwise > 5e-4,
            $"Transformer produces identical logits across all 8 distinct " +
            $"inputs at V=8 / Adam LR=0.001 / 500 iters (issue #1221, " +
            $"matching #1208 V3 variant): max pairwise L2 = {maxPairwise:E3}. " +
            $"Pre-fix this is exactly 0 (uniform output regardless of input). " +
            $"The lower LR (0.001 vs the 0.01 in #1208 tests) keeps the loss " +
            $"trajectory near the random-uniform-output basin, surfacing any " +
            $"weak/missing gradient signal that LR=0.01 would mask via " +
            $"larger update steps.");
    }

    /// <summary>
    /// Production-scale repro from #1221: V=256 / d=64 / 2 encoder layers /
    /// 4 heads / FF=256 — same architecture as the 1MB Shakespeare byte-LM
    /// failing case. Scaled-down sample count (64 distinct sequences instead
    /// of 62k) so the test stays under CI wall budget; the failure mode is
    /// "uniform output regardless of iteration count" so this still catches
    /// the bug.
    ///
    /// User reports eval NLL = ln(256) = 5.5452 exactly after 519s training,
    /// meaning the model emits a uniform distribution across all 256 vocab
    /// classes for every input. Pre-fix logit dispersion is exactly 0 across
    /// all input pairs; post-fix healthy runs see dispersion proportional to
    /// embedding gradient magnitude × Adam step count.
    /// </summary>
    [Fact(Timeout = 90000)]
    public async Task Transformer_V256_d64_L2_DifferentiatesInputs_ProductionConfig()
    {
        await Task.Yield();
        // Match #1221 Cycle30Repro: V=256, d=64, ff=256, heads=4, layers=2,
        // ctx=32. Adam LR=0.0003 (issue's training config).
        const int vocab = 256;
        const int seqLen = 32;
        var model = BuildTransformer(
            vocab: vocab, modelDim: 64, feedForwardDim: 256, seqLen: seqLen,
            numEncoderLayers: 2, numHeads: 4, learningRate: 0.0003);
        model.SetTrainingMode(true);

        // 64 distinct inputs (classes 0..63 from the 256-vocab) — enough
        // diversity to expose a "frozen network" failure mode without
        // requiring the full 62k-sample 1MB corpus.
        const int numClasses = 64;
        const int epochs = 30;
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int k = 0; k < numClasses; k++)
            {
                model.Train(
                    BuildIdentityInput(k, seqLen: seqLen),
                    BuildOneHotTarget(k, vocab: vocab));
            }
        }

        // Eval on all 64 trained classes — read full logit vectors and
        // measure pairwise dispersion. Pre-fix every pair is exactly 0
        // (uniform output for every input regardless of class).
        model.SetTrainingMode(false);
        var logits = new float[numClasses][];
        for (int k = 0; k < numClasses; k++)
        {
            var pred = model.Predict(BuildIdentityInput(k, seqLen: seqLen));
            logits[k] = new float[pred.Length];
            for (int j = 0; j < pred.Length; j++) logits[k][j] = pred[j];
        }

        // Compute max pairwise L2 across the 64 trained inputs (2016 pairs).
        // We track max rather than mean so single-input divergence still
        // counts as "the gradient is flowing" — this is a binary "frozen
        // network or not" probe, not an "is training fully converged" probe.
        double maxPairwise = 0.0;
        int pairsWithDispersion = 0;
        for (int i = 0; i < numClasses; i++)
        {
            for (int j = i + 1; j < numClasses; j++)
            {
                double d = L2Distance(logits[i], logits[j]);
                if (d > maxPairwise) maxPairwise = d;
                if (d > 1e-3) pairsWithDispersion++;
            }
        }

        _output.WriteLine($"V=256 / d=64 / L=2 / 30 epochs × 64 classes:");
        _output.WriteLine($"  max pairwise L2 = {maxPairwise:E3}");
        _output.WriteLine($"  pairs with L2 > 1e-3: {pairsWithDispersion}/{numClasses * (numClasses - 1) / 2}");

        Assert.True(maxPairwise > 5e-4,
            $"Transformer produces identical logits across all {numClasses} " +
            $"distinct inputs at production-grade config V=256 / d=64 / L=2 " +
            $"(issue #1221): max pairwise L2 = {maxPairwise:E3}. Pre-fix this " +
            $"is exactly 0 (eval NLL = ln(V) = {Math.Log(vocab):F4}, PPL = " +
            $"{vocab} = uniform output). The smoking gun reported in the issue " +
            $"is precisely this: training completes successfully but the model " +
            $"emits a uniform distribution regardless of input.");

        // Stronger signal: at least 10% of pairs must show non-trivial
        // dispersion. A single divergent pair could be float-noise; broad
        // dispersion across the input space proves the encoder/attention/FFN
        // chain is differentiating inputs.
        int totalPairs = numClasses * (numClasses - 1) / 2;
        int requiredPairs = totalPairs / 10;
        Assert.True(pairsWithDispersion >= requiredPairs,
            $"Only {pairsWithDispersion}/{totalPairs} input pairs show " +
            $"meaningful logit dispersion (> 1e-3) at production-grade " +
            $"config (issue #1221). Required: {requiredPairs}. " +
            $"With working gradient flow at scale, the encoder must " +
            $"differentiate distinct token ids across a non-trivial fraction " +
            $"of the input space. Pre-fix this is 0/N (every pair " +
            $"identical).");
    }

    /// <summary>
    /// Direct probe of the embedding table — at production scale (V=256),
    /// after training the embedding rows for trained classes must show
    /// non-zero deltas from initialisation. This is the most narrow possible
    /// signal that the bug-2 fix from #1208 (Tensors 0.58.2 tape-aware
    /// TensorEmbeddingLookup) is still working at V=256.
    ///
    /// Pre-fix the embedding table received zero gradient, so per-row
    /// delta magnitudes were bit-identical to initialisation noise. Post-fix
    /// (and assuming the bug is fixed) Adam-shaped updates accumulate on
    /// every row touched by the supervision signal.
    /// </summary>
    [Fact(Timeout = 60000)]
    public async Task EmbeddingTable_V256_ReceivesNonZeroGradient_AtProductionScale()
    {
        await Task.Yield();
        const int vocab = 256;
        const int seqLen = 32;
        var model = BuildTransformer(
            vocab: vocab, modelDim: 64, feedForwardDim: 256, seqLen: seqLen,
            numEncoderLayers: 2, numHeads: 4, learningRate: 0.0003);

        var embedding = model.Layers.OfType<EmbeddingLayer<float>>().FirstOrDefault()
            ?? throw new InvalidOperationException(
                "Transformer was expected to construct an EmbeddingLayer<float> as its token-input layer.");

        model.SetTrainingMode(true);

        // Warm up: one Train call materializes the lazy [V, D] embedding tensor.
        model.Train(BuildIdentityInput(0, seqLen: seqLen), BuildOneHotTarget(0, vocab: vocab));

        var beforeParams = embedding.GetTrainableParameters();
        var embBefore = beforeParams[0];
        Assert.True(embBefore.Length > 0,
            $"Embedding tensor must be materialised after warm-up; got Length={embBefore.Length}.");
        var snapshot = new float[embBefore.Length];
        for (int i = 0; i < embBefore.Length; i++) snapshot[i] = embBefore[i];

        // Run several training steps across diverse classes — touching at
        // least 32 distinct embedding rows so the test isn't sensitive to a
        // single row's update path being broken in isolation.
        const int steps = 64;
        for (int i = 0; i < steps; i++)
        {
            int k = i % 32;
            model.Train(BuildIdentityInput(k, seqLen: seqLen), BuildOneHotTarget(k, vocab: vocab));
        }

        var afterParams = embedding.GetTrainableParameters();
        var embAfter = afterParams[0];
        Assert.Equal(snapshot.Length, embAfter.Length);

        int movedEntries = 0;
        double maxDelta = 0.0;
        for (int i = 0; i < embAfter.Length; i++)
        {
            float delta = embAfter[i] - snapshot[i];
            double absDelta = Math.Abs(delta);
            if (absDelta > maxDelta) maxDelta = absDelta;
            if (absDelta > 1e-5) movedEntries++;
        }

        _output.WriteLine($"V=256 embedding moved entries: {movedEntries}/{embAfter.Length}");
        _output.WriteLine($"V=256 embedding max abs delta: {maxDelta:E3}");

        // At V=256 / d=64 the table is 16,384 entries. Touching 32 distinct
        // rows over 64 steps (each row × d=64 = 64 entries per visited row)
        // means at most ~2,048 entries can move via row-gather; we require
        // at least 25% of THAT to move (~512) to have a strong signal.
        // Pre-fix this is 0; post-fix healthy runs see all touched rows
        // moving substantially.
        const int requiredMoved = 512;
        Assert.True(movedEntries >= requiredMoved,
            $"Embedding table at V=256 received insufficient gradient updates " +
            $"(issue #1221): only {movedEntries}/{embAfter.Length} entries " +
            $"moved by > 1e-5 after {steps} Adam steps across 32 distinct " +
            $"classes. Required: {requiredMoved}. Max delta = {maxDelta:E3}. " +
            $"Pre-fix this is bit-identical (zero movement at production " +
            $"scale despite the V=8 case working) — the user-reported " +
            $"smoking gun for #1221.");
    }
}
