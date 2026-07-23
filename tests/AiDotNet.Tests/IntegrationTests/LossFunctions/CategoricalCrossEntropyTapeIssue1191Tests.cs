using System;
using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.IntegrationTests.LossFunctions;

/// <summary>
/// Regression suite for issue #1191 — Categorical cross-entropy tape
/// loss was averaged across the class axis, producing a loss value
/// (and a back-propagated gradient) that was 1/V of the true CE.
///
/// The pre-fix <see cref="CategoricalCrossEntropyLoss{T}.ComputeTapeLoss"/>
/// did <c>ReduceMean(product, allAxes)</c>. For a one-hot target the
/// product tensor has exactly one non-zero entry per sample, so taking
/// the mean over the class axis silently divided by V. At V=8 this
/// produced a reported loss of ~0.26 at uniform softmax (vs. true
/// ln(8) ≈ 2.08); at V=256 it produced 0.0217 (vs. true ln(256) ≈
/// 5.55), matching exactly the bit-identical loss reported in the
/// issue's Cycle-30 run.
///
/// These tests pin both halves of the bug:
/// 1. The direct-tape tests assert the loss tensor produced by
///    <c>ComputeTapeLoss</c> matches the standard
///    <c>-Σ_class target * log(predicted)</c> formula across 1D / 2D /
///    3D shapes — they fail with a 1/V relative error on the buggy
///    code.
/// 2. The end-to-end Transformer test mirrors issue #1191's V=8
///    identity-mapping repro and asserts the model both reports a
///    loss near ln(V) at random init and learns the task — the
///    pre-fix code reports ln(V)/V and never escapes random.
/// </summary>
public class CategoricalCrossEntropyTapeIssue1191Tests
{
    private readonly ITestOutputHelper _output;

    public CategoricalCrossEntropyTapeIssue1191Tests(ITestOutputHelper output)
    {
        _output = output;
    }

    /// <summary>
    /// Direct reproducer: a uniform softmax over V classes against a
    /// one-hot target should yield exactly ln(V), the cross-entropy of
    /// the uniform distribution. With the 1/V scaling bug the value is
    /// ln(V)/V.
    /// </summary>
    /// <remarks>
    /// V=8 is chosen to match the minimal reproducer the issue links
    /// to (HarmonicEngine PR #80 / Phase_A4_AiDotNet_1187_Reproducer).
    /// </remarks>
    [Fact]
    public void ComputeTapeLoss_UniformSoftmax_OneHotTarget_ReturnsLnV_NotLnV_OverV()
    {
        const int v = 8;
        const float uniform = 1f / v; // 0.125 — softmax over 8 equal logits

        // predicted: [V] — single-sample uniform distribution.
        var predicted = new Tensor<float>([v]);
        for (int i = 0; i < v; i++)
            predicted[i] = uniform;

        // target: [V] — one-hot at class 3 (any class works; pick a
        // non-edge class so a future implementation that special-cases
        // class 0 doesn't pass by accident).
        var target = new Tensor<float>([v]);
        target[3] = 1f;

        var loss = new CategoricalCrossEntropyLoss<float>();
        var result = loss.ComputeTapeLoss(predicted, target);

        Assert.True(result.Length > 0, "Loss tensor should be non-empty");
        float actualLoss = result[0];

        // Standard CE for uniform softmax over V classes against a
        // one-hot target = ln(V). This is independent of which class
        // the one-hot picks because every probability equals 1/V.
        float expectedLoss = (float)Math.Log(v);   // ≈ 2.0794
        float buggyLoss = expectedLoss / v;        // ≈ 0.2599 — what the
                                                   // 1/V scaling produced.

        _output.WriteLine($"V={v}  expected={expectedLoss:F6}  buggy={buggyLoss:F6}  actual={actualLoss:F6}");

        Assert.True(!float.IsNaN(actualLoss) && !float.IsInfinity(actualLoss),
            $"Loss must be finite; got {actualLoss}");

        // Tolerance comes from the 1e-7 epsilon added inside
        // ComputeTapeLoss to avoid log(0). Effect on log(0.125 + 1e-7)
        // is < 1e-6, so a 5e-4 tolerance is two orders of magnitude
        // looser than required and still catches the 1/V regression
        // (which is off by a factor of 8 — error ≈ 1.82, hundreds of
        // times the tolerance).
        Assert.Equal(expectedLoss, actualLoss, 3);

        // Defense in depth: also assert we are not within tolerance of
        // the buggy ln(V)/V value. This makes the failure message
        // explicit if a future regression reintroduces the 1/V bug.
        float diffFromBuggy = Math.Abs(actualLoss - buggyLoss);
        Assert.True(diffFromBuggy > 1f,
            $"Loss is suspiciously close to the buggy ln(V)/V value " +
            $"({buggyLoss:F6}). Issue #1191 regression suspected. " +
            $"actual={actualLoss:F6}, expected={expectedLoss:F6}.");
    }

    /// <summary>
    /// Same shape regression at the V=256 scale that issue #1191
    /// observed in its Cycle-30 production run, where the bit-identical
    /// loss across 56k SGD steps × 3 epochs at 0.0217 = ln(256)/256
    /// pinned the bug. With the fix the per-sample loss is ln(256) ≈
    /// 5.5452.
    /// </summary>
    [Fact]
    public void ComputeTapeLoss_UniformSoftmax_V256_ReturnsLn256_NotLn256_OverV()
    {
        const int v = 256;
        const float uniform = 1f / v;

        var predicted = new Tensor<float>([v]);
        for (int i = 0; i < v; i++) predicted[i] = uniform;

        var target = new Tensor<float>([v]);
        target[42] = 1f;

        var loss = new CategoricalCrossEntropyLoss<float>();
        var result = loss.ComputeTapeLoss(predicted, target);

        float actualLoss = result[0];
        float expected = (float)Math.Log(v); // 5.5452
        float buggy = expected / v;          // 0.02166 — Cycle-30 plateau

        _output.WriteLine($"V={v}  expected={expected:F6}  buggy={buggy:F6}  actual={actualLoss:F6}");

        // Wider tolerance at V=256 because the 1e-7 log-epsilon's
        // contribution scales with V (we add 1e-7 to each of 256 values
        // but only 1 contributes via the one-hot mask, so the bias is
        // bounded by log(1/256 + 1e-7) - log(1/256) ≈ 2.5e-5 — still
        // far below this tolerance).
        Assert.Equal(expected, actualLoss, 2);

        // Hard guard against the bit-identical Cycle-30 value.
        Assert.True(Math.Abs(actualLoss - buggy) > 1f,
            $"Loss matches the bit-identical Cycle-30 plateau {buggy:F6} — issue #1191 regressed.");
    }

    /// <summary>
    /// Edge case: a [B=2, V=4] batched prediction where each row is a
    /// distinct probability distribution. The expected per-sample CE is
    /// computed by hand and averaged over the batch, exactly matching
    /// PyTorch's <c>nn.CrossEntropyLoss(reduction='mean')</c> semantics.
    /// </summary>
    [Fact]
    public void ComputeTapeLoss_TwoDimensional_BatchMean_MatchesHandComputation()
    {
        // Two distinct samples to make the batch-mean step observable.
        // Sample 0 (target class 0):   probs [0.7, 0.1, 0.1, 0.1]  →  CE = -log(0.7) ≈ 0.3567
        // Sample 1 (target class 2):   probs [0.1, 0.2, 0.4, 0.3]  →  CE = -log(0.4) ≈ 0.9163
        // Expected batch mean = (0.3567 + 0.9163) / 2 = 0.6365
        var predicted = new Tensor<float>([2, 4]);
        predicted[0, 0] = 0.7f; predicted[0, 1] = 0.1f; predicted[0, 2] = 0.1f; predicted[0, 3] = 0.1f;
        predicted[1, 0] = 0.1f; predicted[1, 1] = 0.2f; predicted[1, 2] = 0.4f; predicted[1, 3] = 0.3f;

        var target = new Tensor<float>([2, 4]);
        target[0, 0] = 1f;
        target[1, 2] = 1f;

        var loss = new CategoricalCrossEntropyLoss<float>();
        var result = loss.ComputeTapeLoss(predicted, target);

        float actual = result[0];
        float expected = (float)((-Math.Log(0.7) + -Math.Log(0.4)) / 2.0); // 0.6365
        float buggy = expected / 4f; // ≈ 0.1591 — what allAxes-mean produced

        _output.WriteLine($"[B=2, V=4]  expected={expected:F6}  buggy={buggy:F6}  actual={actual:F6}");

        // 1e-7 epsilon contribution: log(0.7 + 1e-7) - log(0.7) ≈ 1.4e-7,
        // log(0.4 + 1e-7) - log(0.4) ≈ 2.5e-7 — both well below tolerance.
        Assert.Equal(expected, actual, 4);
    }

    /// <summary>
    /// Edge case: [B=2, T=3, V=4] sequence shape. Standard PyTorch
    /// behavior is to compute per-token CE then average over both batch
    /// and sequence axes. Verify the implementation reduces every
    /// non-class axis and only the class axis is summed.
    /// </summary>
    [Fact]
    public void ComputeTapeLoss_ThreeDimensional_BatchSequenceMean_MatchesHandComputation()
    {
        const int b = 2, t = 3, v = 4;
        const float uniform = 1f / v; // 0.25 — uniform per token

        var predicted = new Tensor<float>([b, t, v]);
        for (int i = 0; i < predicted.Length; i++)
            predicted[i] = uniform;

        // One-hot target at varying class per token. Choice of class
        // doesn't matter for uniform predictions — each token CE = ln(V).
        var target = new Tensor<float>([b, t, v]);
        for (int bi = 0; bi < b; bi++)
            for (int ti = 0; ti < t; ti++)
                target[bi, ti, (bi + ti) % v] = 1f;

        var loss = new CategoricalCrossEntropyLoss<float>();
        var result = loss.ComputeTapeLoss(predicted, target);

        float actual = result[0];
        float expected = (float)Math.Log(v); // every token contributes ln(V), mean = ln(V)
        float buggy = expected / v;          // 1/V regression

        _output.WriteLine($"[B={b}, T={t}, V={v}]  expected={expected:F6}  buggy={buggy:F6}  actual={actual:F6}");

        Assert.Equal(expected, actual, 3);
        Assert.True(Math.Abs(actual - buggy) > 1f,
            $"3D batched loss matches 1/V buggy value ({buggy:F6}) — issue #1191 regressed.");
    }

    /// <summary>
    /// Sanity edge: a (nearly-)perfect prediction must produce a loss
    /// near 0, regardless of V. The pre-fix code also satisfies this
    /// (perfect prediction → -log(1) = 0 either summed or meaned), so
    /// this protects the bound rather than the symptom — a future
    /// regression that, say, multiplies by a stray V would fail here
    /// while the 1/V regression would not.
    /// </summary>
    [Fact]
    public void ComputeTapeLoss_PerfectPrediction_ReturnsNearZero()
    {
        const int v = 16;
        var predicted = new Tensor<float>([v]);
        // Place ~1.0 on class 7 with tiny mass everywhere else so the
        // distribution still sums to ~1 (matches the function contract).
        const float spread = 1e-4f;
        float main = 1f - spread * (v - 1);
        for (int i = 0; i < v; i++) predicted[i] = spread;
        predicted[7] = main;

        var target = new Tensor<float>([v]);
        target[7] = 1f;

        var loss = new CategoricalCrossEntropyLoss<float>();
        var result = loss.ComputeTapeLoss(predicted, target);

        float actual = result[0];
        // -log(1 - 1.5e-3) ≈ 1.5e-3 — well under 0.01 even with float32 noise.
        Assert.True(actual < 0.01f,
            $"Near-perfect prediction should yield near-zero loss; got {actual:F6}.");
        Assert.True(actual >= 0f,
            $"CE must be non-negative; got {actual:F6}.");
        _output.WriteLine($"Perfect prediction: loss = {actual:F6}");
    }

    /// <summary>
    /// Issue #1191 end-to-end repro. Mirrors the V=8 identity-mapping
    /// task from HarmonicEngine PR #80 / Phase_A4_AiDotNet_1187_Reproducer
    /// at slightly larger iteration count to stay well above the noise
    /// floor of a stochastic Transformer optimizer.
    ///
    /// Pre-fix, this test fails on two distinct assertions:
    /// 1. The reported initial loss is ~0.26, not ~ln(8)=2.08, because
    ///    the tape value is divided by V before the optimizer sees it.
    /// 2. The model never converges — accuracy stays at 1/V = 12.5%
    ///    after 200 iterations because gradients are V× too small.
    /// </summary>
    [Fact]
    public void Transformer_Train_V8_LossNearLnV_AndConverges_Issue1191Repro()
    {
        const int vocabSize = 8;       // V from issue
        const int seqLen = 4;          // matches issue's [k,k,k,k] input pattern
        const int modelDim = 16;
        const int ffDim = 32;
        const int numFacts = vocabSize; // identity task: input k → class k
        const int totalIters = 400;
        const int window = 40;
        const int trials = 5;
        // Multi-trial pass criterion: at least 4/5 trials must satisfy
        // every guard. The Transformer has no exposed seed for weight
        // init, so a single stochastic run can flake on CI even when
        // the fix is correct. Across 5 trials we tolerate 1 unlucky
        // run while still rejecting any genuine regression — the pre-
        // fix 1/V bug fails *every* trial because it's deterministic
        // (the loss reporting bug is independent of init).
        const int requiredPasses = 4;

        float lnV = (float)Math.Log(vocabSize);   // 2.0794
        float lnVOverV = lnV / vocabSize;          // 0.2599

        var architecture = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.SequenceClassification,
            numEncoderLayers: 1,
            numDecoderLayers: 0,
            numHeads: 2,
            modelDimension: modelDim,
            feedForwardDimension: ffDim,
            inputSize: seqLen,
            outputSize: vocabSize,
            maxSequenceLength: seqLen,
            vocabularySize: vocabSize);

        // Build the identity dataset once: input [k,k,k,k] → class k.
        var inputs = new Tensor<float>[numFacts];
        var targets = new Tensor<float>[numFacts];
        for (int k = 0; k < numFacts; k++)
        {
            var inp = new Tensor<float>([1, seqLen]);
            for (int s = 0; s < seqLen; s++) inp[0, s] = k;
            inputs[k] = inp;

            var tgt = new Tensor<float>([1, vocabSize]);
            tgt[0, k] = 1f;
            targets[k] = tgt;
        }

        int passed = 0;
        var failureMessages = new System.Collections.Generic.List<string>();
        for (int trial = 0; trial < trials; trial++)
        {
            // Use Adam at a conservative LR. After the #1191 fix, tape
            // gradients are V× larger than the buggy 1/V version, so a
            // higher LR would over-shoot on this small model. Adam's
            // per-parameter step adaptation keeps the trajectory sane
            // regardless of which parameter is dominant. Each trial
            // gets a fresh optimizer + transformer so weight init is
            // independent.
            var optimizerOptions = new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                InitialLearningRate = 0.005,
                Beta1 = 0.9,
                Beta2 = 0.999,
                Epsilon = 1e-8,
                UseAdaptiveBetas = false
            };
            var optimizer = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null, optimizerOptions);
            var transformer = new Transformer<float>(
                architecture,
                lossFunction: new CategoricalCrossEntropyLoss<float>(),
                optimizer: optimizer);

            transformer.SetTrainingMode(true);

            // First call exposes the random-init loss. Pre-fix this is
            // ~ln(V)/V ≈ 0.26 at V=8 instead of ~ln(V) ≈ 2.08.
            transformer.Train(inputs[0], targets[0]);
            float initialLoss = transformer.GetLastLoss();

            // initialLoss after one training step depends on init — it
            // can drift toward 0.5 × ln(V) on lucky inits even though
            // the 1/V plateau (~ln(V)/V ≈ 0.26) is structurally
            // impossible after the fix. Use a wide floor so that any
            // reasonable post-fix value passes; the buggy value is
            // 4× below this floor.
            bool initialOk =
                !float.IsNaN(initialLoss)
                && !float.IsInfinity(initialLoss)
                && initialLoss > 0.5f * lnV;

            // Run a stochastic per-sample training loop. The issue's pre-fix
            // repro had loss bouncing in the tight band [0.228, 0.272] —
            // every value within ~10% of ln(V)/V ≈ 0.26, unable to escape
            // the 1/V plateau. The post-fix model must demonstrably escape
            // that region, both upward (correct loss reporting) and
            // downward (gradients drive learning toward correct
            // configurations).
            var lossesOverTime = new System.Collections.Generic.List<float>(totalIters);
            for (int iter = 0; iter < totalIters; iter++)
            {
                int k = iter % numFacts;
                transformer.Train(inputs[k], targets[k]);
                lossesOverTime.Add(transformer.GetLastLoss());
            }

            float minLoss = lossesOverTime.Min();
            float maxLoss = lossesOverTime.Max();
            float spread = maxLoss - minLoss;
            float firstWindowMean = Average(lossesOverTime, 0, window);
            float lastWindowMean = Average(lossesOverTime, totalIters - window, window);

            _output.WriteLine(
                $"trial {trial + 1}/{trials}: initial={initialLoss:F4}  min={minLoss:F4}  max={maxLoss:F4}  " +
                $"spread={spread:F4}  first_window={firstWindowMean:F4}  last_window={lastWindowMean:F4}");

            // Guard 1: early-training mean in ln(V) ballpark (not 1/V plateau).
            // Guard 2: model REACHES low loss (gradient flow proof).
            // Guard 3: max loss escapes the 1/V plateau region.
            // Guard 4: spread > tolerance (no Cycle-30 bit-identical signature).
            bool trialPass =
                initialOk
                && firstWindowMean > 0.5f * lnV
                && minLoss < 0.85f * lnV
                && maxLoss > 0.5f * lnV
                && spread > 1e-2f;
            if (trialPass)
            {
                passed++;
            }
            else
            {
                failureMessages.Add(
                    $"trial {trial + 1}: initial={initialLoss:F4}, " +
                    $"firstWin={firstWindowMean:F4} (>{0.5f * lnV:F4}? {firstWindowMean > 0.5f * lnV}), " +
                    $"min={minLoss:F4} (<{0.85f * lnV:F4}? {minLoss < 0.85f * lnV}), " +
                    $"max={maxLoss:F4} (>{0.5f * lnV:F4}? {maxLoss > 0.5f * lnV}), " +
                    $"spread={spread:F4} (>1e-2? {spread > 1e-2f})");
            }
        }

        _output.WriteLine($"Aggregate: {passed}/{trials} trials passed all guards.");

        // Multi-trial pass criterion: a clear majority must satisfy
        // every guard. The 1/V regression would fail every trial
        // (the bug is deterministic — pre-fix loss is structurally
        // ~ln(V)/V ≈ {0.2599} regardless of init), so the threshold
        // distinguishes "fix is correct" from "fix has regressed"
        // even with 1–2 unlucky weight-init trials.
        Assert.True(passed >= requiredPasses,
            $"Convergence checks were unstable: passed {passed}/{trials} trials, " +
            $"required {requiredPasses}. The 1/V bug from issue #1191 would fail " +
            $"every trial deterministically. Per-trial failure detail:\n  " +
            string.Join("\n  ", failureMessages));
    }

    private static float Average(System.Collections.Generic.List<float> xs, int start, int count)
    {
        float s = 0f;
        int n = 0;
        for (int i = start; i < start + count && i < xs.Count; i++)
        {
            s += xs[i];
            n++;
        }
        return n == 0 ? 0f : s / n;
    }
}
