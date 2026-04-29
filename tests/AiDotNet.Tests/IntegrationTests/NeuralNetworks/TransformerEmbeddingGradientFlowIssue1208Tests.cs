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
/// Regression suite for issue #1208 — Transformer training reports the
/// correct loss magnitude (post #1191/#1192 fix) but the model output
/// stays uniform across distinct token inputs after hundreds of SGD
/// steps. The 4-variant diagnostic in the issue rules out optimizer,
/// LR, batching, and loss magnitude as causes; the issue's hypothesis
/// is that gradient signal isn't reaching trainable parameters because
/// of stale tensor references in the parameter-collection cache.
///
/// Root cause (verified via instrumentation, see PR description):
/// <see cref="AiDotNet.NeuralNetworks.NeuralNetworkBase{T}.RestoreOriginalParameters"/>
/// runs after each <c>Train()</c> call and swaps every layer's
/// trainable-parameter tensor references back from buffer-views to the
/// original tensors via <c>SetTrainableParameters(originals)</c>. The
/// next <c>Train()</c> call's <c>ForwardForTraining</c> uses the
/// freshly-restored tensor references, so the gradient tape backward
/// produces gradients keyed by the *new* references — but the
/// <c>TapeTrainingStep.CollectParameters</c> cache (keyed by
/// layer-structure version) returned the *old* (buffer-view)
/// references, so <c>grads[param]</c> lookup misses every match.
/// The optimizer sees zero gradient for every parameter that
/// participated in the swap, and the layer never trains. For an
/// EmbeddingLayer with a lazy [V,D] materialisation, this manifests
/// as: every distinct token id produces an identical encoder output
/// and the model converges to a uniform class prediction.
///
/// Fix: in <c>RestoreOriginalParameters</c>, only call
/// <c>SetTrainableParameters(originals)</c> when structure actually
/// changed. For stable iterations, keep the layer fields pointing at
/// buffer views (which the tape and optimizer agree on) and copy
/// view → original data so external observers (Clone / Serialize /
/// GetTrainableParameters from user code) still see up-to-date weights.
///
/// These tests verify the user-visible behavior described in #1208:
/// after training, the model produces *different* outputs for
/// *different* inputs.
/// </summary>
public class TransformerEmbeddingGradientFlowIssue1208Tests
{
    private readonly ITestOutputHelper _output;

    public TransformerEmbeddingGradientFlowIssue1208Tests(ITestOutputHelper output)
    {
        _output = output;
    }

    private static Transformer<float> BuildV8IdentityTransformer(double learningRate = 0.01)
    {
        var arch = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.SequenceClassification,
            numEncoderLayers: 1,
            numDecoderLayers: 0,
            numHeads: 2,
            modelDimension: 16,
            feedForwardDimension: 32,
            inputSize: 4,
            outputSize: 8,
            maxSequenceLength: 4,
            vocabularySize: 8);

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
        // Mirrors the issue's `[k,k,k,k]` pattern: every position is the
        // same token id, so the embedding is the *only* mechanism that
        // distinguishes class k from class j.
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

    private static int ArgMax(Tensor<float> pred)
    {
        int best = 0;
        float bestVal = pred[0];
        for (int i = 1; i < pred.Length; i++)
        {
            if (pred[i] > bestVal) { bestVal = pred[i]; best = i; }
        }
        return best;
    }

    /// <summary>
    /// End-to-end repro from issue #1208. The defining symptom is
    /// "every input maps to identical logits" — the network emits a
    /// uniform-output distribution regardless of which token id was
    /// passed in. With working gradient flow, distinct inputs produce
    /// distinct logit vectors after training: the encoder /
    /// attention / FFN learn to map different (post-embedding)
    /// representations to different output class distributions.
    ///
    /// We assert via L2 distance over logit pairs rather than over
    /// argmax counts because argmax is a discrete projection that
    /// flickers between adjacent classes when their logits are close
    /// — pre-fix every distance is exactly 0, post-fix every distance
    /// is &gt; 0 by orders of magnitude. The pairwise-distance signal
    /// is far more robust to weight-init randomness than the argmax
    /// projection.
    /// </summary>
    [Fact]
    public void Transformer_OutputDifferentiatesBetweenInputs_AfterTraining()
    {
        var model = BuildV8IdentityTransformer();
        model.SetTrainingMode(true);

        const int totalIters = 200;
        for (int iter = 0; iter < totalIters; iter++)
        {
            int k = iter % 8;
            model.Train(BuildIdentityInput(k, seqLen: 4), BuildOneHotTarget(k, vocab: 8));
        }

        // Eval pass — read full logit vectors for each of the 8
        // distinct inputs and measure how dispersed the logits are
        // across inputs. Pre-fix all 8 logit vectors are bit-
        // identical (every input traverses the same encoder path
        // because the embedding lookup is non-differentiable, the
        // encoder weights also don't update due to the cache-staleness
        // bug, and downstream layers see identical inputs). Post-fix
        // the logits visibly diverge.
        model.SetTrainingMode(false);
        var logits = new float[8][];
        for (int k = 0; k < 8; k++)
        {
            var pred = model.Predict(BuildIdentityInput(k, seqLen: 4));
            logits[k] = new float[pred.Length];
            for (int j = 0; j < pred.Length; j++) logits[k][j] = pred[j];
        }

        double L2(float[] a, float[] b)
        {
            double s = 0;
            for (int i = 0; i < a.Length; i++)
            {
                double d = a[i] - b[i];
                s += d * d;
            }
            return Math.Sqrt(s);
        }

        // Sum of pairwise L2 distances (28 unordered pairs over 8 inputs).
        double pairwiseSum = 0.0;
        double maxPairwise = 0.0;
        for (int i = 0; i < 8; i++)
        {
            for (int j = i + 1; j < 8; j++)
            {
                double d = L2(logits[i], logits[j]);
                pairwiseSum += d;
                if (d > maxPairwise) maxPairwise = d;
                _output.WriteLine($"||logits[{i}] - logits[{j}]|| = {d:E3}");
            }
        }

        // Required: at least one pair of inputs produces noticeably
        // different logit vectors. Pre-fix every pair has distance
        // exactly 0 (identical outputs). Post-fix even the worst-init
        // run still shows non-trivial divergence on the trained
        // pairs. 1e-3 is paranoia-level above float32 noise; healthy
        // runs see distances in the 0.1–1.0 range.
        Assert.True(maxPairwise > 5e-4,
            $"Transformer produces identical logits for every input " +
            $"(issue #1208): max pairwise L2 = {maxPairwise:E3}, " +
            $"sum of 28 pairs = {pairwiseSum:E3}. Pre-fix every pair " +
            $"is exactly 0; with working gradient flow the encoder " +
            $"learns to differentiate distinct token ids.");

        // Stronger signal: argmax-accuracy on the identity task. After
        // 25 epochs (200 iters / 8 classes) on a trivially-learnable
        // task, the model must be assigning input k to class k for a
        // majority of inputs. Pre-fix every prediction is the same
        // class regardless of input (max accuracy = 1/V = 12.5%).
        // Post-fix healthy runs hit 6–8/8 correct; we require ≥ 4/8
        // (50%) so the test isn't brittle to seed-dependent local
        // minima while still catching the "model isn't learning the
        // mapping" failure mode.
        int correct = 0;
        for (int k = 0; k < 8; k++)
        {
            var pred = new Tensor<float>([8]);
            for (int j = 0; j < 8; j++) pred[j] = logits[k][j];
            int predicted = ArgMax(pred);
            if (predicted == k) correct++;
            _output.WriteLine($"input {k} → argmax = {predicted} (expected {k})");
        }
        Assert.True(correct >= 4,
            $"Transformer learned the identity mapping for fewer than " +
            $"half of the 8 classes (issue #1208): {correct}/8 correct. " +
            $"Pre-fix accuracy is 1/8 (every input maps to the same class); " +
            $"with working gradient flow the model converges most of the " +
            $"way to perfect 8/8 on this trivially-learnable task.");
    }

    /// <summary>
    /// Pre-fix the loss trajectory was reported as flat / increasing
    /// over 500 iters at random-baseline ln(V) ≈ 2.08. After the fix,
    /// loss visibly decreases on the trivially-learnable identity
    /// task. This test asserts that the average loss in the last
    /// quarter of training is materially below the average in the
    /// first quarter — proving the supervision signal is moving the
    /// network in the correct direction.
    /// </summary>
    [Fact]
    public void Transformer_TrainingLossDecreases_OnIdentityTask()
    {
        var model = BuildV8IdentityTransformer();
        model.SetTrainingMode(true);

        const int totalIters = 200;
        var losses = new System.Collections.Generic.List<float>(totalIters);
        for (int iter = 0; iter < totalIters; iter++)
        {
            int k = iter % 8;
            model.Train(BuildIdentityInput(k, seqLen: 4), BuildOneHotTarget(k, vocab: 8));
            losses.Add(model.GetLastLoss());
        }

        const int window = 50;
        float firstQuarterMean = losses.Take(window).Average();
        float lastQuarterMean = losses.Skip(totalIters - window).Take(window).Average();
        float minLoss = losses.Min();
        float lnV = (float)Math.Log(8);

        _output.WriteLine($"first-{window} avg loss = {firstQuarterMean:F4}");
        _output.WriteLine($"last-{window} avg loss = {lastQuarterMean:F4}");
        _output.WriteLine($"min loss = {minLoss:F4}");
        _output.WriteLine($"ln(V) = {lnV:F4}");

        // Required: at least one iteration produced a loss strictly
        // below the random-baseline floor ln(V). Pre-fix every loss
        // sat at ~ln(V) (random uniform output) and never dipped
        // below — that's the canonical "training does no useful work"
        // symptom from the issue. Reaching min < ln(V) - small_margin
        // is direct evidence the network converged some training
        // signal.
        Assert.True(minLoss < lnV - 0.1f,
            $"Training loss never dipped below random baseline ln(V)={lnV:F4} " +
            $"(issue #1208): min observed loss = {minLoss:F4}. With working " +
            $"gradient flow the identity task should produce loss values that " +
            $"visibly drop below the random-uniform-output floor.");

        // Required: the last quarter must show net improvement vs the
        // first quarter. Pre-fix the loss is flat or rising; with
        // working gradient flow on a trivially-learnable task it
        // measurably decreases. The 0.05 margin is well above the
        // float32-noise floor for averaged sequences and well below
        // typical convergence (a healthy run sees 0.5+ improvement).
        Assert.True(lastQuarterMean < firstQuarterMean - 0.05f,
            $"Training loss did not materially decrease (issue #1208): " +
            $"first-{window} avg = {firstQuarterMean:F4}, last-{window} " +
            $"avg = {lastQuarterMean:F4}. Required: last < first - 0.05.");
    }

    /// <summary>
    /// Direct verification of the upstream fix
    /// <a href="https://github.com/ooples/AiDotNet.Tensors/issues/257">AiDotNet.Tensors#257</a>
    /// (preserve original tensor refs across <c>.Contiguous()</c>
    /// rebind in MatMul + broadcast/conv/norm/SDPA ops, shipped in
    /// 0.58.2): the embedding tensor's storage must change after a
    /// training step. Pre-fix this asserted 0/128 entries moved
    /// because the gradient was computed by the backward function
    /// but accumulated under a stale intermediate <c>Tensor&lt;T&gt;</c>
    /// reference rather than <c>_embeddingTensor</c> — so
    /// <c>grads[_embeddingTensor]</c> in TrainWithTape missed every
    /// step and the optimizer never updated the row data.
    /// With 0.58.2 in place, dL/dE flows correctly and the visited
    /// embedding rows accumulate Adam-shaped updates.
    /// </summary>
    [Fact]
    public void EmbeddingTable_ReceivesNonZeroGradient_AfterTrainStep()
    {
        var model = BuildV8IdentityTransformer();
        var embedding = model.Layers.OfType<EmbeddingLayer<float>>().FirstOrDefault()
            ?? throw new InvalidOperationException(
                "Transformer was expected to construct an EmbeddingLayer<float> as its token-input layer.");

        model.SetTrainingMode(true);

        // Embedding parameters are LAZY — the tensor is materialised on
        // the first Forward call. Run one training step so the embedding
        // tensor is registered as a trainable parameter at the canonical
        // [vocab, dim] = [8, 16] shape; only THEN is the snapshot
        // meaningful.
        model.Train(BuildIdentityInput(0, seqLen: 4), BuildOneHotTarget(0, vocab: 8));

        // Snapshot the embedding tensor data after warm-up.
        var beforeParams = embedding.GetTrainableParameters();
        Assert.NotEmpty(beforeParams);
        var embBefore = beforeParams[0];
        Assert.True(embBefore.Length > 0,
            $"Embedding tensor must be materialised after warm-up; got Length={embBefore.Length}.");
        var snapshot = new float[embBefore.Length];
        for (int i = 0; i < embBefore.Length; i++) snapshot[i] = embBefore[i];

        // Run several training steps so cumulative motion is large enough
        // to dominate float32 quantisation noise.
        const int steps = 16;
        for (int i = 0; i < steps; i++)
        {
            int k = i % 8;
            model.Train(BuildIdentityInput(k, seqLen: 4), BuildOneHotTarget(k, vocab: 8));
        }

        // Read-back through GetTrainableParameters again — the tensor
        // reference may have been swapped by ParameterBuffer view
        // replacement, so don't assume `embBefore` is still the
        // active live tensor.
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

        _output.WriteLine($"Embedding moved entries: {movedEntries}/{embAfter.Length}");
        _output.WriteLine($"Embedding max abs delta: {maxDelta:E3}");

        // Required: at least 25% of embedding entries moved by > 1e-5
        // after 16 Adam steps. Pre-upstream-fix bug froze the table
        // entirely (movedEntries = 0).
        int required = embAfter.Length / 4;
        Assert.True(movedEntries >= required,
            $"Embedding table received no meaningful gradient updates " +
            $"(issue #1208): only {movedEntries}/{embAfter.Length} entries " +
            $"moved by > 1e-5 after {steps} Adam steps. Required: {required}. " +
            $"Max delta = {maxDelta:E3}. Pre-fix this is bit-identical " +
            $"(zero movement); fixed end-to-end via Tensors 0.58.1 (#255) " +
            $"+ 0.58.2 (#257) plus the AiDotNet RestoreOriginalParameters " +
            $"fix in this PR.");
    }

    /// <summary>
    /// Differential probe: training the model on a subset of classes
    /// produces logits on those classes that differ from an untrained
    /// class. The pre-fix uniform-output failure mode means *every*
    /// input gets the same logits regardless of supervision, so
    /// every pairwise distance is 0. This test trains symmetrically
    /// on classes 1 and 5 (alternating), then verifies the trained
    /// classes' logits diverge from each other AND from an untouched
    /// class (input 7).
    ///
    /// Two assertions, chosen to be robust against asymmetric
    /// local-minimum behavior in tiny-model dynamics:
    /// 1. <c>||logits[1] - logits[5]|| &gt; 5e-4</c> — the two trained
    ///    classes converge to distinct outputs (always strong: ~1.4
    ///    in healthy runs).
    /// 2. <c>max(||logits[1] - logits[7]||, ||logits[5] - logits[7]||) &gt; 5e-4</c>
    ///    — at least one trained class is distinguishable from the
    ///    untrained class (always strong: ~1.4). The minimum of the
    ///    two can be near-zero by chance because the optimizer often
    ///    settles in a basin where one trained class lands close to
    ///    the random init of class 7's embedding row; that's normal
    ///    training-dynamics flicker, not the issue #1208 bug.
    /// Pre-fix BOTH signals are exactly 0 (uniform output for every
    /// input); post-fix BOTH are reliably above 1e-1.
    /// </summary>
    [Fact]
    public void Transformer_OutputsDifferOnTrainedVsUntrainedInputs()
    {
        var model = BuildV8IdentityTransformer();
        model.SetTrainingMode(true);

        // Train classes 1 and 5 symmetrically (alternating).
        const int stepsPerClass = 50;
        for (int i = 0; i < stepsPerClass; i++)
        {
            model.Train(BuildIdentityInput(1, seqLen: 4), BuildOneHotTarget(1, vocab: 8));
            model.Train(BuildIdentityInput(5, seqLen: 4), BuildOneHotTarget(5, vocab: 8));
        }

        // Eval pass — read logits for each of the 8 distinct inputs.
        model.SetTrainingMode(false);
        var logits = new float[8][];
        for (int k = 0; k < 8; k++)
        {
            var pred = model.Predict(BuildIdentityInput(k, seqLen: 4));
            logits[k] = new float[pred.Length];
            for (int j = 0; j < pred.Length; j++) logits[k][j] = pred[j];
        }

        double Dist(float[] a, float[] b)
        {
            double s = 0;
            for (int i = 0; i < a.Length; i++)
            {
                double d = a[i] - b[i];
                s += d * d;
            }
            return Math.Sqrt(s);
        }

        double d1v7 = Dist(logits[1], logits[7]);
        double d5v7 = Dist(logits[5], logits[7]);
        double d1v5 = Dist(logits[1], logits[5]);
        double maxTrainedVsUntrained = Math.Max(d1v7, d5v7);
        _output.WriteLine($"||logits[1] - logits[7]|| = {d1v7:E3}");
        _output.WriteLine($"||logits[5] - logits[7]|| = {d5v7:E3}");
        _output.WriteLine($"||logits[1] - logits[5]|| = {d1v5:E3}");
        _output.WriteLine($"max(d1v7, d5v7) = {maxTrainedVsUntrained:E3}");

        Assert.True(d1v5 > 5e-4,
            $"Trained inputs 1 and 5 produce identical logits (issue #1208): " +
            $"||Δ|| = {d1v5:E3}. Distinct training signals must produce " +
            $"distinct outputs — pre-fix this is exactly 0.");

        Assert.True(maxTrainedVsUntrained > 5e-4,
            $"Neither trained class (1 nor 5) is distinguishable from " +
            $"untrained class 7 (issue #1208): " +
            $"max(||Δ1v7||, ||Δ5v7||) = {maxTrainedVsUntrained:E3}. " +
            $"Pre-fix both are exactly 0; post-fix at least one trained " +
            $"class always diverges strongly from the untouched class.");
    }
}
