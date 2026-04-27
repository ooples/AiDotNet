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
        Assert.True(maxPairwise > 1e-3,
            $"Transformer produces identical logits for every input " +
            $"(issue #1208): max pairwise L2 = {maxPairwise:E3}, " +
            $"sum of 28 pairs = {pairwiseSum:E3}. Pre-fix every pair " +
            $"is exactly 0; with working gradient flow the encoder " +
            $"learns to differentiate distinct token ids.");
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
    /// Differential probe: training the model on subsets of classes
    /// produces different logits on those classes than on untrained
    /// classes. The pre-fix uniform-output failure mode means *every*
    /// input gets the same logits regardless of supervision. This
    /// test trains exclusively on classes 1 and 5 (asymmetrically),
    /// then verifies the logits for inputs 1 and 5 differ from the
    /// logits for an untrained input (e.g., input 7). If the network
    /// is truly frozen post-training, all three logit vectors are
    /// identical; with working gradient flow they diverge.
    /// </summary>
    [Fact]
    public void Transformer_OutputsDifferOnTrainedVsUntrainedInputs()
    {
        var model = BuildV8IdentityTransformer();
        model.SetTrainingMode(true);

        // Train only classes 1 and 5, asymmetrically.
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

        // Compute pairwise L2 distances. Pre-fix all 8 logit vectors
        // are bit-identical (every input flows the same encoder
        // path), so all distances are zero. Post-fix the trained
        // inputs (1, 5) produce different logits from the untrained
        // input 7.
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
        _output.WriteLine($"||logits[1] - logits[7]|| = {d1v7:E3}");
        _output.WriteLine($"||logits[5] - logits[7]|| = {d5v7:E3}");
        _output.WriteLine($"||logits[1] - logits[5]|| = {d1v5:E3}");

        Assert.True(d1v7 > 1e-3,
            $"Trained input 1 produces logits identical to untrained input 7 " +
            $"(issue #1208): ||Δ|| = {d1v7:E3}. With working gradient flow " +
            $"the supervised inputs differ from the untouched ones.");

        Assert.True(d5v7 > 1e-3,
            $"Trained input 5 produces logits identical to untrained input 7 " +
            $"(issue #1208): ||Δ|| = {d5v7:E3}.");

        Assert.True(d1v5 > 1e-3,
            $"Trained inputs 1 and 5 produce identical logits (issue #1208): " +
            $"||Δ|| = {d1v5:E3}. Distinct training signals must produce " +
            $"distinct outputs.");
    }
}
