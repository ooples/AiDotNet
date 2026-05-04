using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.IntegrationTests.Optimizers;

/// <summary>
/// Regression coverage for issue #1245 — <c>AdamOptimizer.UpdateSolution</c>
/// throws <c>ArgumentException: Vector lengths must match. Got 431360 and
/// 524288</c> when training a Transformer with <c>numEncoderLayers=8</c>.
///
/// <para>
/// Root cause: <see cref="NeuralNetworkBase{T}.GetParameters"/> walks
/// <c>layer.GetParameters()</c> per layer (which can include non-trainable
/// persistent state plus trainable weights), but
/// <see cref="NeuralNetworkBase{T}.ComputeGradients"/> walks
/// <c>trainable.GetTrainableParameters()</c> for ITrainableLayer layers
/// — which only returns tensors registered via
/// <c>RegisterTrainableParameter</c>. When a layer has more entries in
/// <c>GetParameters()</c> than in <c>GetTrainableParameters()</c>, the
/// gradient vector ends up shorter than the parameter vector, and
/// Adam's vector ops crash on length mismatch.
/// </para>
///
/// <para>
/// At low encoder counts (L=1/2/4) the discrepancy is small enough to
/// not crash but produces silently-degenerate training (uniform output,
/// see related #1232). At L=8 the cumulative discrepancy crosses the
/// vector-op tolerance and crashes outright.
/// </para>
/// </summary>
public class AdamOptimizerLengthMismatchIssue1245Tests
{
    private readonly ITestOutputHelper _output;

    public AdamOptimizerLengthMismatchIssue1245Tests(ITestOutputHelper output)
    {
        _output = output;
    }

    /// <summary>
    /// Builds an L=8 Transformer and computes a gradient, then verifies
    /// the gradient vector and parameter vector have matching lengths
    /// (the precondition for any optimizer's element-wise update). Pre-
    /// fix: gradient length is 431360, parameter length is 524288, the
    /// next AdamOptimizer.UpdateSolution call would throw on the first
    /// vector op. Post-fix: lengths match and the optimizer runs.
    /// </summary>
    [Fact(Timeout = 60000)]
    public async Task ComputeGradients_L8Transformer_GradientLengthMatchesParameterLength()
    {
        await Task.Yield();

        // Mirrors the issue's repro config: dModel=64, dFf=256, numHeads=4,
        // L=8, vocab=256. The user's exact numbers (params=524288,
        // gradients=431360) come from this config. Pre-fix: divergent
        // lengths, AdamOptimizer.UpdateSolution crashes. Post-fix:
        // matched lengths, optimizer can run.
        var arch = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.SequenceClassification,
            numEncoderLayers: 8,
            numDecoderLayers: 0,
            numHeads: 4,
            modelDimension: 64,
            feedForwardDimension: 256,
            inputSize: 64,
            outputSize: 256,
            maxSequenceLength: 64,
            vocabularySize: 256);

        var model = new Transformer<float>(
            arch,
            lossFunction: new CategoricalCrossEntropyLoss<float>());

        // Build a representative input + target. The actual gradient
        // values don't matter for this test — we only assert that the
        // gradient vector LENGTH matches the parameter vector LENGTH.
        var input = new Tensor<float>(new[] { 1, 64 });
        for (int s = 0; s < 64; s++) input[0, s] = s % 8;
        var target = new Tensor<float>(new[] { 1, 256 });
        target[0, 0] = 1f;

        // ComputeGradients runs a forward pass first which materializes
        // every lazy layer's parameter tensors. After this call, every
        // encoder layer's _isInitialized is true and GetParameters
        // returns the full network parameter vector.
        var gradient = model.ComputeGradients(input, target, new CategoricalCrossEntropyLoss<float>());
        int gradLength = gradient.Length;

        // Now query parameter length AFTER initialization so we're
        // comparing the post-forward state (which is what AdamOptimizer
        // sees in its UpdateSolution loop).
        var paramVector = model.GetParameters();
        int paramLength = paramVector.Length;

        _output.WriteLine($"L=8 Transformer: parameters={paramLength}, gradients={gradLength}, diff={paramLength - gradLength}");

        Assert.Equal(paramLength, gradLength);
    }

    /// <summary>
    /// End-to-end smoke: instantiate AdamOptimizer's UpdateSolution path
    /// at L=8 and confirm it doesn't throw. Pre-fix this is the exact
    /// crash in the issue's stack trace ("Vector lengths must match.
    /// Got 431360 and 524288"). Post-fix the call returns a model with
    /// updated parameters.
    /// </summary>
    [Fact(Timeout = 60000)]
    public async Task AdamOptimizer_L8Transformer_UpdateSolutionDoesNotThrow()
    {
        await Task.Yield();

        var arch = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.SequenceClassification,
            numEncoderLayers: 8,
            numDecoderLayers: 0,
            numHeads: 4,
            modelDimension: 64,
            feedForwardDimension: 256,
            inputSize: 64,
            outputSize: 256,
            maxSequenceLength: 64,
            vocabularySize: 256);

        var optOptions = new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
        {
            InitialLearningRate = 0.001,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8,
        };

        var model = new Transformer<float>(
            arch,
            lossFunction: new CategoricalCrossEntropyLoss<float>(),
            optimizer: new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null, optOptions));

        var input = new Tensor<float>(new[] { 1, 64 });
        for (int s = 0; s < 64; s++) input[0, s] = s % 8;
        var target = new Tensor<float>(new[] { 1, 256 });
        target[0, 0] = 1f;

        // Train() drives the full forward + gradient + Adam update path
        // that crashes pre-fix. Should complete without throwing.
        model.Train(input, target);

        // If we got here without throwing, the length-mismatch bug is
        // resolved. Sanity: model still produces a same-shape prediction.
        var pred = model.Predict(input);
        Assert.Equal(256, pred.Length);
    }

    /// <summary>
    /// Companion regression for issue #1232 (referenced from #1245). The
    /// user reported flat-softmax convergence (top-1 = 0%, PPL = V
    /// exactly) at L=1/2/4 Transformer with Adam — meaning training
    /// silently produces uniform output regardless of input. The
    /// connection to #1245: when ComputeGradients pre-fix emitted a
    /// gradient vector shorter than the parameter vector, Adam's update
    /// path was either silently zero-padding or updating only a prefix
    /// of parameters, leaving the bulk of the encoder stack untouched.
    /// At L=8 the discrepancy crossed a threshold and crashed; at
    /// L=1/2/4 it survived but no real training happened, hence flat
    /// softmax.
    ///
    /// <para>This test trains a small L=2 Transformer for 500 Adam
    /// steps on an identity-mapping task and verifies that training
    /// caused a meaningful CHANGE in the model's output behavior. We
    /// snapshot pairwise logit divergence before and after training
    /// and assert that the after-divergence is materially larger.
    /// Asserting only post-training divergence is too weak — a randomly
    /// initialized Transformer already produces input-dependent logits
    /// (different token IDs hit different embedding rows), so the
    /// post-only check could pass even when training never updated
    /// the encoder stack. The before/after delta directly pins
    /// "training is doing something" rather than "init is non-trivial".
    /// </para>
    ///
    /// <para>Flake-safety: the assertion uses a before/after DELTA
    /// rather than an absolute magnitude. The library's
    /// <c>RandomHelper.ThreadSafeRandom</c> has no public seed API, so
    /// we can't deterministically pin weight init across CI agents —
    /// but the delta metric is robust to init variance because both
    /// snapshots use the same initial weights, and the post-training
    /// growth signal dominates init noise by 3+ orders of magnitude.</para>
    /// </summary>
    [Fact(Timeout = 60000)]
    public async Task L2Transformer_AdamTraining_OutputsLearnRatherThanFlatSoftmax()
    {
        await Task.Yield();

        const int vocab = 8;
        const int seqLen = 4;

        var arch = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.SequenceClassification,
            numEncoderLayers: 2,
            numDecoderLayers: 0,
            numHeads: 2,
            modelDimension: 16,
            feedForwardDimension: 32,
            inputSize: seqLen,
            outputSize: vocab,
            maxSequenceLength: seqLen,
            vocabularySize: vocab);

        var optOptions = new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
        {
            InitialLearningRate = 0.01,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8,
        };

        var model = new Transformer<float>(
            arch,
            lossFunction: new CategoricalCrossEntropyLoss<float>(),
            optimizer: new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null, optOptions));

        Tensor<float> Identity(int classIndex)
        {
            var t = new Tensor<float>(new[] { 1, seqLen });
            for (int s = 0; s < seqLen; s++) t[0, s] = classIndex;
            return t;
        }
        Tensor<float> OneHot(int classIndex)
        {
            var t = new Tensor<float>(new[] { 1, vocab });
            t[0, classIndex] = 1f;
            return t;
        }

        double MaxPairwiseLogitL2()
        {
            var logits = new float[vocab][];
            for (int k = 0; k < vocab; k++)
            {
                var pred = model.Predict(Identity(k));
                logits[k] = new float[pred.Length];
                for (int j = 0; j < pred.Length; j++) logits[k][j] = pred[j];
            }
            double maxD = 0.0;
            for (int i = 0; i < vocab; i++)
            {
                for (int j = i + 1; j < vocab; j++)
                {
                    double s = 0;
                    for (int d = 0; d < logits[i].Length; d++)
                    {
                        double diff = logits[i][d] - logits[j][d];
                        s += diff * diff;
                    }
                    double dist = System.Math.Sqrt(s);
                    if (dist > maxD) maxD = dist;
                }
            }
            return maxD;
        }

        // Snapshot divergence BEFORE training. A randomly-initialized
        // Transformer already gives different logits for different
        // token IDs (embedding lookup hits different rows), so this
        // baseline is typically > 0 — exactly the value we need to
        // distinguish "incidental init diversity" from "training did
        // something".
        double divergenceBefore = MaxPairwiseLogitL2();

        // Train. Pre-fix: gradient updates never reach the encoder
        // stack so the divergence stays at its init level. Post-fix:
        // gradients flow through every parameter and divergence grows
        // as the model learns class-specific output distributions.
        model.SetTrainingMode(true);
        for (int iter = 0; iter < 500; iter++)
        {
            int k = iter % vocab;
            model.Train(Identity(k), OneHot(k));
        }
        model.SetTrainingMode(false);

        double divergenceAfter = MaxPairwiseLogitL2();
        double delta = divergenceAfter - divergenceBefore;

        _output.WriteLine($"L=2 Transformer, 500 Adam iters: pairwise logit L2 before={divergenceBefore:E3}, after={divergenceAfter:E3}, delta={delta:E3}");

        // Pre-fix: delta ≈ 0 (training is a no-op for the encoder, so
        // post-training divergence ~ pre-training divergence). Post-fix:
        // training amplifies divergence by orders of magnitude as the
        // model fits the identity mapping. 1e-3 is well above any
        // floating-point noise and well below typical post-training
        // values (which land in 0.1–1.0 range). Margin chosen to
        // tolerate seed-init variance on the absolute floor without
        // false-positive flake.
        Assert.True(delta > 1e-3,
            $"L=2 Transformer with Adam should learn meaningfully — divergence after training " +
            $"must materially exceed pre-training baseline. Got before={divergenceBefore:E3}, " +
            $"after={divergenceAfter:E3}, delta={delta:E3} (issue #1232 flat-softmax symptom — " +
            $"gradient was not flowing through encoder stack pre-fix).");
    }
}
