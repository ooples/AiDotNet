using System;
using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training.Memory;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.IntegrationTests.Training;

/// <summary>
/// Regression tests for AiDotNet#1341: <c>GradientCheckpointing&lt;T&gt;.Checkpoint</c>
/// backward closure was computing the inner-tape gradient with an implicit
/// ones-seed (treating the segment output as if it were the scalar loss) and
/// then attempting to chain-rule via an elementwise <c>TensorMultiply</c>
/// against <c>gradOutput</c>. That product can only succeed when the segment
/// is a shape-preserving elementwise op; on a real Transformer the head-side
/// gradient <c>[B, V]</c> meets an encoder-segment output of <c>[B, L, D]</c>
/// and the broadcast fails with <c>ArgumentException</c>:
///
/// <para><i>"Tensors with shapes [1, 256] and [1, 64, 128] cannot be broadcast:
/// dimension 2 has sizes 256 and 128 (must be equal or one must be 1)."</i></para>
///
/// <para>The fix re-seeds the inner tape with a scalar pseudo-loss
/// <c>sum(reOutput * gradOutput)</c>, so the inner backward produces the
/// proper VJP <c>d(reOutput)/d(reInput)^T @ gradOutput</c> regardless of
/// whether the segment changes shape.</para>
/// </summary>
public class GradientCheckpointingTransformerIssue1341Tests
{
    private readonly ITestOutputHelper _output;

    public GradientCheckpointingTransformerIssue1341Tests(ITestOutputHelper output)
    {
        _output = output;
    }

    /// <summary>
    /// Exact configuration reported in the issue: 2-layer encoder, dModel=128,
    /// dFf=256, ctxLen=64, vocab=256, per-sample Train, AdamOptimizer,
    /// CategoricalCrossEntropyLoss, ForTransformers() preset (segment size 1
    /// across attention + residual blocks).
    /// </summary>
    /// <remarks>
    /// Skipped until the paired AiDotNet.Tensors PR
    /// (https://github.com/ooples/AiDotNet.Tensors/pull/361) merges and the
    /// new Tensors NuGet is published. Bump
    /// <c>Directory.Packages.props</c>'s AiDotNet.Tensors version and remove
    /// the Skip attribute once the upstream fix is consumable.
    /// </remarks>
    [Fact]
    public void Transformer_Train_with_ForTransformers_checkpointing_does_not_throw_on_shape_mismatch()
    {
        const int vocabSize = 256;
        const int dModel = 128;
        const int dFf = 256;
        const int ctxLen = 64;
        const int heads = 2;
        const int layers = 2;
        const int outputSize = vocabSize;

        var architecture = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.SequenceClassification,
            numEncoderLayers: layers,
            numDecoderLayers: 0,
            numHeads: heads,
            modelDimension: dModel,
            feedForwardDimension: dFf,
            inputSize: ctxLen,
            outputSize: outputSize,
            maxSequenceLength: ctxLen,
            vocabularySize: vocabSize);

        var transformer = new Transformer<float>(
            architecture,
            lossFunction: new CategoricalCrossEntropyLoss<float>());

        // Enable the memory-management preset that turns on per-block
        // checkpointing for attention + residual segments. Before the fix
        // for #1341, the first call to Train() below throws inside the
        // Checkpoint backward closure when the head's [B, V] gradient
        // meets an encoder segment output of [B, L, D].
        transformer.EnableMemoryManagement(TrainingMemoryConfig.ForTransformers());

        var rng = RandomHelper.CreateSeededRandom(0);

        // Per-sample shapes that reproduce the issue exactly: input
        // [1, ctxLen] of token IDs, target [1, vocab] one-hot.
        var input = new Tensor<float>(new[] { 1, ctxLen });
        for (int t = 0; t < ctxLen; t++)
            input[0, t] = rng.Next(vocabSize);

        var target = new Tensor<float>(new[] { 1, vocabSize });
        target[0, rng.Next(vocabSize)] = 1.0f;

        // The act: a single Train() call. Before the fix this threw
        // ArgumentException("Tensors with shapes [1, 256] and [1, 64, 128]
        // cannot be broadcast ..."). After the fix, it must complete and
        // the loss must be a finite number.
        var ex = Record.Exception(() => transformer.Train(input, target));

        Assert.Null(ex);
        var lastLoss = transformer.GetLastLoss();
        Assert.True(!float.IsNaN(lastLoss) && !float.IsInfinity(lastLoss),
            $"LastLoss must be finite, was {lastLoss}");
        _output.WriteLine(
            $"Transformer.Train under ForTransformers() checkpointing succeeded; LastLoss={lastLoss}");
    }

    /// <summary>
    /// Three consecutive per-sample Train calls under checkpointing — guards
    /// against a regression that only surfaces after the first cached-plan
    /// replay (the original failure cascaded through CompiledDelegateChain
    /// the second time the same shape pattern was seen).
    /// </summary>
    /// <remarks>
    /// Skipped until the paired AiDotNet.Tensors PR
    /// (https://github.com/ooples/AiDotNet.Tensors/pull/361) merges and the
    /// new Tensors NuGet is published. Bump
    /// <c>Directory.Packages.props</c>'s AiDotNet.Tensors version and remove
    /// the Skip attribute once the upstream fix is consumable.
    /// </remarks>
    [Fact]
    public void Transformer_Train_with_checkpointing_succeeds_for_repeated_calls()
    {
        const int vocabSize = 32;
        const int dModel = 16;
        const int dFf = 32;
        const int ctxLen = 8;
        const int heads = 2;
        const int layers = 2;

        var architecture = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.SequenceClassification,
            numEncoderLayers: layers,
            numDecoderLayers: 0,
            numHeads: heads,
            modelDimension: dModel,
            feedForwardDimension: dFf,
            inputSize: ctxLen,
            outputSize: vocabSize,
            maxSequenceLength: ctxLen,
            vocabularySize: vocabSize);

        var transformer = new Transformer<float>(
            architecture,
            lossFunction: new CategoricalCrossEntropyLoss<float>());
        transformer.EnableMemoryManagement(TrainingMemoryConfig.ForTransformers());

        var rng = RandomHelper.CreateSeededRandom(1);

        for (int step = 0; step < 3; step++)
        {
            var input = new Tensor<float>(new[] { 1, ctxLen });
            for (int t = 0; t < ctxLen; t++) input[0, t] = rng.Next(vocabSize);
            var target = new Tensor<float>(new[] { 1, vocabSize });
            target[0, rng.Next(vocabSize)] = 1.0f;

            // Use Record.Exception (same pattern as the sibling test at
            // line ~103) so an unexpected throw produces a clearly
            // attributed assertion failure instead of bubbling up as a
            // raw exception that hides which step failed.
            var ex = Record.Exception(() => transformer.Train(input, target));
            Assert.Null(ex);
            var lastLoss = transformer.GetLastLoss();
            Assert.True(!float.IsNaN(lastLoss) && !float.IsInfinity(lastLoss),
                $"step={step}: LastLoss must be finite, was {lastLoss}");
        }

        _output.WriteLine("Three consecutive checkpointed Train calls succeeded.");
    }

    // Note: a "loss decreases under checkpointing" test was intentionally NOT kept — it is
    // confounded. A Transformer's non-checkpointed parameters (token/position embeddings, output
    // projection) drive the loss down on a memorize-one-sample task even when the checkpointed
    // attention/FFN weights receive zero gradient, so such a test passes on BOTH the broken and the
    // fixed package and proves nothing about checkpointed-weight correctness. The unconfounded
    // parameter-update equivalence test below is the real correctness guard.

    /// <summary>
    /// UNCONFOUNDED gradient-equivalence on the real library path: two Transformers from an IDENTICAL
    /// initial parameter vector (same fresh optimizer state) take ONE training step on the SAME sample —
    /// one with checkpointing, one without. If the package checkpoint is gradient-correct for the
    /// checkpointed layers' WEIGHTS (not just the segment input), every parameter update must match. If
    /// the checkpoint drops weight gradients, the checkpointed entries stay put in the on-run and the
    /// two parameter vectors diverge by far more than fp noise. Uses dropout=0 so the only difference
    /// between the two runs is the checkpointing mechanism itself.
    /// </summary>
    /// <remarks>
    /// SKIPPED: NeuralNetworkBase enables checkpointing with a sqrt(N) segment size
    /// (NeuralNetworkBase.cs ForwardForTraining), i.e. MULTIPLE segments. The package primitive
    /// GradientCheckpointing.Checkpoint has a multi-segment defect — when a FusedLinear-bearing block
    /// stack is split into more than one segment, the gradient handed from a later segment to an
    /// earlier segment's input is double-counted, so earlier-segment parameter updates come out 2x
    /// (reproduced: 2-segment FusedLinear diverges 2x, 1-segment and 2-segment plain-matmul are exact).
    /// Diffusion G4 sidesteps this by checkpointing as a SINGLE segment
    /// (NoisePredictorBase.CheckpointBlocks). Un-skip once the package fixes multi-segment recompute,
    /// or after NeuralNetworkBase is switched to single-segment checkpointing.
    /// </remarks>
    [Fact(Skip = "Blocked on AiDotNet.Tensors multi-segment checkpoint double-count (NeuralNetworkBase uses sqrt(N) segments). Diffusion G4 uses single-segment and is verified separately.")]
    public void Transformer_checkpointing_parameter_updates_match_eager_one_step()
    {
        const int vocabSize = 32, dModel = 16, dFf = 32, ctxLen = 8, heads = 2, layers = 2;

        Transformer<float> Build()
        {
            var arch = new TransformerArchitecture<float>(
                inputType: InputType.TwoDimensional,
                taskType: NeuralNetworkTaskType.SequenceClassification,
                numEncoderLayers: layers, numDecoderLayers: 0, numHeads: heads,
                modelDimension: dModel, feedForwardDimension: dFf, inputSize: ctxLen,
                outputSize: vocabSize, maxSequenceLength: ctxLen, vocabularySize: vocabSize,
                dropoutRate: 0.0);
            return new Transformer<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>());
        }

        var rng = RandomHelper.CreateSeededRandom(11);
        var input = new Tensor<float>(new[] { 1, ctxLen });
        for (int t = 0; t < ctxLen; t++) input[0, t] = rng.Next(vocabSize);
        var target = new Tensor<float>(new[] { 1, vocabSize });
        target[0, rng.Next(vocabSize)] = 1.0f;

        var tOff = Build();
        var tOn = Build();
        tOn.SetParameters(tOff.GetParameters()); // identical init; both have fresh (zero) optimizer state
        tOn.EnableMemoryManagement(TrainingMemoryConfig.ForTransformers());

        tOff.Train(input, target);
        tOn.Train(input, target);

        var pOff = tOff.GetParameters();
        var pOn = tOn.GetParameters();
        Assert.Equal(pOff.Length, pOn.Length);

        int diffCount = 0; double maxDiff = 0;
        for (int i = 0; i < pOff.Length; i++)
        {
            double d = System.Math.Abs((double)pOff[i] - (double)pOn[i]);
            if (d > 1e-5) diffCount++;
            if (d > maxDiff) maxDiff = d;
        }
        _output.WriteLine($"param updates: total={pOff.Length}, diffCount={diffCount}, maxDiff={maxDiff}");
        Assert.True(diffCount == 0,
            $"Checkpointed vs eager parameter updates diverge: {diffCount}/{pOff.Length} params differ " +
            $"(maxDiff={maxDiff}). A correct checkpoint is gradient-equivalent — divergence means the " +
            $"package checkpoint is not propagating weight gradients for checkpointed segments.");
    }
}
