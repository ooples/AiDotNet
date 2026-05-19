using System.Threading.Tasks;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Consumer-side regression coverage for AiDotNet#1346 (FlashAttentionLayer
/// degenerate output on the compiled fused-Adam path). The original root
/// cause — Engine.FlashAttention missing its GraphMode.IsActive lazy-graph
/// recording branch — was fixed by AiDotNet.Tensors PR #362 and ships in
/// AiDotNet.Tensors NuGet 0.81.3. This file pins TWO things:
/// <list type="number">
/// <item>The engine-side fix actually reaches the AiDotNet fused training
/// path: a Transformer&lt;float&gt; whose layer stack contains
/// <see cref="FlashAttentionLayer{T}"/> engages
/// <see cref="AiDotNet.Training.CompiledTapeTrainingStep{T}.TryStepWithFusedOptimizer"/>
/// when trained via the public network API (canary test below).</item>
/// <item>The remaining consumer-side gap that the #1346 investigation
/// surfaced — Tensors plan-loss-readout silently returning literal 0
/// instead of the actual NaN/Inf when a CCE-style chain produces NaN under
/// many trainable parameters — is tracked at
/// <a href="https://github.com/ooples/AiDotNet.Tensors/issues/396">AiDotNet.Tensors#396</a>.
/// The Skip'd regression test below auto-enables once that fix lands and
/// the consuming NuGet version bumps.</item>
/// </list>
/// </summary>
/// <remarks>
/// PR #1386 review (CodeRabbit C8Bm6 + Copilot Drjj5): both tests reset and
/// read <see cref="AiDotNet.Training.CompiledTapeTrainingStep{T}"/>'s
/// thread-static fused-step counter and cache. Default xUnit per-class
/// parallelization would race those resets/reads against any other test
/// touching the same global state (FusedOptimizerIntegrationTests etc.),
/// producing flaky engaged-count assertions or cross-test counter leak.
/// Join the existing "FusedOptimizerGlobalState" collection (defined in
/// <see cref="FusedOptimizerCollection"/>) so xUnit serializes every test
/// in this class with every other CompiledTapeTrainingStep-mutating test.
/// </remarks>
[Collection("FusedOptimizerGlobalState")]
public class FlashAttentionFusedCompiledTrainingIssue1346Tests
{
    private readonly ITestOutputHelper _output;

    public FlashAttentionFusedCompiledTrainingIssue1346Tests(ITestOutputHelper output)
    {
        _output = output;
    }

    private const int SeqLen = 4;
    private const int EmbedDim = 16;
    private const int HeadCount = 2;
    private const int NumClasses = 8;

    /// <summary>
    /// Builds a small Transformer whose explicit layer list contains
    /// <see cref="FlashAttentionLayer{T}"/> as the attention block — the same
    /// drop-in-replacement pattern AiDotNet#1346 documented as broken on the
    /// fused-Adam path before AiDotNet.Tensors PR #362 landed.
    /// </summary>
    private static Transformer<float> BuildFlashAttentionTransformer(double learningRate = 0.01)
    {
        // No EmbeddingLayer: input is continuous-valued [1, seq, embed].
        // EmbeddingLayer-first trips a pre-existing TransformerArchitecture
        // input-dim validator quirk (see TransformerCustomLayerValidationIssue1317IntegrationTests
        // .CustomTransformerLayerStack_AcceptsFlashAttentionLayerAsDropInReplacement)
        // that is unrelated to #1346.
        var layers = new List<ILayer<float>>
        {
            new FlashAttentionLayer<float>(SeqLen, EmbedDim, HeadCount),
            new LayerNormalizationLayer<float>(),
            new SequenceTokenSliceLayer<float>(SequenceTokenSliceLayer<float>.Position.Last),
            new DenseLayer<float>(NumClasses, (IActivationFunction<float>)new IdentityActivation<float>())
        };

        var arch = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.SequenceClassification,
            numEncoderLayers: 0, // explicit layers: list replaces the auto-built encoder block (#1382)
            numDecoderLayers: 0,
            numHeads: HeadCount,
            modelDimension: EmbedDim,
            feedForwardDimension: EmbedDim,
            complexity: NetworkComplexity.Medium,
            inputSize: SeqLen * EmbedDim,
            outputSize: NumClasses,
            dropoutRate: 0.0,
            maxSequenceLength: SeqLen,
            vocabularySize: NumClasses,
            usePositionalEncoding: false,
            temperature: 1.0,
            sequencePooling: null,
            layers: layers);

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

    private static Tensor<float> BuildFingerprintInput(int classIndex, int seed)
    {
        var t = new Tensor<float>([1, SeqLen, EmbedDim]);
        var rng = new System.Random(seed * 1000 + classIndex);
        for (int s = 0; s < SeqLen; s++)
        {
            for (int e = 0; e < EmbedDim; e++)
            {
                t[0, s, e] = (float)(classIndex + 0.05 * rng.NextDouble());
            }
        }
        return t;
    }

    private static Tensor<float> BuildOneHotTarget(int classIndex)
    {
        var t = new Tensor<float>([1, NumClasses]);
        t[0, classIndex] = 1f;
        return t;
    }

    /// <summary>
    /// CANARY for AiDotNet.Tensors PR #362's reach into the public AiDotNet
    /// fused-Adam training path. A Transformer whose layer stack contains
    /// FlashAttentionLayer must engage
    /// <see cref="AiDotNet.Training.CompiledTapeTrainingStep{T}.TryStepWithFusedOptimizer"/>
    /// on the first Train() call. Pre-fix (before #362) the GraphMode lazy
    /// trace inside the fused path would still record everything except
    /// FlashAttention; the fused step would run successfully (so this
    /// counter would still increment) but downstream gradient flow would
    /// be broken. This test specifically verifies the canary — a regression
    /// that prevents fused-path engagement at all (e.g. a future Tensors
    /// change that throws during GraphMode trace) would flip this red.
    /// </summary>
    [Fact(Timeout = 60000)]
    public async Task FlashAttentionLayer_TrainViaFusedCompiledAdam_EngagesFusedPath()
    {
        await Task.Yield();
        AiDotNet.Training.CompiledTapeTrainingStep<float>.ResetFusedStepCount();
        AiDotNet.Training.CompiledTapeTrainingStep<float>.Invalidate();

        var model = BuildFlashAttentionTransformer();
        model.SetTrainingMode(true);

        var input = BuildFingerprintInput(0, seed: 7);
        var target = BuildOneHotTarget(0);

        model.Train(input, target);

        long fusedSteps = AiDotNet.Training.CompiledTapeTrainingStep<float>.GetFusedStepCount();
        _output.WriteLine($"Fused step count after 1 Train() call: {fusedSteps}");

        Assert.True(fusedSteps > 0,
            $"FlashAttentionLayer Transformer fell back to eager on first Train() — " +
            $"CompiledTapeTrainingStep<float>.GetFusedStepCount() = {fusedSteps}. " +
            "This indicates Engine.FlashAttention threw during GraphMode trace OR a " +
            "downstream compile gate rejected the FA-containing graph. See AiDotNet.Tensors " +
            "PR #362 and AiDotNet issue #1346.");
    }

    /// <summary>
    /// Future-fix regression for the consumer-side gap surfaced during AiDotNet#1346
    /// investigation. Tracks <a href="https://github.com/ooples/AiDotNet.Tensors/issues/396">AiDotNet.Tensors#396</a>:
    /// when a model with multiple trainable parameters routes raw logits through
    /// <see cref="CategoricalCrossEntropyLoss{T}"/> on the fused-Adam path,
    /// the loss chain's <c>log(negative_logit + eps)</c> produces NaN that
    /// SHOULD propagate to <see cref="NeuralNetworkBase{T}.GetLastLoss"/> but
    /// instead surfaces as literal float 0. This silent zeroing was the
    /// actual reason the original #1346 reporter's HE PathB sanity test
    /// stayed at top1=0% / top5=100% / ppl=V after the engine-side
    /// FlashAttention fix landed — the consumer's loss-curve looked
    /// "converged" while gradients were corrupted.
    /// <para>
    /// Skipped until AiDotNet.Tensors#396 ships and the NuGet version bumps.
    /// On enabling: this test must report <c>lastLoss = NaN</c> (or a finite
    /// positive value if the underlying chain doesn't actually NaN at this seed),
    /// but MUST NOT report literal 0.
    /// </para>
    /// </summary>
    [Fact(Timeout = 60000, Skip = "Blocked on AiDotNet.Tensors#396 — fused-Adam loss-readout returns literal 0 instead of NaN under CCE+raw-logits+many-params. Unskip once that fix lands and the AiDotNet.Tensors NuGet version bumps past the build containing it.")]
    public async Task DenseIdentity_CCE_OnFusedAdam_DoesNotSilentlyZeroNaN()
    {
        await Task.Yield();
        AiDotNet.Training.CompiledTapeTrainingStep<float>.ResetFusedStepCount();
        AiDotNet.Training.CompiledTapeTrainingStep<float>.Invalidate();

        // Force-negative-logit setup: Dense layer with IdentityActivation passes
        // raw logits (potentially negative) into CCE, whose log(p + 1e-7) goes NaN.
        var layers = new List<ILayer<float>>
        {
            new DenseLayer<float>(EmbedDim, (IActivationFunction<float>)new IdentityActivation<float>()),
            new LayerNormalizationLayer<float>(),
            new SequenceTokenSliceLayer<float>(SequenceTokenSliceLayer<float>.Position.Last),
            new DenseLayer<float>(NumClasses, (IActivationFunction<float>)new IdentityActivation<float>())
        };
        var arch = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.SequenceClassification,
            numEncoderLayers: 0, numDecoderLayers: 0,
            numHeads: HeadCount, modelDimension: EmbedDim, feedForwardDimension: EmbedDim,
            complexity: NetworkComplexity.Medium,
            inputSize: SeqLen * EmbedDim, outputSize: NumClasses,
            dropoutRate: 0.0, maxSequenceLength: SeqLen, vocabularySize: NumClasses,
            usePositionalEncoding: false, temperature: 1.0,
            sequencePooling: null, layers: layers);
        var optimizer = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null,
            new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                InitialLearningRate = 0.01,
                Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8
            });
        var model = new Transformer<float>(arch,
            lossFunction: new CategoricalCrossEntropyLoss<float>(),
            optimizer: optimizer);
        model.SetTrainingMode(true);

        var input = BuildFingerprintInput(0, seed: 42);
        var target = BuildOneHotTarget(0);

        model.Train(input, target);
        long fusedSteps = AiDotNet.Training.CompiledTapeTrainingStep<float>.GetFusedStepCount();
        float lastLoss = model.GetLastLoss();

        _output.WriteLine($"Identity+CCE on fused-Adam: fusedSteps={fusedSteps}, lastLoss={lastLoss}, " +
            $"IsNaN={float.IsNaN(lastLoss)}, IsInfinity={float.IsInfinity(lastLoss)}, " +
            $"IsZero={lastLoss == 0f}");

        Assert.True(fusedSteps > 0, "Fused path must have engaged");

        // The signal: lastLoss must be either a sane positive number OR NaN/Inf.
        // Literal 0 means the silent-failure mode behind AiDotNet#1346 / Tensors#396 —
        // NaN was produced inside the loss chain but the fused readout silently
        // zeroed it, so the consumer thinks training is converging while
        // gradients are corrupted and the model never moves off random init.
        bool isSilentlyZero = lastLoss == 0f && !float.IsNaN(lastLoss);
        Assert.False(isSilentlyZero,
            $"AiDotNet.Tensors#396 regression: Identity+CCE on fused-Adam reports " +
            $"lastLoss=0 (literal 0, not NaN). The fused readout is silently zeroing " +
            "the NaN that the CCE log(negative_logit+eps) chain produces. Consumer " +
            "would see 'loss converged' while gradients are corrupted.");
    }
}
