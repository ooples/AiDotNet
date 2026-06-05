using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Repro for the HRE PAPER-A chain shape-validator failure reported in
/// HarmonicEngine PR #149: when a chain ends with
/// <c>MultiHeadAttention -> SequenceTokenSliceLayer -> custom-readout(rank-2 specific shape)</c>
/// the custom-chain validator rejects both pairs with "Layer N is not compatible with Layer N+1".
///
/// <para>The validator is exercised through its real entry points
/// (<c>new Transformer&lt;float&gt;(arch)</c> at construction time and the
/// post-forward <c>DeepCopy</c> revalidation path inside the optimizer
/// pipeline), not by re-implementing the compatibility algorithm in the
/// test. The mock readout layer (<see cref="FixedShapeRank2Layer"/>)
/// declares the same shape contract as HarmonicEngine's <c>HreReadoutLayer</c>,
/// which is what surfaces the validator regression.</para>
/// </summary>
public class HrePaperAChainShapeValidatorTests
{
    /// <summary>
    /// Validator runs at chain-construction time and rejects the chain if any
    /// adjacent-layer pair fails <c>AreShapesCompatible</c>. This is the first
    /// time the rank-N declared (MHA <c>[-1, embDim]</c>) vs rank-(N+1)
    /// resolved (Slice <c>[batch, ctxLen, embDim]</c>) shape pair is checked.
    /// Pre-fix the chain construction threw with
    /// "Layer N (MultiHeadAttention) is not compatible with Layer N+1 (SequenceTokenSliceLayer)".
    /// </summary>
    [Fact]
    public void HreChain_ConstructTransformer_PassesValidator()
    {
        var arch = BuildHreLikeArchitecture();

        // ctor runs the chain validator. A regression of the rank-mismatch
        // fix would throw here with the "Layer N is not compatible with Layer M"
        // diagnostic.
        var model = new Transformer<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>());

        Assert.NotNull(model);
        Assert.True(model.Layers.Count >= MinExpectedLayers,
            $"expected ≥{MinExpectedLayers} layers in built chain, got {model.Layers.Count}");
    }

    /// <summary>
    /// After a real <c>Forward</c> mutates the slice layer's shape from
    /// <c>[-1, -1, -1]</c> to a concrete <c>[batch, ctxLen, embDim]</c>, a
    /// downstream <c>DeepCopy</c> re-runs the validator on the (now resolved)
    /// chain — this is the exact failure path from HarmonicEngine PR #149,
    /// triggered by <c>AdamOptimizer.Optimize</c> →
    /// <c>OptimizerBase.PrepareAndEvaluateSolution</c> → <c>DeepCopy</c>.
    /// Pre-fix the post-forward revalidation threw because the validator
    /// couldn't reconcile MHA's still-declared rank-2 output with Slice's
    /// now-rank-3 resolved input.
    /// </summary>
    [Fact]
    public void HreChain_DeepCopyAfterForward_PassesRevalidation()
    {
        var arch = BuildHreLikeArchitecture();
        var model = new Transformer<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>());

        // Real forward — this is what causes SequenceTokenSliceLayer to call
        // ResolveShapes with a concrete batch dim, mutating the chain's
        // observable shape state.
        var input = new Tensor<float>(new[] { Batch, CtxLen });
        var span = input.AsWritableSpan();
        for (int i = 0; i < span.Length; i++) span[i] = i % VocabSize;
        _ = model.Predict(input);

        // DeepCopy re-instantiates the chain (and re-runs ValidateCustomLayers
        // internally) against the same architecture but now with resolved
        // upstream shape state. A regression here would throw
        // ArgumentException with "Layer N is not compatible with Layer M".
        var copy = (NeuralNetworkBase<float>)model.DeepCopy();

        Assert.NotNull(copy);
        // DeepCopy must return a distinct top-level model instance — without
        // that, parameter updates on the copy would mutate the original.
        Assert.NotSame(model, copy);
        Assert.Equal(model.Layers.Count, copy.Layers.Count);
        // The chain must be structurally equivalent — same layer types in the
        // same positions. Per-instance reference-equality on individual layers
        // is implementation-defined (parameter-less layers are legitimately
        // shared) and not part of the DeepCopy contract.
        for (int i = 0; i < model.Layers.Count; i++)
        {
            Assert.Equal(model.Layers[i].GetType(), copy.Layers[i].GetType());
        }
    }

    /// <summary>
    /// Negative-direction guard: when the shorter shape's leading dim is a
    /// concrete positive (not the wildcard <c>-1</c>), the validator must
    /// still reject. Without this guard the rank-mismatch fix would bless
    /// genuinely incompatible chains like <c>[32, 32]</c> vs <c>[3, 32, 32]</c>.
    /// </summary>
    [Fact]
    public void HreChain_RankMismatchWithoutWildcard_RejectedByValidator()
    {
        // Chain crafted so that ONLY the 1→2 edge fails compatibility — every
        // other transition (0→1, 2→3) is shape-compatible. Without this
        // careful construction, a "Layer X is not compatible with Layer Y"
        // assertion could be triggered by an unrelated edge and the regression
        // we're guarding (rank-mismatch wildcard guard) would silently pass
        // even if it broke.
        //
        //   0: InputLayer(64)          out [64]
        //   1: ProbeLayer [64] → [32, 32]                     ← 0→1 OK
        //   2: ProbeLayer [3, 32, 32] → [3, 32, 32]           ← 1→2 FAILS:
        //         Layer 1 out [32, 32] (rank 2, no wildcard) vs
        //         Layer 2 in  [3, 32, 32] (rank 3, concrete leading 3).
        //         Rank-mismatch wildcard guard MUST reject — without it the
        //         validator would silently strip the leading 3 and accept.
        //   3: ProbeLayer [3, 32, 32] → [vocab]                ← 2→3 OK
        var layers = new List<ILayer<float>>
        {
            new InputLayer<float>(64),
            new FixedShapeRank2Layer(
                inputShape: new[] { 64 },
                outputShape: new[] { 32, 32 }),
            new FixedShapeRank2Layer(
                inputShape: new[] { 3, 32, 32 },
                outputShape: new[] { 3, 32, 32 }),
            new FixedShapeRank2Layer(
                inputShape: new[] { 3, 32, 32 },
                outputShape: new[] { VocabSize }),
        };

        var arch = new TransformerArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.SequenceClassification,
            // Custom layers: REPLACE the auto-built encoder, so numEncoderLayers must be 0 (#1382).
            numEncoderLayers: 0,
            numDecoderLayers: 0,
            numHeads: 1,
            modelDimension: 32,
            feedForwardDimension: 64,
            inputSize: 64,
            outputSize: VocabSize,
            maxSequenceLength: 64,
            vocabularySize: VocabSize,
            layers: layers);

        var ex = Assert.Throws<ArgumentException>(() =>
            new Transformer<float>(arch, lossFunction: new CategoricalCrossEntropyLoss<float>()));

        // Pin the assertion to the SPECIFIC 1→2 rejection, with both the
        // structural tokens AND the concrete shapes that surface in the
        // shape-enriched diagnostic. Without these, the test would also
        // pass on any other edge-N "not compatible" failure — meaning a
        // regression of the rank-mismatch wildcard guard could ship while
        // a different bug at edge 0→1 keeps the assertion green.
        Assert.Contains("Layer 1", ex.Message);
        Assert.Contains("output [32, 32]", ex.Message);
        Assert.Contains("is not compatible with Layer 2", ex.Message);
        Assert.Contains("input [3, 32, 32]", ex.Message);

        // Also assert there is NO reported failure on the 0→1 edge — the
        // chain is constructed so 0→1 ([64] → [64]) is compatible and we
        // rely on that to isolate the 1→2 regression.
        Assert.DoesNotContain("Layer 0", ex.Message);
    }

    // ============================================================================
    // Test fixtures
    // ============================================================================

    private const int CtxLen = 8;
    private const int EmbDim = 16;
    private const int Heads = 2;
    private const int VocabSize = 64;
    private const int Batch = 2;
    private const int MinExpectedLayers = 4;

    /// <summary>
    /// Builds an architecture isomorphic to HarmonicEngine's
    /// <c>ApplesToApplesHreChain</c>: <c>InputLayer → embed → MHA × 2 →
    /// SequenceTokenSliceLayer → readout</c>. We use AiDotNet's stock
    /// <see cref="EmbeddingLayer{T}"/> as the embed and a
    /// <see cref="FixedShapeRank2Layer"/> as the readout — the shape
    /// declarations are what the validator checks; the math inside is
    /// irrelevant for this regression.
    /// </summary>
    private static TransformerArchitecture<float> BuildHreLikeArchitecture()
    {
        var layers = new List<ILayer<float>>
        {
            new InputLayer<float>(CtxLen),
            new EmbeddingLayer<float>(vocabularySize: VocabSize, embeddingDimension: EmbDim),
            new MultiHeadAttentionLayer<float>(Heads, EmbDim / Heads,
                activationFunction: (IActivationFunction<float>)new IdentityActivation<float>()),
            new MultiHeadAttentionLayer<float>(Heads, EmbDim / Heads,
                activationFunction: (IActivationFunction<float>)new IdentityActivation<float>()),
            new SequenceTokenSliceLayer<float>(SequenceTokenSliceLayer<float>.Position.Last),
            new FixedShapeRank2Layer(
                inputShape: new[] { -1, EmbDim },
                outputShape: new[] { VocabSize }),
        };

        return new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.SequenceClassification,
            // Custom layers: REPLACE the auto-built encoder, so numEncoderLayers must be 0 (#1382).
            numEncoderLayers: 0,
            numDecoderLayers: 0,
            numHeads: Heads,
            modelDimension: EmbDim,
            feedForwardDimension: 2 * EmbDim,
            inputSize: CtxLen,
            outputSize: VocabSize,
            maxSequenceLength: CtxLen,
            vocabularySize: VocabSize,
            layers: layers);
    }

    /// <summary>
    /// No-op layer with explicit input/output shapes. Mimics HarmonicEngine's
    /// <c>HreReadoutLayer</c> shape contract (rank-2 input <c>[batch, embDim]</c>
    /// or <c>[-1, embDim]</c>, rank-1 output <c>[vocabSize]</c>) for the
    /// validator-only regression tests. Forward is identity so we can run a
    /// real <c>Predict</c> through the chain without depending on HRE math.
    /// </summary>
    private sealed class FixedShapeRank2Layer : LayerBase<float>
    {
        public FixedShapeRank2Layer(int[] inputShape, int[] outputShape)
            : base(inputShape, outputShape)
        {
        }

        public override long ParameterCount => 0;
        public override bool SupportsTraining => false;

        public override Tensor<float> Forward(Tensor<float> input)
        {
            // Reshape the input flat-buffer to the declared output shape.
            // The declared output is [vocabSize] (rank 1) for the readout-mock
            // case, but for diagnostic chains we may have [-1, embDim] →
            // [-1, embDim] etc. Producing the right rank/dims by reshape from
            // the input's total element count keeps the chain well-formed.
            var outShape = GetOutputShape();
            // Resolve wildcards from the input shape's batch dim if needed.
            var resolved = new int[outShape.Length];
            int knownProduct = 1;
            int wildcardIdx = -1;
            for (int i = 0; i < outShape.Length; i++)
            {
                if (outShape[i] <= 0) { wildcardIdx = i; resolved[i] = -1; }
                else { resolved[i] = outShape[i]; knownProduct *= outShape[i]; }
            }
            int inputElements = 1;
            for (int i = 0; i < input.Shape.Length; i++) inputElements *= input.Shape[i];
            if (wildcardIdx >= 0)
            {
                resolved[wildcardIdx] = Math.Max(1, inputElements / Math.Max(1, knownProduct));
            }
            // Produce a zero tensor of the resolved shape — this layer is a
            // shape-contract stand-in, not a real layer.
            return new Tensor<float>(resolved);
        }

        public override void UpdateParameters(float learningRate) { }
        public override Vector<float> GetParameters() => new Vector<float>(0);
        public override void SetParameters(Vector<float> parameters) { }
        public override Vector<float> GetParameterGradients() => new Vector<float>(0);
        public override void ResetState() { }
        public override LayerBase<float> Clone() => new FixedShapeRank2Layer(GetInputShape(), GetOutputShape());
    }
}
