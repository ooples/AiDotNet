using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Inputs;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNetTests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Integration coverage for issue #1325 — InputLayer needs to be able to
/// declare a multi-dimensional input shape so that the strict
/// <c>AreLayersCompatible</c> SequenceEqual check passes for chains like
/// <c>InputLayer → MultiHeadAttentionLayer</c> (where MHA expects 2D
/// <c>[seq_len, embed_dim]</c> input).
///
/// Bug class: PR #1324 (closing #1321/#1322/#1323) relaxed shape
/// validation for Embedding-category targets via
/// <c>IsBroadcastInputCategory</c>. MultiHeadAttentionLayer is NOT
/// Embedding-category, so chains where the user pre-encodes a 2D tensor
/// outside the network and feeds it into MHA via <c>InputLayer(N)</c>
/// still failed: InputLayer's declared output shape <c>[N]</c> is 1D
/// rank while MHA's declared input shape <c>[seq_len, embed_dim]</c>
/// is 2D rank, and <c>AreShapesCompatible</c> rejects rank-mismatched
/// shapes even when their total element count matches.
///
/// Fix: a new <c>InputLayer(int[] inputShape)</c> constructor lets the
/// user declare the multi-dimensional shape natively. The validator then
/// sees matching ranks and shapes — no semantic relaxation needed.
/// </summary>
public class InputLayerMultiDimShapeIssue1325IntegrationTests
{
    // ====================================================================
    // ISSUE #1325 — main repro: InputLayer → MultiHeadAttention works
    //               when InputLayer is constructed with the matching 2D
    //               shape declared explicitly.
    // ====================================================================

    [Fact]
    public void Issue1325_InputLayerWithShape_AcceptsMultiHeadAttention()
    {
        const int CtxLen = 64;
        const int EmbedDim = 128;
        const int Heads = 2;
        const int VocabSize = 256;

        // Pre-encoded float input is [CtxLen, EmbedDim] = [64, 128].
        // The old 1D constructor InputLayer(CtxLen * EmbedDim) = InputLayer(8192)
        // produced output shape [8192] (rank 1), which was incompatible with
        // MultiHeadAttention's expected [64, 128] (rank 2) input shape under
        // the strict SequenceEqual check.
        //
        // The new int[] constructor lets the user declare [CtxLen, EmbedDim]
        // natively so the rank matches the downstream layer's contract.
        var layers = new List<ILayer<float>>
        {
            new InputLayer<float>(new[] { CtxLen, EmbedDim }),
            new MultiHeadAttentionLayer<float>(Heads, EmbedDim / Heads,
                activationFunction: (IActivationFunction<float>)new IdentityActivation<float>()),
            new SequenceTokenSliceLayer<float>(SequenceTokenSliceLayer<float>.Position.Last),
            new DenseLayer<float>(VocabSize, (IActivationFunction<float>)new IdentityActivation<float>()),
        };

        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: CtxLen,
            inputWidth: EmbedDim,
            outputSize: VocabSize,
            layers: layers);

        // Issue repro from #1325: previously threw
        //   "Layer 0 is not compatible with Layer 1." because InputLayer(8192)
        //   output [8192] mismatched MHA input [64, 128] under strict check.
        var network = new FeedForwardNeuralNetwork<float>(arch);

        Assert.Equal(layers.Count, network.Layers.Count);
        Assert.IsType<InputLayer<float>>(network.Layers[0]);
        Assert.IsType<MultiHeadAttentionLayer<float>>(network.Layers[1]);

        // Verify the InputLayer's declared shape matches what was passed.
        Assert.Equal(new[] { CtxLen, EmbedDim }, network.Layers[0].GetOutputShape());
    }

    [Fact]
    public void Issue1325_InputLayerWithShape_ProducesDeclaredOutputShape()
    {
        // The new constructor's output shape must match exactly the int[]
        // passed in (modulo defensive cloning).
        var layer = new InputLayer<float>(new[] { 8, 16, 32 });
        Assert.Equal(new[] { 8, 16, 32 }, layer.GetOutputShape());
        Assert.Equal(new[] { 8, 16, 32 }, layer.GetInputShape());
    }

    [Fact]
    public void Issue1325_InputLayerWithShape_ForwardPassThroughUnchanged()
    {
        // InputLayer is a pure pass-through. The new constructor must
        // preserve that semantics — Forward returns the input tensor
        // unchanged regardless of declared shape.
        var layer = new InputLayer<float>(new[] { 4, 8 });
        var input = new Tensor<float>([1, 4, 8]);
        for (int i = 0; i < input.Data.Length; i++) input.Data.Span[i] = i;

        var output = layer.Forward(input);
        Assert.Same(input, output);
    }

    [Fact]
    public void Issue1325_InputLayerWithShape_ClonesArrayDefensively()
    {
        // The constructor must clone its int[] argument so that external
        // mutation of the caller's array doesn't bleed into the layer's
        // internal state — guards against accidental shape corruption.
        var shape = new[] { 4, 8 };
        var layer = new InputLayer<float>(shape);
        shape[0] = 999;

        Assert.Equal(new[] { 4, 8 }, layer.GetOutputShape());
    }

    [Fact]
    public void Issue1325_InputLayerWithShape_CtorThrowsOnNull()
    {
        Assert.Throws<ArgumentNullException>(() => new InputLayer<float>((int[])null!));
    }

    [Fact]
    public void Issue1325_InputLayerWithShape_CtorThrowsOnEmpty()
    {
        var ex = Assert.Throws<ArgumentException>(() => new InputLayer<float>(Array.Empty<int>()));
        Assert.Contains("at least one dimension", ex.Message);
    }

    [Fact]
    public void Issue1325_InputLayerWithShape_CtorThrowsOnZeroDim()
    {
        var ex = Assert.Throws<ArgumentException>(() => new InputLayer<float>(new[] { 4, 0, 16 }));
        Assert.Contains("must be positive", ex.Message);
    }

    [Fact]
    public void Issue1325_InputLayerWithShape_CtorThrowsOnNegativeDim()
    {
        var ex = Assert.Throws<ArgumentException>(() => new InputLayer<float>(new[] { 4, -2, 16 }));
        Assert.Contains("must be positive", ex.Message);
    }

    // ====================================================================
    // REGRESSION GUARDS — the existing int constructor still works the
    // same way (produces 1D [inputSize] shape).
    // ====================================================================

    [Fact]
    public void Issue1325_RegressionGuard_IntConstructorStillProducesFlatShape()
    {
        // The existing InputLayer(int inputSize) constructor must keep its
        // 1D semantics — back-compat for every chain that used it.
        var layer = new InputLayer<float>(64);
        Assert.Equal(new[] { 64 }, layer.GetOutputShape());
        Assert.Equal(new[] { 64 }, layer.GetInputShape());
    }

    [Fact]
    public void Issue1325_RegressionGuard_IntConstructor_InputLayerToEmbeddingLayerStillWorks()
    {
        // The #1323 fix path (InputLayer(int) → EmbeddingLayer via the
        // broadcast-input bypass) must still pass after the new constructor
        // is added — no regression to PR #1324's behaviour.
        const int VocabSize = 256;
        const int EmbDim = 64;
        const int CtxLen = 64;

        var layers = new List<ILayer<float>>
        {
            new InputLayer<float>(CtxLen),
            new EmbeddingLayer<float>(vocabularySize: VocabSize, embeddingDimension: EmbDim),
            new DenseLayer<float>(VocabSize, (IActivationFunction<float>)new IdentityActivation<float>()),
        };

        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: CtxLen,
            outputSize: VocabSize,
            layers: layers);

        var network = new FeedForwardNeuralNetwork<float>(arch);
        Assert.Equal(3, network.Layers.Count);
    }

    // ====================================================================
    // EDGE CASES — verify the shape constructor works for higher ranks
    // and in chains that mix with other rank-aware layers.
    // ====================================================================

    [Fact]
    public void Issue1325_EdgeCase_ThreeDInput_Works()
    {
        // A 3D input shape, e.g. for a video / volumetric processing path.
        // The constructor must accept any positive rank.
        var layer = new InputLayer<float>(new[] { 16, 8, 4 });
        Assert.Equal(new[] { 16, 8, 4 }, layer.GetOutputShape());
        Assert.Equal(3, layer.GetOutputShape().Length);
    }

    [Fact]
    public void Issue1325_EdgeCase_SingleDimViaShapeCtor_EquivalentToIntCtor()
    {
        // InputLayer(new[] { N }) should behave identically to
        // InputLayer(N) — both produce 1D shape [N]. Verifies the new
        // ctor is a strict generalization, not a divergent path.
        var shapeCtor = new InputLayer<float>(new[] { 42 });
        var intCtor = new InputLayer<float>(42);

        Assert.Equal(intCtor.GetOutputShape(), shapeCtor.GetOutputShape());
        Assert.Equal(intCtor.GetInputShape(), shapeCtor.GetInputShape());
    }
}
