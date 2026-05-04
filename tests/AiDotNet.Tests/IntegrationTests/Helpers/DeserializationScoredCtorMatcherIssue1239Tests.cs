using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNetTests.IntegrationTests.Helpers;

/// <summary>
/// Regression coverage for the scored constructor matcher landed in
/// issue #1239. Pre-fix, <c>DeserializationHelper.TryConstructByMatchingMetadata</c>
/// iterated public ctors by descending parameter count and returned the
/// first ctor whose params were all fillable — a heuristic that let a
/// broader overload accepting heuristic / defaulted arguments beat a
/// narrower overload whose parameters would have been an exact metadata
/// match purely on arity.
///
/// <para>
/// Post-fix, ctors are scored by metadata-match count (×1000), then
/// shape-derived match count (×100), with arity as a final tie-breaker.
/// The highest-scoring fully-resolvable ctor wins. These tests exercise
/// the matcher fallback path on layers NOT in the explicit-branch list,
/// proving:
/// </para>
/// <list type="bullet">
///   <item>The matcher still produces a non-null layer when metadata is
///     fully populated (positive case).</item>
///   <item>The matcher still produces a non-null layer when metadata is
///     absent (graceful fallback to defaults — preserves the legacy
///     longest-fillable-first behavior as the floor).</item>
///   <item>Round-tripping a metadata-rich serialization preserves the
///     persisted hyperparameter values (the layer reconstructed from
///     metadata reports those values via its public properties / shape).</item>
/// </list>
/// </summary>
public class DeserializationScoredCtorMatcherIssue1239Tests
{
    /// <summary>
    /// Layer with multiple ctors (scalar vs vector activation overloads)
    /// + full hyperparameter metadata. The scored matcher resolves
    /// OutputDepth, KernelSize, Stride, Padding via direct metadata key
    /// hits (4 × 1000 = 4000 base) and picks the scalar-activation
    /// overload via <c>TryRestoreActivation</c> + <c>ScalarActivationType</c>.
    /// </summary>
    [Fact]
    public void ScoredMatcher_SeparableConv_FullMetadata_ProducesValidLayer()
    {
        var metadata = new Dictionary<string, object>
        {
            ["OutputDepth"] = 16,
            ["KernelSize"] = 3,
            ["Stride"] = 2,
            ["Padding"] = 1,
            ["ScalarActivationType"] = "AiDotNet.ActivationFunctions.ReLUActivation`1",
        };

        var layer = DeserializationHelper.CreateLayerFromType<float>(
            layerType: typeof(SeparableConvolutionalLayer<>).Name,
            inputShape: new[] { 8, 8, 8 },
            outputShape: new[] { 16, 4, 4 },
            additionalParams: metadata);

        Assert.NotNull(layer);
        var sepConv = Assert.IsType<SeparableConvolutionalLayer<float>>(layer);

        // Verify the scored matcher selected a ctor that honored the
        // metadata. Parameter count is the cleanest observable: it
        // depends solely on the metadata-supplied OutputDepth (16),
        // KernelSize (3), and inputDepth (8 from inputShape[0]).
        // SeparableConv parameter count = depthwise kernels (kernelSize²
        // × inputDepth) + pointwise kernels (inputDepth × outputDepth) +
        // biases (outputDepth). With our metadata: 3·3·8 + 8·16 + 16 = 216.
        // A wrong-ctor pick (e.g. one that resolved kernelSize from
        // defaults) would land on a different parameter count.
        Assert.Equal(3 * 3 * 8 + 8 * 16 + 16, sepConv.ParameterCount);
    }

    /// <summary>
    /// Same layer, metadata-less. Scored matcher gracefully falls back
    /// to shape-derived matches and ML-domain defaults (the
    /// <c>TryDefaultMlIntHyperparameter</c> path). With metadata absent
    /// each ctor still earns shape-derived hits (×100) where the
    /// inputShape / outputShape supply parameters by name (e.g.
    /// outputDepth from outputShape[0]); only the metadata-hit term
    /// (×1000) goes to zero. Ranking can therefore differ from pure
    /// arity ordering, but the matcher still produces a valid layer
    /// because shape-derived defaults plus
    /// <c>TryDefaultMlIntHyperparameter</c> backfill the rest.
    /// </summary>
    [Fact]
    public void ScoredMatcher_SeparableConv_NoMetadata_StillResolves()
    {
        var layer = DeserializationHelper.CreateLayerFromType<float>(
            layerType: typeof(SeparableConvolutionalLayer<>).Name,
            inputShape: new[] { 8, 8, 8 },
            outputShape: new[] { 16, 4, 4 },
            additionalParams: null);

        Assert.NotNull(layer);
        Assert.IsType<SeparableConvolutionalLayer<float>>(layer);
    }

    /// <summary>
    /// DilatedConvolutionalLayer is another matcher-fallback target.
    /// Verifies the scored matcher handles DilationFactor, OutputDepth,
    /// KernelSize, Stride, Padding metadata round-trip in one of the
    /// non-explicit-branch layer families. Pre-#1239 this exact path
    /// could mis-pick if DilatedConvolutionalLayer ever grew an arity-
    /// mismatched overload; the scored algorithm makes that future-proof.
    /// </summary>
    [Fact]
    public void ScoredMatcher_DilatedConv_FullMetadata_ProducesValidLayer()
    {
        var metadata = new Dictionary<string, object>
        {
            ["OutputDepth"] = 32,
            ["KernelSize"] = 3,
            ["Stride"] = 1,
            ["Padding"] = 2,
            ["DilationFactor"] = 2,
        };

        var layer = DeserializationHelper.CreateLayerFromType<float>(
            layerType: typeof(DilatedConvolutionalLayer<>).Name,
            inputShape: new[] { 16, 16, 16 },
            outputShape: new[] { 32, 16, 16 },
            additionalParams: metadata);

        Assert.NotNull(layer);
        var dilConv = Assert.IsType<DilatedConvolutionalLayer<float>>(layer);

        // Verify the scored matcher honored the metadata-driven
        // hyperparameters via parameter count, which depends solely on
        // metadata values: kernelSize² × inputDepth × outputDepth +
        // outputDepth biases. With KernelSize=3, inputDepth=16,
        // OutputDepth=32: 3·3·16·32 + 32 = 4640. A wrong ctor pick
        // (e.g., a different overload that resolved kernelSize from
        // defaults) would land on a different parameter count. We
        // intentionally don't assert spatial output dims because the
        // layer recomputes them from input + dilation + padding rather
        // than echoing back the supplied outputShape.
        Assert.Equal(3 * 3 * 16 * 32 + 32, dilConv.ParameterCount);
    }

    /// <summary>
    /// Verifies the matcher honors metadata keyed by the *actual* ctor
    /// parameter names. GraphAttentionLayer's ctor takes
    /// (<c>inputFeatures</c>, <c>outputFeatures</c>, <c>numHeads</c>,
    /// <c>alpha</c>, <c>dropoutRate</c>, …). The matcher pascal-cases
    /// parameter names when looking up <c>additionalParams</c>, so
    /// <c>InputFeatures</c> / <c>OutputFeatures</c> / <c>NumHeads</c>
    /// hit the metadata path (×1000 each), exercising the scored
    /// matcher's positive branch.
    /// </summary>
    [Fact]
    public void ScoredMatcher_GraphAttention_FullMetadata_ProducesValidLayer()
    {
        var metadata = new Dictionary<string, object>
        {
            ["InputFeatures"] = 16,
            ["OutputFeatures"] = 8,
            ["NumHeads"] = 2,
            ["Alpha"] = 0.2,
            ["DropoutRate"] = 0.0,
        };

        var layer = DeserializationHelper.CreateLayerFromType<float>(
            layerType: typeof(GraphAttentionLayer<>).Name,
            inputShape: new[] { 16 },
            outputShape: new[] { 8 },
            additionalParams: metadata);

        Assert.NotNull(layer);
        var gat = Assert.IsType<GraphAttentionLayer<float>>(layer);

        // Verify the scored matcher actually piped the metadata
        // through: a wrong-ctor pick (e.g., one that resolved numHeads
        // from defaults instead of metadata) would land on
        // gat.NumHeads != 2. These three properties are the public
        // observables for the ctor parameters that came from metadata.
        Assert.Equal(16, gat.InputFeatures);
        Assert.Equal(8, gat.OutputFeatures);
        Assert.Equal(2, gat.NumHeads);
    }
}
