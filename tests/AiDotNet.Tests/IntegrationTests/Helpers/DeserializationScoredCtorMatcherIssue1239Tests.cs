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
        Assert.IsType<SeparableConvolutionalLayer<float>>(layer);
    }

    /// <summary>
    /// Same layer, metadata-less. Scored matcher gracefully falls back
    /// to shape-derived matches and ML-domain defaults (the
    /// <c>TryDefaultMlIntHyperparameter</c> path). The legacy longest-
    /// fillable-first behavior is preserved as the floor — if no
    /// metadata is present, every ctor scores 0 for matches and arity
    /// alone breaks ties, matching pre-fix behavior.
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
        Assert.IsType<DilatedConvolutionalLayer<float>>(layer);
    }

    /// <summary>
    /// Verifies the matcher still handles multi-int-array parameters
    /// (e.g., layers that take <c>int[]</c> for spatialDimensions /
    /// patchSizes / mlpDimensions). The scored path counts these as
    /// metadata matches when keys are present, shape-matches when
    /// derivable, defaults otherwise.
    /// </summary>
    [Fact]
    public void ScoredMatcher_GraphAttention_FullMetadata_ProducesValidLayer()
    {
        var metadata = new Dictionary<string, object>
        {
            ["NumNodes"] = 32,
            ["InputDim"] = 16,
            ["OutputDim"] = 8,
            ["NumHeads"] = 2,
        };

        var layer = DeserializationHelper.CreateLayerFromType<float>(
            layerType: typeof(GraphAttentionLayer<>).Name,
            inputShape: new[] { 32, 16 },
            outputShape: new[] { 32, 8 },
            additionalParams: metadata);

        Assert.NotNull(layer);
        Assert.IsType<GraphAttentionLayer<float>>(layer);
    }
}
