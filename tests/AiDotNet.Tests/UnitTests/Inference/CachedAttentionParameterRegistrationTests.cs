using System;
using AiDotNet.Inference;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Inference;

/// <summary>
/// The inference-time attention layers must expose their projection weights as parameters.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="CachedMultiHeadAttention{T}"/>, <see cref="CachedGroupedQueryAttention{T}"/> and
/// <see cref="PagedCachedMultiHeadAttention{T}"/> each allocate five weight tensors in their
/// constructors — query, key, value and output projections plus the output bias — and none of them
/// registered those tensors. Every one reported <c>ParameterCount</c> 0 with shapes fully resolved.
/// </para>
/// <para>
/// That is not a cosmetic count. <c>GetParameters</c> returned an empty vector and
/// <c>SetParameters</c> was a silent no-op, so the weights could not be read, written, trained, or
/// saved. <c>InferenceOptimizer</c> rewrites a trained attention layer into these classes and
/// transfers the weights through exactly that round-trip, so the "optimized" model silently kept
/// freshly-initialized projections and computed something different from the model it replaced.
/// </para>
/// <para>
/// Note the shape of the symptom: <c>IsShapeResolved</c> was true and <c>ParameterCount</c> was
/// zero. That reads like a shape-resolution problem and is not one — only probing the layer
/// distinguishes them.
/// </para>
/// </remarks>
public class CachedAttentionParameterRegistrationTests
{
    private const int EmbeddingDimension = 16;
    private const int Heads = 4;

    private static void AssertParametersRoundTrip(dynamic layer, string name)
    {
        Vector<float> before = layer.GetParameters();

        Assert.True(before.Length > 0,
            $"{name} reported {before.Length} parameters; its five projection tensors are not registered, " +
            "so its weights cannot be trained, transferred by InferenceOptimizer, or saved.");

        var replacement = new Vector<float>(before.Length);
        for (int i = 0; i < replacement.Length; i++)
        {
            // A distinct, non-uniform pattern: a constant would still "round-trip" through a layer
            // that ignored the write and happened to be initialized to that constant.
            replacement[i] = 0.25f + (i % 7) * 0.03125f;
        }

        layer.SetParameters(replacement);
        Vector<float> after = layer.GetParameters();

        Assert.Equal(replacement.Length, after.Length);
        for (int i = 0; i < after.Length; i++)
        {
            Assert.True(Math.Abs(after[i] - replacement[i]) < 1e-6f,
                $"{name} did not store parameter {i}: wrote {replacement[i]}, read back {after[i]}.");
        }
    }

    [Fact]
    public void CachedMultiHeadAttention_ExposesItsProjectionWeights()
    {
        var layer = new CachedMultiHeadAttention<float>(
            sequenceLength: 6, embeddingDimension: EmbeddingDimension, headCount: Heads);

        AssertParametersRoundTrip(layer, nameof(CachedMultiHeadAttention<float>));
    }

    [Fact]
    public void CachedGroupedQueryAttention_ExposesItsProjectionWeights()
    {
        // Two KV heads against four query heads: K and V are narrower than Q and O, so this also
        // pins that the registered surface covers the reduced projections rather than assuming
        // every weight is square.
        var layer = new CachedGroupedQueryAttention<float>(
            sequenceLength: 6, embeddingDimension: EmbeddingDimension, numHeads: Heads, numKVHeads: 2);

        AssertParametersRoundTrip(layer, nameof(CachedGroupedQueryAttention<float>));
    }

    [Fact]
    public void PagedCachedMultiHeadAttention_ExposesItsProjectionWeights()
    {
        var layer = new PagedCachedMultiHeadAttention<float>(
            sequenceLength: 6, embeddingDimension: EmbeddingDimension, headCount: Heads,
            useCausalMask: true);

        AssertParametersRoundTrip(layer, nameof(PagedCachedMultiHeadAttention<float>));
    }
}
