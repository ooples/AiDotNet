using System;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Regression tests for the input-domain declaration that lets callers build conforming input
/// without inspecting a layer's type.
/// </summary>
/// <remarks>
/// <para>
/// WHAT THESE BLOCK. Run 31356312540 had 1,041 failing assertions -- 41% of every failure in the
/// run -- all reading "EmbeddingLayer is in Indices mode but element 0 is 0.668..., which is not a
/// token index in [0, 128)". The generic model-family fixture filled every input with
/// <c>rng.NextDouble()</c>, so a model whose first layer is an embedding in lookup mode was handed
/// continuous noise.
/// </para>
/// <para>
/// The layer was right to reject it, and the fix must not make it lenient. EmbeddingLayer is now an
/// index-only lookup; DenseLayer is the explicit continuous projection. The tests below pin that
/// type-level split so parameter sets, domains, and output ranks cannot depend on input data.
/// </para>
/// </remarks>
public class LayerInputDomainTests
{
    private const int VocabularySize = 128;
    private const int EmbeddingDimension = 8;

    private static EmbeddingLayer<double> NewEmbedding() =>
        new EmbeddingLayer<double>(VocabularySize, EmbeddingDimension);

    [Fact]
    public void Continuous_IsTheDefault_ForAnOrdinaryLayer()
    {
        var dense = new DenseLayer<double>(3);

        Assert.False(dense.GetInputDomain(null).IsIndices);
        Assert.Equal(LayerInputDomainKind.Continuous, dense.GetInputDomain(null).Kind);
    }

    [Fact]
    public async Task Embedding_DeclaresTheVocabularyRange()
    {
        await Task.Yield();

        var layer = NewEmbedding();

        var domain = layer.GetInputDomain(null);

        Assert.True(domain.IsIndices);
        Assert.Equal(0, domain.MinInclusive);
        Assert.Equal(VocabularySize, domain.MaxExclusive);
    }

    [Fact]
    public async Task DenseLayer_IsTheExplicitContinuousProjection()
    {
        await Task.Yield();

        var projection = new DenseLayer<double>(EmbeddingDimension);

        Assert.Equal(LayerInputDomainKind.Continuous, projection.GetInputDomain(null).Kind);
    }

    [Fact]
    public async Task Embedding_AlwaysDeclaresIndices_EvenWhenLastAxisEqualsVocabularySize()
    {
        await Task.Yield();

        var layer = NewEmbedding();
        int[] formerlyAmbiguousShape = [2, VocabularySize];

        Assert.True(layer.GetInputDomain(formerlyAmbiguousShape).IsIndices);
        Assert.Equal(VocabularySize, layer.GetInputDomain(formerlyAmbiguousShape).MaxExclusive);
    }

    /// <summary>
    /// THE INVARIANT THE REFACTOR EXISTS TO PROTECT. The declaration must be a function of shape
    /// and configuration only. If it ever consulted the tensor's values, the layer's output rank
    /// would become data-dependent again -- the exact defect that value-sniffing was removed for.
    /// </summary>
    [Fact]
    public void Embedding_Domain_DoesNotDependOnValues()
    {
        var layer = NewEmbedding();

        var beforeAnyData = layer.GetInputDomain(null);

        // Feed legal indices; the declaration must not drift.
        var indices = new Tensor<double>(new[] { 1, 4 });
        for (int i = 0; i < indices.Length; i++) indices[i] = i;
        layer.Forward(indices);

        var afterIndices = layer.GetInputDomain(null);

        Assert.Equal(beforeAnyData.Kind, afterIndices.Kind);
        Assert.Equal(beforeAnyData.MaxExclusive, afterIndices.MaxExclusive);
    }

    /// <summary>
    /// A shape that previously activated Auto projection must now be rejected if it contains
    /// continuous values. Shape alone cannot change the operation represented by the layer type.
    /// </summary>
    [Fact]
    public async Task Embedding_DoesNotProjectContinuousValues_WhenLastAxisEqualsVocabularySize()
    {
        await Task.Yield();

        var layer = NewEmbedding();
        int[] continuousShape = [2, VocabularySize];
        var rng = new Random(20260810);
        var input = new Tensor<double>(continuousShape);
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble();

        var ex = Assert.Throws<ArgumentException>(() => layer.Forward(input));
        Assert.Contains("requires token indices", ex.Message);
    }

    [Fact]
    public void Indices_WithNonPositiveVocabulary_FallsBackToContinuous()
    {
        // A degenerate empty range would make EVERY value illegal rather than every value allowed,
        // which is the wrong failure for a layer that simply is not sized yet.
        Assert.False(LayerInputDomain.Indices(0).IsIndices);
        Assert.False(LayerInputDomain.Indices(-3).IsIndices);
    }

    /// <summary>
    /// End-to-end: values generated from the declared domain must survive the layer's own
    /// validator. This is the assertion that would have caught the original 41% outright.
    /// </summary>
    [Fact]
    public void ValuesGeneratedFromTheDeclaredDomain_AreAcceptedByTheLayer()
    {
        var layer = NewEmbedding();
        var domain = layer.GetInputDomain(null);

        var rng = new Random(20260810);
        var input = new Tensor<double>(new[] { 1, 6 });
        for (int i = 0; i < input.Length; i++)
            input[i] = rng.Next(domain.MinInclusive, domain.MaxExclusive);

        var output = layer.Forward(input);

        Assert.NotNull(output);
        Assert.Equal(EmbeddingDimension, output.Shape[output.Rank - 1]);
    }

    /// <summary>
    /// The negative half of the pair: continuous noise must STILL be rejected. If a future change
    /// makes the layer lenient, the shape contract silently degrades and this fails.
    /// </summary>
    [Fact]
    public async Task ContinuousNoise_IsRejected()
    {
        await Task.Yield();

        var layer = NewEmbedding();

        var rng = new Random(20260810);
        var input = new Tensor<double>(new[] { 1, 6 });
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble();

        var ex = Assert.Throws<ArgumentException>(() => layer.Forward(input));
        Assert.Contains("requires token indices", ex.Message);
    }
}
