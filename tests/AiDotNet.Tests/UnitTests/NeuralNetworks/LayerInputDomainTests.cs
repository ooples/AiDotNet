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
/// The layer was RIGHT to reject it, and the fix must not make it lenient. EmbeddingLayer resolves
/// Indices-vs-Continuous from the input SHAPE alone, on purpose: inferring the mode from VALUES
/// made the layer's output RANK depend on the data and left shape contracts unanalysable. So the
/// tests below pin the complementary direction -- the layer DECLARES what it accepts, and callers
/// generate data to match -- and explicitly pin that no value inspection was reintroduced.
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
    public void Embedding_InIndicesMode_DeclaresTheVocabularyRange()
    {
        var layer = NewEmbedding();
        layer.InputMode = EmbeddingInputMode.Indices;

        var domain = layer.GetInputDomain(null);

        Assert.True(domain.IsIndices);
        Assert.Equal(0, domain.MinInclusive);
        Assert.Equal(VocabularySize, domain.MaxExclusive);
    }

    [Fact]
    public void Embedding_InContinuousMode_DeclaresContinuous()
    {
        var layer = NewEmbedding();
        layer.InputMode = EmbeddingInputMode.Continuous;

        Assert.False(layer.GetInputDomain(null).IsIndices);
    }

    /// <summary>
    /// An unresolved Auto layer must declare Indices, because that is what it will ENFORCE.
    /// Declaring continuous here would hand a token model float noise and reproduce the original
    /// 1,041 failures exactly.
    /// </summary>
    [Fact]
    public void Embedding_InAutoMode_WithNoInputSeenYet_DeclaresIndices()
    {
        var layer = NewEmbedding();

        Assert.Equal(EmbeddingInputMode.Auto, layer.InputMode);
        Assert.True(layer.GetInputDomain(null).IsIndices);
        Assert.Equal(VocabularySize, layer.GetInputDomain(null).MaxExclusive);
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
    /// THE DECLARATION MUST AGREE WITH THE FORWARD PASS, resolved against the shape the caller will
    /// actually feed -- not the layer's own InputShape field.
    /// </summary>
    /// <remarks>
    /// EmbeddingLayer constructs as <c>base([1], [embeddingDimension])</c>, so its InputShape is a
    /// placeholder until the shape system resolves it. An earlier version of this feature read that
    /// field and therefore answered Indices for EVERY Auto layer, including a genuine continuous
    /// projection whose real input is <c>[B, V]</c> with V == vocabulary. That disagreed with what
    /// Forward would do with the same tensor, which is precisely the drift this pins shut.
    /// </remarks>
    [Fact]
    public void Embedding_InAutoMode_AgreesWithForward_ForContinuousByShapeInput()
    {
        var layer = NewEmbedding();

        // Last axis == vocabulary size is the shape rule Forward uses to pick continuous mode.
        int[] continuousShape = [2, VocabularySize];

        Assert.False(layer.GetInputDomain(continuousShape).IsIndices);

        // And Forward agrees: continuous values at this shape are projected, not rejected.
        var rng = new Random(20260810);
        var input = new Tensor<double>(continuousShape);
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble();

        var output = layer.Forward(input);
        Assert.Equal(EmbeddingDimension, output.Shape[output.Rank - 1]);
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
        layer.InputMode = EmbeddingInputMode.Indices;
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
    public void ContinuousNoise_IsStillRejected_InIndicesMode()
    {
        var layer = NewEmbedding();
        layer.InputMode = EmbeddingInputMode.Indices;

        var rng = new Random(20260810);
        var input = new Tensor<double>(new[] { 1, 6 });
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble();

        var ex = Assert.Throws<ArgumentException>(() => layer.Forward(input));
        Assert.Contains("not a token index", ex.Message);
    }
}
