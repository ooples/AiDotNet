using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// Unit tests for the AiDotNet#1370 shape oracle: <see cref="LayerBase{T}.TryDeclareShape"/>
/// virtual + the high-value overrides on
/// <see cref="MultiHeadAttentionLayer{T}"/> and the eager-init
/// <see cref="LayerNormalizationLayer{T}"/> constructor.
/// </summary>
/// <remarks>
/// Joined the <c>LayerSerializationCollection</c> so these tests don't run in
/// parallel with the auto-generated <c>LayerTestBase.TapeGradient_ShouldMatchNumericalGradient</c>
/// tests. The MHA TryDeclareShape override allocates weights via <c>SimdRandom</c>
/// which uses a thread-local seed — concurrent execution with the generated
/// TapeGradient tests shifts their seed and produces flaky finite-difference
/// gradient comparisons. Serializing eliminates the interference without
/// changing layer behavior.
/// </remarks>
[Collection(global::AiDotNet.Tests.Fixtures.LayerSerializationCollection.Name)]
public class ShapeOracleIssue1370Tests
{
    /// <summary>
    /// Default <see cref="LayerBase{T}.TryDeclareShape"/> impl returns
    /// <see cref="LayerBase{T}.IsShapeResolved"/> verbatim. A
    /// <see cref="LayerNormalizationLayer{T}"/> constructed with the eager
    /// (featureSize) ctor is shape-resolved at ctor return, so the default
    /// returns true without any override.
    /// </summary>
    [Fact]
    public void TryDeclareShape_DefaultImpl_MatchesIsShapeResolved_EagerLayerNorm()
    {
        var layer = new LayerNormalizationLayer<float>(featureSize: 16);
        Assert.True(layer.IsShapeResolved);
        Assert.True(layer.TryDeclareShape());
    }

    /// <summary>
    /// Parameter-less <see cref="LayerNormalizationLayer{T}"/> ctor leaves the
    /// feature dim deferred until first forward — IsShapeResolved is false,
    /// so the default <see cref="LayerBase{T}.TryDeclareShape"/> impl
    /// (which mirrors IsShapeResolved) must also return false. This is the
    /// contract that lets <see cref="AiDotNet.AiModelBuilder{T, TInput, TOutput}"/>'s
    /// LoRA wrap path correctly fall back to the warmup forward when any
    /// layer can't self-declare.
    /// </summary>
    [Fact]
    public void TryDeclareShape_LazyLayerNormDefaultCtor_ReturnsFalse()
    {
        var layer = new LayerNormalizationLayer<float>();
        Assert.False(layer.IsShapeResolved);
        Assert.False(layer.TryDeclareShape());
    }

    /// <summary>
    /// Eager-init <see cref="LayerNormalizationLayer{T}"/> ctor (AiDotNet#1370):
    /// passing featureSize at construction allocates gamma/beta immediately and
    /// resolves the layer's input + output shapes. <see cref="LayerBase{T}.IsShapeResolved"/>
    /// is true at construction time, so the default <see cref="LayerBase{T}.TryDeclareShape"/>
    /// impl returns true without needing an override on the layer.
    /// </summary>
    [Theory]
    [InlineData(8)]
    [InlineData(64)]
    [InlineData(768)]
    [InlineData(4096)]
    public void TryDeclareShape_EagerLayerNormCtor_ReturnsTrueImmediately(int featureSize)
    {
        var layer = new LayerNormalizationLayer<float>(featureSize);
        Assert.True(layer.IsShapeResolved);
        Assert.True(layer.TryDeclareShape());

        // Gamma/beta materialised at the requested feature dim — verifies the
        // eager ctor actually allocated, not just set the shape metadata.
        var gamma = layer.GetGammaTensor();
        var beta = layer.GetBetaTensor();
        Assert.Equal(featureSize, gamma.Length);
        Assert.Equal(featureSize, beta.Length);
    }

    /// <summary>
    /// Eager-init <see cref="LayerNormalizationLayer{T}"/> rejects non-positive
    /// featureSize — wrap loop downstream would otherwise build a zero-sized
    /// gamma/beta tensor that explodes on first use.
    /// </summary>
    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    [InlineData(-128)]
    public void EagerLayerNormCtor_RejectsNonPositiveFeatureSize(int featureSize)
    {
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new LayerNormalizationLayer<float>(featureSize));
    }

    /// <summary>
    /// <see cref="MultiHeadAttentionLayer{T}.TryDeclareShape"/> override:
    /// the ctor takes <c>(headCount, headDimension)</c> so the embedding dim
    /// is fully known and the four projection weight matrices (Q/K/V/O) all
    /// have shape <c>[embed, embed]</c>. Calling TryDeclareShape must allocate
    /// the weights AND return true even though the input sequence length is
    /// still <c>-1</c> in InputShape.
    /// </summary>
    [Theory]
    [InlineData(8, 16)]  // small
    [InlineData(12, 64)] // BERT-base scale
    [InlineData(16, 64)] // GPT-2 scale
    public void MHA_TryDeclareShape_AllocatesWeights_AndReturnsTrue_BeforeAnyForward(int headCount, int headDimension)
    {
        var layer = new MultiHeadAttentionLayer<float>(
            headCount: headCount,
            headDimension: headDimension);

        // Before TryDeclareShape: weights are placeholder [0, 0], IsShapeResolved
        // is false (InputShape carries the -1 seq placeholder), and ParameterCount
        // reflects the un-allocated state.
        Assert.False(layer.IsShapeResolved);

        // The shape oracle call.
        bool declared = layer.TryDeclareShape();

        Assert.True(declared,
            "MHA must return true from TryDeclareShape — embedding dim is known from ctor.");

        // After the call: weights allocated to [embed, embed] = [headCount*headDimension, ...].
        int expectedEmbed = headCount * headDimension;
        var qWeights = layer.GetQueryWeights();
        Assert.Equal(expectedEmbed, qWeights.Shape[0]);
        Assert.Equal(expectedEmbed, qWeights.Shape[1]);

        // ParameterCount now reflects allocated state — 4 * embed*embed + embed (bias).
        long expectedParams = 4L * expectedEmbed * expectedEmbed + expectedEmbed;
        Assert.Equal(expectedParams, layer.ParameterCount);

        // Asymmetry: IsShapeResolved remains false (InputShape still has -1 seq),
        // but TryDeclareShape returned true. Both are correct under the AiDotNet#1370
        // contract — TryDeclareShape means "ready for shape-dependent post-processing
        // (e.g. LoRA wrap)", which only needs weight matrix shape.
        Assert.False(layer.IsShapeResolved);
    }

    /// <summary>
    /// <see cref="MultiHeadAttentionLayer{T}.TryDeclareShape"/> is idempotent:
    /// calling it twice (or after a real forward has already resolved shapes)
    /// must succeed without re-allocating weights.
    /// </summary>
    [Fact]
    public void MHA_TryDeclareShape_Idempotent()
    {
        var layer = new MultiHeadAttentionLayer<float>(headCount: 4, headDimension: 16);

        Assert.True(layer.TryDeclareShape());
        var firstQ = layer.GetQueryWeights();
        long firstParamCount = layer.ParameterCount;

        // Second call must not re-allocate or change anything.
        Assert.True(layer.TryDeclareShape());
        var secondQ = layer.GetQueryWeights();

        Assert.Same(firstQ, secondQ);
        Assert.Equal(firstParamCount, layer.ParameterCount);
    }

    /// <summary>
    /// Eager-init <see cref="BatchNormalizationLayer{T}"/> ctor (AiDotNet#1370):
    /// passing numFeatures at construction allocates gamma/beta/runningMean/runningVariance
    /// immediately and resolves the layer's input + output shapes.
    /// </summary>
    [Theory]
    [InlineData(8)]
    [InlineData(64)]
    [InlineData(2048)]
    public void TryDeclareShape_EagerBatchNormCtor_ReturnsTrueImmediately(int numFeatures)
    {
        var layer = new BatchNormalizationLayer<float>(numFeatures);
        Assert.True(layer.IsShapeResolved);
        Assert.True(layer.TryDeclareShape());

        // ParameterCount = 2 * numFeatures (gamma + beta, running stats not trainable).
        Assert.Equal(2L * numFeatures, layer.ParameterCount);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    [InlineData(-128)]
    public void EagerBatchNormCtor_RejectsNonPositiveNumFeatures(int numFeatures)
    {
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new BatchNormalizationLayer<float>(numFeatures));
    }

    /// <summary>
    /// Eager-init <see cref="RMSNormalizationLayer{T}"/> ctor (AiDotNet#1370):
    /// passing featureSize at construction allocates gamma immediately.
    /// </summary>
    [Theory]
    [InlineData(8)]
    [InlineData(4096)]
    public void TryDeclareShape_EagerRMSNormCtor_ReturnsTrueImmediately(int featureSize)
    {
        var layer = new RMSNormalizationLayer<float>(featureSize);
        Assert.True(layer.IsShapeResolved);
        Assert.True(layer.TryDeclareShape());

        // ParameterCount = featureSize (gamma only; RMSNorm has no beta).
        Assert.Equal((long)featureSize, layer.ParameterCount);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void EagerRMSNormCtor_RejectsNonPositiveFeatureSize(int featureSize)
    {
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new RMSNormalizationLayer<float>(featureSize));
    }

    /// <summary>
    /// <see cref="PReLULayer{T}.TryDeclareShape"/> override: α weight tensor is
    /// fully allocated and registered in the constructor (line 98–103 of the
    /// layer). The lazy bit is only the broadcast shape, which is a
    /// forward-runtime concern. LoRA needs only the weight matrix shape, so
    /// TryDeclareShape returns true unconditionally.
    /// </summary>
    [Theory]
    [InlineData(1)]    // shared α
    [InlineData(64)]   // per-channel α
    [InlineData(256)]
    public void PReLU_TryDeclareShape_ReturnsTrue_BeforeAnyForward(int numParameters)
    {
        var layer = new PReLULayer<float>(numParameters: numParameters);

        // IsShapeResolved is false (InputShape still has -1 placeholder), but
        // TryDeclareShape returns true because α is already allocated.
        Assert.False(layer.IsShapeResolved);
        Assert.True(layer.TryDeclareShape());

        // ParameterCount = numParameters (α tensor).
        Assert.Equal((long)numParameters, layer.ParameterCount);
    }

    /// <summary>
    /// <see cref="TransformerEncoderLayer{T}.TryDeclareShape"/> override:
    /// the eager-dimension ctor (passing <c>embeddingSize &gt; 0</c>)
    /// constructs sublayers at ctor time. TryDeclareShape returns true
    /// even though InputShape still has <c>[-1, -1, -1]</c> (sequence +
    /// batch genuinely dynamic).
    /// </summary>
    [Theory]
    [InlineData(2, 32, 16)]
    [InlineData(4, 128, 64)]
    [InlineData(8, 1024, 256)]
    public void TransformerEncoder_TryDeclareShape_EagerCtor_ReturnsTrue(int numHeads, int feedForwardDim, int embeddingSize)
    {
        var layer = new TransformerEncoderLayer<float>(numHeads, feedForwardDim, embeddingSize);

        // Sublayers (MHA + LayerNorm + FFN) constructed at ctor — TryDeclareShape true.
        Assert.True(layer.TryDeclareShape());

        // ParameterCount must reflect the eagerly-constructed sublayers' parameters,
        // not zero (the previous lazy-ctor behavior).
        Assert.True(layer.ParameterCount > 0,
            $"Eager-ctor TransformerEncoderLayer must report ParameterCount > 0 at construction; got {layer.ParameterCount}.");
    }

    /// <summary>
    /// Lazy-ctor TransformerEncoderLayer (no explicit embeddingSize) defers
    /// sublayer construction to first forward → TryDeclareShape returns false →
    /// AiModelBuilder falls back to the warmup forward, as before.
    /// </summary>
    [Fact]
    public void TransformerEncoder_TryDeclareShape_LazyCtor_ReturnsFalse()
    {
        var layer = new TransformerEncoderLayer<float>(numHeads: 4, feedForwardDim: 128);
        Assert.False(layer.IsShapeResolved);
        Assert.False(layer.TryDeclareShape());
    }
}
