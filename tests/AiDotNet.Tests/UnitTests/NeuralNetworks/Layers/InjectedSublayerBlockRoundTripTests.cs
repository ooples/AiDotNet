using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// Decoder blocks that receive their attention (and, for the mixture-of-experts blocks, their feed-forward layer)
/// through the constructor must come back from a standalone save: same weights, same settings (the block's own and
/// every setting of the injected sublayers), and the same outputs. Each test also builds a negative control that
/// differs in one injected setting only, to show the comparison would catch a lost setting.
/// </summary>
public class InjectedSublayerBlockRoundTripTests
{
    private const int Hidden = 8;
    private const int SequenceCapacity = 8;
    private const int Tokens = 4;
    private const double Epsilon = 1e-4;

    /// <summary>
    /// A grouped-query attention with every persisted setting moved off its default: 2 query heads over 1 KV head,
    /// causal, RoPE with a non-default theta and a table longer than the sequence capacity.
    /// </summary>
    private static GroupedQueryAttentionLayer<double> Attention(
        bool causal = true, bool projectionBias = false, double softcap = 0.0)
    {
        var attention = new GroupedQueryAttentionLayer<double>(
            sequenceLength: SequenceCapacity, embeddingDimension: Hidden, numHeads: 2, numKVHeads: 1,
            useProjectionBias: projectionBias, attnLogitSoftcap: softcap, useCausalMask: causal);
        attention.ConfigurePositionalEncoding(PositionalEncodingType.Rotary, ropeTheta: 500000.0, maxSequenceLength: 16);
        return attention;
    }

    private static MoEFeedForwardLayer<double> Moe(int sharedFfnDim = 0) =>
        new MoEFeedForwardLayer<double>(Hidden, 16, numExperts: 3, topK: 2, new GELUActivation<double>(), sharedFfnDim);

    private static Tensor<double> Input()
    {
        var rng = RandomHelper.CreateSeededRandom(2311);
        var input = new Tensor<double>([1, Tokens, Hidden]);
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble() * 2 - 1;
        return input;
    }

    private static double MaxAbsDifference(Tensor<double> a, Tensor<double> b)
    {
        Assert.Equal(a.Shape.ToArray(), b.Shape.ToArray());
        double worst = 0.0;
        for (int i = 0; i < a.Length; i++) worst = Math.Max(worst, Math.Abs(a[i] - b[i]));
        return worst;
    }

    /// <summary>
    /// Saves <paramref name="block"/> through its metadata and flat parameter vector, rebuilds it through
    /// <see cref="DeserializationHelper.CreateLayerFromType{T}"/>, and checks weights, every persisted setting and the
    /// outputs. Then loads the same weights into <paramref name="control"/>, which differs in one injected setting,
    /// and requires both its metadata and its outputs to differ, so the checks above could have failed.
    /// </summary>
    private static TBlock AssertRoundTrip<TBlock>(TBlock block, TBlock control)
        where TBlock : LayerBase<double>
    {
        var x = Input();
        block.SetTrainingMode(false);
        var expected = block.Forward(x);
        var weights = block.GetParameters();
        var metadata = block.GetMetadata();

        var restored = Assert.IsType<TBlock>(DeserializationHelper.CreateLayerFromType<double>(
            typeof(TBlock).Name, block.GetInputShape(), block.GetOutputShape(),
            metadata.ToDictionary(pair => pair.Key, pair => (object)pair.Value)));
        restored.SetTrainingMode(false);
        _ = restored.Forward(x);
        restored.SetParameters(weights);

        Assert.Equal(weights.ToArray(), restored.GetParameters().ToArray());
        Assert.Equal(
            metadata.OrderBy(pair => pair.Key, StringComparer.Ordinal).ToList(),
            restored.GetMetadata().OrderBy(pair => pair.Key, StringComparer.Ordinal).ToList());
        Assert.True(MaxAbsDifference(expected, restored.Forward(x)) < 1e-12,
            "The restored block computes a different function from the same weights.");

        control.SetTrainingMode(false);
        _ = control.Forward(x);
        control.SetParameters(weights);
        Assert.NotEqual(
            metadata.OrderBy(pair => pair.Key, StringComparer.Ordinal).ToList(),
            control.GetMetadata().OrderBy(pair => pair.Key, StringComparer.Ordinal).ToList());
        Assert.True(MaxAbsDifference(expected, control.Forward(x)) > 1e-9,
            "The negative control matched, so the output comparison could not detect a lost setting.");

        return restored;
    }

    private static GroupedQueryAttentionLayer<double> RestoredAttention(LayerBase<double> attention)
    {
        var gqa = Assert.IsType<GroupedQueryAttentionLayer<double>>(attention);
        Assert.Equal(2, gqa.NumHeads);
        Assert.Equal(1, gqa.NumKVHeads);
        Assert.Equal(PositionalEncodingType.Rotary, gqa.PositionalEncoding);
        Assert.Equal(500000.0, gqa.RoPETheta);
        return gqa;
    }

    private static void AssertRestoredMoe(MoEFeedForwardLayer<double> moe, int sharedFfnDim)
    {
        Assert.Equal(3, moe.NumExperts);
        Assert.Equal(2, moe.TopK);
        Assert.Equal(sharedFfnDim > 0, moe.HasSharedExpert);
        Assert.IsType<GELUActivation<double>>(moe.ExpertGate(0).ScalarActivation);
    }

    /// <summary>
    /// CohereDecoderBlock round-trips with its injected causal RoPE attention and LayerNorm epsilon. Fails if the
    /// block stops writing its metadata, the attention sub-blob, or the epsilon, or the deserializer branch is removed.
    /// </summary>
    [Fact]
    public void CohereDecoderBlock_RoundTrips()
    {
        var restored = AssertRoundTrip(
            new CohereDecoderBlock<double>(Hidden, 16, Attention(), Epsilon),
            new CohereDecoderBlock<double>(Hidden, 16, Attention(causal: false), Epsilon));
        RestoredAttention(restored.AttentionLayer);
        Assert.Equal(Hidden, restored.HiddenSize);
    }

    /// <summary>
    /// StarCoder2DecoderBlock round-trips with an attention that has Q/K/V projection biases. Fails if the
    /// projection-bias setting or any other injected setting is lost.
    /// </summary>
    [Fact]
    public void StarCoder2DecoderBlock_RoundTrips()
    {
        var restored = AssertRoundTrip(
            new StarCoder2DecoderBlock<double>(Hidden, 16, Attention(projectionBias: true), Epsilon),
            new StarCoder2DecoderBlock<double>(Hidden, 16, Attention(causal: false, projectionBias: true), Epsilon));
        Assert.True(RestoredAttention(restored.AttentionLayer).UsesProjectionBias);
    }

    /// <summary>
    /// Gemma2DecoderBlock round-trips with an attention-logit soft-cap and its RMSNorm epsilon. Fails if the soft-cap
    /// or the epsilon is lost.
    /// </summary>
    [Fact]
    public void Gemma2DecoderBlock_RoundTrips()
    {
        var restored = AssertRoundTrip(
            new Gemma2DecoderBlock<double>(Hidden, 16, Attention(softcap: 30.0), Epsilon),
            new Gemma2DecoderBlock<double>(Hidden, 16, Attention(causal: false, softcap: 30.0), Epsilon));
        Assert.Equal(30.0, RestoredAttention(restored.AttentionLayer).AttnLogitSoftcap);
    }

    /// <summary>
    /// DbrxDecoderBlock round-trips with both injected sublayers: the attention and the mixture-of-experts FFN (its
    /// expert count, top-k and gate activation). Fails if either sub-blob or the MoE layer's own metadata is lost.
    /// </summary>
    [Fact]
    public void DbrxDecoderBlock_RoundTrips()
    {
        var restored = AssertRoundTrip(
            new DbrxDecoderBlock<double>(Hidden, Attention(), Moe(), Epsilon),
            new DbrxDecoderBlock<double>(Hidden, Attention(causal: false), Moe(), Epsilon));
        RestoredAttention(restored.AttentionLayer);
        AssertRestoredMoe(restored.Moe, sharedFfnDim: 0);
    }

    /// <summary>
    /// MoEDecoderBlock round-trips with a mixture-of-experts FFN that has a shared expert (the Qwen2-MoE layout).
    /// Fails if the shared-expert width is lost, which would also change the parameter count.
    /// </summary>
    [Fact]
    public void MoEDecoderBlock_RoundTrips()
    {
        var restored = AssertRoundTrip(
            new MoEDecoderBlock<double>(Hidden, Attention(), Moe(sharedFfnDim: 12), Epsilon),
            new MoEDecoderBlock<double>(Hidden, Attention(causal: false), Moe(sharedFfnDim: 12), Epsilon));
        RestoredAttention(restored.AttentionLayer);
        AssertRestoredMoe(restored.Moe, sharedFfnDim: 12);
    }

    /// <summary>
    /// PreLNTransformerBlock now writes its attention through the shared helper; its round trip must be unchanged.
    /// Fails if the shared writer and reader disagree on the sub-blob keys.
    /// </summary>
    [Fact]
    public void PreLNTransformerBlock_RoundTrips()
    {
        var restored = AssertRoundTrip(
            new PreLNTransformerBlock<double>(Hidden, 16, Attention(), new SiLUActivation<double>(), gated: true),
            new PreLNTransformerBlock<double>(Hidden, 16, Attention(causal: false), new SiLUActivation<double>(), gated: true));
        RestoredAttention(restored.AttentionLayer);
    }

    /// <summary>A block whose metadata has lost its attention sub-blob is refused rather than rebuilt without it.</summary>
    [Fact]
    public void MissingAttentionSubBlob_IsRejected()
    {
        var block = new CohereDecoderBlock<double>(Hidden, 16, Attention(), Epsilon);
        var metadata = block.GetMetadata()
            .Where(pair => !pair.Key.StartsWith("Attn.", StringComparison.Ordinal))
            .ToDictionary(pair => pair.Key, pair => (object)pair.Value);

        Assert.Throws<InvalidOperationException>(() => DeserializationHelper.CreateLayerFromType<double>(
            typeof(CohereDecoderBlock<double>).Name, block.GetInputShape(), block.GetOutputShape(), metadata));
    }
}
