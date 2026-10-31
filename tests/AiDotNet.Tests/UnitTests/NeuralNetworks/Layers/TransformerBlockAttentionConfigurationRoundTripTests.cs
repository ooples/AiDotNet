using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// A composite transformer block rebuilds its attention from its own constructor arguments, which never include the
/// settings configured on the attention afterwards: the causal mask and the attention-probability dropout. Each block
/// that hosts a <see cref="MultiHeadAttentionLayer{T}"/> must persist and restore them, or a causal block deserialized
/// on its own comes back bidirectional and computes a different function.
/// </summary>
public class TransformerBlockAttentionConfigurationRoundTripTests
{
    private const int Hidden = 16;
    private const int Heads = 4;
    private const int FfnDim = 32;
    private const int Frames = 5;

    private static Tensor<double> Input(int seed, int frames = Frames)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var input = new Tensor<double>([1, frames, Hidden]);
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble() * 2 - 1;
        return input;
    }

    private static Dictionary<string, object> Metadata(LayerBase<double> layer) =>
        layer.GetMetadata().ToDictionary(pair => pair.Key, pair => (object)pair.Value);

    private static double MaxAbsDifference(Tensor<double> a, Tensor<double> b)
    {
        Assert.Equal(a.Shape.ToArray(), b.Shape.ToArray());
        double worst = 0.0;
        for (int i = 0; i < a.Length; i++) worst = System.Math.Max(worst, System.Math.Abs(a[i] - b[i]));
        return worst;
    }

    /// <summary>
    /// TransformerEncoderBlock with a causal, attention-dropout attention survives a deserialize with both settings
    /// and the same eval-mode outputs; a bidirectional block with the same weights computes something else, so the
    /// output match is not vacuous. Fails if the block stops writing or the deserializer stops restoring either
    /// setting.
    /// </summary>
    [Fact]
    public void EncoderBlock_KeepsCausalMaskAndAttentionDropout()
    {
        static TransformerEncoderBlock<double> Build(bool causal)
        {
            var block = new TransformerEncoderBlock<double>(Hidden, Heads, FfnDim, dropoutRate: 0.1);
            var attention = Assert.IsType<MultiHeadAttentionLayer<double>>(block.AttentionLayer);
            attention.UseCausalMask = causal;
            attention.ConfigureAttentionDropout(0.2);
            block.SetTrainingMode(false);
            return block;
        }

        var block = Build(causal: true);
        var x = Input(4101);
        var expected = block.Forward(x);

        var restored = Assert.IsType<TransformerEncoderBlock<double>>(DeserializationHelper.CreateLayerFromType<double>(
            "TransformerEncoderBlock`1", block.GetInputShape(), block.GetOutputShape(), Metadata(block)));
        restored.SetTrainingMode(false);
        _ = restored.Forward(x);
        restored.SetParameters(block.GetParameters());

        var attention = Assert.IsType<MultiHeadAttentionLayer<double>>(restored.AttentionLayer);
        Assert.True(attention.UseCausalMask, "The restored block's attention is bidirectional.");
        Assert.Equal(0.2, attention.AttentionDropoutRate);
        Assert.True(MaxAbsDifference(expected, restored.Forward(x)) < 1e-12);

        var bidirectional = Build(causal: false);
        _ = bidirectional.Forward(x);
        bidirectional.SetParameters(block.GetParameters());
        Assert.True(MaxAbsDifference(expected, bidirectional.Forward(x)) > 1e-9,
            "Causal and bidirectional blocks agree, so the output comparison above proves nothing.");
    }

    /// <summary>
    /// TransformerDecoderBlock keeps its self-attention's causal mask and the attention dropout of both its
    /// self- and cross-attention through a deserialize, with the same eval-mode outputs; a bidirectional
    /// self-attention with the same weights differs. Fails if either attention's settings are lost.
    /// </summary>
    [Fact]
    public void DecoderBlock_KeepsCausalMaskAndAttentionDropoutOfBothAttentions()
    {
        static TransformerDecoderBlock<double> Build(bool causal)
        {
            var block = new TransformerDecoderBlock<double>(Hidden, Heads, FfnDim, dropoutRate: 0.1);
            var self = Assert.IsType<MultiHeadAttentionLayer<double>>(block.SelfAttentionLayer);
            self.UseCausalMask = causal;
            self.ConfigureAttentionDropout(0.2);
            block.CrossAttentionLayer.ConfigureAttentionDropout(0.3);
            block.SetTrainingMode(false);
            return block;
        }

        var block = Build(causal: true);
        var x = Input(4102);
        var memory = Input(4103, frames: 7);
        var expected = block.Forward(x, memory);

        var restored = Assert.IsType<TransformerDecoderBlock<double>>(DeserializationHelper.CreateLayerFromType<double>(
            "TransformerDecoderBlock`1", block.GetInputShape(), block.GetOutputShape(), Metadata(block)));
        restored.SetTrainingMode(false);
        _ = restored.Forward(x, memory);
        restored.SetParameters(block.GetParameters());

        var self = Assert.IsType<MultiHeadAttentionLayer<double>>(restored.SelfAttentionLayer);
        Assert.True(self.UseCausalMask, "The restored decoder's self-attention is bidirectional.");
        Assert.Equal(0.2, self.AttentionDropoutRate);
        Assert.False(restored.CrossAttentionLayer.UseCausalMask);
        Assert.Equal(0.3, restored.CrossAttentionLayer.AttentionDropoutRate);
        Assert.True(MaxAbsDifference(expected, restored.Forward(x, memory)) < 1e-12);

        var bidirectional = Build(causal: false);
        _ = bidirectional.Forward(x, memory);
        bidirectional.SetParameters(block.GetParameters());
        Assert.True(MaxAbsDifference(expected, bidirectional.Forward(x, memory)) > 1e-9,
            "Causal and bidirectional decoders agree, so the output comparison above proves nothing.");
    }

    /// <summary>
    /// Metadata written before these keys existed restores the constructor defaults (bidirectional, no attention
    /// dropout) rather than failing. Fails if the restore starts requiring the new keys.
    /// </summary>
    [Fact]
    public void BlocksSavedWithoutTheKeys_KeepConstructorDefaults()
    {
        var block = new TransformerEncoderBlock<double>(Hidden, Heads, FfnDim, dropoutRate: 0.1);
        var metadata = Metadata(block);
        foreach (var key in metadata.Keys.Where(k => k.StartsWith("Attn.", System.StringComparison.Ordinal)).ToList())
            metadata.Remove(key);

        var restored = Assert.IsType<TransformerEncoderBlock<double>>(DeserializationHelper.CreateLayerFromType<double>(
            "TransformerEncoderBlock`1", block.GetInputShape(), block.GetOutputShape(), metadata));
        var attention = Assert.IsType<MultiHeadAttentionLayer<double>>(restored.AttentionLayer);
        Assert.False(attention.UseCausalMask);
        Assert.Equal(0.0, attention.AttentionDropoutRate);
    }
}
