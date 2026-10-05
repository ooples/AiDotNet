using System.Collections.Generic;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// TransformerEncoderBlock places its LayerNorms before each sublayer (Pre-LN, the default) or after each
/// residual sum (Post-LN, the original Transformer and wav2vec 2.0 BASE). The two compute different
/// functions, so each is checked against its formula built from the block's own sublayers, and the
/// placement must survive a deserialize.
/// </summary>
public class TransformerEncoderBlockNormPlacementTests
{
    private const int Hidden = 16;
    private const int Frames = 5;

    private static Tensor<double> Input(int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var input = new Tensor<double>([1, Frames, Hidden]);
        for (int i = 0; i < input.Length; i++) input[i] = rng.NextDouble() * 2 - 1;
        return input;
    }

    private static TransformerEncoderBlock<double> Block(TransformerNormPlacement placement)
    {
        var block = new TransformerEncoderBlock<double>(
            Hidden, 4, 32, 0.0, new GELUActivation<double>(), placement);
        block.SetTrainingMode(false);
        return block;
    }

    // The block's feed-forward, run on [frames, hidden] rows as the block runs it.
    private static Tensor<double> FeedForward(TransformerEncoderBlock<double> block, Tensor<double> x)
    {
        var rows = new Tensor<double>([Frames, Hidden]);
        for (int i = 0; i < rows.Length; i++) rows[i] = x[i];
        var down = block.FfnDownLayer.Forward(block.FfnUpLayer.Forward(rows));
        var result = new Tensor<double>([1, Frames, Hidden]);
        for (int i = 0; i < result.Length; i++) result[i] = down[i];
        return result;
    }

    private static Tensor<double> Add(Tensor<double> a, Tensor<double> b)
    {
        var sum = new Tensor<double>(a.Shape.ToArray());
        for (int i = 0; i < sum.Length; i++) sum[i] = a[i] + b[i];
        return sum;
    }

    // A fresh LayerNorm has gamma = 1 and beta = 0, as the block's untrained norms do.
    private static Tensor<double> Norm(Tensor<double> x) => new LayerNormalizationLayer<double>(Hidden).Forward(x);

    private static void AssertClose(Tensor<double> expected, Tensor<double> actual)
    {
        Assert.Equal(expected.Shape.ToArray(), actual.Shape.ToArray());
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i], 10);
    }

    [Fact]
    public void PostNorm_NormalizesEachResidualSum()
    {
        var block = Block(TransformerNormPlacement.PostNorm);
        var x = Input(2289);

        var afterAttention = Norm(Add(x, block.AttentionLayer.Forward(x)));
        var expected = Norm(Add(afterAttention, FeedForward(block, afterAttention)));

        AssertClose(expected, block.Forward(x));
    }

    [Fact]
    public void PreNorm_IsTheDefault_AndNormalizesEachSublayerInput()
    {
        var block = new TransformerEncoderBlock<double>(Hidden, 4, 32, 0.0, new GELUActivation<double>());
        block.SetTrainingMode(false);
        Assert.Equal(TransformerNormPlacement.PreNorm, block.NormPlacement);
        var x = Input(2290);

        var afterAttention = Add(x, block.AttentionLayer.Forward(Norm(x)));
        var expected = Add(afterAttention, FeedForward(block, Norm(afterAttention)));

        AssertClose(expected, block.Forward(x));
    }

    [Fact]
    public void Deserialize_KeepsThePlacement_AndTheFunction()
    {
        var block = Block(TransformerNormPlacement.PostNorm);
        var x = Input(2291);
        var expected = block.Forward(x);
        var metadata = block.GetMetadata().ToDictionary(pair => pair.Key, pair => (object)pair.Value);

        var restored = Assert.IsType<TransformerEncoderBlock<double>>(DeserializationHelper.CreateLayerFromType<double>(
            "TransformerEncoderBlock`1", block.GetInputShape(), block.GetOutputShape(), metadata));
        restored.SetTrainingMode(false);
        _ = restored.Forward(x);
        restored.SetParameters(block.GetParameters());

        Assert.Equal(TransformerNormPlacement.PostNorm, restored.NormPlacement);
        AssertClose(expected, restored.Forward(x));
    }

    [Fact]
    public void Deserialize_RefusesAnUnknownPlacement()
    {
        var metadata = Block(TransformerNormPlacement.PostNorm).GetMetadata()
            .ToDictionary(pair => pair.Key, pair => (object)pair.Value);
        metadata["NormPlacement"] = "Sandwich";

        Assert.Throws<System.InvalidOperationException>(() => DeserializationHelper.CreateLayerFromType<double>(
            "TransformerEncoderBlock`1", new[] { Hidden }, new[] { Hidden }, metadata));
    }
}
