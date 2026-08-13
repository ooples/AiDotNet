using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.UnitTests.Layers;

/// <summary>
/// Unit tests for <see cref="LayoutEmbeddingLayer{T}"/>, the text + 2D-layout embedding block from
/// LayoutLM (Xu et al., KDD 2020).
/// </summary>
/// <remarks>
/// The test that matters most here is <c>BoundingBoxes_ChangeTheOutput</c>. Before this layer
/// existed the layout tables were model fields that were allocated, counted as parameters,
/// serialized and stepped by the optimizer while no forward pass ever read them — so every
/// assertion about parameter counts passed and the model still ignored the page completely.
/// A count check cannot distinguish that state from a working one; only feeding two different
/// boxes and demanding two different answers can.
/// </remarks>
public class LayoutEmbeddingLayerTests
{
    private const int VocabSize = 64;
    private const int HiddenDim = 16;
    private const int MaxSeq = 32;
    private const int MaxPos2D = 64;

    private static LayoutEmbeddingLayer<double> CreateLayer()
        => new(VocabSize, HiddenDim, MaxSeq, MaxPos2D);

    private static Tensor<double> Tokens(params double[] ids)
        => new([ids.Length], new Vector<double>(ids));

    /// <summary>Builds a packed [seq, 5] tensor: (tokenId, x0, y0, x1, y1) per row.</summary>
    private static Tensor<double> Packed(double[] ids, double[][] boxes)
    {
        var data = new Vector<double>(ids.Length * LayoutEmbeddingLayer<double>.PackedRowWidth);
        for (int i = 0; i < ids.Length; i++)
        {
            int b = i * LayoutEmbeddingLayer<double>.PackedRowWidth;
            data[b] = ids[i];
            data[b + 1] = boxes[i][0];
            data[b + 2] = boxes[i][1];
            data[b + 3] = boxes[i][2];
            data[b + 4] = boxes[i][3];
        }

        return new Tensor<double>([ids.Length, LayoutEmbeddingLayer<double>.PackedRowWidth], data);
    }

    [Fact(Timeout = 60000)]
    public async Task TokensOnly_ProducesOneEmbeddingPerToken()
    {
        await Task.Yield();
        var layer = CreateLayer();

        var output = layer.Forward(Tokens(1, 2, 3, 4, 5));

        Assert.Equal(2, output.Rank);
        Assert.Equal(5, output.Shape[0]);
        Assert.Equal(HiddenDim, output.Shape[1]);
    }

    [Fact(Timeout = 60000)]
    public async Task PackedInput_ProducesOneEmbeddingPerToken()
    {
        await Task.Yield();
        var layer = CreateLayer();

        var output = layer.Forward(Packed(
            [1, 2, 3],
            [[0, 0, 10, 10], [10, 0, 20, 10], [20, 0, 30, 10]]));

        Assert.Equal(2, output.Rank);
        Assert.Equal(3, output.Shape[0]);
        Assert.Equal(HiddenDim, output.Shape[1]);
    }

    [Fact(Timeout = 60000)]
    public async Task BoundingBoxes_ChangeTheOutput()
    {
        await Task.Yield();
        var layer = CreateLayer();
        double[] ids = [7, 8, 9];

        // Identical tokens in identical reading order; only the page positions differ.
        var topLeft = layer.Forward(Packed(ids,
            [[0, 0, 5, 5], [5, 0, 10, 5], [10, 0, 15, 5]]));
        var bottomRight = layer.Forward(Packed(ids,
            [[40, 40, 45, 45], [45, 40, 50, 45], [50, 40, 55, 45]]));

        bool differs = false;
        for (int i = 0; i < topLeft.Length && !differs; i++)
        {
            if (System.Math.Abs(topLeft.Data.Span[i] - bottomRight.Data.Span[i]) > 1e-12)
                differs = true;
        }

        Assert.True(differs,
            "Same tokens at different page positions produced identical embeddings, which means the " +
            "2D layout tables are not reaching the forward pass — the exact defect this layer exists " +
            "to fix.");
    }

    [Fact(Timeout = 60000)]
    public async Task TokensOnly_IgnoresLayoutTables()
    {
        await Task.Yield();
        var layer = CreateLayer();
        double[] ids = [7, 8, 9];

        // A tokens-only call must not silently pick up an index-0 layout embedding: that would add
        // the same learned vector to every token and mean "no boxes were given".
        var viaRank1 = layer.Forward(Tokens(ids));
        var viaPackedZeroBoxes = layer.Forward(Packed(ids,
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]));

        bool differs = false;
        for (int i = 0; i < viaRank1.Length && !differs; i++)
        {
            if (System.Math.Abs(viaRank1.Data.Span[i] - viaPackedZeroBoxes.Data.Span[i]) > 1e-12)
                differs = true;
        }

        Assert.True(differs,
            "A packed call with all-zero boxes produced the same result as a tokens-only call, so the " +
            "layout tables contributed nothing even when boxes were supplied.");
    }

    [Fact(Timeout = 60000)]
    public async Task ParameterCount_CoversEveryTable()
    {
        await Task.Yield();
        var layer = CreateLayer();

        // The sub-layers allocate lazily, so measure AFTER a forward — a count taken before the
        // tables exist reports zero and would pass a naive "count matches vector" assertion.
        layer.Forward(Packed([1, 2], [[0, 0, 1, 1], [1, 1, 2, 2]]));

        long expected =
            (long)VocabSize * HiddenDim      // word
            + (long)MaxSeq * HiddenDim       // learned 1D position
            + 4L * MaxPos2D * HiddenDim;     // x, y, width, height

        Assert.Equal(expected, layer.ParameterCount);
        Assert.Equal(expected, layer.GetParameters().Length);
    }

    [Fact(Timeout = 60000)]
    public async Task OutOfRangeCoordinates_SaturateInsteadOfThrowing()
    {
        await Task.Yield();
        var layer = CreateLayer();

        // OCR boxes arrive in page pixels and can exceed the table, and negatives show up when a
        // detector's box runs off the edge. Neither is the caller's error to debug.
        var output = layer.Forward(Packed(
            [1, 2],
            [[-40, -40, 99999, 99999], [0, 0, MaxPos2D * 4, MaxPos2D * 4]]));

        Assert.Equal(HiddenDim, output.Shape[1]);
        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output.Data.Span[i]));
        }
    }
}
