using AiDotNet.ComputerVision.OCR.Recognition;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision;

/// <summary>
/// Pins TrOCR's key/value-cached decoding step to the full causal decoder pass: decoding a sequence
/// one token at a time must give, at every position, what the parallel pass over the whole prefix gives.
/// </summary>
public class TrOCRIncrementalDecodingTests
{
    private static Tensor<double> Random(int[] shape, int seed)
    {
        var r = new Random(seed);
        var t = new Tensor<double>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = r.NextDouble() * 2 - 1;
        return t;
    }

    [Theory]
    [InlineData(1, 5, 7)]
    [InlineData(2, 4, 3)]
    public async Task ForwardStep_MatchesTheFullCausalPass(int batch, int length, int patches)
    {
        await Task.Yield();
        const int hidden = 16;
        var layer = new TrOCRDecoderLayer<double>(hidden, numHeads: 2);
        var x = Random(new[] { batch, length, hidden }, 1);
        var encoder = Random(new[] { batch, patches, hidden }, 2);

        var full = layer.Forward(x, encoder);

        var engine = AiDotNetEngine.Current;
        var cache = new TrOCRLayerCache<double>();
        for (int t = 0; t < length; t++)
        {
            var step = layer.ForwardStep(engine.TensorNarrow(x, 1, t, 1), encoder, cache);
            var expected = engine.TensorNarrow(full, 1, t, 1);
            Assert.Equal(expected.Length, step.Length);
            for (int i = 0; i < step.Length; i++)
            {
                Assert.Equal(expected[i], step[i], 10);
            }
        }
    }
}
