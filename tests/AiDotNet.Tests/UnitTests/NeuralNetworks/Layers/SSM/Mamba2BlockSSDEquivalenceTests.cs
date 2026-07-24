using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Tensors;
using Xunit;
using System;
using System.Threading.Tasks;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers.SSM;

/// <summary>
/// Verifies that the chunked semiseparable SSD path (<c>SSDForwardChunked</c>, driven by the configured
/// chunk size) is numerically equivalent to the sequential selective-scan reference
/// (<c>SSDForwardSequential</c>). Both are exposed via internal debug hooks. Because the chunked form
/// carries recurrent state between chunks, matching the reference for every chunk size — including sizes
/// that do not divide the sequence length — proves the chunk boundaries are handled correctly.
/// </summary>
public class Mamba2BlockSSDEquivalenceTests
{
    private static Tensor<double> Random(int[] shape, Random rng)
    {
        var t = new Tensor<double>(shape);
        var span = t.Data.Span;
        for (int i = 0; i < span.Length; i++) span[i] = rng.NextDouble() * 2.0 - 1.0;
        return t;
    }

    [Theory(Timeout = 120000)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    [InlineData(7)]
    [InlineData(13)]
    [InlineData(20)]
    public async Task ChunkedSSD_MatchesSequentialReference_ForEveryChunkSize(int chunkSize)
    {
        await Task.Yield();

        const int batch = 2;
        const int seqLen = 13;
        const int modelDim = 6;
        const int expandFactor = 2;   // innerDim = 12
        const int numHeads = 3;       // headDim = 4
        const int stateDim = 5;

        var block = new Mamba2Block<double>(seqLen, modelDim, stateDim, numHeads, expandFactor,
            convKernelSize: 4, chunkSize: chunkSize);

        int innerDim = block.InnerDimension;      // 12
        var rng = new Random(1234);

        var x = Random(new[] { batch, seqLen, innerDim }, rng);
        // dt must be positive (softplus output in the real forward); use magnitudes so decays stay in (0,1].
        var delta = Random(new[] { batch, seqLen, numHeads }, rng);
        var dSpan = delta.Data.Span;
        for (int i = 0; i < dSpan.Length; i++) dSpan[i] = Math.Abs(dSpan[i]) * 0.5;
        var b = Random(new[] { batch, seqLen, stateDim }, rng);
        var c = Random(new[] { batch, seqLen, stateDim }, rng);

        var reference = block.DebugSSDSequential(x, delta, b, c, batch, seqLen);
        var chunked = block.DebugSSDChunked(x, delta, b, c, batch, seqLen, chunkSize);

        Assert.Equal(reference.Length, chunked.Length);
        var r = reference.Data.Span;
        var g = chunked.Data.Span;
        double maxAbsErr = 0.0;
        for (int i = 0; i < r.Length; i++)
            maxAbsErr = Math.Max(maxAbsErr, Math.Abs(r[i] - g[i]));

        Assert.True(maxAbsErr < 1e-9,
            $"Chunked SSD diverged from the sequential reference at chunkSize={chunkSize} " +
            $"(max abs error {maxAbsErr:E3}).");
    }
}
