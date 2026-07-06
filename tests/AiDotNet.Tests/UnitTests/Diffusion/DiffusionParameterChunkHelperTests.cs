using System;
using System.Collections.Generic;
using AiDotNet.Diffusion;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion;

public class DiffusionParameterChunkHelperTests
{
    [Fact]
    public void BufferToFlatVector_CopiesChunksInStreamOrder()
    {
        var first = new Tensor<double>([2, 2]);
        var second = new Tensor<double>([3]);

        Fill(first, 1.0);
        Fill(second, 5.0);

        var flat = DiffusionParameterChunkHelper.BufferToFlatVector([first, second]);

        Assert.Equal(7, flat.Length);
        Assert.Equal([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], flat.AsSpan().ToArray());
    }

    [Fact]
    public void BufferToFlatVector_NullChunk_ThrowsArgumentException()
    {
        IEnumerable<Tensor<double>> chunks = [new Tensor<double>([1]), null!];

        var ex = Assert.Throws<ArgumentException>(() => DiffusionParameterChunkHelper.BufferToFlatVector(chunks));

        Assert.Contains("null tensor", ex.Message);
    }

    private static void Fill(Tensor<double> tensor, double start)
    {
        var span = tensor.AsWritableSpan();
        for (int i = 0; i < span.Length; i++)
            span[i] = start + i;
    }
}
