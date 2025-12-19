using System;
using AiDotNet.Tensors.Engines.Optimization;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Optimization;

public class CacheOptimizerTests
{
    [Fact]
    public void ComputeOptimalTiling_ReturnsPositiveTiles_WithinDimensions()
    {
        var (tileM, tileN, tileK) = CacheOptimizer.ComputeOptimalTiling(m: 128, n: 256, k: 64);

        Assert.InRange(tileM, 1, 128);
        Assert.InRange(tileN, 1, 256);
        Assert.InRange(tileK, 1, 64);
    }

    [Fact]
    public void TransposeBlocked_TransposesCorrectly()
    {
        const int rows = 3;
        const int cols = 4;

        var src = new float[rows * cols];
        for (int i = 0; i < src.Length; i++)
        {
            src[i] = i + 1;
        }

        var dst = new float[rows * cols];

        CacheOptimizer.TransposeBlocked(src, dst, rows, cols);

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                Assert.Equal(src[r * cols + c], dst[c * rows + r]);
            }
        }
    }

    [Fact]
    public void CopyWithPrefetch_CopiesAllElements()
    {
        var src = new float[] { 1f, 2f, 3f, 4f };
        var dst = new float[src.Length];

        CacheOptimizer.CopyWithPrefetch(src, dst, src.Length);

        Assert.Equal(src, dst);
    }

    [Fact]
    public void MortonEncodeDecode_RoundTrips()
    {
        const int x = 123;
        const int y = 456;

        int code = CacheOptimizer.MortonEncode(x, y);
        var (rx, ry) = CacheOptimizer.MortonDecode(code);

        Assert.Equal(x & 0x0000ffff, rx);
        Assert.Equal(y & 0x0000ffff, ry);
    }
}

