using AiDotNet.Tensors.Engines.Optimization;
using Xunit;

namespace AiDotNet.Tests.InferenceOptimization;

public class CacheOptimizerTests
{
    [Fact]
    public void TransposeBlocked_Transposes2DMatrix()
    {
        // 2x3
        float[] src = new float[] { 1f, 2f, 3f, 4f, 5f, 6f };
        float[] dst = new float[src.Length];

        CacheOptimizer.TransposeBlocked(src, dst, rows: 2, cols: 3);

        // 3x2 (row-major): [ [1,4], [2,5], [3,6] ]
        float[] expected = new float[] { 1f, 4f, 2f, 5f, 3f, 6f };
        Assert.Equal(expected, dst);
    }

    [Fact]
    public void CopyWithPrefetch_CopiesPrefix()
    {
        float[] src = new float[] { 1f, 2f, 3f, 4f, 5f };
        float[] dst = new float[] { 0f, 0f, 0f, 0f, 0f };

        CacheOptimizer.CopyWithPrefetch(src, dst, length: 3);

        Assert.Equal(1f, dst[0]);
        Assert.Equal(2f, dst[1]);
        Assert.Equal(3f, dst[2]);
        Assert.Equal(0f, dst[3]);
        Assert.Equal(0f, dst[4]);
    }
}
