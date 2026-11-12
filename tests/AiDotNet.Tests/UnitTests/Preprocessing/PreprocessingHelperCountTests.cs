using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Preprocessing;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Preprocessing;

/// <summary>
/// Reference tests for the PreprocessingTensorOps count helpers: counts must be exact past 2048 in Half (a 0/1 sum
/// accumulated in Half stops growing at 2048 because 2048 + 1 rounds back to 2048).
/// </summary>
public class PreprocessingHelperCountTests
{
    // 3072 rows of class 0 and 1024 of class 1: every expected value below is exact in Half.
    private const int ClassZeroRows = 3072;
    private const int ClassOneRows = 1024;
    private const int Rows = ClassZeroRows + ClassOneRows;

    private static int[] ClassIndex() =>
        Enumerable.Range(0, Rows).Select(i => i < ClassZeroRows ? 0 : 1).ToArray();

    [Fact(Timeout = 60000)]
    public async Task CategoryCounts_Half_CountsPast2048AndSkipsOutOfRangePositions()
    {
        await Task.CompletedTask;
        var positions = ClassIndex().Concat(new[] { -1, 2, 7 }).ToArray();

        var counts = PreprocessingTensorOps.CategoryCounts<Half>(new Tensor<int>(positions, new[] { positions.Length }), 2);

        Assert.Equal(new[] { 2 }, counts.Shape.ToArray());
        Assert.Equal((double)ClassZeroRows, (double)counts.ToArray()[0]);
        Assert.Equal((double)ClassOneRows, (double)counts.ToArray()[1]);
    }

    // A = B = the class (3072 rows in bin 0, 1024 in bin 1): the joint cells are [3072, 0; 0, 1024] and the mutual
    // information is the class entropy -(0.75 log2 0.75 + 0.25 log2 0.25) = 0.8112781244591328 bits. A joint cell
    // stuck at 2048 in Half gave a different value, and the 3072 * 1024 marginal product overflowed Half.
    [Fact(Timeout = 60000)]
    public async Task HistogramMutualInformation_Half_CountsJointCellsPast2048()
    {
        await Task.CompletedTask;
        var oneHot = PreprocessingTensorOps.ClassOneHot<Half>(ClassIndex(), 2);

        var mi = PreprocessingTensorOps.HistogramMutualInformation(oneHot, oneHot, 2);

        Assert.Equal(new[] { 1, 1 }, mi.Shape.ToArray());
        Assert.Equal(0.8112781244591328, (double)mi.ToArray()[0], 1e-3);
    }
}
