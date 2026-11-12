using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Preprocessing;
using AiDotNet.Preprocessing.FeatureSelection;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Preprocessing;

/// <summary>
/// Reference tests for the shared preprocessing count helpers: row counts must be exact past 2048 in Half (a 0/1 sum
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

    private static Tensor<Half> ClassOneColumn(double value) =>
        new Tensor<Half>(ClassIndex().Select(c => c == 1 ? (Half)value : (Half)0.0).ToArray(), new[] { Rows, 1 });

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

    // x = 1 on class 1 and 0 on class 0: SSB = 3072 * 0.25^2 + 1024 * 0.75^2 = 768 and SSW = 0. With the class-0
    // size stuck at 2048 SSB came out 704.
    [Fact(Timeout = 60000)]
    public async Task AnovaSums_Half_UsesExactClassSizes()
    {
        await Task.CompletedTask;
        var membership = FilterStatisticTensorOps.ClassOneHot<Half>(ClassIndex(), 2);

        var (between, within) = FilterStatisticTensorOps.AnovaSums(ClassOneColumn(1.0), membership);

        Assert.Equal(768.0, (double)between.ToArray()[0]);
        Assert.Equal(0.0, (double)within.ToArray()[0]);
    }
}
