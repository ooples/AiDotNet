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

    private const double Tolerance = 1e-9;

    private static Tensor<Half> HalfTensor(int[] shape, Func<int, double> value)
    {
        int length = shape.Aggregate(1, (a, b) => a * b);
        var data = new Half[length];
        for (int i = 0; i < length; i++)
            data[i] = (Half)value(i);
        return new Tensor<Half>(data, shape);
    }

    [Fact(Timeout = 120000)]
    public async Task Tolerance_RaisesTinyEpsilonsToTheSmallestNormalHalf()
    {
        // 1e-10 rounds to 0 in Half; the smallest normal Half is 2^-14. float and double keep the value.
        await Task.CompletedTask;
        Assert.Equal(Math.Pow(2, -14), (double)PreprocessingTensorOps.Tolerance<Half>(1e-10), Tolerance);
        Assert.Equal(0.25, (double)PreprocessingTensorOps.Tolerance<Half>(0.25), Tolerance);
        Assert.Equal(1e-10, PreprocessingTensorOps.Tolerance<double>(1e-10));
        Assert.Equal(1e-10f, PreprocessingTensorOps.Tolerance<float>(1e-10));
    }

    [Fact(Timeout = 120000)]
    public async Task EqualWidthBinOneHot_Half_ConstantColumnIsNotVaryingAndBinsAtZero()
    {
        // A constant column has range 0 < 1e-10: not varying, every row in bin 0 (width 0 / bins + 1e-10). With the
        // 1e-10 rounded to 0 in Half the width was 0, the bins 0 / 0 = NaN and the column counted as varying.
        await Task.CompletedTask;
        var x = HalfTensor(new[] { 4, 1 }, _ => 3);

        var oneHot = PreprocessingTensorOps.EqualWidthBinOneHot(x, 3, out var varying);

        Assert.Equal(0.0, (double)varying.GetFlat(0), Tolerance);
        for (int i = 0; i < 4; i++)
        {
            Assert.Equal(1.0, (double)oneHot[i, 0], Tolerance);
            Assert.Equal(0.0, (double)oneHot[i, 1], Tolerance);
            Assert.Equal(0.0, (double)oneHot[i, 2], Tolerance);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task AtLeastHelpers_NaNIsNeverAtLeastTheMinimum()
    {
        // Master tests value >= minimum, which is false for NaN; 1 - (value < minimum) made NaN qualify.
        await Task.CompletedTask;
        var values = new Tensor<double>(new[] { double.NaN, 1.0, 2.0, 0.5 }, new[] { 4 });
        var atLeast = PreprocessingTensorOps.AtLeast(values, 1.0);
        Assert.Equal(new[] { 0.0, 1.0, 1.0, 0.0 }, atLeast.AsSpan().ToArray());

        var rows = new Tensor<double>(new[] { double.NaN, 2.0, 0.5 }, new[] { 3, 1 });
        var row = new Tensor<double>(new[] { 1.0 }, new[] { 1 });
        Assert.Equal(new[] { 0.0, 1.0, 0.0 }, PreprocessingTensorOps.AtLeastRow(rows, row).AsSpan().ToArray());

        var scores = new Tensor<double>(new[] { double.NaN, 0.9, 0.2, 0.7 }, new[] { 4 });
        Assert.Equal(new[] { 1, 3 }, PreprocessingTensorOps.TopKAtLeastSorted(scores, 0.5, 3));
    }
}
