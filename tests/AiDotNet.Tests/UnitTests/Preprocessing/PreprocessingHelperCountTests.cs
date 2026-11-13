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

    // x = (-1)^i over 3000 rows and y = x on the first 1000 rows, 0 after: x'y = 1000, x'x = 3000 and y'y = 1000, so
    // r = 1000 / sqrt(3000 * 1000) = 1 / sqrt(3). Max-scaling keeps Half from overflowing but not from stalling: x'x
    // accumulated in Half stops at 2048 (2048 + 1 rounds back to 2048), which gave 1000 / sqrt(2048 * 1000) = 0.699.
    [Fact(Timeout = 60000)]
    public async Task CenteredCorrelations_Half_SumsMoreThan2048Squares()
    {
        await Task.CompletedTask;
        const int n = 3000;
        var xc = new Tensor<Half>(Enumerable.Range(0, n).Select(i => (Half)(i % 2 == 0 ? 1 : -1)).ToArray(), new[] { n, 1 });
        var yc = new Tensor<Half>(
            Enumerable.Range(0, n).Select(i => (Half)(i >= 1000 ? 0 : i % 2 == 0 ? 1 : -1)).ToArray(), new[] { n, 1 });

        var r = PreprocessingTensorOps.CenteredCorrelations(xc, yc);

        Assert.Equal(1.0 / Math.Sqrt(3.0), (double)r[0], 1e-3);
    }

    // x = 300 (-1)^i over ten rows, y = x: r = 1. sum x y = 900000 is past Half's 65504, so a correlation formed in
    // Half from the unscaled cross product was inf / inf = NaN even with the roots taken before the product.
    [Fact(Timeout = 60000)]
    public async Task TargetCorrelations_Half_CrossProductPast65504IsOne()
    {
        await Task.CompletedTask;
        var values = Enumerable.Range(0, 10).Select(i => (Half)(i % 2 == 0 ? 300 : -300)).ToArray();
        var x = new Tensor<Half>(values, new[] { 10, 1 });
        var y = new Tensor<Half>(values, new[] { 10 });

        Assert.Equal(1.0, (double)PreprocessingTensorOps.TargetCorrelations(x, y)[0], 1e-3);
        Assert.Equal(1.0, (double)PreprocessingTensorOps.CorrelationsWithDenominatorGuard(x, y)[0], 1e-3);
    }

    // The tensor library's Linalg.Solve runs only in float and double, so a Half system threw NotSupportedException.
    // diag(2, 4) x = (2, 8) has x = (1, 2).
    [Fact(Timeout = 60000)]
    public async Task Solve_Half_WidensTheSystemToFloat()
    {
        await Task.CompletedTask;
        var a = new Tensor<Half>(new[] { (Half)2, (Half)0, (Half)0, (Half)4 }, new[] { 2, 2 });
        var b = new Tensor<Half>(new[] { (Half)2, (Half)8 }, new[] { 2, 1 });

        var x = PreprocessingTensorOps.Solve(a, b);

        Assert.Equal(1.0, (double)x[0, 0], 1e-3);
        Assert.Equal(2.0, (double)x[1, 0], 1e-3);
    }

    // Linalg.Eigh runs only in float and double, so a Half eigendecomposition threw NotSupportedException. The
    // eigenvalues of [[2, 1], [1, 2]] are 1 and 3 (ascending), with eigenvectors (1, -1) / sqrt(2) and (1, 1) / sqrt(2).
    [Fact(Timeout = 60000)]
    public async Task SymmetricEigh_Half_WidensTheDecompositionToFloat()
    {
        await Task.CompletedTask;
        var symmetric = new Tensor<Half>(new[] { (Half)2, (Half)1, (Half)1, (Half)2 }, new[] { 2, 2 });

        var (values, vectors) = PreprocessingTensorOps.SymmetricEigh(symmetric);

        Assert.Equal(1.0, (double)values[0], 1e-3);
        Assert.Equal(3.0, (double)values[1], 1e-3);
        Assert.Equal(1.0 / Math.Sqrt(2.0), Math.Abs((double)vectors[0, 1]), 1e-3);
        Assert.Equal((double)vectors[0, 1], (double)vectors[1, 1], 1e-3);
        Assert.Equal(-(double)vectors[0, 0], (double)vectors[1, 0], 1e-3);
    }

    // Linalg.Inv runs only in float and double, so a Half inverse threw NotSupportedException.
    [Fact(Timeout = 60000)]
    public async Task Inverse_Half_WidensTheInversionToFloat()
    {
        await Task.CompletedTask;
        var matrix = new Tensor<Half>(new[] { (Half)2, (Half)1, (Half)0, (Half)4 }, new[] { 2, 2 });

        var inverse = PreprocessingTensorOps.Inverse(matrix);

        // [[2, 1], [0, 4]]^-1 = [[1/2, -1/8], [0, 1/4]].
        Assert.Equal(0.5, (double)inverse[0, 0], 1e-3);
        Assert.Equal(-0.125, (double)inverse[0, 1], 1e-3);
        Assert.Equal(0.0, (double)inverse[1, 0], 1e-3);
        Assert.Equal(0.25, (double)inverse[1, 1], 1e-3);
    }

    // Linalg.Svd runs only in float and double, so a Half SVD threw NotSupportedException.
    [Fact(Timeout = 60000)]
    public async Task Svd_Half_WidensTheDecompositionToFloat()
    {
        await Task.CompletedTask;
        var matrix = new Tensor<Half>(new[] { (Half)3, (Half)0, (Half)4, (Half)5 }, new[] { 2, 2 });

        var (u, s, vh) = PreprocessingTensorOps.Svd(matrix, false);

        // A'A = [[25, 20], [20, 25]] has eigenvalues 45 and 5, so the singular values are sqrt(45) and sqrt(5).
        Assert.Equal(Math.Sqrt(45.0), (double)s[0], 1e-2);
        Assert.Equal(Math.Sqrt(5.0), (double)s[1], 1e-2);
        double[] expected = { 3, 0, 4, 5 };
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                double rebuilt = (double)u[i, 0] * (double)s[0] * (double)vh[0, j]
                    + (double)u[i, 1] * (double)s[1] * (double)vh[1, j];
                Assert.Equal(expected[i * 2 + j], rebuilt, 2e-2);
            }
        }
    }
}
