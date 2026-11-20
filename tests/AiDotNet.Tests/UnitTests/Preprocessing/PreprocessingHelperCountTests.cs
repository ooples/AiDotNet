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
        await Task.Yield();
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
        await Task.Yield();
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
        await Task.Yield();
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
        await Task.Yield();
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
        await Task.Yield();
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
        await Task.Yield();
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
        await Task.Yield();
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
        await Task.Yield();
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
        await Task.Yield();
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
        await Task.Yield();
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
        await Task.Yield();
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

    // x = [0, 2^-15, 2^-14] in two bins: width = 2^-14 / 2 + 1e-10, so 2^-14 lands in bin 1 (2^-14 / width just below
    // 2). With 1e-10 raised to Half's smallest normal 2^-14 the width was 3 * 2^-15 and every value fell in bin 0.
    [Fact(Timeout = 60000)]
    public async Task EqualWidthBinOneHot_Half_SmallRangeKeepsTheUpperBin()
    {
        await Task.Yield();
        var x = new Tensor<Half>(new[] { (Half)0, (Half)Math.Pow(2, -15), (Half)Math.Pow(2, -14) }, new[] { 3, 1 });

        var oneHot = PreprocessingTensorOps.EqualWidthBinOneHot(x, 2, out var varying);

        Assert.Equal(1.0, (double)varying.GetFlat(0), Tolerance);
        Assert.Equal(new[] { 1.0, 0.0, 1.0, 0.0, 0.0, 1.0 }, oneHot.ToArray().Select(v => (double)v).ToArray());
    }

    // [-40000, 0, 40000] in three bins: the range 80000 passes Half's 65504, so in Half the width was infinite, 0 and
    // 40000 both scaled to 0 and the maximum to inf / inf = NaN. In float the bins are 0, 1 and 2.
    [Fact(Timeout = 60000)]
    public async Task EqualWidthBinOneHot_Half_RangePast65504KeepsEveryBin()
    {
        await Task.Yield();
        var x = new Tensor<Half>(new[] { (Half)(-40000), (Half)0, (Half)40000 }, new[] { 3, 1 });

        var oneHot = PreprocessingTensorOps.EqualWidthBinOneHot(x, 3, out _);

        Assert.Equal(new[] { 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 }, oneHot.ToArray().Select(v => (double)v).ToArray());
    }

    // Range-scaled bins (x - min) / (max - min) * (bins - 1) of [-40000, 0, 40000] with three bins are 0, 1 and 2;
    // the Half range 80000 overflowed to inf, scaling 0 to bin 0 and 40000 to inf / inf = NaN.
    [Fact(Timeout = 60000)]
    public async Task RangeScaledBinOneHot_Half_RangePast65504KeepsEveryBin()
    {
        await Task.Yield();
        var x = new Tensor<Half>(new[] { (Half)(-40000), (Half)0, (Half)40000 }, new[] { 3, 1 });

        var oneHot = PreprocessingTensorOps.RangeScaledBinOneHot(x, 3);

        Assert.Equal(new[] { 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 }, oneHot.ToArray().Select(v => (double)v).ToArray());
    }

    // 2049 categories: label 2048 is valid and becomes its category (7). In Half the count 2049 rounds to 2048, so the
    // label failed 2048 < 2048 and kept its own value.
    [Fact(Timeout = 60000)]
    public async Task GatherCategories_Half_LabelPast2048IsACategory()
    {
        await Task.Yield();
        var categories = PreprocessingTensorOps.Filled(new[] { 2049 }, (Half)7);
        var labels = new Tensor<Half>(new[] { (Half)2048, (Half)(-1) }, new[] { 2 });

        var gathered = PreprocessingTensorOps.GatherCategories(categories, labels);

        Assert.Equal(new[] { 7.0, -1.0 }, gathered.ToArray().Select(v => (double)v).ToArray());
    }

    // Per-category sums by index-add: categories 0 and 2 collect 1 + 3 and 2, category 1 has no rows.
    [Fact(Timeout = 60000)]
    public async Task CategorySums_AddsEveryRowToItsCategory()
    {
        await Task.Yield();
        var index = new Tensor<int>(new[] { 0, 2, 0 }, new[] { 3 });
        var values = new Tensor<double>(new[] { 1.0, 2.0, 3.0 }, new[] { 3 });

        var sums = PreprocessingTensorOps.CategorySums(index, values, 3);

        Assert.Equal(new[] { 4.0, 0.0, 2.0 }, sums.ToArray());
    }

    // A 2050-column mask whose only set entry is column 2049: the first non-zero is 2049, and an all-zero row gets the
    // width 2050. Positions held in Half round 2049 to 2048.
    [Fact(Timeout = 60000)]
    public async Task FirstNonZeroPerRow_Half_PositionPast2048IsExact()
    {
        await Task.Yield();
        var mask = HalfTensor(new[] { 2, 2050 }, i => i == 2049 ? 1 : 0);

        var first = PreprocessingTensorOps.FirstNonZeroPerRow(mask);

        Assert.Equal(new[] { 2049, 2050 }, first);
    }

    // [-40000, 0, 40000] normalizes to [0, 0.5, 1]; the Half range 80000 overflowed to inf, giving 0, 0 and NaN.
    [Fact(Timeout = 60000)]
    public async Task MinMaxNormalize_Half_RangePast65504()
    {
        await Task.Yield();
        var x = new Tensor<Half>(new[] { (Half)(-40000), (Half)0, (Half)40000 }, new[] { 3 });

        var normalized = PreprocessingTensorOps.MinMaxNormalize(x);

        Assert.Equal(new[] { 0.0, 0.5, 1.0 }, normalized.ToArray().Select(v => (double)v).ToArray());
    }

    // [-40000, -20000, 0, 20000, 40000] in five blocks of width 16000 gives blocks 0, 1, 2, 3 and 4 (80000 / 16000 = 5
    // clamps to 4). In Half the range overflowed, the width was infinite and the blocks collapsed to 0.
    [Fact(Timeout = 60000)]
    public async Task EqualWidthBlockIndex_Half_RangePast65504()
    {
        await Task.Yield();
        var x = new Tensor<Half>(new[] { (Half)(-40000), (Half)(-20000), (Half)0, (Half)20000, (Half)40000 }, new[] { 5 });

        Assert.Equal(new[] { 0, 1, 2, 3, 4 }, PreprocessingTensorOps.EqualWidthBlockIndex(x, 5));
    }

    // A row of 65,536 ones has norm 256. Its sum of squares, 65,536, passes Half's 65504 (or stalls at 2048 when
    // accumulated in Half), so the norm came out inf or 45.25 instead.
    [Fact(Timeout = 60000)]
    public async Task RowNorms_Half_SumOfSquaresPast65504()
    {
        await Task.Yield();
        var x = HalfTensor(new[] { 1, 65536 }, _ => 1);

        var norms = PreprocessingTensorOps.RowNorms(x);

        Assert.Equal(256.0, (double)norms.GetFlat(0), Tolerance);
    }

    // Column [-65504, 65504, 65504]: mean 21834.67, differences -87338.67, 43669.33, 43669.33 and population std
    // 61758.04, so the standardized distances are sqrt(2), 1 / sqrt(2) and 1 / sqrt(2). In Half the column sum and
    // the first difference passed 65504 and the distances came out inf or NaN.
    [Fact(Timeout = 60000)]
    public async Task CentroidDistances_Half_DifferencesPast65504()
    {
        await Task.Yield();
        var data = new Matrix<Half>(new[,] { { (Half)(-65504) }, { (Half)65504 }, { (Half)65504 } });

        var distances = PreprocessingTensorOps.CentroidDistances(data, standardize: true);

        Assert.Equal(Math.Sqrt(2), (double)distances.GetFlat(0), 1e-2);
        Assert.Equal(1 / Math.Sqrt(2), (double)distances.GetFlat(1), 1e-2);
        Assert.Equal(1 / Math.Sqrt(2), (double)distances.GetFlat(2), 1e-2);
    }
}
