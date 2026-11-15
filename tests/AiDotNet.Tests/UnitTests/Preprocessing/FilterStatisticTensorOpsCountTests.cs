using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Preprocessing.FeatureSelection;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Preprocessing;

/// <summary>
/// Reference tests for the FilterStatisticTensorOps class-size counts: they must be exact past 2048 in Half (a 0/1
/// sum accumulated in Half stops growing at 2048 because 2048 + 1 rounds back to 2048).
/// </summary>
public class FilterStatisticTensorOpsCountTests
{
    // 3072 rows of class 0 and 1024 of class 1: every expected value below is exact in Half.
    private const int ClassZeroRows = 3072;
    private const int ClassOneRows = 1024;
    private const int Rows = ClassZeroRows + ClassOneRows;

    private static int[] ClassIndex() =>
        Enumerable.Range(0, Rows).Select(i => i < ClassZeroRows ? 0 : 1).ToArray();

    private static Tensor<Half> ClassOneColumn(double value) =>
        new Tensor<Half>(ClassIndex().Select(c => c == 1 ? (Half)value : (Half)0.0).ToArray(), new[] { Rows, 1 });

    // x = 1 on class 1 and 0 on class 0: SSB = 3072 * 0.25^2 + 1024 * 0.75^2 = 768 and SSW = 0. With the class-0
    // size stuck at 2048 SSB came out 704.
    [Fact(Timeout = 60000)]
    public async Task AnovaSums_Half_UsesExactClassSizes()
    {
        await Task.Yield();
        var membership = FilterStatisticTensorOps.ClassOneHot<Half>(ClassIndex(), 2);

        var (between, within) = FilterStatisticTensorOps.AnovaSums(ClassOneColumn(1.0), membership);

        Assert.Equal(768.0, (double)between.ToArray()[0]);
        Assert.Equal(0.0, (double)within.ToArray()[0]);
    }

    // [200, -200, 200, -200] has population variance 40,000, which Half holds; its sum of squared deviations,
    // 160,000, does not, so dividing the Half sum gave inf.
    [Fact(Timeout = 60000)]
    public async Task ColumnPopulationVariance_Half_SumPast65504()
    {
        await Task.Yield();
        var x = new Tensor<Half>(new[] { (Half)200, (Half)(-200), (Half)200, (Half)(-200) }, new[] { 4, 1 });

        Assert.Equal(40000.0, (double)FilterStatisticTensorOps.ColumnPopulationVariance(x).ToArray()[0]);
    }

    // 3072 class-0 rows of 32 and 1024 class-1 rows of 33: overall mean 32.25, SSB = 3072 * 0.25^2 + 1024 * 0.75^2 =
    // 768 and SSW = 0. The Half class-0 sum 98,304 overflowed before the class mean was formed.
    [Fact(Timeout = 60000)]
    public async Task AnovaSums_Half_ClassSumPast65504()
    {
        await Task.Yield();
        var membership = FilterStatisticTensorOps.ClassOneHot<Half>(ClassIndex(), 2);
        var x = new Tensor<Half>(ClassIndex().Select(c => c == 1 ? (Half)33 : (Half)32).ToArray(), new[] { Rows, 1 });

        var (between, within) = FilterStatisticTensorOps.AnovaSums(x, membership);

        Assert.Equal(768.0, (double)between.ToArray()[0]);
        Assert.Equal(0.0, (double)within.ToArray()[0]);
    }

    // scipy.stats.f.sf(1, 1000, 1000) = 0.5 (F(d, d) has median 1). The continued fraction with a = b = 500 forms
    // 501 * 1001 and 502 * 503, past Half's 65504 (inf / inf = NaN).
    [Fact(Timeout = 60000)]
    public async Task FSurvival_Half_LargeDegreesOfFreedom()
    {
        await Task.Yield();
        var f = new Tensor<Half>(new[] { (Half)1 }, new[] { 1 });

        Assert.Equal(0.5, (double)FilterStatisticTensorOps.FSurvival(f, 1000, 1000).ToArray()[0], 1e-3);
    }

    // scipy.stats.t.sf(300, 1) * 2 = 2 / pi * atan(1 / 300) = 0.0021220. t^2 = 90000 overflowed Half, the beta
    // argument became 0 and the p-value 0, which passed a 0.001 threshold.
    [Fact(Timeout = 60000)]
    public async Task StudentTTwoSided_Half_SquaredStatisticPast65504()
    {
        await Task.Yield();
        var t = new Tensor<Half>(new[] { (Half)300 }, new[] { 1 });

        Assert.Equal(2.0 / Math.PI * Math.Atan(1.0 / 300.0),
            (double)FilterStatisticTensorOps.StudentTTwoSided(t, 1.0).ToArray()[0], 1e-5);
    }
}
