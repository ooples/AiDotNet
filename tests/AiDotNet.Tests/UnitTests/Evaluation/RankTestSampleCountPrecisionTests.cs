using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Evaluation.Statistics;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Evaluation;

/// <summary>
/// Regression tests for integer overflow in the sample-count products of the non-parametric
/// statistical tests (N*(N+1), N^3 - N, nr*(nr+1)*(2nr+1), grandTotal^2). Each product used to be
/// evaluated in 32-bit int arithmetic and only then widened to double, so it wrapped (often to a
/// negative value) at sample sizes that are ordinary for ML evaluation.
/// </summary>
public class RankTestSampleCountPrecisionTests
{
    [Fact]
    public void CochranQ_LargeSampleCount_MatchesClosedForm()
    {
        // Two classifiers, all labels positive. Classifier A is correct on all n samples; B on the
        // first m. Then Q reduces to (n - m)^2 / (n - m) = n - m. grandTotal = n + m = 59,990, whose
        // square (3.6e9) overflowed int32 and made Q ~ 4.3e8 instead of 10.
        const int n = 30000;
        const int m = 29990;
        var actuals = Enumerable.Repeat(1.0, n).ToArray();
        var classifierA = Enumerable.Repeat(1.0, n).ToArray();
        var classifierB = Enumerable.Range(0, n).Select(i => i < m ? 1.0 : 0.0).ToArray();

        var result = new CochranQTest<double>().Test(new[] { classifierA, classifierB }, actuals);

        Assert.Equal((double)(n - m), result.Statistic, 6);
    }

    [Fact]
    public void KruskalWallis_TieCorrection_LargeN_MatchesDoublePrecisionReference()
    {
        // N = 1500 pooled observations with heavy ties. The tie-correction denominator N^3 - N
        // (3.4e9) overflowed int32 to a negative number, turning the correction factor from ~0.994
        // into ~1.023 and shrinking H by ~3%.
        const int perGroup = 750;
        var groupA = Enumerable.Range(0, perGroup).Select(i => (double)(i % 10)).ToArray();
        var groupB = Enumerable.Range(0, perGroup).Select(i => (double)(i % 10) + 5.0).ToArray();

        var result = new KruskalWallisTest<double>().Test(new[] { groupA, groupB });

        double expected = ReferenceKruskalWallisH(groupA, groupB);
        Assert.Equal(expected, result.Statistic, 1e-9 * Math.Abs(expected));
    }

    [Fact]
    public void KruskalWallis_DunnPostHoc_LargeN_ReturnsFinitePValue()
    {
        // N = 50,000 identical observations: every mean rank is equal, so z = 0 and p = 1.
        // N*(N+1) = 2.5e9 overflowed int32 to a negative number; its square root was NaN, so the
        // p-value was NaN.
        const int perGroup = 25000;
        var groupA = Enumerable.Repeat(1.0, perGroup).ToArray();
        var groupB = Enumerable.Repeat(1.0, perGroup).ToArray();

        var results = new KruskalWallisTest<double>().DunnPostHoc(new[] { groupA, groupB });

        double pValue = results[(0, 1)];
        Assert.InRange(pValue, 0.999, 1.001);
    }

    [Fact]
    public void WilcoxonSignedRank_LargeNonZeroPairCount_ReturnsFinitePValue()
    {
        // 2000 non-zero differences with magnitudes 1..2000 and alternating signs:
        // W+ = 1,000,000, W- = 1,001,000, mean = 1,000,500, std = sqrt(2000*2001*4001/24) = 25,829.57,
        // z = -0.01934, two-sided p = 0.9846. nr*(nr+1)*(2nr+1) = 1.6e10 overflowed int32 to a
        // negative number, so std (and the p-value) was NaN.
        const int n = 2000;
        var sample1 = Enumerable.Range(0, n).Select(i => (i % 2 == 0 ? 1.0 : -1.0) * (i + 1)).ToArray();
        var sample2 = new double[n];

        var result = new WilcoxonSignedRankTest<double>().Test(sample1, sample2);

        Assert.Equal(1_000_000.0, result.Statistic, 6);
        Assert.InRange(result.PValue, 0.983, 0.986);
    }

    private static double ReferenceKruskalWallisH(double[] groupA, double[] groupB)
    {
        var pooled = groupA.Select(v => (value: v, group: 0))
            .Concat(groupB.Select(v => (value: v, group: 1)))
            .OrderBy(x => x.value)
            .ToList();
        double n = pooled.Count;

        var rankSums = new double[2];
        var tieSizes = new List<double>();
        int start = 0;
        while (start < pooled.Count)
        {
            int end = start;
            while (end < pooled.Count && pooled[end].value == pooled[start].value)
                end++;

            double midRank = (start + 1 + end) / 2.0;
            for (int i = start; i < end; i++)
                rankSums[pooled[i].group] += midRank;
            if (end - start > 1)
                tieSizes.Add(end - start);
            start = end;
        }

        double h = 12.0 / (n * (n + 1))
                   * (rankSums[0] * rankSums[0] / groupA.Length + rankSums[1] * rankSums[1] / groupB.Length)
                   - 3.0 * (n + 1);
        double tieCorrection = 1.0 - tieSizes.Sum(t => t * t * t - t) / (n * n * n - n);
        return h / tieCorrection;
    }
}
