using AiDotNet.Preprocessing.FeatureSelection.Filter.Correlation;
using AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate;
using Xunit;
using System.Threading.Tasks;
using StatisticalWilcoxonSignedRank = AiDotNet.Preprocessing.FeatureSelection.Filter.Statistical.WilcoxonSignedRank<double>;
using UnivariateWilcoxonSignedRank = AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate.WilcoxonSignedRank<double>;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// Regression tests for int32 overflow in sample-count products used by rank and
/// correlation feature-selection statistics (the products were computed in int and
/// only then widened to double).
/// </summary>
public class FeatureSelectionRankStatisticPrecisionTests
{
    [Fact(Timeout = 120000)]
    public async Task KruskalWallisTest_LargeTieGroups_AppliesTieCorrection()
    {
        // n = 3000, two classes of 1500, feature equal to the class label: each class is one
        // tie group of 1500. 1500^3 = 3.375e9 overflowed int32 to a negative tie sum, so the tie
        // correction was skipped (H = 2249.25). With the correction, a perfectly separating
        // two-group feature gives H = n - 1 = 2999.
        const int n = 3000;
        var data = new Matrix<double>(n, 1);
        var target = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            double label = i < n / 2 ? 0.0 : 1.0;
            data[i, 0] = label;
            target[i] = label;
        }

        var selector = new KruskalWallisTest<double>(nFeaturesToSelect: 1);
        selector.Fit(data, target);

        Assert.NotNull(selector.HStatistics);
        Assert.Equal(n - 1.0, selector.HStatistics[0], 1e-6);
    }

    [Fact(Timeout = 120000)]
    public async Task StatisticalWilcoxonSignedRank_TwoThousandPairs_ComputesFiniteStdDev()
    {
        // 2000 strictly positive, distinct differences: W = 0, mean = 1,000,500,
        // stdDev = sqrt(2000*2001*4001/24) ~ 25,830, z ~ -38.7, p ~ 0.
        // 2000*2001*4001 = 1.6e10 overflowed int32 to a negative value, stdDev was NaN and the
        // p-value fell back to 1.0.
        const int n = 2000;
        var (data, target) = BuildStrictlyPositiveDifferences(n);

        var selector = new StatisticalWilcoxonSignedRank(nFeaturesToSelect: 1);
        selector.Fit(data, target);

        Assert.NotNull(selector.PValues);
        Assert.True(selector.PValues[0] < 1e-6, $"Expected p-value near 0, got {selector.PValues[0]}");
    }

    [Fact(Timeout = 120000)]
    public async Task UnivariateWilcoxonSignedRank_TwoThousandPairs_ComputesFiniteSigma()
    {
        // Same construction as above for the Univariate variant: old sigma was NaN so the
        // p-value fell back to 1.0 instead of ~0.
        const int n = 2000;
        var (data, target) = BuildStrictlyPositiveDifferences(n);

        var selector = new UnivariateWilcoxonSignedRank(nFeaturesToSelect: 1);
        selector.Fit(data, target);

        Assert.NotNull(selector.PValues);
        Assert.True(selector.PValues[0] < 1e-6, $"Expected p-value near 0, got {selector.PValues[0]}");
    }

    [Fact(Timeout = 120000)]
    public async Task PointBiserial_HundredThousandSamples_PerfectSeparationGivesCorrelationOne()
    {
        // n = 100,000 balanced classes, feature equal to the class label: r_pb = 1.
        // n0 * n1 = 2.5e9 and n * n = 1e10 both overflowed int32 (pq = -1.27), so the
        // correlation was NaN.
        const int n = 100000;
        var data = new Matrix<double>(n, 1);
        var target = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            double label = i < n / 2 ? 0.0 : 1.0;
            data[i, 0] = label;
            target[i] = label;
        }

        var selector = new PointBiserial<double>(nFeaturesToSelect: 1);
        selector.Fit(data, target);

        Assert.NotNull(selector.Correlations);
        Assert.Equal(1.0, selector.Correlations[0], 1e-9);
    }

    private static (Matrix<double> Data, Vector<double> Target) BuildStrictlyPositiveDifferences(int n)
    {
        var data = new Matrix<double>(n, 1);
        var target = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            data[i, 0] = i + 1.0;
            target[i] = 0.0;
        }

        return (data, target);
    }
}
