using AiDotNet.Clustering.Evaluation;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Clustering;

/// <summary>
/// Regression tests for the expected-index term of <see cref="AdjustedRandIndex{T}"/>. The product
/// sum(C(a_i,2)) * sum(C(b_j,2)) was formed in 64-bit integer arithmetic before being widened, which
/// overflows once each sum passes ~3e9 (a few hundred thousand points in a handful of clusters).
/// </summary>
public class AdjustedRandIndexPrecisionTests
{
    [Fact]
    public void Compute_IndependentLabelingsOnLargeDataset_ReturnsNearZero()
    {
        const int n = 200_000;
        var trueLabels = new Vector<double>(n);
        var predLabels = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            trueLabels[i] = i < n / 2 ? 0 : 1;
            predLabels[i] = i % 2;
        }

        double result = new AdjustedRandIndex<double>().Compute(trueLabels, predLabels);

        // Every contingency cell holds 50,000 points; each row/column sum is 100,000.
        double sumNij = 4.0 * (50_000.0 * 49_999.0 / 2.0);          // 4,999,900,000
        double sumA = 2.0 * (100_000.0 * 99_999.0 / 2.0);           // 9,999,900,000
        double sumB = sumA;
        double totalPairs = (double)n * (n - 1) / 2.0;              // 19,999,900,000
        double expectedIndex = sumA * sumB / totalPairs;
        double expected = (sumNij - expectedIndex) / (0.5 * (sumA + sumB) - expectedIndex);

        // sumA * sumB = 9.9998e19 exceeds long.MaxValue (9.22e18); the wrapped product gave ~0.48.
        Assert.Equal(expected, result, 9);
        Assert.True(System.Math.Abs(result) < 1e-3, $"Independent labelings must give ARI ~ 0, got {result}.");
    }
}
