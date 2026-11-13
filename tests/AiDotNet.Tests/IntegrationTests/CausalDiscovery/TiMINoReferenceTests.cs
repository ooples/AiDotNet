using AiDotNet.CausalDiscovery.TimeSeries;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.CausalDiscovery;

/// <summary>
/// Hand-derived reference values for TiMINo (Peters, Janzing and Schölkopf 2013): the HSIC gamma
/// test of Gretton et al. (2008) it uses for independence, and the sink-first order search.
/// </summary>
public class TiMINoReferenceTests
{
    private static Tensor<double> Series(params double[] values) => new Tensor<double>(values, new[] { values.Length });

    [Fact]
    public void GaussianKernel_UsesMedianOfPositiveSquaredDistances()
    {
        // x = [0, 1, 3]: the positive squared distances are {1, 4, 9} (each twice), median 4,
        // so 2σ² = 4 (σ² = median / 2 = 2).
        var algo = new TiMINoAlgorithm<double>();
        var kernel = algo.BuildGaussianKernel(Series(0, 1, 3));
        Assert.Equal(4.0, kernel.SquaredWidth, 12);

        // Off-diagonal mean of K: (2e^{-1/4} + 2e^{-9/4} + 2e^{-1}) / 6.
        double expected = (2 * Math.Exp(-0.25) + 2 * Math.Exp(-2.25) + 2 * Math.Exp(-1.0)) / 6.0;
        Assert.Equal(expected, kernel.OffDiagonalMean, 12);
    }

    [Fact]
    public void HsicGammaTest_TwoGroupSeries_MatchesClosedForm()
    {
        // x = y = [0,0,0,1,1,1], m = 6. Squared distances are 0 or 1, so 2σ² = 1 and
        // K = q·11ᵀ + (1-q)·B with q = e^{-1} and B the same-group indicator. With s = ±1 the group
        // signs, HBH = ssᵀ/2, so K̃ = L̃ = (1-q)/2 · ssᵀ and every entry of K̃∘L̃ is c = (1-q)²/4.
        //   statistic = Σ(K̃∘L̃)/m = 36c/6 = 1.5(1-q)²
        //   variance  = 72·2·1/(6·5·4·3) · (30c²/36)/30 = c²/90
        //   μ_K = μ_L = (ΣK - m)/(m(m-1)) = (12 + 18q)/30, so 1 - μ = 3(1-q)/5
        //   mean      = (1-μ)²/m = 3(1-q)²/50
        //   shape     = mean²/variance = 5.184, rate·statistic = mean·statistic/(m·variance) = 21.6
        // p-value = Q(5.184, 21.6) = 6.147046506e-6 (regularized upper incomplete gamma).
        var algo = new TiMINoAlgorithm<double>();
        var kernel = algo.BuildGaussianKernel(Series(0, 0, 0, 1, 1, 1));
        var (statistic, pValue) = algo.HsicGammaTest(kernel, kernel);

        double q = Math.Exp(-1.0);
        Assert.Equal(1.5 * (1 - q) * (1 - q), statistic, 12);
        Assert.Equal(6.147046506e-6, pValue, 1e-10);
    }

    [Fact]
    public void HsicGammaTest_ConstantSeries_IsIndependent()
    {
        // A constant series has no positive distance and a zero centered kernel: HSIC = 0, p = 1.
        var algo = new TiMINoAlgorithm<double>();
        var constant = algo.BuildGaussianKernel(Series(2, 2, 2, 2, 2, 2));
        var other = algo.BuildGaussianKernel(Series(0, 1, 3, 2, 5, 4));
        Assert.Null(constant.Centered);
        Assert.Equal(1.0, algo.HsicGammaTest(constant, other).PValue);
    }

    /// <summary>
    /// x0[t] iid uniform on [-1, 1) and x1[t] = 0.8·x0[t] + 0.5·u[t] with u iid uniform: an
    /// instantaneous linear effect with non-Gaussian noise, which TiMINo identifies. Regressing x1 on
    /// x0 leaves the noise u, independent of x0 (p-value 1 after Bonferroni); regressing x0 on x1
    /// leaves a residual that depends on x1 (p ≈ 1.5e-9), so x1 is the sink. Pruning keeps x0 as a
    /// parent: without it the residual 0.8·x0 + 0.5·u depends on x0 (p ≈ 0).
    /// </summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void DiscoverStructure_UniformNoiseChain_FindsOnlyTheCausalDirection(bool swapColumns)
    {
        const int n = 300;
        var values = new double[n, 2];
        long state = 12345;
        double Uniform()
        {
            state = (state * 1103515245 + 12345) % 2147483648L;
            return 2.0 * state / 2147483648.0 - 1.0;
        }

        for (int t = 0; t < n; t++)
        {
            double cause = Uniform();
            double effect = 0.8 * cause + 0.5 * Uniform();
            values[t, swapColumns ? 1 : 0] = cause;
            values[t, swapColumns ? 0 : 1] = effect;
        }

        int causeIndex = swapColumns ? 1 : 0;
        int effectIndex = 1 - causeIndex;
        var algo = new TiMINoAlgorithm<double>(new CausalDiscoveryOptions { MaxLag = 1 });
        var graph = algo.DiscoverStructure(new Matrix<double>(values));

        Assert.Equal(1.0, graph.AdjacencyMatrix[causeIndex, effectIndex]);
        Assert.Equal(0.0, graph.AdjacencyMatrix[effectIndex, causeIndex]);
        Assert.False(algo.ModelFitRejected);
    }

    [Fact]
    public void DiscoverStructure_DeterministicTrends_ReportsBadModelFit()
    {
        // Exactly collinear trends leave residuals that are a (rounding-level) linear function of
        // time, which every regressor series also is, so no sink candidate passes the test: the
        // paper's "I do not know - bad model fit" outcome.
        const int n = 50;
        var values = new double[n, 3];
        for (int i = 0; i < n; i++)
        {
            double x = i * 0.1;
            values[i, 0] = x;
            values[i, 1] = 2.0 * x + 0.5;
            values[i, 2] = x + values[i, 1] * 0.3;
        }

        var algo = new TiMINoAlgorithm<double>();
        algo.DiscoverStructure(new Matrix<double>(values));
        Assert.True(algo.ModelFitRejected);
    }
}
