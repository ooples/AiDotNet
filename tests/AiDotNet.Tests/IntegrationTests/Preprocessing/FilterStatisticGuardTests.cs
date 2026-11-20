using System;
using System.Threading.Tasks;
using AiDotNet.Preprocessing.FeatureSelection;
using AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// Guards of the shared filter statistics that the per-element loops had and the tensor rewrite has to keep: a
/// chi-square test with no degrees of freedom, and ln Gamma past the point where Gamma itself overflows.
/// </summary>
public class FilterStatisticGuardTests
{
    // A one-class target leaves the contingency test (bins - 1) * (classes - 1) = 0 degrees of freedom. The loop
    // this replaced returned p = 1 for df <= 0; Q(0, x) from the incomplete gamma is NaN.
    [Fact(Timeout = 60000)]
    public async Task ChiSquareTest_OneClassTarget_PValuesAreOne()
    {
        await Task.CompletedTask;
        var data = new Matrix<double>(new double[,] { { 1, 5 }, { 2, 5 }, { 3, 7 }, { 4, 9 } });
        var target = new Vector<double>(new[] { 1.0, 1.0, 1.0, 1.0 });
        var selector = new ChiSquareTest<double>(1);

        selector.Fit(data, target);

        Assert.NotNull(selector.PValues);
        Assert.NotNull(selector.ChiSquareStats);
        Assert.NotNull(selector.SelectedIndices);
        Assert.Equal(new[] { 1.0, 1.0 }, selector.PValues.ToArray());
        Assert.Equal(new[] { 0.0, 0.0 }, selector.ChiSquareStats.ToArray());
        Assert.Equal(new[] { 0 }, selector.SelectedIndices);
    }

    [Fact(Timeout = 60000)]
    public async Task ChiSquareSurvival_NoDegreesOfFreedom_IsOne()
    {
        await Task.CompletedTask;
        var statistic = new Tensor<double>(new[] { 0.0, 2.5 }, new[] { 2 });

        Assert.Equal(new[] { 1.0, 1.0 }, FilterStatisticTensorOps.ChiSquareSurvival(statistic, 0).ToArray());
        // chi2.sf(2.5, 1) = 0.11384629800665813 (scipy): the guard leaves a real test untouched.
        Assert.Equal(0.11384629800665813, FilterStatisticTensorOps.ChiSquareSurvival(statistic, 1)[1], 12);
    }

    // scipy.special.gammaln: Gamma(172) and Gamma(201) overflow double, Gamma(36) and Gamma(50) overflow float.
    [Fact(Timeout = 60000)]
    public async Task LogGamma_Double_PastTheGammaOverflow()
    {
        await Task.CompletedTask;
        var x = new Tensor<double>(new[] { 0.5, 1.0, 10.0, 150.0, 171.0, 172.0, 201.0, 1e6 }, new[] { 8 });

        var actual = FilterStatisticTensorOps.LogGamma(x).ToArray();

        var expected = new[]
        {
            0.5723649429247001, 0.0, 12.801827480081469, 600.0094705553274, 706.5730622457874,
            711.71472580229, 863.2319871924054, 12815504.569147611,
        };
        for (int i = 0; i < expected.Length; i++)
        {
            double tolerance = Math.Max(1e-12, Math.Abs(expected[i]) * 1e-13);
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance, $"lgamma({x[i]}) = {actual[i]}, expected {expected[i]}");
        }
    }

    [Fact(Timeout = 60000)]
    public async Task LogGamma_Float_PastTheGammaOverflow()
    {
        await Task.CompletedTask;
        var x = new Tensor<float>(new[] { 2f, 30f, 36f, 50f, 101f, 201f }, new[] { 6 });

        var actual = FilterStatisticTensorOps.LogGamma(x).ToArray();

        var expected = new[] { 0.0, 71.25703896716801, 92.13617560368708, 144.5657439463449, 363.7393755555635, 863.2319871924054 };
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-5 * Math.Max(1.0, expected[i]), $"lgamma({x[i]}) = {actual[i]}, expected {expected[i]}");
    }

    // Fisher's exact test on 400 rows: ln 400! needs lgamma(401), where Gamma overflows double and the p-value
    // became NaN. For [[120, 80], [90, 110]] the odds ratio is 120 * 110 / (80 * 90) and the two-sided p-value, the
    // exact sum of the hypergeometric probabilities no larger than the observed one (scipy.stats.fisher_exact's
    // rule, summed with integer binomials), is 0.0036351855558814394.
    [Fact(Timeout = 60000)]
    public async Task FisherExact_Double_MoreRowsThanGammaCanHold()
    {
        await Task.CompletedTask;
        static Tensor<double> One(double value) => new(new[] { value }, new[] { 1 });

        var (pValues, oddsRatios) = FeatureScoringTensorOps.FisherExact(One(120), One(80), One(90), One(110), 400);

        Assert.Equal(120.0 * 110.0 / (80.0 * 90.0), oddsRatios[0], 12);
        Assert.Equal(0.0036351855558814394, pValues[0], 9);
    }
}
