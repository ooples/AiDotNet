using AiDotNet.Preprocessing.FeatureSelection.Moment;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// Reference-value tests for feature selectors, with expected values computed by SciPy / scikit-learn
/// (the generating Python snippet is quoted in each test).
/// </summary>
public class FeatureSelectionScipyReferenceTests
{
    private const double Tolerance = 1e-10;

    // Column 0 is bimodal, column 1 has one outlier, column 2 is evenly spaced.
    private static Matrix<double> MomentData() => new Matrix<double>(new double[,]
    {
        { 1.0, 2.0, 0.5 },
        { 2.0, 2.5, 1.5 },
        { 1.5, 9.0, 2.0 },
        { 8.0, 3.0, 2.5 },
        { 9.0, 2.0, 3.0 },
        { 8.5, 1.0, 3.5 },
        { 1.2, 2.2, 4.0 },
        { 8.8, 2.8, 4.5 },
    });

    [Fact]
    public void BimodalitySelector_UsesBiasAdjustedSampleMoments()
    {
        // G1 = scipy.stats.skew(X, bias=False); G2 = scipy.stats.kurtosis(X, bias=False)
        // BC = (G1**2 + 1) / (G2 + 3*(n-1)**2/((n-2)*(n-3)))   (SAS; Pfister et al. 2013)
        var selector = new BimodalitySelector<double>(nFeaturesToSelect: 1);
        selector.Fit(MomentData());

        var values = selector.BimodalityValues;
        Assert.NotNull(values);
        Assert.Equal(0.45728728624065296, values[0], Tolerance);
        Assert.Equal(0.6184213023584525, values[1], Tolerance);
        Assert.Equal(0.2559685384158708, values[2], Tolerance);
        Assert.Equal(new[] { 1 }, selector.SelectedIndices);
    }

    [Fact]
    public void KurtosisAndSkewnessSelectors_MatchScipyPopulationMoments()
    {
        // scipy.stats.kurtosis(X, fisher=True, bias=True); scipy.stats.skew(X, bias=True)
        var kurtosis = new KurtosisSelector<double>(nFeaturesToSelect: 2);
        kurtosis.Fit(MomentData());
        var skewness = new SkewnessSelector<double>(nFeaturesToSelect: 1, preferHighSkewness: false);
        skewness.Fit(MomentData());

        Assert.NotNull(kurtosis.KurtosisValues);
        Assert.Equal(-1.958662332151795, kurtosis.KurtosisValues[0], Tolerance);
        Assert.Equal(2.495359954619423, kurtosis.KurtosisValues[1], Tolerance);
        Assert.Equal(-0.976915974145891, kurtosis.KurtosisValues[2], Tolerance);
        Assert.Equal(new[] { 0, 1 }, kurtosis.SelectedIndices);

        Assert.NotNull(skewness.SkewnessValues);
        Assert.Equal(0.0, skewness.SkewnessValues[0], Tolerance);
        Assert.Equal(1.986214288094205, skewness.SkewnessValues[1], Tolerance);
        Assert.Equal(-0.23713871415241208, skewness.SkewnessValues[2], Tolerance);
        Assert.Equal(new[] { 0 }, skewness.SelectedIndices);
    }
}
