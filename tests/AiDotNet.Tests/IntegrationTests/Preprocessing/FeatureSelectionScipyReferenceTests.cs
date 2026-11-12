using AiDotNet.Preprocessing.FeatureSelection.Moment;
using AiDotNet.Preprocessing.FeatureSelection.Morphological;
using AiDotNet.Preprocessing.FeatureSelection.NLP;
using AiDotNet.Preprocessing.FeatureSelection.Quantile;
using AiDotNet.Preprocessing.FeatureSelection.Range;
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

    [Fact]
    public void QuantileBasedSelectors_MatchNumpyLinearPercentiles()
    {
        // scipy.stats.iqr(X, axis=0); scipy.stats.median_abs_deviation(X, axis=0)
        // q1, q3 = np.percentile(X, [25, 75], axis=0); qcd = (q3 - q1) / abs(q3 + q1)
        // lo, hi = np.percentile(X, [10, 90], axis=0); range = hi - lo; ratio = abs(hi) / abs(lo)
        var iqr = new IQRSelector<double>(nFeaturesToSelect: 1);
        iqr.Fit(MomentData());
        var mad = new MADSelector<double>(nFeaturesToSelect: 1);
        mad.Fit(MomentData());
        var qcd = new QuantileDispersionSelector<double>(nFeaturesToSelect: 1);
        qcd.Fit(MomentData());
        var range = new PercentileRangeSelector<double>(nFeaturesToSelect: 1);
        range.Fit(MomentData());
        var ratio = new QuantileRatioSelector<double>(nFeaturesToSelect: 1);
        ratio.Fit(MomentData());

        AssertValues(new[] { 7.1499999999999995, 0.8499999999999996, 1.75 }, iqr.IQRValues);
        AssertValues(new[] { 3.65, 0.3999999999999999, 1.0 }, mad.MADValues);
        AssertValues(new[] { 0.715, 0.17525773195876282, 0.3181818181818182 }, qcd.QCDValues);
        AssertValues(new[] { 7.7200000000000015, 3.0999999999999988, 2.95 }, range.RangeValues);
        AssertValues(new[] { 7.771929824561405, 2.823529411764705, 3.458333333333333 }, ratio.RatioValues);
        Assert.Equal(new[] { 0 }, iqr.SelectedIndices);
        Assert.Equal(new[] { 0 }, ratio.SelectedIndices);
    }

    [Fact]
    public void MorphologicalSelectors_MatchScipyNdimageFlatStructuringElements()
    {
        // gradient: (grey_dilation(c, size=3, mode='nearest') - grey_erosion(c, size=3, mode='nearest')).mean()
        // granulometry: c normalized to [0, 1]; for ws in 1..5 the opening grey_opening(c, size=2*(ws//2)+1,
        // mode='nearest') gives the pattern spectrum ps[ws] = sum(previous) - sum(opening); score = entropy(ps, base 2)
        var gradient = new ErosionDilationSelector<double>(nFeaturesToSelect: 1, windowSize: 3);
        gradient.Fit(MomentData());
        var granulometry = new GranulometrySelector<double>(nFeaturesToSelect: 1, maxWindowSize: 5);
        granulometry.Fit(MomentData());

        AssertValues(new[] { 5.0, 3.325, 1.0 }, gradient.MorphologicalScores);
        AssertValues(new[] { 0.9148210087144519, 0.9239841591414172, 0.9182958340544896 }, granulometry.GranulometryScores);
        Assert.Equal(new[] { 0 }, gradient.SelectedIndices);
        Assert.Equal(new[] { 1 }, granulometry.SelectedIndices);
    }

    [Fact]
    public void TextSelectors_MatchScikitLearnTfidfAndDocumentFrequency()
    {
        // TfidfTransformer(smooth_idf=False, norm=None).fit_transform(C).mean(axis=0); df = (C > 0).sum(axis=0)
        var counts = new Matrix<double>(new double[,]
        {
            { 3, 0, 1, 0 },
            { 2, 0, 0, 1 },
            { 0, 1, 0, 1 },
            { 4, 0, 2, 1 },
            { 0, 0, 0, 1 },
        });
        var tfidf = new TfIdfFS<double>(nFeaturesToSelect: 2);
        tfidf.Fit(counts);
        var frequency = new DocumentFrequencyFS<double>(nFeaturesToSelect: 3, minDf: 2, maxDf: 3);
        frequency.Fit(counts);

        AssertValues(new[] { 2.7194861227787834, 0.5218875824868201, 1.149774439124493, 0.9785148410513678 }, tfidf.TfIdfScores);
        Assert.Equal(new[] { 0, 2 }, tfidf.SelectedIndices);
        Assert.Equal(new[] { 3, 1, 2, 4 }, frequency.DocumentFrequencies);
        Assert.Equal(new[] { 0, 2 }, frequency.SelectedIndices);
    }

    private static void AssertValues(double[] expected, Vector<double>? actual)
    {
        Assert.NotNull(actual);
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], Tolerance);
    }
}
