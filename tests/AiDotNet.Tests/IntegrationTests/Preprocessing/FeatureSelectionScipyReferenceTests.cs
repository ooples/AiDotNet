using AiDotNet.Preprocessing.FeatureSelection.Kernel;
using AiDotNet.Preprocessing.FeatureSelection.Manifold;
using AiDotNet.Preprocessing.FeatureSelection.Moment;
using AiDotNet.Preprocessing.FeatureSelection.Morphological;
using AiDotNet.Preprocessing.FeatureSelection.NLP;
using AiDotNet.Preprocessing.FeatureSelection.Nonparametric;
using AiDotNet.Preprocessing.FeatureSelection.Projection;
using AiDotNet.Preprocessing.FeatureSelection.Quantile;
using AiDotNet.Preprocessing.FeatureSelection.Range;
using AiDotNet.Preprocessing.FeatureSelection.Redundancy;
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

    // A 2-D spiral (columns 0 and 1) plus a small noise column; geodesic distances along the spiral differ
    // from the straight-line ones.
    private static Matrix<double> SpiralData() => new Matrix<double>(new double[,]
    {
        { 0.439, 0.24, 0.3 },
        { 0.118, 1.487, -0.2 },
        { -1.964, 1.519, 0.5 },
        { -3.284, -1.137, 0.1 },
        { -1.087, -4.332, -0.4 },
        { 3.704, -4.009, 0.2 },
        { 6.36, 1.07, 0.0 },
        { 2.983, 6.817, -0.1 },
        { -4.616, 7.058, 0.4 },
        { -9.425, 0.0, -0.3 },
    });

    [Fact]
    public void IsomapSelectors_UseGeodesicDistancesOfTheKnnGraph()
    {
        // geo = scipy.sparse.csgraph.shortest_path(kneighbors_graph(X, 3, mode='distance'), directed=False)
        // IsomapSelector: corrcoef(|x_i - x_j|, geo_ij) over i < j for each column;
        // IsomapBasedSelector: mean(|x_i - x_j| * geo_ij) over i < j, divided by its maximum
        var isomap = new IsomapSelector<double>(nFeaturesToSelect: 1, nNeighbors: 3);
        isomap.Fit(SpiralData());
        var isomapBased = new IsomapBasedSelector<double>(nFeaturesToSelect: 2, nNeighbors: 3);
        isomapBased.Fit(SpiralData());

        AssertValues(new[] { 0.7230409799705054, 0.47619987520071577, -0.08682100377897621 }, isomap.ManifoldScores, 1e-9);
        AssertValues(new[] { 1.0, 0.7785377100203102, 0.05534937113140767 }, isomapBased.FeatureScores, 1e-9);
        Assert.Equal(new[] { 0 }, isomap.SelectedIndices);
        Assert.Equal(new[] { 0, 1 }, isomapBased.SelectedIndices);
    }

    [Fact]
    public void LleSelectors_UseScikitLearnBarycenterWeights()
    {
        // W = sklearn.manifold._locally_linear.barycenter_kneighbors_graph(X, k, reg=1e-3).toarray()
        // LLEBasedSelector (k = 4): mean(1 / ((X - W X)**2 + 1e-10), axis=0)
        // LLESelector (k = 3): mean(1 / (1 + |X - W X|), axis=0)
        var lleBased = new LLEBasedSelector<double>(nFeaturesToSelect: 1, nNeighbors: 4);
        lleBased.Fit(SpiralData());
        var lle = new LLESelector<double>(nFeaturesToSelect: 1, nNeighbors: 3);
        lle.Fit(SpiralData());

        var based = lleBased.FeatureScores;
        Assert.NotNull(based);
        var expectedBased = new[] { 2379.942822908662, 164185.0579222155, 157.612105775534 };
        for (int i = 0; i < expectedBased.Length; i++)
            Assert.Equal(1.0, based[i] / expectedBased[i], 1e-7);
        AssertValues(new[] { 0.7860009082860653, 0.7460063540614652, 0.5726743102664745 }, lle.LocalLinearityScores, 1e-9);
        Assert.Equal(new[] { 1 }, lleBased.SelectedIndices);
        Assert.Equal(new[] { 0 }, lle.SelectedIndices);
    }

    [Fact]
    public void LaplacianScoreSelector_MatchesHeCaiNiyogiFormula()
    {
        // W = A * exp(-||x_i - x_j||^2 / (2 sigma^2)), A = symmetrized kneighbors_graph(X, 3); D = W 1;
        // f~ = f - (f'D1 / 1'D1); score = f~'(D - W)f~ / f~'D f~   (He, Cai and Niyogi 2005)
        var laplacian = new LaplacianScoreSelector<double>(nFeaturesToSelect: 1, nNeighbors: 3, sigma: 1.0);
        laplacian.Fit(SpiralData());

        AssertValues(new[] { 0.8526099922737964, 1.6058679731353886, 1.8084026457213203 }, laplacian.LaplacianScores, 1e-9);
        Assert.Equal(new[] { 0 }, laplacian.SelectedIndices);
    }

    [Fact]
    public void ProjectionSelectors_MatchScikitLearnPcaAndPrincipalAxisFactoring()
    {
        // pca = sklearn.decomposition.PCA(n_components=2).fit(X)
        // PCAVarianceSelector: (explained_variance_ / explained_variance_.sum()) @ components_**2
        // SVDBasedSelector: (s**2 / (s**2).sum()) @ components_**2 with s = singular_values_
        // FactorAnalysisSelector: R = corrcoef with diag max|r_jk|^2; top-2 eigh(R): (V**2) @ max(lambda, 0)
        var pca = new PCAVarianceSelector<double>(nFeaturesToSelect: 1, nComponents: 2);
        pca.Fit(SpiralData());
        var svd = new SVDBasedSelector<double>(nFeaturesToSelect: 1, nComponents: 2);
        svd.Fit(SpiralData());
        var factor = new FactorAnalysisSelector<double>(nFeaturesToSelect: 2, nFactors: 2);
        factor.Fit(SpiralData());

        AssertValues(new[] { 0.5856747286422438, 0.4140705686003831, 0.0002547027573734128 }, pca.VarianceContributions, 1e-9);
        AssertValues(new[] { 0.5856747286420579, 0.41407056860025176, 0.00025470275737333195 }, svd.FeatureScores, 1e-9);
        AssertValues(new[] { 0.03798463307245823, 0.1911529724857474, 0.19576025573400804 }, factor.Communalities, 1e-9);
        Assert.Equal(new[] { 0 }, pca.SelectedIndices);
        Assert.Equal(new[] { 0 }, svd.SelectedIndices);
        Assert.Equal(new[] { 1, 2 }, factor.SelectedIndices);
    }

    private static Vector<double> SpiralClasses() => new Vector<double>(new[] { 0.0, 0, 1, 1, 2, 2, 0, 1, 2, 0 });

    [Fact]
    public void KernelSelectors_MatchNumpyKernelStatistics()
    {
        // k(a, b) = exp(-0.5 (a - b)^2) per feature; H = I - 1/n
        // HSIC: trace(Kx H Ky H) / (n-1)^2; MMD: mean over class pairs of biased MMD^2;
        // Fisher: between/within scatter of the kernel row sums; RBF/poly alignment on z-scored columns
        // over the pairs i < j; kernel PCA: sklearn KernelPCA(2, kernel='rbf', gamma=0.5).eigenvectors_ V,
        // importance_j = sum_k sum_ii' |V_ik V_i'k| (x_ij - x_i'j)^2 / n^2
        var x = SpiralData();
        var y = SpiralClasses();
        var hsic = new HSICSelector<double>(nFeaturesToSelect: 1, gammaX: 0.5, gammaY: 0.5);
        hsic.Fit(x, y);
        var mmd = new MMDBasedSelector<double>(nFeaturesToSelect: 1, gamma: 0.5);
        mmd.Fit(x, y);
        var fisher = new KernelFisherSelector<double>(nFeaturesToSelect: 1, gamma: 0.5);
        fisher.Fit(x, y);
        var rbf = new RBFKernelSelector<double>(nFeaturesToSelect: 1, gamma: 0.5);
        rbf.Fit(x, y);
        var poly = new PolynomialKernelSelector<double>(nFeaturesToSelect: 1, degree: 3, coef0: 1.0);
        poly.Fit(x, y);
        var kernelPca = new KernelPCASelector<double>(nFeaturesToSelect: 1, gamma: 0.5, nComponents: 2);
        kernelPca.Fit(x);

        AssertValues(new[] { 0.04674325635262055, 0.08544211390443947, 0.0016084374149811273 }, hsic.HSICScores, 1e-9);
        AssertValues(new[] { 0.5464649312352773, 0.8066777822109324, 0.020898895933488593 }, mmd.MMDScores, 1e-9);
        AssertValues(new[] { 0.04976180952869986, 3.4641453265995543, 0.10499430026916166 }, fisher.FisherScores, 1e-9);
        AssertValues(new[] { -3.254204875043939, -1.8410813889749136, -2.963950548001502 }, rbf.KernelScores, 1e-9);
        AssertValues(new[] { 1.0341152591004186, 1.1676654798342967, 1.5716128973860009 }, poly.KernelScores, 1e-9);
        AssertValues(new[] { 3.588857017343745, 2.4779158031076856, 0.017252035195986903 }, kernelPca.FeatureImportances, 1e-8);
        Assert.Equal(new[] { 1 }, hsic.SelectedIndices);
        Assert.Equal(new[] { 0 }, rbf.SelectedIndices);
        Assert.Equal(new[] { 2 }, poly.SelectedIndices);
        Assert.Equal(new[] { 0 }, kernelPca.SelectedIndices);
    }

    [Fact]
    public void KendallTauSelector_UsesTieCorrectedTauB()
    {
        // abs(scipy.stats.kendalltau(X[:, d], y).statistic) - tau-b; the class target has many ties, so the
        // tie-free tau-a (concordant - discordant) / (n (n - 1) / 2) would be smaller
        var kendall = new KendallTauSelector<double>(nFeaturesToSelect: 1);
        kendall.Fit(SpiralData(), SpiralClasses());

        AssertValues(new[] { 0.0778498944161523, 0.0778498944161523, 0.18164975363768868 }, kendall.TauScores, 1e-12);
        Assert.Equal(new[] { 2 }, kendall.SelectedIndices);
    }

    [Fact]
    public void DistanceCorrelationSelectors_MatchSzekelyDistanceCorrelation()
    {
        // A, B = double-centred |x_i - x_j| and |y_i - y_j|; dCor = sqrt(mean(A*B)) / sqrt(sqrt(mean(A*A) * mean(B*B)))
        // (Szekely, Rizzo and Bakirov 2007; the same value as dcor.distance_correlation)
        var nonlinear = new AiDotNet.Preprocessing.FeatureSelection.Nonlinear.DistanceCorrelationSelector<double>(nFeaturesToSelect: 1);
        nonlinear.Fit(SpiralData(), SpiralClasses());
        var nonparametric = new DistanceCorrelationSelector<double>(nFeaturesToSelect: 1);
        nonparametric.Fit(SpiralData(), SpiralClasses());

        var expected = new[] { 0.33735611474717664, 0.5793364927353657, 0.36195935229966436 };
        AssertValues(expected, nonlinear.DCorValues, 1e-12);
        AssertValues(expected, nonparametric.DcorScores, 1e-12);
        Assert.Equal(new[] { 1 }, nonlinear.SelectedIndices);
        Assert.Equal(new[] { 1 }, nonparametric.SelectedIndices);
    }

    [Fact]
    public void RedundancySelectors_MatchGreedyNumpyReference()
    {
        // Columns: the spiral data plus column 3 = 2 * column 1 + 0.1 * column 2 (redundant with column 1).
        // relevance = |corrcoef(x_j, y)|, C = |corrcoef(X)|; mRMR picks argmax relevance, then repeatedly
        // argmax(relevance_j - mean_s C[j, s]); clustering walks features by relevance and lets each
        // unassigned one absorb the unassigned features with C >= 0.9.
        var spiral = SpiralData();
        var x = new Matrix<double>(spiral.Rows, 4);
        for (int i = 0; i < spiral.Rows; i++)
        {
            for (int j = 0; j < 3; j++)
                x[i, j] = spiral[i, j];
            x[i, 3] = 2 * spiral[i, 1] + 0.1 * spiral[i, 2];
        }

        var mrmr = new MinimalRedundancySelector<double>(nFeaturesToSelect: 3);
        mrmr.Fit(x, SpiralClasses());
        var cluster = new CorrelationClusterSelector<double>(nFeaturesToSelect: 2, correlationThreshold: 0.9);
        cluster.Fit(x, SpiralClasses());

        AssertValues(new[] { -0.0758795489570683, -0.06823528274167653, 0.18860838403857944, 0.0 }, mrmr.MRMRScores, 1e-12);
        Assert.Equal(new[] { 0, 1, 2 }, mrmr.SelectedIndices);
        AssertValues(new[] { 0.0, 0.10691055606830033, 0.18860838403857944, 0.0 }, cluster.ClusterRepresentativeScores, 1e-12);
        Assert.Equal(new[] { 1, 2 }, cluster.SelectedIndices);
    }

    private static void AssertValues(double[] expected, Vector<double>? actual, double tolerance = Tolerance)
    {
        Assert.NotNull(actual);
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], tolerance);
    }
}
