using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Preprocessing.FeatureSelection.Clustering;
using AiDotNet.Preprocessing.FeatureSelection.Compression;
using AiDotNet.Preprocessing.FeatureSelection.Entropy;
using AiDotNet.Preprocessing.FeatureSelection.Fairness;
using AiDotNet.Preprocessing.FeatureSelection.Filter.InformationTheory;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using BorutaFS = AiDotNet.Preprocessing.FeatureSelection.Ensemble.BorutaFS<double>;
using ChiSquaredSelection = AiDotNet.Preprocessing.FeatureSelection.Filter.ChiSquaredSelection<double>;
using ClassificationChiSquare = AiDotNet.Preprocessing.FeatureSelection.Filter.Classification.ChiSquare<double>;
using DiverseEnsemble = AiDotNet.Preprocessing.FeatureSelection.Ensemble.DiverseEnsemble<double>;
using ElasticNetFS = AiDotNet.Preprocessing.FeatureSelection.Embedded.ElasticNetFS<double>;
using EmbeddedPermutationImportance = AiDotNet.Preprocessing.FeatureSelection.Embedded.PermutationImportance<double>;
using FClassification = AiDotNet.Preprocessing.FeatureSelection.Filter.Classification.FClassification<double>;
using FilterDistanceCorrelation = AiDotNet.Preprocessing.FeatureSelection.Filter.DistanceCorrelation<double>;
using FilterKnockoffFilter = AiDotNet.Preprocessing.FeatureSelection.Filter.HighDimensional.KnockoffFilter<double>;
using KendallTauSelector = AiDotNet.Preprocessing.FeatureSelection.Filter.Correlation.KendallTau<double>;
using RidgeSelector = AiDotNet.Preprocessing.FeatureSelection.Embedded.RidgeSelector<double>;
using ScadSelector = AiDotNet.Preprocessing.FeatureSelection.Embedded.SCAD<double>;
using TreeBasedFS = AiDotNet.Preprocessing.FeatureSelection.Embedded.TreeBasedFS<double>;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// Reference tests for the feature-selection bugs fixed in the element-wise loop sweep. Expected values come
/// from scipy / scikit-learn (quoted in each test) or from the closed form of the published method.
/// </summary>
public class FeatureSelectionSweepReferenceTests
{
    private const double Tolerance = 1e-9;

    private static Matrix<double> Column(params double[] values)
    {
        var m = new Matrix<double>(values.Length, 1);
        for (int i = 0; i < values.Length; i++)
            m[i, 0] = values[i];
        return m;
    }

    private static Matrix<double> Columns(params double[][] columns)
    {
        var m = new Matrix<double>(columns[0].Length, columns.Length);
        for (int j = 0; j < columns.Length; j++)
            for (int i = 0; i < columns[j].Length; i++)
                m[i, j] = columns[j][i];
        return m;
    }

    private static Vector<double> V(params double[] values) => Vector<double>.FromArray(values);

    [Fact(Timeout = 120000)]
    public async Task SilhouetteBasedSelector_SingletonClassScoresZero()
    {
        // sklearn.metrics.silhouette_score([[0], [1], [10]], [0, 0, 1]) = 0.5962962962962962:
        // s = 0.9, 8/9 and 0 for the sample alone in its cluster (Rousseeuw 1987).
        var selector = new SilhouetteBasedSelector<double>(nFeaturesToSelect: 1);
        selector.Fit(Column(0, 1, 10), V(0, 0, 1));

        Assert.NotNull(selector.SilhouetteScores);
        Assert.Equal(0.5962962962962962, selector.SilhouetteScores[0], Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task NormalizedCompressionSelector_ConstantFeatureHasZeroMutualInformation()
    {
        // MI(X; Y) = 0 for a constant X (sklearn mutual_info_score); the old code returned H(Y) = 1 bit.
        var data = Columns(new double[] { 5, 5, 5, 5 }, new double[] { 0, 1, 0, 1 });
        var selector = new NormalizedCompressionSelector<double>(nFeaturesToSelect: 1, nBins: 2);
        selector.Fit(data, V(0, 1, 0, 1));

        Assert.NotNull(selector.NCDScores);
        Assert.Equal(0.0, selector.NCDScores[0], Tolerance);
        Assert.Equal(1.0, selector.NCDScores[1], Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task ElasticNetFS_MatchesSklearnElasticNetOnStandardizedFeature()
    {
        // sklearn ElasticNet(alpha=0.1, l1_ratio=0.5).fit(z, 2x) with z the standardized x = 1..5 gives
        // coef = 2.6461...; divided by std(x) = sqrt(2) this is 1.8710901532768311.
        var selector = new ElasticNetFS(nFeaturesToSelect: 1, alpha: 0.1, l1Ratio: 0.5, maxIterations: 1000, tolerance: 1e-12);
        selector.Fit(Column(1, 2, 3, 4, 5), V(2, 4, 6, 8, 10));

        Assert.NotNull(selector.Coefficients);
        Assert.Equal(1.8710901532768311, selector.Coefficients[0], 1e-8);
    }

    [Fact(Timeout = 120000)]
    public async Task Scad_UsesSoftThresholdUpToTwiceLambda()
    {
        // Fan and Li (2001): for |z| <= 2*lambda the SCAD solution is soft thresholding. With x = 1..5 and
        // y = c (x - 3), z = x_std . y / n = c * sqrt(2); c = 1.5 / sqrt(2) gives z = 1.5 and lambda = 1 -> 0.5.
        double c = 1.5 / Math.Sqrt(2);
        var selector = new ScadSelector(nFeaturesToSelect: 1, lambda: 1.0, a: 3.7, maxIterations: 100, tolerance: 1e-12);
        selector.Fit(Column(1, 2, 3, 4, 5), V(-2 * c, -c, 0, c, 2 * c));

        Assert.NotNull(selector.Coefficients);
        Assert.Equal(0.5, selector.Coefficients[0], 1e-9);
    }

    [Fact(Timeout = 120000)]
    public async Task Scad_UsesTransitionFormulaBetweenTwiceLambdaAndALambda()
    {
        // 2*lambda < |z| = 3 <= a*lambda = 3.7: ((a - 1) z - sign(z) a lambda) / (a - 2) = 2.588235294117648.
        double c = 3.0 / Math.Sqrt(2);
        var selector = new ScadSelector(nFeaturesToSelect: 1, lambda: 1.0, a: 3.7, maxIterations: 100, tolerance: 1e-12);
        selector.Fit(Column(1, 2, 3, 4, 5), V(-2 * c, -c, 0, c, 2 * c));

        Assert.NotNull(selector.Coefficients);
        Assert.Equal(2.588235294117648, selector.Coefficients[0], 1e-9);
    }

    [Fact(Timeout = 120000)]
    public async Task RidgeSelector_SingularNormalEquationsGiveMinimumNormSolution()
    {
        // Two identical standardized columns with alpha = 0: numpy.linalg.lstsq returns the minimum-norm
        // solution [sqrt(2), sqrt(2)] for y = 2x, x = 1..5.
        var x = new double[] { 1, 2, 3, 4, 5 };
        var selector = new RidgeSelector(nFeaturesToSelect: 2, alpha: 0.0);
        selector.Fit(Columns(x, x), V(2, 4, 6, 8, 10));

        Assert.NotNull(selector.Coefficients);
        Assert.Equal(Math.Sqrt(2), selector.Coefficients[0], 1e-8);
        Assert.Equal(Math.Sqrt(2), selector.Coefficients[1], 1e-8);
    }

    [Fact(Timeout = 120000)]
    public async Task EmbeddedPermutationImportance_DefaultScorerGivesPerfectR2ForLinearTarget()
    {
        // y = 3x + 1 is predicted exactly by yMean + r * z * yStd (r = 1), so the baseline R^2 is 1.
        var selector = new EmbeddedPermutationImportance(nFeaturesToSelect: 1, nRepeats: 1, randomState: 42);
        selector.Fit(Column(1, 2, 3, 4, 5, 6), V(4, 7, 10, 13, 16, 19));

        Assert.Equal(1.0, selector.BaselineScore, 1e-9);
    }

    [Fact(Timeout = 120000)]
    public async Task TreeBasedFS_BinaryFeatureCanSplit()
    {
        // A binary feature equal to the class: the only boundary (between 0 and 1) is a perfect Gini split,
        // so all importance goes to it (sklearn DecisionTreeClassifier.feature_importances_ = [1]).
        var x = Enumerable.Range(0, 20).Select(i => i < 10 ? 0.0 : 1.0).ToArray();
        var selector = new TreeBasedFS(nFeaturesToSelect: 1, maxDepth: 1, minSamplesLeaf: 1, randomState: 7);
        selector.Fit(Column(x), V(x));

        Assert.NotNull(selector.FeatureImportances);
        Assert.Equal(1.0, selector.FeatureImportances[0], Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task BorutaFS_PercentileThresholdUsesPositiveNormalQuantile()
    {
        // 20 iterations, percentile 99: threshold = 10 + z(0.99) * sqrt(5) = 10 + 2.3263 * 2.2361 = 15.2 hits
        // (scipy.stats.norm.ppf(0.99) = 2.3263). Feature 0 beats the shadows in 10 iterations, feature 1 in all
        // 20, so only feature 1 is confirmed. The old z(0.99) = -2.34 confirmed both (threshold 4.8).
        int call = 0;
        Vector<double> Importances(Matrix<double> data, Vector<double> target)
        {
            double first = call++ % 2 == 0 ? 1.0 : 0.0;
            return V(first, 1.0, 0.5, 0.5);
        }

        var data = Columns(new double[] { 1, 2, 3, 4 }, new double[] { 4, 3, 2, 1 });
        var selector = new BorutaFS(nIterations: 20, percentile: 99.0, getImportances: Importances, randomState: 1);
        selector.Fit(data, V(1, 2, 3, 4));

        Assert.NotNull(selector.SelectedIndices);
        Assert.Equal(new[] { 1 }, selector.SelectedIndices);
    }

    [Fact(Timeout = 120000)]
    public async Task DiverseEnsemble_DistanceCorrelationIsScaleInvariant()
    {
        // Feature 1 = 1000 * feature 0 = 1000 * y. Correlation, MI, F and distance correlation (dCor is scale
        // invariant, Szekely et al. 2007) tie, so only the variance method separates them: scores [0, 1/5].
        // The old 'distance correlation' was dCov, which grows with scale and gave [0, 2/5].
        var y = new double[] { 1, 3, 2, 5, 4, 7, 6, 9, 8, 10 };
        var scaled = y.Select(v => 1000 * v).ToArray();
        var selector = new DiverseEnsemble(nFeaturesToSelect: 1, randomState: 3);
        selector.Fit(Columns(y, scaled), V(y));

        Assert.NotNull(selector.AggregatedScores);
        Assert.Equal(0.0, selector.AggregatedScores[0], 1e-9);
        Assert.Equal(0.2, selector.AggregatedScores[1], 1e-9);
    }

    [Fact(Timeout = 120000)]
    public async Task TsallisEntropySelector_QEqualsOneIsShannonEntropyInNats()
    {
        // lim_{q->1} (1 - sum p^q) / (q - 1) = -sum p ln p; two equally likely bins give ln 2.
        var selector = new TsallisEntropySelector<double>(nFeaturesToSelect: 1, nBins: 2, q: 1.0);
        selector.Fit(Column(0, 0, 1, 1));

        Assert.NotNull(selector.TsallisEntropies);
        Assert.Equal(Math.Log(2), selector.TsallisEntropies[0], Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task SampleEntropySelector_CountsAAndBOverTheSameTemplates()
    {
        // Richman and Moorman (2000) with m = 2, r = 0.2 * std (population): over the same n - m = 12
        // templates B = 10 and A = 6 pairs match, SampEn = -ln(6/10) = 0.5108256237659907.
        var selector = new SampleEntropySelector<double>(nFeaturesToSelect: 1, embeddingDimension: 2, tolerance: 0.2);
        selector.Fit(Column(1, 2, 3, 1, 2, 4, 1, 2, 3, 1, 2, 4, 1, 2));

        Assert.NotNull(selector.SampEnValues);
        Assert.Equal(0.5108256237659907, selector.SampEnValues[0], Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task FairFeatureSelector_DisparateImpactSplitsAtTheSortedMedian()
    {
        // Protected attribute = feature = [10, 1, 2, 3, 4, 5]; the n/2-th smallest value is 4, so the groups
        // are {1, 2, 3, 4} (mean 2.5) and {10, 5} (mean 7.5): disparate impact 2.5 / 7.5 = 1/3.
        var selector = new FairFeatureSelector<double>(nFeaturesToSelect: 1, protectedAttributeIndex: 0);

        Assert.Equal(1.0 / 3.0, selector.GetDisparateImpact(Column(10, 1, 2, 3, 4, 5), 0), Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task ClassificationChiSquare_PValueIsChiSquaredSurvival()
    {
        // Perfect 2x2 table, n = 8: chi2 = 8 with df = 1, scipy.stats.chi2.sf(8, 1) = 0.004677734981047276.
        var x = new double[] { 0, 0, 0, 0, 1, 1, 1, 1 };
        var selector = new ClassificationChiSquare(nFeaturesToSelect: 1, nBins: 2);
        selector.Fit(Column(x), V(x));

        Assert.NotNull(selector.ChiSquareScores);
        Assert.NotNull(selector.PValues);
        Assert.Equal(8.0, selector.ChiSquareScores[0], Tolerance);
        Assert.Equal(0.004677734981047276, selector.PValues[0], 1e-12);
    }

    [Fact(Timeout = 120000)]
    public async Task ChiSquaredSelection_PValueIsExactChiSquaredSurvival()
    {
        // Same table through the default 10 bins (two occupied, df = 1): chi2.sf(8, 1) = 0.004677734981047276,
        // as sklearn.feature_selection.chi2.
        var x = new double[] { 0, 0, 0, 0, 1, 1, 1, 1 };
        var selector = new ChiSquaredSelection(nFeaturesToSelect: 1, nBins: 10);
        selector.Fit(Column(x), V(x));

        Assert.NotNull(selector.PValues);
        Assert.Equal(0.004677734981047276, selector.PValues[0], 1e-12);
    }

    [Fact(Timeout = 120000)]
    public async Task FClassification_PValueIsExactFSurvival()
    {
        // Groups {1, 2, 3} and {4, 5, 6}: F = 13.5 on (1, 4) df; scipy.stats.f.sf(13.5, 1, 4) = 0.021311641128756713
        // (sklearn f_classif).
        var selector = new FClassification(nFeaturesToSelect: 1);
        selector.Fit(Column(1, 2, 3, 4, 5, 6), V(0, 0, 0, 1, 1, 1));

        Assert.NotNull(selector.FScores);
        Assert.NotNull(selector.PValues);
        Assert.Equal(13.5, selector.FScores[0], 1e-9);
        Assert.Equal(0.021311641128756713, selector.PValues[0], 1e-10);
    }

    [Fact(Timeout = 120000)]
    public async Task FilterDistanceCorrelation_ReturnsDistanceCorrelationNotItsSquare()
    {
        // x = 1..5, y = x^2: dCor = sqrt(dCov^2 / sqrt(dVar^2(x) dVar^2(y))) = 0.9869160440537484
        // (V-statistic, Szekely, Rizzo and Bakirov 2007; numpy reference).
        var selector = new FilterDistanceCorrelation(nFeaturesToSelect: 1);
        selector.Fit(Column(1, 2, 3, 4, 5), V(1, 4, 9, 16, 25));

        Assert.NotNull(selector.DistanceCorrelations);
        Assert.Equal(0.9869160440537484, selector.DistanceCorrelations[0], Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task KendallTau_TauBCountsJointTiesInBothTieTerms()
    {
        // scipy.stats.kendalltau([1, 1, 2, 3], [1, 1, 3, 2]).statistic = 0.6.
        var selector = new KendallTauSelector(nFeaturesToSelect: 1, useTauB: true);
        selector.Fit(Column(1, 1, 2, 3), V(1, 1, 3, 2));

        Assert.NotNull(selector.TauScores);
        Assert.Equal(0.6, selector.TauScores[0], Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task KnockoffFilter_KnockoffPlusThresholdIsTheSmallestQualifyingT()
    {
        // W = [4, 3, 2, 1], q = 0.5: (1 + #{W <= -t}) / max(1, #{W >= t}) is 1, 0.5, 1/3, 1/4 for t = 4, 3, 2, 1,
        // so the knockoff+ threshold (Barber and Candes 2015) is min{t : ratio <= q} = 1 and all four are kept.
        Vector<double> Importances(Matrix<double> data, Vector<double> target) => V(4, 3, 2, 1, 0, 0, 0, 0);

        var data = Columns(
            new double[] { 1, 2, 3, 4, 5 }, new double[] { 2, 1, 4, 3, 5 },
            new double[] { 5, 4, 3, 2, 1 }, new double[] { 1, 3, 5, 2, 4 });
        var selector = new FilterKnockoffFilter(fdr: 0.5, importanceFunc: Importances, randomState: 5);
        selector.Fit(data, V(1, 2, 3, 4, 5));

        Assert.Equal(1.0, selector.Threshold, Tolerance);
        Assert.Equal(new[] { 0, 1, 2, 3 }, selector.SelectedIndices);
    }

    [Fact(Timeout = 120000)]
    public async Task AdjustedMutualInformation_MatchesSklearnWithExactExpectedMutualInformation()
    {
        // Feature values 0..9 fall one per bin (10 range-scaled bins), so the score is
        // sklearn.metrics.adjusted_mutual_info_score(y, x, average_method='max') = 0.03951269381827764.
        var x = new double[] { 0, 9, 3, 3, 7, 9, 3, 5, 2, 4, 7, 6, 8, 8, 1, 6, 7, 7, 8, 1,
                               5, 9, 8, 9, 4, 3, 0, 3, 5, 0, 2, 3, 8, 1, 3, 3, 3, 7, 0, 1 };
        var y = new double[] { 1, 1, 0, 2, 0, 2, 2, 0, 2, 0, 0, 0, 1, 1, 2, 0, 0, 1, 0, 1,
                               2, 2, 0, 1, 1, 1, 1, 2, 2, 2, 0, 2, 1, 0, 1, 2, 0, 0, 2, 0 };
        var selector = new AdjustedMutualInformation<double>(nFeaturesToSelect: 1, nBins: 10);
        selector.Fit(Column(x), V(y));

        Assert.NotNull(selector.AMIScores);
        Assert.Equal(0.03951269381827764, selector.AMIScores[0], 1e-9);
    }

    private static Tensor<Half> HalfTensor(int[] shape, Func<int, double> value)
    {
        int length = shape.Aggregate(1, (a, b) => a * b);
        var data = new Half[length];
        for (int i = 0; i < length; i++)
            data[i] = (Half)value(i);
        return new Tensor<Half>(data, shape);
    }

    // 500 rows of class 0 then 2500 of class 1: a Half sum of the 0/1 one-hot stops at 2048.
    private static int[] ImbalancedClasses() => Enumerable.Range(0, 3000).Select(i => i < 500 ? 0 : 1).ToArray();

    [Fact(Timeout = 120000)]
    public async Task KendallTau_Half_SumsMoreThan2048PairSigns()
    {
        // scipy.stats.kendalltau(range(100), range(100)).statistic = 1 (tau-a and tau-b): 9900 ordered pairs, all
        // concordant. Summed in Half the +1 signs stop at 2048.
        await Task.CompletedTask;
        var x = HalfTensor(new[] { 100, 1 }, i => i);
        var y = HalfTensor(new[] { 100 }, i => i);

        Assert.Equal(1.0, (double)AiDotNet.Preprocessing.FeatureSelection.FeatureSelectionTensorOps.KendallTau(x, y, tauB: false).GetFlat(0), Tolerance);
        Assert.Equal(1.0, (double)AiDotNet.Preprocessing.FeatureSelection.FeatureSelectionTensorOps.KendallTau(x, y, tauB: true).GetFlat(0), Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task ClassStatistics_Half_CountMoreThan2048PerClass()
    {
        // Class sizes 500 and 2500 of 3000: Gini 1 - (1/6)^2 - (5/6)^2 = 0.277778 and entropy
        // -(1/6 ln 1/6 + 5/6 ln 5/6) = 0.450561 nats (scipy.stats.entropy([500, 2500])). A bin one-hot equal to the
        // classes is perfectly associated, so scipy.stats.chi2_contingency(correction=False) gives n phi^2 = 3000.
        await Task.CompletedTask;
        var classes = ImbalancedClasses();
        var oneHot = AiDotNet.Preprocessing.PreprocessingTensorOps.ClassOneHot<Half>(classes, 2);

        Assert.Equal(0.277778, (double)AiDotNet.Preprocessing.FeatureSelection.FeatureSelectionTensorOps.GiniImpurity(oneHot), 1e-3);
        Assert.Equal(0.450561, (double)AiDotNet.Preprocessing.FeatureSelection.FeatureSelectionTensorOps.ClassEntropyNats(oneHot), 1e-3);

        var counts = AiDotNet.Preprocessing.FeatureSelection.FeatureSelectionTensorOps.BinCounts(oneHot, 2);
        Assert.Equal(500.0, (double)counts.GetFlat(0), Tolerance);
        Assert.Equal(2500.0, (double)counts.GetFlat(1), Tolerance);

        var chiSquared = AiDotNet.Preprocessing.FeatureSelection.FeatureSelectionTensorOps.ContingencyChiSquared(oneHot, oneHot, 2, out var occupied);
        Assert.Equal(3000.0, (double)chiSquared.GetFlat(0), 2.0);
        Assert.Equal(2.0, (double)occupied.GetFlat(0), Tolerance);

        // The split between the classes leaves two pure children, so the reduction is the parent Gini
        var feature = HalfTensor(new[] { 3000, 1 }, i => classes[i]);
        Assert.Equal(0.277778, (double)AiDotNet.Preprocessing.FeatureSelection.FeatureSelectionTensorOps.BestGiniSplitReduction(feature, classes, 2).GetFlat(0), 1e-3);
    }

    [Fact(Timeout = 120000)]
    public async Task MeanPairwiseDifference_Half_ManyRowsDoNotOverflowThePairCount()
    {
        // Rows 0..299: the mean |i - j| over pairs is (k + 1) / 3 = 100.333. The ordered pair count
        // k (k - 1) = 89700 overflows Half (max 65504).
        await Task.CompletedTask;
        var rows = HalfTensor(new[] { 300, 1 }, i => i);

        Assert.Equal(301.0 / 3.0, (double)AiDotNet.Preprocessing.FeatureSelection.FeatureSelectionTensorOps.MeanPairwiseDifference(rows, squared: false).GetFlat(0), 0.1);
    }

    [Fact(Timeout = 120000)]
    public async Task NaiveBayesSelector_Half_CountsMoreThan2048PerClass()
    {
        // Class 0: {0, 2, 0, 2} (mean 1, variance 1). Class 1: 2500 rows, two at 8, two at -8, the rest 0 (mean 0,
        // variance 256 / 2500 = 0.1024). Symmetric Gaussian KL = 0.5 (KL01 + KL10) with
        // KL01 = 0.5 (ln 0.1024 + 1 / 0.1024 + 1 / 0.1024 - 1) = 8.126190 and KL10 = 0.5 (-ln 0.1024 + 0.1024) =
        // 1.190635, so 4.658412. A Half class count of 2048 gives variance 0.125 and 3.78.
        await Task.CompletedTask;
        const int n = 2504;
        var data = new Matrix<Half>(n, 1);
        var target = new Vector<Half>(n);
        for (int i = 0; i < n; i++)
        {
            double value = i < 4 ? (i % 2) * 2 : i < 6 ? 8 : i < 8 ? -8 : 0;
            data[i, 0] = (Half)value;
            target[i] = (Half)(i < 4 ? 0 : 1);
        }

        var selector = new AiDotNet.Preprocessing.FeatureSelection.Classification.NaiveBayesSelector<Half>(nFeaturesToSelect: 1);
        selector.Fit(data, target);

        Assert.NotNull(selector.DiscriminativePower);
        Assert.Equal(4.658412, (double)selector.DiscriminativePower[0], 0.02);
    }

    [Fact(Timeout = 120000)]
    public async Task TreeBasedImportance_ImportanceStdIsAPopulationStdNeverNaN()
    {
        // numpy.std(importances, axis=0) is a mean squared deviation, never negative. E[x^2] - E[x]^2 can cancel to
        // a tiny negative value when the trees agree, and its square root is NaN.
        await Task.CompletedTask;
        const int n = 60;
        var data = new Matrix<double>(n, 2);
        var target = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            data[i, 0] = i % 2 == 0 ? 0.1 * i : 10 + 0.1 * i;
            data[i, 1] = (i * 7) % 5;
            target[i] = i % 2;
        }

        var selector = new AiDotNet.Preprocessing.FeatureSelection.Embedded.TreeBasedImportance<double>(
            nFeaturesToSelect: 1, nTrees: 20, randomState: 3);
        selector.Fit(data, target);

        Assert.NotNull(selector.ImportanceStds);
        for (int j = 0; j < 2; j++)
            Assert.True(selector.ImportanceStds[j] >= 0, $"std[{j}] = {selector.ImportanceStds[j]}");
    }
}
