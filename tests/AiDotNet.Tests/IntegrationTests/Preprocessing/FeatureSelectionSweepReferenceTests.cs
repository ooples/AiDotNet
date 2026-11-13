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

    [Fact(Timeout = 120000)]
    public async Task ColumnStd_Float_IsTwoPassForNearlyEqualValues()
    {
        // numpy.std([4096 - 2^-11, 4096 + 2^-11], dtype=float32) = 2^-11. One pass loses it: the float squares round
        // to 2^24 - 4 and 2^24 + 4, so E[x^2] - E[x]^2 = 2^24 - 2^24 = 0 (or below 0, NaN after the square root).
        await Task.CompletedTask;
        float d = (float)Math.Pow(2, -11);
        var x = new Tensor<float>(new[] { 4096f - d, 4096f + d }, new[] { 2, 1 });

        var std = AiDotNet.Preprocessing.FeatureSelection.FeatureSelectionTensorOps.ColumnStd(x);

        Assert.Equal(Math.Pow(2, -11), std.GetFlat(0), Tolerance);
    }

    // ---- Review fixes: Half (and other narrow-T) traps in the converted selectors ----

    private static Matrix<TNum> Build<TNum>(int n, Func<double, TNum> convert, params Func<int, double>[] columns)
    {
        var m = new Matrix<TNum>(n, columns.Length);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < columns.Length; j++)
                m[i, j] = convert(columns[j](i));
        return m;
    }

    private static Vector<TNum> BuildVector<TNum>(int n, Func<double, TNum> convert, Func<int, double> value)
    {
        var v = new Vector<TNum>(n);
        for (int i = 0; i < n; i++)
            v[i] = convert(value(i));
        return v;
    }

    private static Matrix<Half> HalfMatrix(int n, params Func<int, double>[] columns) => Build(n, d => (Half)d, columns);
    private static Vector<Half> HalfVector(int n, Func<int, double> value) => BuildVector(n, d => (Half)d, value);
    private static Matrix<double> DoubleMatrix(int n, params Func<int, double>[] columns) => Build(n, d => d, columns);
    private static Vector<double> DoubleVector(int n, Func<int, double> value) => BuildVector(n, d => d, value);

    private static double[] ToDoubles(Vector<Half>? values)
    {
        Assert.NotNull(values);
        return values.ToArray().Select(h => (double)h).ToArray();
    }

    private static void AssertAllClose(double[] expected, double[] actual, double tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int j = 0; j < expected.Length; j++)
            Assert.True(Math.Abs(expected[j] - actual[j]) <= tolerance,
                $"[{j}]: expected {expected[j]}, actual {actual[j]} (tolerance {tolerance})");
    }

    // 500 rows of class 0 then 2500 of class 1 (the ImbalancedClasses split) as a target vector
    private static double ImbalancedLabel(int i) => i < 500 ? 0 : 1;

    private static readonly double[] RidgeColumn0 = { 1, 2, 3, 4, 5, 6 };
    private static readonly double[] RidgeColumn1 = { 2, 1, 2, 1, 2, 1 };
    private static readonly double[] RidgeTarget = { 1, 3, 2, 5, 4, 6 };

    [Fact(Timeout = 120000)]
    public async Task RidgeFS_Half_SolvesTheNormalEquationsInsteadOfThrowing()
    {
        // numpy.linalg.solve(Xc'Xc + I, Xc'yc) for the centred data = [0.761364, -0.943182]. The engine Cholesky
        // runs only in float and double; a Half system threw NotSupportedException.
        await Task.CompletedTask;
        var selector = new AiDotNet.Preprocessing.FeatureSelection.Embedded.RidgeFS<Half>(nFeaturesToSelect: 1, alpha: 1.0);
        selector.Fit(HalfMatrix(6, i => RidgeColumn0[i], i => RidgeColumn1[i]), HalfVector(6, i => RidgeTarget[i]));

        AssertAllClose(new[] { 0.761364, -0.943182 }, ToDoubles(selector.Coefficients), 2e-3);
    }

    [Fact(Timeout = 120000)]
    public async Task RidgeSelector_Half_MatchesTheDoubleSolveInsteadOfThrowing()
    {
        // The LU solve runs only in float and double; a Half system threw NotSupportedException.
        await Task.CompletedTask;
        var reference = new RidgeSelector(nFeaturesToSelect: 1, alpha: 1.0);
        reference.Fit(DoubleMatrix(6, i => RidgeColumn0[i], i => RidgeColumn1[i]), DoubleVector(6, i => RidgeTarget[i]));
        var selector = new AiDotNet.Preprocessing.FeatureSelection.Embedded.RidgeSelector<Half>(nFeaturesToSelect: 1, alpha: 1.0);
        selector.Fit(HalfMatrix(6, i => RidgeColumn0[i], i => RidgeColumn1[i]), HalfVector(6, i => RidgeTarget[i]));

        Assert.NotNull(reference.Coefficients);
        AssertAllClose(reference.Coefficients.ToArray(), ToDoubles(selector.Coefficients), 1e-2);
    }

    [Fact(Timeout = 120000)]
    public async Task PartialCorrelation_Half_MatchesTheDoublePrecisionMatrix()
    {
        // The precision matrix comes from an LU factorization, which ran in T and threw for Half.
        await Task.CompletedTask;
        double[] a = { 1, 2, 3, 4, 5, 6, 7, 8 };
        double[] b = { 2, 1, 4, 3, 6, 5, 8, 7 };
        double[] c = { 1, 1, 2, 3, 5, 8, 13, 21 };
        double[] y = { 3, 2, 6, 7, 9, 12, 16, 22 };
        var reference = new AiDotNet.Preprocessing.FeatureSelection.Filter.Correlation.PartialCorrelation<double>(nFeaturesToSelect: 1);
        reference.Fit(DoubleMatrix(8, i => a[i], i => b[i], i => c[i]), DoubleVector(8, i => y[i]));
        var selector = new AiDotNet.Preprocessing.FeatureSelection.Filter.Correlation.PartialCorrelation<Half>(nFeaturesToSelect: 1);
        selector.Fit(HalfMatrix(8, i => a[i], i => b[i], i => c[i]), HalfVector(8, i => y[i]));

        Assert.NotNull(reference.PartialCorrelations);
        AssertAllClose(reference.PartialCorrelations.ToArray(), ToDoubles(selector.PartialCorrelations), 1e-2);
    }

    [Fact(Timeout = 120000)]
    public async Task SpearmanCorrelation_Half_RanksPastTheHalfSumOfSquaresLimit()
    {
        // scipy.stats.spearmanr(range(400), range(400)) = 1. The centred rank sum of squares n (n^2 - 1) / 12 =
        // 5.3e6 overflows Half (max 65504), which turned the correlation into inf / inf = NaN.
        await Task.CompletedTask;
        var selector = new AiDotNet.Preprocessing.FeatureSelection.Filter.Correlation.SpearmanCorrelation<Half>(nFeaturesToSelect: 1);
        selector.Fit(HalfMatrix(400, i => i), HalfVector(400, i => i));

        Assert.Equal(1.0, ToDoubles(selector.CorrelationScores)[0], 2e-3);
    }

    [Fact(Timeout = 120000)]
    public async Task CorrelationHelpers_Half_DoNotOverflowTheSumsOfSquares()
    {
        // numpy.corrcoef(range(400), 2 * range(400)) = 1; sum (x - mean)^2 = 5.3e6 overflows Half.
        await Task.CompletedTask;
        var x = HalfTensor(new[] { 400, 1 }, i => i);
        var y = HalfTensor(new[] { 400 }, i => i);
        var pair = HalfTensor(new[] { 400, 2 }, i => (i % 2 + 1) * (i / 2));

        Assert.Equal(1.0, (double)AiDotNet.Preprocessing.PreprocessingTensorOps.CorrelationsWithDenominatorGuard(x, y).GetFlat(0), 2e-3);
        Assert.Equal(1.0, (double)AiDotNet.Preprocessing.FeatureSelection.FeatureSelectionTensorOps.CorrelationMatrix(pair).GetFlat(1), 2e-3);
    }

    [Fact(Timeout = 120000)]
    public async Task ColumnStd_Half_SumsMoreThan2048Deviations()
    {
        // numpy.std([0, 2] * 1500) = 1 and the sum of squared deviations is 3000. Summed in Half the unit squares
        // stop growing at 2048 (2049 rounds back to 2048).
        await Task.CompletedTask;
        var x = HalfTensor(new[] { 3000, 1 }, i => 2 * (i % 2));

        Assert.Equal(3000.0, (double)AiDotNet.Preprocessing.FeatureSelection.FeatureSelectionTensorOps.ColumnSumSquaredDeviations(x).GetFlat(0), 2.0);
        Assert.Equal(1.0, (double)AiDotNet.Preprocessing.FeatureSelection.FeatureSelectionTensorOps.ColumnStd(x).GetFlat(0), 1e-3);
    }

    [Fact(Timeout = 120000)]
    public async Task DistanceCorrelation_Half_VarianceProductDoesNotOverflow()
    {
        // dCor(x, x) = 1 (Szekely, Rizzo and Bakirov 2007). For x = 0..99, dVar^2(x) = 444.56, so the product of the
        // two distance variances (197630) overflows Half, and the mean of the 10000 squared centred distances
        // (sum 4.4e6) overflowed before that.
        await Task.CompletedTask;
        var correlationSelector = new AiDotNet.Preprocessing.FeatureSelection.Filter.Correlation.DistanceCorrelation<Half>(nFeaturesToSelect: 1);
        correlationSelector.Fit(HalfMatrix(100, i => i), HalfVector(100, i => i));
        var filterSelector = new AiDotNet.Preprocessing.FeatureSelection.Filter.DistanceCorrelation<Half>(nFeaturesToSelect: 1);
        filterSelector.Fit(HalfMatrix(100, i => i), HalfVector(100, i => i));

        Assert.Equal(1.0, ToDoubles(correlationSelector.DistCorrelations)[0], 2e-3);
        Assert.Equal(1.0, ToDoubles(filterSelector.DistanceCorrelations)[0], 2e-3);
    }

    [Fact(Timeout = 120000)]
    public async Task DiverseEnsemble_Half_DistanceCorrelationMethodCounts()
    {
        // Two features, so every method normalizes to exactly 1 for the better feature and 0 for the other. x0 = y
        // wins the correlation, mutual information and distance correlation methods; the F test (100 classes, no
        // within-class degrees of freedom) and the variance (x1 is a permutation of x0) have zero range and are
        // skipped. Average over the 5 methods: x0 = 3 / 5. The Half distance correlation overflowed to NaN and
        // dropped out, giving 2 / 5.
        await Task.CompletedTask;
        var selector = new AiDotNet.Preprocessing.FeatureSelection.Ensemble.DiverseEnsemble<Half>(nFeaturesToSelect: 1, randomState: 7);
        selector.Fit(HalfMatrix(100, i => i, i => (i * 37) % 100), HalfVector(100, i => i));

        var scores = ToDoubles(selector.AggregatedScores);
        Assert.Equal(0.6, scores[0], 1e-3);
        Assert.Equal(0.0, scores[1], 1e-3);
    }

    [Fact(Timeout = 120000)]
    public async Task BootstrapSelector_Half_CorrelationDenominatorDoesNotOverflow()
    {
        // x1 = y = +/-17, so corr(x1, y) = 1 on every bootstrap sample; x0 = 0, 0.1, 0.2 repeating is nearly
        // uncorrelated with y. The variance product 289 * 289 = 83521 overflows Half, which made corr(x1, y) = 0 and
        // let x0 win every bootstrap.
        await Task.CompletedTask;
        var selector = new AiDotNet.Preprocessing.FeatureSelection.CrossValidation.BootstrapSelector<Half>(
            nFeaturesToSelect: 1, nBootstraps: 20, sampleRatio: 0.8, randomState: 1);
        selector.Fit(HalfMatrix(60, i => 0.1 * (i % 3), i => i % 2 == 0 ? 17 : -17), HalfVector(60, i => i % 2 == 0 ? 17 : -17));

        AssertAllClose(new[] { 0.0, 1.0 }, ToDoubles(selector.StabilityScores), 1e-9);
    }

    [Fact(Timeout = 120000)]
    public async Task EmbeddedPermutationImportance_Half_DefaultScorerDoesNotOverflow()
    {
        // y = x (0..49 twice) is predicted exactly, so R^2 = 1. sxx = syy = 20825 and their product overflows Half,
        // which zeroed the weights and gave R^2 = 0.
        await Task.CompletedTask;
        var selector = new AiDotNet.Preprocessing.FeatureSelection.Embedded.PermutationImportance<Half>(
            nFeaturesToSelect: 1, nRepeats: 1, randomState: 42);
        selector.Fit(HalfMatrix(100, i => i % 50), HalfVector(100, i => i % 50));

        Assert.Equal(1.0, (double)selector.BaselineScore, 1e-2);
    }

    [Fact(Timeout = 120000)]
    public async Task DropColumnImportance_Half_CrossValidatedR2DoesNotOverflow()
    {
        // y = x (0..49 twice): every fold's predictor is exact, so the mean R^2 is 1. sxx * yVar * nTrain
        // (16660 * 208 * 80) overflows Half, which zeroed the correlations and gave a negative R^2.
        await Task.CompletedTask;
        var selector = new AiDotNet.Preprocessing.FeatureSelection.Embedded.DropColumnImportance<Half>(nFeaturesToSelect: 1);
        selector.Fit(HalfMatrix(100, i => i % 50), HalfVector(100, i => i % 50));

        Assert.Equal(1.0, (double)selector.BaselineScore, 1e-2);
    }

    [Fact(Timeout = 120000)]
    public async Task NormalizedCompressionSelector_Half_CountsTheJointHistogramPast2048()
    {
        // x = y = class (500 / 2500): MI = H(y) = 0.650022 bits. The Half joint-count matmul stopped at 2048.
        await Task.CompletedTask;
        var selector = new NormalizedCompressionSelector<Half>(nFeaturesToSelect: 1, nBins: 2);
        selector.Fit(HalfMatrix(3000, ImbalancedLabel), HalfVector(3000, ImbalancedLabel));

        Assert.Equal(0.650022, ToDoubles(selector.NCDScores)[0], 2e-3);
    }

    [Fact(Timeout = 120000)]
    public async Task TextFeatureSelector_Half_CountsDocumentsPast2048()
    {
        // The term occurs in the 2500 class-1 documents: document frequency 2500 / 3000 = 0.833333 and, perfectly
        // associated with the class, chi^2 = n phi^2 = 3000 (scipy.stats.chi2_contingency(correction=False)). Half
        // sums of the 0/1 presence stopped at 2048.
        await Task.CompletedTask;
        var selector = new AiDotNet.Preprocessing.FeatureSelection.DomainSpecific.TextFeatureSelector<Half>(nFeaturesToSelect: 1);
        selector.Fit(HalfMatrix(3000, ImbalancedLabel), HalfVector(3000, ImbalancedLabel));

        Assert.Equal(2500.0 / 3000.0, ToDoubles(selector.DocumentFrequencies)[0], 1e-3);
        Assert.Equal(3000.0, ToDoubles(selector.FeatureScores)[0], 2.0);
    }

    [Fact(Timeout = 120000)]
    public async Task FClassification_Half_CountsClassSizesPast2048()
    {
        // Class 0: 500 rows of 1. Class 1: 2500 rows of 0 except 10, -10, 10, -10.
        // sklearn.feature_selection.f_classif gives F = 3122.916667. A Half class size of 2048 gives 3028.8.
        await Task.CompletedTask;
        var selector = new AiDotNet.Preprocessing.FeatureSelection.Filter.Classification.FClassification<Half>(nFeaturesToSelect: 1);
        selector.Fit(HalfMatrix(3000, i => i < 500 ? 1 : i < 504 ? (i % 2 == 0 ? 10 : -10) : 0), HalfVector(3000, ImbalancedLabel));

        Assert.Equal(3122.916667, ToDoubles(selector.FScores)[0], 10.0);
    }

    [Fact(Timeout = 120000)]
    public async Task MutualInfoClassification_Half_CountsClassSizesPast2048()
    {
        // x = class: MI = H(y) = 0.450561 nats. A Half class probability of 2048 / 3000 gave 0.617.
        await Task.CompletedTask;
        var selector = new AiDotNet.Preprocessing.FeatureSelection.Filter.Classification.MutualInfoClassification<Half>(nFeaturesToSelect: 1);
        selector.Fit(HalfMatrix(3000, ImbalancedLabel), HalfVector(3000, ImbalancedLabel));

        Assert.Equal(0.450561, ToDoubles(selector.MIScores)[0], 2e-3);
    }

    [Fact(Timeout = 120000)]
    public async Task GiniImpuritySelection_Half_CountsSplitSidesPast2048()
    {
        // Class 0: 250 rows at 0, 250 at 1. Class 1: 100 at 0, 2400 at 1. Every threshold splits {0} | {1}:
        // parent Gini 0.277778 minus the weighted child Gini 0.198563 = 0.079215. A Half class total of 2048
        // gives 0.082452.
        await Task.CompletedTask;
        var selector = new AiDotNet.Preprocessing.FeatureSelection.Filter.GiniImpuritySelection<Half>(nFeaturesToSelect: 1);
        selector.Fit(HalfMatrix(3000, i => i < 250 || (i >= 500 && i < 600) ? 0 : 1), HalfVector(3000, ImbalancedLabel));

        Assert.Equal(0.079215, ToDoubles(selector.GiniReductionScores)[0], 5e-4);
    }

    [Fact(Timeout = 120000)]
    public async Task KnockoffFilter_Half_CountsTheKnockoffPlusTermsPast2048()
    {
        // W = 1 for 2300 features and -1 for 200. At t = 1 the knockoff+ estimate is (1 + 200) / 2300 = 0.0874 <=
        // fdr = 0.09, so the threshold is 1. A Half count of 2048 positives gives 201 / 2048 = 0.098 > 0.09 and no
        // valid threshold.
        await Task.CompletedTask;
        const int p = 2500;
        var importances = new Vector<Half>(2 * p);
        for (int j = 0; j < p; j++)
        {
            importances[j] = (Half)(j < 2300 ? 1 : 0);
            importances[p + j] = (Half)(j < 2300 ? 0 : 1);
        }

        var selector = new AiDotNet.Preprocessing.FeatureSelection.Filter.HighDimensional.KnockoffFilter<Half>(
            fdr: 0.09, importanceFunc: (_, _) => importances, randomState: 3);
        var data = new Matrix<Half>(2, p);
        for (int j = 0; j < p; j++)
            data[1, j] = (Half)1;
        selector.Fit(data, HalfVector(2, i => i));

        Assert.Equal(1.0, (double)selector.Threshold, Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task EnsembleFS_Half_RanksPast2048StayExact()
    {
        // Each of the 4 base methods votes for exactly its top 2060 ranks, so the votes sum to 4 * 2060 = 8240.
        // Ranks held in Half round 2059 up to 2060, so only 2059 ranks per method passed rank < 2060.
        await Task.CompletedTask;
        const int p = 2100;
        var selector = new AiDotNet.Preprocessing.FeatureSelection.Ensemble.EnsembleFS<Half>(nFeaturesToSelect: 2060);
        selector.Fit(new Matrix<Half>(10, p), HalfVector(10, i => i));

        Assert.Equal(8240.0, ToDoubles(selector.AggregatedScores).Sum(), Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task EnsembleFeatureSelector_Half_BordaPointsPast2048StayExact()
    {
        // One selection [1, 0, 2, 3, ...] of length 2100: feature 1 gets 2100 Borda points and feature 0 gets
        // 2099. In Half 2099 rounds to 2100, the two tie, and the lower index (0) was selected.
        await Task.CompletedTask;
        const int p = 2100;
        var order = new[] { 1, 0 }.Concat(Enumerable.Range(2, p - 2)).ToArray();
        var selector = new AiDotNet.Preprocessing.FeatureSelection.Ensemble.EnsembleFeatureSelector<Half>(
            nFeaturesToSelect: 1, selectors: new System.Collections.Generic.List<Func<Matrix<Half>, Vector<Half>, int[]>> { (_, _) => order });
        selector.Fit(new Matrix<Half>(2, p), HalfVector(2, i => i));

        Assert.Equal(new[] { 1 }, selector.SelectedIndices);
    }

    // 1 at every 10th position, else 0
    private static double SparseSpikes(int i) => i % 10 == 0 ? 1 : 0;

    [Fact(Timeout = 120000)]
    public async Task ApproximateEntropySelector_Half_CountsTemplateMatchesPast2048()
    {
        // Pincus (1991) ApEn(m = 2, r = 0.2 std) of 3000 samples of the spike train = 0.301062 (direct double
        // evaluation). About 2400 templates match the all-zero template; summed in Half that stopped at 2048 (0.19).
        await Task.CompletedTask;
        var selector = new AiDotNet.Preprocessing.FeatureSelection.Entropy.ApproximateEntropySelector<Half>(nFeaturesToSelect: 1);
        selector.Fit(HalfMatrix(3000, SparseSpikes));

        Assert.Equal(0.301062, ToDoubles(selector.ApEnValues)[0], 2e-3);
    }

    [Fact(Timeout = 120000)]
    public async Task SampleEntropySelector_Half_CountsTemplatePairsPast2048()
    {
        // Richman and Moorman (2000) SampEn(m = 2, r = 0.2 std) of the first 300 spike-train samples = 0.233166
        // (direct double evaluation). Both pair counts exceed 2048; summed in Half they stopped at the same value
        // and gave -ln(1) = 0.
        await Task.CompletedTask;
        var selector = new AiDotNet.Preprocessing.FeatureSelection.Entropy.SampleEntropySelector<Half>(nFeaturesToSelect: 1);
        selector.Fit(HalfMatrix(300, SparseSpikes));

        Assert.Equal(0.233166, ToDoubles(selector.SampEnValues)[0], 2e-3);
    }

    [Fact(Timeout = 120000)]
    public async Task PermutationEntropySelector_Half_CountsPatternsPast2048()
    {
        // x = i mod 10 (3000 samples, m = 3): 2400 increasing windows and 299 / 299 of the two wrap-around patterns.
        // Normalized permutation entropy (Bandt and Pompe 2002) = 0.356031. The increasing-pattern count stopped at
        // 2048 in Half.
        await Task.CompletedTask;
        var selector = new AiDotNet.Preprocessing.FeatureSelection.Entropy.PermutationEntropySelector<Half>(nFeaturesToSelect: 1);
        selector.Fit(HalfMatrix(3000, i => i % 10));

        Assert.Equal(0.356031, ToDoubles(selector.EntropyValues)[0], 1e-3);
    }

    // Quadratic residues mod 11: an integer series exact in Half with an irregular pattern
    private static double QuadraticResidue(int i) => (double)((long)i * i % 11);

    [Fact(Timeout = 120000)]
    public async Task LyapunovSelector_Half_MatchesDoubleBeyond2048Points()
    {
        // About 10000 divergence terms are averaged: counted in Half they stopped at 2048, and time indices above
        // 2048 held in Half broke the temporal separation test.
        await Task.CompletedTask;
        const int n = 2200;
        var reference = new AiDotNet.Preprocessing.FeatureSelection.Complexity.LyapunovSelector<double>(nFeaturesToSelect: 1);
        reference.Fit(DoubleMatrix(n, QuadraticResidue));
        var selector = new AiDotNet.Preprocessing.FeatureSelection.Complexity.LyapunovSelector<Half>(nFeaturesToSelect: 1);
        selector.Fit(HalfMatrix(n, QuadraticResidue));

        Assert.NotNull(reference.LyapunovValues);
        AssertAllClose(reference.LyapunovValues.ToArray(), ToDoubles(selector.LyapunovValues), 1e-2);
    }

    [Fact(Timeout = 120000)]
    public async Task FractalDimensionSelector_Half_CurveLengthNormalizerDoesNotOverflow()
    {
        // Higuchi's normalizer (n - 1) / (k^2 numSegments): at k = 10, k^2 numSegments = 100 * 699 = 69900
        // overflows Half, which made L(10) = 0 and the log-log slope non-finite.
        await Task.CompletedTask;
        const int n = 7000;
        var reference = new AiDotNet.Preprocessing.FeatureSelection.Complexity.FractalDimensionSelector<double>(nFeaturesToSelect: 1);
        reference.Fit(DoubleMatrix(n, QuadraticResidue));
        var selector = new AiDotNet.Preprocessing.FeatureSelection.Complexity.FractalDimensionSelector<Half>(nFeaturesToSelect: 1);
        selector.Fit(HalfMatrix(n, QuadraticResidue));

        Assert.NotNull(reference.FractalDimensions);
        AssertAllClose(reference.FractalDimensions.ToArray(), ToDoubles(selector.FractalDimensions), 3e-2);
    }

    [Fact(Timeout = 120000)]
    public async Task StabilitySelection_Half_CountsSelectionsPast2048Iterations()
    {
        // The base selector always picks feature 0, so its selection probability is 2100 / 2100 = 1. A Half count
        // stopped at 2048 (0.975).
        await Task.CompletedTask;
        var selector = new AiDotNet.Preprocessing.FeatureSelection.Ensemble.StabilitySelection<Half>(
            nFeaturesToSelect: 1, nIterations: 2100, baseSelector: (_, _, _) => new[] { 0 }, randomState: 5);
        selector.Fit(HalfMatrix(4, i => i, i => -i), HalfVector(4, i => i));

        Assert.Equal(1.0, ToDoubles(selector.SelectionProbabilities)[0], Tolerance);
    }

    [Fact(Timeout = 120000)]
    public async Task HierarchicalFS_Half_MergeIndicesPast2048StayExact()
    {
        // Features 2049 and 2051 are identical (distance 0) and merge first; feature 5 = 10 v + noise is next closest
        // to that cluster, so the second merge adds 5 and leaves feature 2048 (an unrelated low-variance column) on its
        // own. Positions held in Half map 2049 to 2048, so the first merge also overwrote feature 2048's distances
        // with the cluster's, merged 5 with 2048, and dropped 2048 for the higher-variance 5.
        await Task.CompletedTask;
        const int n = 12;
        const int p = 2052;
        var random = RandomHelper.CreateSeededRandom(11);
        var values = new double[n, p];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                values[i, j] = random.NextDouble() * 2 - 1;
        for (int i = 0; i < n; i++)
        {
            values[i, 2051] = values[i, 2049];
            values[i, 5] = 10 * values[i, 2049] + values[i, 7];
            values[i, 2048] *= 0.1;
        }

        var data = new Matrix<Half>(n, p);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                data[i, j] = (Half)values[i, j];

        var selector = new AiDotNet.Preprocessing.FeatureSelection.Clustering.HierarchicalFS<Half>(nFeaturesToSelect: p - 2);
        selector.Fit(data);

        Assert.NotNull(selector.SelectedIndices);
        Assert.Contains(2048, selector.SelectedIndices);
        Assert.Contains(5, selector.SelectedIndices);
        Assert.DoesNotContain(2049, selector.SelectedIndices);
        Assert.DoesNotContain(2051, selector.SelectedIndices);
    }

    [Fact(Timeout = 120000)]
    public async Task ImageFeatureSelector_Half_PixelIndicesPast2048StayExact()
    {
        // 50 x 50 image. Pixel 2049 = y scores 1.0, its neighbour 2048 = y / 2 scores 0.85 and pixel 0 = y / 4 scores
        // 0.775 (0.7 |r| + 0.3 std). Picking 2049 penalizes 2048 to 0.85 (1 - exp(-0.5) / 2) = 0.592, so the top two
        // are 0 and 2049. Positions held in Half map 2049 to 2048, so 2048 was marked picked and never penalized.
        await Task.CompletedTask;
        const int p = 2500;
        var data = new Matrix<Half>(2, p);
        data[1, 2049] = (Half)2;
        data[1, 2048] = (Half)1;
        data[1, 0] = (Half)0.5;
        var selector = new AiDotNet.Preprocessing.FeatureSelection.DomainSpecific.ImageFeatureSelector<Half>(
            nFeaturesToSelect: 2, imageWidth: 50, imageHeight: 50);
        selector.Fit(data, HalfVector(2, i => 2 * i));

        Assert.Equal(new[] { 0, 2049 }, selector.SelectedIndices);
    }

    [Fact(Timeout = 120000)]
    public async Task GenomicFeatureSelector_Half_ClassMeansDoNotOverflow()
    {
        // Class 0 (500 rows) at 15, class 1 (2500 rows) alternating 29 / 31: log2 fold change log2(30 / 15) = 1.
        // The Half class-1 sum (75000) overflowed to infinity.
        await Task.CompletedTask;
        var selector = new AiDotNet.Preprocessing.FeatureSelection.DomainSpecific.GenomicFeatureSelector<Half>(nFeaturesToSelect: 1);
        selector.Fit(HalfMatrix(3000, i => i < 500 ? 15 : (i % 2 == 0 ? 29 : 31)), HalfVector(3000, ImbalancedLabel));

        Assert.Equal(1.0, ToDoubles(selector.FoldChanges)[0], 2e-3);
    }

    [Fact(Timeout = 120000)]
    public async Task PointBiserial_Half_ClassMeansDoNotOverflow()
    {
        // Class 0 alternates 0 / 2 (500 rows), class 1 alternates 30 / 32 (2500 rows): r_pb = 0.996024
        // (|m1 - m0| / std * sqrt(n0 n1) / n). The Half class-1 sum (77500) overflowed to infinity.
        await Task.CompletedTask;
        var selector = new AiDotNet.Preprocessing.FeatureSelection.Filter.Correlation.PointBiserial<Half>(nFeaturesToSelect: 1);
        selector.Fit(HalfMatrix(3000, i => i < 500 ? 2 * (i % 2) : 30 + 2 * (i % 2)), HalfVector(3000, ImbalancedLabel));

        Assert.Equal(0.996024, ToDoubles(selector.Correlations)[0], 2e-3);
    }

    [Fact(Timeout = 120000)]
    public async Task NaiveBayesSelector_Half_ClassMeansDoNotOverflow()
    {
        // Class 0: {0, 2, 0, 2} (mean 1, variance 1). Class 1: 2500 rows alternating 30 / 32 (mean 31, variance 1).
        // Symmetric Gaussian KL = 0.5 (KL01 + KL10) = (31 - 1)^2 / 2 = 450. The Half class-1 sum (77500) overflowed.
        await Task.CompletedTask;
        const int n = 2504;
        var selector = new AiDotNet.Preprocessing.FeatureSelection.Classification.NaiveBayesSelector<Half>(nFeaturesToSelect: 1);
        selector.Fit(HalfMatrix(n, i => i < 4 ? 2 * (i % 2) : 30 + 2 * (i % 2)), HalfVector(n, i => i < 4 ? 0 : 1));

        Assert.NotNull(selector.DiscriminativePower);
        Assert.Equal(450.0, (double)selector.DiscriminativePower[0], 1.0);
    }

    [Fact(Timeout = 120000)]
    public async Task SilhouetteBasedSelector_Half_CountsClassSizesPast2048()
    {
        // Class 0: two rows at 3. Class 1: 1500 rows at 0 and 1000 at 1.
        // sklearn.metrics.silhouette_score = 0.800080. A Half class size of 2048 gives 0.756.
        await Task.CompletedTask;
        const int n = 2502;
        var selector = new SilhouetteBasedSelector<Half>(nFeaturesToSelect: 1);
        selector.Fit(HalfMatrix(n, i => i < 2 ? 3 : i < 1502 ? 0 : 1), HalfVector(n, i => i < 2 ? 0 : 1));

        Assert.Equal(0.800080, ToDoubles(selector.SilhouetteScores)[0], 2e-3);
    }

    [Fact(Timeout = 120000)]
    public async Task FairFeatureSelector_Half_CountsGroupsPast2048()
    {
        // Protected 0 for 2500 rows (median 0, so they form the lower group) and 1 for 500. Feature mean: lower
        // 1000 / 2500 = 0.4, upper 400 / 500 = 0.8, disparate impact min(0.5, 2) = 0.5. A Half group size of 2048
        // gives 0.4883 / 0.8 = 0.61.
        await Task.CompletedTask;
        var data = HalfMatrix(3000, i => i < 2500 ? 0 : 1, i => i < 100 ? 10 : i >= 2500 && i < 2600 ? 4 : 0);
        var selector = new FairFeatureSelector<Half>(nFeaturesToSelect: 1, protectedAttributeIndex: 0);

        Assert.Equal(0.5, (double)selector.GetDisparateImpact(data, 1), 1e-3);
    }
}
