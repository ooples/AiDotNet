using AiDotNet.Preprocessing.DataPreparation.Splitting.ActiveLearning;
using AiDotNet.Preprocessing.DataPreparation.Splitting.Specialized;
using AiDotNet.Preprocessing.Encoders;
using AiDotNet.Preprocessing.FeatureGeneration;
using AiDotNet.Preprocessing.FeatureSelection.Association;
using AiDotNet.Preprocessing.FeatureSelection.Bayesian;
using AiDotNet.Preprocessing.FeatureSelection.Boundary;
using AiDotNet.Preprocessing.FeatureSelection.Causal;
using AiDotNet.Preprocessing.FeatureSelection.Classification;
using Xunit;
using System.Threading.Tasks;
using BorutaSelector = AiDotNet.Preprocessing.FeatureSelection.Bioinformatics.Boruta<System.Half>;
using SamSelector = AiDotNet.Preprocessing.FeatureSelection.Bioinformatics.SAM<System.Half>;
using SignificanceAnalysisSelector = AiDotNet.Preprocessing.FeatureSelection.Bioinformatics.SignificanceAnalysis<System.Half>;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// Reference tests for the element-wise sweep of the preprocessing encoders, splines, splitters and feature
/// selectors: Half counts past 2048, overflow-free posteriors and model weights, per-category ordered sums,
/// NaN isolation and fitted state that survives a tensor arena.
/// </summary>
public class PreprocessingSweepReviewTests
{
    private const double Tolerance = 1e-9;

    private static Matrix<double> M(double[,] values) => new Matrix<double>(values);

    private static Matrix<Half> HalfColumn(int rows, Func<int, double> value)
    {
        var data = new Half[rows, 1];
        for (int i = 0; i < rows; i++)
            data[i, 0] = (Half)value(i);
        return new Matrix<Half>(data);
    }

    private static Vector<Half> HalfVector(int length, Func<int, double> value)
    {
        var data = new Half[length];
        for (int i = 0; i < length; i++)
            data[i] = (Half)value(i);
        return new Vector<Half>(data);
    }

    private static void ChurnArena(int length)
    {
        using (AiDotNet.Tensors.Helpers.TensorArena.Create())
        {
            var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
            for (int i = 0; i < 64; i++)
            {
                _ = engine.TensorAddScalar(new Tensor<double>(new[] { length }), 1e6 + i);
                _ = engine.TensorAddScalar(new Tensor<double>(new[] { length, 1 }), -1e6 - i);
            }
        }
    }

    // CatBoost ordered target statistics (Prokhorenkova et al. 2018) use, for each row, only the earlier rows of
    // its own category. With prior 0 every category-1 row after the first is the mean of earlier targets of 1,
    // exactly 1; a running sum shared with category 0 (targets 1e17) loses the +1 steps to rounding.
    [Fact(Timeout = 120000)]
    public async Task CatBoostEncoder_OrderedSum_RestartsAtEveryCategory()
    {
        await Task.CompletedTask;
        var data = M(new double[,] { { 0 }, { 0 }, { 0 }, { 1 }, { 1 }, { 1 }, { 1 } });
        var target = new Vector<double>(new double[] { 1e17, 1e17, 1e17, 1, 1, 1, 1 });

        var encoded = new CatBoostEncoder<double>(prior: 0.0).FitTransform(data, target);

        int firstVisits = 0;
        for (int i = 3; i < 7; i++)
        {
            if (double.IsNaN(encoded[i, 0]))
                firstVisits++; // 0 / 0: no earlier row of the category and no prior
            else
                Assert.Equal(1.0, encoded[i, 0], Tolerance);
        }

        Assert.Equal(1, firstVisits);
    }

    // The fitted categories are engine results; they must not alias scratch that a tensor arena hands out again.
    [Fact(Timeout = 120000)]
    public async Task LabelAndOrdinalEncoders_FittedInArena_KeepTheirCategories()
    {
        await Task.CompletedTask;
        var data = M(new double[,] { { 30, 7 }, { 10, 5 }, { 20, 7 }, { 10, 6 } });
        var expectedLabel = new LabelEncoder<double>().FitTransform(data);
        var expectedOrdinal = new OrdinalEncoder<double>().FitTransform(data);

        var label = new LabelEncoder<double>();
        var ordinal = new OrdinalEncoder<double>();
        using (AiDotNet.Tensors.Helpers.TensorArena.Create())
        {
            label.Fit(data);
            ordinal.Fit(data);
        }

        ChurnArena(3);

        var actualLabel = label.Transform(data);
        var actualOrdinal = ordinal.Transform(data);
        for (int i = 0; i < data.Rows; i++)
        {
            for (int j = 0; j < data.Columns; j++)
            {
                Assert.Equal(expectedLabel[i, j], actualLabel[i, j], Tolerance);
                Assert.Equal(expectedOrdinal[i, j], actualOrdinal[i, j], Tolerance);
            }
        }
    }

    // scikit-learn SplineTransformer(knots="quantile") places knots at np.percentile positions q (n - 1) in float64.
    // n = 4000, 3 knots: positions 999.75, 1999.5, 2999.25. The last one lies between sorted[2999] = 0 and
    // sorted[3000] = 1, so the knot is 0.25; in Half the position 2999.25 rounds to 3000 and gives 1.
    [Fact(Timeout = 120000)]
    public async Task SplineTransformer_Half_QuantileKnotsUseExactPositions()
    {
        await Task.CompletedTask;
        var spline = new SplineTransformer<Half>(nKnots: 3, degree: 3, knotStrategy: SplineKnotStrategy.Quantile);
        spline.Fit(HalfColumn(4000, i => i < 3000 ? 0.0 : 1.0));

        Assert.NotNull(spline.Knots);
        var knots = spline.Knots[0];
        Assert.Equal(0.0, (double)knots[4], Tolerance);
        Assert.Equal(0.0, (double)knots[5], Tolerance);
        Assert.Equal(0.25, (double)knots[6], Tolerance);
    }

    // Class means 1.125 and 0.125 with variance 1/64 give t^2 / 2 above the cap of 50, so the posterior is
    // sigmoid(50) = 1. Formed as odds / (1 + odds), exp(50) overflows Half to inf / inf = NaN.
    [Fact(Timeout = 120000)]
    public async Task BayesianFeatureSelector_Half_StrongFeatureHasPosteriorOne()
    {
        await Task.CompletedTask;
        double[] x = { 1, 1.25, 1, 1.25, 0, 0.25, 0, 0.25 };
        var selector = new BayesianFeatureSelector<Half>(nFeaturesToSelect: 1);
        selector.Fit(HalfColumn(8, i => x[i]), HalfVector(8, i => i < 4 ? 1.0 : 0.0));

        Assert.NotNull(selector.PosteriorProbabilities);
        Assert.Equal(1.0, (double)selector.PosteriorProbabilities[0], 1e-3);
    }

    // With slabVariance 0.001 and spikeVariance 1, x = y = 1..8 gives log BF = 0.5 log((vr + 1) / (vr + 0.001))
    // + 0.999 * fit / rv = about 18.7 (vr = 6 / 204, fit = 102, rv = 6), so the inclusion probability is
    // sigmoid(18.7) = 1 - 8e-9. exp(18.7) overflows Half, which made the odds ratio NaN.
    [Fact(Timeout = 120000)]
    public async Task SpikeAndSlabSelector_Half_LargeBayesFactorHasInclusionOne()
    {
        await Task.CompletedTask;
        var selector = new SpikeAndSlabSelector<Half>(nFeaturesToSelect: 1, slabVariance: 0.001, spikeVariance: 1.0);
        selector.Fit(HalfColumn(8, i => i + 1), HalfVector(8, i => i + 1));

        Assert.NotNull(selector.InclusionProbabilities);
        Assert.Equal(1.0, (double)selector.InclusionProbabilities[0], 1e-3);
    }

    // BIC = n log(1 - R^2) + k log n. With n = 400 and R^2 = 0.99, exp(-BIC / 2) = exp(921) overflows double, so
    // absolute weights gave NaN. Relative to the best model the {feature 0} model dominates: feature 0 equals the
    // target and features 1 and 2 are exactly uncorrelated with it.
    [Fact(Timeout = 120000)]
    public async Task BayesianModelAveraging_ManySamples_GivesFiniteInclusionProbabilities()
    {
        await Task.CompletedTask;
        const int n = 400;
        var data = new Matrix<double>(n, 3);
        var target = new Vector<double>(n);
        for (int i = 0; i < n; i++)
        {
            target[i] = i % 2 == 0 ? 1 : -1;
            data[i, 0] = target[i];
            data[i, 1] = (i / 2) % 2 == 0 ? 1 : -1;
            data[i, 2] = (i / 4) % 2 == 0 ? 1 : -1;
        }

        var selector = new BayesianModelAveraging<double>(nFeaturesToSelect: 1);
        selector.Fit(data, target);

        Assert.NotNull(selector.InclusionProbabilities);
        for (int j = 0; j < 3; j++)
        {
            double probability = selector.InclusionProbabilities[j];
            Assert.False(double.IsNaN(probability) || double.IsInfinity(probability));
            Assert.InRange(probability, 0.0, 1.0 + 1e-9);
        }

        Assert.True(selector.InclusionProbabilities[0] > 0.99, $"Got {selector.InclusionProbabilities[0]}");
        Assert.Equal(new[] { 0 }, selector.SelectedIndices);
    }

    // Every column has equal class means (d = 0), so each of the 100 x 30 permuted |d| is at or above every
    // observed |d| and SAM's FDR is (3000 / 100) / 30 = 1. Counted in Half, 3000 saturates at 2048 (FDR 0.68).
    [Fact(Timeout = 120000)]
    public async Task SAM_Half_CountsMoreThan2048FalsePositives()
    {
        await Task.CompletedTask;
        const int p = 30;
        var data = new Half[4, p];
        for (int j = 0; j < p; j++)
        {
            data[0, j] = (Half)0;
            data[1, j] = (Half)1;
            data[2, j] = (Half)0;
            data[3, j] = (Half)1;
        }

        var selector = new SamSelector(nFeaturesToSelect: 1, nPermutations: 100, randomState: 7);
        selector.Fit(new Matrix<Half>(data), HalfVector(4, i => i < 2 ? 0.0 : 1.0));

        Assert.NotNull(selector.FDR);
        for (int j = 0; j < p; j++)
            Assert.Equal(1.0, (double)selector.FDR[j], Tolerance);
    }

    // Equal class means (d = 0): all 3000 permuted |d| are at or above it, so q = 3000 / 3000 = 1. A Half count
    // stops at 2048 (q = 0.68).
    [Fact(Timeout = 120000)]
    public async Task SignificanceAnalysis_Half_CountsMoreThan2048Exceedances()
    {
        await Task.CompletedTask;
        var selector = new SignificanceAnalysisSelector(nFeaturesToSelect: 1, nPermutations: 3000, randomState: 7);
        selector.Fit(HalfColumn(4, i => i % 2), HalfVector(4, i => i < 2 ? 0.0 : 1.0));

        Assert.NotNull(selector.QValues);
        Assert.Equal(1.0, (double)selector.QValues[0], Tolerance);
    }

    // Rule bin 1 -> positive: 3000 of 4000 rows, all positive. support 0.75, confidence 1, lift 1 / 0.75, so the
    // score is 1. Summed in Half the 3000 counts saturate at 2048 (score 0.68).
    [Fact(Timeout = 120000)]
    public async Task AprioriBasedSelector_Half_CountsMoreThan2048Transactions()
    {
        await Task.CompletedTask;
        var selector = new AprioriBasedSelector<Half>(nFeaturesToSelect: 1, minSupport: 0.1, minConfidence: 0.5, nBins: 2);
        selector.Fit(HalfColumn(4000, i => i < 3000 ? 1.0 : 0.0), HalfVector(4000, i => i < 3000 ? 1.0 : 0.0));

        Assert.NotNull(selector.AssociationScores);
        Assert.Equal(1.0, (double)selector.AssociationScores[0], 2e-3);
    }

    // The median split keeps the 3000 ones, all positive, so the 1-itemset support is 3000 / 4000 = 0.75 as in the
    // double fit; a Half sum saturates the count at 2048 (support 0.512).
    [Fact(Timeout = 120000)]
    public async Task FrequentPatternSelector_Half_MatchesDouble()
    {
        await Task.CompletedTask;
        Func<int, double> value = i => i < 3000 ? 1.0 : 0.0;
        var reference = new FrequentPatternSelector<double>(nFeaturesToSelect: 1);
        var doubleData = new Matrix<double>(4000, 1);
        var doubleTarget = new Vector<double>(4000);
        for (int i = 0; i < 4000; i++)
        {
            doubleData[i, 0] = value(i);
            doubleTarget[i] = value(i);
        }

        reference.Fit(doubleData, doubleTarget);

        var selector = new FrequentPatternSelector<Half>(nFeaturesToSelect: 1);
        selector.Fit(HalfColumn(4000, value), HalfVector(4000, value));

        Assert.NotNull(reference.PatternScores);
        Assert.NotNull(selector.PatternScores);
        Assert.Equal(0.75, reference.PatternScores[0], Tolerance);
        Assert.Equal(0.75, (double)selector.PatternScores[0], Tolerance);
    }

    // Shared data for the class-count tests: class 0 has 400 rows at 0 and 100 at 10, class 1 has 2500 rows at 5.
    private static (Matrix<Half> Data, Vector<Half> Target) ImbalancedHalfClasses()
    {
        const int n = 3000;
        return (HalfColumn(n, i => i < 400 ? 0.0 : i < 500 ? 10.0 : 5.0), HalfVector(n, i => i < 500 ? 0.0 : 1.0));
    }

    // sklearn's DecisionTreeClassifier Gini: parent 1 - (1/6)^2 - (5/6)^2 = 0.277778. The best split x <= 0.909
    // leaves class 0's 400 zeros on the left (Gini 0) and 100 + 2500 rows on the right with Gini
    // 1 - (100/2600)^2 - (2500/2600)^2 = 0.073964, so the reduction is 0.277778 - 2600/3000 * 0.073964 = 0.213675.
    [Fact(Timeout = 120000)]
    public async Task GiniImpuritySelector_Half_CountsMoreThan2048PerClass()
    {
        await Task.CompletedTask;
        var (data, target) = ImbalancedHalfClasses();
        var selector = new GiniImpuritySelector<Half>(nFeaturesToSelect: 1);
        selector.Fit(data, target);

        Assert.NotNull(selector.GiniReductions);
        Assert.Equal(0.213675, (double)selector.GiniReductions[0], 1e-3);
    }

    // Class means 2 and 5, boundary 3.5: class 0 has 400 of 500 rows below it and class 1 all 2500 above, so the
    // boundary accuracy is 2900 / 3000. A Half class count stops at 2048 ((400 + 2048) / 2548 = 0.9608).
    [Fact(Timeout = 120000)]
    public async Task DecisionBoundarySelector_Half_CountsMoreThan2048PerClass()
    {
        await Task.CompletedTask;
        var (data, target) = ImbalancedHalfClasses();
        var selector = new DecisionBoundarySelector<Half>(nFeaturesToSelect: 1);
        selector.Fit(data, target);

        Assert.NotNull(selector.BoundaryClarityScores);
        Assert.Equal(2900.0 / 3000.0, (double)selector.BoundaryClarityScores[0], 1e-3);
    }

    // Between-class scatter 500 (2 - 4.5)^2 + 2500 (5 - 4.5)^2 = 3750 over within-class scatter
    // 400 * 2^2 + 100 * 8^2 = 8000 gives 0.46875; the class counts and sums need more than Half's 2048.
    [Fact(Timeout = 120000)]
    public async Task LDAProjectionSelector_Half_CountsMoreThan2048PerClass()
    {
        await Task.CompletedTask;
        var (data, target) = ImbalancedHalfClasses();
        var selector = new LDAProjectionSelector<Half>(nFeaturesToSelect: 1);
        selector.Fit(data, target);

        Assert.NotNull(selector.LDAScores);
        Assert.Equal(0.46875, (double)selector.LDAScores[0], 1e-3);
    }

    // Master scores a pair with a zero or NaN norm as similarity 0. Cosines: rows 0-1 and 1-2 are 0.7071, row 0-2
    // is 0, and the NaN row 3 is 0 to all, so row 1 (average 0.4714) is the most similar sample. Multiplying
    // NaN cosines by a 0 mask kept the NaN and poisoned every average.
    [Fact(Timeout = 120000)]
    public async Task SimilarityBasedSplitter_NaNRow_DoesNotPoisonOtherRows()
    {
        await Task.CompletedTask;
        var data = M(new double[,] { { 1, 0 }, { 1, 1 }, { 0, 1 }, { double.NaN, 1 } });
        var splitter = new SimilarityBasedSplitter<double>(testSize: 0.25, testSimilar: true, shuffle: false);

        var result = splitter.Split(data);

        Assert.Equal(new[] { 1 }, result.TestIndices);
    }

    // WOE with regularization 0.5 over 3 events and 3 non-events, 2 categories: category 0 (2 events, 1 non-event)
    // is ln((2.5 / 4) / (1.5 / 4)) = 0.510826 and category 1 the negative. The |y - 1| < 1e-10 event test needs a
    // tolerance Half can hold: with 1e-10 rounded to 0 no target was an event and Fit rejected binary targets.
    [Fact(Timeout = 120000)]
    public async Task WOEEncoder_Half_RecognisesBinaryTargets()
    {
        await Task.CompletedTask;
        double[] categories = { 0, 0, 0, 1, 1, 1 };
        double[] labels = { 1, 1, 0, 1, 0, 0 };
        var data = HalfColumn(6, i => categories[i]);
        var encoder = new WOEEncoder<Half>();
        encoder.Fit(data, HalfVector(6, i => labels[i]));

        var encoded = encoder.Transform(data);
        for (int i = 0; i < 6; i++)
            Assert.Equal(i < 3 ? 0.510826 : -0.510826, (double)encoded[i, 0], 2e-3);
    }

    // Master compares with NumOps.Compare (double.CompareTo), which orders NaN below every number. Uniform: the NaN is
    // the minimum, so every edge is NaN, no number is at most an edge (last bin) and the NaN is in bin 0.
    [Fact(Timeout = 120000)]
    public async Task KBinsDiscretizer_Uniform_NaNMakesEveryEdgeNaNAsInMaster()
    {
        await Task.CompletedTask;
        var data = M(new double[,] { { 1 }, { double.NaN }, { 3 }, { 5 }, { 7 } });
        var discretizer = new AiDotNet.Preprocessing.Discretizers.KBinsDiscretizer<double>(
            nBins: 2, strategy: AiDotNet.Preprocessing.Discretizers.BinningStrategy.Uniform);

        var bins = discretizer.FitTransform(data);

        Assert.Equal(new[] { 1.0, 0.0, 1.0, 1.0, 1.0 }, Enumerable.Range(0, 5).Select(i => bins[i, 0]).ToArray());
    }

    // Quantile: master sorts NaN first, [NaN, 1, 3, 5, 7], so the edges at positions 0, 2, 4 are [NaN, 3, 7] and the
    // upper edges 3 and 7 give bins 0, 0, 0, 1, 1 (the NaN value is in bin 0).
    [Fact(Timeout = 120000)]
    public async Task KBinsDiscretizer_Quantile_SortsNaNFirstAsInMaster()
    {
        await Task.CompletedTask;
        var data = M(new double[,] { { 1 }, { double.NaN }, { 3 }, { 5 }, { 7 } });
        var discretizer = new AiDotNet.Preprocessing.Discretizers.KBinsDiscretizer<double>(
            nBins: 2, strategy: AiDotNet.Preprocessing.Discretizers.BinningStrategy.Quantile);

        var bins = discretizer.FitTransform(data);

        Assert.Equal(new[] { 0.0, 0.0, 0.0, 1.0, 1.0 }, Enumerable.Range(0, 5).Select(i => bins[i, 0]).ToArray());
    }

    // Master's inverse takes (int)NaN, clamped to bin 0: edges [1, 5, 9] give the bin 0 midpoint 3.
    [Fact(Timeout = 120000)]
    public async Task KBinsDiscretizer_InverseOfNaNCodeIsTheFirstBinMidpoint()
    {
        await Task.CompletedTask;
        var discretizer = new AiDotNet.Preprocessing.Discretizers.KBinsDiscretizer<double>(
            nBins: 2, strategy: AiDotNet.Preprocessing.Discretizers.BinningStrategy.Quantile);
        discretizer.Fit(M(new double[,] { { 1 }, { 3 }, { 5 }, { 7 }, { 9 } }));

        var restored = discretizer.InverseTransform(M(new double[,] { { double.NaN }, { 1 } }));

        Assert.Equal(3.0, restored[0, 0], Tolerance);
        Assert.Equal(7.0, restored[1, 0], Tolerance);
    }

    // Master keys categories by double (double.NaN.Equals(double.NaN) is true) and sorts them with OrderBy, which puts
    // NaN first: the column [NaN, 1, NaN, 2] has the categories [NaN, 1, 2], each NaN row is a known category and
    // transforming the fitted data with unknown-category errors enabled does not throw.
    [Fact(Timeout = 120000)]
    public async Task Encoders_NaN_IsOneCategoryOrderedFirstAsInMaster()
    {
        await Task.CompletedTask;
        var data = M(new double[,] { { double.NaN }, { 1 }, { double.NaN }, { 2 } });

        var oneHot = new OneHotEncoder<double>(handleUnknown: OneHotUnknownHandling.Error).FitTransform(data);
        double[,] expectedOneHot = { { 1, 0, 0 }, { 0, 1, 0 }, { 1, 0, 0 }, { 0, 0, 1 } };
        Assert.Equal(3, oneHot.Columns);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 3; j++)
                Assert.Equal(expectedOneHot[i, j], oneHot[i, j], Tolerance);

        var labels = new LabelEncoder<double>().FitTransform(data);
        double[] expectedLabels = { 0, 1, 0, 2 };
        for (int i = 0; i < 4; i++)
            Assert.Equal(expectedLabels[i], labels[i, 0], Tolerance);

        var counts = new CountEncoder<double>(handleUnknown: CountEncoderHandleUnknown.Error).FitTransform(data);
        double[] expectedCounts = { 2, 1, 2, 1 };
        for (int i = 0; i < 4; i++)
            Assert.Equal(expectedCounts[i], counts[i, 0], Tolerance);
    }

    // Category 0 has 3000 events and 1000 non-events, category 1 the reverse (4000 of each in total). With
    // regularization 0.5 the WOE of category 0 is ln(3000.5 / 1000.5) = 1.098279 and category 1 the negative. An
    // event count summed in Half stops at 2048.
    [Fact(Timeout = 120000)]
    public async Task WOEEncoder_Half_CountsMoreThan2048EventsPerCategory()
    {
        await Task.CompletedTask;
        const int n = 8000;
        var data = HalfColumn(n, i => i < 4000 ? 0.0 : 1.0);
        var target = HalfVector(n, i => i < 3000 || (i >= 4000 && i < 5000) ? 1.0 : 0.0);
        var encoder = new WOEEncoder<Half>();
        encoder.Fit(data, target);

        var encoded = encoder.Transform(data);
        Assert.Equal(1.098279, (double)encoded[0, 0], 5e-3);
        Assert.Equal(-1.098279, (double)encoded[n - 1, 0], 5e-3);
    }

    // The feature always beats its shadow, so all 4200 iterations are hits: far above Boruta's acceptance bound it
    // is confirmed, and its mean importance is 1. Counted in Half, the hits and the summed importance stop at 2048
    // (tentative, mean 0.49).
    [Fact(Timeout = 120000)]
    public async Task Boruta_Half_CountsMoreThan2048Hits()
    {
        await Task.CompletedTask;
        var selector = new BorutaSelector(
            maxIterations: 4200,
            importanceFunction: (extended, _) => new Vector<Half>(new[] { (Half)1, (Half)0 }),
            randomState: 3);
        selector.Fit(HalfColumn(4, i => i), HalfVector(4, i => i % 2));

        var decisions = selector.Decisions;
        var importances = selector.Importances;
        Assert.NotNull(decisions);
        Assert.NotNull(importances);
        if (decisions is null || importances is null)
            return;
        Assert.Equal(BorutaSelector.BorutaDecision.Confirmed, decisions[0]);
        Assert.Equal(1.0, (double)importances[0], Tolerance);
    }

    // Master sorts the shift column with OrderBy (NaN first): [NaN, 0..9] has median 4, q25 1 and q75 7, so with
    // strength 0.5 the train threshold is 2.5 and the test threshold 5.5. A NaN fails both comparisons and is a
    // middle sample; the 4 test and 4 middle samples fill the 8 test slots and only rows 1-3 (values 0-2) train.
    [Fact(Timeout = 120000)]
    public async Task CovariateShiftSplitter_NaN_IsAMiddleSampleAsInMaster()
    {
        await Task.CompletedTask;
        var values = new double[11, 1];
        values[0, 0] = double.NaN;
        for (int i = 1; i < 11; i++)
            values[i, 0] = i - 1;

        var result = new CovariateShiftSplitter<double>(testSize: 0.8, shiftStrength: 0.5).Split(M(values));

        Assert.Equal(new[] { 1, 2, 3 }, result.TrainIndices.OrderBy(i => i).ToArray());
        Assert.Contains(0, result.TestIndices);
    }

    // Diverse selection adds the sample farthest from the first one. The points 0, 300 and -310 are more than 256
    // apart, so their squared distances overflow Half: summed directly every distance is inf and the tie went to
    // the lowest index instead of the farthest point.
    [Fact(Timeout = 120000)]
    public async Task PoolBasedSplitter_Half_DiverseSelectionDoesNotOverflow()
    {
        await Task.CompletedTask;
        double[] points = { 0, 300, -310 };
        var splitter = new PoolBasedSplitter<Half>(
            initialLabeledRatio: 0.1, minLabeledSamples: 2, strategy: PoolBasedSplitter<Half>.SelectionStrategy.Diverse);

        var result = splitter.Split(HalfColumn(3, i => points[i]));

        Assert.Equal(2, result.TrainIndices.Length);
        int first = result.TrainIndices[0];
        int farthest = Enumerable.Range(0, 3).Where(j => j != first)
            .OrderByDescending(j => Math.Abs(points[j] - points[first])).First();
        Assert.Equal(farthest, result.TrainIndices[1]);
    }

    // Mean 5, so the standardized distances are |x - 5| / 168 (population std sqrt(282250 / 10) = 168): row 8 (400)
    // is the most distant and the only test sample. Squared in Half, 395^2 overflows, the std became inf and every
    // distance 0, which put row 0 in the test set.
    [Fact(Timeout = 120000)]
    public async Task OutOfDistributionSplitter_Half_StandardizationDoesNotOverflow()
    {
        await Task.CompletedTask;
        var splitter = new OutOfDistributionSplitter<Half>(testSize: 0.1, oodPercentile: 0.5);

        var result = splitter.Split(HalfColumn(10, i => i == 8 ? 400.0 : i == 9 ? -350.0 : 0.0));

        Assert.Equal(new[] { 8 }, result.TestIndices);
    }

    // Feature 1 (MI 0.121) joins the blanket first. Feature 0 given feature 1: the 4100 rows with feature 1 at 0
    // have cells 3000 / 1000 / 100 and MI 0.032934, weighted 4100 / 8100 to 0.016670 > 0.015, so feature 0 joins
    // too. Counted in Half the 3000-row cell stops at 2048, giving 0.013362 and dropping feature 0.
    [Fact(Timeout = 120000)]
    public async Task MarkovBlanketSelector_Half_CountsMoreThan2048PerGroup()
    {
        await Task.CompletedTask;
        const int n = 8100;
        var data = new Half[n, 2];
        var target = new Half[n];
        for (int i = 0; i < n; i++)
        {
            // (feature 0, feature 1, target): 3000 x (0,0,0), 1000 x (0,0,1), 1000 x (0,1,0), 3000 x (0,1,1),
            // 100 x (1,0,1)
            int a = i >= 8000 ? 1 : 0;
            int b = i >= 4000 && i < 8000 ? 1 : 0;
            int y = (i >= 3000 && i < 4000) || i >= 5000 ? 1 : 0;
            data[i, 0] = (Half)a;
            data[i, 1] = (Half)b;
            target[i] = (Half)y;
        }

        var selector = new MarkovBlanketSelector<Half>(threshold: 0.015, nBins: 2);
        selector.Fit(new Matrix<Half>(data), new Vector<Half>(target));

        var selected = selector.SelectedIndices;
        Assert.NotNull(selected);
        if (selected is null)
            return;
        Assert.Equal(new[] { 0, 1 }, selected.OrderBy(i => i).ToArray());
    }

    // Uniform knots are min + (i + 1) / (nKnots + 1) * (max - min): 15000, 30000 and 45000 for the range [0, 60000].
    // Formed as (i + 1) * (max - min) first, 120000 and 180000 overflow Half to inf.
    [Fact(Timeout = 120000)]
    public async Task SplineTransformer_Half_UniformKnotsDoNotOverflow()
    {
        await Task.CompletedTask;
        var spline = new SplineTransformer<Half>(nKnots: 3, degree: 3, knotStrategy: SplineKnotStrategy.Uniform);
        spline.Fit(HalfColumn(2, i => i * 60000.0));

        var allKnots = spline.Knots;
        Assert.NotNull(allKnots);
        if (allKnots is null)
            return;
        var knots = allKnots[0];
        Assert.Equal(15000.0, (double)knots[4], Tolerance);
        Assert.Equal(30000.0, (double)knots[5], Tolerance);
        Assert.Equal(45000.0, (double)knots[6], Tolerance);
    }
}
