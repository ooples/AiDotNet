using AiDotNet.Preprocessing.PowerTransforms;
using AiDotNet.Preprocessing.ImbalancedLearning;
using AiDotNet.Preprocessing.Imputers;
using AiDotNet.AnomalyDetection.Statistical;
using AiDotNet.Preprocessing.OutlierHandling;
using AiDotNet.Preprocessing.Scalers;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Preprocessing;

namespace AiDotNet.Tests.IntegrationTests.Preprocessing;

/// <summary>
/// Reference tests that pin the preprocessing transformers to scikit-learn / imbalanced-learn (or the original
/// paper) on small inputs whose expected values were computed by hand from the reference definitions.
/// If any test fails, the CODE must be fixed - never adjust expected values.
/// </summary>
public class PreprocessingReferenceTests
{
    private const double Tolerance = 1e-10;

    private static Matrix<double> M(double[,] data) => new Matrix<double>(data);

    // sklearn.preprocessing.StandardScaler().fit_transform([[1], [2], [3], [4]]):
    // mean_ = 2.5, var_ = 1.25 (ddof=0), scale_ = sqrt(1.25)
    [Fact(Timeout = 120000)]
    public async Task StandardScaler_UsesPopulationVariance_LikeSklearn()
    {
        await Task.CompletedTask;
        var scaler = new StandardScaler<double>();
        var result = scaler.FitTransform(M(new double[,] { { 1 }, { 2 }, { 3 }, { 4 } }));

        Assert.Equal(Math.Sqrt(1.25), scaler.StandardDeviation?[0] ?? double.NaN, Tolerance);
        Assert.Equal(-1.3416407864998738, result[0, 0], Tolerance);
        Assert.Equal(-0.4472135954999579, result[1, 0], Tolerance);
        Assert.Equal(0.4472135954999579, result[2, 0], Tolerance);
        Assert.Equal(1.3416407864998738, result[3, 0], Tolerance);
    }

    // sklearn ignores NaN in fit and keeps it in transform:
    // StandardScaler().fit_transform([[1], [nan], [3]]) == [[-1], [nan], [1]]
    [Fact(Timeout = 120000)]
    public async Task StandardScaler_IgnoresNaNInFit_KeepsItInTransform()
    {
        await Task.CompletedTask;
        var scaler = new StandardScaler<double>();
        var result = scaler.FitTransform(M(new double[,] { { 1 }, { double.NaN }, { 3 } }));

        Assert.Equal(2.0, scaler.Mean?[0] ?? double.NaN, Tolerance);
        Assert.Equal(-1.0, result[0, 0], Tolerance);
        Assert.True(double.IsNaN(result[1, 0]));
        Assert.Equal(1.0, result[2, 0], Tolerance);
    }

    // sklearn's _is_constant_feature: a column whose variance is only floating-point noise is left unscaled
    // (scale_ = 1) instead of being blown up by a ~1e-17 standard deviation.
    [Fact(Timeout = 120000)]
    public async Task StandardScaler_NearConstantColumn_IsNotBlownUp()
    {
        await Task.CompletedTask;
        double v = 0.1 + 0.2;
        var scaler = new StandardScaler<double>();
        var result = scaler.FitTransform(M(new double[,] { { v }, { v }, { v }, { 0.3 } }));

        Assert.Equal(1.0, scaler.StandardDeviation?[0] ?? double.NaN, Tolerance);
        for (int i = 0; i < 4; i++)
            Assert.True(Math.Abs(result[i, 0]) < 1e-12, $"row {i}: {result[i, 0]}");
    }

    // sklearn.preprocessing.MinMaxScaler().fit_transform([[5, 1], [5, 2], [5, 3]]): a zero data range is
    // treated as 1, so the constant column maps to the feature-range minimum, and inverse_transform recovers it.
    [Fact(Timeout = 120000)]
    public async Task MinMaxScaler_ConstantColumn_MapsToRangeMin_AndInverts()
    {
        await Task.CompletedTask;
        var scaler = new MinMaxScaler<double>(-1.0, 1.0);
        var data = M(new double[,] { { 5, 1 }, { 5, 2 }, { 5, 3 } });
        var result = scaler.FitTransform(data);

        for (int i = 0; i < 3; i++)
            Assert.Equal(-1.0, result[i, 0], Tolerance);
        Assert.Equal(-1.0, result[0, 1], Tolerance);
        Assert.Equal(0.0, result[1, 1], Tolerance);
        Assert.Equal(1.0, result[2, 1], Tolerance);

        var restored = scaler.InverseTransform(result);
        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(5.0, restored[i, 0], Tolerance);
            Assert.Equal(i + 1.0, restored[i, 1], Tolerance);
        }
    }

    // sklearn MinMaxScaler uses nanmin / nanmax: fit_transform([[0], [nan], [10]]) == [[0], [nan], [1]]
    [Fact(Timeout = 120000)]
    public async Task MinMaxScaler_IgnoresNaNInFit()
    {
        await Task.CompletedTask;
        var scaler = new MinMaxScaler<double>();
        var result = scaler.FitTransform(M(new double[,] { { 0 }, { double.NaN }, { 10 } }));

        Assert.Equal(0.0, result[0, 0], Tolerance);
        Assert.True(double.IsNaN(result[1, 0]));
        Assert.Equal(1.0, result[2, 0], Tolerance);
    }

    // Columns outside columnIndices pass through untouched even with a non-[0, 1] feature range.
    [Fact(Timeout = 120000)]
    public async Task MinMaxScaler_UnselectedColumns_PassThrough()
    {
        await Task.CompletedTask;
        var scaler = new MinMaxScaler<double>(-1.0, 1.0, new[] { 1 });
        var result = scaler.FitTransform(M(new double[,] { { 3, 0 }, { 7, 4 } }));

        Assert.Equal(3.0, result[0, 0], Tolerance);
        Assert.Equal(7.0, result[1, 0], Tolerance);
        Assert.Equal(-1.0, result[0, 1], Tolerance);
        Assert.Equal(1.0, result[1, 1], Tolerance);
    }

    // sklearn.preprocessing.MaxAbsScaler: max_abs_ = nanmax(|X|); NaN is kept by transform.
    [Fact(Timeout = 120000)]
    public async Task MaxAbsScaler_MatchesSklearn_WithNaN()
    {
        await Task.CompletedTask;
        var scaler = new MaxAbsScaler<double>();
        var result = scaler.FitTransform(M(new double[,] { { 1, -2 }, { double.NaN, 4 }, { -3, 0 } }));

        Assert.Equal(1.0 / 3.0, result[0, 0], Tolerance);
        Assert.Equal(-0.5, result[0, 1], Tolerance);
        Assert.True(double.IsNaN(result[1, 0]));
        Assert.Equal(1.0, result[1, 1], Tolerance);
        Assert.Equal(-1.0, result[2, 0], Tolerance);
        Assert.Equal(0.0, result[2, 1], Tolerance);
    }

    // sklearn.preprocessing.RobustScaler().fit_transform([[1], [2], [nan], [3], [4], [100]]):
    // center_ = nanmedian = 3, scale_ = nanpercentile(75) - nanpercentile(25) = 4 - 2 = 2
    [Fact(Timeout = 120000)]
    public async Task RobustScaler_MatchesSklearn_WithNaN()
    {
        await Task.CompletedTask;
        var scaler = new RobustScaler<double>();
        var result = scaler.FitTransform(M(new double[,] { { 1 }, { 2 }, { double.NaN }, { 3 }, { 4 }, { 100 } }));

        Assert.Equal(3.0, scaler.Median?[0] ?? double.NaN, Tolerance);
        Assert.Equal(2.0, scaler.InterquartileRange?[0] ?? double.NaN, Tolerance);
        Assert.Equal(-1.0, result[0, 0], Tolerance);
        Assert.Equal(-0.5, result[1, 0], Tolerance);
        Assert.True(double.IsNaN(result[2, 0]));
        Assert.Equal(48.5, result[5, 0], Tolerance);
    }

    // sklearn.preprocessing.Normalizer: rows divided by their norm, all-zero rows left at zero.
    [Fact(Timeout = 120000)]
    public async Task Normalizer_MatchesSklearn_L2AndMax()
    {
        await Task.CompletedTask;
        var l2 = new Normalizer<double>(NormType.L2).FitTransform(M(new double[,] { { 3, 4 }, { 0, 0 } }));
        Assert.Equal(0.6, l2[0, 0], Tolerance);
        Assert.Equal(0.8, l2[0, 1], Tolerance);
        Assert.Equal(0.0, l2[1, 0], Tolerance);
        Assert.Equal(0.0, l2[1, 1], Tolerance);

        var max = new Normalizer<double>(NormType.Max).FitTransform(M(new double[,] { { 1, -4 } }));
        Assert.Equal(0.25, max[0, 0], Tolerance);
        Assert.Equal(-1.0, max[0, 1], Tolerance);

        var l1 = new Normalizer<double>(NormType.L1).FitTransform(M(new double[,] { { 1, -3 } }));
        Assert.Equal(0.25, l1[0, 0], Tolerance);
        Assert.Equal(-0.75, l1[0, 1], Tolerance);
    }

    // Decimal scaling (Han & Kamber): divide by the smallest power of ten above max|x|.
    [Fact(Timeout = 120000)]
    public async Task DecimalScaler_UsesSmallestPowerOfTenAboveMaxAbs()
    {
        await Task.CompletedTask;
        var scaler = new DecimalScaler<double>();
        scaler.Fit(M(new double[,] { { 999, 1000, 0.5, 1, 0 }, { -5, 3, -0.25, 0, 0 } }));

        Assert.Equal(1000.0, scaler.Scale?[0] ?? double.NaN, Tolerance);
        Assert.Equal(10000.0, scaler.Scale?[1] ?? double.NaN, Tolerance);
        Assert.Equal(1.0, scaler.Scale?[2] ?? double.NaN, Tolerance);
        Assert.Equal(10.0, scaler.Scale?[3] ?? double.NaN, Tolerance);
        Assert.Equal(1.0, scaler.Scale?[4] ?? double.NaN, Tolerance);
    }

    // numpy.percentile([1, 2, 3, 4, 100], [25, 75]) == [2, 4] -> IQR 2, bounds 2 - 3 = -1 and 4 + 3 = 7.
    [Fact(Timeout = 120000)]
    public async Task IQRClipper_UsesNumpyPercentiles_AndKeepsNaN()
    {
        await Task.CompletedTask;
        var clipper = new IQRClipper<double>();
        clipper.Fit(M(new double[,] { { 1 }, { 2 }, { double.NaN }, { 3 }, { 4 }, { 100 } }));

        Assert.Equal(2.0, clipper.Q1Values?[0] ?? double.NaN, Tolerance);
        Assert.Equal(4.0, clipper.Q3Values?[0] ?? double.NaN, Tolerance);
        Assert.Equal(-1.0, clipper.LowerBounds?[0] ?? double.NaN, Tolerance);
        Assert.Equal(7.0, clipper.UpperBounds?[0] ?? double.NaN, Tolerance);

        var result = clipper.Transform(M(new double[,] { { -5 }, { double.NaN }, { 100 } }));
        Assert.Equal(-1.0, result[0, 0], Tolerance);
        Assert.True(double.IsNaN(result[1, 0]));
        Assert.Equal(7.0, result[2, 0], Tolerance);
        Assert.Equal(new[] { 2 }, clipper.CountOutliersPerFeature(M(new double[,] { { -5 }, { 0 }, { 100 } })));
    }

    // Iglewicz & Hoaglin: the clip bound is exactly where the modified Z-score Phi^-1(3/4) * (x - median) / MAD
    // reaches the threshold, so GetModifiedZScores at the bound returns the threshold.
    [Fact(Timeout = 120000)]
    public async Task MADClipper_BoundsAgreeWithModifiedZScore()
    {
        await Task.CompletedTask;
        var clipper = new MADClipper<double>(threshold: 3.5);
        clipper.Fit(M(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 100 } }));

        double upper = clipper.UpperBounds?[0] ?? double.NaN;
        double lower = clipper.LowerBounds?[0] ?? double.NaN;
        Assert.Equal(3.0 + 3.5 / 0.6744897501960817, upper, Tolerance);
        var z = clipper.GetModifiedZScores(M(new double[,] { { upper }, { lower } }));
        Assert.Equal(3.5, z[0, 0], Tolerance);
        Assert.Equal(-3.5, z[1, 0], Tolerance);
    }

    // ZScoreClipper: population std, NaN ignored by the fit and kept by transform.
    [Fact(Timeout = 120000)]
    public async Task ZScoreClipper_IgnoresNaNInFit()
    {
        await Task.CompletedTask;
        var clipper = new ZScoreClipper<double>(threshold: 1.0);
        clipper.Fit(M(new double[,] { { 1 }, { double.NaN }, { 3 } }));

        Assert.Equal(2.0, clipper.Means?[0] ?? double.NaN, Tolerance);
        Assert.Equal(1.0, clipper.StandardDeviations?[0] ?? double.NaN, Tolerance);
        var result = clipper.Transform(M(new double[,] { { 10 }, { double.NaN }, { -10 } }));
        Assert.Equal(3.0, result[0, 0], Tolerance);
        Assert.True(double.IsNaN(result[1, 0]));
        Assert.Equal(1.0, result[2, 0], Tolerance);
    }

    // feature-engine Winsorizer(capping_method="iqr", fold=1.5): Q1 - 1.5 IQR, Q3 + 1.5 IQR (pandas quantiles).
    [Fact(Timeout = 120000)]
    public async Task Winsorizer_IqrLimits_MatchFeatureEngine()
    {
        await Task.CompletedTask;
        var winsorizer = new Winsorizer<double>(1.5, 1.5, WinsorizerLimitType.IQR);
        winsorizer.Fit(M(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 100 } }));

        Assert.Equal(-1.0, winsorizer.LowerBounds?[0] ?? double.NaN, Tolerance);
        Assert.Equal(7.0, winsorizer.UpperBounds?[0] ?? double.NaN, Tolerance);
    }

    // OutlierClipper: numpy.percentile([0..10], [10, 90]) == [1, 9].
    [Fact(Timeout = 120000)]
    public async Task OutlierClipper_UsesNumpyPercentiles()
    {
        await Task.CompletedTask;
        var clipper = new OutlierClipper<double>(10, 90);
        var data = new double[11, 1];
        for (int i = 0; i <= 10; i++)
            data[i, 0] = i;
        var result = clipper.FitTransform(M(data));

        Assert.Equal(1.0, result[0, 0], Tolerance);
        Assert.Equal(5.0, result[5, 0], Tolerance);
        Assert.Equal(9.0, result[10, 0], Tolerance);
    }

    // columnIndices of DetectorBasedFilter are "the columns to consider for anomaly detection": the detector
    // must be fitted on and score only those columns.
    [Fact(Timeout = 120000)]
    public async Task DetectorBasedFilter_DetectsOnSelectedColumnsOnly()
    {
        await Task.CompletedTask;
        var data = M(new double[,]
        {
            { 1, 0 }, { 2, 0 }, { 3, 0 }, { 4, 0 }, { 50, 0 }, { 6, 0 }, { 7, 0 }, { 8, 0 }, { 9, 1000 }, { 10, 0 }
        });
        var filter = new DetectorBasedFilter<double>(new ZScoreDetector<double>(), FilterMode.Flag, new[] { 0 });
        filter.Fit(data);

        var reference = new ZScoreDetector<double>();
        var column0 = M(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 50 }, { 6 }, { 7 }, { 8 }, { 9 }, { 10 } });
        reference.Fit(column0);
        var expected = reference.ScoreAnomalies(column0);
        var actual = filter.GetAnomalyScores(data);
        for (int i = 0; i < 10; i++)
            Assert.Equal(expected[i], actual[i], Tolerance);

        var flagged = filter.Transform(data);
        Assert.Equal(3, flagged.Columns);
        var restored = filter.InverseTransform(flagged);
        Assert.Equal(2, restored.Columns);
        Assert.Equal(1000.0, restored[8, 1], Tolerance);
    }

    // sklearn SimpleImputer(strategy="most_frequent"): "If there is more than one such value, only the smallest
    // is returned." Column [3, 1, 3, 1, nan] -> 1.
    [Fact(Timeout = 120000)]
    public async Task SimpleImputer_MostFrequentTie_ReturnsSmallest()
    {
        await Task.CompletedTask;
        var imputer = new SimpleImputer<double>(ImputationStrategy.MostFrequent);
        var result = imputer.FitTransform(M(new double[,] { { 3 }, { 1 }, { 3 }, { 1 }, { double.NaN } }));

        Assert.Equal(1.0, result[4, 0], Tolerance);
        Assert.Equal(3.0, result[0, 0], Tolerance);
    }

    // sklearn SimpleImputer(strategy="median") ignores NaN: median of [1, 2, 10] = 2.
    [Fact(Timeout = 120000)]
    public async Task SimpleImputer_Median_IgnoresNaN()
    {
        await Task.CompletedTask;
        var imputer = new SimpleImputer<double>(ImputationStrategy.Median);
        var result = imputer.FitTransform(M(new double[,] { { 1 }, { double.NaN }, { 2 }, { 10 } }));

        Assert.Equal(2.0, result[1, 0], Tolerance);
    }

    // sklearn MissingIndicator compares X == missing_values exactly; 1e-12 is not the missing value 0.
    [Fact(Timeout = 120000)]
    public async Task MissingIndicator_UsesExactMissingValueMatch()
    {
        await Task.CompletedTask;
        var indicator = new MissingIndicator<double>(MissingIndicatorFeatures.All, missingValue: 0.0);
        var result = indicator.FitTransform(M(new double[,] { { 0.0, 1 }, { 1e-12, 0 } }));

        Assert.Equal(1.0, result[0, 0], Tolerance);
        Assert.Equal(0.0, result[1, 0], Tolerance);
        Assert.Equal(0.0, result[0, 1], Tolerance);
        Assert.Equal(1.0, result[1, 1], Tolerance);
    }

    // sklearn KNNImputer only takes donors that have the column being imputed. Training rows
    // [0, nan], [1, 10], [2, 20], [100, 30]; query [0, nan], k = 2: nan_euclidean distances to the donors are
    // sqrt(2), 2 sqrt(2), 100 sqrt(2) -> uniform mean of 10 and 20 = 15; inverse-distance (2*10 + 1*20) / 3.
    [Fact(Timeout = 120000)]
    public async Task KNNImputer_UsesDonorsThatHaveTheColumn()
    {
        await Task.CompletedTask;
        var train = M(new double[,] { { 0, double.NaN }, { 1, 10 }, { 2, 20 }, { 100, 30 } });
        var query = M(new double[,] { { 0, double.NaN } });

        var uniform = new KNNImputer<double>(nNeighbors: 2);
        uniform.Fit(train);
        Assert.Equal(15.0, uniform.Transform(query)[0, 1], Tolerance);

        var weighted = new KNNImputer<double>(nNeighbors: 2, weights: KNNWeights.Distance);
        weighted.Fit(train);
        Assert.Equal(40.0 / 3.0, weighted.Transform(query)[0, 1], Tolerance);
    }

    // sklearn KNNImputer(weights="distance") gives donors at distance 0 weight 1 and the others weight 0.
    [Fact(Timeout = 120000)]
    public async Task KNNImputer_ZeroDistanceDonors_TakeAllTheWeight()
    {
        await Task.CompletedTask;
        var imputer = new KNNImputer<double>(nNeighbors: 3, weights: KNNWeights.Distance);
        imputer.Fit(M(new double[,] { { 0, 10 }, { 0, 30 }, { 1, 1000 } }));

        Assert.Equal(20.0, imputer.Transform(M(new double[,] { { 0, double.NaN } }))[0, 1], Tolerance);
    }

    // sklearn IterativeImputer honours the estimator: with LinearRegression an exactly linear column
    // (y = 2x) is imputed exactly; with Ridge(alpha=1) the closed form on rows x = [1, 2, 4], y = [2, 4, 8]
    // gives coef 28/17 and intercept 42/51, so x = 3 -> 98/17.
    [Fact(Timeout = 120000)]
    public async Task IterativeImputer_HonoursTheEstimator()
    {
        await Task.CompletedTask;
        var data = M(new double[,] { { 1, 2 }, { 2, 4 }, { 3, double.NaN }, { 4, 8 } });

        var ols = new IterativeImputer<double>(estimator: IterativeImputerEstimator.LinearRegression);
        Assert.Equal(6.0, ols.FitTransform(data)[2, 1], 1e-8);

        var ridge = new IterativeImputer<double>(estimator: IterativeImputerEstimator.Ridge);
        Assert.Equal(98.0 / 17.0, ridge.FitTransform(data)[2, 1], 1e-8);

        // transform replays the fitted models on new rows
        Assert.Equal(28.0 / 17.0 * 5 + 42.0 / 51.0,
            ridge.Transform(M(new double[,] { { 5, double.NaN } }))[0, 1], 1e-8);
    }

    private static Vector<double> Labels(params double[] values) => new Vector<double>(new List<double>(values));

    // imbalanced-learn ADASYN: g_i = rint(r_i / sum(r) * G) with no re-balancing of the rounded counts.
    // Minority (label 1) at (0,0), (0,10), (0,20), each with one majority sample at distance 1 and the other
    // majority samples far away: with k = 2 every r_i = 1/2, so g_i = rint(4/3) = 1 and only 3 samples are
    // generated for G = 7 - 3 = 4.
    [Fact(Timeout = 120000)]
    public async Task ADASYN_UsesRoundedCountsWithoutRebalancing()
    {
        await Task.CompletedTask;
        var x = M(new double[,]
        {
            { 0, 0 }, { 0, 10 }, { 0, 20 },
            { 1, 0 }, { 1, 10 }, { 1, 20 }, { 100, 100 }, { 100, 101 }, { 100, 102 }, { 100, 103 }
        });
        var y = Labels(1, 1, 1, 0, 0, 0, 0, 0, 0, 0);

        var adasyn = new ADASYN<double>(kNeighbors: 2, seed: 7);
        var (resampledX, resampledY) = adasyn.Resample(x, y);

        Assert.Equal(13, resampledX.Rows);
        Assert.Equal(13, resampledY.Length);
        Assert.Equal(3, adasyn.GetStatistics().SamplesAddedPerClass[1.0]);
    }

    // imbalanced-learn ADASYN raises when no minority sample has a neighbour of another class (the density
    // distribution would divide by zero).
    [Fact(Timeout = 120000)]
    public async Task ADASYN_NoMajorityNeighbours_Throws()
    {
        await Task.CompletedTask;
        var x = M(new double[,] { { 0, 0 }, { 0, 1 }, { 0, 2 }, { 100, 100 }, { 100, 101 }, { 100, 102 }, { 100, 103 } });
        var y = Labels(1, 1, 1, 0, 0, 0, 0);

        Assert.Throws<InvalidOperationException>(() => new ADASYN<double>(kNeighbors: 2, seed: 7).Resample(x, y));
    }

    // imbalanced-learn BorderlineSMOTE (Han et al., 2005): DANGER is m/2 <= m' < m with real division, and
    // with no DANGER sample nothing is generated. m = 3: each minority sample has m' = 1 < 1.5, so the data
    // is returned unchanged (integer division would wrongly make 1 >= 3/2 = 1 a DANGER sample).
    [Fact(Timeout = 120000)]
    public async Task BorderlineSMOTE_OddM_UsesRealHalfAndGeneratesNothingWithoutDanger()
    {
        await Task.CompletedTask;
        var x = M(new double[,] { { 0, 0 }, { 0, 1 }, { 0, 2 }, { 100, 100 }, { 100, 101 }, { 100, 102 }, { 100, 103 } });
        var y = Labels(1, 1, 1, 0, 0, 0, 0);

        var smote = new BorderlineSMOTE<double>(kNeighbors: 1, mNeighbors: 3, seed: 7);
        var (resampledX, _) = smote.Resample(x, y);

        Assert.Equal(7, resampledX.Rows);
        Assert.Equal(0, smote.GetStatistics().SamplesAddedPerClass[1.0]);
    }

    // imbalanced-learn NearMiss(version=1, n_neighbors=3): minority {0, 10, 20}; majority {1, -3, 15, 12, 100}
    // have summed distances to their 3 nearest minority samples 29, 39, 25, 22, 270, so the kept majority
    // samples are 12, 15 and 1 (the single nearest distance would wrongly keep 1, 12 and -3).
    [Fact(Timeout = 120000)]
    public async Task NearMiss1_AveragesTheKNearestMinorityDistances()
    {
        await Task.CompletedTask;
        var x = M(new double[,] { { 0 }, { 10 }, { 20 }, { 1 }, { -3 }, { 15 }, { 12 }, { 100 } });
        var y = Labels(1, 1, 1, 0, 0, 0, 0, 0);

        var (resampledX, _) = new NearMiss<double>(version: NearMissVersion.Version1, kNeighbors: 3).Resample(x, y);

        Assert.Equal(new[] { 0.0, 10, 20, 1, 15, 12 }, resampledX.GetColumn(0).ToArray());
    }

    // imbalanced-learn NearMiss(version=2, n_neighbors=3): with 3 minority samples the 3 farthest are all of
    // them, so the summed distances are again 29, 39, 25, 22, 270 and 12, 15 and 1 are kept.
    [Fact(Timeout = 120000)]
    public async Task NearMiss2_AveragesTheKFarthestMinorityDistances()
    {
        await Task.CompletedTask;
        var x = M(new double[,] { { 0 }, { 10 }, { 20 }, { 1 }, { -3 }, { 15 }, { 12 }, { 100 } });
        var y = Labels(1, 1, 1, 0, 0, 0, 0, 0);

        var (resampledX, _) = new NearMiss<double>(version: NearMissVersion.Version2, kNeighbors: 3).Resample(x, y);

        Assert.Equal(new[] { 0.0, 10, 20, 1, 15, 12 }, resampledX.GetColumn(0).ToArray());
    }

    // imbalanced-learn NearMiss(version=3, n_neighbors=2, n_neighbors_ver3=2): the 2 nearest majority samples
    // of 0, 10, 20 short-list {1, -3, 15, 12}; their summed distances to the 2 nearest minority samples are
    // 10, 16, 10, 10 and the largest are kept (ties in sample order): -3, 1, 15.
    [Fact(Timeout = 120000)]
    public async Task NearMiss3_KeepsTheShortListedSamplesFarthestFromTheMinority()
    {
        await Task.CompletedTask;
        var x = M(new double[,] { { 0 }, { 10 }, { 20 }, { 1 }, { -3 }, { 15 }, { 12 }, { 100 } });
        var y = Labels(1, 1, 1, 0, 0, 0, 0, 0);

        var (resampledX, _) = new NearMiss<double>(version: NearMissVersion.Version3, kNeighbors: 2).Resample(x, y);

        Assert.Equal(new[] { 0.0, 10, 20, 1, -3, 15 }, resampledX.GetColumn(0).ToArray());
    }

    // imbalanced-learn under-sampling with a float sampling_strategy keeps int(n_minority / ratio) majority
    // samples: int(3 / 0.7) = 4 (not ceil = 5).
    [Fact(Timeout = 120000)]
    public async Task UnderSampling_TruncatesTheTargetMajorityCount()
    {
        await Task.CompletedTask;
        var x = M(new double[,] { { 0 }, { 10 }, { 20 }, { 1 }, { -3 }, { 15 }, { 12 }, { 100 } });
        var y = Labels(1, 1, 1, 0, 0, 0, 0, 0);

        var (resampledX, _) = new RandomUnderSampler<double>(samplingStrategy: 0.7, seed: 3).Resample(x, y);

        Assert.Equal(7, resampledX.Rows);
    }

    // imbalanced-learn EditedNearestNeighbours(kind_sel="all") removes a sample when ANY of its k neighbours
    // has another class. Majority {0, 1, 2, 20, 21}, minority {3, 4}, k = 3: every majority sample has at
    // least one minority neighbour, so only the minority remains; kind_sel="mode" keeps 0, 1 and 2.
    [Fact(Timeout = 120000)]
    public async Task EditedNearestNeighbors_AllRemovesOnAnyDisagreement()
    {
        await Task.CompletedTask;
        var x = M(new double[,] { { 0 }, { 1 }, { 2 }, { 20 }, { 21 }, { 3 }, { 4 } });
        var y = Labels(0, 0, 0, 0, 0, 1, 1);

        var (all, _) = new EditedNearestNeighbors<double>(kNeighbors: 3, kind: ENNKind.All).Resample(x, y);
        Assert.Equal(new[] { 3.0, 4 }, all.GetColumn(0).ToArray());

        var (mode, _) = new EditedNearestNeighbors<double>(kNeighbors: 3, kind: ENNKind.Mode).Resample(x, y);
        Assert.Equal(new[] { 0.0, 1, 2, 3, 4 }, mode.GetColumn(0).ToArray());
    }

    // sklearn QuantileTransformer(n_quantiles=10) on 0, 1, ..., 20: quantiles_ = np.nanpercentile(X, linspace(0, 1, 10)
    // * 100) (linear interpolation) = 20 i / 9, so quantiles_[1] = 2.2222... (not the floor-indexed sample 2), and
    // transform maps x to x / 20 (2 -> 0.1).
    [Fact(Timeout = 120000)]
    public async Task QuantileTransformer_InterpolatesQuantilesLinearly()
    {
        await Task.CompletedTask;
        var data = new double[21, 1];
        for (int i = 0; i <= 20; i++)
        {
            data[i, 0] = i;
        }

        var transformer = new QuantileTransformer<double>(nQuantiles: 10);
        transformer.Fit(M(data));

        var quantiles = transformer.Quantiles;
        Assert.NotNull(quantiles);
        Assert.Equal(10, quantiles[0].Length);
        Assert.Equal(20.0 / 9.0, quantiles[0][1], Tolerance);
        Assert.Equal(0.1, transformer.Transform(M(new double[,] { { 2.0 } }))[0, 0], Tolerance);
        Assert.Equal(2.0, transformer.InverseTransform(M(new double[,] { { 0.1 } }))[0, 0], Tolerance);
    }

    // sklearn PowerTransformer fits lambda by maximum likelihood with scipy.optimize.brent (not a 0.1 grid) and
    // standardizes with StandardScaler. Reference values from scikit-learn 1.9.1 on [1, 2, 3, 5, 8, 13, 21]:
    // Yeo-Johnson lambda -0.10907583494172068, transform(4) -0.2551259365003003; Box-Cox lambda
    // 0.055747783607017924, transform(4) -0.2265160870380528. On [-3, -1, 0, 0.5, 2, 7, 20] (Yeo-Johnson):
    // lambda 0.43543662095254776, transform([-2, 1]) = [-1.1220341395151565, 0.012290038542442674].
    [Fact(Timeout = 120000)]
    public async Task PowerTransformer_MaximumLikelihoodLambda_MatchesSklearn()
    {
        await Task.CompletedTask;
        var data = M(new double[,] { { 1 }, { 2 }, { 3 }, { 5 }, { 8 }, { 13 }, { 21 } });

        var yeoJohnson = new PowerTransformer<double>(PowerTransformMethod.YeoJohnson);
        yeoJohnson.Fit(data);
        var yeoJohnsonLambdas = yeoJohnson.Lambdas;
        Assert.NotNull(yeoJohnsonLambdas);
        Assert.Equal(-0.10907583494172068, yeoJohnsonLambdas[0], 1e-6);
        Assert.Equal(-0.2551259365003003, yeoJohnson.Transform(M(new double[,] { { 4 } }))[0, 0], 1e-6);

        var boxCox = new PowerTransformer<double>(PowerTransformMethod.BoxCox);
        boxCox.Fit(data);
        var boxCoxLambdas = boxCox.Lambdas;
        Assert.NotNull(boxCoxLambdas);
        Assert.Equal(0.055747783607017924, boxCoxLambdas[0], 1e-6);
        Assert.Equal(-0.2265160870380528, boxCox.Transform(M(new double[,] { { 4 } }))[0, 0], 1e-6);

        var mixed = new PowerTransformer<double>(PowerTransformMethod.YeoJohnson);
        mixed.Fit(M(new double[,] { { -3 }, { -1 }, { 0 }, { 0.5 }, { 2 }, { 7 }, { 20 } }));
        var mixedLambdas = mixed.Lambdas;
        Assert.NotNull(mixedLambdas);
        Assert.Equal(0.43543662095254776, mixedLambdas[0], 1e-6);
        var transformed = mixed.Transform(M(new double[,] { { -2 }, { 1 } }));
        Assert.Equal(-1.1220341395151565, transformed[0, 0], 1e-6);
        Assert.Equal(0.012290038542442674, transformed[1, 0], 1e-6);
    }

    // sklearn train_test_split: n_test = ceil(test_size * n_samples), so 10 samples at test_size=0.25 give
    // 3 test samples (the last 3 without shuffling) and 7 train samples; KFold(3) on 10 samples gives folds of
    // 4, 3, 3 (the first n % k folds take one extra sample).
    [Fact(Timeout = 120000)]
    public async Task TrainTestSplit_SizesMatchSklearn()
    {
        await Task.CompletedTask;
        var data = new double[10, 1];
        for (int i = 0; i < 10; i++)
        {
            data[i, 0] = i;
        }

        var x = M(data);
        var y = new Vector<double>(new List<double> { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });

        var (xTrain, xTest, _, yTest) = TrainTestSplit<double>.Split(x, y, testSize: 0.25, shuffle: false);
        Assert.Equal(7, xTrain.Rows);
        Assert.Equal(new[] { 7.0, 8, 9 }, xTest.GetColumn(0).ToArray());
        Assert.Equal(new[] { 7.0, 8, 9 }, yTest.ToArray());

        var folds = TrainTestSplit<double>.KFoldSplit(x, y, k: 3, shuffle: false);
        Assert.Equal(new[] { 0.0, 1, 2, 3 }, folds[0].XTest.GetColumn(0).ToArray());
        Assert.Equal(new[] { 4.0, 5, 6 }, folds[1].XTest.GetColumn(0).ToArray());
        Assert.Equal(new[] { 7.0, 8, 9 }, folds[2].XTest.GetColumn(0).ToArray());
        Assert.Equal(new[] { 0.0, 1, 2, 3, 7, 8, 9 }, folds[1].XTrain.GetColumn(0).ToArray());
    }
    private static Matrix<Half> HalfColumn(int rows, Func<int, double> value)
    {
        var data = new Half[rows, 1];
        for (int i = 0; i < rows; i++)
        {
            data[i, 0] = (Half)value(i);
        }

        return new Matrix<Half>(data);
    }

    // Counting a 0/1 mask must not saturate for Half: 2048 + 1 rounds back to 2048 in Half, so a count kept in
    // T reports 2048 for 2049 outliers.
    [Fact(Timeout = 120000)]
    public async Task IQRClipper_Half_CountsMoreThan2048Outliers()
    {
        await Task.CompletedTask;
        var clipper = new IQRClipper<Half>();
        clipper.Fit(HalfColumn(5, i => i)); // Q1 = 1, Q3 = 3, bounds [-2, 6]

        var counts = clipper.CountOutliersPerFeature(HalfColumn(2049, _ => 100));

        Assert.Equal(2049, counts[0]);
    }

    // imbalanced-learn counts the class sizes exactly; summed in Half the 2049-sample class would count as 2048.
    [Fact(Timeout = 120000)]
    public async Task ResamplingClassCounts_Half_CountMoreThan2048Samples()
    {
        await Task.CompletedTask;
        var labels = new Half[2052];
        for (int i = 2049; i < labels.Length; i++)
        {
            labels[i] = (Half)1;
        }

        var counts = ResamplingTensorOps.ClassCounts(new Vector<Half>(labels));

        Assert.Equal(2049, counts[(Half)0]);
        Assert.Equal(3, counts[(Half)1]);
    }

    // scikit-learn IterativeImputer(imputation_order="ascending") imputes the feature with fewer missing values
    // first; with 2050 and 2049 missing entries a Half count saturates both at 2048 and ties them.
    [Fact(Timeout = 120000)]
    public async Task IterativeImputer_Half_OrdersMissingCountsAbove2048()
    {
        await Task.CompletedTask;
        var mask = new Half[2050, 2];
        for (int i = 0; i < 2050; i++)
        {
            mask[i, 0] = (Half)1;
            mask[i, 1] = i < 2049 ? (Half)1 : (Half)0;
        }

        var order = IterativeImputer<Half>.AscendingMissingOrder(Tensor<Half>.FromMatrix(new Matrix<Half>(mask)));

        Assert.Equal(new[] { 1, 0 }, order);
    }

    // The fitted imputation sequence is replayed by every Transform, so it must not alias engine scratch that
    // a tensor arena hands out again after the fit.
    [Fact(Timeout = 120000)]
    public async Task IterativeImputer_FittedInArena_KeepsItsModels()
    {
        await Task.CompletedTask;
        var data = M(new double[,]
        {
            { 1, 2 }, { 2, 4.1 }, { 3, 6.2 }, { 4, double.NaN }, { 5, 9.9 }, { double.NaN, 12 }
        });
        var reference = new IterativeImputer<double>(estimator: IterativeImputerEstimator.LinearRegression);
        var expected = reference.FitTransform(data);

        var imputer = new IterativeImputer<double>(estimator: IterativeImputerEstimator.LinearRegression);
        using (AiDotNet.Tensors.Helpers.TensorArena.Create())
        {
            imputer.Fit(data);
        }

        using (AiDotNet.Tensors.Helpers.TensorArena.Create())
        {
            var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
            for (int i = 0; i < 64; i++)
            {
                _ = engine.TensorAddScalar(new Tensor<double>(new[] { 2 }), 1e6 + i);
                _ = engine.TensorAddScalar(new Tensor<double>(new[] { 2, 1 }), -1e6 - i);
            }
        }

        var actual = imputer.Transform(data);
        for (int i = 0; i < data.Rows; i++)
        {
            for (int j = 0; j < data.Columns; j++)
            {
                Assert.Equal(expected[i, j], actual[i, j], Tolerance);
            }
        }
    }

    // sklearn.preprocessing.normalize only reads the columns it normalizes: a NaN or infinity in a
    // passed-through column must not reach the norm of the selected ones.
    [Fact(Timeout = 120000)]
    public async Task Normalizer_NaNInUnselectedColumn_DoesNotPoisonTheRow()
    {
        await Task.CompletedTask;
        var normalizer = new Normalizer<double>(NormType.L2, columnIndices: new[] { 0, 1 });
        var result = normalizer.FitTransform(M(new double[,] { { 3, 4, double.NaN }, { 6, 8, double.PositiveInfinity } }));

        Assert.Equal(0.6, result[0, 0], Tolerance);
        Assert.Equal(0.8, result[0, 1], Tolerance);
        Assert.True(double.IsNaN(result[0, 2]));
        Assert.Equal(0.6, result[1, 0], Tolerance);
        Assert.Equal(0.8, result[1, 1], Tolerance);
        Assert.True(double.IsPositiveInfinity(result[1, 2]));
    }

    // numpy computes the percentile positions q (n - 1) in float64; in Half they overflow past 65504 samples
    // (and integers past 2048 are inexact), so the quantiles were read from invalid positions.
    [Fact(Timeout = 120000)]
    public async Task QuantileTransformer_Half_MoreThan65504Samples_MatchesDouble()
    {
        await Task.CompletedTask;
        const int rows = 70000;
        Func<int, double> value = i => (i % 1000) / 1000.0;
        var halfTransformer = new QuantileTransformer<Half>(nQuantiles: 11);
        halfTransformer.Fit(HalfColumn(rows, value));

        var doubleData = new double[rows, 1];
        for (int i = 0; i < rows; i++)
        {
            doubleData[i, 0] = (double)(Half)value(i);
        }

        var doubleTransformer = new QuantileTransformer<double>(nQuantiles: 11);
        doubleTransformer.Fit(M(doubleData));

        var probes = new[] { 0.0, 0.25, 0.5, 0.75, 0.999 };
        var halfResult = halfTransformer.Transform(HalfColumn(probes.Length, i => probes[i]));
        var doubleResult = doubleTransformer.Transform(M(new double[,] { { 0.0 }, { 0.25 }, { 0.5 }, { 0.75 }, { 0.999 } }));
        for (int i = 0; i < probes.Length; i++)
        {
            Assert.Equal(doubleResult[i, 0], (double)halfResult[i, 0], 5e-3);
        }
    }

    // ---- Review fixes for the element-wise sweep (#2514) ----

    private static Matrix<Half> HalfAlternating(int rows, double even, double odd) =>
        HalfColumn(rows, i => i % 2 == 0 ? even : odd);

    // numpy reduces float16 with float32 intermediates (np.mean / np.var / np.nanmean): 3000 Half samples of
    // 100 and 300 have mean 200 and std 100. Summed in Half the column total overflows past 65504.
    [Fact(Timeout = 120000)]
    public async Task HalfColumnStatistics_ManySamples_MatchNumpy()
    {
        await Task.CompletedTask;
        var data = HalfAlternating(3000, 100, 300);

        var standard = new StandardScaler<Half>();
        standard.Fit(data);
        Assert.Equal(200.0, (double)(standard.Mean?[0] ?? Half.NaN), 1e-6);
        Assert.Equal(100.0, (double)(standard.StandardDeviation?[0] ?? Half.NaN), 1e-6);
        Assert.Equal(1.0, (double)standard.Transform(HalfColumn(1, _ => 300))[0, 0], 1e-6);

        var global = new GlobalContrastScaler<Half>();
        global.Fit(data);
        Assert.Equal(200.0, (double)(global.Mean?[0] ?? Half.NaN), 1e-6);
        Assert.Equal(100.0, (double)(global.StdDev?[0] ?? Half.NaN), 1e-6);

        var clipper = new ZScoreClipper<Half>(threshold: 1.0);
        clipper.Fit(data);
        Assert.Equal(200.0, (double)(clipper.Means?[0] ?? Half.NaN), 1e-6);
        Assert.Equal(100.0, (double)(clipper.StandardDeviations?[0] ?? Half.NaN), 1e-6);

        var imputer = new SimpleImputer<Half>(ImputationStrategy.Mean);
        imputer.Fit(data);
        Assert.Equal(200.0, (double)(imputer.Statistics?[0] ?? Half.NaN), 1e-6);
    }

    // The 1e-10 variance floor of LogMeanVarianceScaler is 0 in Half; a constant column then divided by a zero
    // standard deviation and transformed every other value to infinity.
    [Fact(Timeout = 120000)]
    public async Task LogMeanVarianceScaler_Half_ConstantColumn_TransformsToFiniteValues()
    {
        await Task.CompletedTask;
        var scaler = new LogMeanVarianceScaler<Half>();
        scaler.Fit(HalfColumn(4, _ => 5));

        var result = scaler.Transform(HalfColumn(1, _ => 6));

        Assert.True(Half.IsFinite(result[0, 0]));
    }

    // scikit-learn SimpleImputer(strategy="most_frequent") returns the smallest of the most frequent values and
    // is sort-based (O(n log n)); a column of 400000 distinct values must not take a quadratic per-value count.
    [Fact(Timeout = 60000)]
    public async Task SimpleImputer_MostFrequent_ManyDistinctValues_IsSortBased()
    {
        await Task.CompletedTask;
        var tie = new SimpleImputer<double>(ImputationStrategy.MostFrequent);
        tie.Fit(M(new double[,] { { 3 }, { 1 }, { 3 }, { 1 }, { 2 }, { double.NaN } }));
        Assert.Equal(1.0, tie.Statistics?[0] ?? double.NaN, Tolerance);

        const int rows = 400000;
        var data = new double[rows, 1];
        for (int i = 0; i < rows; i++)
        {
            data[i, 0] = rows - i;
        }

        data[rows - 1, 0] = 7;
        var imputer = new SimpleImputer<double>(ImputationStrategy.MostFrequent);
        imputer.Fit(M(data));
        Assert.Equal(7.0, imputer.Statistics?[0] ?? double.NaN, Tolerance);
    }

    // sklearn.impute.KNNImputer(n_neighbors=1) computes nan_euclidean distances with wide intermediates: the
    // nearest donor of [302, nan] is [300, 1], so the imputed value is 1. In Half the squares overflowed
    // (inf - inf = NaN), no donor was usable and the column mean 34.33 was used.
    [Fact(Timeout = 120000)]
    public async Task KNNImputer_Half_LargeValues_UsesNearestDonor()
    {
        await Task.CompletedTask;
        var train = new Matrix<Half>(new[,]
        {
            { (Half)300, (Half)1 }, { (Half)310, (Half)2 }, { (Half)1000, (Half)100 }
        });
        var imputer = new KNNImputer<Half>(nNeighbors: 1);
        imputer.Fit(train);

        var result = imputer.Transform(new Matrix<Half>(new[,] { { (Half)302, Half.NaN } }));

        Assert.Equal(1.0, (double)result[0, 1], 1e-6);
    }

    // imbalanced-learn's NearestNeighbors computes Euclidean distances in float64; in Half the squared gaps of
    // 1000 and 300 both overflowed to infinity and tied, so the farther candidate (first in order) won.
    [Fact(Timeout = 120000)]
    public async Task ResamplingNearest_Half_LargeGaps_PicksTheNearest()
    {
        await Task.CompletedTask;
        var values = new[] { 0.0, 1000.0, 300.0 };
        var data = Tensor<Half>.FromMatrix(HalfColumn(3, i => values[i]));

        var nearest = ResamplingTensorOps.Nearest(data, 0, new[] { 1, 2 }, 1);

        Assert.Equal(new[] { 2 }, nearest);
    }

    // sklearn.preprocessing.normalize([[300, 400]], norm="l2") = [[0.6, 0.8]]; in Half 300^2 overflowed.
    [Fact(Timeout = 120000)]
    public async Task Normalizer_Half_LargeValues_MatchesSklearn()
    {
        await Task.CompletedTask;
        var result = new Normalizer<Half>(NormType.L2).FitTransform(
            new Matrix<Half>(new[,] { { (Half)300, (Half)400 } }));

        Assert.Equal(0.6, (double)result[0, 0], 1e-3);
        Assert.Equal(0.8, (double)result[0, 1], 1e-3);
    }

    // The L2 norm of the column [300, 400] is 500; in Half 300^2 + 400^2 overflowed to infinity.
    [Fact(Timeout = 120000)]
    public async Task LpNormScaler_Half_LargeValues_HasTheColumnNorm()
    {
        await Task.CompletedTask;
        var scaler = new LpNormScaler<Half>(2.0);
        scaler.Fit(HalfColumn(2, i => i == 0 ? 300 : 400));

        Assert.Equal(500.0, (double)(scaler.ColumnNorms?[0] ?? Half.NaN), 1e-6);
    }

    // sklearn IterativeImputer(estimator=LinearRegression(), max_iter=10, tol=0.0229) stops when
    // np.linalg.norm(Xt - Xt_previous, ord=np.inf) (the largest ROW SUM of absolute changes, not the largest
    // element) drops below tol * max|X observed|; it runs n_iter_ = 4 rounds on this data and returns these
    // values. Comparing the largest element stopped one round early.
    [Fact(Timeout = 120000)]
    public async Task IterativeImputer_Convergence_UsesMatrixInfinityNorm()
    {
        await Task.CompletedTask;
        var data = M(new double[,]
        {
            { 5, 0, -1 }, { 6, double.NaN, 4 }, { -7, 0, -2 }, { 1, 4, 0 }, { double.NaN, -8, double.NaN }, { 4, 0, -1 }
        });
        var imputer = new IterativeImputer<double>(
            maxIterations: 10, tolerance: 0.0229, estimator: IterativeImputerEstimator.LinearRegression);

        var result = imputer.FitTransform(data);

        Assert.Equal(-7.924393756742558, result[1, 1], 1e-9);
        Assert.Equal(5.333282926154977, result[4, 0], 1e-9);
        Assert.Equal(3.0288912480644274, result[4, 2], 1e-9);
    }

    // sklearn IterativeImputer(estimator=LinearRegression()) with exactly collinear predictors b = 2a: lstsq
    // returns the minimum-norm coefficients, which predict 2.273333333333333 for [2.5, 4]. Keeping the
    // rounding-noise eigenvalue of the collinear direction blows the coefficients up.
    [Fact(Timeout = 120000)]
    public async Task IterativeImputer_LinearRegression_CollinearFeatures_IsMinimumNorm()
    {
        await Task.CompletedTask;
        var data = M(new double[,]
        {
            { 1, 2, 3 }, { 2, 4, 1 }, { 3, 6, 4 }, { 4, 8, 1 }, { 5, 10, 5 }, { 6, 12, 9 }, { 2.5, 4, double.NaN }
        });
        var imputer = new IterativeImputer<double>(estimator: IterativeImputerEstimator.LinearRegression);

        var result = imputer.FitTransform(data);

        Assert.Equal(2.273333333333333, result[6, 2], 1e-9);
    }

    // sklearn QuantileTransformer(output_distribution="normal") clips to
    // norm.ppf(1 - (1e-7 - np.spacing(1))) = 5.199337582605575, computed in float64 for every dtype.
    [Fact(Timeout = 120000)]
    public async Task QuantileTransformer_Normal_ClipBoundIsComputedInFloat64()
    {
        await Task.CompletedTask;
        var floatData = new Matrix<float>(10, 1);
        for (int i = 0; i < 10; i++)
        {
            floatData[i, 0] = i;
        }

        var floatTransformer = new QuantileTransformer<float>(OutputDistributionType.Normal, nQuantiles: 10);
        floatTransformer.Fit(floatData);
        var floatResult = floatTransformer.Transform(new Matrix<float>(new[,] { { 100f }, { -100f } }));
        Assert.Equal(5.199337582605575, floatResult[0, 0], 1e-5);
        Assert.Equal(-5.199337582605575, floatResult[1, 0], 1e-5);

        var halfTransformer = new QuantileTransformer<Half>(OutputDistributionType.Normal, nQuantiles: 10);
        halfTransformer.Fit(HalfColumn(10, i => i));
        var halfResult = halfTransformer.Transform(HalfColumn(2, i => i == 0 ? 100 : -100));
        Assert.Equal(5.199337582605575, (double)halfResult[0, 0], 5e-3);
        Assert.Equal(-5.199337582605575, (double)halfResult[1, 0], 5e-3);
    }

    // sklearn PowerTransformer: a constant feature gets lambda 1 for Yeo-Johnson (_is_constant_feature), so
    // [[2], [3]] standardizes against mean 2, scale 1 to [0, 1]; Box-Cox raises "Data must not be constant."
    [Fact(Timeout = 120000)]
    public async Task PowerTransformer_ConstantFeature_MatchesSklearn()
    {
        await Task.CompletedTask;
        var constant = M(new double[,] { { 2 }, { 2 }, { 2 }, { 2 } });
        var yeoJohnson = new PowerTransformer<double>(PowerTransformMethod.YeoJohnson);
        yeoJohnson.Fit(constant);

        Assert.Equal(1.0, yeoJohnson.Lambdas?[0] ?? double.NaN, Tolerance);
        var result = yeoJohnson.Transform(M(new double[,] { { 2 }, { 3 } }));
        Assert.Equal(0.0, result[0, 0], Tolerance);
        Assert.Equal(1.0, result[1, 0], Tolerance);

        var boxCox = new PowerTransformer<double>(PowerTransformMethod.BoxCox);
        Assert.Throws<ArgumentException>(() => boxCox.Fit(constant));
    }

    // The Yeo-Johnson log-likelihood reduces in float64 (scipy); in Half the variance of 3000 transformed
    // values near 300 overflowed and the lambda search returned a wrong optimum.
    [Fact(Timeout = 120000)]
    public async Task PowerTransformer_Half_ManySamples_MatchesDoubleLambda()
    {
        await Task.CompletedTask;
        const int rows = 3000;
        Func<int, double> value = i => (double)(Half)(i * 0.1);
        var half = new PowerTransformer<Half>(PowerTransformMethod.YeoJohnson);
        half.Fit(HalfColumn(rows, value));

        var doubleData = new double[rows, 1];
        for (int i = 0; i < rows; i++)
        {
            doubleData[i, 0] = value(i);
        }

        var reference = new PowerTransformer<double>(PowerTransformMethod.YeoJohnson);
        reference.Fit(M(doubleData));

        Assert.Equal(reference.Lambdas?[0] ?? double.NaN, (double)(half.Lambdas?[0] ?? Half.NaN), 1e-2);
    }

    private sealed class FirstRowAnomalyDetector : AiDotNet.AnomalyDetection.AnomalyDetectorBase<Half>
    {
        public override void Fit(Matrix<Half> X)
        {
        }

        public override Vector<Half> ScoreAnomalies(Matrix<Half> X) => new Vector<Half>(X.Rows);

        public override Vector<Half> Predict(Matrix<Half> X)
        {
            var labels = new Half[X.Rows];
            for (int i = 0; i < labels.Length; i++)
            {
                labels[i] = i == 0 ? (Half)(-1) : (Half)1;
            }

            return new Vector<Half>(labels);
        }
    }

    // ReplaceWithMean replaces an anomalous row by the column mean, 200 for 3000 Half samples of 100 and 300;
    // averaged in Half the column total overflowed.
    [Fact(Timeout = 120000)]
    public async Task DetectorBasedFilter_Half_ReplaceWithMean_ManySamples()
    {
        await Task.CompletedTask;
        var data = HalfAlternating(3000, 100, 300);
        var filter = new DetectorBasedFilter<Half>(new FirstRowAnomalyDetector(), FilterMode.ReplaceWithMean);

        var result = filter.FitTransform(data);

        Assert.Equal(200.0, (double)result[0, 0], 1e-6);
        Assert.Equal(300.0, (double)result[1, 0], 1e-6);
    }

    // sklearn _handle_zeros_in_scale(scale, constant_mask=...) only replaces the features flagged constant:
    // StandardScaler().fit([[1e-16], [2e-16]]).scale_ = 5e-17 and the transform is [-1, 1].
    [Fact(Timeout = 120000)]
    public async Task StandardScaler_TinyNonConstantFeature_KeepsItsScale()
    {
        await Task.CompletedTask;
        var scaler = new StandardScaler<double>();

        var result = scaler.FitTransform(M(new double[,] { { 1e-16 }, { 2e-16 } }));

        Assert.Equal(5e-17, scaler.StandardDeviation?[0] ?? double.NaN, 1e-30);
        Assert.Equal(-1.0, result[0, 0], Tolerance);
        Assert.Equal(1.0, result[1, 0], Tolerance);
    }

    // _handle_zeros_in_scale uses 10 * np.finfo(scale.dtype).eps: a float32 scale of 1e-7 is below it and
    // becomes 1, and a Half all-zero column gets scale 1 (10 * float64 eps is 0 in Half, which left 0 / 0 = NaN).
    [Fact(Timeout = 120000)]
    public async Task MaxAbsScaler_ZeroScaleThreshold_UsesTheDataTypeEpsilon()
    {
        await Task.CompletedTask;
        var floatScaler = new MaxAbsScaler<float>();
        floatScaler.Fit(new Matrix<float>(new[,] { { 1e-7f }, { 5e-8f } }));
        Assert.Equal(1f, floatScaler.MaxAbsolute?[0] ?? float.NaN);
        Assert.Equal(1e-7f, floatScaler.Transform(new Matrix<float>(new[,] { { 1e-7f } }))[0, 0]);

        var halfScaler = new MaxAbsScaler<Half>();
        var zeros = halfScaler.FitTransform(HalfColumn(3, _ => 0));
        Assert.Equal(0.0, (double)zeros[0, 0]);
    }
}
