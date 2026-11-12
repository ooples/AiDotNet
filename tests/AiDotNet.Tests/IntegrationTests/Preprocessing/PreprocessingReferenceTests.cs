using AiDotNet.Preprocessing.PowerTransforms;
using AiDotNet.Preprocessing.ImbalancedLearning;
using AiDotNet.Preprocessing.Imputers;
using AiDotNet.AnomalyDetection.Statistical;
using AiDotNet.Preprocessing.OutlierHandling;
using AiDotNet.Preprocessing.Scalers;
using Xunit;
using System.Threading.Tasks;

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

    // Iglewicz & Hoaglin: the clip bound is exactly where the modified Z-score 0.6745 * (x - median) / MAD
    // reaches the threshold, so GetModifiedZScores at the bound returns the threshold.
    [Fact(Timeout = 120000)]
    public async Task MADClipper_BoundsAgreeWithModifiedZScore()
    {
        await Task.CompletedTask;
        var clipper = new MADClipper<double>(threshold: 3.5);
        clipper.Fit(M(new double[,] { { 1 }, { 2 }, { 3 }, { 4 }, { 100 } }));

        double upper = clipper.UpperBounds?[0] ?? double.NaN;
        double lower = clipper.LowerBounds?[0] ?? double.NaN;
        Assert.Equal(3.0 + 3.5 / 0.6745, upper, Tolerance);
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
}
