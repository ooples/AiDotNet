using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.TimeSeries;

/// <summary>
/// Detects anomalies using Matrix Profile for time series discord detection.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Matrix Profile computes the distance to the nearest neighbor subsequence
/// for every subsequence in a time series. Subsequences with no similar matches (discords)
/// are anomalies. It's one of the most powerful time series anomaly detection methods.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Extract all subsequences of length m
/// 2. For each subsequence, find the nearest neighbor (excluding trivial matches)
/// 3. Store these distances as the Matrix Profile
/// 4. High values in the Matrix Profile indicate discords (anomalies)
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Time series pattern anomaly detection
/// - Finding unusual subsequences
/// - Discord discovery in time series
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Subsequence length: 50 (or approximately one period)
/// - Exclusion zone: m/4
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Yeh, C.C.M., et al. (2016). "Matrix Profile I: All Pairs Similarity Joins for
/// Time Series: A Unifying View." IEEE ICDM.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.TimeSeriesModel)]
[ModelCategory(ModelCategory.InstanceBased)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Matrix Profile I: All Pairs Similarity Joins for Time Series: A Unifying View", "https://doi.org/10.1109/ICDM.2016.0179", Year = 2016, Authors = "Chin-Chia Michael Yeh, Yan Zhu, Liudmila Ulanova, Nurjahan Begum, Yifei Ding, Hoang Anh Dau, Diego Furtado Silva, Abdullah Mueen, Eamonn Keogh")]
public partial class MatrixProfileDetector<T> : AnomalyDetectorBase<T>
{
    /// <summary>Weight for the value-deviation component in the combined anomaly score.</summary>
    private const double ValueDeviationWeight = 0.1;

    private readonly int _subsequenceLength;
    private readonly int _exclusionZone;
    [Buffer]
    private Vector<T>? _matrixProfile;
    [Buffer]
    private Vector<T>? _trainingValues;
    private T _trainingChecksum;

    /// <summary>
    /// Gets the subsequence length.
    /// </summary>
    public int SubsequenceLength => _subsequenceLength;

    /// <summary>
    /// Gets the exclusion zone size.
    /// </summary>
    public int ExclusionZone => _exclusionZone;

    /// <summary>
    /// Creates a new Matrix Profile anomaly detector.
    /// </summary>
    /// <param name="subsequenceLength">Length of subsequences. Default is 50.</param>
    /// <param name="exclusionZone">
    /// Size of exclusion zone for trivial match prevention. -1 means auto (m/4). Default is -1.
    /// </param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public MatrixProfileDetector(int subsequenceLength = 50, int exclusionZone = -1,
        double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (subsequenceLength < 4)
        {
            throw new ArgumentOutOfRangeException(nameof(subsequenceLength),
                "SubsequenceLength must be at least 4. Recommended is 50 or one period length.");
        }

        if (exclusionZone < -1)
        {
            throw new ArgumentOutOfRangeException(nameof(exclusionZone),
                "ExclusionZone must be -1 (auto), 0 (no exclusion), or a positive value.");
        }

        _subsequenceLength = subsequenceLength;
        // -1 = auto (m/4), 0 = no exclusion zone, positive = explicit value
        _exclusionZone = exclusionZone == -1 ? subsequenceLength / 4 : exclusionZone;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        if (X.Columns != 1)
        {
            throw new ArgumentException(
                "Matrix Profile expects univariate time series (1 column).",
                nameof(X));
        }

        int n = X.Rows;

        if (n < _subsequenceLength + 1)
        {
            throw new ArgumentException(
                $"Time series length ({n}) must be greater than subsequence length ({_subsequenceLength}).",
                nameof(X));
        }

        // Extract values
        var engine = Engine;
        var series = engine.Reshape(AnomalyTensorOps.ToTensor(X), new[] { n });
        _trainingChecksum = Checksum(series);
        _trainingValues = series.ToVector();

        // Compute Matrix Profile (all-pairs z-normalized subsequence distances)
        _matrixProfile = ComputeMatrixProfile(series).ToVector();

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    // sum_i values[i] * (i + 1), identifying the training series.
    private T Checksum(Tensor<T> series)
    {
        var engine = Engine;
        int n = series.Length;
        return engine.TensorSum(engine.TensorMultiply(series, engine.TensorArange<T>(1, n + 1, 1)));
    }

    // The z-normalized subsequences of length m ([n - m + 1, m]); population std floored at 1e-10.
    private Tensor<T> NormalizedSubsequences(Tensor<T> series)
    {
        var engine = Engine;
        var windows = engine.TensorUnfold(series, 0, _subsequenceLength, 1);
        var (std, mean) = engine.TensorStdMean(windows, new[] { 1 }, 0, false);
        var flooredStd = engine.TensorMax(std, NumOps.FromDouble(1e-10));
        return AnomalyTensorOps.DivideColumn(engine.TensorSubtract(windows,
            AnomalyTensorOps.BroadcastColumn(mean, _subsequenceLength)), flooredStd);
    }

    private Tensor<T> ComputeMatrixProfile(Tensor<T> series)
    {
        var engine = Engine;
        var normalized = NormalizedSubsequences(series);
        int profileLength = normalized.Shape[0];

        // Nearest-neighbour distance of every subsequence, skipping the exclusion zone |i - j| <= zone
        // (MaxValue when every other subsequence is excluded)
        var distances = AnomalyTensorOps.PairwiseDistances(normalized, normalized);
        var ones = Tensor<T>.CreateOnes(new[] { profileLength, profileLength });
        int zone = Math.Min(_exclusionZone, profileLength);
        var band = engine.TensorSubtract(engine.TensorTril(ones, zone), engine.TensorTril(ones, -zone - 1));
        var excluded = engine.TensorEqScalar(band, NumOps.One);
        return engine.TensorAmin(engine.TensorMaskedFill(distances, excluded, NumOps.MaxValue), new[] { 1 }, false);
    }

    /// <inheritdoc/>
    public override Vector<T> ScoreAnomalies(Matrix<T> X)
    {
        EnsureFitted();
        return ScoreAnomaliesInternal(X);
    }

    private Vector<T> ScoreAnomaliesInternal(Matrix<T> X)
    {
        ValidateInput(X);

        if (X.Columns != 1)
        {
            throw new ArgumentException(
                "Matrix Profile expects univariate time series (1 column).",
                nameof(X));
        }

        int n = X.Rows;
        var engine = Engine;
        var values = engine.Reshape(AnomalyTensorOps.ToTensor(X), new[] { n });

        // For same data as training, use precomputed profile
        var trainingValues = _trainingValues;
        if (trainingValues == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        // Check if this is the same data as training using length and checksum
        // (allowing a small floating point tolerance in the checksum comparison)
        bool isSameData = n == trainingValues.Length
            && NumOps.LessThan(NumOps.Abs(NumOps.Subtract(Checksum(values), _trainingChecksum)), NumOps.FromDouble(1e-6));

        int m = _subsequenceLength;
        T scale = NumOps.FromDouble(2 * Math.Sqrt(m));
        T one = NumOps.One;
        if (isSameData)
        {
            var matrixProfile = _matrixProfile;
            if (matrixProfile == null)
            {
                throw new InvalidOperationException("Model not properly fitted.");
            }

            // Return profile-based scores.
            // For each time point, use the MAXIMUM profile value across all
            // subsequences that contain that point (at least 0). Also add a point-level
            // value deviation signal to disambiguate points within the same
            // high-discord subsequence (e.g., the spike vs. adjacent normal values).
            var padding = Tensor<T>.CreateZeros(new[] { Math.Max(m - 1, 1) });
            var padded = m > 1
                ? engine.TensorConcatenate(new[] { padding, matrixProfile.AsTensor(), padding }, 0)
                : matrixProfile.AsTensor();
            var maxProfileValue = engine.TensorMax(
                engine.ReduceMax(engine.TensorUnfold(padded, 0, m, 1), new[] { 1 }, false), NumOps.Zero);

            // Profile component: normalized discord distance
            var profileScore = engine.TensorDivideScalar(maxProfileValue, scale);

            // Value deviation component: how extreme this specific point is (global population std,
            // 1 when below 1e-10). This disambiguates the actual anomaly point from adjacent normal
            // points that share the same high-discord subsequence.
            var (globalStd, globalMean) = engine.TensorStdMean(values, new[] { 0 }, 0, true);
            var floorMask = engine.TensorLessThan(globalStd, NumOps.FromDouble(1e-10));
            var safeStd = engine.TensorWhere(floorMask, Tensor<T>.CreateOnes(new[] { 1 }), globalStd);
            var valueDeviation = engine.TensorDivide(
                engine.TensorAbs(engine.TensorSubtract(values, engine.TensorBroadcastTo(globalMean, new[] { n }))),
                engine.TensorBroadcastTo(safeStd, new[] { n }));

            // Combined score (weighted sum), at most 1
            var score = engine.TensorAdd(profileScore,
                engine.TensorMultiplyScalar(valueDeviation, NumOps.FromDouble(ValueDeviationWeight)));
            return engine.TensorMin(score, one).ToVector();
        }

        // For new data, compute distances to training subsequences
        return ComputeNewDataScores(values);
    }

    private Vector<T> ComputeNewDataScores(Tensor<T> values)
    {
        var trainingValues = _trainingValues;
        if (trainingValues == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        var engine = Engine;
        int n = values.Length;
        int m = _subsequenceLength;
        T scale = NumOps.FromDouble(2 * Math.Sqrt(m));
        if (n < m)
        {
            // No test subsequence contains any point: the distance stays MaxValue and the score saturates at 1
            return engine.TensorMin(engine.TensorDivideScalar(engine.Fill(n, NumOps.MaxValue).AsTensor(), scale), NumOps.One).ToVector();
        }

        // Minimum distance of every test subsequence to any training subsequence
        var trainNormalized = NormalizedSubsequences(trainingValues.AsTensor());
        var testNormalized = NormalizedSubsequences(values);
        var subsequenceMin = engine.TensorAmin(
            AnomalyTensorOps.PairwiseDistances(testNormalized, trainNormalized), new[] { 1 }, false);

        // For each point, the minimum over the test subsequences containing it
        var padding = engine.Fill(Math.Max(m - 1, 1), NumOps.MaxValue).AsTensor();
        var padded = m > 1 ? engine.TensorConcatenate(new[] { padding, subsequenceMin, padding }, 0) : subsequenceMin;
        var minDist = engine.TensorAmin(engine.TensorUnfold(padded, 0, m, 1), new[] { 1 }, false);

        // Normalize score
        return engine.TensorMin(engine.TensorDivideScalar(minDist, scale), NumOps.One).ToVector();
    }
}
