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
public class MatrixProfileDetector<T> : AnomalyDetectorBase<T>
{
    /// <summary>Weight for the value-deviation component in the combined anomaly score.</summary>
    private const double ValueDeviationWeight = 0.1;

    private readonly int _subsequenceLength;
    private readonly int _exclusionZone;
    private double[]? _matrixProfile;
    private double[]? _trainingValues;
    private double _trainingChecksum;

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
        _trainingValues = new double[n];
        double checksum = 0;
        for (int i = 0; i < n; i++)
        {
            _trainingValues[i] = NumOps.ToDouble(X[i, 0]);
            // Compute a weighted checksum for same-data detection
            checksum += _trainingValues[i] * (i + 1);
        }
        _trainingChecksum = checksum;

        // Compute Matrix Profile using STOMP algorithm (simplified)
        _matrixProfile = ComputeMatrixProfile(_trainingValues);

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    private double[] ComputeMatrixProfile(double[] values)
    {
        int n = values.Length;
        int m = _subsequenceLength;
        int profileLength = n - m + 1;

        var profile = new double[profileLength];
        for (int i = 0; i < profileLength; i++)
        {
            profile[i] = double.MaxValue;
        }

        // Compute means and standard deviations for all subsequences
        var means = new double[profileLength];
        var stds = new double[profileLength];

        for (int i = 0; i < profileLength; i++)
        {
            double sum = 0, sumSq = 0;
            for (int j = 0; j < m; j++)
            {
                double v = values[i + j];
                sum += v;
                sumSq += v * v;
            }
            means[i] = sum / m;
            double variance = sumSq / m - means[i] * means[i];
            stds[i] = Math.Sqrt(Math.Max(0, variance));
            if (stds[i] < 1e-10) stds[i] = 1e-10;
        }

        // Compute distance profile for each starting position
        for (int i = 0; i < profileLength; i++)
        {
            // Extract and normalize query subsequence
            var query = new double[m];
            for (int j = 0; j < m; j++)
            {
                query[j] = (values[i + j] - means[i]) / stds[i];
            }

            // Compute distances to all other subsequences
            for (int j = 0; j < profileLength; j++)
            {
                // Skip exclusion zone
                if (Math.Abs(i - j) <= _exclusionZone)
                {
                    continue;
                }

                // Compute normalized Euclidean distance
                double dist = 0;
                for (int k = 0; k < m; k++)
                {
                    double normalizedK = (values[j + k] - means[j]) / stds[j];
                    double diff = query[k] - normalizedK;
                    dist += diff * diff;
                }
                dist = Math.Sqrt(dist);

                // Update matrix profile
                if (dist < profile[i])
                {
                    profile[i] = dist;
                }
                if (dist < profile[j])
                {
                    profile[j] = dist;
                }
            }
        }

        return profile;
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
        var values = new double[n];
        for (int i = 0; i < n; i++)
        {
            values[i] = NumOps.ToDouble(X[i, 0]);
        }

        // For same data as training, use precomputed profile
        var trainingValues = _trainingValues;
        if (trainingValues == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        // Check if this is the same data as training using length and checksum
        bool isSameData = false;
        if (n == trainingValues.Length)
        {
            double checksum = 0;
            for (int i = 0; i < n; i++)
            {
                checksum += values[i] * (i + 1);
            }
            // Allow small floating point tolerance in checksum comparison
            isSameData = Math.Abs(checksum - _trainingChecksum) < 1e-6;
        }

        if (isSameData)
            {
                var matrixProfile = _matrixProfile;
                if (matrixProfile == null)
                {
                    throw new InvalidOperationException("Model not properly fitted.");
                }

                // Return profile-based scores.
                // For each time point, use the MAXIMUM profile value across all
                // subsequences that contain that point. Also add a point-level
                // value deviation signal to disambiguate points within the same
                // high-discord subsequence (e.g., the spike vs. adjacent normal values).
                var scores = new Vector<T>(n);
                int m = _subsequenceLength;
                int profileLength = n - m + 1;

                // Compute global mean and std for value deviation scoring
                double globalMean = 0;
                for (int i = 0; i < n; i++)
                {
                    globalMean += values[i];
                }
                globalMean /= n;

                double globalVar = 0;
                for (int i = 0; i < n; i++)
                {
                    double diff = values[i] - globalMean;
                    globalVar += diff * diff;
                }
                double globalStd = Math.Sqrt(globalVar / n);
                if (globalStd < 1e-10) globalStd = 1.0;

                for (int i = 0; i < n; i++)
                {
                    double maxProfileValue = 0;

                    // Find all subsequences that contain point i
                    int subStart = Math.Max(0, i - m + 1);
                    int subEnd = Math.Min(i, profileLength - 1);

                    for (int s = subStart; s <= subEnd; s++)
                    {
                        if (matrixProfile[s] > maxProfileValue)
                        {
                            maxProfileValue = matrixProfile[s];
                        }
                    }

                    // Profile component: normalized discord distance
                    double profileScore = maxProfileValue / (2 * Math.Sqrt(m));

                    // Value deviation component: how extreme this specific point is.
                    // This disambiguates the actual anomaly point from adjacent normal
                    // points that share the same high-discord subsequence.
                    double valueDeviation = Math.Abs(values[i] - globalMean) / globalStd;

                    // Combined score (weighted sum)
                    double score = profileScore + ValueDeviationWeight * valueDeviation;
                    scores[i] = NumOps.FromDouble(Math.Min(score, 1.0));
                }

                return scores;
        }

        // For new data, compute distances to training subsequences
        return ComputeNewDataScores(values);
    }

    private Vector<T> ComputeNewDataScores(double[] values)
    {
        var trainingValues = _trainingValues;
        if (trainingValues == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        int n = values.Length;
        int m = _subsequenceLength;
        int nTrain = trainingValues.Length;

        // Compute training subsequence statistics
        int trainProfileLen = nTrain - m + 1;
        var trainMeans = new double[trainProfileLen];
        var trainStds = new double[trainProfileLen];

        for (int i = 0; i < trainProfileLen; i++)
        {
            double sum = 0, sumSq = 0;
            for (int j = 0; j < m; j++)
            {
                double v = trainingValues[i + j];
                sum += v;
                sumSq += v * v;
            }
            trainMeans[i] = sum / m;
            double variance = sumSq / m - trainMeans[i] * trainMeans[i];
            trainStds[i] = Math.Sqrt(Math.Max(0, variance));
            if (trainStds[i] < 1e-10) trainStds[i] = 1e-10;
        }

        var scores = new Vector<T>(n);

        // For each point, find minimum distance to any training subsequence
        for (int i = 0; i < n; i++)
        {
            double minDist = double.MaxValue;

            // Check subsequences ending at or after this point
            int start = Math.Max(0, i - m + 1);
            int end = Math.Min(n - m + 1, i + 1);

            for (int s = start; s < end; s++)
            {
                // Compute mean and std of test subsequence
                double sum = 0, sumSq = 0;
                for (int j = 0; j < m; j++)
                {
                    double v = values[s + j];
                    sum += v;
                    sumSq += v * v;
                }
                double mean = sum / m;
                double variance = sumSq / m - mean * mean;
                double std = Math.Sqrt(Math.Max(0, variance));
                if (std < 1e-10) std = 1e-10;

                // Find nearest training subsequence
                for (int t = 0; t < trainProfileLen; t++)
                {
                    double dist = 0;
                    for (int k = 0; k < m; k++)
                    {
                        double testNorm = (values[s + k] - mean) / std;
                        double trainNorm = (trainingValues[t + k] - trainMeans[t]) / trainStds[t];
                        double diff = testNorm - trainNorm;
                        dist += diff * diff;
                    }
                    dist = Math.Sqrt(dist);

                    if (dist < minDist)
                    {
                        minDist = dist;
                    }
                }
            }

            // Normalize score
            double score = minDist / (2 * Math.Sqrt(m));
            scores[i] = NumOps.FromDouble(Math.Min(score, 1.0));
        }

        return scores;
    }
}
