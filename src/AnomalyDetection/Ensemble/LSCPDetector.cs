using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.Ensemble;

/// <summary>
/// Detects anomalies using LSCP (Locally Selective Combination in Parallel Outlier Ensembles).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> LSCP improves ensemble detection by selecting the most competent
/// detectors for each test point locally. Instead of combining all detectors equally,
/// it selects the best performing detectors based on local pseudo ground truth.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Train multiple diverse base detectors
/// 2. For each test point, find local region
/// 3. Evaluate detector competence in that region
/// 4. Combine scores from most competent detectors
/// </para>
/// <para>
/// <b>When to use:</b>
/// - When different detectors work better in different regions
/// - For heterogeneous anomaly types
/// - When ensemble averaging isn't optimal
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - N estimators: 10
/// - Local region size: 30 neighbors
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Zhao, Y., et al. (2019). "LSCP: Locally Selective Combination in Parallel
/// Outlier Ensembles." SDM.
/// </para>
/// </remarks>
public class LSCPDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _nEstimators;
    private readonly int _localRegionSize;
    private List<IAnomalyDetector<T>>? _baseDetectors;
    private double[][]? _trainingData;
    private double[][]? _detectorScores;

    /// <summary>
    /// Gets the number of estimators.
    /// </summary>
    public int NEstimators => _nEstimators;

    /// <summary>
    /// Gets the local region size.
    /// </summary>
    public int LocalRegionSize => _localRegionSize;

    /// <summary>
    /// Creates a new LSCP anomaly detector.
    /// </summary>
    /// <param name="nEstimators">Number of base detectors. Default is 10.</param>
    /// <param name="localRegionSize">
    /// Number of neighbors for local region. Default is 30.
    /// </param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public LSCPDetector(int nEstimators = 10, int localRegionSize = 30,
        double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (nEstimators < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(nEstimators),
                "NEstimators must be at least 2. Recommended is 10.");
        }

        if (localRegionSize < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(localRegionSize),
                "LocalRegionSize must be at least 2. Recommended is 30.");
        }

        _nEstimators = nEstimators;
        _localRegionSize = localRegionSize;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int n = X.Rows;

        // Convert to double array
        _trainingData = new double[n][];
        for (int i = 0; i < n; i++)
        {
            _trainingData[i] = new double[X.Columns];
            for (int j = 0; j < X.Columns; j++)
            {
                _trainingData[i][j] = NumOps.ToDouble(X[i, j]);
            }
        }

        // Create diverse base detectors
        _baseDetectors = new List<IAnomalyDetector<T>>();
        _detectorScores = new double[_nEstimators][];

        int k = Math.Min(10, n - 1);

        for (int e = 0; e < _nEstimators; e++)
        {
            IAnomalyDetector<T> detector;

            // Create diverse detector types
            switch (e % 3)
            {
                case 0:
                    detector = new DistanceBased.LocalOutlierFactor<T>(
                        numNeighbors: Math.Max(1, k + e - 1),
                        contamination: _contamination,
                        randomSeed: _randomSeed + e);
                    break;
                case 1:
                    detector = new DistanceBased.KNNDetector<T>(
                        k: Math.Max(1, k + e - 1),
                        contamination: _contamination,
                        randomSeed: _randomSeed + e);
                    break;
                default:
                    detector = new TreeBased.IsolationForest<T>(
                        numTrees: 50 + e * 10,
                        maxSamples: Math.Min(256, n),
                        contamination: _contamination,
                        randomSeed: _randomSeed + e);
                    break;
            }

            detector.Fit(X);
            _baseDetectors.Add(detector);

            // Store training scores
            var scores = detector.ScoreAnomalies(X);
            _detectorScores[e] = new double[n];
            for (int i = 0; i < n; i++)
            {
                _detectorScores[e][i] = NumOps.ToDouble(scores[i]);
            }
        }

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
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

        var trainingData = _trainingData;
        var baseDetectors = _baseDetectors;
        if (trainingData == null || baseDetectors == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        var scores = new Vector<T>(X.Rows);
        int effectiveRegionSize = Math.Min(_localRegionSize, trainingData.Length);

        // Get scores from all detectors for test data
        var testScores = new double[_nEstimators][];
        for (int e = 0; e < _nEstimators; e++)
        {
            var detectorScores = baseDetectors[e].ScoreAnomalies(X);
            testScores[e] = new double[X.Rows];
            for (int i = 0; i < X.Rows; i++)
            {
                testScores[e][i] = NumOps.ToDouble(detectorScores[i]);
            }
        }

        for (int i = 0; i < X.Rows; i++)
        {
            var point = new double[X.Columns];
            for (int j = 0; j < X.Columns; j++)
            {
                point[j] = NumOps.ToDouble(X[i, j]);
            }

            // Find local region (nearest neighbors in training data)
            var distances = new (double dist, int idx)[trainingData.Length];
            for (int t = 0; t < trainingData.Length; t++)
            {
                double dist = EuclideanDistance(point, trainingData[t]);
                distances[t] = (dist, t);
            }

            var localRegion = distances
                .OrderBy(d => d.dist)
                .Take(effectiveRegionSize)
                .Select(d => d.idx)
                .ToArray();

            // Evaluate detector competence in local region
            // Use Pearson correlation between scores and distance rank as competence
            var competence = new double[_nEstimators];

            for (int e = 0; e < _nEstimators; e++)
            {
                var detectorScores = _detectorScores;
                if (detectorScores == null)
                {
                    throw new InvalidOperationException("Model not properly fitted.");
                }

                var localScores = localRegion.Select(idx => detectorScores[e][idx]).ToArray();
                var localDistances = localRegion.Select(idx => distances[idx].dist).ToArray();

                // Higher scores should correlate with higher distances for good detectors
                competence[e] = ComputeCorrelation(localScores, localDistances);
            }

            // Select top detectors and combine their scores
            int nSelect = Math.Max(1, _nEstimators / 2);
            var selectedDetectors = competence
                .Select((c, idx) => (c, idx))
                .OrderByDescending(x => x.c)
                .Take(nSelect)
                .Select(x => x.idx)
                .ToArray();

            // Average scores from selected detectors
            double combinedScore = selectedDetectors.Average(e => testScores[e][i]);
            scores[i] = NumOps.FromDouble(combinedScore);
        }

        return scores;
    }

    private static double EuclideanDistance(double[] a, double[] b)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.Sqrt(sum);
    }

    private static double ComputeCorrelation(double[] x, double[] y)
    {
        int n = x.Length;
        if (n < 2) return 0;

        double meanX = x.Average();
        double meanY = y.Average();

        double cov = 0, varX = 0, varY = 0;
        for (int i = 0; i < n; i++)
        {
            double dx = x[i] - meanX;
            double dy = y[i] - meanY;
            cov += dx * dy;
            varX += dx * dx;
            varY += dy * dy;
        }

        double denom = Math.Sqrt(varX * varY);
        return denom > 1e-10 ? cov / denom : 0;
    }
}
