using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.DistanceBased;

/// <summary>
/// Detects anomalies using Local Outlier Probability (LoOP).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> LoOP improves on LOF by providing a probability score between 0 and 1,
/// making results easier to interpret. A score of 0 means the point is definitely not an outlier,
/// while 1 means it's definitely an outlier.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Compute probabilistic set distance (PLOF) using Gaussian error function
/// 2. Normalize using local standard deviation of LOF values
/// 3. Convert to probability using error function
/// </para>
/// <para>
/// <b>When to use:</b>
/// - When you need interpretable probability scores
/// - When comparing outlier scores across different datasets
/// - Similar use cases as LOF but with better interpretability
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - K (neighbors): 10
/// - Lambda: 3 (controls sensitivity)
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Kriegel, H., et al. (2009). "LoOP: Local Outlier Probabilities." CIKM.
/// </para>
/// </remarks>
public class LoOPDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _k;
    private readonly double _lambda;
    private Matrix<T>? _trainingData;
    private double[]? _probabilisticDistances;

    /// <summary>
    /// Gets the number of neighbors used for detection.
    /// </summary>
    public int K => _k;

    /// <summary>
    /// Gets the lambda parameter (standard deviations for probability).
    /// </summary>
    public double Lambda => _lambda;

    /// <summary>
    /// Creates a new LoOP anomaly detector.
    /// </summary>
    /// <param name="k">Number of nearest neighbors to consider. Default is 10.</param>
    /// <param name="lambda">
    /// Number of standard deviations for the probability estimate. Default is 3.
    /// Higher values make the detector less sensitive.
    /// </param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public LoOPDetector(int k = 10, double lambda = 3.0, double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (k < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(k),
                "K must be at least 1. Recommended value is 10.");
        }

        if (lambda <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(lambda),
                "Lambda must be positive. Recommended value is 3.");
        }

        _k = k;
        _lambda = lambda;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        if (X.Rows <= _k)
        {
            throw new ArgumentException(
                $"Number of samples ({X.Rows}) must be greater than k ({_k}).",
                nameof(X));
        }

        _trainingData = X;

        // Compute probabilistic distances for training data
        _probabilisticDistances = ComputeProbabilisticDistances(X);

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

        var scores = new Vector<T>(X.Rows);

        for (int i = 0; i < X.Rows; i++)
        {
            // Compute probabilistic distance for this point
            var neighbors = GetKNearestNeighbors(X, i);
            double pdist = ComputePDist(X, i, neighbors);

            // Compute expected value E[pdist] over neighbors
            double ePdist = 0;
            foreach (int neighborIdx in neighbors)
            {
                ePdist += _probabilisticDistances![neighborIdx];
            }
            ePdist /= neighbors.Count;

            // Compute PLOF (Probabilistic LOF)
            double plof = ePdist > 0 ? (pdist / ePdist) - 1 : 0;

            // Compute standard deviation of PLOF in neighborhood
            double nplof = ComputeNPLOF(neighbors);

            // Compute LoOP using error function
            // LoOP = max(0, erf(plof / (sqrt(2) * nplof)))
            double loop = 0;
            if (nplof > 0)
            {
                double z = plof / (Math.Sqrt(2) * _lambda * nplof);
                loop = Math.Max(0, ErrorFunction(z));
            }
            else if (plof > 0)
            {
                loop = 1.0;
            }

            scores[i] = NumOps.FromDouble(loop);
        }

        return scores;
    }

    private double[] ComputeProbabilisticDistances(Matrix<T> X)
    {
        var pdist = new double[X.Rows];

        for (int i = 0; i < X.Rows; i++)
        {
            var neighbors = GetKNearestNeighborsFromTraining(i);
            pdist[i] = ComputePDistFromTraining(i, neighbors);
        }

        return pdist;
    }

    private double ComputePDist(Matrix<T> X, int pointIdx, List<int> neighbors)
    {
        // Probabilistic set distance: sqrt(sum(d^2) / k)
        double sumSquaredDist = 0;
        var point = new Vector<T>(X.Columns);
        for (int j = 0; j < X.Columns; j++)
        {
            point[j] = X[pointIdx, j];
        }

        var trainingData = _trainingData;
        if (trainingData == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        foreach (int neighborIdx in neighbors)
        {
            var neighbor = new Vector<T>(trainingData.Columns);
            for (int j = 0; j < trainingData.Columns; j++)
            {
                neighbor[j] = trainingData[neighborIdx, j];
            }

            double dist = NumOps.ToDouble(StatisticsHelper<T>.EuclideanDistance(point, neighbor));
            sumSquaredDist += dist * dist;
        }

        return Math.Sqrt(sumSquaredDist / neighbors.Count);
    }

    private double ComputePDistFromTraining(int pointIdx, List<int> neighbors)
    {
        var trainingData = _trainingData;
        if (trainingData == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        double sumSquaredDist = 0;
        var point = new Vector<T>(trainingData.Columns);
        for (int j = 0; j < trainingData.Columns; j++)
        {
            point[j] = trainingData[pointIdx, j];
        }

        foreach (int neighborIdx in neighbors)
        {
            var neighbor = new Vector<T>(trainingData.Columns);
            for (int j = 0; j < trainingData.Columns; j++)
            {
                neighbor[j] = trainingData[neighborIdx, j];
            }

            double dist = NumOps.ToDouble(StatisticsHelper<T>.EuclideanDistance(point, neighbor));
            sumSquaredDist += dist * dist;
        }

        return neighbors.Count > 0 ? Math.Sqrt(sumSquaredDist / neighbors.Count) : 0;
    }

    private double ComputeNPLOF(List<int> neighbors)
    {
        // Normalization factor for PLOF
        var probabilisticDistances = _probabilisticDistances;
        if (probabilisticDistances == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        double sumSquaredRatio = 0;

        foreach (int neighborIdx in neighbors)
        {
            var neighborNeighbors = GetKNearestNeighborsFromTraining(neighborIdx);
            double pdist = probabilisticDistances[neighborIdx];

            double ePdist = 0;
            foreach (int nn in neighborNeighbors)
            {
                ePdist += probabilisticDistances[nn];
            }
            ePdist /= neighborNeighbors.Count;

            double plof = ePdist > 0 ? (pdist / ePdist) - 1 : 0;
            sumSquaredRatio += plof * plof;
        }

        return _lambda * Math.Sqrt(sumSquaredRatio / neighbors.Count);
    }

    private List<int> GetKNearestNeighbors(Matrix<T> X, int pointIdx)
    {
        var trainingData = _trainingData;
        if (trainingData == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        var point = new Vector<T>(X.Columns);
        for (int j = 0; j < X.Columns; j++)
        {
            point[j] = X[pointIdx, j];
        }

        return Enumerable.Range(0, trainingData.Rows)
            .Select(i =>
            {
                var refPoint = new Vector<T>(trainingData.Columns);
                for (int j = 0; j < trainingData.Columns; j++)
                {
                    refPoint[j] = trainingData[i, j];
                }
                return (Index: i, Dist: NumOps.ToDouble(StatisticsHelper<T>.EuclideanDistance(point, refPoint)));
            })
            .Where(x => x.Dist > 0)
            .OrderBy(x => x.Dist)
            .Take(_k)
            .Select(x => x.Index)
            .ToList();
    }

    private List<int> GetKNearestNeighborsFromTraining(int pointIdx)
    {
        var trainingData = _trainingData;
        if (trainingData == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        var point = new Vector<T>(trainingData.Columns);
        for (int j = 0; j < trainingData.Columns; j++)
        {
            point[j] = trainingData[pointIdx, j];
        }

        return Enumerable.Range(0, trainingData.Rows)
            .Where(i => i != pointIdx)
            .Select(i =>
            {
                var refPoint = new Vector<T>(trainingData.Columns);
                for (int j = 0; j < trainingData.Columns; j++)
                {
                    refPoint[j] = trainingData[i, j];
                }
                return (Index: i, Dist: NumOps.ToDouble(StatisticsHelper<T>.EuclideanDistance(point, refPoint)));
            })
            .OrderBy(x => x.Dist)
            .Take(_k)
            .Select(x => x.Index)
            .ToList();
    }

    /// <summary>
    /// Approximation of the error function (erf).
    /// </summary>
    private static double ErrorFunction(double x)
    {
        // Horner form approximation
        double a1 = 0.254829592;
        double a2 = -0.284496736;
        double a3 = 1.421413741;
        double a4 = -1.453152027;
        double a5 = 1.061405429;
        double p = 0.3275911;

        int sign = x < 0 ? -1 : 1;
        x = Math.Abs(x);

        double t = 1.0 / (1.0 + p * x);
        double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.Exp(-x * x);

        return sign * y;
    }
}
