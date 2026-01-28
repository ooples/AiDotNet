using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.Ensemble;

/// <summary>
/// Combines multiple anomaly detectors using score averaging.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This ensemble method combines the predictions of multiple anomaly
/// detectors by averaging their normalized scores. This is one of the simplest and most
/// effective ensemble techniques.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Train each base detector on the data
/// 2. Normalize scores from each detector to [0,1]
/// 3. Average the normalized scores
/// 4. Points with high average score are anomalies
/// </para>
/// <para>
/// <b>When to use:</b>
/// - When you want robust predictions from multiple detectors
/// - To reduce variance in anomaly scores
/// - As a simple baseline ensemble
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Default detectors: LOF, IsolationForest, KNN
/// - Contamination: 0.1 (10%)
/// </para>
/// </remarks>
public class AveragingDetector<T> : AnomalyDetectorBase<T>
{
    private List<IAnomalyDetector<T>>? _baseDetectors;

    /// <summary>
    /// Creates a new Averaging ensemble anomaly detector with default base detectors.
    /// </summary>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public AveragingDetector(double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int n = X.Rows;
        int k = Math.Min(10, n - 1);

        // Create default base detectors
        _baseDetectors = new List<IAnomalyDetector<T>>
        {
            new DistanceBased.LocalOutlierFactor<T>(
                numNeighbors: k,
                contamination: _contamination,
                randomSeed: _randomSeed),

            new TreeBased.IsolationForest<T>(
                numTrees: 100,
                maxSamples: Math.Min(256, n),
                contamination: _contamination,
                randomSeed: _randomSeed + 1),

            new DistanceBased.KNNDetector<T>(
                k: k,
                contamination: _contamination,
                randomSeed: _randomSeed + 2)
        };

        // Fit all base detectors
        foreach (var detector in _baseDetectors)
        {
            detector.Fit(X);
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

        var baseDetectors = _baseDetectors;
        if (baseDetectors == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        // Collect and normalize scores from all detectors
        var allScores = new List<double[]>();

        foreach (var detector in baseDetectors)
        {
            var scores = detector.ScoreAnomalies(X);
            var doubleScores = NormalizeScores(scores);
            allScores.Add(doubleScores);
        }

        // Average the normalized scores
        var avgScores = new Vector<T>(X.Rows);

        for (int i = 0; i < X.Rows; i++)
        {
            double sum = 0;
            foreach (var scores in allScores)
            {
                sum += scores[i];
            }
            avgScores[i] = NumOps.FromDouble(sum / allScores.Count);
        }

        return avgScores;
    }

    private double[] NormalizeScores(Vector<T> scores)
    {
        int n = scores.Length;
        var result = new double[n];

        double min = double.MaxValue;
        double max = double.MinValue;

        for (int i = 0; i < n; i++)
        {
            result[i] = NumOps.ToDouble(scores[i]);
            if (result[i] < min) min = result[i];
            if (result[i] > max) max = result[i];
        }

        double range = max - min;
        if (range > 1e-10)
        {
            for (int i = 0; i < n; i++)
            {
                result[i] = (result[i] - min) / range;
            }
        }

        return result;
    }
}
