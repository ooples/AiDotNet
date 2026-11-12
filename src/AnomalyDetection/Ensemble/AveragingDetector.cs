using AiDotNet.Attributes;
using AiDotNet.Enums;
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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
    [ResearchPaper("Outlier Ensembles: An Introduction", "https://doi.org/10.1007/978-3-319-54765-7")]
public partial class AveragingDetector<T> : AnomalyDetectorBase<T>
{
    private List<IAnomalyDetector<T>>? _baseDetectors;
    private List<(T Min, T Max)>? _trainingScoreRanges;

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

        // Guard against tiny datasets - need at least 2 samples for neighbor-based methods
        if (n < 2)
        {
            throw new ArgumentException(
                $"AveragingDetector requires at least 2 samples, but got {n}.",
                nameof(X));
        }

        int k = Math.Max(1, Math.Min(10, n - 1));

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

        // Fit all base detectors and compute training score ranges
        _trainingScoreRanges = new List<(T Min, T Max)>();

        foreach (var detector in _baseDetectors)
        {
            detector.Fit(X);

            // Compute training scores to establish min/max for normalization
            var scores = detector.ScoreAnomalies(X);
            _trainingScoreRanges.Add(AnomalyTensorOps.MinMaxIgnoringNaN(scores));
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
        var trainingRanges = _trainingScoreRanges;
        if (baseDetectors == null || trainingRanges == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        // Collect and normalize scores from all detectors using training min/max
        var allScores = new List<Vector<T>>();

        for (int d = 0; d < baseDetectors.Count; d++)
        {
            var scores = baseDetectors[d].ScoreAnomalies(X);
            var (trainMin, trainMax) = trainingRanges[d];
            var normalizedScores = NormalizeScores(scores, trainMin, trainMax);
            allScores.Add(normalizedScores);
        }

        // Average the normalized scores
        return AnomalyTensorOps.ElementwiseMean(allScores);
    }

    private Vector<T> NormalizeScores(Vector<T> scores, T trainMin, T trainMax)
    {
        T range = NumOps.Subtract(trainMax, trainMin);
        T eps = NumOps.FromDouble(1e-10);
        T half = NumOps.FromDouble(0.5);

        if (!NumOps.GreaterThan(range, eps))
        {
            return Engine.Fill(scores.Length, half); // Constant scores map to midpoint
        }

        if (scores.Length == 0)
        {
            return new Vector<T>(0);
        }

        // Normalize using training min/max, clamp to [0, 1]
        var normalized = Engine.TensorDivideScalar(Engine.TensorSubtractScalar(scores.AsTensor(), trainMin), range);
        return Engine.TensorClamp(normalized, NumOps.Zero, NumOps.One).ToVector();
    }
}
