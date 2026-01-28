using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection;

/// <summary>
/// Base class for algorithmic anomaly detectors providing common functionality.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This base class provides shared functionality for all machine learning-based
/// anomaly detectors. It handles common tasks like parameter validation, threshold computation,
/// and distance calculations, allowing specific algorithms (like Isolation Forest or LOF) to focus
/// on their unique detection logic.
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Contamination: 0.1 (10%) - the expected proportion of anomalies
/// - Random Seed: 42 - for reproducibility
/// </para>
/// </remarks>
public abstract class AnomalyDetectorBase<T> : IAnomalyDetector<T>
{
    /// <summary>
    /// Provides numeric operations for the generic type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// The contamination parameter representing the expected proportion of anomalies in the data.
    /// Industry standard default is 0.1 (10%).
    /// </summary>
    protected readonly double _contamination;

    /// <summary>
    /// The random seed used for reproducibility.
    /// </summary>
    protected readonly int _randomSeed;

    /// <summary>
    /// Random number generator for algorithms that require randomization.
    /// </summary>
    protected readonly Random _random;

    /// <summary>
    /// The threshold for classifying samples as inliers or outliers.
    /// Initialized to zero using NumOps for proper generic type handling.
    /// </summary>
    protected T _threshold;

    /// <summary>
    /// Indicates whether the detector has been fitted to data.
    /// </summary>
    protected bool _isFitted;

    /// <summary>
    /// Initializes a new instance of the <see cref="AnomalyDetectorBase{T}"/> class.
    /// </summary>
    /// <param name="contamination">
    /// The expected proportion of anomalies in the training data (between 0 and 0.5).
    /// Default is 0.1 (10%), which is the industry standard for most anomaly detection scenarios.
    /// </param>
    /// <param name="randomSeed">
    /// Random seed for reproducibility. Default is 42.
    /// </param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The contamination parameter tells the algorithm roughly what percentage
    /// of your data you expect to be anomalous:
    /// - 0.1 (10%): Common default, good starting point for most datasets
    /// - 0.05 (5%): Use when anomalies are less frequent
    /// - 0.01 (1%): Use when anomalies are rare (e.g., fraud detection)
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when contamination is not between 0 (exclusive) and 0.5 (inclusive).
    /// </exception>
    protected AnomalyDetectorBase(double contamination = 0.1, int randomSeed = 42)
    {
        if (contamination <= 0 || contamination > 0.5)
        {
            throw new ArgumentOutOfRangeException(nameof(contamination),
                "Contamination must be between 0 (exclusive) and 0.5 (inclusive). " +
                "A value of 0.1 means you expect about 10% of your data to be anomalies.");
        }

        _contamination = contamination;
        _randomSeed = randomSeed;
        _random = RandomHelper.CreateSeededRandom(randomSeed);
        _threshold = NumOps.Zero;
        _isFitted = false;
    }

    /// <inheritdoc/>
    public T Threshold => _threshold;

    /// <inheritdoc/>
    public bool IsFitted => _isFitted;

    /// <inheritdoc/>
    public abstract void Fit(Matrix<T> X);

    /// <inheritdoc/>
    public abstract Vector<T> ScoreAnomalies(Matrix<T> X);

    /// <inheritdoc/>
    public virtual Vector<T> Predict(Matrix<T> X)
    {
        EnsureFitted();

        var scores = ScoreAnomalies(X);
        var predictions = new Vector<T>(scores.Length);

        for (int i = 0; i < scores.Length; i++)
        {
            // Scores above threshold are anomalies (-1), below are inliers (1)
            predictions[i] = NumOps.GreaterThan(scores[i], _threshold)
                ? NumOps.FromDouble(-1)
                : NumOps.FromDouble(1);
        }

        return predictions;
    }

    /// <summary>
    /// Validates that the input matrix is not null and has valid dimensions.
    /// </summary>
    /// <param name="X">The matrix to validate.</param>
    /// <param name="paramName">The parameter name for exception messages.</param>
    /// <exception cref="ArgumentNullException">Thrown when X is null.</exception>
    /// <exception cref="ArgumentException">Thrown when X has no rows or columns.</exception>
    protected static void ValidateInput(Matrix<T> X, string paramName = "X")
    {
        if (X is null)
        {
            throw new ArgumentNullException(paramName);
        }

        if (X.Rows == 0 || X.Columns == 0)
        {
            throw new ArgumentException(
                $"Input matrix must have at least one row and one column. Got {X.Rows} rows and {X.Columns} columns.",
                paramName);
        }
    }

    /// <summary>
    /// Ensures the detector has been fitted before prediction.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown when the detector has not been fitted.</exception>
    protected void EnsureFitted()
    {
        if (!_isFitted)
        {
            throw new InvalidOperationException(
                "This detector has not been fitted yet. Call Fit() with training data before calling Predict() or ScoreAnomalies().");
        }
    }

    /// <summary>
    /// Calculates the threshold based on the contamination parameter and scores.
    /// </summary>
    /// <param name="scores">The anomaly scores from the training data.</param>
    /// <remarks>
    /// Sets the threshold at the (1 - contamination) percentile of scores, so that
    /// approximately <c>contamination * 100</c>% of training samples will be
    /// classified as anomalies (those with scores above the threshold).
    /// </remarks>
    protected void SetThresholdFromContamination(Vector<T> scores)
    {
        // Sort scores to find the percentile
        var sortedScores = scores.ToArray();
        Array.Sort(sortedScores, (a, b) => NumOps.LessThan(a, b) ? -1 : (NumOps.GreaterThan(a, b) ? 1 : 0));

        // Find the index corresponding to the (1 - contamination) percentile
        // Higher scores = more anomalous, so we want the top contamination% to be anomalies
        int thresholdIndex = (int)Math.Floor((1 - _contamination) * sortedScores.Length);
        thresholdIndex = Math.Max(0, Math.Min(thresholdIndex, sortedScores.Length - 1));

        _threshold = sortedScores[thresholdIndex];
    }

}
