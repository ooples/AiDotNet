using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.OutlierRemoval;

/// <summary>
/// Base class for algorithmic outlier detectors providing common functionality.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This base class provides shared functionality for all machine learning-based
/// outlier detectors. It handles common tasks like parameter validation and type conversions,
/// allowing specific algorithms (like Isolation Forest or LOF) to focus on their unique logic.
/// </para>
/// </remarks>
public abstract class OutlierDetectorBase<T> : IOutlierDetector<T>
{
    /// <summary>
    /// Provides numeric operations for the generic type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// The contamination parameter representing the expected proportion of outliers in the data.
    /// </summary>
    protected readonly double _contamination;

    /// <summary>
    /// Random number generator for algorithms that require randomization.
    /// </summary>
    protected readonly Random _random;

    /// <summary>
    /// The threshold for classifying samples as inliers or outliers.
    /// </summary>
    protected T _threshold;

    /// <summary>
    /// Indicates whether the detector has been fitted to data.
    /// </summary>
    protected bool _isFitted;

    /// <summary>
    /// Initializes a new instance of the <see cref="OutlierDetectorBase{T}"/> class.
    /// </summary>
    /// <param name="contamination">
    /// The expected proportion of outliers in the training data (between 0 and 0.5).
    /// Default is 0.1 (10%), which is a common assumption in anomaly detection.
    /// </param>
    /// <param name="randomSeed">
    /// Random seed for reproducibility. Default is 42.
    /// </param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The contamination parameter tells the algorithm roughly what percentage
    /// of your data you expect to be outliers. For example, if you think about 10% of your data
    /// might be anomalous, use 0.1. This helps the algorithm set an appropriate threshold.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when contamination is not between 0 (exclusive) and 0.5 (inclusive).
    /// </exception>
    protected OutlierDetectorBase(double contamination = 0.1, int randomSeed = 42)
    {
        if (contamination <= 0 || contamination > 0.5)
        {
            throw new ArgumentOutOfRangeException(nameof(contamination),
                "Contamination must be between 0 (exclusive) and 0.5 (inclusive). " +
                "A value of 0.1 means you expect about 10% of your data to be outliers.");
        }

        _contamination = contamination;
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
    public abstract Vector<T> DecisionFunction(Matrix<T> X);

    /// <inheritdoc/>
    public virtual Vector<T> Predict(Matrix<T> X)
    {
        EnsureFitted();

        var scores = DecisionFunction(X);
        var predictions = new Vector<T>(scores.Length);

        for (int i = 0; i < scores.Length; i++)
        {
            // Scores below threshold are outliers (-1), above are inliers (1)
            predictions[i] = NumOps.LessThan(scores[i], _threshold)
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
                "This detector has not been fitted yet. Call Fit() with training data before calling Predict() or DecisionFunction().");
        }
    }

    /// <summary>
    /// Calculates the threshold based on the contamination parameter and scores.
    /// </summary>
    /// <param name="scores">The anomaly scores from the training data.</param>
    /// <remarks>
    /// Sets the threshold at the contamination percentile of scores, so that
    /// approximately <c>contamination * 100</c>% of training samples will be
    /// classified as outliers.
    /// </remarks>
    protected void SetThresholdFromContamination(Vector<T> scores)
    {
        // Sort scores to find the percentile
        var sortedScores = scores.ToArray();
        Array.Sort(sortedScores, (a, b) => NumOps.LessThan(a, b) ? -1 : (NumOps.GreaterThan(a, b) ? 1 : 0));

        // Find the index corresponding to the contamination percentile
        int thresholdIndex = (int)Math.Floor(_contamination * sortedScores.Length);
        thresholdIndex = Math.Max(0, Math.Min(thresholdIndex, sortedScores.Length - 1));

        _threshold = sortedScores[thresholdIndex];
    }

    /// <summary>
    /// Calculates the Euclidean distance between two vectors.
    /// </summary>
    /// <param name="a">The first vector.</param>
    /// <param name="b">The second vector.</param>
    /// <returns>The Euclidean distance between the vectors.</returns>
    protected static T EuclideanDistance(Vector<T> a, Vector<T> b)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < a.Length; i++)
        {
            T diff = NumOps.Subtract(a[i], b[i]);
            sum = NumOps.Add(sum, NumOps.Multiply(diff, diff));
        }

        return NumOps.Sqrt(sum);
    }

    /// <summary>
    /// Calculates the Euclidean distance between a vector and a matrix row.
    /// </summary>
    /// <param name="a">The vector.</param>
    /// <param name="X">The matrix.</param>
    /// <param name="rowIndex">The row index in the matrix.</param>
    /// <returns>The Euclidean distance.</returns>
    protected static T EuclideanDistance(Vector<T> a, Matrix<T> X, int rowIndex)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < a.Length; i++)
        {
            T diff = NumOps.Subtract(a[i], X[rowIndex, i]);
            sum = NumOps.Add(sum, NumOps.Multiply(diff, diff));
        }

        return NumOps.Sqrt(sum);
    }
}
