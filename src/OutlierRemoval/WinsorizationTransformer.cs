using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.OutlierRemoval;

/// <summary>
/// Implements Winsorization (clipping) to cap extreme values at specified quantile limits.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Winsorization is a technique to handle outliers without removing them.
/// Instead of deleting extreme values, it "clips" them to more reasonable values at the edges
/// of your data distribution.
/// </para>
/// <para>
/// For example, if you set lower quantile to 0.05 and upper quantile to 0.95:
/// - Any value below the 5th percentile is raised to the 5th percentile value
/// - Any value above the 95th percentile is lowered to the 95th percentile value
/// </para>
/// <para>
/// <b>When to use:</b> Winsorization is useful when:
/// - You want to reduce the impact of outliers without losing data points
/// - Your model is sensitive to extreme values (like linear regression)
/// - You believe extreme values might be measurement errors but aren't sure
/// </para>
/// <para>
/// The term "Winsorization" is named after the statistician Charles P. Winsor.
/// </para>
/// </remarks>
public class WinsorizationTransformer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly double _lowerQuantile;
    private readonly double _upperQuantile;

    private Vector<T>? _lowerBounds;
    private Vector<T>? _upperBounds;
    private bool _isFitted;

    /// <summary>
    /// Creates a new Winsorization transformer.
    /// </summary>
    /// <param name="lowerQuantile">
    /// The lower quantile for clipping (0 to 0.5). Default is 0.05 (5th percentile).
    /// Values below this percentile will be clipped to the percentile value.
    /// </param>
    /// <param name="upperQuantile">
    /// The upper quantile for clipping (0.5 to 1). Default is 0.95 (95th percentile).
    /// Values above this percentile will be clipped to the percentile value.
    /// </param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The quantile parameters control how aggressive the clipping is:
    /// - Default (0.05, 0.95): Clips the most extreme 5% on each end
    /// - Conservative (0.01, 0.99): Only clips the most extreme 1% on each end
    /// - Aggressive (0.10, 0.90): Clips the most extreme 10% on each end
    ///
    /// If you're unsure, start with the defaults (0.05, 0.95) which provide moderate protection
    /// against outliers while preserving most of your data's natural variation.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when lowerQuantile is not in [0, 0.5), upperQuantile is not in (0.5, 1],
    /// or lowerQuantile >= upperQuantile.
    /// </exception>
    public WinsorizationTransformer(double lowerQuantile = 0.05, double upperQuantile = 0.95)
    {
        if (lowerQuantile < 0 || lowerQuantile >= 0.5)
        {
            throw new ArgumentOutOfRangeException(nameof(lowerQuantile),
                "Lower quantile must be in the range [0, 0.5). " +
                "A value of 0.05 means the lowest 5% of values will be clipped.");
        }

        if (upperQuantile <= 0.5 || upperQuantile > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(upperQuantile),
                "Upper quantile must be in the range (0.5, 1]. " +
                "A value of 0.95 means the highest 5% of values will be clipped.");
        }

        if (lowerQuantile >= upperQuantile)
        {
            throw new ArgumentException(
                "Lower quantile must be less than upper quantile.",
                nameof(lowerQuantile));
        }

        _lowerQuantile = lowerQuantile;
        _upperQuantile = upperQuantile;
        _isFitted = false;
    }

    /// <summary>
    /// Gets the lower quantile used for clipping.
    /// </summary>
    public double LowerQuantile => _lowerQuantile;

    /// <summary>
    /// Gets the upper quantile used for clipping.
    /// </summary>
    public double UpperQuantile => _upperQuantile;

    /// <summary>
    /// Gets a value indicating whether the transformer has been fitted to data.
    /// </summary>
    public bool IsFitted => _isFitted;

    /// <summary>
    /// Gets the lower bounds computed during fitting (one per feature).
    /// </summary>
    /// <remarks>
    /// Returns null if the transformer has not been fitted yet.
    /// </remarks>
    public Vector<T>? LowerBounds => _lowerBounds;

    /// <summary>
    /// Gets the upper bounds computed during fitting (one per feature).
    /// </summary>
    /// <remarks>
    /// Returns null if the transformer has not been fitted yet.
    /// </remarks>
    public Vector<T>? UpperBounds => _upperBounds;

    /// <summary>
    /// Fits the transformer by computing the clipping bounds from the data.
    /// </summary>
    /// <param name="X">
    /// The data matrix where each row is a sample and each column is a feature.
    /// </param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method analyzes your data to determine where to clip.
    /// It computes the lower and upper percentile values for each feature (column).
    /// These values will be used as the clipping limits when you call Transform.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when X is null.</exception>
    /// <exception cref="ArgumentException">Thrown when X has no rows or columns.</exception>
    public void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int numFeatures = X.Columns;
        _lowerBounds = new Vector<T>(numFeatures);
        _upperBounds = new Vector<T>(numFeatures);

        for (int j = 0; j < numFeatures; j++)
        {
            var column = X.GetColumn(j);
            var (lower, upper) = ComputeQuantiles(column);
            _lowerBounds[j] = lower;
            _upperBounds[j] = upper;
        }

        _isFitted = true;
    }

    /// <summary>
    /// Transforms the data by clipping values to the fitted bounds.
    /// </summary>
    /// <param name="X">
    /// The data matrix to transform.
    /// </param>
    /// <returns>
    /// A new matrix with the same dimensions as X, where extreme values have been clipped.
    /// </returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method applies the clipping limits computed by Fit.
    /// Any value below the lower bound is raised to the lower bound.
    /// Any value above the upper bound is lowered to the upper bound.
    /// Values between the bounds are left unchanged.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown when the transformer has not been fitted.</exception>
    /// <exception cref="ArgumentNullException">Thrown when X is null.</exception>
    /// <exception cref="ArgumentException">
    /// Thrown when X has different number of features than the training data.
    /// </exception>
    public Matrix<T> Transform(Matrix<T> X)
    {
        EnsureFitted();
        ValidateInput(X);

        if (X.Columns != _lowerBounds!.Length)
        {
            throw new ArgumentException(
                $"Input has {X.Columns} features, but transformer was fitted with {_lowerBounds.Length} features.",
                nameof(X));
        }

        var result = new Matrix<T>(X.Rows, X.Columns);

        for (int i = 0; i < X.Rows; i++)
        {
            for (int j = 0; j < X.Columns; j++)
            {
                T value = X[i, j];
                T lower = _lowerBounds[j];
                T upper = _upperBounds![j];

                // Clip to [lower, upper]
                if (NumOps.LessThan(value, lower))
                {
                    result[i, j] = lower;
                }
                else if (NumOps.GreaterThan(value, upper))
                {
                    result[i, j] = upper;
                }
                else
                {
                    result[i, j] = value;
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Fits the transformer and transforms the data in one step.
    /// </summary>
    /// <param name="X">
    /// The data matrix to fit and transform.
    /// </param>
    /// <returns>
    /// A new matrix with the same dimensions as X, where extreme values have been clipped.
    /// </returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is a convenience method that combines Fit and Transform.
    /// Use this when you want to compute the clipping bounds and apply them to the same data.
    /// </para>
    /// </remarks>
    public Matrix<T> FitTransform(Matrix<T> X)
    {
        Fit(X);
        return Transform(X);
    }

    private (T lower, T upper) ComputeQuantiles(Vector<T> column)
    {
        // Sort the values
        var sorted = column.ToArray();
        Array.Sort(sorted, (a, b) => NumOps.LessThan(a, b) ? -1 : (NumOps.GreaterThan(a, b) ? 1 : 0));

        int n = sorted.Length;

        // Compute quantile indices using linear interpolation
        T lower = ComputeQuantileValue(sorted, _lowerQuantile);
        T upper = ComputeQuantileValue(sorted, _upperQuantile);

        return (lower, upper);
    }

    private T ComputeQuantileValue(T[] sorted, double quantile)
    {
        int n = sorted.Length;

        if (n == 1)
        {
            return sorted[0];
        }

        // Position in the sorted array (0-indexed, continuous)
        double position = quantile * (n - 1);
        int lowerIndex = (int)Math.Floor(position);
        int upperIndex = (int)Math.Ceiling(position);

        if (lowerIndex == upperIndex)
        {
            return sorted[lowerIndex];
        }

        // Linear interpolation
        double fraction = position - lowerIndex;
        T lowerValue = sorted[lowerIndex];
        T upperValue = sorted[upperIndex];

        T diff = NumOps.Subtract(upperValue, lowerValue);
        T interpolated = NumOps.Add(lowerValue, NumOps.Multiply(NumOps.FromDouble(fraction), diff));

        return interpolated;
    }

    private static void ValidateInput(Matrix<T> X)
    {
        if (X is null)
        {
            throw new ArgumentNullException(nameof(X));
        }

        if (X.Rows == 0 || X.Columns == 0)
        {
            throw new ArgumentException(
                $"Input matrix must have at least one row and one column. Got {X.Rows} rows and {X.Columns} columns.",
                nameof(X));
        }
    }

    private void EnsureFitted()
    {
        if (!_isFitted)
        {
            throw new InvalidOperationException(
                "This transformer has not been fitted yet. Call Fit() with training data before calling Transform().");
        }
    }
}
