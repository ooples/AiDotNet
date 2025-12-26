using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.Encoders;

/// <summary>
/// Encodes categorical features using leave-one-out target encoding.
/// </summary>
/// <remarks>
/// <para>
/// LeaveOneOutEncoder is similar to TargetEncoder but uses leave-one-out statistics
/// to prevent overfitting. For each sample, the encoding is computed using all other
/// samples in the same category, excluding the current sample.
/// </para>
/// <para>
/// This reduces the risk of target leakage during training while still capturing
/// the relationship between categories and the target variable.
/// </para>
/// <para><b>For Beginners:</b> Regular target encoding can overfit because it uses
/// the same data to encode and train. Leave-one-out encoding prevents this:
/// - When encoding row 1, it uses the average of all OTHER rows with the same category
/// - This prevents the model from "cheating" by memorizing individual samples
///
/// Example: If "Category A" has 3 samples with targets [1, 0, 1]:
/// - Row 1 gets encoded as average of [0, 1] = 0.5
/// - Row 2 gets encoded as average of [1, 1] = 1.0
/// - Row 3 gets encoded as average of [1, 0] = 0.5
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class LeaveOneOutEncoder<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly double _smoothing;
    private readonly LeaveOneOutHandleUnknown _handleUnknown;

    // Fitted parameters
    private Dictionary<int, Dictionary<double, (double Sum, int Count)>>? _categoryStats;
    private double _globalMean;
    private int _nInputFeatures;

    /// <summary>
    /// Gets the smoothing parameter.
    /// </summary>
    public double Smoothing => _smoothing;

    /// <summary>
    /// Gets how unknown categories are handled.
    /// </summary>
    public LeaveOneOutHandleUnknown HandleUnknown => _handleUnknown;

    /// <summary>
    /// Gets the global target mean.
    /// </summary>
    public double GlobalMean => _globalMean;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="LeaveOneOutEncoder{T}"/>.
    /// </summary>
    /// <param name="smoothing">Smoothing parameter for regularization. Defaults to 1.0.</param>
    /// <param name="handleUnknown">How to handle unknown categories. Defaults to UseGlobalMean.</param>
    /// <param name="columnIndices">The column indices to encode, or null for all columns.</param>
    public LeaveOneOutEncoder(
        double smoothing = 1.0,
        LeaveOneOutHandleUnknown handleUnknown = LeaveOneOutHandleUnknown.UseGlobalMean,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (smoothing < 0)
        {
            throw new ArgumentException("Smoothing must be non-negative.", nameof(smoothing));
        }

        _smoothing = smoothing;
        _handleUnknown = handleUnknown;
    }

    /// <summary>
    /// Fits the encoder (requires target via specialized Fit method).
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "LeaveOneOutEncoder requires target values for fitting. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <summary>
    /// Fits the encoder by computing category statistics.
    /// </summary>
    /// <param name="data">The feature matrix to fit.</param>
    /// <param name="target">The target values.</param>
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
        {
            throw new ArgumentException("Target length must match the number of rows in data.");
        }

        _nInputFeatures = data.Columns;
        var columnsToProcess = GetColumnsToProcess(_nInputFeatures);

        // Compute global mean
        double targetSum = 0;
        for (int i = 0; i < target.Length; i++)
        {
            targetSum += NumOps.ToDouble(target[i]);
        }
        _globalMean = targetSum / target.Length;

        // Compute category statistics
        _categoryStats = new Dictionary<int, Dictionary<double, (double Sum, int Count)>>();

        foreach (int col in columnsToProcess)
        {
            var stats = new Dictionary<double, (double Sum, int Count)>();

            for (int i = 0; i < data.Rows; i++)
            {
                double categoryValue = NumOps.ToDouble(data[i, col]);
                double targetValue = NumOps.ToDouble(target[i]);

                if (!stats.TryGetValue(categoryValue, out var existing))
                {
                    existing = (0, 0);
                }

                stats[categoryValue] = (existing.Sum + targetValue, existing.Count + 1);
            }

            _categoryStats[col] = stats;
        }

        IsFitted = true;
    }

    /// <summary>
    /// Fits and transforms using leave-one-out encoding.
    /// </summary>
    /// <param name="data">The feature matrix.</param>
    /// <param name="target">The target values.</param>
    /// <returns>The encoded data with leave-one-out statistics.</returns>
    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return TransformWithTarget(data, target);
    }

    /// <summary>
    /// Transforms the data using leave-one-out encoding (requires target for training data).
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <param name="target">The target values (for leave-one-out calculation).</param>
    /// <returns>The encoded data.</returns>
    public Matrix<T> TransformWithTarget(Matrix<T> data, Vector<T> target)
    {
        if (_categoryStats is null)
        {
            throw new InvalidOperationException("Encoder has not been fitted.");
        }

        if (data.Rows != target.Length)
        {
            throw new ArgumentException("Target length must match data rows.");
        }

        int numRows = data.Rows;
        int numColumns = data.Columns;
        var result = new T[numRows, numColumns];
        var columnsToProcess = GetColumnsToProcess(numColumns);
        var processSet = new HashSet<int>(columnsToProcess);

        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numColumns; j++)
            {
                if (!processSet.Contains(j))
                {
                    result[i, j] = data[i, j];
                    continue;
                }

                double categoryValue = NumOps.ToDouble(data[i, j]);
                double targetValue = NumOps.ToDouble(target[i]);

                if (_categoryStats.TryGetValue(j, out var stats) &&
                    stats.TryGetValue(categoryValue, out var catStats))
                {
                    // Leave-one-out: exclude current sample
                    double looSum = catStats.Sum - targetValue;
                    int looCount = catStats.Count - 1;

                    double encodedValue;
                    if (looCount > 0)
                    {
                        double looMean = looSum / looCount;
                        // Apply smoothing
                        encodedValue = (looCount * looMean + _smoothing * _globalMean) / (looCount + _smoothing);
                    }
                    else
                    {
                        // Only sample in category, use global mean
                        encodedValue = _globalMean;
                    }

                    result[i, j] = NumOps.FromDouble(encodedValue);
                }
                else
                {
                    result[i, j] = NumOps.FromDouble(_globalMean);
                }
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Transforms the data using standard target encoding (for test/inference data).
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The encoded data.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_categoryStats is null)
        {
            throw new InvalidOperationException("Encoder has not been fitted.");
        }

        int numRows = data.Rows;
        int numColumns = data.Columns;
        var result = new T[numRows, numColumns];
        var columnsToProcess = GetColumnsToProcess(numColumns);
        var processSet = new HashSet<int>(columnsToProcess);

        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numColumns; j++)
            {
                if (!processSet.Contains(j))
                {
                    result[i, j] = data[i, j];
                    continue;
                }

                double categoryValue = NumOps.ToDouble(data[i, j]);

                if (_categoryStats.TryGetValue(j, out var stats) &&
                    stats.TryGetValue(categoryValue, out var catStats))
                {
                    // Use full category statistics for test data
                    double catMean = catStats.Sum / catStats.Count;
                    double encodedValue = (catStats.Count * catMean + _smoothing * _globalMean) /
                                         (catStats.Count + _smoothing);
                    result[i, j] = NumOps.FromDouble(encodedValue);
                }
                else
                {
                    // Unknown category
                    switch (_handleUnknown)
                    {
                        case LeaveOneOutHandleUnknown.UseGlobalMean:
                            result[i, j] = NumOps.FromDouble(_globalMean);
                            break;
                        case LeaveOneOutHandleUnknown.Error:
                            throw new ArgumentException($"Unknown category value: {categoryValue} in column {j}");
                        default:
                            result[i, j] = NumOps.FromDouble(_globalMean);
                            break;
                    }
                }
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("LeaveOneOutEncoder does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }
}

/// <summary>
/// Specifies how to handle unknown categories during transformation.
/// </summary>
public enum LeaveOneOutHandleUnknown
{
    /// <summary>
    /// Use the global target mean for unknown categories.
    /// </summary>
    UseGlobalMean,

    /// <summary>
    /// Raise an error when an unknown category is encountered.
    /// </summary>
    Error
}
