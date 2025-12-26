using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.Encoders;

/// <summary>
/// Encodes categorical features using target mean encoding.
/// </summary>
/// <remarks>
/// <para>
/// TargetEncoder replaces each category with the mean of the target variable for that category.
/// This creates a continuous feature that captures the relationship between the category and target.
/// </para>
/// <para>
/// To prevent overfitting, especially with rare categories, smoothing is applied:
/// encoding = (count * category_mean + smoothing * global_mean) / (count + smoothing)
/// </para>
/// <para><b>For Beginners:</b> Instead of one-hot encoding (many columns), target encoding
/// creates a single column per feature containing the average target value for each category:
/// - Category "A" with average target 0.8 becomes 0.8
/// - Category "B" with average target 0.3 becomes 0.3
///
/// This is especially useful for high-cardinality features where one-hot would create
/// too many columns.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class TargetEncoder<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly double _smoothing;
    private readonly double _minSamplesLeaf;
    private readonly TargetEncoderHandleUnknown _handleUnknown;

    // Fitted parameters
    private Dictionary<int, Dictionary<double, double>>? _encodingMaps;
    private double[]? _globalMeans;
    private int _nInputFeatures;

    /// <summary>
    /// Gets the smoothing parameter used during encoding.
    /// </summary>
    public double Smoothing => _smoothing;

    /// <summary>
    /// Gets how unknown categories are handled during transform.
    /// </summary>
    public TargetEncoderHandleUnknown HandleUnknown => _handleUnknown;

    /// <summary>
    /// Gets the encoding maps for each column.
    /// </summary>
    public Dictionary<int, Dictionary<double, double>>? EncodingMaps => _encodingMaps;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="TargetEncoder{T}"/>.
    /// </summary>
    /// <param name="smoothing">Smoothing parameter. Higher values give more weight to global mean. Defaults to 1.0.</param>
    /// <param name="minSamplesLeaf">Minimum samples to compute category mean. Categories below this use global mean. Defaults to 1.</param>
    /// <param name="handleUnknown">How to handle unknown categories during transform. Defaults to UseGlobalMean.</param>
    /// <param name="columnIndices">The column indices to encode, or null for all columns.</param>
    public TargetEncoder(
        double smoothing = 1.0,
        double minSamplesLeaf = 1,
        TargetEncoderHandleUnknown handleUnknown = TargetEncoderHandleUnknown.UseGlobalMean,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (smoothing < 0)
        {
            throw new ArgumentException("Smoothing must be non-negative.", nameof(smoothing));
        }

        if (minSamplesLeaf < 1)
        {
            throw new ArgumentException("Minimum samples per leaf must be at least 1.", nameof(minSamplesLeaf));
        }

        _smoothing = smoothing;
        _minSamplesLeaf = minSamplesLeaf;
        _handleUnknown = handleUnknown;
    }

    /// <summary>
    /// Fits the encoder by learning the target means for each category.
    /// </summary>
    /// <param name="data">The feature matrix to fit.</param>
    /// <param name="target">The target values used to compute means.</param>
    /// <exception cref="ArgumentException">If target length doesn't match data rows.</exception>
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
        {
            throw new ArgumentException("Target length must match the number of rows in data.");
        }

        _nInputFeatures = data.Columns;
        var columnsToProcess = GetColumnsToProcess(_nInputFeatures);

        _encodingMaps = new Dictionary<int, Dictionary<double, double>>();
        _globalMeans = new double[_nInputFeatures];

        // Calculate global means for each column's targets
        double globalTargetMean = 0;
        for (int i = 0; i < target.Length; i++)
        {
            globalTargetMean += NumOps.ToDouble(target[i]);
        }
        globalTargetMean /= target.Length;

        foreach (int col in columnsToProcess)
        {
            _globalMeans[col] = globalTargetMean;

            // Group by category value and compute statistics
            var categoryStats = new Dictionary<double, (double Sum, int Count)>();

            for (int i = 0; i < data.Rows; i++)
            {
                double categoryValue = NumOps.ToDouble(data[i, col]);
                double targetValue = NumOps.ToDouble(target[i]);

                if (!categoryStats.TryGetValue(categoryValue, out var stats))
                {
                    stats = (0, 0);
                }

                categoryStats[categoryValue] = (stats.Sum + targetValue, stats.Count + 1);
            }

            // Compute smoothed means for each category
            var encodingMap = new Dictionary<double, double>();

            foreach (var kvp in categoryStats)
            {
                double categoryValue = kvp.Key;
                double sum = kvp.Value.Sum;
                int count = kvp.Value.Count;

                double categoryMean;
                if (count >= _minSamplesLeaf)
                {
                    categoryMean = sum / count;
                    // Apply smoothing: (count * category_mean + smoothing * global_mean) / (count + smoothing)
                    double smoothedMean = (count * categoryMean + _smoothing * globalTargetMean) / (count + _smoothing);
                    encodingMap[categoryValue] = smoothedMean;
                }
                else
                {
                    // Use global mean for rare categories
                    encodingMap[categoryValue] = globalTargetMean;
                }
            }

            _encodingMaps[col] = encodingMap;
        }

        IsFitted = true;
    }

    /// <summary>
    /// Fits the encoder using the base Fit method (requires target via FitWithTarget).
    /// </summary>
    /// <param name="data">The feature matrix.</param>
    /// <exception cref="InvalidOperationException">Always thrown. Use Fit(Matrix, Vector) instead.</exception>
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "TargetEncoder requires target values for fitting. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <summary>
    /// Fits the encoder and transforms the data in one step.
    /// </summary>
    /// <param name="data">The feature matrix to fit and transform.</param>
    /// <param name="target">The target values used to compute means.</param>
    /// <returns>The encoded data.</returns>
    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    /// <summary>
    /// Transforms the data by replacing categories with their target means.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The encoded data.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_encodingMaps is null || _globalMeans is null)
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
                    // Pass-through: copy value directly
                    result[i, j] = data[i, j];
                    continue;
                }

                double categoryValue = NumOps.ToDouble(data[i, j]);
                double encodedValue;

                if (_encodingMaps.TryGetValue(j, out var encodingMap) &&
                    encodingMap.TryGetValue(categoryValue, out encodedValue))
                {
                    result[i, j] = NumOps.FromDouble(encodedValue);
                }
                else
                {
                    // Unknown category
                    switch (_handleUnknown)
                    {
                        case TargetEncoderHandleUnknown.UseGlobalMean:
                            result[i, j] = NumOps.FromDouble(_globalMeans[j]);
                            break;
                        case TargetEncoderHandleUnknown.Error:
                            throw new ArgumentException($"Unknown category value: {categoryValue} in column {j}");
                        default:
                            result[i, j] = NumOps.FromDouble(_globalMeans[j]);
                            break;
                    }
                }
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Inverse transformation is not supported for target encoding.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("TargetEncoder does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (inputFeatureNames is null)
        {
            var names = new string[_nInputFeatures];
            for (int i = 0; i < _nInputFeatures; i++)
            {
                names[i] = $"x{i}";
            }
            return names;
        }

        return inputFeatureNames;
    }
}

/// <summary>
/// Specifies how to handle unknown categories during transformation.
/// </summary>
public enum TargetEncoderHandleUnknown
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
