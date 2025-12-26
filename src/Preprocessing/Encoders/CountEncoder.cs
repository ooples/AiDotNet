using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.Encoders;

/// <summary>
/// Encodes categorical features using frequency counts.
/// </summary>
/// <remarks>
/// <para>
/// CountEncoder replaces each category with its frequency count (number of occurrences)
/// in the training data. This creates a continuous feature that captures category popularity.
/// </para>
/// <para>
/// Options include normalizing counts to probabilities (0-1 range) or log-transforming
/// the counts to handle highly skewed distributions.
/// </para>
/// <para><b>For Beginners:</b> Instead of creating multiple columns, frequency encoding
/// replaces each category with how often it appears:
/// - Category "common" appearing 1000 times → 1000 (or 0.5 if normalized)
/// - Category "rare" appearing 10 times → 10 (or 0.005 if normalized)
///
/// This is useful when the popularity of a category is predictive of the target.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class CountEncoder<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly bool _normalize;
    private readonly bool _logTransform;
    private readonly CountEncoderHandleUnknown _handleUnknown;
    private readonly double _unknownValue;

    // Fitted parameters
    private Dictionary<int, Dictionary<double, double>>? _countMaps;
    private int _nInputFeatures;
    private int _nTrainingSamples;

    /// <summary>
    /// Gets whether counts are normalized to probabilities.
    /// </summary>
    public bool Normalize => _normalize;

    /// <summary>
    /// Gets whether counts are log-transformed.
    /// </summary>
    public bool LogTransform => _logTransform;

    /// <summary>
    /// Gets how unknown categories are handled.
    /// </summary>
    public CountEncoderHandleUnknown HandleUnknown => _handleUnknown;

    /// <summary>
    /// Gets the count maps for each column.
    /// </summary>
    public Dictionary<int, Dictionary<double, double>>? CountMaps => _countMaps;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="CountEncoder{T}"/>.
    /// </summary>
    /// <param name="normalize">If true, normalize counts to probabilities (0-1). Defaults to false.</param>
    /// <param name="logTransform">If true, apply log1p transform to counts. Defaults to false.</param>
    /// <param name="handleUnknown">How to handle unknown categories. Defaults to UseValue.</param>
    /// <param name="unknownValue">Value to use for unknown categories when HandleUnknown is UseValue. Defaults to 1.</param>
    /// <param name="columnIndices">The column indices to encode, or null for all columns.</param>
    public CountEncoder(
        bool normalize = false,
        bool logTransform = false,
        CountEncoderHandleUnknown handleUnknown = CountEncoderHandleUnknown.UseValue,
        double unknownValue = 1.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (unknownValue < 0)
        {
            throw new ArgumentException("Unknown value must be non-negative.", nameof(unknownValue));
        }

        _normalize = normalize;
        _logTransform = logTransform;
        _handleUnknown = handleUnknown;
        _unknownValue = unknownValue;
    }

    /// <summary>
    /// Learns the frequency counts from the training data.
    /// </summary>
    /// <param name="data">The training data matrix.</param>
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        _nTrainingSamples = data.Rows;
        var columnsToProcess = GetColumnsToProcess(_nInputFeatures);

        _countMaps = new Dictionary<int, Dictionary<double, double>>();

        foreach (int col in columnsToProcess)
        {
            // Count occurrences of each category
            var counts = new Dictionary<double, int>();

            for (int i = 0; i < data.Rows; i++)
            {
                double categoryValue = NumOps.ToDouble(data[i, col]);

                if (!counts.TryGetValue(categoryValue, out int count))
                {
                    count = 0;
                }
                counts[categoryValue] = count + 1;
            }

            // Convert to final encoding values
            var countMap = new Dictionary<double, double>();

            foreach (var kvp in counts)
            {
                double categoryValue = kvp.Key;
                double count = kvp.Value;

                // Apply normalization if requested
                if (_normalize)
                {
                    count /= _nTrainingSamples;
                }

                // Apply log transform if requested
                if (_logTransform)
                {
                    count = Math.Log(1 + count); // log1p to handle zero
                }

                countMap[categoryValue] = count;
            }

            _countMaps[col] = countMap;
        }
    }

    /// <summary>
    /// Transforms the data by replacing categories with their frequency counts.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The frequency encoded data.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_countMaps is null)
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

                if (_countMaps.TryGetValue(j, out var countMap) &&
                    countMap.TryGetValue(categoryValue, out encodedValue))
                {
                    result[i, j] = NumOps.FromDouble(encodedValue);
                }
                else
                {
                    // Unknown category
                    switch (_handleUnknown)
                    {
                        case CountEncoderHandleUnknown.UseValue:
                            double unknownVal = _unknownValue;
                            if (_normalize)
                            {
                                unknownVal = _unknownValue / _nTrainingSamples;
                            }
                            if (_logTransform)
                            {
                                unknownVal = Math.Log(1 + unknownVal);
                            }
                            result[i, j] = NumOps.FromDouble(unknownVal);
                            break;
                        case CountEncoderHandleUnknown.Error:
                            throw new ArgumentException($"Unknown category value: {categoryValue} in column {j}");
                    }
                }
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Inverse transformation is not supported for frequency encoding.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("CountEncoder does not support inverse transformation.");
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
public enum CountEncoderHandleUnknown
{
    /// <summary>
    /// Use a specified value for unknown categories.
    /// </summary>
    UseValue,

    /// <summary>
    /// Raise an error when an unknown category is encountered.
    /// </summary>
    Error
}
