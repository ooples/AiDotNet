using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.Encoders;

/// <summary>
/// Specifies how to handle unknown categories during transformation.
/// </summary>
public enum UnknownValueHandling
{
    /// <summary>
    /// Raise an error when an unknown category is encountered.
    /// </summary>
    Error,

    /// <summary>
    /// Use a specific value for unknown categories.
    /// </summary>
    UseEncodedValue
}

/// <summary>
/// Encodes categorical values as ordinal integers with optional custom ordering.
/// </summary>
/// <remarks>
/// <para>
/// OrdinalEncoder transforms categorical values to consecutive integers based on order.
/// Unlike LabelEncoder, it can accept custom category orderings and handle unknown values.
/// </para>
/// <para><b>For Beginners:</b> This encoder converts categories to ordered numbers:
/// - You can specify the order of categories
/// - Useful when categories have a natural ordering (e.g., low, medium, high)
///
/// Example with custom order ["small", "medium", "large"]:
/// ["large", "small", "medium", "large"] → [2, 0, 1, 2]
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class OrdinalEncoder<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly List<double[]>? _categories;
    private readonly UnknownValueHandling _handleUnknown;
    private readonly double _unknownValue;

    // Fitted parameters: mapping from value to ordinal for each column
    private List<Dictionary<double, int>>? _valueToOrdinal;
    private List<Dictionary<int, double>>? _ordinalToValue;

    /// <summary>
    /// Gets how unknown categories are handled.
    /// </summary>
    public UnknownValueHandling HandleUnknown => _handleUnknown;

    /// <summary>
    /// Gets the value used for unknown categories.
    /// </summary>
    public double UnknownValue => _unknownValue;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Creates a new instance of <see cref="OrdinalEncoder{T}"/>.
    /// </summary>
    /// <param name="categories">Optional list of category orderings for each column. If null, categories are inferred from data.</param>
    /// <param name="handleUnknown">How to handle unknown categories. Defaults to Error.</param>
    /// <param name="unknownValue">The value to use for unknown categories when handling is UseEncodedValue. Defaults to -1.</param>
    /// <param name="columnIndices">The column indices to encode, or null for all columns.</param>
    public OrdinalEncoder(
        List<double[]>? categories = null,
        UnknownValueHandling handleUnknown = UnknownValueHandling.Error,
        double unknownValue = -1,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        _categories = categories;
        _handleUnknown = handleUnknown;
        _unknownValue = unknownValue;
    }

    /// <summary>
    /// Learns the encoding mapping from the training data or uses provided categories.
    /// </summary>
    /// <param name="data">The training data matrix where each column is a feature.</param>
    protected override void FitCore(Matrix<T> data)
    {
        int numColumns = data.Columns;
        var columnsToProcess = GetColumnsToProcess(numColumns);

        _valueToOrdinal = new List<Dictionary<double, int>>();
        _ordinalToValue = new List<Dictionary<int, double>>();

        for (int col = 0; col < numColumns; col++)
        {
            if (!columnsToProcess.Contains(col))
            {
                _valueToOrdinal.Add(new Dictionary<double, int>());
                _ordinalToValue.Add(new Dictionary<int, double>());
                continue;
            }

            var column = data.GetColumn(col);
            var valueToOrdinal = new Dictionary<double, int>();
            var ordinalToValue = new Dictionary<int, double>();

            // Get category ordering
            double[] categoryOrder;
            if (_categories is not null && col < _categories.Count && _categories[col].Length > 0)
            {
                // Use provided category ordering
                categoryOrder = _categories[col];
            }
            else
            {
                // Infer from data (sorted)
                var uniqueValues = new HashSet<double>();
                for (int i = 0; i < column.Length; i++)
                {
                    uniqueValues.Add(NumOps.ToDouble(column[i]));
                }
                categoryOrder = uniqueValues.OrderBy(v => v).ToArray();
            }

            // Create mappings
            for (int i = 0; i < categoryOrder.Length; i++)
            {
                valueToOrdinal[categoryOrder[i]] = i;
                ordinalToValue[i] = categoryOrder[i];
            }

            _valueToOrdinal.Add(valueToOrdinal);
            _ordinalToValue.Add(ordinalToValue);
        }
    }

    /// <summary>
    /// Transforms the data by encoding categorical values as ordinal integers.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The encoded data with ordinal integers.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_valueToOrdinal is null)
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
                T value = data[i, j];

                if (processSet.Contains(j) && _valueToOrdinal[j].Count > 0)
                {
                    double doubleValue = NumOps.ToDouble(value);
                    if (_valueToOrdinal[j].TryGetValue(doubleValue, out int ordinal))
                    {
                        value = NumOps.FromDouble(ordinal);
                    }
                    else
                    {
                        // Unknown value
                        if (_handleUnknown == UnknownValueHandling.Error)
                        {
                            throw new ArgumentException($"Unknown category value: {doubleValue} in column {j}");
                        }
                        value = NumOps.FromDouble(_unknownValue);
                    }
                }

                result[i, j] = value;
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Reverses the ordinal encoding to get original values.
    /// </summary>
    /// <param name="data">The encoded data.</param>
    /// <returns>The original categorical values.</returns>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_ordinalToValue is null)
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
                T value = data[i, j];

                if (processSet.Contains(j) && _ordinalToValue[j].Count > 0)
                {
                    int ordinal = (int)NumOps.ToDouble(value);
                    if (_ordinalToValue[j].TryGetValue(ordinal, out double originalValue))
                    {
                        value = NumOps.FromDouble(originalValue);
                    }
                    // Unknown ordinal - keep as is
                }

                result[i, j] = value;
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    /// <param name="inputFeatureNames">The input feature names.</param>
    /// <returns>The same feature names (OrdinalEncoder doesn't change number of features).</returns>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }
}
