using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.Encoders;

/// <summary>
/// Encodes categorical values as integer labels (0, 1, 2, ...).
/// </summary>
/// <remarks>
/// <para>
/// LabelEncoder transforms categorical values to consecutive integers.
/// Each unique value is assigned a unique integer starting from 0.
/// This is useful for encoding target labels or ordinal features.
/// </para>
/// <para><b>For Beginners:</b> This encoder converts categories to numbers:
/// - Each unique value gets a unique number starting from 0
/// - Values are sorted alphabetically/numerically before encoding
///
/// Example:
/// ["cat", "dog", "cat", "bird", "dog"] → [1, 2, 1, 0, 2]
/// (Mapping: bird=0, cat=1, dog=2)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class LabelEncoder<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    // Fitted parameters: mapping from value to label for each column
    private List<Dictionary<double, int>>? _valueToLabel;
    private List<Dictionary<int, double>>? _labelToValue;

    /// <summary>
    /// Gets the number of unique classes for each encoded column.
    /// </summary>
    public int[]? NClasses
    {
        get
        {
            if (_valueToLabel is null) return null;
            var result = new int[_valueToLabel.Count];
            for (int i = 0; i < _valueToLabel.Count; i++)
            {
                result[i] = _valueToLabel[i].Count;
            }
            return result;
        }
    }

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Creates a new instance of <see cref="LabelEncoder{T}"/>.
    /// </summary>
    /// <param name="columnIndices">The column indices to encode, or null for all columns.</param>
    public LabelEncoder(int[]? columnIndices = null)
        : base(columnIndices)
    {
    }

    /// <summary>
    /// Learns the encoding mapping from the training data.
    /// </summary>
    /// <param name="data">The training data matrix where each column is a feature.</param>
    protected override void FitCore(Matrix<T> data)
    {
        int numColumns = data.Columns;
        var columnsToProcess = GetColumnsToProcess(numColumns);

        _valueToLabel = new List<Dictionary<double, int>>();
        _labelToValue = new List<Dictionary<int, double>>();

        for (int col = 0; col < numColumns; col++)
        {
            if (!columnsToProcess.Contains(col))
            {
                _valueToLabel.Add(new Dictionary<double, int>());
                _labelToValue.Add(new Dictionary<int, double>());
                continue;
            }

            var column = data.GetColumn(col);

            // Collect unique values
            var uniqueValues = new HashSet<double>();
            for (int i = 0; i < column.Length; i++)
            {
                uniqueValues.Add(NumOps.ToDouble(column[i]));
            }

            // Sort and create mapping
            var sortedValues = uniqueValues.OrderBy(v => v).ToList();
            var valueToLabel = new Dictionary<double, int>();
            var labelToValue = new Dictionary<int, double>();

            for (int i = 0; i < sortedValues.Count; i++)
            {
                valueToLabel[sortedValues[i]] = i;
                labelToValue[i] = sortedValues[i];
            }

            _valueToLabel.Add(valueToLabel);
            _labelToValue.Add(labelToValue);
        }
    }

    /// <summary>
    /// Transforms the data by encoding categorical values as integers.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The encoded data with integer labels.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_valueToLabel is null)
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

                if (processSet.Contains(j) && _valueToLabel[j].Count > 0)
                {
                    double doubleValue = NumOps.ToDouble(value);
                    if (_valueToLabel[j].TryGetValue(doubleValue, out int label))
                    {
                        value = NumOps.FromDouble(label);
                    }
                    else
                    {
                        // Unknown value - use -1 or handle error
                        value = NumOps.FromDouble(-1);
                    }
                }

                result[i, j] = value;
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Reverses the label encoding to get original values.
    /// </summary>
    /// <param name="data">The encoded data.</param>
    /// <returns>The original categorical values.</returns>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_labelToValue is null)
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

                if (processSet.Contains(j) && _labelToValue[j].Count > 0)
                {
                    int label = (int)NumOps.ToDouble(value);
                    if (_labelToValue[j].TryGetValue(label, out double originalValue))
                    {
                        value = NumOps.FromDouble(originalValue);
                    }
                    // Unknown label - keep as is
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
    /// <returns>The same feature names (LabelEncoder doesn't change number of features).</returns>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }
}
