using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.Encoders;

/// <summary>
/// Encodes categorical features using binary representation.
/// </summary>
/// <remarks>
/// <para>
/// BinaryEncoder first ordinal encodes the categories, then converts those integers
/// to their binary representation. This creates log2(n) columns instead of n columns
/// that one-hot encoding would create.
/// </para>
/// <para><b>For Beginners:</b> If you have 8 categories, one-hot creates 8 columns.
/// Binary encoding uses only 3 columns (since 8 = 2^3):
/// - Category 0 → [0, 0, 0]
/// - Category 1 → [0, 0, 1]
/// - Category 2 → [0, 1, 0]
/// - Category 7 → [1, 1, 1]
///
/// This is useful for high-cardinality categorical features where one-hot
/// would create too many columns.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class BinaryEncoder<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly BinaryEncoderHandleUnknown _handleUnknown;

    // Fitted parameters
    private Dictionary<int, Dictionary<double, int>>? _categoryMaps;
    private Dictionary<int, int>? _bitsPerColumn;
    private int _nInputFeatures;
    private int _nOutputFeatures;
    private List<int>? _featureIndicesStart;

    /// <summary>
    /// Gets how unknown categories are handled.
    /// </summary>
    public BinaryEncoderHandleUnknown HandleUnknown => _handleUnknown;

    /// <summary>
    /// Gets the number of output features after transformation.
    /// </summary>
    public int NOutputFeatures => _nOutputFeatures;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Creates a new instance of <see cref="BinaryEncoder{T}"/>.
    /// </summary>
    /// <param name="handleUnknown">How to handle unknown categories. Defaults to AllZeros.</param>
    /// <param name="columnIndices">The column indices to encode, or null for all columns.</param>
    public BinaryEncoder(
        BinaryEncoderHandleUnknown handleUnknown = BinaryEncoderHandleUnknown.AllZeros,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        _handleUnknown = handleUnknown;
    }

    /// <summary>
    /// Learns the categories and binary encoding from the training data.
    /// </summary>
    /// <param name="data">The training data matrix.</param>
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        var columnsToProcess = GetColumnsToProcess(_nInputFeatures);

        _categoryMaps = new Dictionary<int, Dictionary<double, int>>();
        _bitsPerColumn = new Dictionary<int, int>();
        _featureIndicesStart = new List<int>();

        int currentOutputIndex = 0;

        for (int col = 0; col < _nInputFeatures; col++)
        {
            _featureIndicesStart.Add(currentOutputIndex);

            if (!columnsToProcess.Contains(col))
            {
                // Pass-through column
                _categoryMaps[col] = new Dictionary<double, int>();
                _bitsPerColumn[col] = 0;
                currentOutputIndex += 1;
                continue;
            }

            // Collect unique values
            var uniqueValues = new HashSet<double>();
            for (int i = 0; i < data.Rows; i++)
            {
                uniqueValues.Add(NumOps.ToDouble(data[i, col]));
            }

            // Sort and create ordinal mapping
            var sortedValues = uniqueValues.OrderBy(v => v).ToList();
            var categoryMap = new Dictionary<double, int>();
            for (int i = 0; i < sortedValues.Count; i++)
            {
                categoryMap[sortedValues[i]] = i + 1; // Start from 1 to leave 0 for unknown
            }

            _categoryMaps[col] = categoryMap;

            // Calculate bits needed: ceil(log2(n + 1)) where +1 is for unknown category
            int numCategories = sortedValues.Count + 1; // +1 for unknown
            int bitsNeeded = Math.Max(1, (int)Math.Ceiling(Math.Log(numCategories + 1, 2)));
            _bitsPerColumn[col] = bitsNeeded;

            currentOutputIndex += bitsNeeded;
        }

        _nOutputFeatures = currentOutputIndex;
    }

    /// <summary>
    /// Transforms the data by converting categories to binary representation.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The binary encoded data.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_categoryMaps is null || _bitsPerColumn is null || _featureIndicesStart is null)
        {
            throw new InvalidOperationException("Encoder has not been fitted.");
        }

        int numRows = data.Rows;
        var result = new T[numRows, _nOutputFeatures];
        var columnsToProcess = GetColumnsToProcess(_nInputFeatures);
        var processSet = new HashSet<int>(columnsToProcess);

        // Initialize to zero
        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < _nOutputFeatures; j++)
            {
                result[i, j] = NumOps.Zero;
            }
        }

        for (int i = 0; i < numRows; i++)
        {
            for (int col = 0; col < _nInputFeatures; col++)
            {
                int outputStart = _featureIndicesStart[col];

                if (!processSet.Contains(col) || _bitsPerColumn[col] == 0)
                {
                    // Pass-through: copy value directly
                    result[i, outputStart] = data[i, col];
                    continue;
                }

                double categoryValue = NumOps.ToDouble(data[i, col]);
                int ordinalValue;

                if (_categoryMaps[col].TryGetValue(categoryValue, out ordinalValue))
                {
                    // Known category - convert ordinal to binary
                    WriteBinaryRepresentation(result, i, outputStart, _bitsPerColumn[col], ordinalValue);
                }
                else
                {
                    // Unknown category
                    switch (_handleUnknown)
                    {
                        case BinaryEncoderHandleUnknown.AllZeros:
                            // Leave as zeros (already initialized)
                            break;
                        case BinaryEncoderHandleUnknown.Error:
                            throw new ArgumentException($"Unknown category value: {categoryValue} in column {col}");
                    }
                }
            }
        }

        return new Matrix<T>(result);
    }

    private void WriteBinaryRepresentation(T[,] result, int row, int startCol, int numBits, int value)
    {
        for (int bit = 0; bit < numBits; bit++)
        {
            int bitValue = (value >> (numBits - 1 - bit)) & 1;
            result[row, startCol + bit] = bitValue == 1 ? NumOps.One : NumOps.Zero;
        }
    }

    /// <summary>
    /// Reverses the binary encoding to get original category values.
    /// </summary>
    /// <param name="data">The binary encoded data.</param>
    /// <returns>The original categorical values.</returns>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_categoryMaps is null || _bitsPerColumn is null || _featureIndicesStart is null)
        {
            throw new InvalidOperationException("Encoder has not been fitted.");
        }

        int numRows = data.Rows;
        var result = new T[numRows, _nInputFeatures];
        var columnsToProcess = GetColumnsToProcess(_nInputFeatures);
        var processSet = new HashSet<int>(columnsToProcess);

        // Create reverse lookup
        var reverseMaps = new Dictionary<int, Dictionary<int, double>>();
        foreach (var kvp in _categoryMaps)
        {
            var reverseMap = new Dictionary<int, double>();
            foreach (var catKvp in kvp.Value)
            {
                reverseMap[catKvp.Value] = catKvp.Key;
            }
            reverseMaps[kvp.Key] = reverseMap;
        }

        for (int i = 0; i < numRows; i++)
        {
            for (int col = 0; col < _nInputFeatures; col++)
            {
                int outputStart = _featureIndicesStart[col];

                if (!processSet.Contains(col) || _bitsPerColumn[col] == 0)
                {
                    // Pass-through: copy value back
                    result[i, col] = data[i, outputStart];
                    continue;
                }

                // Read binary representation
                int ordinalValue = ReadBinaryRepresentation(data, i, outputStart, _bitsPerColumn[col]);

                // Look up original category
                if (reverseMaps[col].TryGetValue(ordinalValue, out double categoryValue))
                {
                    result[i, col] = NumOps.FromDouble(categoryValue);
                }
                else
                {
                    // Unknown - return 0
                    result[i, col] = NumOps.Zero;
                }
            }
        }

        return new Matrix<T>(result);
    }

    private int ReadBinaryRepresentation(Matrix<T> data, int row, int startCol, int numBits)
    {
        int value = 0;
        for (int bit = 0; bit < numBits; bit++)
        {
            double bitValue = NumOps.ToDouble(data[row, startCol + bit]);
            if (bitValue > 0.5)
            {
                value |= 1 << (numBits - 1 - bit);
            }
        }
        return value;
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_bitsPerColumn is null || _featureIndicesStart is null)
        {
            return Array.Empty<string>();
        }

        var columnsToProcess = GetColumnsToProcess(_nInputFeatures);
        var processSet = new HashSet<int>(columnsToProcess);

        var names = new List<string>();

        for (int col = 0; col < _nInputFeatures; col++)
        {
            string baseName = inputFeatureNames is not null && col < inputFeatureNames.Length
                ? inputFeatureNames[col]
                : $"x{col}";

            if (!processSet.Contains(col) || _bitsPerColumn[col] == 0)
            {
                // Pass-through column
                names.Add(baseName);
                continue;
            }

            // Add names for each bit
            for (int bit = 0; bit < _bitsPerColumn[col]; bit++)
            {
                names.Add($"{baseName}_bit{bit}");
            }
        }

        return names.ToArray();
    }
}

/// <summary>
/// Specifies how to handle unknown categories during transformation.
/// </summary>
public enum BinaryEncoderHandleUnknown
{
    /// <summary>
    /// Encode unknown categories as all zeros.
    /// </summary>
    AllZeros,

    /// <summary>
    /// Raise an error when an unknown category is encountered.
    /// </summary>
    Error
}
