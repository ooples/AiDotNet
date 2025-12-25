using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.Encoders;

/// <summary>
/// Specifies how to handle unknown categories during transformation.
/// </summary>
public enum OneHotUnknownHandling
{
    /// <summary>
    /// Raise an error when an unknown category is encountered.
    /// </summary>
    Error,

    /// <summary>
    /// Ignore unknown categories (all zeros in the one-hot encoding).
    /// </summary>
    Ignore
}

/// <summary>
/// Encodes categorical values as one-hot (binary) vectors.
/// </summary>
/// <remarks>
/// <para>
/// OneHotEncoder transforms categorical values into binary indicator columns.
/// Each unique category value becomes a separate column with 1s and 0s indicating presence.
/// This encoding is required for many machine learning algorithms that cannot work directly with categories.
/// </para>
/// <para><b>For Beginners:</b> This encoder converts categories into binary columns:
/// - Each unique value gets its own column
/// - A 1 indicates the category is present, 0 means it's not
///
/// Example for colors [red, green, blue, red]:
/// Becomes:
/// [1, 0, 0]  (red)
/// [0, 1, 0]  (green)
/// [0, 0, 1]  (blue)
/// [1, 0, 0]  (red)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class OneHotEncoder<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly bool _dropFirst;
    private readonly OneHotUnknownHandling _handleUnknown;

    // Fitted parameters
    private List<double[]>? _categories;
    private int _nInputFeatures;
    private int _nOutputFeatures;
    private List<int>? _featureIndicesStart;

    /// <summary>
    /// Gets whether the first category is dropped (to avoid multicollinearity).
    /// </summary>
    public bool DropFirst => _dropFirst;

    /// <summary>
    /// Gets how unknown categories are handled.
    /// </summary>
    public OneHotUnknownHandling HandleUnknown => _handleUnknown;

    /// <summary>
    /// Gets the number of output features after transformation.
    /// </summary>
    public int NOutputFeatures => _nOutputFeatures;

    /// <summary>
    /// Gets the categories for each encoded column.
    /// </summary>
    public List<double[]>? Categories => _categories;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Creates a new instance of <see cref="OneHotEncoder{T}"/>.
    /// </summary>
    /// <param name="dropFirst">If true, drops the first category to avoid multicollinearity. Defaults to false.</param>
    /// <param name="handleUnknown">How to handle unknown categories. Defaults to Error.</param>
    /// <param name="columnIndices">The column indices to encode, or null for all columns.</param>
    public OneHotEncoder(
        bool dropFirst = false,
        OneHotUnknownHandling handleUnknown = OneHotUnknownHandling.Error,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        _dropFirst = dropFirst;
        _handleUnknown = handleUnknown;
    }

    /// <summary>
    /// Learns the categories from the training data.
    /// </summary>
    /// <param name="data">The training data matrix where each column is a feature.</param>
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        var columnsToProcess = GetColumnsToProcess(_nInputFeatures);

        _categories = new List<double[]>();
        _featureIndicesStart = new List<int>();

        int currentOutputIndex = 0;

        for (int col = 0; col < _nInputFeatures; col++)
        {
            _featureIndicesStart.Add(currentOutputIndex);

            if (!columnsToProcess.Contains(col))
            {
                // Pass-through column (not encoded)
                _categories.Add(Array.Empty<double>());
                currentOutputIndex += 1;
                continue;
            }

            var column = data.GetColumn(col);

            // Collect unique values
            var uniqueValues = new HashSet<double>();
            for (int i = 0; i < column.Length; i++)
            {
                uniqueValues.Add(NumOps.ToDouble(column[i]));
            }

            // Sort and store categories
            var sortedCategories = uniqueValues.OrderBy(v => v).ToArray();
            _categories.Add(sortedCategories);

            // Calculate output columns for this feature
            int nOutputCols = _dropFirst ? sortedCategories.Length - 1 : sortedCategories.Length;
            nOutputCols = Math.Max(1, nOutputCols); // At least 1 column
            currentOutputIndex += nOutputCols;
        }

        _nOutputFeatures = currentOutputIndex;
    }

    /// <summary>
    /// Transforms the data by applying one-hot encoding.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The one-hot encoded data.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_categories is null || _featureIndicesStart is null)
        {
            throw new InvalidOperationException("Encoder has not been fitted.");
        }

        int numRows = data.Rows;
        var result = new T[numRows, _nOutputFeatures];
        var columnsToProcess = GetColumnsToProcess(_nInputFeatures);
        var processSet = new HashSet<int>(columnsToProcess);

        // Initialize all to zero
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
                T value = data[i, col];
                int outputStart = _featureIndicesStart[col];

                if (!processSet.Contains(col) || _categories[col].Length == 0)
                {
                    // Pass-through: copy value directly
                    result[i, outputStart] = value;
                    continue;
                }

                // Find category index
                double doubleValue = NumOps.ToDouble(value);
                int categoryIndex = Array.IndexOf(_categories[col], doubleValue);

                if (categoryIndex < 0)
                {
                    // Unknown category
                    if (_handleUnknown == OneHotUnknownHandling.Error)
                    {
                        throw new ArgumentException($"Unknown category value: {doubleValue} in column {col}");
                    }
                    // Ignore: leave all zeros
                    continue;
                }

                // Apply one-hot encoding
                if (_dropFirst)
                {
                    // Skip the first category (becomes the reference)
                    if (categoryIndex > 0)
                    {
                        result[i, outputStart + categoryIndex - 1] = NumOps.One;
                    }
                }
                else
                {
                    result[i, outputStart + categoryIndex] = NumOps.One;
                }
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Reverses the one-hot encoding to get original category values.
    /// </summary>
    /// <param name="data">The one-hot encoded data.</param>
    /// <returns>The original categorical values.</returns>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_categories is null || _featureIndicesStart is null)
        {
            throw new InvalidOperationException("Encoder has not been fitted.");
        }

        int numRows = data.Rows;
        var result = new T[numRows, _nInputFeatures];
        var columnsToProcess = GetColumnsToProcess(_nInputFeatures);
        var processSet = new HashSet<int>(columnsToProcess);

        for (int i = 0; i < numRows; i++)
        {
            for (int col = 0; col < _nInputFeatures; col++)
            {
                int outputStart = _featureIndicesStart[col];

                if (!processSet.Contains(col) || _categories[col].Length == 0)
                {
                    // Pass-through: copy value back
                    result[i, col] = data[i, outputStart];
                    continue;
                }

                // Find which category has the 1
                int nCategories = _categories[col].Length;
                int foundCategory = -1;

                if (_dropFirst)
                {
                    // Check if any of the (n-1) columns has a 1
                    bool allZeros = true;
                    for (int c = 0; c < nCategories - 1; c++)
                    {
                        if (NumOps.Compare(data[i, outputStart + c], NumOps.FromDouble(0.5)) > 0)
                        {
                            foundCategory = c + 1; // +1 because first category is dropped
                            allZeros = false;
                            break;
                        }
                    }
                    if (allZeros)
                    {
                        foundCategory = 0; // First (dropped) category
                    }
                }
                else
                {
                    for (int c = 0; c < nCategories; c++)
                    {
                        if (NumOps.Compare(data[i, outputStart + c], NumOps.FromDouble(0.5)) > 0)
                        {
                            foundCategory = c;
                            break;
                        }
                    }
                }

                if (foundCategory >= 0 && foundCategory < _categories[col].Length)
                {
                    result[i, col] = NumOps.FromDouble(_categories[col][foundCategory]);
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

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    /// <param name="inputFeatureNames">The input feature names.</param>
    /// <returns>The output feature names with category suffixes.</returns>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_categories is null || _featureIndicesStart is null)
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

            if (!processSet.Contains(col) || _categories[col].Length == 0)
            {
                // Pass-through column
                names.Add(baseName);
                continue;
            }

            // Add names for each category
            int startIdx = _dropFirst ? 1 : 0;
            for (int c = startIdx; c < _categories[col].Length; c++)
            {
                names.Add($"{baseName}_{_categories[col][c]}");
            }
        }

        return names.ToArray();
    }
}
