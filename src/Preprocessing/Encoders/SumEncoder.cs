using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.Encoders;

/// <summary>
/// Encodes categorical features using sum (deviation) coding.
/// </summary>
/// <remarks>
/// <para>
/// SumEncoder (also known as deviation coding or effect coding) compares each level
/// to the grand mean of the dependent variable. The last category serves as the
/// reference and is encoded as -1 in all columns.
/// </para>
/// <para>
/// For k categories, creates k-1 columns. Unlike one-hot encoding:
/// - Each category (except reference) gets 1 in its column, 0 elsewhere
/// - The reference category gets -1 in ALL columns
/// - The sum of coefficients equals 0
/// </para>
/// <para><b>For Beginners:</b> Sum coding is useful in ANOVA-style analysis:
/// - Shows how each category differs from the overall average
/// - The reference category's effect is the negative sum of others
/// - Coefficients are easier to interpret as deviations from mean
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class SumEncoder<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly SumEncoderHandleUnknown _handleUnknown;

    // Fitted parameters
    private Dictionary<int, List<double>>? _categories;
    private int _nInputFeatures;
    private int _nOutputFeatures;

    /// <summary>
    /// Gets how unknown categories are handled.
    /// </summary>
    public SumEncoderHandleUnknown HandleUnknown => _handleUnknown;

    /// <summary>
    /// Gets the number of output features.
    /// </summary>
    public int NOutputFeatures => _nOutputFeatures;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="SumEncoder{T}"/>.
    /// </summary>
    /// <param name="handleUnknown">How to handle unknown categories. Defaults to UseZeros.</param>
    /// <param name="columnIndices">The column indices to encode, or null for all columns.</param>
    public SumEncoder(
        SumEncoderHandleUnknown handleUnknown = SumEncoderHandleUnknown.UseZeros,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        _handleUnknown = handleUnknown;
    }

    /// <summary>
    /// Fits the encoder by learning categories.
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        var columnsToProcess = GetColumnsToProcess(_nInputFeatures);

        _categories = new Dictionary<int, List<double>>();

        foreach (int col in columnsToProcess)
        {
            var uniqueValues = new HashSet<double>();

            for (int i = 0; i < data.Rows; i++)
            {
                double value = NumOps.ToDouble(data[i, col]);
                uniqueValues.Add(value);
            }

            _categories[col] = uniqueValues.OrderBy(v => v).ToList();
        }

        // Calculate total output features
        _nOutputFeatures = 0;
        var processSet = new HashSet<int>(columnsToProcess);

        for (int col = 0; col < _nInputFeatures; col++)
        {
            if (processSet.Contains(col))
            {
                int nCategories = _categories[col].Count;
                // k-1 columns for k categories (sum coding)
                _nOutputFeatures += Math.Max(1, nCategories - 1);
            }
            else
            {
                _nOutputFeatures += 1;
            }
        }

        IsFitted = true;
    }

    /// <summary>
    /// Transforms the data using sum coding.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_categories is null)
        {
            throw new InvalidOperationException("SumEncoder has not been fitted.");
        }

        int numRows = data.Rows;
        var result = new T[numRows, _nOutputFeatures];
        var columnsToProcess = GetColumnsToProcess(_nInputFeatures);
        var processSet = new HashSet<int>(columnsToProcess);

        for (int i = 0; i < numRows; i++)
        {
            int outCol = 0;

            for (int j = 0; j < _nInputFeatures; j++)
            {
                if (!processSet.Contains(j))
                {
                    result[i, outCol++] = data[i, j];
                    continue;
                }

                double categoryValue = NumOps.ToDouble(data[i, j]);
                var categories = _categories[j];
                int nCategories = categories.Count;
                int nCols = Math.Max(1, nCategories - 1);

                int categoryIndex = categories.IndexOf(categoryValue);

                if (categoryIndex < 0)
                {
                    // Unknown category
                    switch (_handleUnknown)
                    {
                        case SumEncoderHandleUnknown.UseZeros:
                            for (int c = 0; c < nCols; c++)
                            {
                                result[i, outCol++] = NumOps.FromDouble(0);
                            }
                            break;
                        case SumEncoderHandleUnknown.Error:
                            throw new ArgumentException($"Unknown category value: {categoryValue} in column {j}");
                        default:
                            for (int c = 0; c < nCols; c++)
                            {
                                result[i, outCol++] = NumOps.FromDouble(0);
                            }
                            break;
                    }
                }
                else if (categoryIndex == nCategories - 1)
                {
                    // Reference category (last one) - encode as -1 in all columns
                    for (int c = 0; c < nCols; c++)
                    {
                        result[i, outCol++] = NumOps.FromDouble(-1);
                    }
                }
                else
                {
                    // Non-reference category - encode as 1 in its column, 0 elsewhere
                    for (int c = 0; c < nCols; c++)
                    {
                        double value = (c == categoryIndex) ? 1 : 0;
                        result[i, outCol++] = NumOps.FromDouble(value);
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
        throw new NotSupportedException("SumEncoder does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_categories is null)
        {
            return Array.Empty<string>();
        }

        var names = new List<string>();
        var columnsToProcess = GetColumnsToProcess(_nInputFeatures);
        var processSet = new HashSet<int>(columnsToProcess);

        for (int col = 0; col < _nInputFeatures; col++)
        {
            string baseName = inputFeatureNames is not null && col < inputFeatureNames.Length
                ? inputFeatureNames[col]
                : $"x{col}";

            if (processSet.Contains(col))
            {
                var categories = _categories[col];
                int nCols = Math.Max(1, categories.Count - 1);

                for (int c = 0; c < nCols; c++)
                {
                    names.Add($"{baseName}_sum_{c}");
                }
            }
            else
            {
                names.Add(baseName);
            }
        }

        return names.ToArray();
    }
}

/// <summary>
/// Specifies how to handle unknown categories in SumEncoder.
/// </summary>
public enum SumEncoderHandleUnknown
{
    /// <summary>
    /// Encode unknown categories as all zeros.
    /// </summary>
    UseZeros,

    /// <summary>
    /// Raise an error when an unknown category is encountered.
    /// </summary>
    Error
}
