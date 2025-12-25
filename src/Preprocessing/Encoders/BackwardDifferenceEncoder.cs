using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.Encoders;

/// <summary>
/// Encodes categorical features using backward difference coding.
/// </summary>
/// <remarks>
/// <para>
/// BackwardDifferenceEncoder compares each level of a categorical variable to
/// the previous level. This is useful for ordinal variables where the
/// difference between adjacent levels is meaningful.
/// </para>
/// <para>
/// For k categories, creates k-1 columns. Each column represents the difference
/// between level n and level n-1.
/// </para>
/// <para><b>For Beginners:</b> Backward difference coding is useful for ordinal data:
/// - Education: High School vs Some College vs Bachelor's vs Master's
/// - Each coefficient shows the "step up" from the previous level
/// - Good when you expect gradual progression through levels
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class BackwardDifferenceEncoder<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly BackwardDifferenceHandleUnknown _handleUnknown;

    // Fitted parameters
    private Dictionary<int, List<double>>? _categories;
    private Dictionary<int, double[,]>? _contrastMatrices;
    private int _nInputFeatures;
    private int _nOutputFeatures;

    /// <summary>
    /// Gets how unknown categories are handled.
    /// </summary>
    public BackwardDifferenceHandleUnknown HandleUnknown => _handleUnknown;

    /// <summary>
    /// Gets the number of output features.
    /// </summary>
    public int NOutputFeatures => _nOutputFeatures;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="BackwardDifferenceEncoder{T}"/>.
    /// </summary>
    /// <param name="handleUnknown">How to handle unknown categories. Defaults to UseZeros.</param>
    /// <param name="columnIndices">The column indices to encode, or null for all columns.</param>
    public BackwardDifferenceEncoder(
        BackwardDifferenceHandleUnknown handleUnknown = BackwardDifferenceHandleUnknown.UseZeros,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        _handleUnknown = handleUnknown;
    }

    /// <summary>
    /// Fits the encoder by learning categories and building contrast matrices.
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        var columnsToProcess = GetColumnsToProcess(_nInputFeatures);

        _categories = new Dictionary<int, List<double>>();
        _contrastMatrices = new Dictionary<int, double[,]>();

        foreach (int col in columnsToProcess)
        {
            var uniqueValues = new HashSet<double>();

            for (int i = 0; i < data.Rows; i++)
            {
                double value = NumOps.ToDouble(data[i, col]);
                uniqueValues.Add(value);
            }

            var sortedCategories = uniqueValues.OrderBy(v => v).ToList();
            _categories[col] = sortedCategories;

            // Build backward difference contrast matrix
            int k = sortedCategories.Count;
            if (k > 1)
            {
                var contrastMatrix = BuildBackwardDifferenceMatrix(k);
                _contrastMatrices[col] = contrastMatrix;
            }
        }

        // Calculate total output features
        _nOutputFeatures = 0;
        var processSet = new HashSet<int>(columnsToProcess);

        for (int col = 0; col < _nInputFeatures; col++)
        {
            if (processSet.Contains(col))
            {
                int nCategories = _categories[col].Count;
                _nOutputFeatures += Math.Max(1, nCategories - 1);
            }
            else
            {
                _nOutputFeatures += 1;
            }
        }

        IsFitted = true;
    }

    private double[,] BuildBackwardDifferenceMatrix(int k)
    {
        // Backward difference contrast matrix
        // Each row represents a category level
        // Each column represents a contrast (level i vs level i-1)
        var matrix = new double[k, k - 1];

        for (int row = 0; row < k; row++)
        {
            for (int col = 0; col < k - 1; col++)
            {
                // The contrast for column c compares level c+1 to level c
                // Levels <= col get -(k-col-1)/k
                // Levels > col get (col+1)/k
                if (row <= col)
                {
                    matrix[row, col] = -(double)(k - col - 1) / k;
                }
                else
                {
                    matrix[row, col] = (double)(col + 1) / k;
                }
            }
        }

        return matrix;
    }

    /// <summary>
    /// Transforms the data using backward difference coding.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_categories is null || _contrastMatrices is null)
        {
            throw new InvalidOperationException("BackwardDifferenceEncoder has not been fitted.");
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
                        case BackwardDifferenceHandleUnknown.UseZeros:
                            for (int c = 0; c < nCols; c++)
                            {
                                result[i, outCol++] = NumOps.FromDouble(0);
                            }
                            break;
                        case BackwardDifferenceHandleUnknown.Error:
                            throw new ArgumentException($"Unknown category value: {categoryValue} in column {j}");
                        default:
                            for (int c = 0; c < nCols; c++)
                            {
                                result[i, outCol++] = NumOps.FromDouble(0);
                            }
                            break;
                    }
                }
                else if (nCategories <= 1)
                {
                    result[i, outCol++] = NumOps.FromDouble(0);
                }
                else
                {
                    var contrastMatrix = _contrastMatrices[j];
                    for (int c = 0; c < nCols; c++)
                    {
                        result[i, outCol++] = NumOps.FromDouble(contrastMatrix[categoryIndex, c]);
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
        throw new NotSupportedException("BackwardDifferenceEncoder does not support inverse transformation.");
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
                    names.Add($"{baseName}_diff_{c + 1}_vs_{c}");
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
/// Specifies how to handle unknown categories in BackwardDifferenceEncoder.
/// </summary>
public enum BackwardDifferenceHandleUnknown
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
