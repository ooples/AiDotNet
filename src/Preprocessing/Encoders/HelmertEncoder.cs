using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.Encoders;

/// <summary>
/// Encodes categorical features using Helmert coding.
/// </summary>
/// <remarks>
/// <para>
/// HelmertEncoder compares each level of a categorical variable to the mean of
/// all subsequent levels. This is useful when you want to understand how each
/// level differs from the average of all levels that come after it.
/// </para>
/// <para>
/// For k categories, creates k-1 columns. Column i compares level i to the
/// mean of levels i+1 through k.
/// </para>
/// <para><b>For Beginners:</b> Helmert coding is useful when order matters:
/// - Compares each level to the "future average"
/// - First level vs. average of all others
/// - Second level vs. average of third, fourth, etc.
/// - Good for detecting trends or cumulative effects
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class HelmertEncoder<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly HelmertHandleUnknown _handleUnknown;
    private readonly bool _reversed;

    // Fitted parameters
    private Dictionary<int, List<double>>? _categories;
    private Dictionary<int, double[,]>? _contrastMatrices;
    private int _nInputFeatures;
    private int _nOutputFeatures;

    /// <summary>
    /// Gets how unknown categories are handled.
    /// </summary>
    public HelmertHandleUnknown HandleUnknown => _handleUnknown;

    /// <summary>
    /// Gets whether reversed Helmert coding is used.
    /// </summary>
    public bool Reversed => _reversed;

    /// <summary>
    /// Gets the number of output features.
    /// </summary>
    public int NOutputFeatures => _nOutputFeatures;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="HelmertEncoder{T}"/>.
    /// </summary>
    /// <param name="handleUnknown">How to handle unknown categories. Defaults to UseZeros.</param>
    /// <param name="reversed">If true, compare each level to mean of previous levels. Defaults to false.</param>
    /// <param name="columnIndices">The column indices to encode, or null for all columns.</param>
    public HelmertEncoder(
        HelmertHandleUnknown handleUnknown = HelmertHandleUnknown.UseZeros,
        bool reversed = false,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        _handleUnknown = handleUnknown;
        _reversed = reversed;
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

            // Build Helmert contrast matrix
            int k = sortedCategories.Count;
            if (k > 1)
            {
                var contrastMatrix = _reversed
                    ? BuildReversedHelmertMatrix(k)
                    : BuildHelmertMatrix(k);
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

    private double[,] BuildHelmertMatrix(int k)
    {
        // Standard Helmert contrast matrix
        // Compares each level to the mean of subsequent levels
        var matrix = new double[k, k - 1];

        for (int col = 0; col < k - 1; col++)
        {
            // For contrast col, we compare level col to mean of levels col+1 through k-1
            int nSubsequent = k - col - 1;

            for (int row = 0; row < k; row++)
            {
                if (row < col)
                {
                    // Levels before the comparison: 0
                    matrix[row, col] = 0;
                }
                else if (row == col)
                {
                    // The level being compared: positive weight
                    matrix[row, col] = (double)nSubsequent / (nSubsequent + 1);
                }
                else
                {
                    // Subsequent levels: negative weight (share of -1)
                    matrix[row, col] = -1.0 / (nSubsequent + 1);
                }
            }
        }

        return matrix;
    }

    private double[,] BuildReversedHelmertMatrix(int k)
    {
        // Reversed Helmert contrast matrix
        // Compares each level to the mean of previous levels
        var matrix = new double[k, k - 1];

        for (int col = 0; col < k - 1; col++)
        {
            // For contrast col, we compare level col+1 to mean of levels 0 through col
            int nPrevious = col + 1;

            for (int row = 0; row < k; row++)
            {
                if (row < col + 1)
                {
                    // Previous levels: negative weight
                    matrix[row, col] = -1.0 / (nPrevious + 1);
                }
                else if (row == col + 1)
                {
                    // The level being compared: positive weight
                    matrix[row, col] = (double)nPrevious / (nPrevious + 1);
                }
                else
                {
                    // Subsequent levels: 0
                    matrix[row, col] = 0;
                }
            }
        }

        return matrix;
    }

    /// <summary>
    /// Transforms the data using Helmert coding.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_categories is null || _contrastMatrices is null)
        {
            throw new InvalidOperationException("HelmertEncoder has not been fitted.");
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
                        case HelmertHandleUnknown.UseZeros:
                            for (int c = 0; c < nCols; c++)
                            {
                                result[i, outCol++] = NumOps.FromDouble(0);
                            }
                            break;
                        case HelmertHandleUnknown.Error:
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
        throw new NotSupportedException("HelmertEncoder does not support inverse transformation.");
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
                string suffix = _reversed ? "rev_helmert" : "helmert";

                for (int c = 0; c < nCols; c++)
                {
                    names.Add($"{baseName}_{suffix}_{c}");
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
/// Specifies how to handle unknown categories in HelmertEncoder.
/// </summary>
public enum HelmertHandleUnknown
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
