using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

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
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class HelmertEncoder<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly HelmertHandleUnknown _handleUnknown;
    private readonly bool _reversed;

    // Fitted parameters
    private Dictionary<int, Tensor<T>>? _categories;
    private Dictionary<int, Tensor<T>>? _contrastMatrices;
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

        _categories = new Dictionary<int, Tensor<T>>();
        _contrastMatrices = new Dictionary<int, Tensor<T>>();

        foreach (int col in columnsToProcess)
        {
            var column = PreprocessingTensorOps.Column(data, col);
            var sortedCategories = column.Length == 0 ? new Tensor<T>(new[] { 0 }) : PreprocessingTensorOps.SortedUnique(column);
            _categories[col] = CategoryValueMap<T>.Owned(sortedCategories);

            // Build the contrast matrix (one zero column when there are fewer than two levels)
            int k = sortedCategories.Length;
            _contrastMatrices[col] = CategoryValueMap<T>.Owned(
                k > 1 ? (_reversed ? BuildReversedHelmertMatrix(k) : BuildHelmertMatrix(k)) : new Tensor<T>(new[] { k, 1 }));
        }

        // Calculate total output features
        _nOutputFeatures = 0;
        var processSet = new HashSet<int>(columnsToProcess);

        for (int col = 0; col < _nInputFeatures; col++)
        {
            if (processSet.Contains(col))
            {
                int nCategories = _categories[col].Length;
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
    /// Standard Helmert contrast matrix: compares each level to the mean of subsequent levels.
    /// Contrast col gives level col the weight nSubsequent / (nSubsequent + 1), each later level
    /// -1 / (nSubsequent + 1) and earlier levels 0, where nSubsequent = k - col - 1.
    /// </summary>
    private static Tensor<T> BuildHelmertMatrix(int k) =>
        PreprocessingTensorOps.RationalMatrix<T>(k, k - 1,
            (row, col) => row < col ? 0 : row == col ? k - col - 1 : -1,
            col => k - col);

    /// <summary>
    /// Reversed Helmert contrast matrix: compares each level to the mean of previous levels.
    /// Contrast col gives levels 0..col the weight -1 / (nPrevious + 1), level col + 1 the weight
    /// nPrevious / (nPrevious + 1) and later levels 0, where nPrevious = col + 1.
    /// </summary>
    private static Tensor<T> BuildReversedHelmertMatrix(int k) =>
        PreprocessingTensorOps.RationalMatrix<T>(k, k - 1,
            (row, col) => row < col + 1 ? -1 : row == col + 1 ? col + 1 : 0,
            col => col + 2);

    /// <summary>
    /// Transforms the data using Helmert coding.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_categories is null || _contrastMatrices is null)
        {
            throw new InvalidOperationException("HelmertEncoder has not been fitted.");
        }

        var engine = Engine;
        int numRows = data.Rows;
        var processSet = new HashSet<int>(GetColumnsToProcess(_nInputFeatures));
        var blocks = new List<Tensor<T>>(_nInputFeatures);

        for (int j = 0; j < _nInputFeatures; j++)
        {
            var column = PreprocessingTensorOps.Column(data, j);
            if (!processSet.Contains(j))
            {
                blocks.Add(engine.Reshape(column, new[] { numRows, 1 }));
                continue;
            }

            // Known categories take their contrast row; unknown ones are zeros (or throw)
            blocks.Add(PreprocessingTensorOps.ContrastBlock(_categories[j], _contrastMatrices[j], column,
                _handleUnknown == HelmertHandleUnknown.Error, j));
        }

        if (blocks.Count == 0)
            return new Matrix<T>(numRows, 0);
        return engine.TensorConcatenate(blocks.ToArray(), 1).ToMatrix();
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
                int nCols = Math.Max(1, categories.Length - 1);
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
