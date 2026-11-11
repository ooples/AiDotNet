using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

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
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class BackwardDifferenceEncoder<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly BackwardDifferenceHandleUnknown _handleUnknown;

    // Fitted parameters
    private Dictionary<int, Tensor<T>>? _categories;
    private Dictionary<int, Tensor<T>>? _contrastMatrices;
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

        _categories = new Dictionary<int, Tensor<T>>();
        _contrastMatrices = new Dictionary<int, Tensor<T>>();

        foreach (int col in columnsToProcess)
        {
            var column = PreprocessingTensorOps.Column(data, col);
            var sortedCategories = column.Length == 0 ? new Tensor<T>(new[] { 0 }) : PreprocessingTensorOps.SortedUnique(column);
            _categories[col] = sortedCategories;

            // Build the contrast matrix (one zero column when there are fewer than two levels)
            int k = sortedCategories.Length;
            _contrastMatrices[col] = k > 1 ? BuildBackwardDifferenceMatrix(k) : new Tensor<T>(new[] { k, 1 });
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
    /// Backward difference contrast matrix: contrast col compares level col + 1 to level col. Levels up to col
    /// get -(k - col - 1) / k and later levels get (col + 1) / k.
    /// </summary>
    private static Tensor<T> BuildBackwardDifferenceMatrix(int k) =>
        PreprocessingTensorOps.RationalMatrix<T>(k, k - 1,
            (row, col) => row <= col ? -(k - col - 1) : col + 1,
            _ => k);

    /// <summary>
    /// Transforms the data using backward difference coding.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_categories is null || _contrastMatrices is null)
        {
            throw new InvalidOperationException("BackwardDifferenceEncoder has not been fitted.");
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
                _handleUnknown == BackwardDifferenceHandleUnknown.Error, j));
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
                int nCols = Math.Max(1, categories.Length - 1);

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
