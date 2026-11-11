using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

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
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class LabelEncoder<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    // Fitted parameters: mapping from value to label for each column
    private List<Tensor<T>>? _categories;

    /// <summary>
    /// Gets the number of unique classes for each encoded column.
    /// </summary>
    public int[]? NClasses
    {
        get
        {
            return _categories?.Select(c => c.Length).ToArray();
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
        var columnsToProcess = new HashSet<int>(GetColumnsToProcess(numColumns));

        // Sorted distinct values per processed column; the label of a value is its position
        _categories = new List<Tensor<T>>();
        for (int col = 0; col < numColumns; col++)
        {
            _categories.Add(columnsToProcess.Contains(col)
                ? PreprocessingTensorOps.SortedUnique(PreprocessingTensorOps.Column(data, col))
                : new Tensor<T>(new[] { 0 }));
        }
    }

    /// <summary>
    /// Transforms the data by replacing categorical values with their integer labels.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The transformed data with integer labels.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_categories is null)
        {
            throw new InvalidOperationException("Encoder has not been fitted.");
        }

        var engine = Engine;
        int numRows = data.Rows;
        int numColumns = data.Columns;
        var processSet = new HashSet<int>(GetColumnsToProcess(numColumns));
        var unknown = PreprocessingTensorOps.Filled(new[] { numRows }, NumOps.FromDouble(-1));
        var columns = new List<Tensor<T>>(numColumns);

        for (int j = 0; j < numColumns; j++)
        {
            var column = PreprocessingTensorOps.Column(data, j);
            if (processSet.Contains(j) && _categories[j].Length > 0)
            {
                // Unknown values get -1
                var labels = PreprocessingTensorOps.LookupSorted(_categories[j], column, out var found);
                column = engine.TensorWhere(found, labels.Cast<T>(), unknown);
            }

            columns.Add(column);
        }

        return PreprocessingTensorOps.StackColumns(columns, numRows);
    }

    /// <summary>
    /// Reverses the label encoding to recover original values.
    /// </summary>
    /// <param name="data">The label-encoded data.</param>
    /// <returns>The data with original categorical values.</returns>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_categories is null)
        {
            throw new InvalidOperationException("Encoder has not been fitted.");
        }

        var engine = Engine;
        int numRows = data.Rows;
        int numColumns = data.Columns;
        var processSet = new HashSet<int>(GetColumnsToProcess(numColumns));
        var columns = new List<Tensor<T>>(numColumns);

        for (int j = 0; j < numColumns; j++)
        {
            var column = PreprocessingTensorOps.Column(data, j);
            if (processSet.Contains(j) && _categories[j].Length > 0)
            {
                // The label is the value truncated toward zero; unknown labels keep their value
                column = PreprocessingTensorOps.GatherCategories(_categories[j], column);
            }

            columns.Add(column);
        }

        return PreprocessingTensorOps.StackColumns(columns, numRows);
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
