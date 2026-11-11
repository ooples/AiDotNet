using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

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
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class OrdinalEncoder<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly List<Vector<T>>? _categories;
    private readonly UnknownValueHandling _handleUnknown;
    private readonly double _unknownValue;

    // Fitted parameters: mapping from value to ordinal for each column
    private List<Tensor<T>>? _ordinalCategories;
    private List<Tensor<T>>? _sortedCategories;
    private List<Tensor<T>>? _sortedOrdinals;

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
        List<Vector<T>>? categories = null,
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
        var columnsToProcess = new HashSet<int>(GetColumnsToProcess(numColumns));

        _ordinalCategories = new List<Tensor<T>>();
        _sortedCategories = new List<Tensor<T>>();
        _sortedOrdinals = new List<Tensor<T>>();

        for (int col = 0; col < numColumns; col++)
        {
            Tensor<T> categoryOrder;
            if (!columnsToProcess.Contains(col))
            {
                categoryOrder = new Tensor<T>(new[] { 0 });
            }
            else if (_categories is not null && col < _categories.Count && _categories[col].Length > 0)
            {
                // Use provided category ordering
                categoryOrder = Tensor<T>.FromVector(_categories[col]);
            }
            else
            {
                // Infer from data (sorted)
                categoryOrder = PreprocessingTensorOps.SortedUnique(PreprocessingTensorOps.Column(data, col));
            }

            // Sorted view of the categories (for lookups) with the ordinal of each sorted entry
            _ordinalCategories.Add(categoryOrder);
            if (categoryOrder.Length == 0)
            {
                _sortedCategories.Add(categoryOrder);
                _sortedOrdinals.Add(categoryOrder);
            }
            else
            {
                var (sorted, order) = Engine.TensorSort(categoryOrder, 0, false);
                _sortedCategories.Add(sorted);
                _sortedOrdinals.Add(order.Cast<T>());
            }
        }
    }

    /// <summary>
    /// Transforms the data by replacing categorical values with their ordinal integers.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The transformed data with ordinal values.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_ordinalCategories is null || _sortedCategories is null || _sortedOrdinals is null)
        {
            throw new InvalidOperationException("Encoder has not been fitted.");
        }

        var engine = Engine;
        int numRows = data.Rows;
        int numColumns = data.Columns;
        var processSet = new HashSet<int>(GetColumnsToProcess(numColumns));
        var unknown = PreprocessingTensorOps.Filled(new[] { numRows }, NumOps.FromDouble(_unknownValue));
        var columns = new List<Tensor<T>>(numColumns);

        for (int j = 0; j < numColumns; j++)
        {
            var column = PreprocessingTensorOps.Column(data, j);
            if (processSet.Contains(j) && _ordinalCategories[j].Length > 0)
            {
                var position = PreprocessingTensorOps.LookupSorted(_sortedCategories[j], column, out var found);
                var ordinals = engine.TensorIndexSelect(_sortedOrdinals[j], position, 0);

                if (_handleUnknown == UnknownValueHandling.Error)
                {
                    var missing = PreprocessingTensorOps.MaskPositions(PreprocessingTensorOps.Not(found));
                    if (missing.Length > 0)
                        throw new ArgumentException($"Unknown category value: {column.GetFlat(missing[0])} in column {j}");
                }

                column = engine.TensorWhere(found, ordinals, unknown);
            }

            columns.Add(column);
        }

        return PreprocessingTensorOps.StackColumns(columns, numRows);
    }

    /// <summary>
    /// Reverses the ordinal encoding to recover original values.
    /// </summary>
    /// <param name="data">The ordinal-encoded data.</param>
    /// <returns>The data with original categorical values.</returns>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_ordinalCategories is null)
        {
            throw new InvalidOperationException("Encoder has not been fitted.");
        }

        int numRows = data.Rows;
        int numColumns = data.Columns;
        var processSet = new HashSet<int>(GetColumnsToProcess(numColumns));
        var columns = new List<Tensor<T>>(numColumns);

        for (int j = 0; j < numColumns; j++)
        {
            var column = PreprocessingTensorOps.Column(data, j);
            if (processSet.Contains(j) && _ordinalCategories[j].Length > 0)
            {
                // The ordinal is the value truncated toward zero; unknown ordinals keep their value
                column = PreprocessingTensorOps.GatherCategories(_ordinalCategories[j], column);
            }

            columns.Add(column);
        }

        return PreprocessingTensorOps.StackColumns(columns, numRows);
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
