using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

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
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class OneHotEncoder<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly bool _dropFirst;
    private readonly OneHotUnknownHandling _handleUnknown;

    // Fitted parameters
    private List<Vector<T>>? _categories;
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
    public List<Vector<T>>? Categories => _categories;

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
        var columnsToProcess = new HashSet<int>(GetColumnsToProcess(_nInputFeatures));

        _categories = new List<Vector<T>>();
        _featureIndicesStart = new List<int>();
        int currentOutputIndex = 0;

        for (int col = 0; col < _nInputFeatures; col++)
        {
            _featureIndicesStart.Add(currentOutputIndex);

            if (!columnsToProcess.Contains(col))
            {
                // Pass-through column (not encoded)
                _categories.Add(new Vector<T>(0));
                currentOutputIndex += 1;
                continue;
            }

            // Sorted distinct values are the categories
            var sortedCategories = PreprocessingTensorOps.SortedUnique(PreprocessingTensorOps.Column(data, col));
            _categories.Add(sortedCategories.ToVector());

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

        var engine = Engine;
        int numRows = data.Rows;
        var processSet = new HashSet<int>(GetColumnsToProcess(_nInputFeatures));
        var blocks = new List<Tensor<T>>(_nInputFeatures);

        for (int col = 0; col < _nInputFeatures; col++)
        {
            var column = PreprocessingTensorOps.Column(data, col);
            int nCategories = _categories[col].Length;

            if (!processSet.Contains(col) || nCategories == 0)
            {
                // Pass-through: copy value directly
                blocks.Add(engine.Reshape(column, new[] { numRows, 1 }));
                continue;
            }

            var position = PreprocessingTensorOps.LookupSorted(Tensor<T>.FromVector(_categories[col]), column, out var found);
            if (_handleUnknown == OneHotUnknownHandling.Error)
            {
                var missing = PreprocessingTensorOps.MaskPositions(PreprocessingTensorOps.Not(found));
                if (missing.Length > 0)
                    throw new ArgumentException($"Unknown category value: {column.GetFlat(missing[0])} in column {col}");
            }

            // One-hot rows; unknown values (Ignore) leave all zeros
            var oneHot = numRows == 0
                ? new Tensor<T>(new[] { 0, nCategories })
                : engine.TensorMultiply(
                    engine.TensorOneHot<T>(position, nCategories),
                    engine.TensorBroadcastTo(engine.Reshape(found, new[] { numRows, 1 }), new[] { numRows, nCategories }));

            if (!_dropFirst)
                blocks.Add(oneHot);
            else if (nCategories > 1)
                blocks.Add(engine.TensorSlice(oneHot, new[] { 0, 1 }, new[] { numRows, nCategories - 1 })); // first category is the reference
            else
                blocks.Add(new Tensor<T>(new[] { numRows, 1 }));
        }

        if (blocks.Count == 0)
            return new Matrix<T>(numRows, 0);
        return engine.TensorConcatenate(blocks.ToArray(), 1).ToMatrix();
    }

    /// <summary>
    /// Reverses the one-hot encoding to recover original categorical values.
    /// </summary>
    /// <param name="data">The one-hot encoded data.</param>
    /// <returns>The original categorical data.</returns>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_categories is null || _featureIndicesStart is null)
        {
            throw new InvalidOperationException("Encoder has not been fitted.");
        }

        var engine = Engine;
        int numRows = data.Rows;
        var processSet = new HashSet<int>(GetColumnsToProcess(_nInputFeatures));
        var X = Tensor<T>.FromMatrix(data);
        var columns = new List<Tensor<T>>(_nInputFeatures);
        T half = NumOps.FromDouble(0.5);

        for (int col = 0; col < _nInputFeatures; col++)
        {
            int outputStart = _featureIndicesStart[col];
            int nCategories = _categories[col].Length;

            if (!processSet.Contains(col) || nCategories == 0)
            {
                // Pass-through: copy value back
                columns.Add(PreprocessingTensorOps.ColumnVector(X, outputStart));
                continue;
            }

            var categories = Tensor<T>.FromVector(_categories[col]);
            if (_dropFirst && nCategories == 1)
            {
                // Only the dropped reference category exists
                columns.Add(engine.TensorIndexSelect(categories, new Tensor<int>(new int[numRows], new[] { numRows }), 0));
                continue;
            }

            // The category is the first column above 0.5 (shifted by one when the first category is dropped)
            int width = _dropFirst ? nCategories - 1 : nCategories;
            var hot = engine.TensorGreaterThan(engine.TensorSlice(X, new[] { 0, outputStart }, new[] { numRows, width }), half);
            int[] first = PreprocessingTensorOps.FirstNonZeroPerRow(hot);

            var index = new int[numRows];
            var known = new bool[numRows];
            for (int i = 0; i < numRows; i++)
            {
                bool any = first[i] < width;
                index[i] = _dropFirst ? (any ? first[i] + 1 : 0) : (any ? first[i] : 0);
                known[i] = _dropFirst || any;
            }

            // Unknown - return 0
            var values = engine.TensorIndexSelect(categories, new Tensor<int>(index, new[] { numRows }), 0);
            columns.Add(engine.TensorWhere(known, values, new Tensor<T>(new[] { numRows })));
        }

        return PreprocessingTensorOps.StackColumns(columns, numRows);
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
            var categoryValues = _categories[col].ToArray();
            for (int c = startIdx; c < categoryValues.Length; c++)
            {
                names.Add($"{baseName}_{categoryValues[c]}");
            }
        }

        return names.ToArray();
    }
}
