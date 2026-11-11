using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

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
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class BinaryEncoder<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly BinaryEncoderHandleUnknown _handleUnknown;

    // Fitted parameters
    private List<Tensor<T>>? _categories;
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

        _categories = new List<Tensor<T>>();
        _bitsPerColumn = new Dictionary<int, int>();
        _featureIndicesStart = new List<int>();

        int currentOutputIndex = 0;

        for (int col = 0; col < _nInputFeatures; col++)
        {
            _featureIndicesStart.Add(currentOutputIndex);

            if (!columnsToProcess.Contains(col))
            {
                // Pass-through column
                _categories.Add(new Tensor<T>(new[] { 0 }));
                _bitsPerColumn[col] = 0;
                currentOutputIndex += 1;
                continue;
            }

            // Sorted unique values; category i gets ordinal i + 1 (0 is left for unknown)
            var column = PreprocessingTensorOps.Column(data, col);
            var categories = column.Length == 0 ? new Tensor<T>(new[] { 0 }) : PreprocessingTensorOps.SortedUnique(column);
            _categories.Add(categories);

            // Calculate bits needed: ceil(log2(n + 1)) where +1 is for unknown category
            int numCategories = categories.Length + 1; // +1 for unknown
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
        if (_categories is null || _bitsPerColumn is null || _featureIndicesStart is null)
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
            if (!processSet.Contains(col) || _bitsPerColumn[col] == 0)
            {
                // Pass-through: copy value directly
                blocks.Add(engine.Reshape(column, new[] { numRows, 1 }));
                continue;
            }

            // Known categories get their ordinal's bits; unknown ones stay all zeros (or throw)
            var ordinals = PreprocessingTensorOps.OrdinalCodes(_categories[col], column, out var found);
            if (_handleUnknown == BinaryEncoderHandleUnknown.Error)
            {
                var missing = PreprocessingTensorOps.MaskPositions(PreprocessingTensorOps.Not(found));
                if (missing.Length > 0)
                    throw new ArgumentException($"Unknown category value: {column.GetFlat(missing[0])} in column {col}");
            }

            blocks.Add(PreprocessingTensorOps.DigitBlock<T>(ordinals, _bitsPerColumn[col], 2));
        }

        if (blocks.Count == 0)
            return new Matrix<T>(numRows, 0);
        return engine.TensorConcatenate(blocks.ToArray(), 1).ToMatrix();
    }

    /// <summary>
    /// Reverses the binary encoding to get original category values.
    /// </summary>
    /// <param name="data">The binary encoded data.</param>
    /// <returns>The original categorical values.</returns>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_categories is null || _bitsPerColumn is null || _featureIndicesStart is null)
        {
            throw new InvalidOperationException("Encoder has not been fitted.");
        }

        var engine = Engine;
        int numRows = data.Rows;
        var processSet = new HashSet<int>(GetColumnsToProcess(_nInputFeatures));
        var encoded = Tensor<T>.FromMatrix(data);
        var columns = new List<Tensor<T>>(_nInputFeatures);

        for (int col = 0; col < _nInputFeatures; col++)
        {
            int outputStart = _featureIndicesStart[col];
            int bits = _bitsPerColumn[col];
            if (!processSet.Contains(col) || bits == 0)
            {
                // Pass-through: copy value back
                columns.Add(PreprocessingTensorOps.Column(data, outputStart));
                continue;
            }

            // Ordinal o in 1..K is category o - 1; any other code (unknown) becomes 0
            var categories = _categories[col];
            int k = categories.Length;
            var codes = PreprocessingTensorOps.ReadBinaryCodes(
                engine.TensorSlice(encoded, new[] { 0, outputStart }, new[] { numRows, bits }));
            for (int i = 0; i < codes.Length; i++)
                codes[i] = codes[i] <= k ? codes[i] : 0;

            var table = engine.TensorConcatenate(new[] { new Tensor<T>(new[] { 1 }), categories }, 0);
            columns.Add(engine.TensorIndexSelect(table, new Tensor<int>(codes, new[] { numRows }), 0));
        }

        return PreprocessingTensorOps.StackColumns(columns, numRows);
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
