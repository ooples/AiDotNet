using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.Encoders;

/// <summary>
/// Encodes categorical features using base-N representation.
/// </summary>
/// <remarks>
/// <para>
/// BaseNEncoder converts category indices to base-N representation, creating
/// multiple columns with digits in the specified base. This is a generalization
/// of binary encoding (base 2).
/// </para>
/// <para>
/// For example, with base=3 and 9 categories (0-8):
/// - Category 0 → [0, 0]
/// - Category 4 → [1, 1] (4 = 1*3 + 1)
/// - Category 8 → [2, 2] (8 = 2*3 + 2)
/// </para>
/// <para><b>For Beginners:</b> BaseNEncoder is like counting in different number systems:
/// - Base 2 (binary): Uses 0 and 1 → most compact
/// - Base 3 (ternary): Uses 0, 1, 2 → slightly more columns
/// - Higher bases = fewer columns but more possible values per column
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class BaseNEncoder<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _base;
    private readonly BaseNHandleUnknown _handleUnknown;

    // Fitted parameters
    private Dictionary<int, Tensor<T>>? _categoryToIndex;
    private Dictionary<int, int>? _nDigitsPerColumn;
    private int _nInputFeatures;
    private int _nOutputFeatures;

    /// <summary>
    /// Gets the base used for encoding.
    /// </summary>
    public int Base => _base;

    /// <summary>
    /// Gets how unknown categories are handled.
    /// </summary>
    public BaseNHandleUnknown HandleUnknown => _handleUnknown;

    /// <summary>
    /// Gets the number of output features.
    /// </summary>
    public int NOutputFeatures => _nOutputFeatures;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="BaseNEncoder{T}"/>.
    /// </summary>
    /// <param name="base_">The base to use for encoding. Defaults to 2 (binary).</param>
    /// <param name="handleUnknown">How to handle unknown categories. Defaults to UseZeros.</param>
    /// <param name="columnIndices">The column indices to encode, or null for all columns.</param>
    public BaseNEncoder(
        int base_ = 2,
        BaseNHandleUnknown handleUnknown = BaseNHandleUnknown.UseZeros,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (base_ < 2)
        {
            throw new ArgumentException("Base must be at least 2.", nameof(base_));
        }

        _base = base_;
        _handleUnknown = handleUnknown;
    }

    /// <summary>
    /// Fits the encoder by learning categories.
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        var columnsToProcess = GetColumnsToProcess(_nInputFeatures);

        _categoryToIndex = new Dictionary<int, Tensor<T>>();
        _nDigitsPerColumn = new Dictionary<int, int>();

        foreach (int col in columnsToProcess)
        {
            // Sorted unique values; category i gets index i
            var column = PreprocessingTensorOps.Column(data, col);
            var sortedValues = column.Length == 0 ? new Tensor<T>(new[] { 0 }) : PreprocessingTensorOps.SortedUnique(column);
            _categoryToIndex[col] = sortedValues;

            // Calculate number of digits needed
            int nCategories = sortedValues.Length;
            int nCategories = sortedValues.Count;
            int nDigits = nCategories > 0 ? (int)Math.Ceiling(Math.Log(nCategories + 1) / Math.Log(_base)) : 1;
            nDigits = Math.Max(1, nDigits);
            _nDigitsPerColumn[col] = nDigits;
        }

        // Calculate total output features
        _nOutputFeatures = 0;
        var processSet = new HashSet<int>(columnsToProcess);

        for (int col = 0; col < _nInputFeatures; col++)
        {
            if (processSet.Contains(col))
            {
                _nOutputFeatures += _nDigitsPerColumn[col];
            }
            else
            {
                _nOutputFeatures += 1;
            }
        }

        IsFitted = true;
    }

    /// <summary>
    /// Transforms the data using base-N encoding.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_categoryToIndex is null || _nDigitsPerColumn is null)
        {
            throw new InvalidOperationException("BaseNEncoder has not been fitted.");
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

            // index + 1 for known categories so that 0 can be reserved for unknown
            var index = PreprocessingTensorOps.OrdinalCodes(_categoryToIndex[j], column, out var found);
            if (_handleUnknown == BaseNHandleUnknown.Error)
            {
                var missing = PreprocessingTensorOps.MaskPositions(PreprocessingTensorOps.Not(found));
                if (missing.Length > 0)
                    throw new ArgumentException($"Unknown category value: {column.GetFlat(missing[0])} in column {j}");
            }

            // Base-N digits, most significant first
            blocks.Add(PreprocessingTensorOps.DigitBlock<T>(index, _nDigitsPerColumn[j], _base));
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
        throw new NotSupportedException("BaseNEncoder does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_nDigitsPerColumn is null)
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
                int nDigits = _nDigitsPerColumn[col];
                for (int d = 0; d < nDigits; d++)
                {
                    names.Add($"{baseName}_base{_base}_{d}");
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
/// Specifies how to handle unknown categories in BaseNEncoder.
/// </summary>
public enum BaseNHandleUnknown
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
