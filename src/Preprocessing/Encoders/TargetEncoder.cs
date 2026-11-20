using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.Encoders;

/// <summary>
/// Encodes categorical features using target mean encoding.
/// </summary>
/// <remarks>
/// <para>
/// TargetEncoder replaces each category with the mean of the target variable for that category.
/// This creates a continuous feature that captures the relationship between the category and target.
/// </para>
/// <para>
/// To prevent overfitting, especially with rare categories, smoothing is applied:
/// encoding = (count * category_mean + smoothing * global_mean) / (count + smoothing)
/// </para>
/// <para><b>For Beginners:</b> Instead of one-hot encoding (many columns), target encoding
/// creates a single column per feature containing the average target value for each category:
/// - Category "A" with average target 0.8 becomes 0.8
/// - Category "B" with average target 0.3 becomes 0.3
///
/// This is especially useful for high-cardinality features where one-hot would create
/// too many columns.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class TargetEncoder<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly double _smoothing;
    private readonly double _minSamplesLeaf;
    private readonly TargetEncoderHandleUnknown _handleUnknown;

    // Fitted parameters
    private Dictionary<int, CategoryValueMap<T>>? _encodingMaps;
    private Tensor<T>? _globalMeans;
    private int _nInputFeatures;

    /// <summary>
    /// Gets the smoothing parameter used during encoding.
    /// </summary>
    public double Smoothing => _smoothing;

    /// <summary>
    /// Gets how unknown categories are handled during transform.
    /// </summary>
    public TargetEncoderHandleUnknown HandleUnknown => _handleUnknown;

    /// <summary>
    /// Gets the encoding maps for each column.
    /// </summary>
    public Dictionary<int, CategoryValueMap<T>>? EncodingMaps => _encodingMaps;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="TargetEncoder{T}"/>.
    /// </summary>
    /// <param name="smoothing">Smoothing parameter. Higher values give more weight to global mean. Defaults to 1.0.</param>
    /// <param name="minSamplesLeaf">Minimum samples to compute category mean. Categories below this use global mean. Defaults to 1.</param>
    /// <param name="handleUnknown">How to handle unknown categories during transform. Defaults to UseGlobalMean.</param>
    /// <param name="columnIndices">The column indices to encode, or null for all columns.</param>
    public TargetEncoder(
        double smoothing = 1.0,
        double minSamplesLeaf = 1,
        TargetEncoderHandleUnknown handleUnknown = TargetEncoderHandleUnknown.UseGlobalMean,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (smoothing < 0)
        {
            throw new ArgumentException("Smoothing must be non-negative.", nameof(smoothing));
        }

        if (minSamplesLeaf < 1)
        {
            throw new ArgumentException("Minimum samples per leaf must be at least 1.", nameof(minSamplesLeaf));
        }

        _smoothing = smoothing;
        _minSamplesLeaf = minSamplesLeaf;
        _handleUnknown = handleUnknown;
    }

    /// <summary>
    /// Fits the encoder by learning the target means for each category.
    /// </summary>
    /// <param name="data">The feature matrix to fit.</param>
    /// <param name="target">The target values used to compute means.</param>
    /// <exception cref="ArgumentException">If target length doesn't match data rows.</exception>
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
        {
            throw new ArgumentException("Target length must match the number of rows in data.");
        }

        _nInputFeatures = data.Columns;
        var engine = Engine;
        var columnsToProcess = GetColumnsToProcess(_nInputFeatures);

        _encodingMaps = new Dictionary<int, CategoryValueMap<T>>();
        var globalMeans = new bool[_nInputFeatures];

        // Global target mean
        var y = Tensor<T>.FromVector(target);
        T globalTargetMean = engine.TensorMean(y);
        T smoothing = NumOps.FromDouble(_smoothing);
        T minSamplesLeaf = NumOps.FromDouble(_minSamplesLeaf);

        foreach (int col in columnsToProcess)
        {
            globalMeans[col] = true;

            // Group by category value: per-category target sums and counts
            var column = PreprocessingTensorOps.Column(data, col);
            var categories = PreprocessingTensorOps.SortedUnique(column);
            var position = PreprocessingTensorOps.LookupSorted(categories, column, out _);
            var counts = PreprocessingTensorOps.CategoryCounts<T>(position, categories.Length);
            var sums = PreprocessingTensorOps.CategorySums(position, y, categories.Length);

            // Smoothed means (count * category_mean + smoothing * global_mean) / (count + smoothing);
            // rare categories (count < minSamplesLeaf) use the global mean
            var categoryMean = engine.TensorDivide(sums, counts);
            var smoothed = engine.TensorDivide(
                engine.TensorAddScalar(engine.TensorMultiply(counts, categoryMean), NumOps.Multiply(smoothing, globalTargetMean)),
                engine.TensorAddScalar(counts, smoothing));
            var global = PreprocessingTensorOps.Filled(new[] { categories.Length }, globalTargetMean);
            var encoded = engine.TensorWhere(PreprocessingTensorOps.AtLeast(counts, minSamplesLeaf), smoothed, global);

            _encodingMaps[col] = new CategoryValueMap<T>(categories, encoded);
        }

        // Global mean per column (0 for columns that are not encoded)
        _globalMeans = CategoryValueMap<T>.Owned(engine.TensorWhere(globalMeans,
            PreprocessingTensorOps.Filled(new[] { _nInputFeatures }, globalTargetMean), new Tensor<T>(new[] { _nInputFeatures })));
        IsFitted = true;
    }

    /// <summary>
    /// Fits the encoder using the base Fit method (requires target via FitWithTarget).
    /// </summary>
    /// <param name="data">The feature matrix.</param>
    /// <exception cref="InvalidOperationException">Always thrown. Use Fit(Matrix, Vector) instead.</exception>
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "TargetEncoder requires target values for fitting. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <summary>
    /// Fits the encoder and transforms the data in one step.
    /// </summary>
    /// <param name="data">The feature matrix to fit and transform.</param>
    /// <param name="target">The target values used to compute means.</param>
    /// <returns>The encoded data.</returns>
    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    /// <summary>
    /// Transforms the data by replacing categories with their target means.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The encoded data.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_encodingMaps is null || _globalMeans is null)
        {
            throw new InvalidOperationException("Encoder has not been fitted.");
        }

        int numRows = data.Rows;
        int numColumns = data.Columns;
        var processSet = new HashSet<int>(GetColumnsToProcess(numColumns));
        var columns = new List<Tensor<T>>(numColumns);

        for (int j = 0; j < numColumns; j++)
        {
            var raw = PreprocessingTensorOps.Column(data, j);
            var column = raw;
            if (processSet.Contains(j))
            {
                // Unknown categories use the column's global mean (or throw)
                var globalMean = PreprocessingTensorOps.Filled(new[] { numRows }, _globalMeans.GetFlat(j));
                Tensor<T> found;
                if (_encodingMaps.TryGetValue(j, out var encodingMap))
                {
                    column = encodingMap.Map(raw, globalMean, out found);
                }
                else
                {
                    found = new Tensor<T>(new[] { numRows });
                    column = globalMean;
                }

                if (_handleUnknown == TargetEncoderHandleUnknown.Error)
                {
                    var missing = PreprocessingTensorOps.MaskPositions(PreprocessingTensorOps.Not(found));
                    if (missing.Length > 0)
                        throw new ArgumentException(
                            $"Unknown category value: {raw.GetFlat(missing[0])} in column {j}");
                }
            }

            columns.Add(column);
        }

        return PreprocessingTensorOps.StackColumns(columns, numRows);
    }

    /// <summary>
    /// Inverse transformation is not supported for target encoding.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("TargetEncoder does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (inputFeatureNames is null)
        {
            var names = new string[_nInputFeatures];
            for (int i = 0; i < _nInputFeatures; i++)
            {
                names[i] = $"x{i}";
            }
            return names;
        }

        return inputFeatureNames;
    }
}

/// <summary>
/// Specifies how to handle unknown categories during transformation.
/// </summary>
public enum TargetEncoderHandleUnknown
{
    /// <summary>
    /// Use the global target mean for unknown categories.
    /// </summary>
    UseGlobalMean,

    /// <summary>
    /// Raise an error when an unknown category is encountered.
    /// </summary>
    Error
}
