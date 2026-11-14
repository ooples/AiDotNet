using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.Encoders;

/// <summary>
/// Encodes categorical features using leave-one-out target encoding.
/// </summary>
/// <remarks>
/// <para>
/// LeaveOneOutEncoder is similar to TargetEncoder but uses leave-one-out statistics
/// to prevent overfitting. For each sample, the encoding is computed using all other
/// samples in the same category, excluding the current sample.
/// </para>
/// <para>
/// This reduces the risk of target leakage during training while still capturing
/// the relationship between categories and the target variable.
/// </para>
/// <para><b>For Beginners:</b> Regular target encoding can overfit because it uses
/// the same data to encode and train. Leave-one-out encoding prevents this:
/// - When encoding row 1, it uses the average of all OTHER rows with the same category
/// - This prevents the model from "cheating" by memorizing individual samples
///
/// Example: If "Category A" has 3 samples with targets [1, 0, 1]:
/// - Row 1 gets encoded as average of [0, 1] = 0.5
/// - Row 2 gets encoded as average of [1, 1] = 1.0
/// - Row 3 gets encoded as average of [1, 0] = 0.5
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class LeaveOneOutEncoder<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly double _smoothing;
    private readonly LeaveOneOutHandleUnknown _handleUnknown;

    // Fitted parameters
    private Dictionary<int, CategoryStatistics>? _categoryStats;
    private T _globalMean = MathHelper.GetNumericOperations<T>().Zero;
    private int _nInputFeatures;

    /// <summary>
    /// Gets the smoothing parameter.
    /// </summary>
    public double Smoothing => _smoothing;

    /// <summary>
    /// Gets how unknown categories are handled.
    /// </summary>
    public LeaveOneOutHandleUnknown HandleUnknown => _handleUnknown;

    /// <summary>
    /// Gets the global target mean.
    /// </summary>
    public T GlobalMean => _globalMean;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="LeaveOneOutEncoder{T}"/>.
    /// </summary>
    /// <param name="smoothing">Smoothing parameter for regularization. Defaults to 1.0.</param>
    /// <param name="handleUnknown">How to handle unknown categories. Defaults to UseGlobalMean.</param>
    /// <param name="columnIndices">The column indices to encode, or null for all columns.</param>
    public LeaveOneOutEncoder(
        double smoothing = 1.0,
        LeaveOneOutHandleUnknown handleUnknown = LeaveOneOutHandleUnknown.UseGlobalMean,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (smoothing < 0)
        {
            throw new ArgumentException("Smoothing must be non-negative.", nameof(smoothing));
        }

        _smoothing = smoothing;
        _handleUnknown = handleUnknown;
    }

    /// <summary>
    /// Fits the encoder (requires target via specialized Fit method).
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "LeaveOneOutEncoder requires target values for fitting. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <summary>
    /// Fits the encoder by computing category statistics.
    /// </summary>
    /// <param name="data">The feature matrix to fit.</param>
    /// <param name="target">The target values.</param>
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
        {
            throw new ArgumentException("Target length must match the number of rows in data.");
        }

        _nInputFeatures = data.Columns;
        var engine = Engine;
        var columnsToProcess = GetColumnsToProcess(_nInputFeatures);

        // Compute global mean
        var y = Tensor<T>.FromVector(target);
        _globalMean = engine.TensorMean(y);

        // Compute category statistics (per-category target sums and counts)
        _categoryStats = new Dictionary<int, CategoryStatistics>();
        foreach (int col in columnsToProcess)
        {
            var column = PreprocessingTensorOps.Column(data, col);
            var categories = PreprocessingTensorOps.SortedUnique(column);
            var position = PreprocessingTensorOps.LookupSorted(categories, column, out _);
            _categoryStats[col] = new CategoryStatistics(
                categories,
                PreprocessingTensorOps.CategorySums(position, y, categories.Length),
                PreprocessingTensorOps.CategoryCounts<T>(position, categories.Length));
        }

        IsFitted = true;
    }

    /// <summary>
    /// Fits and transforms using leave-one-out encoding.
    /// </summary>
    /// <param name="data">The feature matrix.</param>
    /// <param name="target">The target values.</param>
    /// <returns>The encoded data with leave-one-out statistics.</returns>
    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return TransformWithTarget(data, target);
    }

    /// <summary>
    /// Transforms the data using leave-one-out encoding (requires target for training data).
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <param name="target">The target values (for leave-one-out calculation).</param>
    /// <returns>The encoded data.</returns>
    public Matrix<T> TransformWithTarget(Matrix<T> data, Vector<T> target)
    {
        if (_categoryStats is null)
        {
            throw new InvalidOperationException("Encoder has not been fitted.");
        }

        if (data.Rows != target.Length)
        {
            throw new ArgumentException("Target length must match data rows.");
        }

        var engine = Engine;
        int numRows = data.Rows;
        int numColumns = data.Columns;
        var processSet = new HashSet<int>(GetColumnsToProcess(numColumns));
        var y = Tensor<T>.FromVector(target);
        var globalMean = PreprocessingTensorOps.Filled(new[] { numRows }, _globalMean);
        T smoothing = NumOps.FromDouble(_smoothing);
        T smoothedPrior = NumOps.Multiply(smoothing, _globalMean);
        var columns = new List<Tensor<T>>(numColumns);

        for (int j = 0; j < numColumns; j++)
        {
            var column = PreprocessingTensorOps.Column(data, j);
            if (processSet.Contains(j))
            {
                if (_categoryStats.TryGetValue(j, out var stats) && stats.Categories.Length > 0 && numRows > 0)
                {
                    // Leave-one-out: exclude the current sample from the statistics of its category
                    var position = PreprocessingTensorOps.LookupSorted(stats.Categories, column, out var found);
                    var looSum = engine.TensorSubtract(engine.TensorIndexSelect(stats.Sums, position, 0), y);
                    var looCount = engine.TensorSubtractScalar(engine.TensorIndexSelect(stats.Counts, position, 0), NumOps.One);
                    var looMean = engine.TensorDivide(looSum, looCount);

                    // Apply smoothing; the only sample of its category uses the global mean
                    var smoothed = engine.TensorDivide(
                        engine.TensorAddScalar(engine.TensorMultiply(looCount, looMean), smoothedPrior),
                        engine.TensorAddScalar(looCount, smoothing));
                    var encoded = engine.TensorWhere(engine.TensorGreaterThan(looCount, NumOps.Zero), smoothed, globalMean);
                    column = engine.TensorWhere(found, encoded, globalMean);
                }
                else
                {
                    column = globalMean;
                }
            }

            columns.Add(column);
        }

        return PreprocessingTensorOps.StackColumns(columns, numRows);
    }

    /// <summary>
    /// Transforms new data using full category statistics (for test data).
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The encoded data.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_categoryStats is null)
        {
            throw new InvalidOperationException("Encoder has not been fitted.");
        }

        // Use full category statistics for test data: (count * mean + smoothing * global) / (count + smoothing)
        var engine = Engine;
        T smoothing = NumOps.FromDouble(_smoothing);
        T smoothedPrior = NumOps.Multiply(smoothing, _globalMean);
        var maps = new Dictionary<int, CategoryValueMap<T>>();
        foreach (var entry in _categoryStats)
        {
            var stats = entry.Value;
            var mean = engine.TensorDivide(stats.Sums, stats.Counts);
            var encoded = engine.TensorDivide(
                engine.TensorAddScalar(engine.TensorMultiply(stats.Counts, mean), smoothedPrior),
                engine.TensorAddScalar(stats.Counts, smoothing));
            maps[entry.Key] = new CategoryValueMap<T>(stats.Categories, encoded);
        }

        var processSet = new HashSet<int>(GetColumnsToProcess(data.Columns));
        return CategoryValueMap<T>.EncodeColumns(data, processSet, maps, _ => _globalMean,
            _handleUnknown == LeaveOneOutHandleUnknown.Error);
    }

    private sealed class CategoryStatistics
    {
        public CategoryStatistics(Tensor<T> categories, Tensor<T> sums, Tensor<T> counts)
        {
            // Engine results kept across calls: copy them out of any tensor arena
            Categories = CategoryValueMap<T>.Owned(categories);
            Sums = CategoryValueMap<T>.Owned(sums);
            Counts = CategoryValueMap<T>.Owned(counts);
        }

        public Tensor<T> Categories { get; }

        public Tensor<T> Sums { get; }

        public Tensor<T> Counts { get; }
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("LeaveOneOutEncoder does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }
}

/// <summary>
/// Specifies how to handle unknown categories during transformation.
/// </summary>
public enum LeaveOneOutHandleUnknown
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
