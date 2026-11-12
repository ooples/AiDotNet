using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Newtonsoft.Json;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.Scalers;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.Preprocessing.Imputers;

/// <summary>
/// Specifies the strategy for imputing missing values.
/// </summary>
public enum ImputationStrategy
{
    /// <summary>
    /// Replace missing values with the mean of each feature.
    /// </summary>
    Mean,

    /// <summary>
    /// Replace missing values with the median of each feature.
    /// </summary>
    Median,

    /// <summary>
    /// Replace missing values with the most frequent value of each feature.
    /// </summary>
    MostFrequent,

    /// <summary>
    /// Replace missing values with a constant value.
    /// </summary>
    Constant
}

/// <summary>
/// Imputes missing values using simple strategies like mean, median, or constant.
/// </summary>
/// <remarks>
/// <para>
/// SimpleImputer fills in missing values (represented as NaN) using simple strategies:
/// - Mean: Replace with column mean
/// - Median: Replace with column median
/// - MostFrequent: Replace with most common value
/// - Constant: Replace with a specified value
/// </para>
/// <para><b>For Beginners:</b> This transformer fills in gaps in your data:
/// - If you have missing ages, replace them with average age (Mean)
/// - If you have missing incomes with outliers, use median income (Median)
/// - If you have missing categories, use most common category (MostFrequent)
/// - Or fill with a specific value like 0 or -1 (Constant)
///
/// Example with Mean strategy:
/// [1, 2, NaN, 4, 5] → [1, 2, 3, 4, 5] (NaN replaced with mean=3)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SimpleImputer<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    [JsonProperty]
    private readonly ImputationStrategy _strategy;
    [JsonProperty]
    private readonly T? _fillValue;
    [JsonProperty]
    private readonly T _missingValue;

    // Fitted parameters: statistics for each column
    [JsonProperty]
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>? _statistics;

    /// <summary>
    /// Gets the imputation strategy used.
    /// </summary>
    [JsonIgnore]
    public ImputationStrategy Strategy => _strategy;

    /// <summary>
    /// Gets the fill value used for Constant strategy.
    /// </summary>
    [JsonIgnore]
    public T? FillValue => _fillValue;

    /// <summary>
    /// Gets the value considered as missing.
    /// </summary>
    [JsonIgnore]
    public T MissingValue => _missingValue;

    /// <summary>
    /// Gets the computed statistics for each feature.
    /// </summary>
    [JsonIgnore]
    public Vector<T>? Statistics => _statistics;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    /// <remarks>
    /// Inverse transform is not supported because we don't know which values were originally missing.
    /// </remarks>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="SimpleImputer{T}"/>.
    /// </summary>
    /// <param name="strategy">The imputation strategy. Defaults to Mean.</param>
    /// <param name="fillValue">The fill value for Constant strategy. Ignored for other strategies.</param>
    /// <param name="columnIndices">The column indices to impute, or null for all columns.</param>
    public SimpleImputer(
        ImputationStrategy strategy = ImputationStrategy.Mean,
        T? fillValue = default,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        _strategy = strategy;
        _fillValue = fillValue;
        _missingValue = NumOps.FromDouble(double.NaN);
        // Note: For value types with unconstrained generics, fillValue defaults to
        // default(T) when not explicitly provided. The nullable pattern (T?) doesn't
        // work as Nullable<T> for unconstrained generics, so if no fillValue is
        // provided, the imputer falls back to NumOps.Zero in the transform logic.
    }

    /// <summary>
    /// Computes the statistics for each feature from the training data.
    /// </summary>
    /// <param name="data">The training data matrix where each column is a feature.</param>
    protected override void FitCore(Matrix<T> data)
    {
        int numColumns = data.Columns;
        var columnsToProcess = GetColumnsToProcess(numColumns);
        var mask = ScalerTensorOps.ColumnMask<T>(numColumns, columnsToProcess);
        T emptyValue = _strategy == ImputationStrategy.Constant && _fillValue is not null ? _fillValue : NumOps.Zero;
        if (data.Rows == 0 || numColumns == 0)
        {
            _statistics = ScalerTensorOps.MaskedParameters(
                mask, PreprocessingTensorOps.Filled(new[] { numColumns }, emptyValue), NumOps.Zero);
            return;
        }

        // Statistics of the non-missing (non-NaN) values; a column with no observed value gets 0 (the fill
        // value for the constant strategy), like scikit-learn with keep_empty_features=True.
        var x = Tensor<T>.FromMatrix(data);
        Tensor<T> stats;
        switch (_strategy)
        {
            case ImputationStrategy.Mean:
                stats = ScalerTensorOps.NanColumnMean(x);
                break;
            case ImputationStrategy.Median:
                stats = Engine.TensorNanQuantile(x, 0.5, 0, false, QuantileInterpolation.Linear);
                break;
            case ImputationStrategy.MostFrequent:
                stats = ComputeMostFrequent(data, columnsToProcess);
                break;
            case ImputationStrategy.Constant:
                stats = PreprocessingTensorOps.Filled(new[] { numColumns }, _fillValue ?? NumOps.Zero);
                break;
            default:
                throw new ArgumentException($"Unknown imputation strategy: {_strategy}");
        }

        var empty = Engine.TensorEquals(ScalerTensorOps.NanColumnCount(x), NumOps.Zero);
        stats = Engine.TensorWhere(empty, PreprocessingTensorOps.Filled(new[] { numColumns }, emptyValue), stats);
        _statistics = ScalerTensorOps.MaskedParameters(mask, stats, NumOps.Zero);
    }

    /// <summary>
    /// The most frequent non-missing value of each processed column; ties go to the smallest value, as in
    /// scikit-learn's SimpleImputer(strategy="most_frequent"). Columns with no observed value get 0.
    /// </summary>
    private Tensor<T> ComputeMostFrequent(Matrix<T> data, int[] columnsToProcess)
    {
        var modes = new Tensor<T>(new[] { data.Columns });
        foreach (var col in columnsToProcess)
        {
            var column = PreprocessingTensorOps.Column(data, col);
            var observed = PreprocessingTensorOps.MaskPositions(PreprocessingTensorOps.Not(ScalerTensorOps.NanMask(column)));
            if (observed.Length == 0)
            {
                continue;
            }

            var values = Engine.TensorIndexSelect(column, new Tensor<int>(observed, new[] { observed.Length }), 0);
            var (mode, _) = Engine.TensorMode(values);
            var single = PreprocessingTensorOps.Filled(new[] { 1 }, mode);
            modes = Engine.TensorSetSlice(modes, single, new[] { col });
        }

        return modes;
    }

    /// <summary>
    /// Transforms the data by imputing missing values.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The data with missing values imputed.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_statistics is null)
        {
            throw new InvalidOperationException("Imputer has not been fitted.");
        }

        ScalerTensorOps.EnsureColumnCount(data, _statistics.Length);
        int numRows = data.Rows;
        int numColumns = data.Columns;
        if (numRows == 0 || numColumns == 0)
        {
            return new Matrix<T>(numRows, numColumns);
        }

        // Missing (NaN) entries of the processed columns take the column statistic.
        var x = Tensor<T>.FromMatrix(data);
        var mask = ScalerTensorOps.ColumnMask<T>(numColumns, GetColumnsToProcess(numColumns));
        var fill = Engine.TensorMultiply(ScalerTensorOps.NanMask(x), ScalerTensorOps.BroadcastRow(mask, numRows));
        return Engine.TensorWhere(fill, ScalerTensorOps.BroadcastRow(_statistics, numRows), x).ToMatrix();
    }

    /// <summary>
    /// Inverse transformation is not supported for imputation.
    /// </summary>
    /// <param name="data">The imputed data.</param>
    /// <returns>Never returns - always throws.</returns>
    /// <exception cref="NotSupportedException">Always thrown because we don't track which values were missing.</exception>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException(
            "SimpleImputer does not support inverse transformation. " +
            "We don't track which values were originally missing.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    /// <param name="inputFeatureNames">The input feature names.</param>
    /// <returns>The same feature names (SimpleImputer doesn't change number of features).</returns>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }
}
