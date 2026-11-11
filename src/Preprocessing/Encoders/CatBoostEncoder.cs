using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.Encoders;

/// <summary>
/// Encodes categorical features using ordered (CatBoost-style) target encoding.
/// </summary>
/// <remarks>
/// <para>
/// CatBoostEncoder applies an ordered approach to target encoding that prevents
/// target leakage by only using target values from previous samples when encoding.
/// This is the same technique used in the CatBoost gradient boosting library.
/// </para>
/// <para>
/// For each sample, the encoding is computed as:
/// (sum of targets for previous samples with same category + prior) / (count + 1)
/// </para>
/// <para><b>For Beginners:</b> Regular target encoding can "cheat" by using future
/// information. CatBoost encoding prevents this:
/// - When encoding row 10, it only uses data from rows 1-9
/// - Row 1 always gets the prior (global mean) since there's nothing before it
/// - This prevents overfitting and works better with gradient boosting
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class CatBoostEncoder<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly double _prior;
    private readonly int _randomState;
    private readonly CatBoostHandleUnknown _handleUnknown;

    // Fitted parameters
    private Dictionary<int, CategoryValueMap<T>>? _categoryStats;
    private T _globalMean = MathHelper.GetNumericOperations<T>().Zero;
    private int _nInputFeatures;

    /// <summary>
    /// Gets the prior value (regularization).
    /// </summary>
    public double Prior => _prior;

    /// <summary>
    /// Gets how unknown categories are handled.
    /// </summary>
    public CatBoostHandleUnknown HandleUnknown => _handleUnknown;

    /// <summary>
    /// Gets the global target mean.
    /// </summary>
    public T GlobalMean => _globalMean;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="CatBoostEncoder{T}"/>.
    /// </summary>
    /// <param name="prior">Prior weight for regularization. Higher values add more smoothing. Defaults to 1.0.</param>
    /// <param name="handleUnknown">How to handle unknown categories. Defaults to UseGlobalMean.</param>
    /// <param name="randomState">Random seed for shuffling order. Defaults to 0.</param>
    /// <param name="columnIndices">The column indices to encode, or null for all columns.</param>
    public CatBoostEncoder(
        double prior = 1.0,
        CatBoostHandleUnknown handleUnknown = CatBoostHandleUnknown.UseGlobalMean,
        int randomState = 0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (prior < 0)
        {
            throw new ArgumentException("Prior must be non-negative.", nameof(prior));
        }

        _prior = prior;
        _handleUnknown = handleUnknown;
        _randomState = randomState;
    }

    /// <summary>
    /// Fits the encoder (requires target via specialized Fit method).
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "CatBoostEncoder requires target values for fitting. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        T prior = NumOps.FromDouble(_prior);
        T priorMean = NumOps.Multiply(prior, _globalMean);

        // Full category statistics (for transform on new data): (sum + prior * mean) / (count + prior)
        _categoryStats = new Dictionary<int, CategoryValueMap<T>>();
        foreach (int col in columnsToProcess)
        {
            _categoryStats[col] = CategoryValueMap<T>.FromTargetStatistics(PreprocessingTensorOps.Column(data, col), y,
                (counts, sums) => engine.TensorDivide(engine.TensorAddScalar(sums, priorMean), engine.TensorAddScalar(counts, prior)));
        }

        IsFitted = true;
    }

    /// <summary>
    /// Fits and transforms using ordered target encoding (CatBoost style).
    /// </summary>
    /// <param name="data">The feature matrix.</param>
    /// <param name="target">The target values.</param>
    /// <returns>The encoded data with ordered target statistics.</returns>
    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return TransformWithTarget(data, target);
    }

    /// <summary>
    /// Transforms training data using ordered encoding (only uses previous samples).
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <param name="target">The target values (for ordered calculation).</param>
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
        T prior = NumOps.FromDouble(_prior);
        T priorMean = NumOps.Multiply(prior, _globalMean);

        // Create shuffled order for randomization; orderPosition[row] is when the row is visited
        var random = RandomHelper.CreateSeededRandom(_randomState);
        var order = Enumerable.Range(0, numRows).ToArray();
        for (int i = numRows - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (order[i], order[j]) = (order[j], order[i]);
        }

        var orderPosition = new int[numRows];
        for (int t = 0; t < numRows; t++)
            orderPosition[order[t]] = t;

        var columns = new List<Tensor<T>>(numColumns);
        for (int col = 0; col < numColumns; col++)
        {
            var column = PreprocessingTensorOps.Column(data, col);
            if (processSet.Contains(col) && numRows > 0)
            {
                column = OrderedEncoding(column, y, orderPosition, priorMean, prior);
            }

            columns.Add(column);
        }

        return PreprocessingTensorOps.StackColumns(columns, numRows);
    }

    /// <summary>
    /// Ordered target encoding of one column: each row is encoded from the target sum and count of the rows of
    /// its category visited before it, <c>(sum + prior * mean) / (count + prior)</c>.
    /// </summary>
    private Tensor<T> OrderedEncoding(Tensor<T> column, Tensor<T> y, int[] orderPosition, T priorMean, T prior)
    {
        var engine = Engine;
        int numRows = column.Length;
        var categories = PreprocessingTensorOps.SortedUnique(column);
        var category = PreprocessingTensorOps.LookupSorted(categories, column, out _).AsSpan().ToArray();

        // Visit the rows grouped by category and, within a category, in shuffled order
        var keys = new long[numRows];
        var visit = new int[numRows];
        for (int r = 0; r < numRows; r++)
        {
            keys[r] = (long)category[r] * numRows + orderPosition[r];
            visit[r] = r;
        }

        Array.Sort(keys, visit);

        // Start of each category run in the visit order, and each row's place in the visit order
        var runStart = new int[numRows];
        var previousCount = new int[numRows];
        var visitPosition = new int[numRows];
        for (int t = 0; t < numRows; t++)
        {
            bool newRun = t == 0 || category[visit[t]] != category[visit[t - 1]];
            runStart[t] = newRun ? t : runStart[t - 1];
            previousCount[t] = t - runStart[t];
            visitPosition[visit[t]] = t;
        }

        // Target sum of the earlier rows of the same category: exclusive prefix sum minus its value at the run start
        var visited = engine.TensorIndexSelect(y, new Tensor<int>(visit, new[] { numRows }), 0);
        var exclusive = engine.TensorSubtract(engine.TensorCumSum(visited, 0), visited);
        var previousSum = engine.TensorSubtract(
            exclusive, engine.TensorIndexSelect(exclusive, new Tensor<int>(runStart, new[] { numRows }), 0));

        // Ordered target encoding formula
        var encoded = engine.TensorDivide(
            engine.TensorAddScalar(previousSum, priorMean),
            engine.TensorAddScalar(new Tensor<int>(previousCount, new[] { numRows }).Cast<T>(), prior));
        return engine.TensorIndexSelect(encoded, new Tensor<int>(visitPosition, new[] { numRows }), 0);
    }

    /// <summary>
    /// Transforms new data using full category statistics.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The encoded data.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_categoryStats is null)
        {
            throw new InvalidOperationException("Encoder has not been fitted.");
        }

        var processSet = new HashSet<int>(GetColumnsToProcess(data.Columns));
        return CategoryValueMap<T>.EncodeColumns(data, processSet, _categoryStats, _ => _globalMean,
            _handleUnknown == CatBoostHandleUnknown.Error);
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("CatBoostEncoder does not support inverse transformation.");
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
public enum CatBoostHandleUnknown
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
