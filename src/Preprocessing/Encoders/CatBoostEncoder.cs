using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

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
public class CatBoostEncoder<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly double _prior;
    private readonly int _randomState;
    private readonly CatBoostHandleUnknown _handleUnknown;

    // Fitted parameters
    private Dictionary<int, Dictionary<double, (double Sum, int Count)>>? _categoryStats;
    private double _globalMean;
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
    public double GlobalMean => _globalMean;

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
        var columnsToProcess = GetColumnsToProcess(_nInputFeatures);

        // Compute global mean
        double targetSum = 0;
        for (int i = 0; i < target.Length; i++)
        {
            targetSum += NumOps.ToDouble(target[i]);
        }
        _globalMean = targetSum / target.Length;

        // Compute category statistics (for transform on new data)
        _categoryStats = new Dictionary<int, Dictionary<double, (double Sum, int Count)>>();

        foreach (int col in columnsToProcess)
        {
            var stats = new Dictionary<double, (double Sum, int Count)>();

            for (int i = 0; i < data.Rows; i++)
            {
                double categoryValue = NumOps.ToDouble(data[i, col]);
                double targetValue = NumOps.ToDouble(target[i]);

                if (!stats.TryGetValue(categoryValue, out var existing))
                {
                    existing = (0, 0);
                }

                stats[categoryValue] = (existing.Sum + targetValue, existing.Count + 1);
            }

            _categoryStats[col] = stats;
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

        int numRows = data.Rows;
        int numColumns = data.Columns;
        var result = new T[numRows, numColumns];
        var columnsToProcess = GetColumnsToProcess(numColumns);
        var processSet = new HashSet<int>(columnsToProcess);

        // Create shuffled order for randomization
        var random = RandomHelper.CreateSeededRandom(_randomState);
        var order = Enumerable.Range(0, numRows).ToArray();
        for (int i = numRows - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (order[i], order[j]) = (order[j], order[i]);
        }

        // For each column, apply ordered encoding
        foreach (int col in columnsToProcess)
        {
            // Running statistics per category
            var runningSum = new Dictionary<double, double>();
            var runningCount = new Dictionary<double, int>();

            foreach (int originalIdx in order)
            {
                double categoryValue = NumOps.ToDouble(data[originalIdx, col]);
                double targetValue = NumOps.ToDouble(target[originalIdx]);

                // Compute encoding using PREVIOUS samples only
                double sum = runningSum.GetValueOrDefault(categoryValue, 0);
                int count = runningCount.GetValueOrDefault(categoryValue, 0);

                // Ordered target encoding formula
                double encodedValue = (sum + _prior * _globalMean) / (count + _prior);
                result[originalIdx, col] = NumOps.FromDouble(encodedValue);

                // Update running statistics (for next samples)
                runningSum[categoryValue] = sum + targetValue;
                runningCount[categoryValue] = count + 1;
            }
        }

        // Copy non-processed columns
        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numColumns; j++)
            {
                if (!processSet.Contains(j))
                {
                    result[i, j] = data[i, j];
                }
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Transforms test data using full category statistics (for inference).
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_categoryStats is null)
        {
            throw new InvalidOperationException("Encoder has not been fitted.");
        }

        int numRows = data.Rows;
        int numColumns = data.Columns;
        var result = new T[numRows, numColumns];
        var columnsToProcess = GetColumnsToProcess(numColumns);
        var processSet = new HashSet<int>(columnsToProcess);

        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numColumns; j++)
            {
                if (!processSet.Contains(j))
                {
                    result[i, j] = data[i, j];
                    continue;
                }

                double categoryValue = NumOps.ToDouble(data[i, j]);

                if (_categoryStats.TryGetValue(j, out var colStats) &&
                    colStats.TryGetValue(categoryValue, out var stats))
                {
                    // Use full statistics for test data
                    double encodedValue = (stats.Sum + _prior * _globalMean) / (stats.Count + _prior);
                    result[i, j] = NumOps.FromDouble(encodedValue);
                }
                else
                {
                    // Unknown category
                    switch (_handleUnknown)
                    {
                        case CatBoostHandleUnknown.UseGlobalMean:
                            result[i, j] = NumOps.FromDouble(_globalMean);
                            break;
                        case CatBoostHandleUnknown.Error:
                            throw new ArgumentException($"Unknown category value: {categoryValue} in column {j}");
                        default:
                            result[i, j] = NumOps.FromDouble(_globalMean);
                            break;
                    }
                }
            }
        }

        return new Matrix<T>(result);
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
