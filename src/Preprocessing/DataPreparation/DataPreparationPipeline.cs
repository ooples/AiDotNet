using AiDotNet.Preprocessing.DataPreparation.Splitting.Basic;
using AiDotNet.Preprocessing.DataPreparation.Splitting.CrossValidation;
using AiDotNet.Preprocessing.DataPreparation.Splitting.TimeSeries;
using AiDotNet.Preprocessing.DataPreparation.Splitting.Stratified;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation;

/// <summary>
/// Chains multiple row operations into a sequential data preparation pipeline.
/// </summary>
/// <remarks>
/// <para>
/// DataPreparationPipeline handles operations that change the number of rows in a dataset,
/// such as outlier removal, data augmentation (SMOTE), and data splitting. These operations
/// must process both features (X) and labels (y) together to maintain alignment.
/// </para>
/// <para>
/// <b>Data Preparation vs Data Preprocessing:</b>
/// <list type="bullet">
/// <item><b>Data Preparation (this pipeline):</b> Changes the NUMBER of rows - outlier removal,
/// augmentation, splitting. Only happens during training.</item>
/// <item><b>Data Preprocessing:</b> Transforms values WITHOUT changing row count - scaling,
/// encoding. Happens during both training and prediction.</item>
/// </list>
/// </para>
/// <para>
/// <b>Pipeline Flow:</b>
/// <code>
/// Raw Data → [Outlier Removal] → [Augmentation] → [Splitting] → Train/Val/Test Sets
/// </code>
/// </para>
/// <para>
/// <b>For Beginners:</b> Before training a model, you might want to:
/// - Remove outliers (unusual data points that could confuse the model)
/// - Add synthetic samples (SMOTE) to balance classes
/// - Split data into train/validation/test sets
///
/// This pipeline handles all these operations in sequence, making sure your features
/// and labels stay properly aligned.
/// </para>
/// <para>
/// <b>Usage:</b> Users interact with this through AiModelBuilder.ConfigureDataPreparation().
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class DataPreparationPipeline<T>
{
    private readonly List<(string Name, IRowOperation<T> Operation)> _operations;
    private IDataSplitter<T>? _splitter;
    private bool _isFitted;

    /// <summary>
    /// Gets whether this pipeline has been fitted to data.
    /// </summary>
    public bool IsFitted => _isFitted;

    /// <summary>
    /// Gets the number of operations in the pipeline.
    /// </summary>
    public int Count => _operations.Count;

    /// <summary>
    /// Gets the named operations in the pipeline.
    /// </summary>
    public IReadOnlyList<(string Name, IRowOperation<T> Operation)> Operations => _operations.AsReadOnly();

    /// <summary>
    /// Gets the configured data splitter, or null if none is set.
    /// </summary>
    public IDataSplitter<T>? Splitter => _splitter;

    /// <summary>
    /// Gets whether this pipeline has a splitter configured.
    /// </summary>
    public bool HasSplitter => _splitter != null;

    /// <summary>
    /// Creates a new empty data preparation pipeline.
    /// </summary>
    public DataPreparationPipeline()
    {
        _operations = new List<(string, IRowOperation<T>)>();
        _splitter = null;
        _isFitted = false;
    }

    /// <summary>
    /// Adds a row operation to the pipeline.
    /// </summary>
    /// <param name="operation">The row operation to add.</param>
    /// <returns>This pipeline for method chaining.</returns>
    public DataPreparationPipeline<T> Add(IRowOperation<T> operation)
    {
        return Add($"step_{_operations.Count}", operation);
    }

    /// <summary>
    /// Adds a named row operation to the pipeline.
    /// </summary>
    /// <param name="name">The name for this operation.</param>
    /// <param name="operation">The row operation to add.</param>
    /// <returns>This pipeline for method chaining.</returns>
    public DataPreparationPipeline<T> Add(string name, IRowOperation<T> operation)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("Operation name cannot be null or whitespace.", nameof(name));
        }

        if (operation is null)
        {
            throw new ArgumentNullException(nameof(operation));
        }

        // Check for duplicate names
        foreach (var op in _operations)
        {
            if (op.Name.Equals(name, StringComparison.Ordinal))
            {
                throw new ArgumentException($"An operation with name '{name}' already exists.", nameof(name));
            }
        }

        _operations.Add((name, operation));
        _isFitted = false;

        return this;
    }

    /// <summary>
    /// Gets an operation by name.
    /// </summary>
    /// <param name="name">The operation name.</param>
    /// <returns>The operation, or null if not found.</returns>
    public IRowOperation<T>? GetOperation(string name)
    {
        foreach (var op in _operations)
        {
            if (op.Name.Equals(name, StringComparison.Ordinal))
            {
                return op.Operation;
            }
        }
        return null;
    }

    #region Splitter Configuration

    /// <summary>
    /// Configures a custom data splitter for this pipeline.
    /// </summary>
    /// <param name="splitter">The data splitter to use.</param>
    /// <returns>This pipeline for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The splitter divides your data into train/validation/test sets.
    /// You can use one of the many built-in splitters (K-Fold, Stratified, Time Series, etc.)
    /// or create your own.
    /// </para>
    /// </remarks>
    public DataPreparationPipeline<T> WithSplitter(IDataSplitter<T> splitter)
    {
        _splitter = splitter ?? throw new ArgumentNullException(nameof(splitter));
        _isFitted = false;
        return this;
    }

    /// <summary>
    /// Configures a simple train/test split.
    /// </summary>
    /// <param name="testSize">Proportion for test set. Default is 0.2 (20%).</param>
    /// <param name="shuffle">Whether to shuffle before splitting. Default is true.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    /// <returns>This pipeline for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// <b>Industry Standard:</b> 80/20 split is the most common default.
    /// </para>
    /// </remarks>
    public DataPreparationPipeline<T> WithTrainTestSplit(double testSize = 0.2, bool shuffle = true, int randomSeed = 42)
    {
        _splitter = new TrainTestSplitter<T>(testSize, shuffle, randomSeed);
        _isFitted = false;
        return this;
    }

    /// <summary>
    /// Configures a three-way train/validation/test split.
    /// </summary>
    /// <param name="trainSize">Proportion for training. Default is 0.7 (70%).</param>
    /// <param name="validationSize">Proportion for validation. Default is 0.15 (15%).</param>
    /// <param name="shuffle">Whether to shuffle before splitting. Default is true.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    /// <returns>This pipeline for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// <b>Industry Standard:</b> 70/15/15 is common for medium-sized datasets.
    /// </para>
    /// </remarks>
    public DataPreparationPipeline<T> WithTrainValTestSplit(
        double trainSize = 0.7,
        double validationSize = 0.15,
        bool shuffle = true,
        int randomSeed = 42)
    {
        _splitter = new TrainValTestSplitter<T>(trainSize, validationSize, shuffle, randomSeed);
        _isFitted = false;
        return this;
    }

    /// <summary>
    /// Configures K-Fold cross-validation.
    /// </summary>
    /// <param name="k">Number of folds. Default is 5.</param>
    /// <param name="shuffle">Whether to shuffle before splitting. Default is true.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    /// <returns>This pipeline for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// <b>Industry Standard:</b> k=5 for large datasets, k=10 for smaller datasets.
    /// </para>
    /// </remarks>
    public DataPreparationPipeline<T> WithKFold(int k = 5, bool shuffle = true, int randomSeed = 42)
    {
        _splitter = new KFoldSplitter<T>(k, shuffle, randomSeed);
        _isFitted = false;
        return this;
    }

    /// <summary>
    /// Configures Stratified K-Fold cross-validation that preserves class distribution.
    /// </summary>
    /// <param name="k">Number of folds. Default is 5.</param>
    /// <param name="shuffle">Whether to shuffle before splitting. Default is true.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    /// <returns>This pipeline for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// <b>Industry Standard:</b> For classification tasks, ALWAYS prefer stratified splits.
    /// </para>
    /// </remarks>
    public DataPreparationPipeline<T> WithStratifiedKFold(int k = 5, bool shuffle = true, int randomSeed = 42)
    {
        _splitter = new StratifiedKFoldSplitter<T>(k, shuffle, randomSeed);
        _isFitted = false;
        return this;
    }

    /// <summary>
    /// Configures a stratified train/test split for classification.
    /// </summary>
    /// <param name="testSize">Proportion for test set. Default is 0.2 (20%).</param>
    /// <param name="shuffle">Whether to shuffle before splitting. Default is true.</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    /// <returns>This pipeline for method chaining.</returns>
    public DataPreparationPipeline<T> WithStratifiedSplit(double testSize = 0.2, bool shuffle = true, int randomSeed = 42)
    {
        _splitter = new StratifiedTrainTestSplitter<T>(testSize, shuffle, randomSeed);
        _isFitted = false;
        return this;
    }

    /// <summary>
    /// Configures a time series split with expanding window (no shuffling).
    /// </summary>
    /// <param name="nSplits">Number of splits. Default is 5.</param>
    /// <param name="gap">Samples to skip between train and test (prevents leakage). Default is 0.</param>
    /// <returns>This pipeline for method chaining.</returns>
    /// <remarks>
    /// <para>
    /// <b>Important:</b> Time series data must NOT be shuffled. This splitter respects temporal order.
    /// </para>
    /// </remarks>
    public DataPreparationPipeline<T> WithTimeSeriesSplit(int nSplits = 5, int gap = 0)
    {
        _splitter = new TimeSeriesSplitter<T>(nSplits, gap: gap);
        _isFitted = false;
        return this;
    }

    #endregion

    /// <summary>
    /// Fits all operations and applies row modifications to the data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Each operation is fitted and applied in sequence. The output of one operation
    /// becomes the input to the next.
    /// </para>
    /// <para>
    /// <b>Important:</b> This is the ONLY time row operations are applied. There is no
    /// separate Transform method because row operations cannot be applied during prediction.
    /// </para>
    /// </remarks>
    /// <param name="X">The feature matrix.</param>
    /// <param name="y">The label vector.</param>
    /// <returns>A tuple containing the modified (X, y) after all operations.</returns>
    public (Matrix<T> X, Vector<T> y) FitResample(Matrix<T> X, Vector<T> y)
    {
        if (X is null)
        {
            throw new ArgumentNullException(nameof(X));
        }

        if (y is null)
        {
            throw new ArgumentNullException(nameof(y));
        }

        if (X.Rows != y.Length)
        {
            throw new ArgumentException(
                $"X has {X.Rows} rows but y has {y.Length} elements. They must match.",
                nameof(y));
        }

        // If no operations, return data unchanged
        if (_operations.Count == 0)
        {
            _isFitted = true;
            return (X, y);
        }

        // Apply each operation in sequence
        Matrix<T> currentX = X;
        Vector<T> currentY = y;

        foreach (var (name, operation) in _operations)
        {
            (currentX, currentY) = operation.FitResample(currentX, currentY);
        }

        _isFitted = true;
        return (currentX, currentY);
    }

    /// <summary>
    /// Applies row operations and then splits the data.
    /// </summary>
    /// <param name="X">The feature matrix.</param>
    /// <param name="y">The label vector.</param>
    /// <returns>The split result containing train/test (and optionally validation) sets.</returns>
    /// <exception cref="InvalidOperationException">If no splitter is configured.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method combines row operations (outlier removal, augmentation)
    /// with data splitting in a single call. The operations are applied first, then the data is split.
    /// </para>
    /// </remarks>
    public DataSplitResult<T> FitResampleAndSplit(Matrix<T> X, Vector<T> y)
    {
        if (_splitter is null)
        {
            throw new InvalidOperationException(
                "No splitter configured. Use WithSplitter(), WithTrainTestSplit(), WithKFold(), or similar methods first.");
        }

        // Apply row operations first
        var (preparedX, preparedY) = FitResample(X, y);

        // Then split
        return _splitter.Split(preparedX, preparedY);
    }

    /// <summary>
    /// Applies row operations and returns multiple splits (for cross-validation).
    /// </summary>
    /// <param name="X">The feature matrix.</param>
    /// <param name="y">The label vector.</param>
    /// <returns>An enumerable of split results, one for each fold/iteration.</returns>
    /// <exception cref="InvalidOperationException">If no splitter is configured.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this for cross-validation. It applies row operations once,
    /// then generates multiple train/test splits for you to evaluate.
    /// </para>
    /// </remarks>
    public IEnumerable<DataSplitResult<T>> FitResampleAndGetSplits(Matrix<T> X, Vector<T> y)
    {
        if (_splitter is null)
        {
            throw new InvalidOperationException(
                "No splitter configured. Use WithSplitter(), WithKFold(), or similar methods first.");
        }

        // Apply row operations first
        var (preparedX, preparedY) = FitResample(X, y);

        // Then generate splits
        return _splitter.GetSplits(preparedX, preparedY);
    }

    /// <summary>
    /// Clears all operations and splitter from the pipeline.
    /// </summary>
    public void Clear()
    {
        _operations.Clear();
        _splitter = null;
        _isFitted = false;
    }

    /// <summary>
    /// Gets a summary of the pipeline operations and splitter.
    /// </summary>
    /// <returns>A string describing all operations and the splitter configuration.</returns>
    public string GetSummary()
    {
        var lines = new List<string> { "DataPreparationPipeline:" };

        if (_operations.Count == 0 && _splitter is null)
        {
            lines.Add("  (empty - pass-through)");
            return string.Join(Environment.NewLine, lines);
        }

        // Operations
        if (_operations.Count > 0)
        {
            lines.Add("  Row Operations:");
            for (int i = 0; i < _operations.Count; i++)
            {
                var (name, op) = _operations[i];
                lines.Add($"    {i + 1}. [{name}] {op.Description}");
            }
        }
        else
        {
            lines.Add("  Row Operations: (none)");
        }

        // Splitter
        if (_splitter != null)
        {
            lines.Add($"  Splitter: {_splitter.Description}");
            if (_splitter.NumSplits > 1)
            {
                lines.Add($"    Generates {_splitter.NumSplits} splits");
            }
        }
        else
        {
            lines.Add("  Splitter: (not configured)");
        }

        return string.Join(Environment.NewLine, lines);
    }
}
