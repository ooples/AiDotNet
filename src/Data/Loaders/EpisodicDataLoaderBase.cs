using System.Runtime.CompilerServices;
using System.Threading.Channels;
using AiDotNet.Data.Structures;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Data.Loaders;

/// <summary>
/// Provides a base implementation for episodic data loaders with common functionality for N-way K-shot meta-learning.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type for tasks (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;, double[]).</typeparam>
/// <typeparam name="TOutput">The output data type for tasks (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;, double[]).</typeparam>
/// <remarks>
/// <para>
/// This abstract class implements the IEpisodicDataLoader interface and provides common functionality
/// for episodic task sampling. It handles dataset validation, class-to-indices preprocessing, and
/// parameter validation while allowing derived classes to focus on implementing specific sampling strategies.
/// </para>
/// <para><b>For Beginners:</b> This is the foundation that all episodic data loaders build upon.
///
/// Think of it like a template for creating meta-learning tasks:
/// - It handles common tasks (validating inputs, organizing data by class, checking requirements)
/// - Specific loaders just implement how they sample tasks
/// - This ensures all episodic loaders work consistently and follow SOLID principles
///
/// The base class takes care of:
/// - Validating that your dataset has enough classes and examples
/// - Building an efficient index (class to example indices) for fast sampling
/// - Storing configuration (N-way, K-shot, query shots)
/// - Providing protected access to the dataset and configuration
/// </para>
/// </remarks>
public abstract class EpisodicDataLoaderBase<T, TInput, TOutput> :
    DataLoaderBase<T>,
    IEpisodicDataLoader<T, TInput, TOutput>
{
    /// <summary>
    /// The feature matrix containing all examples.
    /// </summary>
    protected readonly Matrix<T> DatasetX;

    /// <summary>
    /// The label vector containing class labels for all examples.
    /// </summary>
    protected readonly Vector<T> DatasetY;

    /// <summary>
    /// The number of classes per task (N in N-way).
    /// </summary>
    protected readonly int _nWay;

    /// <summary>
    /// The number of support examples per class (K in K-shot).
    /// </summary>
    protected readonly int _kShot;

    /// <summary>
    /// The number of query examples per class.
    /// </summary>
    protected readonly int _queryShots;

    /// <summary>
    /// Random number generator for task sampling.
    /// </summary>
    protected Random RandomInstance;

    /// <summary>
    /// Lock object for thread-safe access to RandomInstance during batch generation.
    /// </summary>
    private readonly object _randomLock = new object();

    /// <summary>
    /// Mapping from class label to list of example indices for that class.
    /// </summary>
    protected readonly Dictionary<int, List<int>> ClassToIndices;

    /// <summary>
    /// Array of all available class labels in the dataset.
    /// </summary>
    protected readonly int[] _availableClasses;

    /// <summary>
    /// Provides mathematical operations for the numeric type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private int _batchSize = 1;
    private int _currentBatchStartIndex;

    /// <inheritdoc/>
    public override string Name => "EpisodicDataLoader";

    /// <inheritdoc/>
    public override string Description => "Episodic data loader for N-way K-shot meta-learning";

    /// <inheritdoc/>
    public override int TotalCount => DatasetX.Rows;

    /// <inheritdoc/>
    public int NWay => _nWay;

    /// <inheritdoc/>
    public int KShot => _kShot;

    /// <inheritdoc/>
    public int QueryShots => _queryShots;

    /// <inheritdoc/>
    public int AvailableClasses => _availableClasses.Length;

    /// <inheritdoc/>
    public override int BatchSize
    {
        get => _batchSize;
        set => _batchSize = Math.Max(1, value);
    }

    /// <inheritdoc/>
    public bool HasNext => _currentBatchStartIndex < TotalCount;

    /// <summary>
    /// Initializes a new instance of the EpisodicDataLoaderBase class with industry-standard defaults.
    /// </summary>
    /// <param name="datasetX">The feature matrix where each row is an example. Shape: [num_examples, num_features].</param>
    /// <param name="datasetY">The label vector containing class labels for each example. Length: num_examples.</param>
    /// <param name="nWay">The number of unique classes per task. Default is 5 (standard in meta-learning).</param>
    /// <param name="kShot">The number of support examples per class. Default is 5 (balanced difficulty).</param>
    /// <param name="queryShots">The number of query examples per class. Default is 15 (3x kShot).</param>
    /// <param name="seed">Optional random seed for reproducible task sampling. If null, uses a time-based seed.</param>
    /// <exception cref="ArgumentNullException">Thrown when datasetX or datasetY is null.</exception>
    /// <exception cref="ArgumentException">Thrown when dimensions are invalid or dataset is too small.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This constructor sets up the base infrastructure for episodic sampling.
    ///
    /// It performs several important tasks:
    /// 1. Validates all inputs to catch configuration errors early
    /// 2. Builds an index mapping each class to its example indices for fast lookup
    /// 3. Verifies the dataset has enough classes and examples per class
    /// 4. Stores all configuration for use by derived classes
    ///
    /// After construction, the derived class can use the protected fields to implement
    /// its specific sampling strategy.
    /// </para>
    /// </remarks>
    protected EpisodicDataLoaderBase(
        Matrix<T> datasetX,
        Vector<T> datasetY,
        int nWay = 5,        // Default: 5-way (standard in meta-learning literature)
        int kShot = 5,       // Default: 5-shot (balanced between 1-shot and 10-shot)
        int queryShots = 15, // Default: 15 queries (3x kShot is common practice)
        int? seed = null)
    {
        // Validate inputs
        ValidateDataset(datasetX, datasetY);
        ValidateConfiguration(nWay, kShot, queryShots);

        DatasetX = datasetX;
        DatasetY = datasetY;
        _nWay = nWay;
        _kShot = kShot;
        _queryShots = queryShots;
        RandomInstance = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();

        // Preprocess: Build class-to-indices mapping
        ClassToIndices = BuildClassIndex(datasetY);
        _availableClasses = ClassToIndices.Keys.ToArray();

        // Validate dataset has sufficient classes and examples
        ValidateDatasetSize(ClassToIndices, nWay, kShot, queryShots);
    }

    /// <inheritdoc/>
    public MetaLearningTask<T, TInput, TOutput> GetNextTask()
    {
        return GetNextTaskCore();
    }

    /// <inheritdoc/>
    public IReadOnlyList<MetaLearningTask<T, TInput, TOutput>> GetTaskBatch(int numTasks)
    {
        var tasks = new List<MetaLearningTask<T, TInput, TOutput>>(numTasks);
        for (int i = 0; i < numTasks; i++)
        {
            tasks.Add(GetNextTask());
        }
        return tasks.AsReadOnly();
    }

    /// <inheritdoc/>
    public void SetSeed(int seed)
    {
        RandomInstance = RandomHelper.CreateSeededRandom(seed);
    }

    /// <inheritdoc/>
    public MetaLearningTask<T, TInput, TOutput> GetNextBatch()
    {
        if (!HasNext)
        {
            throw new InvalidOperationException("No more batches available. Call Reset() to start over.");
        }

        var task = GetNextTask();
        _currentBatchStartIndex += BatchSize;
        return task;
    }

    /// <inheritdoc/>
    public bool TryGetNextBatch(out MetaLearningTask<T, TInput, TOutput> batch)
    {
        if (!HasNext)
        {
            batch = new MetaLearningTask<T, TInput, TOutput>();
            return false;
        }

        batch = GetNextBatch();
        return true;
    }

    /// <inheritdoc/>
    protected override Task LoadDataCoreAsync(CancellationToken cancellationToken)
    {
        // Data is provided via constructor, so nothing to load asynchronously
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    protected override void UnloadDataCore()
    {
        // Data is managed externally via constructor, nothing to unload
    }

    /// <inheritdoc/>
    protected override void OnReset()
    {
        _currentBatchStartIndex = 0;
    }

    /// <summary>
    /// Core task sampling logic to be implemented by derived classes.
    /// </summary>
    /// <returns>A MetaLearningTask with sampled support and query sets.</returns>
    /// <remarks>
    /// <para><b>For Implementers:</b> This is where you implement your specific task sampling strategy.
    ///
    /// You have access to:
    /// - DatasetX and DatasetY: The full dataset
    /// - ClassToIndices: Fast lookup of examples by class
    /// - _availableClasses: Array of all class labels
    /// - _nWay, _kShot, _queryShots: Task configuration
    /// - RandomInstance: For randomized sampling
    /// - NumOps: For numeric operations
    ///
    /// Your implementation should:
    /// 1. Select NWay classes
    /// 2. Sample KShot + QueryShots examples per class
    /// 3. Split into support and query sets
    /// 4. Build and return a MetaLearningTask
    /// </para>
    /// </remarks>
    protected abstract MetaLearningTask<T, TInput, TOutput> GetNextTaskCore();

    /// <summary>
    /// Validates the dataset inputs.
    /// </summary>
    private void ValidateDataset(Matrix<T> datasetX, Vector<T> datasetY)
    {
        if (datasetX == null)
        {
            throw new ArgumentNullException(nameof(datasetX), "Dataset features cannot be null");
        }

        if (datasetY == null)
        {
            throw new ArgumentNullException(nameof(datasetY), "Dataset labels cannot be null");
        }

        if (datasetX.Rows != datasetY.Length)
        {
            throw new ArgumentException(
                $"Number of examples in features ({datasetX.Rows}) must match number of labels ({datasetY.Length})",
                nameof(datasetX));
        }
    }

    /// <summary>
    /// Validates the N-way K-shot configuration.
    /// </summary>
    private void ValidateConfiguration(int nWay, int kShot, int queryShots)
    {
        if (nWay < 2)
        {
            throw new ArgumentException("nWay must be at least 2 (need at least 2 classes per task)", nameof(nWay));
        }

        if (kShot < 1)
        {
            throw new ArgumentException("kShot must be at least 1 (need at least 1 support example per class)", nameof(kShot));
        }

        if (queryShots < 1)
        {
            throw new ArgumentException("queryShots must be at least 1 (need at least 1 query example per class)", nameof(queryShots));
        }
    }

    /// <summary>
    /// Builds a dictionary mapping each class label to its example indices.
    /// </summary>
    private Dictionary<int, List<int>> BuildClassIndex(Vector<T> datasetY)
    {
        var classToIndices = new Dictionary<int, List<int>>();

        for (int i = 0; i < datasetY.Length; i++)
        {
            int classLabel = NumOps.ToInt32(datasetY[i]);

            if (!classToIndices.ContainsKey(classLabel))
            {
                classToIndices[classLabel] = new List<int>();
            }

            classToIndices[classLabel].Add(i);
        }

        return classToIndices;
    }

    /// <summary>
    /// Validates that the dataset has sufficient classes and examples per class.
    /// </summary>
    private void ValidateDatasetSize(Dictionary<int, List<int>> classToIndices, int nWay, int kShot, int queryShots)
    {
        var availableClasses = classToIndices.Keys.ToArray();

        // Validate dataset has enough classes
        if (availableClasses.Length < nWay)
        {
            throw new ArgumentException(
                $"Dataset has only {availableClasses.Length} classes, but nWay={nWay} requires at least {nWay} classes",
                nameof(nWay));
        }

        // Validate each class has enough examples
        int requiredExamplesPerClass = kShot + queryShots;
        var insufficientClasses = classToIndices
            .Where(kvp => kvp.Value.Count < requiredExamplesPerClass)
            .Select(kvp => $"Class {kvp.Key}: {kvp.Value.Count} examples")
            .ToList();

        if (insufficientClasses.Count > 0)
        {
            throw new ArgumentException(
                $"Some classes have insufficient examples. Need at least {requiredExamplesPerClass} " +
                $"(kShot={kShot} + queryShots={queryShots}) per class. Insufficient classes: " +
                string.Join(", ", insufficientClasses),
                nameof(nWay));
        }
    }

    /// <summary>
    /// Builds a MetaLearningTask from lists of examples and labels.
    /// </summary>
    protected MetaLearningTask<T, TInput, TOutput> BuildMetaLearningTask(
        List<Vector<T>> supportExamples,
        List<T> supportLabels,
        List<Vector<T>> queryExamples,
        List<T> queryLabels)
    {
        // Build matrices from examples
        int numFeatures = DatasetX.Columns;
        var supportMatrix = new Matrix<T>(supportExamples.Count, numFeatures);
        var supportVector = new Vector<T>(supportLabels.Count);

        for (int i = 0; i < supportExamples.Count; i++)
        {
            for (int j = 0; j < numFeatures; j++)
            {
                supportMatrix[i, j] = supportExamples[i][j];
            }
            supportVector[i] = supportLabels[i];
        }

        var queryMatrix = new Matrix<T>(queryExamples.Count, numFeatures);
        var queryVector = new Vector<T>(queryLabels.Count);

        for (int i = 0; i < queryExamples.Count; i++)
        {
            for (int j = 0; j < numFeatures; j++)
            {
                queryMatrix[i, j] = queryExamples[i][j];
            }
            queryVector[i] = queryLabels[i];
        }

        // Convert to TInput/TOutput using type checking
        TInput supportX = ConvertMatrixToInput(supportMatrix);
        TOutput supportY = ConvertVectorToOutput(supportVector);
        TInput queryX = ConvertMatrixToInput(queryMatrix);
        TOutput queryY = ConvertVectorToOutput(queryVector);

        return new MetaLearningTask<T, TInput, TOutput>
        {
            SupportSetX = supportX,
            SupportSetY = supportY,
            QuerySetX = queryX,
            QuerySetY = queryY
        };
    }

    /// <summary>
    /// Converts a Matrix to TInput type using the same pattern as ConversionsHelper.
    /// </summary>
    private static TInput ConvertMatrixToInput(Matrix<T> matrix)
    {
        if (typeof(TInput) == typeof(Matrix<T>))
        {
            return (TInput)(object)matrix;
        }

        if (typeof(TInput) == typeof(Tensor<T>))
        {
            // Use Tensor.FromMatrix for efficient conversion
            var tensor = Tensor<T>.FromRowMatrix(matrix);
            return (TInput)(object)tensor;
        }

        throw new NotSupportedException(
            $"Conversion from Matrix<T> to {typeof(TInput).Name} is not supported. " +
            $"Supported types: Matrix<T>, Tensor<T>");
    }

    /// <summary>
    /// Converts a Vector to TOutput type using the same pattern as ConversionsHelper.
    /// </summary>
    private static TOutput ConvertVectorToOutput(Vector<T> vector)
    {
        if (typeof(TOutput) == typeof(Vector<T>))
        {
            return (TOutput)(object)vector;
        }

        if (typeof(TOutput) == typeof(Tensor<T>))
        {
            // Use Tensor.FromVector for efficient conversion
            return (TOutput)(object)Tensor<T>.FromVector(vector);
        }

        throw new NotSupportedException(
            $"Conversion from Vector<T> to {typeof(TOutput).Name} is not supported. " +
            $"Supported types: Vector<T>, Tensor<T>");
    }

    #region PyTorch-Style Batch Iteration for Meta-Learning

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// For episodic data loaders, each "batch" is a meta-learning task. This method yields
    /// the specified number of tasks using lazy evaluation.
    /// </para>
    /// <para><b>For Beginners:</b> In meta-learning, each batch is a complete learning task:
    ///
    /// <code>
    /// foreach (var task in episodicLoader.GetBatches(batchSize: 10))
    /// {
    ///     // Each task has support and query sets
    ///     var (supportX, supportY) = (task.SupportSetX, task.SupportSetY);
    ///     var (queryX, queryY) = (task.QuerySetX, task.QuerySetY);
    ///
    ///     // Train inner loop on support, evaluate on query
    ///     model.Adapt(supportX, supportY);
    ///     var loss = model.Evaluate(queryX, queryY);
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    public virtual IEnumerable<MetaLearningTask<T, TInput, TOutput>> GetBatches(
        int? batchSize = null,
        bool shuffle = true,
        bool dropLast = false,
        int? seed = null)
    {
        // Validate that shuffle and dropLast are default values - episodic loaders
        // generate inherently random tasks, so these parameters don't apply
        if (!shuffle)
        {
            throw new ArgumentException(
                "Episodic data loaders generate inherently random tasks. The shuffle parameter must be true (default).",
                nameof(shuffle));
        }

        if (dropLast)
        {
            throw new ArgumentException(
                "Episodic data loaders generate tasks on-demand. The dropLast parameter must be false (default).",
                nameof(dropLast));
        }

        int numTasks = batchSize ?? BatchSize;
        if (numTasks < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(batchSize), "Number of tasks must be at least 1.");
        }

        // Use lock for thread-safe random access when seed is provided
        // This ensures concurrent enumerations don't interfere with each other
        lock (_randomLock)
        {
            if (seed.HasValue)
            {
                SetSeed(seed.Value);
            }

            // Yield tasks using lazy evaluation
            for (int i = 0; i < numTasks; i++)
            {
                yield return GetNextTaskCore();
            }
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// Asynchronously generates meta-learning tasks with prefetching support.
    /// Tasks are generated in the background while the current task is being processed.
    /// </para>
    /// <para>
    /// <b>Thread Safety:</b> Task generation uses locking to ensure thread-safe
    /// access to the random number generator. This prevents interference between
    /// concurrent enumerations.
    /// </para>
    /// </remarks>
    public virtual async IAsyncEnumerable<MetaLearningTask<T, TInput, TOutput>> GetBatchesAsync(
        int? batchSize = null,
        bool shuffle = true,
        bool dropLast = false,
        int? seed = null,
        int prefetchCount = 2,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        // Validate that shuffle and dropLast are default values - episodic loaders
        // generate inherently random tasks, so these parameters don't apply
        if (!shuffle)
        {
            throw new ArgumentException(
                "Episodic data loaders generate inherently random tasks. The shuffle parameter must be true (default).",
                nameof(shuffle));
        }

        if (dropLast)
        {
            throw new ArgumentException(
                "Episodic data loaders generate tasks on-demand. The dropLast parameter must be false (default).",
                nameof(dropLast));
        }

        if (prefetchCount < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(prefetchCount), "Prefetch count must be at least 1.");
        }

        int numTasks = batchSize ?? BatchSize;
        if (numTasks < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(batchSize), "Number of tasks must be at least 1.");
        }

        // Create bounded channel for prefetching
        var channel = Channel.CreateBounded<MetaLearningTask<T, TInput, TOutput>>(
            new BoundedChannelOptions(prefetchCount)
            {
                FullMode = BoundedChannelFullMode.Wait,
                SingleReader = true,
                SingleWriter = true
            });

        // Start producer task with thread-safe random access
        var producerTask = Task.Run(async () =>
        {
            try
            {
                // Use lock for thread-safe random access
                lock (_randomLock)
                {
                    if (seed.HasValue)
                    {
                        SetSeed(seed.Value);
                    }

                    for (int i = 0; i < numTasks; i++)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        var task = GetNextTaskCore();
                        // Note: We hold the lock while generating all tasks to ensure
                        // reproducibility when a seed is provided. This is acceptable
                        // because task generation is typically fast compared to model training.
                        channel.Writer.TryWrite(task);
                    }
                }
            }
            finally
            {
                channel.Writer.Complete();
            }
        }, cancellationToken);

        // Consume tasks (net471 compatible - no ReadAllAsync)
        while (await channel.Reader.WaitToReadAsync(cancellationToken))
        {
            while (channel.Reader.TryRead(out var task))
            {
                yield return task;
            }
        }

        // Ensure producer completed without errors
        await producerTask;
    }

    #endregion
}
