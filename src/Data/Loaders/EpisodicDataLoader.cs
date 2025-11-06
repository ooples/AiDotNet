using AiDotNet.Data.Abstractions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Data.Loaders;

/// <summary>
/// Provides episodic task sampling for N-way K-shot meta-learning scenarios.
/// </summary>
/// <typeparam name="T">The numeric data type used for features and labels (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// The EpisodicDataLoader transforms a standard supervised learning dataset (features + labels) into
/// a stream of meta-learning tasks. Each task contains a support set (for quick adaptation) and a
/// query set (for evaluation), enabling algorithms like MAML, Reptile, and SEAL to learn how to
/// learn from limited examples.
/// </para>
/// <para><b>For Beginners:</b> Meta-learning is training an AI to be a "fast learner." Instead of
/// training a model once on lots of data, we train it on many small tasks, each with very few examples.
///
/// This loader helps create those small tasks:
/// - <b>N-way:</b> How many different categories each task should have (e.g., 5-way = 5 classes)
/// - <b>K-shot:</b> How many examples per category to use for learning (e.g., 3-shot = 3 examples/class)
/// - <b>Query shots:</b> How many examples per category to use for testing (e.g., 10 queries/class)
///
/// Example: 5-way 3-shot with 10 queries
/// - Support set: 5 classes × 3 examples = 15 total examples to learn from
/// - Query set: 5 classes × 10 examples = 50 total examples to test on
///
/// Why this matters:
/// - Mimics real-world scenarios where you have limited labeled data
/// - Teaches models to generalize from very few examples
/// - Enables rapid adaptation to new tasks
///
/// The same underlying dataset is resampled many times to create different tasks, each
/// presenting a unique few-shot learning challenge.
/// </para>
/// <para>
/// <b>Thread Safety:</b> This class is not thread-safe due to internal Random state.
/// Create separate instances for concurrent task generation.
/// </para>
/// <para>
/// <b>Performance:</b> Task creation is O(nWay × (kShot + queryShots)) for sampling and
/// tensor construction. Preprocessing is O(n) where n is dataset size.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Load a dataset (e.g., Omniglot, Mini-ImageNet)
/// var features = new Matrix&lt;double&gt;(1000, 784);  // 1000 examples, 784 features
/// var labels = new Vector&lt;double&gt;(1000);         // Class labels 0-9
///
/// // Create 5-way 3-shot loader with 10 query examples per class
/// var loader = new EpisodicDataLoader&lt;double&gt;(
///     datasetX: features,
///     datasetY: labels,
///     nWay: 5,        // 5 classes per task
///     kShot: 3,       // 3 support examples per class
///     queryShots: 10, // 10 query examples per class
///     seed: 42        // Optional: for reproducibility
/// );
///
/// // Sample tasks for meta-training
/// for (int episode = 0; episode &lt; 1000; episode++)
/// {
///     var task = loader.GetNextTask();
///
///     // Support set: [15, 784] and [15]
///     Console.WriteLine($"Support: {task.SupportSetX.Shape[0]} examples");
///
///     // Query set: [50, 784] and [50]
///     Console.WriteLine($"Query: {task.QuerySetX.Shape[0]} examples");
///
///     // Use task with MAML, Reptile, or SEAL
///     model.MetaTrainStep(task);
/// }
/// </code>
/// </example>
public class EpisodicDataLoader<T>
{
    private readonly Matrix<T> _datasetX;
    private readonly Vector<T> _datasetY;
    private readonly int _nWay;
    private readonly int _kShot;
    private readonly int _queryShots;
    private readonly Random _random;
    private readonly Dictionary<int, List<int>> _classToIndices;
    private readonly int[] _availableClasses;

    /// <summary>
    /// Numeric operations helper for type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Initializes a new instance of the EpisodicDataLoader for N-way K-shot task sampling.
    /// </summary>
    /// <param name="datasetX">The feature matrix where each row is an example. Shape: [num_examples, num_features].</param>
    /// <param name="datasetY">The label vector containing class labels for each example. Length: num_examples.</param>
    /// <param name="nWay">The number of unique classes per task. Must be at least 2.</param>
    /// <param name="kShot">The number of support examples per class. Must be at least 1.</param>
    /// <param name="queryShots">The number of query examples per class. Must be at least 1.</param>
    /// <param name="seed">Optional random seed for reproducible task sampling. If null, uses a time-based seed.</param>
    /// <exception cref="ArgumentNullException">Thrown when datasetX or datasetY is null.</exception>
    /// <exception cref="ArgumentException">Thrown when dimensions are invalid or dataset is too small.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This constructor prepares the dataset for episodic sampling.
    ///
    /// <b>Parameters explained:</b>
    /// - <b>datasetX:</b> Your input features (images, embeddings, etc.) - one example per row
    /// - <b>datasetY:</b> The category/class for each example (must be integers or convertible to integers)
    /// - <b>nWay:</b> How many categories in each task (e.g., 5 means each task has 5 different classes)
    /// - <b>kShot:</b> How many training examples per category (e.g., 3 means 3 examples to learn from)
    /// - <b>queryShots:</b> How many test examples per category (e.g., 10 means 10 examples to evaluate on)
    /// - <b>seed:</b> Controls randomness - same seed = same task sequence (useful for debugging/testing)
    ///
    /// <b>Requirements:</b>
    /// - Your dataset must have at least nWay different classes
    /// - Each class must have at least (kShot + queryShots) examples
    /// - Labels in datasetY should be integers (0, 1, 2, ...) or convertible to integers
    ///
    /// <b>What happens during initialization:</b>
    /// The loader organizes your data by class, creating an index that maps each class label
    /// to the row numbers of all examples from that class. This makes sampling fast later.
    /// </para>
    /// </remarks>
    public EpisodicDataLoader(
        Matrix<T> datasetX,
        Vector<T> datasetY,
        int nWay,
        int kShot,
        int queryShots,
        int? seed = null)
    {
        // Validate inputs
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

        _datasetX = datasetX;
        _datasetY = datasetY;
        _nWay = nWay;
        _kShot = kShot;
        _queryShots = queryShots;
        _random = seed.HasValue ? new Random(seed.Value) : new Random();

        // Preprocess: Build class-to-indices mapping
        _classToIndices = new Dictionary<int, List<int>>();

        for (int i = 0; i < datasetY.Length; i++)
        {
            int classLabel = NumOps.ToInt32(datasetY[i]);

            if (!_classToIndices.ContainsKey(classLabel))
            {
                _classToIndices[classLabel] = new List<int>();
            }

            _classToIndices[classLabel].Add(i);
        }

        _availableClasses = _classToIndices.Keys.ToArray();

        // Validate dataset has enough classes
        if (_availableClasses.Length < nWay)
        {
            throw new ArgumentException(
                $"Dataset has only {_availableClasses.Length} classes, but nWay={nWay} requires at least {nWay} classes",
                nameof(datasetY));
        }

        // Validate each class has enough examples
        int requiredExamplesPerClass = kShot + queryShots;
        var insufficientClasses = _classToIndices
            .Where(kvp => kvp.Value.Count < requiredExamplesPerClass)
            .Select(kvp => $"Class {kvp.Key}: {kvp.Value.Count} examples")
            .ToList();

        if (insufficientClasses.Count > 0)
        {
            throw new ArgumentException(
                $"Some classes have insufficient examples. Need at least {requiredExamplesPerClass} " +
                $"(kShot={kShot} + queryShots={queryShots}) per class. Insufficient classes: " +
                string.Join(", ", insufficientClasses),
                nameof(datasetY));
        }
    }

    /// <summary>
    /// Samples and returns the next N-way K-shot meta-learning task.
    /// </summary>
    /// <returns>
    /// A MetaLearningTask containing support and query sets with the specified N-way K-shot configuration.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method performs the following steps:
    /// 1. Randomly selects N unique classes from the dataset
    /// 2. For each selected class, randomly samples (K + queryShots) examples
    /// 3. Shuffles the sampled examples to avoid ordering bias
    /// 4. Splits the shuffled examples: first K go to support set, remaining go to query set
    /// 5. Constructs tensors and returns them as a MetaLearningTask
    /// </para>
    /// <para><b>For Beginners:</b> Each call to this method creates a new random mini-problem (task)
    /// from your dataset.
    ///
    /// <b>How it works:</b>
    /// 1. <b>Pick random classes:</b> Select N different categories (e.g., for 5-way, pick classes like cat, dog, bird, fish, rabbit)
    /// 2. <b>Sample examples:</b> For each class, grab K+queryShots random examples (e.g., 3+10=13 images per class)
    /// 3. <b>Shuffle:</b> Mix up the examples within each class (prevents the model from learning based on example order)
    /// 4. <b>Split:</b> First K examples become support (training), rest become query (testing)
    /// 5. <b>Package:</b> Combine all support examples and all query examples into tensors
    ///
    /// <b>Important:</b> Each task has completely different classes sampled from the dataset.
    /// The goal is to teach the model to adapt quickly to whatever classes appear in the task.
    ///
    /// <b>Example output shapes (5-way 3-shot, 10 queries):</b>
    /// - SupportSetX: [15, num_features] - (5 classes × 3 shots)
    /// - SupportSetY: [15] - Labels for support set
    /// - QuerySetX: [50, num_features] - (5 classes × 10 queries)
    /// - QuerySetY: [50] - Labels for query set
    /// </para>
    /// <para>
    /// <b>Performance:</b> O(nWay × (kShot + queryShots)) time complexity.
    /// Creates new tensor objects on each call.
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// var loader = new EpisodicDataLoader&lt;double&gt;(features, labels, nWay: 5, kShot: 3, queryShots: 10);
    ///
    /// // Generate a batch of tasks for meta-training
    /// var tasks = new List&lt;MetaLearningTask&lt;double&gt;&gt;();
    /// for (int i = 0; i &lt; 32; i++)  // Meta-batch size of 32
    /// {
    ///     tasks.Add(loader.GetNextTask());
    /// }
    ///
    /// // Each task has different randomly-selected classes
    /// // Train your meta-learner to perform well across all these diverse tasks
    /// </code>
    /// </example>
    public MetaLearningTask<T> GetNextTask()
    {
        // Step 1: Randomly select nWay unique classes
        var selectedClasses = _availableClasses
            .OrderBy(_ => _random.Next())
            .Take(_nWay)
            .ToArray();

        // Prepare storage for support and query sets
        var supportExamples = new List<Vector<T>>();
        var supportLabels = new List<T>();
        var queryExamples = new List<Vector<T>>();
        var queryLabels = new List<T>();

        // Step 2: For each selected class, sample and split examples
        for (int classIdx = 0; classIdx < selectedClasses.Length; classIdx++)
        {
            int classLabel = selectedClasses[classIdx];
            var classIndices = _classToIndices[classLabel];

            // Step 3: Sample (kShot + queryShots) examples and shuffle
            var sampledIndices = classIndices
                .OrderBy(_ => _random.Next())
                .Take(_kShot + _queryShots)
                .ToList();

            // Shuffle the sampled indices to prevent ordering bias
            sampledIndices = sampledIndices.OrderBy(_ => _random.Next()).ToList();

            // Step 4: Split into support (first kShot) and query (remaining queryShots)
            var supportIndices = sampledIndices.Take(_kShot);
            var queryIndices = sampledIndices.Skip(_kShot);

            // Add support examples
            foreach (var idx in supportIndices)
            {
                supportExamples.Add(_datasetX.GetRow(idx));
                supportLabels.Add(NumOps.FromDouble(classIdx)); // Use index 0..nWay-1 for the task
            }

            // Add query examples
            foreach (var idx in queryIndices)
            {
                queryExamples.Add(_datasetX.GetRow(idx));
                queryLabels.Add(NumOps.FromDouble(classIdx)); // Use index 0..nWay-1 for the task
            }
        }

        // Step 5: Convert to tensors
        int numFeatures = _datasetX.Columns;
        int supportSize = _nWay * _kShot;
        int querySize = _nWay * _queryShots;

        // Build support set tensors
        var supportSetX = new Tensor<T>(new[] { supportSize, numFeatures });
        var supportSetY = new Tensor<T>(new[] { supportSize });

        for (int i = 0; i < supportSize; i++)
        {
            for (int j = 0; j < numFeatures; j++)
            {
                supportSetX[new[] { i, j }] = supportExamples[i][j];
            }
            supportSetY[new[] { i }] = supportLabels[i];
        }

        // Build query set tensors
        var querySetX = new Tensor<T>(new[] { querySize, numFeatures });
        var querySetY = new Tensor<T>(new[] { querySize });

        for (int i = 0; i < querySize; i++)
        {
            for (int j = 0; j < numFeatures; j++)
            {
                querySetX[new[] { i, j }] = queryExamples[i][j];
            }
            querySetY[new[] { i }] = queryLabels[i];
        }

        // Return the completed task
        return new MetaLearningTask<T>
        {
            SupportSetX = supportSetX,
            SupportSetY = supportSetY,
            QuerySetX = querySetX,
            QuerySetY = querySetY
        };
    }
}
