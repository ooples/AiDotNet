using AiDotNet.Data.Abstractions;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Data.Loaders;

/// <summary>
/// Provides balanced episodic task sampling that ensures equal class representation across multiple tasks.
/// </summary>
/// <typeparam name="T">The numeric data type used for features and labels (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// The BalancedEpisodicDataLoader extends the standard episodic loader by tracking class usage
/// across multiple tasks and preferentially sampling under-represented classes. This ensures
/// that over many episodes, all classes appear roughly the same number of times, preventing
/// bias toward frequently-sampled classes.
/// </para>
/// <para><b>For Beginners:</b> Standard random sampling might pick some classes more often than others
/// by chance. This can cause problems in meta-learning:
/// - The model might learn some classes better than others
/// - Training could be biased toward frequently-sampled classes
/// - Evaluation metrics might be skewed
///
/// The balanced loader solves this by:
/// - Tracking how many times each class has been selected
/// - Preferring classes that haven't been used as much
/// - Ensuring fair representation across all classes over time
///
/// <b>When to use this:</b>
/// - Long meta-training runs where balanced class exposure matters
/// - When your dataset has many classes and you want uniform coverage
/// - When evaluating meta-learning algorithms fairly across all classes
///
/// <b>Trade-off:</b> Less random than uniform sampling, but more balanced. Good for training,
/// but you might want standard EpisodicDataLoader for final evaluation to match real-world randomness.
/// </para>
/// <para>
/// <b>Thread Safety:</b> This class is not thread-safe due to internal state tracking.
/// Create separate instances for concurrent task generation.
/// </para>
/// <para>
/// <b>Performance:</b> Slightly slower than standard EpisodicDataLoader due to usage tracking
/// and weighted sampling, but still O(nWay × (kShot + queryShots)) for task creation.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Load a dataset with imbalanced class distribution
/// var features = new Matrix&lt;double&gt;(1000, 784);
/// var labels = new Vector&lt;double&gt;(1000);  // 20 classes with varying frequencies
///
/// // Create balanced loader - all classes will be sampled equally over time
/// var loader = new BalancedEpisodicDataLoader&lt;double&gt;(
///     datasetX: features,
///     datasetY: labels,
///     nWay: 5,
///     kShot: 3,
///     queryShots: 10,
///     seed: 42
/// );
///
/// // Over 1000 episodes, each class will appear in roughly the same number of tasks
/// for (int episode = 0; episode &lt; 1000; episode++)
/// {
///     var task = loader.GetNextTask();
///     // Train on balanced distribution of classes
/// }
/// </code>
/// </example>
public class BalancedEpisodicDataLoader<T> : EpisodicDataLoaderBase<T>
{
    private readonly Dictionary<int, int> _classUsageCount;

    /// <summary>
    /// Initializes a new instance of the BalancedEpisodicDataLoader for balanced N-way K-shot task sampling.
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
    /// <para><b>For Beginners:</b> This constructor sets up balanced sampling by initializing
    /// usage tracking for all classes. Each class starts with a count of zero, and as tasks are
    /// generated, the loader keeps track of which classes have been used and prefers less-used ones.
    ///
    /// The balancing happens automatically - you don't need to do anything special. Just call
    /// GetNextTask() repeatedly and the loader will ensure balanced class distribution over time.
    /// </para>
    /// </remarks>
    public BalancedEpisodicDataLoader(
        Matrix<T> datasetX,
        Vector<T> datasetY,
        int nWay = 5,
        int kShot = 5,
        int queryShots = 15,
        int? seed = null)
        : base(datasetX, datasetY, nWay, kShot, queryShots, seed)
    {
        // Initialize usage tracking - all classes start with count 0
        _classUsageCount = new Dictionary<int, int>();
        foreach (var classLabel in AvailableClasses)
        {
            _classUsageCount[classLabel] = 0;
        }
    }

    /// <summary>
    /// Core implementation of balanced N-way K-shot task sampling with weighted class selection.
    /// </summary>
    /// <returns>A MetaLearningTask with balanced class sampling over time.</returns>
    /// <remarks>
    /// <para>
    /// This method extends the standard sampling algorithm with balanced selection:
    /// 1. Calculates selection weights: classes with lower usage get higher weights
    /// 2. Performs weighted random selection of N classes (favoring under-used classes)
    /// 3. For each selected class, randomly samples (K + queryShots) examples
    /// 4. Shuffles and splits into support and query sets
    /// 5. Updates usage counts for selected classes
    /// 6. Constructs and returns MetaLearningTask
    /// </para>
    /// <para><b>For Beginners:</b> The balancing works like this:
    ///
    /// Imagine you have 10 classes. After 5 tasks:
    /// - Classes 0, 2, 5 have been used 3 times each
    /// - Classes 1, 4, 7 have been used 2 times each
    /// - Classes 3, 6, 8, 9 have been used 1 time each
    ///
    /// For the next task, the loader will heavily favor classes 3, 6, 8, 9 (used least),
    /// moderately favor classes 1, 4, 7, and avoid classes 0, 2, 5 (used most).
    ///
    /// Over many tasks, this ensures all classes get approximately equal representation,
    /// leading to more balanced meta-learning training.
    /// </para>
    /// </remarks>
    protected override MetaLearningTask<T> GetNextTaskCore()
    {
        // Step 1: Calculate selection weights based on usage (inverse weighting)
        // Classes used less get higher weights
        var minUsage = _classUsageCount.Values.Min();
        var maxUsage = _classUsageCount.Values.Max();
        var usageRange = maxUsage - minUsage + 1; // Add 1 to avoid division by zero

        var weights = new Dictionary<int, double>();
        foreach (var classLabel in AvailableClasses)
        {
            // Inverse weight: less used = higher weight
            weights[classLabel] = usageRange - (_classUsageCount[classLabel] - minUsage);
        }

        // Step 2: Perform weighted random selection of nWay classes
        var selectedClasses = WeightedSample(AvailableClasses, weights, NWay);

        // Step 3: Update usage counts for selected classes
        foreach (var classLabel in selectedClasses)
        {
            _classUsageCount[classLabel]++;
        }

        // Step 4: Sample examples for each selected class (same as standard loader)
        var supportExamples = new List<Vector<T>>();
        var supportLabels = new List<T>();
        var queryExamples = new List<Vector<T>>();
        var queryLabels = new List<T>();

        for (int classIdx = 0; classIdx < selectedClasses.Length; classIdx++)
        {
            int classLabel = selectedClasses[classIdx];
            var classIndices = ClassToIndices[classLabel];

            // Sample (kShot + queryShots) examples and shuffle
            var sampledIndices = classIndices
                .OrderBy(_ => Random.Next())
                .Take(KShot + QueryShots)
                .ToList();

            // Shuffle the sampled indices to prevent ordering bias
            sampledIndices = sampledIndices.OrderBy(_ => Random.Next()).ToList();

            // Split into support and query
            var supportIndices = sampledIndices.Take(KShot);
            var queryIndices = sampledIndices.Skip(KShot);

            // Add support examples
            foreach (var idx in supportIndices)
            {
                supportExamples.Add(DatasetX.GetRow(idx));
                supportLabels.Add(NumOps.FromDouble(classIdx));
            }

            // Add query examples
            foreach (var idx in queryIndices)
            {
                queryExamples.Add(DatasetX.GetRow(idx));
                queryLabels.Add(NumOps.FromDouble(classIdx));
            }
        }

        // Step 5: Convert to tensors and return
        return BuildMetaLearningTask(supportExamples, supportLabels, queryExamples, queryLabels);
    }

    /// <summary>
    /// Performs weighted random sampling without replacement.
    /// </summary>
    private int[] WeightedSample(int[] population, Dictionary<int, double> weights, int sampleSize)
    {
        var selected = new List<int>();
        var remaining = population.ToList();
        var remainingWeights = new Dictionary<int, double>(weights);

        for (int i = 0; i < sampleSize; i++)
        {
            // Calculate cumulative weights
            var totalWeight = remainingWeights.Values.Sum();
            var randomValue = Random.NextDouble() * totalWeight;

            // Select based on weighted probability
            double cumulative = NumOps.ToInt32(NumOps.Zero);
            int selectedClass = remaining[0];

            foreach (var classLabel in remaining)
            {
                cumulative += remainingWeights[classLabel];
                if (randomValue <= cumulative)
                {
                    selectedClass = classLabel;
                    break;
                }
            }

            selected.Add(selectedClass);
            remaining.Remove(selectedClass);
            remainingWeights.Remove(selectedClass);
        }

        return selected.ToArray();
    }

    /// <summary>
    /// Builds a MetaLearningTask from lists of examples and labels.
    /// </summary>
    private MetaLearningTask<T> BuildMetaLearningTask(
        List<Vector<T>> supportExamples,
        List<T> supportLabels,
        List<Vector<T>> queryExamples,
        List<T> queryLabels)
    {
        int numFeatures = DatasetX.Columns;
        int supportSize = NWay * KShot;
        int querySize = NWay * QueryShots;

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

        return new MetaLearningTask<T>
        {
            SupportSetX = supportSetX,
            SupportSetY = supportSetY,
            QuerySetX = querySetX,
            QuerySetY = querySetY
        };
    }
}
