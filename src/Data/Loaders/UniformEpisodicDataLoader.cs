using AiDotNet.Data.Structures;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Data.Loaders;

/// <summary>
/// Provides uniform random episodic task sampling for N-way K-shot meta-learning scenarios.
/// </summary>
/// <typeparam name="T">The numeric data type used for features and labels (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type for tasks (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;, double[]).</typeparam>
/// <typeparam name="TOutput">The output data type for tasks (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;, double[]).</typeparam>
/// <remarks>
/// <para>
/// The UniformEpisodicDataLoader transforms a standard supervised learning dataset (features + labels) into
/// a stream of meta-learning tasks using uniform random sampling. Each task contains a support set (for quick adaptation) and a
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
/// var loader = new UniformEpisodicDataLoader&lt;double&gt;(
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
public class UniformEpisodicDataLoader<T, TInput, TOutput> : EpisodicDataLoaderBase<T, TInput, TOutput>
{
    private const int MaxResampleAttempts = 32;

    private bool _hasLastTaskSignature;
    private ulong _lastTaskSignature;
    private ulong _lastClassSignature;
    private int _lastFirstSupportIndex;

    /// <summary>
    /// Initializes a new instance of the UniformEpisodicDataLoader for N-way K-shot task sampling with industry-standard defaults.
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
    public UniformEpisodicDataLoader(
        Matrix<T> datasetX,
        Vector<T> datasetY,
        int nWay = 5,
        int kShot = 5,
        int queryShots = 15,
        int? seed = null)
        : base(datasetX, datasetY, nWay, kShot, queryShots, seed)
    {
    }

    /// <summary>
    /// Core implementation of N-way K-shot task sampling with uniform random selection.
    /// </summary>
    /// <returns>A MetaLearningTask with randomly sampled support and query sets.</returns>
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
    protected override MetaLearningTask<T, TInput, TOutput> GetNextTaskCore()
    {
        (MetaLearningTask<T, TInput, TOutput> Task, ulong Signature, ulong ClassSignature, int FirstSupportIndex) SampleTaskCandidate()
        {
            // Step 1: Randomly select nWay unique classes
            var selectedClasses = _availableClasses
                .OrderBy(_ => RandomInstance.Next())
                .Take(NWay)
                .ToArray();

            // Prepare storage for support and query sets
            var supportExamples = new List<Vector<T>>();
            var supportLabels = new List<T>();
            var queryExamples = new List<Vector<T>>();
            var queryLabels = new List<T>();
            int firstSupportIndex = -1;

            ulong signature = 14695981039346656037UL;
            for (int i = 0; i < selectedClasses.Length; i++)
            {
                signature = HashCombine(signature, selectedClasses[i]);
            }

            // Step 2: For each selected class, sample and split examples
            for (int classIdx = 0; classIdx < selectedClasses.Length; classIdx++)
            {
                int classLabel = selectedClasses[classIdx];
                var classIndices = ClassToIndices[classLabel];

                // Step 3: Sample (kShot + queryShots) examples and shuffle
                var sampledIndices = classIndices
                    .OrderBy(_ => RandomInstance.Next())
                    .Take(KShot + QueryShots)
                    .ToList();

                for (int i = 0; i < sampledIndices.Count; i++)
                {
                    signature = HashCombine(signature, sampledIndices[i]);
                }

                if (classIdx == 0 && sampledIndices.Count > 0)
                {
                    firstSupportIndex = sampledIndices[0];
                }

                // Step 4: Split into support (first kShot) and query (remaining queryShots)
                var supportIndices = sampledIndices.Take(KShot);
                var queryIndices = sampledIndices.Skip(KShot);

                // Add support examples
                foreach (var idx in supportIndices)
                {
                    supportExamples.Add(DatasetX.GetRow(idx));
                    supportLabels.Add(NumOps.FromDouble(classIdx)); // Use index 0..nWay-1 for the task
                }

                // Add query examples
                foreach (var idx in queryIndices)
                {
                    queryExamples.Add(DatasetX.GetRow(idx));
                    queryLabels.Add(NumOps.FromDouble(classIdx)); // Use index 0..nWay-1 for the task
                }
            }

            return (
                BuildMetaLearningTask(supportExamples, supportLabels, queryExamples, queryLabels),
                signature,
                ComputeClassSignature(selectedClasses),
                firstSupportIndex);
        }

        bool IsDistinctFromLast(in (MetaLearningTask<T, TInput, TOutput> Task, ulong Signature, ulong ClassSignature, int FirstSupportIndex) candidate)
        {
            if (!_hasLastTaskSignature)
            {
                return true;
            }

            if (candidate.Signature == _lastTaskSignature)
            {
                return false;
            }

            return !(candidate.ClassSignature == _lastClassSignature &&
                     candidate.FirstSupportIndex == _lastFirstSupportIndex);
        }

        void UpdateLast(in (MetaLearningTask<T, TInput, TOutput> Task, ulong Signature, ulong ClassSignature, int FirstSupportIndex) candidate)
        {
            _lastTaskSignature = candidate.Signature;
            _lastClassSignature = candidate.ClassSignature;
            _hasLastTaskSignature = true;
            _lastFirstSupportIndex = candidate.FirstSupportIndex;
        }

        for (int attempt = 0; attempt < MaxResampleAttempts; attempt++)
        {
            var candidate = SampleTaskCandidate();

            if (!IsDistinctFromLast(candidate))
            {
                continue;
            }

            UpdateLast(candidate);
            return candidate.Task;
        }

        var fallbackCandidate = SampleTaskCandidate();
        for (int attempt = 0; attempt < MaxResampleAttempts && !IsDistinctFromLast(fallbackCandidate); attempt++)
        {
            fallbackCandidate = SampleTaskCandidate();
        }

        UpdateLast(fallbackCandidate);
        return fallbackCandidate.Task;
    }

    private static ulong ComputeClassSignature(int[] selectedClasses)
    {
        var sorted = (int[])selectedClasses.Clone();
        System.Array.Sort(sorted);

        ulong signature = 14695981039346656037UL;
        for (int i = 0; i < sorted.Length; i++)
        {
            signature = HashCombine(signature, sorted[i]);
        }

        return signature;
    }

    private static ulong HashCombine(ulong hash, int value)
    {
        unchecked
        {
            hash ^= (uint)value;
            hash *= 1099511628211UL;
            return hash;
        }
    }
}
