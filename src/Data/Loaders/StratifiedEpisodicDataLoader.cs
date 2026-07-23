using AiDotNet.Data.Structures;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Data.Loaders;

/// <summary>
/// Provides stratified episodic task sampling that maintains dataset class proportions across tasks.
/// </summary>
/// <typeparam name="T">The numeric data type used for features and labels (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// The StratifiedEpisodicDataLoader extends the standard episodic loader by sampling classes
/// proportionally to their representation in the dataset. If a class represents 30% of the dataset,
/// it will appear in approximately 30% of tasks over many episodes, preserving the natural
/// class distribution.
/// </para>
/// <para><b>For Beginners:</b> Real-world datasets often have imbalanced class distributions:
/// - Medical datasets might have 90% healthy cases, 10% disease cases
/// - E-commerce might have 80% browsing, 15% cart additions, 5% purchases
/// - Image datasets might have common objects (cars, trees) and rare ones (exotic animals)
///
/// Standard random sampling treats all classes equally, which doesn't reflect reality.
/// Stratified sampling maintains these natural proportions:
/// - Common classes appear in more tasks
/// - Rare classes appear in fewer tasks
/// - The model learns to handle the real-world distribution
///
/// <b>When to use this:</b>
/// - When your dataset has natural class imbalance that you want to preserve
/// - When training for real-world deployment where class frequencies matter
/// - When you want meta-learning to reflect actual data distributions
///
/// <b>When NOT to use this:</b>
/// - When you want equal exposure to all classes (use BalancedEpisodicDataLoader)
/// - When evaluating few-shot learning fairly across all classes
/// - When class frequencies in deployment differ from training data
/// </para>
/// <para>
/// <b>Thread Safety:</b> This class is not thread-safe due to internal Random state.
/// Create separate instances for concurrent task generation.
/// </para>
/// <para>
/// <b>Performance:</b> Similar to standard EpisodicDataLoader with O(nWay × (kShot + queryShots))
/// complexity. Slightly slower due to proportional weight calculation during initialization.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Dataset with imbalanced classes
/// // Class 0: 500 examples (50%)
/// // Class 1: 300 examples (30%)
/// // Class 2: 200 examples (20%)
/// var features = new Matrix&lt;double&gt;(1000, 784);
/// var labels = new Vector&lt;double&gt;(1000);
///
/// // Create stratified loader - maintains 50%/30%/20% distribution
/// var loader = new StratifiedEpisodicDataLoader&lt;double&gt;(
///     datasetX: features,
///     datasetY: labels,
///     nWay: 2,  // 2-way tasks
///     kShot: 5,
///     queryShots: 15,
///     seed: 42
/// );
///
/// // Over many tasks, class 0 will appear in ~50% of tasks,
/// // class 1 in ~30%, and class 2 in ~20%
/// for (int episode = 0; episode &lt; 1000; episode++)
/// {
///     var task = loader.GetNextTask();
///     // Train on distribution matching real-world data
/// }
/// </code>
/// </example>
public class StratifiedEpisodicDataLoader<T, TInput, TOutput> : EpisodicDataLoaderBase<T, TInput, TOutput>
{
    private readonly Dictionary<int, double> _classWeights;

    /// <summary>
    /// Initializes a new instance of the StratifiedEpisodicDataLoader for proportional N-way K-shot task sampling.
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
    /// <para><b>For Beginners:</b> This constructor calculates the proportion of each class in your dataset
    /// and stores these as weights for sampling.
    ///
    /// For example, if you have:
    /// - 1000 total examples
    /// - Class A: 500 examples (50%)
    /// - Class B: 300 examples (30%)
    /// - Class C: 200 examples (20%)
    ///
    /// The weights will be 0.5, 0.3, 0.2 respectively. When generating tasks, classes will be
    /// selected randomly but weighted by these proportions, so Class A appears most often.
    /// </para>
    /// </remarks>
    public StratifiedEpisodicDataLoader(
        Matrix<T> datasetX,
        Vector<T> datasetY,
        int nWay = 5,
        int kShot = 5,
        int queryShots = 15,
        int? seed = null)
        : base(datasetX, datasetY, nWay, kShot, queryShots, seed)
    {
        // Calculate class weights based on their proportion in the dataset
        int totalExamples = datasetY.Length;
        _classWeights = new Dictionary<int, double>();

        foreach (var classLabel in _availableClasses)
        {
            int classCount = ClassToIndices[classLabel].Count;
            _classWeights[classLabel] = (double)classCount / totalExamples;
        }
    }

    /// <summary>
    /// Core implementation of stratified N-way K-shot task sampling with proportional class selection.
    /// </summary>
    /// <returns>A MetaLearningTask with classes sampled proportionally to their dataset representation.</returns>
    /// <remarks>
    /// <para>
    /// This method implements proportional sampling:
    /// 1. Selects N classes using weighted random sampling (proportional to class frequency)
    /// 2. For each selected class, randomly samples (K + queryShots) examples
    /// 3. Shuffles and splits into support and query sets
    /// 4. Constructs and returns MetaLearningTask
    /// </para>
    /// <para><b>For Beginners:</b> Think of it like a lottery where each class has tickets
    /// proportional to how common it is:
    ///
    /// - Class A (50% of data): Gets 500 lottery tickets
    /// - Class B (30% of data): Gets 300 lottery tickets
    /// - Class C (20% of data): Gets 200 lottery tickets
    ///
    /// When selecting classes for a task, we draw from this lottery. Class A is most likely
    /// to be drawn, Class B moderately likely, and Class C least likely.
    ///
    /// This means:
    /// - Common real-world scenarios appear in more training tasks
    /// - Rare scenarios appear less frequently, matching reality
    /// - The model learns the true distribution it will see in deployment
    /// </para>
    /// </remarks>
    protected override MetaLearningTask<T, TInput, TOutput> GetNextTaskCore()
    {
        // Step 1: Perform weighted random selection of nWay classes (proportional to class frequency)
        var selectedClasses = WeightedSampleClasses(NWay);

        // Step 2: Sample examples for each selected class
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
                .OrderBy(_ => RandomInstance.Next())
                .Take(KShot + QueryShots)
                .ToList();

            // Shuffle the sampled indices to prevent ordering bias
            sampledIndices = sampledIndices.OrderBy(_ => RandomInstance.Next()).ToList();

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

        // Step 3: Convert to tensors and return
        return BuildMetaLearningTask(supportExamples, supportLabels, queryExamples, queryLabels);
    }

    /// <summary>
    /// Performs weighted random sampling of classes based on their dataset proportions.
    /// </summary>
    private int[] WeightedSampleClasses(int sampleSize)
    {
        var selected = new List<int>();
        var remaining = _availableClasses.ToList();
        var remainingWeights = new Dictionary<int, double>(_classWeights);

        for (int i = 0; i < sampleSize; i++)
        {
            // Normalize remaining weights to sum to 1
            var totalWeight = remainingWeights.Values.Sum();

            // Generate random value between 0 and totalWeight
            var randomValue = RandomInstance.NextDouble() * totalWeight;

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
}
