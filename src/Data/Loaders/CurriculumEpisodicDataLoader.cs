using AiDotNet.Data.Abstractions;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Data.Loaders;

/// <summary>
/// Provides curriculum-based episodic task sampling that progressively increases task difficulty during training.
/// </summary>
/// <typeparam name="T">The numeric data type used for features and labels (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// The CurriculumEpisodicDataLoader implements curriculum learning for meta-learning by starting
/// with easier tasks and progressively increasing difficulty. Easier tasks have fewer classes (lower N-way)
/// and more examples per class (higher K-shot), while harder tasks approach the target N-way K-shot configuration.
/// This gradual progression helps models learn more effectively by building on simpler concepts first.
/// </para>
/// <para><b>For Beginners:</b> Curriculum learning is inspired by how humans learn:
/// - You don't start learning math with calculus - you start with counting, then addition, etc.
/// - You don't learn a language by reading novels - you start with basic vocabulary and grammar
/// - Complex skills are built on simpler foundations
///
/// This loader applies the same principle to meta-learning:
/// - <b>Easy tasks:</b> 2-way 10-shot (2 classes, 10 examples each) - lots of examples, few classes
/// - <b>Medium tasks:</b> 4-way 5-shot (4 classes, 5 examples each) - balanced difficulty
/// - <b>Hard tasks:</b> 5-way 1-shot (5 classes, 1 example each) - few examples, many classes
///
/// <b>How difficulty progression works:</b>
/// - Progress is tracked from 0.0 (start) to 1.0 (end)
/// - At progress 0.0: Easy tasks (2-way, lots of shots)
/// - At progress 0.5: Medium tasks (halfway to target)
/// - At progress 1.0: Target difficulty (full N-way K-shot)
///
/// <b>When to use this:</b>
/// - When training struggles to converge on hard tasks from the start
/// - When you want to improve sample efficiency and training stability
/// - When implementing human-inspired learning strategies
/// - Research has shown curriculum learning often leads to better final performance
///
/// <b>Note:</b> You must manually update the progress as training proceeds (e.g., based on
/// episode number, training loss, or validation accuracy).
/// </para>
/// <para>
/// <b>Thread Safety:</b> This class is not thread-safe due to internal state.
/// Create separate instances for concurrent task generation.
/// </para>
/// <para>
/// <b>Performance:</b> Same complexity as standard EpisodicDataLoader, with minor overhead
/// for calculating current difficulty parameters based on progress.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var features = new Matrix&lt;double&gt;(1000, 784);
/// var labels = new Vector&lt;double&gt;(1000);
///
/// // Create curriculum loader: start easy (2-way 10-shot), end hard (5-way 1-shot)
/// var loader = new CurriculumEpisodicDataLoader&lt;double&gt;(
///     datasetX: features,
///     datasetY: labels,
///     targetNWay: 5,         // Final target: 5-way
///     targetKShot: 1,        // Final target: 1-shot
///     queryShots: 15,        // Fixed query examples
///     initialNWay: 2,        // Start easy: 2-way
///     initialKShot: 10,      // Start easy: 10-shot
///     seed: 42
/// );
///
/// int totalEpisodes = 10000;
/// for (int episode = 0; episode &lt; totalEpisodes; episode++)
/// {
///     // Update curriculum progress (0.0 to 1.0)
///     double progress = (double)episode / totalEpisodes;
///     loader.SetProgress(progress);
///
///     var task = loader.GetNextTask();
///     // Train on progressively harder tasks
///
///     // At episode 0: 2-way 10-shot (easy)
///     // At episode 5000: ~3-way 5-shot (medium)
///     // At episode 10000: 5-way 1-shot (hard/target)
/// }
/// </code>
/// </example>
public class CurriculumEpisodicDataLoader<T> : EpisodicDataLoaderBase<T>
{
    private readonly int _targetNWay;
    private readonly int _targetKShot;
    private readonly int _initialNWay;
    private readonly int _initialKShot;
    private double _progress;

    /// <summary>
    /// Gets the current curriculum progress (0.0 = easiest, 1.0 = target difficulty).
    /// </summary>
    public double Progress => _progress;

    /// <summary>
    /// Initializes a new instance of the CurriculumEpisodicDataLoader for progressive N-way K-shot task sampling.
    /// </summary>
    /// <param name="datasetX">The feature matrix where each row is an example. Shape: [num_examples, num_features].</param>
    /// <param name="datasetY">The label vector containing class labels for each example. Length: num_examples.</param>
    /// <param name="targetNWay">The target number of classes per task (final difficulty). Must be at least 2.</param>
    /// <param name="targetKShot">The target number of support examples per class (final difficulty). Must be at least 1.</param>
    /// <param name="queryShots">The number of query examples per class (constant across curriculum). Must be at least 1.</param>
    /// <param name="initialNWay">The initial number of classes per task (easy start). Must be at least 2 and less than or equal to targetNWay.</param>
    /// <param name="initialKShot">The initial number of support examples per class (easy start). Must be at least targetKShot.</param>
    /// <param name="seed">Optional random seed for reproducible task sampling. If null, uses a time-based seed.</param>
    /// <exception cref="ArgumentNullException">Thrown when datasetX or datasetY is null.</exception>
    /// <exception cref="ArgumentException">Thrown when dimensions are invalid, dataset is too small, or curriculum parameters are inconsistent.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This constructor sets up a curriculum from easy to hard:
    ///
    /// - <b>initialNWay, initialKShot:</b> Starting difficulty (easy tasks with few classes, many examples)
    /// - <b>targetNWay, targetKShot:</b> Ending difficulty (hard tasks with many classes, few examples)
    /// - <b>queryShots:</b> Stays constant throughout (doesn't affect difficulty much)
    ///
    /// <b>Parameter guidelines:</b>
    /// - initialNWay should be smaller than targetNWay (e.g., 2-way → 5-way)
    /// - initialKShot should be larger than targetKShot (e.g., 10-shot → 1-shot)
    /// - This creates an easy-to-hard progression
    ///
    /// <b>Example progressions:</b>
    /// - Gentle: 3-way 5-shot → 5-way 3-shot
    /// - Moderate: 2-way 10-shot → 5-way 3-shot
    /// - Aggressive: 2-way 20-shot → 10-way 1-shot
    ///
    /// The loader starts at progress 0.0 (easy). Call SetProgress() to advance the curriculum.
    /// </para>
    /// </remarks>
    public CurriculumEpisodicDataLoader(
        Matrix<T> datasetX,
        Vector<T> datasetY,
        int targetNWay = 5,      // Default final difficulty: 5-way
        int targetKShot = 1,     // Default final difficulty: 1-shot (challenging)
        int queryShots = 15,     // Default: 15 queries
        int initialNWay = 2,     // Default initial difficulty: 2-way (easier)
        int initialKShot = 10,   // Default initial difficulty: 10-shot (easier)
        int? seed = null)
        : base(datasetX, datasetY, targetNWay, targetKShot + initialKShot, queryShots, seed)  // Ensure dataset has enough examples
    {
        // Validate curriculum parameters
        if (initialNWay < 2)
        {
            throw new ArgumentException("initialNWay must be at least 2", nameof(initialNWay));
        }

        if (initialNWay > targetNWay)
        {
            throw new ArgumentException($"initialNWay ({initialNWay}) must be less than or equal to targetNWay ({targetNWay})", nameof(initialNWay));
        }

        if (initialKShot < targetKShot)
        {
            throw new ArgumentException($"initialKShot ({initialKShot}) must be greater than or equal to targetKShot ({targetKShot})", nameof(initialKShot));
        }

        _targetNWay = targetNWay;
        _targetKShot = targetKShot;
        _initialNWay = initialNWay;
        _initialKShot = initialKShot;
        _progress = NumOps.ToInt32(NumOps.Zero);  // Start at 0.0 (easiest)
    }

    /// <summary>
    /// Sets the curriculum progress to control task difficulty.
    /// </summary>
    /// <param name="progress">Progress value from 0.0 (easiest) to 1.0 (target difficulty).</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when progress is outside [0.0, 1.0] range.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method controls where you are in the curriculum.
    ///
    /// Call this method periodically during training to advance the difficulty:
    /// - Progress 0.0: Easiest tasks (initialNWay, initialKShot)
    /// - Progress 0.5: Halfway between easy and hard
    /// - Progress 1.0: Target tasks (targetNWay, targetKShot)
    ///
    /// <b>Common strategies for updating progress:</b>
    /// - Linear: progress = current_episode / total_episodes
    /// - Step-based: increase progress every N episodes
    /// - Performance-based: increase when validation accuracy reaches thresholds
    /// - Adaptive: increase when training loss stabilizes
    ///
    /// The curriculum automatically interpolates between initial and target difficulty.
    /// </para>
    /// </remarks>
    public void SetProgress(double progress)
    {
        if (progress < 0.0 || progress > 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(progress), $"Progress must be between 0.0 and 1.0, got {progress}");
        }

        _progress = progress;
    }

    /// <summary>
    /// Core implementation of curriculum-based N-way K-shot task sampling with progressive difficulty.
    /// </summary>
    /// <returns>A MetaLearningTask with difficulty based on current curriculum progress.</returns>
    /// <remarks>
    /// <para>
    /// This method implements progressive difficulty:
    /// 1. Calculates current N-way and K-shot based on progress (linear interpolation)
    /// 2. Randomly selects currentNWay classes
    /// 3. For each class, samples (currentKShot + queryShots) examples
    /// 4. Shuffles and splits into support and query sets
    /// 5. Constructs and returns MetaLearningTask
    /// </para>
    /// <para><b>For Beginners:</b> The difficulty calculation works like this:
    ///
    /// Assume: initialNWay=2, targetNWay=5, initialKShot=10, targetKShot=1
    ///
    /// At progress 0.0:
    /// - currentNWay = 2 + (5-2) * 0.0 = 2
    /// - currentKShot = 10 + (1-10) * 0.0 = 10
    /// - Result: 2-way 10-shot (easiest)
    ///
    /// At progress 0.5:
    /// - currentNWay = 2 + (5-2) * 0.5 = 3.5 → 4 (rounded)
    /// - currentKShot = 10 + (1-10) * 0.5 = 5.5 → 5 (rounded)
    /// - Result: 4-way 5-shot (medium)
    ///
    /// At progress 1.0:
    /// - currentNWay = 2 + (5-2) * 1.0 = 5
    /// - currentKShot = 10 + (1-10) * 1.0 = 1
    /// - Result: 5-way 1-shot (target/hardest)
    /// </para>
    /// </remarks>
    protected override MetaLearningTask<T> GetNextTaskCore()
    {
        // Step 1: Calculate current difficulty based on progress
        int currentNWay = CalculateCurrentNWay();
        int currentKShot = CalculateCurrentKShot();

        // Step 2: Randomly select currentNWay classes
        var selectedClasses = AvailableClasses
            .OrderBy(_ => Random.Next())
            .Take(currentNWay)
            .ToArray();

        // Step 3: Sample examples for each selected class
        var supportExamples = new List<Vector<T>>();
        var supportLabels = new List<T>();
        var queryExamples = new List<Vector<T>>();
        var queryLabels = new List<T>();

        for (int classIdx = 0; classIdx < selectedClasses.Length; classIdx++)
        {
            int classLabel = selectedClasses[classIdx];
            var classIndices = ClassToIndices[classLabel];

            // Sample (currentKShot + queryShots) examples and shuffle
            var sampledIndices = classIndices
                .OrderBy(_ => Random.Next())
                .Take(currentKShot + QueryShots)
                .ToList();

            // Shuffle the sampled indices to prevent ordering bias
            sampledIndices = sampledIndices.OrderBy(_ => Random.Next()).ToList();

            // Split into support (first currentKShot) and query
            var supportIndices = sampledIndices.Take(currentKShot);
            var queryIndices = sampledIndices.Skip(currentKShot);

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

        // Step 4: Convert to tensors and return
        return BuildMetaLearningTask(supportExamples, supportLabels, queryExamples, queryLabels, currentNWay, currentKShot);
    }

    /// <summary>
    /// Calculates the current N-way based on curriculum progress.
    /// </summary>
    private int CalculateCurrentNWay()
    {
        // Linear interpolation from initialNWay to targetNWay
        double interpolated = _initialNWay + (_targetNWay - _initialNWay) * _progress;
        return (int)Math.Round(interpolated);
    }

    /// <summary>
    /// Calculates the current K-shot based on curriculum progress.
    /// </summary>
    private int CalculateCurrentKShot()
    {
        // Linear interpolation from initialKShot to targetKShot
        double interpolated = _initialKShot + (_targetKShot - _initialKShot) * _progress;
        int result = (int)Math.Round(interpolated);

        // Ensure at least 1 shot
        return Math.Max(1, result);
    }

    /// <summary>
    /// Builds a MetaLearningTask from lists of examples and labels with current curriculum difficulty.
    /// </summary>
    private MetaLearningTask<T> BuildMetaLearningTask(
        List<Vector<T>> supportExamples,
        List<T> supportLabels,
        List<Vector<T>> queryExamples,
        List<T> queryLabels,
        int currentNWay,
        int currentKShot)
    {
        int numFeatures = DatasetX.Columns;
        int supportSize = currentNWay * currentKShot;
        int querySize = currentNWay * QueryShots;

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
