using AiDotNet.Data.Abstractions;
using AiDotNet.Models.Results;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for meta-learning algorithms that train models to quickly adapt to new tasks.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The type of input data (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;, double[]).</typeparam>
/// <typeparam name="TOutput">The type of output data (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;, double[]).</typeparam>
/// <remarks>
/// <para>
/// Meta-learning, or "learning to learn," trains models across multiple tasks to develop
/// rapid adaptation capabilities. This enables few-shot learning where models can learn
/// new tasks from just a handful of examples.
/// </para>
/// <para><b>For Beginners:</b> Meta-learning is like teaching someone how to learn, not just what to learn.
///
/// Traditional vs Meta-Learning:
/// - <b>Traditional:</b> Train on thousands of cat/dog images → classify cats vs dogs well
/// - <b>Meta-Learning:</b> Train on many classification tasks → learn ANY new category from 5 examples
///
/// Real-world applications:
/// - Few-shot image classification (recognize new objects from 1-5 images)
/// - Rapid robot adaptation (new environments with minimal data)
/// - Personalized recommendations (adapt to new users quickly)
/// - Drug discovery (predict properties of new molecules)
/// </para>
/// <para>
/// <b>Architecture - Two-Loop Optimization:</b>
///
/// <b>Inner Loop (Task Adaptation):</b>
/// - Given: New task with support set (K examples per class)
/// - Process: Few gradient steps (1-10) to adapt model
/// - Output: Task-specific adapted parameters
/// - Goal: Quickly learn this specific task
///
/// <b>Outer Loop (Meta-Optimization):</b>
/// - Given: Batch of tasks from task distribution
/// - Process: For each task, adapt (inner loop) and evaluate on query set
/// - Output: Updated meta-parameters
/// - Goal: Learn parameters that enable fast adaptation across all tasks
///
/// This two-loop structure is what enables "learning to learn."
/// </para>
/// <para>
/// <b>Production Considerations:</b>
/// - Use MetaTrainStep() for training loop with proper batch sizes (2-32 tasks)
/// - Monitor Evaluate() metrics every N iterations to detect overfitting
/// - Use AdaptAndEvaluate() for deployment to quickly adapt to new tasks
/// - Save/Load models after meta-training for deployment
/// - Thread Safety: Not thread-safe, use separate instances for concurrent training
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // 1. Setup: Create episodic data loader for 5-way 5-shot tasks
/// var dataLoader = new UniformEpisodicDataLoader&lt;double&gt;(
///     datasetX: trainingFeatures,
///     datasetY: trainingLabels,
///     nWay: 5,          // 5 classes per class
///     kShot: 5,         // 5 support examples per class
///     queryShots: 15    // 15 query examples per class
/// );
///
/// // 2. Configure: Setup meta-learner (Reptile example)
/// var config = new ReptileTrainerConfig&lt;double&gt;
/// {
///     InnerLearningRate = 0.01,      // Task adaptation rate
///     MetaLearningRate = 0.001,      // Meta-optimization rate
///     InnerSteps = 5,                // Gradient steps per task
///     MetaBatchSize = 4              // Tasks per meta-update
/// };
///
/// var metaLearner = new ReptileTrainer&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;(
///     metaModel: neuralNetwork,
///     lossFunction: new CrossEntropyLoss&lt;double&gt;(),
///     config: config
/// );
///
/// // 3. Meta-Training: Outer loop for 1000 iterations
/// for (int iter = 0; iter &lt; 1000; iter++)
/// {
///     // One meta-update on batch of 4 tasks
///     var stepResult = metaLearner.MetaTrainStep(dataLoader, batchSize: 4);
///
///     Console.WriteLine($"Iter {iter}: MetaLoss={stepResult.MetaLoss:F4}, " +
///                      $"Accuracy={stepResult.Accuracy:P2}");
///
///     // Periodic evaluation on held-out tasks
///     if (iter % 100 == 0)
///     {
///         var evalResult = metaLearner.Evaluate(evalDataLoader, numTasks: 100);
///         Console.WriteLine($"Eval: {evalResult.AccuracyStats.Mean:P2} ± {evalResult.AccuracyStats.StandardDeviation:P2}");
///     }
/// }
///
/// // 4. Save meta-trained model
/// metaLearner.Save("meta_model.bin");
///
/// // 5. Deployment: Adapt to new task with 5 examples
/// var newTask = dataLoader.GetNextTask();  // Unseen task
/// var adaptResult = metaLearner.AdaptAndEvaluate(newTask);
/// Console.WriteLine($"New Task Accuracy: {adaptResult.QueryAccuracy:P2}");
/// // Expected: High accuracy (>70%) from just 5 examples per class!
/// </code>
/// </example>
public interface IMetaLearner<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the base model being meta-trained.
    /// </summary>
    IFullModel<T, TInput, TOutput> BaseModel { get; }

    /// <summary>
    /// Gets the meta-learner configuration.
    /// </summary>
    IMetaLearnerConfig<T> Config { get; }

    /// <summary>
    /// Gets the current meta-training iteration count.
    /// </summary>
    int CurrentIteration { get; }

    /// <summary>
    /// Performs one meta-training step (outer loop update) on a batch of tasks.
    /// </summary>
    /// <param name="dataLoader">The episodic data loader providing meta-learning tasks.</param>
    /// <param name="batchSize">Number of tasks to sample for this meta-update.</param>
    /// <returns>Metrics including meta-loss, task loss, accuracy, and timing information.</returns>
    MetaTrainingStepResult<T> MetaTrainStep(IEpisodicDataLoader<T> dataLoader, int batchSize);

    /// <summary>
    /// Evaluates meta-learning performance on multiple held-out tasks.
    /// </summary>
    /// <param name="dataLoader">Episodic data loader providing evaluation tasks.</param>
    /// <param name="numTasks">Number of tasks to evaluate (100-1000 recommended for statistics).</param>
    /// <returns>Comprehensive metrics including mean accuracy, confidence intervals, and per-task statistics.</returns>
    MetaEvaluationResult<T> Evaluate(IEpisodicDataLoader<T> dataLoader, int numTasks);

    /// <summary>
    /// Adapts the model to a specific task and evaluates adaptation quality.
    /// </summary>
    /// <param name="task">Meta-learning task with support set (for adaptation) and query set (for evaluation).</param>
    /// <returns>Detailed metrics about adaptation performance and timing.</returns>
    MetaAdaptationResult<T> AdaptAndEvaluate(MetaLearningTask<T> task);

    /// <summary>
    /// Saves the meta-trained model to disk for later deployment.
    /// </summary>
    /// <param name="filePath">File path where model should be saved.</param>
    void Save(string filePath);

    /// <summary>
    /// Loads a previously meta-trained model from disk.
    /// </summary>
    /// <param name="filePath">File path to the saved model.</param>
    void Load(string filePath);

    /// <summary>
    /// Resets the meta-learner to initial untrained state.
    /// </summary>
    void Reset();
}
