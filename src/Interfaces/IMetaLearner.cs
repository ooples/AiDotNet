using AiDotNet.Data.Abstractions;
using AiDotNet.MetaLearning.Metrics;

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
///     nWay: 5,          // 5 classes per task
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
///     var metrics = metaLearner.MetaTrainStep(dataLoader, batchSize: 4);
///
///     Console.WriteLine($"Iter {iter}: Meta-Loss={metrics.MetaLoss:F4}, " +
///                      $"Accuracy={metrics.Accuracy:P2}");
///
///     // Periodic evaluation on held-out tasks
///     if (iter % 100 == 0)
///     {
///         var evalMetrics = metaLearner.Evaluate(evalDataLoader, numTasks: 100);
///         Console.WriteLine($"Eval: {evalMetrics.Accuracy:P2} ± {evalMetrics.AccuracyStd:P2}");
///     }
/// }
///
/// // 4. Save meta-trained model
/// metaLearner.Save("meta_model.bin");
///
/// // 5. Deployment: Adapt to new task with 5 examples
/// var newTask = dataLoader.GetNextTask();  // Unseen task
/// var adaptMetrics = metaLearner.AdaptAndEvaluate(newTask);
/// Console.WriteLine($"New Task Accuracy: {adaptMetrics.QueryAccuracy:P2}");
/// // Expected: High accuracy (>70%) from just 5 examples per class!
/// </code>
/// </example>
public interface IMetaLearner<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the base model being meta-trained.
    /// </summary>
    /// <value>
    /// The model that will be optimized for rapid task adaptation.
    /// After meta-training, this model's parameters serve as a strong initialization
    /// for quick fine-tuning to new tasks.
    /// </value>
    /// <remarks>
    /// <para><b>For Production:</b> This is the model you'll deploy. After meta-training,
    /// clone it and adapt the clone to new tasks, keeping the original as the meta-initialization.
    /// </para>
    /// </remarks>
    IFullModel<T, TInput, TOutput> BaseModel { get; }

    /// <summary>
    /// Gets the meta-learner configuration.
    /// </summary>
    /// <value>
    /// Configuration containing inner/outer learning rates, inner steps, meta-batch size,
    /// and algorithm-specific settings.
    /// </value>
    IMetaLearnerConfig<T> Config { get; }

    /// <summary>
    /// Gets the current meta-training iteration count.
    /// </summary>
    /// <value>
    /// Number of outer loop updates performed. Useful for:
    /// - Learning rate schedules
    /// - Early stopping criteria
    /// - Progress monitoring
    /// </value>
    int CurrentIteration { get; }

    /// <summary>
    /// Performs one meta-training step (outer loop update) on a batch of tasks.
    /// </summary>
    /// <param name="dataLoader">The episodic data loader providing meta-learning tasks.</param>
    /// <param name="batchSize">Number of tasks to sample for this meta-update.</param>
    /// <returns>Metrics including meta-loss, task loss, accuracy, and timing information.</returns>
    /// <exception cref="ArgumentNullException">Thrown when dataLoader is null.</exception>
    /// <exception cref="ArgumentException">Thrown when batchSize &lt; 1.</exception>
    /// <remarks>
    /// <para>
    /// <b>Algorithm:</b>
    /// <code>
    /// MetaTrainStep(dataLoader, batchSize):
    ///   1. meta_gradients = []
    ///   2. for i = 1 to batchSize:
    ///        a. task = dataLoader.GetNextTask()
    ///        b. θ_adapted = InnerLoop(task.SupportSet, θ_meta)  // Adapt to task
    ///        c. loss = Evaluate(task.QuerySet, θ_adapted)        // Measure adaptation quality
    ///        d. grad = ComputeMetaGradient(θ_meta, θ_adapted, loss)
    ///        e. meta_gradients.append(grad)
    ///   3. θ_meta = θ_meta - meta_lr * Average(meta_gradients)  // Meta-update
    ///   4. return metrics
    /// </code>
    /// </para>
    /// <para><b>For Beginners:</b> This is where "learning to learn" happens.
    ///
    /// Process:
    /// 1. Sample several practice tasks (batch)
    /// 2. For each task:
    ///    - Try to solve it with current approach (inner loop)
    ///    - Check how well you did (query set evaluation)
    /// 3. Analyze: What approach works best across all tasks?
    /// 4. Update your meta-strategy (outer loop update)
    ///
    /// After many iterations, you've learned strategies that work across diverse tasks.
    /// </para>
    /// <para>
    /// <b>Production Guidelines:</b>
    /// - <b>Batch Size:</b> 2-16 for fast iteration, 16-32 for stable gradients
    /// - <b>Monitoring:</b> Log metrics every iteration, save checkpoints every N iterations
    /// - <b>Memory:</b> Each task in batch requires model clone, manage memory accordingly
    /// - <b>Performance:</b> O(B × K × P) - B: batch size, K: inner steps, P: parameters
    /// </para>
    /// </remarks>
    MetaTrainingMetrics MetaTrainStep(IEpisodicDataLoader<T> dataLoader, int batchSize);

    /// <summary>
    /// Evaluates meta-learning performance on multiple held-out tasks.
    /// </summary>
    /// <param name="dataLoader">Episodic data loader providing evaluation tasks.</param>
    /// <param name="numTasks">Number of tasks to evaluate (100-1000 recommended for statistics).</param>
    /// <returns>Comprehensive metrics including mean accuracy, confidence intervals, and per-task statistics.</returns>
    /// <exception cref="ArgumentNullException">Thrown when dataLoader is null.</exception>
    /// <exception cref="ArgumentException">Thrown when numTasks &lt; 1.</exception>
    /// <remarks>
    /// <para>
    /// Evaluation measures adaptation quality on unseen tasks from the meta-test set.
    /// This is critical for assessing whether the model learned general adaptation strategies
    /// or just memorized the meta-training task distribution.
    /// </para>
    /// <para><b>For Beginners:</b> After meta-training, test on completely new tasks.
    ///
    /// Good meta-learning shows:
    /// - High accuracy (>70% for 5-shot, >85% for 10-shot)
    /// - Low variance (consistent across different task types)
    /// - Tight confidence intervals (reliable performance estimates)
    ///
    /// Poor meta-learning shows:
    /// - Low accuracy (overfitted to meta-training tasks)
    /// - High variance (works on some tasks, fails on others)
    /// - Wide confidence intervals (unreliable, need more eval tasks)
    /// </para>
    /// <para>
    /// <b>Production Guidelines:</b>
    /// - Run evaluation every 50-100 meta-training iterations
    /// - Use separate meta-validation and meta-test sets
    /// - Minimum 100 tasks for reliable statistics, 1000+ for publication
    /// - Track per-task metrics to identify failure modes
    /// - Use confidence intervals for deployment decisions
    /// - Early stopping: stop if validation accuracy stops improving
    /// </para>
    /// </remarks>
    MetaEvaluationMetrics Evaluate(IEpisodicDataLoader<T> dataLoader, int numTasks);

    /// <summary>
    /// Adapts the model to a specific task and evaluates adaptation quality.
    /// </summary>
    /// <param name="task">Meta-learning task with support set (for adaptation) and query set (for evaluation).</param>
    /// <returns>Detailed metrics about adaptation performance and timing.</returns>
    /// <exception cref="ArgumentNullException">Thrown when task is null.</exception>
    /// <remarks>
    /// <para>
    /// This is the end goal of meta-learning: given a new task with K examples per class,
    /// quickly adapt and achieve high accuracy. This method demonstrates the model's
    /// few-shot learning capability.
    /// </para>
    /// <para><b>For Beginners:</b> This is meta-learning in action!
    ///
    /// Given: New task you've never seen, with just 5 examples per category
    /// Process:
    /// 1. Start from meta-learned parameters (good initialization)
    /// 2. Do 5 gradient steps on those 5 examples (inner loop)
    /// 3. Test on new examples from same task (query set)
    /// 4. Achieve surprisingly high accuracy from so few examples!
    ///
    /// This is impossible with random initialization but works with meta-learning.
    /// </para>
    /// <para>
    /// <b>Production Use Cases:</b>
    /// - <b>Few-Shot Classification:</b> New product categories with limited images
    /// - <b>Personalization:</b> Adapt to new user with minimal interaction data
    /// - <b>Domain Adaptation:</b> Quick adaptation to new domains/environments
    /// - <b>Online Learning:</b> Continuously adapt as task distribution shifts
    ///
    /// <b>Deployment Pattern:</b>
    /// <code>
    /// // Load meta-trained model once at startup
    /// metaLearner.Load("production_model.bin");
    ///
    /// // For each new task in production:
    /// var newTask = CollectFewShotExamples();  // Get K examples from user/environment
    /// var metrics = metaLearner.AdaptAndEvaluate(newTask);
    ///
    /// if (metrics.QueryAccuracy > 0.7)  // Quality check
    /// {
    ///     DeployAdaptedModel();  // Use adapted model for this task
    /// }
    /// else
    /// {
    ///     FallbackStrategy();  // Need more examples or different approach
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    AdaptationMetrics AdaptAndEvaluate(MetaLearningTask<T> task);

    /// <summary>
    /// Saves the meta-trained model to disk for later deployment.
    /// </summary>
    /// <param name="filePath">File path where model should be saved.</param>
    /// <exception cref="ArgumentException">Thrown when filePath is null or empty.</exception>
    /// <exception cref="IOException">Thrown when file cannot be written.</exception>
    /// <remarks>
    /// <para>
    /// Saves the complete meta-trained model including:
    /// - Model parameters (the meta-learned initialization)
    /// - Configuration (inner LR, steps, etc.)
    /// - Training metadata (iterations, best metrics, etc.)
    /// </para>
    /// <para><b>For Production:</b> Save models after meta-training completes.
    /// These serve as the foundation for deployment - load once at startup,
    /// then rapidly adapt to each new task.
    /// </para>
    /// </remarks>
    void Save(string filePath);

    /// <summary>
    /// Loads a previously meta-trained model from disk.
    /// </summary>
    /// <param name="filePath">File path to the saved model.</param>
    /// <exception cref="ArgumentException">Thrown when filePath is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown when file doesn't exist.</exception>
    /// <exception cref="IOException">Thrown when file is corrupted or incompatible.</exception>
    /// <remarks>
    /// <para>
    /// Loads a meta-trained model for immediate use in task adaptation.
    /// No meta-training required after loading.
    /// </para>
    /// <para><b>For Production:</b> Load pre-trained meta-models at deployment.
    /// This enables instant few-shot learning without expensive meta-training.
    /// </para>
    /// </remarks>
    void Load(string filePath);

    /// <summary>
    /// Resets the meta-learner to initial untrained state.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Resets:
    /// - Model parameters to random initialization
    /// - Iteration counter to 0
    /// - Optimizer state (if applicable)
    ///
    /// Use for:
    /// - Multiple training runs with different hyperparameters
    /// - Learning rate warm restarts
    /// - Ensemble training (multiple meta-learned initializations)
    /// </para>
    /// </remarks>
    void Reset();
}
