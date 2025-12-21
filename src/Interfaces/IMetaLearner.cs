using AiDotNet.Data.Structures;
using AiDotNet.MetaLearning;
using AiDotNet.MetaLearning.Data;
using AiDotNet.Models;
using AiDotNet.Models.Results;

namespace AiDotNet.Interfaces;

/// <summary>
/// Unified interface for meta-learning algorithms that train models to quickly adapt to new tasks.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The type of input data (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;, double[]).</typeparam>
/// <typeparam name="TOutput">The type of output data (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;, double[]).</typeparam>
/// <remarks>
/// <para>
/// This is the unified interface for all meta-learning algorithms in the framework.
/// It combines both training infrastructure and algorithm capabilities, enabling
/// seamless integration with PredictionModelBuilder while supporting all 17 meta-learning
/// algorithms (MAML, Reptile, ProtoNets, LEO, MetaOptNet, etc.).
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
/// var dataLoader = new UniformEpisodicDataLoader&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(
///     datasetX: trainingFeatures,
///     datasetY: trainingLabels,
///     nWay: 5,          // 5 classes per task
///     kShot: 5,         // 5 support examples per class
///     queryShots: 15    // 15 query examples per class
/// );
///
/// // 2. Configure: Setup meta-learner with options
/// var options = MetaLearnerOptionsBase&lt;double&gt;.CreateBuilder()
///     .WithInnerLearningRate(0.01)
///     .WithOuterLearningRate(0.001)
///     .WithAdaptationSteps(5)
///     .WithMetaBatchSize(4)
///     .WithNumMetaIterations(1000)
///     .Build();
///
/// var metaLearner = new MAMLAlgorithm&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(
///     metaModel: neuralNetwork,
///     lossFunction: new CrossEntropyLoss&lt;double&gt;(),
///     dataLoader: dataLoader,
///     options: options
/// );
///
/// // 3. Meta-Training: Simply call Train()
/// var trainingResult = metaLearner.Train();
///
/// // 4. Deployment: Adapt to new task with 5 examples
/// var newTask = dataLoader.GetNextTask();
/// var adaptResult = metaLearner.AdaptAndEvaluate(newTask);
/// Console.WriteLine($"New Task Accuracy: {adaptResult.QueryAccuracy:P2}");
/// </code>
/// </example>
public interface IMetaLearner<T, TInput, TOutput>
{
    #region Properties

    /// <summary>
    /// Gets the base model being meta-trained.
    /// </summary>
    IFullModel<T, TInput, TOutput> BaseModel { get; }

    /// <summary>
    /// Gets the meta-learner options (configuration).
    /// </summary>
    IMetaLearnerOptions<T> Options { get; }

    /// <summary>
    /// Gets the current meta-training iteration count.
    /// </summary>
    int CurrentIteration { get; }

    /// <summary>
    /// Gets the type of meta-learning algorithm.
    /// </summary>
    MetaLearningAlgorithmType AlgorithmType { get; }

    /// <summary>
    /// Gets the number of adaptation steps to perform during task adaptation (inner loop).
    /// </summary>
    int AdaptationSteps { get; }

    /// <summary>
    /// Gets the learning rate used for task adaptation (inner loop).
    /// </summary>
    double InnerLearningRate { get; }

    /// <summary>
    /// Gets the learning rate used for meta-learning (outer loop).
    /// </summary>
    double OuterLearningRate { get; }

    #endregion

    #region Core Meta-Learning Methods

    /// <summary>
    /// Performs one meta-training step on a batch of tasks.
    /// </summary>
    /// <param name="taskBatch">The batch of tasks to train on.</param>
    /// <returns>The meta-training loss for this batch.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method updates the model by training on multiple tasks at once.
    /// Each task teaches the model something about how to learn quickly. The returned loss value
    /// indicates how well the model is doing - lower is better.
    /// </para>
    /// </remarks>
    T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch);

    /// <summary>
    /// Adapts the model to a new task using its support set.
    /// </summary>
    /// <param name="task">The task to adapt to.</param>
    /// <returns>A new model instance adapted to the task.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is where the "quick learning" happens. Given a new task
    /// with just a few examples (the support set), this method creates a new model that's
    /// specialized for that specific task.
    /// </para>
    /// </remarks>
    IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task);

    /// <summary>
    /// Evaluates the meta-learning algorithm on a batch of tasks.
    /// </summary>
    /// <param name="taskBatch">The batch of tasks to evaluate on.</param>
    /// <returns>The average evaluation loss across all tasks.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This checks how well the meta-learning algorithm performs.
    /// For each task, it adapts using the support set and then tests on the query set.
    /// The returned value is the average loss across all tasks - lower means better performance.
    /// </para>
    /// </remarks>
    T Evaluate(TaskBatch<T, TInput, TOutput> taskBatch);

    #endregion

    #region Training Infrastructure

    /// <summary>
    /// Performs one meta-training step (outer loop update) using the episodic data loader.
    /// </summary>
    /// <remarks>
    /// Uses the episodic data loader configured during construction to sample tasks for this meta-update.
    /// </remarks>
    /// <param name="batchSize">Number of tasks to sample for this meta-update.</param>
    /// <returns>Metrics including meta-loss, task loss, accuracy, and timing information.</returns>
    MetaTrainingStepResult<T> MetaTrainStep(int batchSize);

    /// <summary>
    /// Trains the meta-learner using the configuration specified during construction.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method performs the complete outer-loop meta-training process, repeatedly calling
    /// MetaTrainStep and collecting metrics across all iterations. All training parameters
    /// are specified in the IMetaLearnerOptions provided during construction.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is the main training method for meta-learning. Unlike traditional
    /// training where you train once on a dataset, this trains your model across many different tasks
    /// so it learns how to quickly adapt to new tasks.
    /// </para>
    /// </remarks>
    /// <returns>Complete training history with loss/accuracy progression and timing information.</returns>
    MetaTrainingResult<T> Train();

    /// <summary>
    /// Evaluates meta-learning performance on multiple held-out tasks.
    /// </summary>
    /// <remarks>
    /// Uses the episodic data loader configured during construction to sample evaluation tasks.
    /// </remarks>
    /// <param name="numTasks">Number of tasks to evaluate (100-1000 recommended for statistics).</param>
    /// <returns>Comprehensive metrics including mean accuracy, confidence intervals, and per-task statistics.</returns>
    MetaEvaluationResult<T> Evaluate(int numTasks);

    /// <summary>
    /// Adapts the model to a specific task and evaluates adaptation quality.
    /// </summary>
    /// <param name="task">Meta-learning task with support set (for adaptation) and query set (for evaluation).</param>
    /// <returns>Detailed metrics about adaptation performance and timing.</returns>
    MetaAdaptationResult<T> AdaptAndEvaluate(MetaLearningTask<T, TInput, TOutput> task);

    #endregion

    #region Model Management

    /// <summary>
    /// Gets the current meta-model.
    /// </summary>
    /// <returns>The current meta-model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This returns the "meta-learned" model that has been trained
    /// on many tasks. This model itself may not be very good at any specific task, but it's
    /// excellent as a starting point for quickly adapting to new tasks.
    /// </para>
    /// </remarks>
    IFullModel<T, TInput, TOutput> GetMetaModel();

    /// <summary>
    /// Sets the base model for this meta-learning algorithm.
    /// </summary>
    /// <param name="model">The model to use as the base.</param>
    void SetMetaModel(IFullModel<T, TInput, TOutput> model);

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

    #endregion
}
