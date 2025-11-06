namespace AiDotNet.Interfaces;

/// <summary>
/// Defines the contract for meta-learning algorithms that train models to quickly adapt to new tasks.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Meta-learning, or "learning to learn," is a paradigm where models are trained across multiple
/// tasks to develop the ability to quickly adapt to new, unseen tasks with minimal data.
/// Unlike traditional learning that optimizes for a single task, meta-learning optimizes for
/// rapid adaptation across a distribution of tasks.
/// </para>
/// <para><b>For Beginners:</b> Meta-learning is like teaching someone how to learn, not just what to learn.
///
/// Imagine two students:
/// - <b>Traditional Learning:</b> A student memorizes facts for one specific test. They know that test well,
///   but struggle when given a different test on a new topic.
/// - <b>Meta-Learning:</b> A student learns study techniques by practicing many different types of tests.
///   When given a new test on an unfamiliar topic, they know how to quickly learn and adapt.
///
/// In machine learning:
/// - Traditional models train on lots of data for one task (e.g., classify cats vs dogs)
/// - Meta-learning models train on many small tasks, learning how to quickly adapt to new tasks
///
/// This is especially useful when:
/// - You have limited data for new tasks (few-shot learning)
/// - You need to adapt to changing environments
/// - You want to transfer knowledge across related problems
///
/// Common meta-learning algorithms include:
/// - <b>MAML (Model-Agnostic Meta-Learning):</b> Learns initial parameters that are easy to fine-tune
/// - <b>Reptile:</b> Simpler version of MAML that directly averages task-specific parameters
/// - <b>SEAL:</b> Uses evolutionary strategies for meta-optimization
/// </para>
/// <para>
/// <b>Thread Safety:</b> Implementations are not required to be thread-safe.
/// Create separate instances for concurrent training.
/// </para>
/// </remarks>
public interface IMetaLearner<T>
{
    /// <summary>
    /// Trains the meta-learning model across multiple tasks to develop rapid adaptation capabilities.
    /// </summary>
    /// <param name="dataLoader">The episodic data loader that provides meta-learning tasks.</param>
    /// <param name="numMetaIterations">The number of meta-training iterations to perform.</param>
    /// <returns>Metadata about the meta-training process including loss history and performance metrics.</returns>
    /// <exception cref="ArgumentNullException">Thrown when dataLoader is null.</exception>
    /// <exception cref="ArgumentException">Thrown when numMetaIterations is less than 1.</exception>
    /// <remarks>
    /// <para>
    /// This method implements the meta-training loop, which typically involves:
    /// 1. Sampling a batch of tasks from the data loader
    /// 2. For each task, adapting the model using the support set
    /// 3. Evaluating adaptation quality on the query set
    /// 4. Updating the meta-parameters to improve adaptation performance
    /// </para>
    /// <para><b>For Beginners:</b> This is where the "learning to learn" happens.
    ///
    /// Think of it like training to become a fast learner:
    /// 1. <b>Sample tasks:</b> Get a variety of practice problems (math, science, history, etc.)
    /// 2. <b>Quick study:</b> For each problem set, quickly study the examples (support set)
    /// 3. <b>Take quiz:</b> Test your understanding on new questions from the same topic (query set)
    /// 4. <b>Reflect and improve:</b> Analyze what study techniques worked best and adjust your approach
    /// 5. <b>Repeat:</b> Do this many times with different topics until you become a fast learner
    ///
    /// The meta-training process:
    /// - <b>Input:</b> A data loader that generates N-way K-shot tasks
    /// - <b>Process:</b> Train on many tasks, learning how to quickly adapt from few examples
    /// - <b>Output:</b> A model that can rapidly adapt to new, unseen tasks
    ///
    /// After meta-training, when you encounter a new task with just a few examples,
    /// the model can quickly fine-tune itself to perform well on that task.
    /// </para>
    /// <para>
    /// <b>Performance:</b> Training time depends on numMetaIterations, task complexity,
    /// and the specific meta-learning algorithm. Progress can be monitored through
    /// the returned metadata.
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// // Create episodic data loader for 5-way 3-shot tasks
    /// var dataLoader = new UniformEpisodicDataLoader&lt;double&gt;(
    ///     datasetX: features,
    ///     datasetY: labels,
    ///     nWay: 5,
    ///     kShot: 3
    /// );
    ///
    /// // Create and configure meta-learner
    /// var metaLearner = new ReptileTrainer&lt;double&gt;(
    ///     metaModel: neuralNetwork,
    ///     lossFunction: new MeanSquaredError&lt;double&gt;(),
    ///     innerSteps: 5,
    ///     innerLearningRate: 0.01,
    ///     metaLearningRate: 0.001
    /// );
    ///
    /// // Meta-train for 1000 iterations
    /// var metadata = metaLearner.Train(dataLoader, numMetaIterations: 1000);
    ///
    /// // Now the model can quickly adapt to new tasks with few examples
    /// Console.WriteLine($"Final meta-loss: {metadata.FinalLoss}");
    /// </code>
    /// </example>
    ModelMetadata<T> Train(IEpisodicDataLoader<T> dataLoader, int numMetaIterations);
}
