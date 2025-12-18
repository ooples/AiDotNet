using AiDotNet.Interfaces;

namespace AiDotNet.Models.Options;

/// <summary>
/// Base configuration options for meta-learning algorithms.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix<T>, Tensor<T>).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector<T>, Tensor<T>).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Meta-learning algorithms learn how to learn quickly from a few examples.
/// This class contains the common configuration settings that all meta-learning algorithms share.
///
/// Key concepts:
/// - Inner Loop: The fast adaptation to a specific task (using the support set)
/// - Outer Loop: The meta-learning update that improves the ability to adapt
/// - Adaptation Steps: How many gradient steps to take when adapting to a new task
/// </para>
/// </remarks>
public class MetaLearningAlgorithmOptions<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the base model to use for meta-learning.
    /// </summary>
    /// <value>The base model, typically a neural network.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the model that will learn to adapt quickly.
    /// It's usually a neural network that starts with random parameters and learns
    /// good initial parameters that can be quickly adapted to new tasks.
    /// </para>
    /// </remarks>
    public IFullModel<T, TInput, TOutput>? BaseModel { get; set; }

    /// <summary>
    /// Gets or sets the learning rate for task adaptation (inner loop).
    /// </summary>
    /// <value>The inner learning rate, defaulting to 0.01.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This controls how fast the model adapts to a specific task.
    /// A higher value means faster adaptation but less stability. The "inner loop" refers
    /// to the process of adapting to each individual task.
    /// </para>
    /// </remarks>
    public double InnerLearningRate { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the learning rate for meta-learning (outer loop).
    /// </summary>
    /// <value>The outer learning rate, defaulting to 0.001.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This controls how fast the meta-learner updates its knowledge
    /// about how to learn. The "outer loop" refers to the process of learning from multiple
    /// tasks to become better at adapting in general.
    /// </para>
    /// </remarks>
    public double OuterLearningRate { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the number of gradient steps for task adaptation.
    /// </summary>
    /// <value>The number of adaptation steps, defaulting to 5.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is how many times the model updates itself when adapting
    /// to a new task. More steps mean more thorough adaptation but take longer.
    /// For few-shot learning, this is typically a small number (1-10).
    /// </para>
    /// </remarks>
    public int AdaptationSteps { get; set; } = 5;

    /// <summary>
    /// Gets or sets whether to use first-order approximation.
    /// </summary>
    /// <value>True to use first-order approximation (faster but less accurate), false otherwise.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is a performance trade-off. When true, the algorithm
    /// uses a simpler (faster) way to calculate gradients, which speeds up training
    /// but may be slightly less accurate. This is often fine in practice and significantly
    /// faster, especially for deep networks.
    /// </para>
    /// </remarks>
    public bool UseFirstOrder { get; set; } = false;

    /// <summary>
    /// Gets or sets the loss function to use for meta-learning.
    /// </summary>
    /// <value>The loss function.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The loss function measures how wrong the model's predictions are.
    /// The meta-learning algorithm tries to minimize this loss. Common choices are mean squared
    /// error for regression or cross-entropy for classification.
    /// </para>
    /// </remarks>
    public ILossFunction<T>? LossFunction { get; set; }

    /// <summary>
    /// Gets or sets the random seed for reproducibility.
    /// </summary>
    /// <value>The random seed value, or null for non-deterministic behavior.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Setting a random seed ensures you get the same results every time
    /// you run the algorithm. This is useful for debugging and comparing different approaches.
    /// If null, the algorithm will produce different results each run.
    /// </para>
    /// </remarks>
    public int? RandomSeed { get; set; }

    /// <summary>
    /// Gets or sets the batch size for meta-training (number of tasks per meta-update).
    /// </summary>
    /// <value>The meta-batch size, defaulting to 4.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is how many tasks the algorithm learns from before
    /// updating its meta-parameters. More tasks per batch gives more stable updates but
    /// requires more memory. 4-32 tasks per batch is typical.
    /// </para>
    /// </remarks>
    public int MetaBatchSize { get; set; } = 4;

    /// <summary>
    /// Gets or sets whether to track gradients for debugging.
    /// </summary>
    /// <value>True to track gradients, false otherwise.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When enabled, the algorithm saves gradient information
    /// that can help diagnose training problems. This slows down training and uses more
    /// memory, so it's typically only used for debugging.
    /// </para>
    /// </remarks>
    public bool TrackGradients { get; set; } = false;

    /// <summary>
    /// Gets or sets the optimizer for meta-learning (outer loop).
    /// </summary>
    /// <value>The meta-optimizer. If null, Adam optimizer will be used.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This optimizer updates the meta-parameters that enable
    /// quick adaptation to new tasks. Different optimizers work better for different
    /// scenarios:
    /// - Adam: Good default choice with adaptive learning rates
    /// - SGD: Simple and effective for some tasks
    /// - RMSProp: Works well with non-stationary objectives
    /// </para>
    /// </remarks>
    public IGradientBasedOptimizer<T, TInput, TOutput>? MetaOptimizer { get; set; }

    /// <summary>
    /// Gets or sets the optimizer for task adaptation (inner loop).
    /// </summary>
    /// <value>The inner optimizer. If null, Adam optimizer will be used.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This optimizer handles the quick adaptation to each
    /// specific task during meta-training. You can use the same optimizer type as
    /// MetaOptimizer or choose a different one based on your needs.
    /// </para>
    /// </remarks>
    public IGradientBasedOptimizer<T, TInput, TOutput>? InnerOptimizer { get; set; }
}
