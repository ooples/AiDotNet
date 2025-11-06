using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.MetaLearning.Trainers;

/// <summary>
/// Provides a base implementation for Reptile meta-learning trainers with common functionality and validation.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This abstract class implements the IMetaLearner interface and provides common functionality
/// for Reptile-based meta-learning trainers. It handles parameter validation, loss computation,
/// and configuration management while allowing derived classes to focus on implementing the
/// specific Reptile algorithm variant.
/// </para>
/// <para><b>For Beginners:</b> This is the foundation that Reptile trainers build upon.
///
/// Reptile is a simple yet powerful meta-learning algorithm published by OpenAI. It works by:
/// 1. Starting with initial model parameters
/// 2. For each task: clone the model, train it on that task, then move the original parameters
///    toward the task-specific parameters
/// 3. After many tasks, the model learns to find parameter values that are easy to fine-tune
///
/// Why Reptile is special:
/// - <b>Simpler than MAML:</b> No second-order derivatives needed
/// - <b>Effective:</b> Achieves comparable performance to MAML in many scenarios
/// - <b>Scalable:</b> Works well with standard gradient descent optimizers
///
/// This base class handles:
/// - Storing and validating configuration (learning rates, number of steps, etc.)
/// - Providing access to the model, loss function, and numeric operations
/// - Ensuring proper parameter validation
/// - Offering protected helper methods for derived classes
/// </para>
/// <para>
/// <b>Thread Safety:</b> This class is not thread-safe. Create separate instances for concurrent training.
/// </para>
/// </remarks>
public abstract class ReptileTrainerBase<T> : IMetaLearner<T>
{
    /// <summary>
    /// The model being meta-trained.
    /// </summary>
    protected readonly IFullModel<T, Tensor<T>, Tensor<T>> MetaModel;

    /// <summary>
    /// The loss function used to evaluate task performance.
    /// </summary>
    protected readonly ILossFunction<T> LossFunction;

    /// <summary>
    /// The number of gradient descent steps to perform on each task's support set.
    /// </summary>
    protected readonly int InnerSteps;

    /// <summary>
    /// The learning rate for inner loop optimization (task-specific training).
    /// </summary>
    protected readonly T InnerLearningRate;

    /// <summary>
    /// The meta-learning rate (epsilon) that controls how much to move toward task-adapted parameters.
    /// </summary>
    protected readonly T MetaLearningRate;

    /// <summary>
    /// Provides mathematical operations for the numeric type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Initializes a new instance of the ReptileTrainerBase class with industry-standard defaults.
    /// </summary>
    /// <param name="metaModel">The model to be meta-trained. Must implement IFullModel for parameter access.</param>
    /// <param name="lossFunction">The loss function used to evaluate predictions during training.</param>
    /// <param name="innerSteps">The number of gradient steps per task. Default is 5 (common in meta-learning).</param>
    /// <param name="innerLearningRate">The learning rate for task-specific training. Default is 0.01.</param>
    /// <param name="metaLearningRate">The meta-learning rate for Reptile updates. Default is 0.001.</param>
    /// <exception cref="ArgumentNullException">Thrown when metaModel or lossFunction is null.</exception>
    /// <exception cref="ArgumentException">Thrown when configuration parameters are invalid.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This constructor sets up the Reptile trainer with all necessary components.
    ///
    /// <b>Parameters explained:</b>
    /// - <b>metaModel:</b> The neural network (or other model) you want to meta-train. After meta-training,
    ///   this model will be good at quickly adapting to new tasks.
    ///
    /// - <b>lossFunction:</b> Measures how wrong the model's predictions are. Common choices:
    ///   - Mean Squared Error for regression (predicting numbers)
    ///   - Cross Entropy for classification (predicting categories)
    ///
    /// - <b>innerSteps:</b> How many times to update the model on each task's support set.
    ///   - More steps = better task-specific performance, but slower training
    ///   - Typical range: 3-10 steps
    ///   - Default: 5 steps (good balance)
    ///
    /// - <b>innerLearningRate:</b> How big of a step to take when training on each task.
    ///   - Larger = faster adaptation but risk of instability
    ///   - Smaller = more stable but slower adaptation
    ///   - Default: 0.01 (conservative, stable choice)
    ///
    /// - <b>metaLearningRate:</b> How much to update the meta-parameters toward task-specific parameters.
    ///   - This is Reptile's "epsilon" parameter
    ///   - Larger = faster meta-learning but risk of forgetting
    ///   - Smaller = more stable but slower meta-learning
    ///   - Default: 0.001 (10x smaller than inner LR, common practice)
    ///
    /// <b>Typical usage:</b>
    /// Start with default parameters and adjust based on results:
    /// - If meta-training is unstable: reduce metaLearningRate
    /// - If adaptation is poor: increase innerSteps or innerLearningRate
    /// - If meta-training is too slow: increase metaLearningRate
    /// </para>
    /// </remarks>
    protected ReptileTrainerBase(
        IFullModel<T, Tensor<T>, Tensor<T>> metaModel,
        ILossFunction<T> lossFunction,
        int innerSteps = 5,
        double innerLearningRate = 0.01,
        double metaLearningRate = 0.001)
    {
        // Validate inputs
        ValidateConstructorParameters(metaModel, lossFunction, innerSteps, innerLearningRate, metaLearningRate);

        MetaModel = metaModel;
        LossFunction = lossFunction;
        InnerSteps = innerSteps;
        InnerLearningRate = NumOps.FromDouble(innerLearningRate);
        MetaLearningRate = NumOps.FromDouble(metaLearningRate);
    }

    /// <summary>
    /// Trains the meta-learning model across multiple tasks to develop rapid adaptation capabilities.
    /// </summary>
    /// <param name="dataLoader">The episodic data loader that provides meta-learning tasks.</param>
    /// <param name="numMetaIterations">The number of meta-training iterations to perform.</param>
    /// <returns>Metadata about the meta-training process including loss history and performance metrics.</returns>
    /// <exception cref="ArgumentNullException">Thrown when dataLoader is null.</exception>
    /// <exception cref="ArgumentException">Thrown when numMetaIterations is less than 1.</exception>
    public ModelMetadata<T> Train(IEpisodicDataLoader<T> dataLoader, int numMetaIterations)
    {
        // Validate parameters
        if (dataLoader == null)
        {
            throw new ArgumentNullException(nameof(dataLoader), "Data loader cannot be null");
        }

        if (numMetaIterations < 1)
        {
            throw new ArgumentException("Number of meta-iterations must be at least 1", nameof(numMetaIterations));
        }

        // Delegate to derived class implementation
        return TrainCore(dataLoader, numMetaIterations);
    }

    /// <summary>
    /// Core meta-training logic to be implemented by derived classes.
    /// </summary>
    /// <param name="dataLoader">The episodic data loader that provides meta-learning tasks.</param>
    /// <param name="numMetaIterations">The number of meta-training iterations to perform.</param>
    /// <returns>Metadata about the meta-training process.</returns>
    /// <remarks>
    /// <para><b>For Implementers:</b> This is where you implement the Reptile algorithm.
    ///
    /// You have access to:
    /// - MetaModel: The model being meta-trained
    /// - LossFunction: For computing loss
    /// - InnerSteps: Number of steps per task
    /// - InnerLearningRate: Learning rate for inner loop
    /// - MetaLearningRate: Epsilon for Reptile updates
    /// - NumOps: For numeric operations
    ///
    /// Your implementation should:
    /// 1. For each meta-iteration:
    ///    a. Sample a task from the data loader
    ///    b. Clone the current model parameters
    ///    c. Train on the task's support set for InnerSteps
    ///    d. Compute the difference between adapted and original parameters
    ///    e. Update meta-parameters: θ ← θ + ε(θ_adapted - θ)
    /// 2. Track and return training metrics
    /// </para>
    /// </remarks>
    protected abstract ModelMetadata<T> TrainCore(IEpisodicDataLoader<T> dataLoader, int numMetaIterations);

    /// <summary>
    /// Validates constructor parameters.
    /// </summary>
    private void ValidateConstructorParameters(
        IFullModel<T, Tensor<T>, Tensor<T>> metaModel,
        ILossFunction<T> lossFunction,
        int innerSteps,
        double innerLearningRate,
        double metaLearningRate)
    {
        if (metaModel == null)
        {
            throw new ArgumentNullException(nameof(metaModel), "Meta-model cannot be null");
        }

        if (lossFunction == null)
        {
            throw new ArgumentNullException(nameof(lossFunction), "Loss function cannot be null");
        }

        if (innerSteps < 1)
        {
            throw new ArgumentException("Inner steps must be at least 1", nameof(innerSteps));
        }

        if (innerLearningRate <= 0)
        {
            throw new ArgumentException("Inner learning rate must be positive", nameof(innerLearningRate));
        }

        if (metaLearningRate <= 0)
        {
            throw new ArgumentException("Meta learning rate must be positive", nameof(metaLearningRate));
        }
    }

    /// <summary>
    /// Computes the loss on a given dataset.
    /// </summary>
    /// <param name="model">The model to evaluate.</param>
    /// <param name="inputX">The input features.</param>
    /// <param name="targetY">The target labels.</param>
    /// <returns>The computed loss value.</returns>
    /// <remarks>
    /// <para><b>For Implementers:</b> This helper method evaluates model performance on a dataset.
    /// It flattens the predictions and targets into vectors for the loss function.
    /// </para>
    /// </remarks>
    protected T ComputeLoss(IFullModel<T, Tensor<T>, Tensor<T>> model, Tensor<T> inputX, Tensor<T> targetY)
    {
        var predictions = model.Predict(inputX);
        var predictedVector = predictions.Flatten();
        var targetVector = targetY.Flatten();
        return LossFunction.CalculateLoss(predictedVector, targetVector);
    }
}
