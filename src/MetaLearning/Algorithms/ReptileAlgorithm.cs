using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of the Reptile meta-learning algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// Reptile is a simple and scalable meta-learning algorithm. Unlike MAML, it doesn't require
/// computing gradients through the adaptation process, making it more efficient and easier
/// to implement while achieving competitive performance.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// 1. Sample a task (or batch of tasks)
/// 2. Perform SGD on the task starting from the current meta-parameters
/// 3. Update meta-parameters by interpolating toward the adapted parameters
/// 4. Repeat
/// </para>
/// <para>
/// <b>For Beginners:</b> Reptile is like learning by averaging your experiences.
/// </para>
/// <para>
/// Imagine learning to cook:
/// - You start with basic knowledge (initial parameters)
/// - You make a specific dish and learn specific techniques
/// - Instead of just remembering that one dish, you update your basic knowledge
///   to include some of what you learned
/// - After cooking many dishes, your basic knowledge becomes really good
///   for learning any new recipe quickly
/// </para>
/// <para>
/// Reptile is simpler than MAML because it just moves toward adapted parameters
/// instead of computing complex gradients through the adaptation process.
/// The key insight is that this simple approach achieves similar performance
/// to more complex methods like MAML.
/// </para>
/// <para>
/// Reference: Nichol, A., Achiam, J., &amp; Schulman, J. (2018).
/// On first-order meta-learning algorithms.
/// </para>
/// </remarks>
public class ReptileAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly ReptileOptions<T, TInput, TOutput> _reptileOptions;

    /// <summary>
    /// Initializes a new instance of the ReptileAlgorithm class.
    /// </summary>
    /// <param name="options">Reptile configuration options containing the model and all hyperparameters.</param>
    /// <exception cref="ArgumentNullException">Thrown when options is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when required components are not set in options.</exception>
    /// <example>
    /// <code>
    /// // Create Reptile with minimal configuration
    /// var options = new ReptileOptions&lt;double, Tensor, Tensor&gt;(myNeuralNetwork);
    /// var reptile = new ReptileAlgorithm&lt;double, Tensor, Tensor&gt;(options);
    ///
    /// // Create Reptile with custom configuration
    /// var options = new ReptileOptions&lt;double, Tensor, Tensor&gt;(myNeuralNetwork)
    /// {
    ///     AdaptationSteps = 10,
    ///     InnerBatches = 2,
    ///     Interpolation = 0.5,
    ///     OuterLearningRate = 0.1
    /// };
    /// var reptile = new ReptileAlgorithm&lt;double, Tensor, Tensor&gt;(options);
    /// </code>
    /// </example>
    public ReptileAlgorithm(ReptileOptions<T, TInput, TOutput> options)
        : base(
            options?.MetaModel ?? throw new ArgumentNullException(nameof(options), "MetaModel must be set in options."),
            options.LossFunction ?? options.MetaModel.DefaultLossFunction,
            options,
            options.DataLoader,
            options.MetaOptimizer,
            options.InnerOptimizer)
    {
        _reptileOptions = options;
    }

    /// <summary>
    /// Gets the algorithm type identifier for this meta-learner.
    /// </summary>
    /// <value>Returns <see cref="MetaLearningAlgorithmType.Reptile"/>.</value>
    /// <remarks>
    /// <para>
    /// This property identifies the algorithm as Reptile, a first-order
    /// meta-learning algorithm that uses parameter interpolation instead
    /// of gradient-based meta-updates.
    /// </para>
    /// </remarks>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.Reptile;

    /// <summary>
    /// Performs one meta-training step using Reptile's parameter interpolation approach.
    /// </summary>
    /// <param name="taskBatch">A batch of tasks to meta-train on, each containing support and query sets.</param>
    /// <returns>The average loss across all tasks in the batch (evaluated on query sets).</returns>
    /// <exception cref="ArgumentException">Thrown when the task batch is null or empty.</exception>
    /// <exception cref="InvalidOperationException">Thrown when parameter update computation fails.</exception>
    /// <remarks>
    /// <para>
    /// Reptile meta-training is simpler than MAML:
    /// </para>
    /// <para>
    /// <b>For each task:</b>
    /// 1. Clone the meta-model with current meta-parameters
    /// 2. Perform K gradient descent steps on the task's support set
    /// 3. Compute the direction: (adapted_params - initial_params)
    /// </para>
    /// <para>
    /// <b>Meta-Update:</b>
    /// 1. Average the adaptation directions across all tasks
    /// 2. Move meta-parameters in that direction: theta_new = theta_old + epsilon * direction
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Reptile simply says: "I adapted to these tasks and ended up
    /// at these new parameters. Let me move my starting point a little bit in that
    /// direction, so next time I'm closer to where I need to be."
    /// </para>
    /// <para>
    /// The beauty of Reptile is that this simple interpolation, when done across many
    /// tasks, converges to a good initialization for few-shot learning.
    /// </para>
    /// </remarks>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null || taskBatch.BatchSize == 0)
        {
            throw new ArgumentException("Task batch cannot be null or empty.", nameof(taskBatch));
        }

        // Accumulate parameter updates across all tasks
        Vector<T>? accumulatedUpdates = null;
        T totalLoss = NumOps.Zero;

        foreach (var task in taskBatch.Tasks)
        {
            // Clone the meta model for this task
            var taskModel = CloneModel();
            var initialParams = taskModel.GetParameters();

            // Inner loop: Adapt to the task using support set
            var adaptedParams = InnerLoopAdaptation(taskModel, task);

            // Compute the parameter update direction (adapted - initial)
            var taskUpdate = new Vector<T>(initialParams.Length);
            for (int i = 0; i < initialParams.Length; i++)
            {
                taskUpdate[i] = NumOps.Subtract(adaptedParams[i], initialParams[i]);
            }

            // Accumulate updates
            if (accumulatedUpdates == null)
            {
                accumulatedUpdates = taskUpdate;
            }
            else
            {
                for (int i = 0; i < accumulatedUpdates.Length; i++)
                {
                    accumulatedUpdates[i] = NumOps.Add(accumulatedUpdates[i], taskUpdate[i]);
                }
            }

            // Evaluate on query set for loss tracking
            taskModel.SetParameters(adaptedParams);
            var queryPredictions = taskModel.Predict(task.QueryInput);
            T taskLoss = ComputeLossFromOutput(queryPredictions, task.QueryOutput);
            totalLoss = NumOps.Add(totalLoss, taskLoss);
        }

        if (accumulatedUpdates == null)
        {
            throw new InvalidOperationException("Failed to compute parameter updates.");
        }

        // Average the parameter updates
        T batchSizeT = NumOps.FromDouble(taskBatch.BatchSize);
        for (int i = 0; i < accumulatedUpdates.Length; i++)
        {
            accumulatedUpdates[i] = NumOps.Divide(accumulatedUpdates[i], batchSizeT);
        }

        // Update meta-parameters using interpolation
        // theta_new = theta_old + (OuterLearningRate * Interpolation) * average_update
        var currentMetaParams = MetaModel.GetParameters();
        var updatedMetaParams = new Vector<T>(currentMetaParams.Length);
        T stepSize = NumOps.FromDouble(_reptileOptions.OuterLearningRate * _reptileOptions.Interpolation);

        for (int i = 0; i < currentMetaParams.Length; i++)
        {
            updatedMetaParams[i] = NumOps.Add(
                currentMetaParams[i],
                NumOps.Multiply(stepSize, accumulatedUpdates[i])
            );
        }

        MetaModel.SetParameters(updatedMetaParams);

        // Return average loss
        return NumOps.Divide(totalLoss, batchSizeT);
    }

    /// <summary>
    /// Adapts the meta-learned model to a new task using gradient descent.
    /// </summary>
    /// <param name="task">The new task containing support set examples for adaptation.</param>
    /// <returns>A new model instance that has been fine-tuned to the given task.</returns>
    /// <exception cref="ArgumentNullException">Thrown when task is null.</exception>
    /// <remarks>
    /// <para>
    /// Reptile adaptation is straightforward: perform SGD on the support set
    /// for K steps. The meta-learned initialization enables fast adaptation.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> When adapting to a new task, Reptile works just
    /// like regular training - take gradient steps on the examples. The magic
    /// is in the initialization, which was learned during meta-training to
    /// be a great starting point for any task.
    /// </para>
    /// <para>
    /// Note: Reptile can use many adaptation steps since there's no need to
    /// backpropagate through them. More steps often leads to better task
    /// performance, especially for harder tasks.
    /// </para>
    /// </remarks>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        if (task == null)
        {
            throw new ArgumentNullException(nameof(task));
        }

        // Clone the meta model
        var adaptedModel = CloneModel();

        // Perform inner loop adaptation
        var adaptedParameters = InnerLoopAdaptation(adaptedModel, task);
        adaptedModel.SetParameters(adaptedParameters);

        return adaptedModel;
    }

    /// <summary>
    /// Performs the inner loop adaptation to a specific task.
    /// </summary>
    /// <param name="model">The model to adapt.</param>
    /// <param name="task">The task to adapt to.</param>
    /// <returns>The adapted parameters after multiple gradient steps.</returns>
    /// <remarks>
    /// <para>
    /// Reptile performs (AdaptationSteps * InnerBatches) total gradient steps
    /// during inner loop adaptation. This flexibility allows for deep adaptation
    /// without the memory constraints of MAML.
    /// </para>
    /// </remarks>
    private Vector<T> InnerLoopAdaptation(IFullModel<T, TInput, TOutput> model, IMetaLearningTask<T, TInput, TOutput> task)
    {
        var parameters = model.GetParameters();

        // Reptile can perform many inner steps since no backprop through them
        int totalSteps = _reptileOptions.AdaptationSteps * _reptileOptions.InnerBatches;

        for (int step = 0; step < totalSteps; step++)
        {
            // Compute gradients on support set
            var gradients = ComputeGradients(model, task.SupportInput, task.SupportOutput);

            // Apply gradients with inner learning rate
            parameters = ApplyGradients(parameters, gradients, _reptileOptions.InnerLearningRate);
            model.SetParameters(parameters);
        }

        return parameters;
    }
}
