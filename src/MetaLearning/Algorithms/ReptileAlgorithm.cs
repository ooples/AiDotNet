using AiDotNet.Interfaces;
using AiDotNet.MetaLearning.Data;
using AiDotNet.Models;
using AiDotNet.Models.Options;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of the Reptile meta-learning algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix<T>, Tensor<T>).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector<T>, Tensor<T>).</typeparam>
/// <remarks>
/// <para>
/// Reptile is a simple and scalable meta-learning algorithm. Unlike MAML, it doesn't require
/// computing gradients through the adaptation process, making it more efficient and easier
/// to implement while achieving competitive performance.
/// </para>
/// <para>
/// Algorithm:
/// 1. Sample a task
/// 2. Perform SGD on the task starting from the current meta-parameters
/// 3. Update meta-parameters by interpolating toward the adapted parameters
/// 4. Repeat
/// </para>
/// <para>
/// <b>For Beginners:</b> Reptile is like learning by averaging your experiences.
///
/// Imagine learning to cook:
/// - You start with basic knowledge (initial parameters)
/// - You make a specific dish and learn specific techniques
/// - Instead of just remembering that one dish, you update your basic knowledge
///   to include some of what you learned
/// - After cooking many dishes, your basic knowledge becomes really good
///   for learning any new recipe quickly
///
/// Reptile is simpler than MAML because it just moves toward adapted parameters
/// instead of computing complex gradients through the adaptation process.
/// </para>
/// <para>
/// Reference: Nichol, A., Achiam, J., & Schulman, J. (2018).
/// On first-order meta-learning algorithms.
/// </para>
/// </remarks>
public class ReptileAlgorithm<T, TInput, TOutput> : MetaLearningBase<T, TInput, TOutput>
{
    private readonly ReptileAlgorithmOptions<T, TInput, TOutput> _reptileOptions;

    /// <summary>
    /// Initializes a new instance of the ReptileAlgorithm class.
    /// </summary>
    /// <param name="options">The configuration options for Reptile.</param>
    public ReptileAlgorithm(ReptileAlgorithmOptions<T, TInput, TOutput> options) : base(options)
    {
        _reptileOptions = options;
    }

    /// <inheritdoc/>
    public override string AlgorithmName => "Reptile";

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null || taskBatch.BatchSize == 0)
        {
            throw new ArgumentException("Task batch cannot be null or empty.", nameof(taskBatch));
        }

        // Accumulate parameter updates across all tasks
        Vector<T>? parameterUpdates = null;
        T totalLoss = NumOps.Zero;

        foreach (var task in taskBatch.Tasks)
        {
            // Clone the meta model for this task
            var taskModel = CloneModel();
            var initialParams = taskModel.GetParameters();

            // Inner loop: Adapt to the task using support set
            var adaptedParams = InnerLoopAdaptation(taskModel, task);

            // Compute the parameter update (adapted params - initial params)
            var taskUpdate = new Vector<T>(initialParams.Length);
            for (int i = 0; i < initialParams.Length; i++)
            {
                taskUpdate[i] = NumOps.Subtract(adaptedParams[i], initialParams[i]);
            }

            // Accumulate updates
            if (parameterUpdates == null)
            {
                parameterUpdates = taskUpdate;
            }
            else
            {
                for (int i = 0; i < parameterUpdates.Length; i++)
                {
                    parameterUpdates[i] = NumOps.Add(parameterUpdates[i], taskUpdate[i]);
                }
            }

            // Evaluate on query set for logging
            taskModel.SetParameters(adaptedParams);
            var queryPredictions = taskModel.Predict(task.QueryInput);
            T taskLoss = LossFunction.CalculateLoss(OutputToVector(queryPredictions), OutputToVector(task.QueryOutput));
            totalLoss = NumOps.Add(totalLoss, taskLoss);
        }

        if (parameterUpdates == null)
        {
            throw new InvalidOperationException("Failed to compute parameter updates.");
        }

        // Average the parameter updates
        T batchSize = NumOps.FromDouble(taskBatch.BatchSize);
        for (int i = 0; i < parameterUpdates.Length; i++)
        {
            parameterUpdates[i] = NumOps.Divide(parameterUpdates[i], batchSize);
        }

        // Update meta-parameters using interpolation
        var currentMetaParams = MetaModel.GetParameters();
        var updatedMetaParams = new Vector<T>(currentMetaParams.Length);
        T interpolation = NumOps.FromDouble(_reptileOptions.Interpolation * Options.OuterLearningRate);

        for (int i = 0; i < currentMetaParams.Length; i++)
        {
            // θ_new = θ_old + interpolation * (θ_adapted - θ_old)
            updatedMetaParams[i] = NumOps.Add(
                currentMetaParams[i],
                NumOps.Multiply(interpolation, parameterUpdates[i])
            );
        }

        MetaModel.SetParameters(updatedMetaParams);

        // Return average loss
        return NumOps.Divide(totalLoss, batchSize);
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(ITask<T, TInput, TOutput> task)
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
    /// <returns>The adapted parameters.</returns>
    private Vector<T> InnerLoopAdaptation(IFullModel<T, TInput, TOutput> model, ITask<T, TInput, TOutput> task)
    {
        var parameters = model.GetParameters();

        // Reptile performs multiple inner batches per task
        int totalSteps = Options.AdaptationSteps * _reptileOptions.InnerBatches;

        for (int step = 0; step < totalSteps; step++)
        {
            // Compute gradients on support set
            var gradients = ComputeGradients(model, task.SupportInput, task.SupportOutput);

            // Use inner optimizer for parameter updates
            parameters = InnerOptimizer.UpdateParameters(parameters, gradients);
            model.SetParameters(parameters);
        }

        return parameters;
    }
}
