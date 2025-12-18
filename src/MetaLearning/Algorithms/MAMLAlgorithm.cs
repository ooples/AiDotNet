using AiDotNet.Interfaces;
using AiDotNet.MetaLearning.Data;
using AiDotNet.Models;
using AiDotNet.Models.Options;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of the MAML (Model-Agnostic Meta-Learning) algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix<T>, Tensor<T>).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector<T>, Tensor<T>).</typeparam>
/// <remarks>
/// <para>
/// MAML (Model-Agnostic Meta-Learning) is a meta-learning algorithm that trains models
/// to be easily fine-tunable. It learns initial parameters such that a small number of
/// gradient steps on a new task will lead to good performance.
/// </para>
/// <para>
/// Key features:
/// - Model-agnostic: works with any model trainable with gradient descent
/// - Learns good initialization rather than learning a fixed feature extractor
/// - Enables few-shot learning with just 1-5 examples per class
/// </para>
/// <para>
/// <b>For Beginners:</b> MAML is like teaching someone how to learn quickly.
///
/// Normal machine learning: Train a model for one specific task
/// MAML: Train a model to be easily trainable for many different tasks
///
/// It's like learning how to learn - by practicing on many tasks, the model
/// learns what kind of parameters make it easy to adapt to new tasks quickly.
/// </para>
/// <para>
/// Reference: Finn, C., Abbeel, P., & Levine, S. (2017).
/// Model-agnostic meta-learning for fast adaptation of deep networks.
/// </para>
/// </remarks>
public class MAMLAlgorithm<T, TInput, TOutput> : MetaLearningBase<T, TInput, TOutput>
{
    private readonly MAMLAlgorithmOptions<T, TInput, TOutput> _mamlOptions;

    /// <summary>
    /// Initializes a new instance of the MAMLAlgorithm class.
    /// </summary>
    /// <param name="options">The configuration options for MAML.</param>
    public MAMLAlgorithm(MAMLAlgorithmOptions<T, TInput, TOutput> options) : base(options)
    {
        _mamlOptions = options;
    }

    /// <inheritdoc/>
    public override string AlgorithmName => "MAML";

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null || taskBatch.BatchSize == 0)
        {
            throw new ArgumentException("Task batch cannot be null or empty.", nameof(taskBatch));
        }

        // Accumulate meta-gradients across all tasks
        Vector<T>? metaGradients = null;
        T totalMetaLoss = NumOps.Zero;

        foreach (var task in taskBatch.Tasks)
        {
            // Clone the meta model for this task
            var taskModel = CloneModel();
            var initialParams = taskModel.GetParameters();

            // Inner loop: Adapt to the task using support set
            var (adaptedParams, adaptationHistory) = InnerLoopAdaptationWithHistory(taskModel, task, initialParams);
            taskModel.UpdateParameters(adaptedParams);

            // Compute meta-loss on query set
            var queryPredictions = taskModel.Predict(task.QueryInput);
            T metaLoss = LossFunction.ComputeLoss(queryPredictions, task.QueryOutput);
            totalMetaLoss = NumOps.Add(totalMetaLoss, metaLoss);

            // Compute meta-gradients (gradients with respect to initial parameters)
            // Pass the already adapted parameters to avoid re-adaptation
            var taskMetaGradients = ComputeMetaGradients(initialParams, task, adaptedParams, adaptationHistory);

            // Accumulate meta-gradients
            if (metaGradients == null)
            {
                metaGradients = taskMetaGradients;
            }
            else
            {
                for (int i = 0; i < metaGradients.Length; i++)
                {
                    metaGradients[i] = NumOps.Add(metaGradients[i], taskMetaGradients[i]);
                }
            }
        }

        if (metaGradients == null)
        {
            throw new InvalidOperationException("Failed to compute meta-gradients.");
        }

        // Average the meta-gradients
        T batchSize = NumOps.FromDouble(taskBatch.BatchSize);
        for (int i = 0; i < metaGradients.Length; i++)
        {
            metaGradients[i] = NumOps.Divide(metaGradients[i], batchSize);
        }

        // Outer loop: Update meta-parameters using the meta-optimizer
        var currentMetaParams = MetaModel.GetParameters();
        var updatedMetaParams = MetaOptimizer.UpdateParameters(currentMetaParams, metaGradients);
        MetaModel.UpdateParameters(updatedMetaParams);

        // Return average meta-loss
        return NumOps.Divide(totalMetaLoss, batchSize);
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
        adaptedModel.UpdateParameters(adaptedParameters);

        return adaptedModel;
    }

    /// <summary>
    /// Performs the inner loop adaptation to a specific task.
    /// </summary>
    /// <param name="model">The model to adapt.</param>
    /// <param name="task">The task to adapt to.</param>
    /// <param name="initialParams">Initial parameters for adaptation.</param>
    /// <returns>The adapted parameters and adaptation history.</returns>
    private (Vector<T> adaptedParams, List<AdaptationStep<T>> adaptationHistory) InnerLoopAdaptationWithHistory(
        IFullModel<T, TInput, TOutput> model,
        ITask<T, TInput, TOutput> task,
        Vector<T> initialParams)
    {
        var parameters = initialParams;
        var history = new List<AdaptationStep<T>>();

        // Perform K gradient steps on the support set
        for (int step = 0; step < Options.AdaptationSteps; step++)
        {
            var stepInfo = new AdaptationStep<T>
            {
                Parameters = Vector<T>.Copy(parameters),
                Step = step
            };

            // Compute gradients on support set
            var gradients = ComputeGradients(model, task.SupportInput, task.SupportOutput);
            stepInfo.Gradients = gradients;

            // Use inner optimizer for parameter updates
            parameters = InnerOptimizer.UpdateParameters(parameters, gradients);
            stepInfo.UpdatedParameters = Vector<T>.Copy(parameters);

            model.UpdateParameters(parameters);
            history.Add(stepInfo);
        }

        return (parameters, history);
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

        // Perform K gradient steps on the support set
        for (int step = 0; step < Options.AdaptationSteps; step++)
        {
            // Compute gradients on support set
            var gradients = ComputeGradients(model, task.SupportInput, task.SupportOutput);

            // Use inner optimizer for parameter updates
            parameters = InnerOptimizer.UpdateParameters(parameters, gradients);
            model.UpdateParameters(parameters);
        }

        return parameters;
    }

    /// <summary>
    /// Represents a single adaptation step for tracking gradients.
    /// </summary>
    private class AdaptationStep<TState>
    {
        public Vector<TState> Parameters { get; set; }
        public Vector<TState> UpdatedParameters { get; set; }
        public Vector<TState> Gradients { get; set; }
        public int Step { get; set; }
    }

    /// <summary>
    /// Computes meta-gradients for the outer loop update.
    /// </summary>
    /// <param name="initialParams">The initial parameters before adaptation.</param>
    /// <param name="task">The task to compute meta-gradients for.</param>
    /// <param name="adaptedParams">Already adapted parameters.</param>
    /// <param name="adaptationHistory">History of adaptation steps.</param>
    /// <returns>The meta-gradient vector.</returns>
    private Vector<T> ComputeMetaGradients(
        Vector<T> initialParams,
        ITask<T, TInput, TOutput> task,
        Vector<T> adaptedParams,
        List<AdaptationStep<T>> adaptationHistory)
    {
        // Clone meta model with adapted parameters (no re-adaptation needed)
        var model = CloneModel();
        model.UpdateParameters(adaptedParams);

        // Compute gradients on query set (this gives us the meta-gradient)
        var metaGradients = ComputeGradients(model, task.QueryInput, task.QueryOutput);

        return metaGradients;
    }
}
