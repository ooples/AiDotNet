using AiDotNet.Interfaces;
using AiDotNet.MetaLearning.Data;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Data.Structures;

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
            var adaptedParams = InnerLoopAdaptation(taskModel, task);
            taskModel.SetParameters(adaptedParams);

            // Compute meta-loss on query set
            var queryPredictions = taskModel.Predict(task.QueryInput);
            T metaLoss = LossFunction.CalculateLoss(queryPredictions, task.QueryOutput);
            totalMetaLoss = NumOps.Add(totalMetaLoss, metaLoss);

            // Compute meta-gradients (gradients with respect to initial parameters)
            var taskMetaGradients = ComputeMetaGradients(initialParams, task);

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
        MetaModel.SetParameters(updatedMetaParams);

        // Return average meta-loss
        return NumOps.Divide(totalMetaLoss, batchSize);
    }

    /// <inheritdoc/>
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
    /// <returns>The adapted parameters.</returns>
    private Vector<T> InnerLoopAdaptation(IFullModel<T, TInput, TOutput> model, IMetaLearningTask<T, TInput, TOutput> task)
    {
        var parameters = model.GetParameters();

        // Perform K gradient steps on the support set
        for (int step = 0; step < Options.AdaptationSteps; step++)
        {
            // Compute gradients on support set
            var gradients = ComputeGradients(model, task.SupportInput, task.SupportOutput);

            // Use inner optimizer for parameter updates
            parameters = InnerOptimizer.UpdateParameters(parameters, gradients);
            model.SetParameters(parameters);
        }

        return parameters;
    }

    /// <summary>
    /// Computes meta-gradients for the outer loop update.
    /// </summary>
    /// <param name="initialParams">The initial parameters before adaptation.</param>
    /// <param name="task">The task to compute meta-gradients for.</param>
    /// <returns>The meta-gradient vector.</returns>
    private Vector<T> ComputeMetaGradients(Vector<T> initialParams, IMetaLearningTask<T, TInput, TOutput> task)
    {
        // Clone meta model
        var model = CloneModel();
        model.SetParameters(initialParams);

        // Adapt to the task
        var adaptedParams = InnerLoopAdaptation(model, task);
        model.SetParameters(adaptedParams);

        // CRITICAL: This implements first-order MAML (FOMAML)
        // The gradient is computed w.r.t. adapted parameters, treating them as constants
        // This is NOT true second-order MAML which would require:
        // 1. Storing the computational graph during inner loop
        // 2. Computing dL/dθ_i = dL/dθ'_i * dθ'_i/dθ_i (chain rule through adaptation)
        // 3. Where θ'_i are adapted parameters and θ_i are initial parameters
        //
        // True second-order would need:
        // - Automatic differentiation or manual gradient computation
        // - Hessian-vector products for efficiency
        // - O(n²) or O(n³) computational cost
        //
        // Current FOMAML approach:
        // - Treats adaptation as a black box
        // - Computes ∇L(θ') where θ' are adapted parameters
        // - Ignores the dependency ∂θ'/∂θ
        // - Reduces from O(n³) to O(n) complexity
        // - Performs nearly as well in practice (Finn et al., 2017)
        var metaGradients = ComputeGradients(model, task.QueryInput, task.QueryOutput);

        return metaGradients;
    }
}
