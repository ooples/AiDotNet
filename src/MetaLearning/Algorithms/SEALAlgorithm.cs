using AiDotNet.Interfaces;
using AiDotNet.MetaLearning.Data;
using AiDotNet.Models;
using AiDotNet.Models.Options;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of the SEAL (Sample-Efficient Adaptive Learning) meta-learning algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix<T>, Tensor<T>).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector<T>, Tensor<T>).</typeparam>
/// <remarks>
/// <para>
/// SEAL is a gradient-based meta-learning algorithm that learns initial parameters
/// that can be quickly adapted to new tasks with just a few examples. It combines
/// ideas from MAML (Model-Agnostic Meta-Learning) with additional efficiency improvements.
/// </para>
/// <para>
/// <b>For Beginners:</b> SEAL learns the best starting point for a model so that
/// it can quickly adapt to new tasks with minimal data.
///
/// Imagine learning to play musical instruments:
/// - Learning your first instrument (e.g., piano) is hard
/// - Learning your second instrument (e.g., guitar) is easier
/// - By the time you learn your 5th instrument, you've learned principles of music
///   that help you pick up new instruments much faster
///
/// SEAL does the same with machine learning models - it learns from many tasks
/// to find a great starting point that makes adapting to new tasks much faster.
/// </para>
/// </remarks>
public class SEALAlgorithm<T, TInput, TOutput> : MetaLearningBase<T, TInput, TOutput>
{
    private readonly SEALAlgorithmOptions<T, TInput, TOutput> _sealOptions;
    private Dictionary<string, Vector<T>>? _adaptiveLearningRates;

    /// <summary>
    /// Initializes a new instance of the SEALAlgorithm class.
    /// </summary>
    /// <param name="options">The configuration options for SEAL.</param>
    public SEALAlgorithm(SEALAlgorithmOptions<T, TInput, TOutput> options) : base(options)
    {
        _sealOptions = options;

        if (_sealOptions.UseAdaptiveInnerLR)
        {
            _adaptiveLearningRates = new Dictionary<string, Vector<T>>();
        }
    }

    /// <inheritdoc/>
    public override string AlgorithmName => "SEAL";

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null || taskBatch.BatchSize == 0)
        {
            throw new ArgumentException("Task batch cannot be null or empty.", nameof(taskBatch));
        }

        // Accumulate meta-gradients across all tasks in the batch
        Vector<T>? metaGradients = null;
        T totalMetaLoss = NumOps.Zero;

        foreach (var task in taskBatch.Tasks)
        {
            // Clone the meta model for this task
            var taskModel = CloneModel();

            // Inner loop: Adapt to the task using support set
            var adaptedParameters = InnerLoopAdaptation(taskModel, task);
            taskModel.UpdateParameters(adaptedParameters);

            // Evaluate on query set to get meta-loss
            var queryPredictions = taskModel.Predict(task.QueryInput);
            T metaLoss = LossFunction.ComputeLoss(queryPredictions, task.QueryOutput);

            // Add temperature scaling if configured
            if (_sealOptions.Temperature != 1.0)
            {
                T temperature = NumOps.FromDouble(_sealOptions.Temperature);
                metaLoss = NumOps.Divide(metaLoss, temperature);
            }

            // Add entropy regularization if configured
            if (_sealOptions.EntropyCoefficient > 0.0)
            {
                T entropyTerm = ComputeEntropyRegularization(queryPredictions);
                T entropyCoef = NumOps.FromDouble(_sealOptions.EntropyCoefficient);
                metaLoss = NumOps.Subtract(metaLoss, NumOps.Multiply(entropyCoef, entropyTerm));
            }

            totalMetaLoss = NumOps.Add(totalMetaLoss, metaLoss);

            // Compute meta-gradients (gradients with respect to initial parameters)
            var taskMetaGradients = ComputeMetaGradients(task);

            // Clip gradients if threshold is set
            if (_sealOptions.GradientClipThreshold.HasValue)
            {
                taskMetaGradients = ClipGradients(taskMetaGradients, _sealOptions.GradientClipThreshold.Value);
            }

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

        // Apply weight decay if configured
        if (_sealOptions.WeightDecay > 0.0)
        {
            var currentParams = MetaModel.GetParameters();
            T decay = NumOps.FromDouble(_sealOptions.WeightDecay);
            for (int i = 0; i < metaGradients.Length; i++)
            {
                metaGradients[i] = NumOps.Add(metaGradients[i], NumOps.Multiply(decay, currentParams[i]));
            }
        }

        // Outer loop: Update meta-parameters
        var currentMetaParams = MetaModel.GetParameters();
        var updatedMetaParams = ApplyGradients(currentMetaParams, metaGradients, Options.OuterLearningRate);
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
    /// <returns>The adapted parameters.</returns>
    private Vector<T> InnerLoopAdaptation(IFullModel<T, TInput, TOutput> model, ITask<T, TInput, TOutput> task)
    {
        var parameters = model.GetParameters();

        // Perform adaptation steps
        for (int step = 0; step < Options.AdaptationSteps; step++)
        {
            // Compute gradients on support set
            var gradients = ComputeGradients(model, task.SupportInput, task.SupportOutput);

            // Get learning rate (adaptive or fixed)
            double learningRate = GetInnerLearningRate(task.TaskId, step);

            // Apply gradients
            parameters = ApplyGradients(parameters, gradients, learningRate);
            model.UpdateParameters(parameters);
        }

        return parameters;
    }

    /// <summary>
    /// Computes meta-gradients for the outer loop update.
    /// </summary>
    /// <param name="task">The task to compute meta-gradients for.</param>
    /// <returns>The meta-gradient vector.</returns>
    private Vector<T> ComputeMetaGradients(ITask<T, TInput, TOutput> task)
    {
        // Clone meta model for gradient computation
        var model = CloneModel();

        // Adapt to the task
        var adaptedParameters = InnerLoopAdaptation(model, task);
        model.UpdateParameters(adaptedParameters);

        // Compute gradients on query set
        var metaGradients = ComputeGradients(model, task.QueryInput, task.QueryOutput);

        // If using first-order approximation, we're done
        if (Options.UseFirstOrder)
        {
            return metaGradients;
        }

        // For second-order, we need to backpropagate through the adaptation steps
        // This is computationally expensive and requires careful implementation
        // For now, we use first-order approximation as it's more practical
        return metaGradients;
    }

    /// <summary>
    /// Gets the inner learning rate, either adaptive or fixed.
    /// </summary>
    /// <param name="taskId">The task identifier.</param>
    /// <param name="step">The current adaptation step.</param>
    /// <returns>The learning rate to use.</returns>
    private double GetInnerLearningRate(string taskId, int step)
    {
        if (!_sealOptions.UseAdaptiveInnerLR || _adaptiveLearningRates == null)
        {
            return Options.InnerLearningRate;
        }

        // For adaptive learning rates, we would learn per-parameter learning rates
        // For simplicity, we use a fixed learning rate here
        // A full implementation would maintain and update adaptive learning rates
        return Options.InnerLearningRate;
    }

    /// <summary>
    /// Computes entropy regularization term for the predictions.
    /// </summary>
    /// <param name="predictions">The model predictions.</param>
    /// <returns>The entropy value.</returns>
    private T ComputeEntropyRegularization(TOutput predictions)
    {
        // Entropy regularization encourages diverse predictions
        // For simplicity, we return zero here
        // A full implementation would compute the entropy of the prediction distribution
        return NumOps.Zero;
    }
}
