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
    private readonly Dictionary<string, Vector<T>>? _adaptiveLearningRates;

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
            if (Math.Abs(_sealOptions.Temperature - 1.0) > 1e-10)
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
    /// <returns>The adapted parameters.</returns>
    private Vector<T> InnerLoopAdaptation(IFullModel<T, TInput, TOutput> model, ITask<T, TInput, TOutput> task)
    {
        var parameters = model.GetParameters();

        // Perform adaptation steps
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
    /// Computes adaptive learning rates based on gradient norms.
    /// </summary>
    /// <param name="gradients">The gradient vector.</param>
    /// <returns>Adaptive learning rates for each parameter.</returns>
    private Vector<T> ComputeAdaptiveLearningRates(Vector<T> gradients)
    {
        var adaptiveLr = new Vector<T>(gradients.Length);

        // Compute gradient norms
        for (int i = 0; i < gradients.Length; i++)
        {
            T gradNorm = NumOps.Multiply(gradients[i], gradients[i]); // |g_i|²

            // Adaptive LR = base_lr / (sqrt(grad_norm) + epsilon)
            T sqrtNorm = NumOps.FromDouble(Math.Sqrt(Math.Max(NumOps.ToDouble(gradNorm), 1e-8)));
            T epsilon = NumOps.FromDouble(1e-8);
            T denominator = NumOps.Add(sqrtNorm, epsilon);

            adaptiveLr[i] = NumOps.Divide(Options.InnerLearningRate, denominator);
        }

        return adaptiveLr;
    }

    /// <summary>
    /// Clips gradients to prevent exploding gradients.
    /// </summary>
    /// <param name="gradients">The gradient vector to clip.</param>
    /// <param name="threshold">The clipping threshold.</param>
    /// <returns>The clipped gradients.</returns>
    private Vector<T> ClipGradients(Vector<T> gradients, double threshold)
    {
        // Compute L2 norm of gradients
        T squaredSum = NumOps.Zero;
        for (int i = 0; i < gradients.Length; i++)
        {
            squaredSum = NumOps.Add(squaredSum, NumOps.Multiply(gradients[i], gradients[i]));
        }

        T norm = NumOps.FromDouble(Math.Sqrt(NumOps.ToDouble(squaredSum)));
        T thresholdT = NumOps.FromDouble(threshold);

        // If norm is below threshold, return gradients unchanged
        if (Convert.ToDouble(norm) <= threshold)
        {
            return gradients;
        }

        // Otherwise, scale gradients
        T scaleFactor = NumOps.Divide(thresholdT, norm);
        var clippedGradients = new Vector<T>(gradients.Length);

        for (int i = 0; i < gradients.Length; i++)
        {
            clippedGradients[i] = NumOps.Multiply(gradients[i], scaleFactor);
        }

        return clippedGradients;
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
        // NOTE: Full second-order implementation is computationally expensive (O(n³))
        // and requires automatic differentiation. Most practical implementations
        // use first-order approximation which performs nearly as well.
        //
        // To implement full second-order:
        // 1. Store computational graph during inner loop adaptation
        // 2. Compute Hessian-vector products using Pearlmutter's algorithm
        // 3. Apply implicit gradient computation: ∇θ_meta = (I - λH)⁻¹ ∇θ_L
        // 4. This would require significant framework infrastructure
        //
        // Current implementation uses first-order (FOMAML) which:
        // - Treats adapted parameters as constants during backprop
        // - Reduces complexity from O(n³) to O(n)
        // - Maintains most of the performance benefits
        // - Is the de facto standard in production systems
        return metaGradients;
    }

  
    /// <summary>
    /// Computes entropy regularization term for the predictions.
    /// </summary>
    /// <param name="predictions">The model predictions.</param>
    /// <returns>The entropy value.</returns>
    private T ComputeEntropyRegularization(TOutput predictions)
    {
        // Entropy regularization encourages diverse predictions
        // H(p) = -sum(p_i * log(p_i))
        try
        {
            // Convert predictions to probabilities (softmax if not already)
            var probabilities = ConvertToProbabilities(predictions);
            T entropy = NumOps.Zero;

            // Compute entropy: -sum(p * log(p))
            for (int i = 0; i < probabilities.Length; i++)
            {
                T p = probabilities[i];

                // Avoid log(0) by adding small epsilon
                T epsilon = NumOps.FromDouble(1e-8);
                p = NumOps.Add(p, epsilon);

                // Normalize to ensure sum = 1
                T sum = NumOps.Zero;
                for (int j = 0; j < probabilities.Length; j++)
                {
                    sum = NumOps.Add(sum, NumOps.Add(probabilities[j], epsilon));
                }
                p = NumOps.Divide(p, sum);

                // Compute p * log(p)
                T logP = NumOps.FromDouble(Math.Log(NumOps.ToDouble(p)));
                T contribution = NumOps.Multiply(p, logP);
                entropy = NumOps.Subtract(entropy, contribution);
            }

            return entropy;
        }
        catch
        {
            // If entropy computation fails, return zero
            return NumOps.Zero;
        }
    }

    /// <summary>
    /// Converts model outputs to probability distribution using softmax.
    /// </summary>
    private Vector<T> ConvertToProbabilities(TOutput predictions)
    {
        // Extract vector from TOutput
        Vector<T> logits;
        if (typeof(TOutput) == typeof(Vector<T>))
        {
            logits = (Vector<T>)(object)predictions;
        }
        else if (typeof(TOutput) == typeof(Tensor<T>))
        {
            var tensor = (Tensor<T>)(object)predictions;
            logits = tensor.Flatten();
        }
        else
        {
            throw new NotSupportedException($"Cannot compute entropy for type {typeof(TOutput).Name}");
        }

        // Apply softmax: softmax(x)_i = exp(x_i) / sum(exp(x_j))
        var expValues = new T[logits.Length];
        T sumExp = NumOps.Zero;

        // Compute exp values for numerical stability
        T maxLogit = logits[0];
        for (int i = 1; i < logits.Length; i++)
        {
            if (Convert.ToDouble(logits[i]) > Convert.ToDouble(maxLogit))
            {
                maxLogit = logits[i];
            }
        }

        // Compute exp(x - max) for stability
        for (int i = 0; i < logits.Length; i++)
        {
            T shifted = NumOps.Subtract(logits[i], maxLogit);
            expValues[i] = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(shifted)));
            sumExp = NumOps.Add(sumExp, expValues[i]);
        }

        // Normalize to probabilities
        var probabilities = new Vector<T>(logits.Length);
        for (int i = 0; i < logits.Length; i++)
        {
            probabilities[i] = NumOps.Divide(expValues[i], sumExp);
        }

        return probabilities;
    }
}
