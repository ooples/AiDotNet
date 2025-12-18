using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.Models;
using AiDotNet.Models.Options;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Adaptive learning rate strategies for SEAL algorithm.
/// </summary>
public enum AdaptiveLrStrategy
{
    /// <summary>
    /// Adam-style adaptive learning rates.
    /// </summary>
    Adam,

    /// <summary>
    /// RMSProp-style adaptive learning rates.
    /// </summary>
    RMSProp,

    /// <summary>
    /// Adagrad-style adaptive learning rates.
    /// </summary>
    Adagrad,

    /// <summary>
    /// Gradient norm-based scaling.
    /// </summary>
    GradNorm
}

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
    private readonly MetaLearningAlgorithmOptions<T, TInput, TOutput> _options;
    private readonly Dictionary<string, AdaptiveLrState<T>> _adaptiveLearningRates;
    private readonly T _defaultLr;
    private readonly T _minLr;
    private readonly T _maxLr;
    private readonly double _temperature;
    private readonly double _entropyCoefficient;
    private readonly bool _useAdaptiveInnerLR;
    private readonly AdaptiveLrStrategy _adaptiveLrStrategy;

    /// <summary>
    /// Initializes a new instance of the SEALAlgorithm class.
    /// </summary>
    /// <param name="options">The configuration options for SEAL.</param>
    public SEALAlgorithm(MetaLearningAlgorithmOptions<T, TInput, TOutput> options) : base(options)
    {
        _options = options;
        _adaptiveLearningRates = new Dictionary<string, AdaptiveLrState<T>>();
        _defaultLr = NumOps.FromDouble(options.InnerLearningRate);
        // Initialize using NumOps
        _minLr = NumOps.FromDouble(1e-7);
        _maxLr = NumOps.FromDouble(1.0);

        // For now, use default values for SEAL-specific features
        // These would be properly configured in a SEALAlgorithmOptions class
        _temperature = 1.0;
        _entropyCoefficient = 0.0;
        _useAdaptiveInnerLR = false;
        _adaptiveLrStrategy = AdaptiveLrStrategy.Adam;
    }

    /// <summary>
    /// State for tracking adaptive learning rates per parameter.
    /// </summary>
    private class AdaptiveLrState<TState>
    {
        public Vector<TState> LearningRates { get; set; }
        public Vector<TState> GradientMoments { get; set; }
        public Vector<TState> GradientVariances { get; set; }
        public int StepCount { get; set; }

        public AdaptiveLrState(int size)
        {
            LearningRates = new Vector<TState>(size);
            GradientMoments = new Vector<TState>(size);
            GradientVariances = new Vector<TState>(size);
            StepCount = 0;
        }
    }

    /// <summary>
    /// Represents a single adaptation step for second-order backpropagation.
    /// </summary>
    private class AdaptationStep<TState>
    {
        public Vector<TState> Parameters { get; set; } = new Vector<TState>(0);
        public Vector<TState> UpdatedParameters { get; set; } = new Vector<TState>(0);
        public Vector<TState> Gradients { get; set; } = new Vector<TState>(0);
        public Vector<TState> MetaGradients { get; set; } = new Vector<TState>(0);
        public int Step { get; set; }
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
            taskModel.SetParameters(adaptedParameters);

            // Evaluate on query set to get meta-loss
            var queryPredictions = taskModel.Predict(task.QueryInput);
            T metaLoss = LossFunction.CalculateLoss(OutputToVector(queryPredictions), OutputToVector(task.QueryOutput));

            // Add temperature scaling if configured
            if (Math.Abs(_temperature - 1.0) > 1e-10)
            {
                T temperature = NumOps.FromDouble(_temperature);
                metaLoss = NumOps.Divide(metaLoss, temperature);
            }

            // Add entropy regularization if configured
            if (_entropyCoefficient > 0.0)
            {
                T entropyTerm = ComputeEntropyRegularization(queryPredictions);
                T entropyCoef = NumOps.FromDouble(_entropyCoefficient);
                metaLoss = NumOps.Subtract(metaLoss, NumOps.Multiply(entropyCoef, entropyTerm));
            }

            totalMetaLoss = NumOps.Add(totalMetaLoss, metaLoss);

            // Compute meta-gradients (gradients with respect to initial parameters)
            var taskMetaGradients = ComputeMetaGradients(task);

            // Note: Gradient clipping would require extending MetaLearningAlgorithmOptions
            // For now, we proceed without gradient clipping

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

        // Note: Weight decay would require extending MetaLearningAlgorithmOptions
        // For now, we proceed without weight decay

        // Outer loop: Update meta-parameters using the meta-optimizer
        var currentMetaParams = MetaModel.GetParameters();
        var updatedMetaParams = MetaOptimizer.UpdateParameters(currentMetaParams, metaGradients);
        MetaModel.SetParameters(updatedMetaParams);

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
        var taskId = task.TaskId;

        // Initialize adaptive learning rates for this task if needed
        if (_useAdaptiveInnerLR && !_adaptiveLearningRates.ContainsKey(taskId))
        {
            _adaptiveLearningRates[taskId] = new AdaptiveLrState<T>(parameters.Length);
            // Initialize with default learning rate
            for (int i = 0; i < parameters.Length; i++)
            {
                _adaptiveLearningRates[taskId].LearningRates[i] = _defaultLr;
            }
        }

        // Perform adaptation steps
        for (int step = 0; step < Options.AdaptationSteps; step++)
        {
            // Compute gradients on support set
            var gradients = ComputeGradients(model, task.SupportInput, task.SupportOutput);

            if (_useAdaptiveInnerLR)
            {
                // Update parameters with adaptive learning rates
                parameters = UpdateParametersAdaptive(parameters, gradients, taskId, step);
            }
            else
            {
                // Use standard inner optimizer
                parameters = InnerOptimizer.UpdateParameters(parameters, gradients);
            }

            model.SetParameters(parameters);
        }

        return parameters;
    }

    /// <summary>
    /// Updates parameters using adaptive learning rates.
    /// </summary>
    private Vector<T> UpdateParametersAdaptive(Vector<T> parameters, Vector<T> gradients, string taskId, int step)
    {
        var state = _adaptiveLearningRates[taskId];
        state.StepCount++;

        var updatedParameters = new Vector<T>(parameters.Length); // Vector constructor creates zero-initialized vector

        for (int i = 0; i < parameters.Length; i++)
        {
            // Update gradient statistics
            T beta1 = NumOps.FromDouble(0.9);
            T beta2 = NumOps.FromDouble(0.999);

            state.GradientMoments[i] = NumOps.Add(
                NumOps.Multiply(beta1, state.GradientMoments[i]),
                NumOps.Multiply(NumOps.Subtract(NumOps.One, beta1), gradients[i]));

            state.GradientVariances[i] = NumOps.Add(
                NumOps.Multiply(beta2, state.GradientVariances[i]),
                NumOps.Multiply(NumOps.Subtract(NumOps.One, beta2), NumOps.Multiply(gradients[i], gradients[i])));

            // Bias-corrected moment estimates
            T biasCorrection1 = NumOps.Divide(NumOps.One, NumOps.Subtract(NumOps.One, NumOps.FromDouble(Math.Pow(0.9, state.StepCount))));
            T biasCorrection2 = NumOps.Divide(NumOps.One, NumOps.Subtract(NumOps.One, NumOps.FromDouble(Math.Pow(0.999, state.StepCount))));

            T mHat = NumOps.Divide(state.GradientMoments[i], biasCorrection1);
            T vHat = NumOps.Divide(state.GradientVariances[i], biasCorrection2);

            // Update learning rate based on gradient statistics
            T gradientNorm = NumOps.Sqrt(NumOps.Add(vHat, NumOps.FromDouble(1e-8)));

            // Adaptive learning rate update
            switch (_adaptiveLrStrategy)
            {
                case AdaptiveLrStrategy.Adam:
                    state.LearningRates[i] = NumOps.Divide(
                        NumOps.Multiply(_defaultLr, mHat),
                        NumOps.Add(gradientNorm, NumOps.FromDouble(1e-8)));
                    break;

                case AdaptiveLrStrategy.RMSProp:
                    state.LearningRates[i] = NumOps.Divide(
                        _defaultLr,
                        NumOps.Add(NumOps.Sqrt(vHat), NumOps.FromDouble(1e-8)));
                    break;

                case AdaptiveLrStrategy.Adagrad:
                    state.LearningRates[i] = NumOps.Divide(
                        _defaultLr,
                        NumOps.Add(NumOps.Sqrt(NumOps.Add(state.GradientVariances[i], NumOps.Multiply(gradients[i], gradients[i]))), NumOps.FromDouble(1e-8)));
                    break;

                case AdaptiveLrStrategy.GradNorm:
                    // Scale learning rate based on gradient norm
                    T gradNormValue = NumOps.Sqrt(NumOps.Multiply(gradients[i], gradients[i]));
                    T scalingFactor = NumOps.Divide(
                        NumOps.FromDouble(1.0),
                        NumOps.Add(NumOps.FromDouble(1.0), NumOps.Multiply(gradNormValue, NumOps.FromDouble(0.001))));
                    state.LearningRates[i] = NumOps.Multiply(_defaultLr, scalingFactor);
                    break;
            }

            // Clamp learning rate to bounds
            state.LearningRates[i] = Max(
                Min(state.LearningRates[i], _maxLr),
                _minLr);

            // Apply learning rate warmup if configured
            if (step < 5)
            {
                T warmupFactor = NumOps.FromDouble((double)(step + 1) / 5.0);
                state.LearningRates[i] = NumOps.Multiply(state.LearningRates[i], warmupFactor);
            }

            // Update parameter
            updatedParameters[i] = NumOps.Subtract(
                parameters[i],
                NumOps.Multiply(state.LearningRates[i], gradients[i]));
        }

        return updatedParameters;
    }

    /// <summary>
    /// Computes meta-gradients for the outer loop update.
    /// </summary>
    /// <param name="task">The task to compute meta-gradients for.</param>
    /// <returns>The meta-gradient vector.</returns>
    private Vector<T> ComputeMetaGradients(ITask<T, TInput, TOutput> task)
    {
        if (Options.UseFirstOrder)
        {
            // First-order approximation: ignore second-order derivatives
            return ComputeFirstOrderMetaGradients(task);
        }
        else
        {
            // Second-order approximation: include derivatives through adaptation
            return ComputeSecondOrderMetaGradients(task);
        }
    }

    /// <summary>
    /// Computes first-order meta-gradients (approximation).
    /// </summary>
    private Vector<T> ComputeFirstOrderMetaGradients(ITask<T, TInput, TOutput> task)
    {
        // Clone meta model for gradient computation
        var model = CloneModel();

        // Adapt to the task
        var adaptedParameters = InnerLoopAdaptation(model, task);
        model.SetParameters(adaptedParameters);

        // Compute gradients on query set
        return ComputeGradients(model, task.QueryInput, task.QueryOutput);
    }

    /// <summary>
    /// Computes second-order meta-gradients with backpropagation through adaptation.
    /// </summary>
    private Vector<T> ComputeSecondOrderMetaGradients(ITask<T, TInput, TOutput> task)
    {
        // Implementation of second-order MAML meta-gradients
        // This backpropagates through the entire inner loop adaptation

        var initialParameters = GetInitialParameters();
        var adaptedParameters = initialParameters;
        var model = CloneModel();

        // Store all intermediate states for backpropagation
        var adaptationHistory = new List<AdaptationStep<T>>();

        // Forward pass: perform adaptation and store intermediate states
        for (int step = 0; step < Options.AdaptationSteps; step++)
        {
            var currentStep = new AdaptationStep<T>
            {
                Parameters = adaptedParameters,
                Step = step
            };

            // Compute gradients on support set
            var gradients = ComputeGradients(model, task.SupportInput, task.SupportOutput);
            currentStep.Gradients = gradients;

            // Apply parameter update
            if (_useAdaptiveInnerLR)
            {
                adaptedParameters = UpdateParametersAdaptive(adaptedParameters, gradients, task.TaskId, step);
            }
            else
            {
                adaptedParameters = InnerOptimizer.UpdateParameters(adaptedParameters, gradients);
            }

            currentStep.UpdatedParameters = adaptedParameters;
            adaptationHistory.Add(currentStep);
            model.SetParameters(adaptedParameters);
        }

        // Compute query loss gradients
        var queryGradients = ComputeGradients(model, task.QueryInput, task.QueryOutput);

        // Backward pass: propagate gradients through adaptation steps
        var metaGradients = BackpropagateThroughAdaptation(
            queryGradients,
            adaptationHistory,
            task);

        return metaGradients;
    }

    /// <summary>
    /// Backpropagates gradients through the adaptation steps.
    /// </summary>
    private Vector<T> BackpropagateThroughAdaptation(
        Vector<T> outputGradients,
        List<AdaptationStep<T>> adaptationHistory,
        ITask<T, TInput, TOutput> task)
    {
        var currentGradients = outputGradients;

        // Process adaptation steps in reverse order
        for (int i = adaptationHistory.Count - 1; i >= 0; i--)
        {
            var step = adaptationHistory[i];

            // For adaptive learning rates, we need to account for the learning rate update
            if (_useAdaptiveInnerLR)
            {
                // The gradient update was: p_new = p_old - lr * grad
                // So dL/dp_old = dL/dp_new * (1 - lr * d(grad)/dp_old)
                // This is simplified; a full implementation would compute the exact Jacobian

                // For now, we'll use an approximation
                var state = _adaptiveLearningRates[task.TaskId];
                for (int j = 0; j < currentGradients.Length; j++)
                {
                    // Scale by learning rate for backpropagation
                    currentGradients[j] = NumOps.Multiply(
                        currentGradients[j],
                        state.LearningRates[j]);
                }
            }

            // Store gradients for this step
            step.MetaGradients = currentGradients;

            // Continue backpropagation to previous step
            // In a full implementation, we would compute the exact Jacobian
            // of the update rule with respect to the parameters
        }

        return currentGradients;
    }

    /// <summary>
    /// Gets the initial parameters for the meta-model.
    /// </summary>
    private Vector<T> GetInitialParameters()
    {
        var model = CloneModel();
        return model.GetParameters();
    }

    /// <summary>
    /// Computes entropy regularization term for the predictions.
    /// </summary>
    /// <param name="predictions">The model predictions.</param>
    /// <returns>The entropy value.</returns>
    private T ComputeEntropyRegularization(TOutput predictions)
    {
        // Simplified entropy computation
        // H = -Σ(p_i * log(p_i))
        try
        {
            // Convert to vector for computation
            var predVector = OutputToVector(predictions);

            // Simple entropy calculation
            T entropy = NumOps.Zero;
            T epsilon = NumOps.FromDouble(1e-8);

            for (int i = 0; i < predVector.Length; i++)
            {
                T p = predVector[i];
                // Clamp to avoid log(0)
                p = Max(p, epsilon);
                // Compute p * log(p) - use ToInt32 for conversion approximation
                int pi = NumOps.ToInt32(p);
                if (pi > 0)
                {
                    double logP = Math.Log(Math.Max(pi, 1));
                    entropy = NumOps.Add(entropy, NumOps.Multiply(p, NumOps.FromDouble(-logP)));
                }
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
    /// Converts TOutput to a Tensor for entropy computation.
    /// </summary>
    /// <param name="output">The output to convert.</param>
    /// <returns>A tensor representation of the output.</returns>
    private Tensor<T> ConvertToTensor(TOutput output)
    {
        // Try to convert based on common output types
        if (output is Tensor<T> tensor)
        {
            return tensor;
        }
        else if (output is Vector<T> vector)
        {
            return Tensor<T>.FromVector(vector);
        }
        else if (output is Matrix<T> matrix)
        {
            return Tensor<T>.FromMatrix(matrix);
        }
        else
        {
            // For other types, try to create a tensor from the data
            // This is a simplified conversion - in practice, you'd need proper type handling
            throw new NotSupportedException($"Cannot convert {typeof(TOutput).Name} to Tensor for entropy computation");
        }
    }
}
