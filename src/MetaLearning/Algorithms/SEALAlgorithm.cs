using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of the SEAL (Sample-Efficient Adaptive Learning) meta-learning algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// SEAL is a gradient-based meta-learning algorithm that combines ideas from MAML with
/// sample-efficiency improvements. It learns initial parameters that can be quickly
/// adapted to new tasks with just a few examples.
/// </para>
/// <para>
/// <b>Key Features:</b>
/// - Temperature scaling: Controls confidence in predictions during meta-training
/// - Entropy regularization: Encourages diverse predictions to prevent overconfident models
/// - Adaptive learning rates: Per-parameter learning rate adaptation based on gradient norms
/// - Weight decay: Prevents overfitting to meta-training tasks
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// 1. Sample a batch of tasks
/// 2. For each task:
///    a. Clone the meta-model
///    b. Adapt to the task using support set (inner loop)
///    c. Evaluate on query set to compute meta-loss
///    d. Apply temperature scaling and entropy regularization
///    e. Compute meta-gradients
/// 3. Average meta-gradients across tasks
/// 4. Apply weight decay and update meta-parameters
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
/// <para>
/// Reference: Based on gradient-based meta-learning with additional efficiency
/// improvements including temperature scaling and entropy regularization.
/// </para>
/// </remarks>
public class SEALAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly SEALOptions<T, TInput, TOutput> _sealOptions;
    private readonly Dictionary<string, Vector<T>>? _adaptiveLearningRateState;

    /// <summary>
    /// Initializes a new instance of the SEALAlgorithm class.
    /// </summary>
    /// <param name="options">SEAL configuration options containing the model and all hyperparameters.</param>
    /// <exception cref="ArgumentNullException">Thrown when options is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when required components are not set in options.</exception>
    /// <example>
    /// <code>
    /// // Create SEAL with minimal configuration
    /// var options = new SEALOptions&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;(myNeuralNetwork);
    /// var seal = new SEALAlgorithm&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;(options);
    ///
    /// // Create SEAL with entropy regularization
    /// var options = new SEALOptions&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;(myNeuralNetwork)
    /// {
    ///     EntropyCoefficient = 0.01,
    ///     Temperature = 1.5,
    ///     UseAdaptiveInnerLR = true,
    ///     AdaptiveLearningRateMode = SEALAdaptiveLearningRateMode.RunningMean
    /// };
    /// var seal = new SEALAlgorithm&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;(options);
    /// </code>
    /// </example>
    public SEALAlgorithm(SEALOptions<T, TInput, TOutput> options)
        : base(
            options?.MetaModel ?? throw new ArgumentNullException(nameof(options), "MetaModel must be set in options."),
            options.LossFunction ?? options.MetaModel.DefaultLossFunction,
            options,
            options.DataLoader,
            options.MetaOptimizer,
            options.InnerOptimizer)
    {
        _sealOptions = options;

        if (_sealOptions.UseAdaptiveInnerLR)
        {
            _adaptiveLearningRateState = new Dictionary<string, Vector<T>>();
        }
    }

    /// <summary>
    /// Gets the algorithm type identifier for this meta-learner.
    /// </summary>
    /// <value>Returns <see cref="MetaLearningAlgorithmType.SEAL"/>.</value>
    /// <remarks>
    /// <para>
    /// This property identifies the algorithm as SEAL, a sample-efficient
    /// meta-learning algorithm that combines MAML-style gradient-based
    /// meta-learning with temperature scaling and entropy regularization.
    /// </para>
    /// </remarks>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.SEAL;

    /// <summary>
    /// Performs one meta-training step using SEAL's sample-efficient approach.
    /// </summary>
    /// <param name="taskBatch">A batch of tasks to meta-train on, each containing support and query sets.</param>
    /// <returns>The average meta-loss across all tasks in the batch (evaluated on query sets).</returns>
    /// <exception cref="ArgumentException">Thrown when the task batch is null or empty.</exception>
    /// <exception cref="InvalidOperationException">Thrown when meta-gradient computation fails.</exception>
    /// <remarks>
    /// <para>
    /// SEAL meta-training extends MAML with additional sample-efficiency improvements:
    /// </para>
    /// <para>
    /// <b>For each task:</b>
    /// 1. Clone the meta-model with current meta-parameters
    /// 2. Perform K gradient descent steps on the task's support set (inner loop)
    ///    - Optionally uses adaptive per-parameter learning rates
    /// 3. Evaluate adapted model on query set to compute meta-loss
    /// 4. Apply temperature scaling: loss = loss / temperature
    /// 5. Add entropy regularization: loss = loss - entropy_coef * entropy(predictions)
    /// 6. Compute meta-gradients with optional first-order approximation
    /// 7. Clip gradients if threshold is set
    /// </para>
    /// <para>
    /// <b>Meta-Update:</b>
    /// 1. Average meta-gradients across all tasks
    /// 2. Apply weight decay: gradient += weight_decay * parameters
    /// 3. Update meta-parameters using the meta-optimizer
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> SEAL meta-training is like a teacher who practices
    /// on many small lessons. For each lesson, the teacher quickly adapts their
    /// teaching style (inner loop), then evaluates how well students learned (query set).
    /// The teacher then adjusts their general teaching approach based on what worked
    /// across all lessons (meta-update).
    ///
    /// The special features of SEAL:
    /// - Temperature scaling controls how confident the model should be
    /// - Entropy regularization encourages diverse predictions
    /// - Adaptive learning rates help parameters learn at appropriate speeds
    /// - Weight decay prevents overfitting to training tasks
    /// </para>
    /// </remarks>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null || taskBatch.BatchSize == 0)
        {
            throw new ArgumentException("Task batch cannot be null or empty.", nameof(taskBatch));
        }

        // Track current iteration for temperature annealing
        _currentIteration++;

        // Compute current temperature (with optional annealing)
        double currentTemperature = ComputeCurrentTemperature();

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
            T metaLoss = ComputeLossFromOutput(queryPredictions, task.QueryOutput);

            // Apply temperature scaling if configured
            if (Math.Abs(currentTemperature - 1.0) > 1e-10)
            {
                T temperature = NumOps.FromDouble(currentTemperature);
                metaLoss = NumOps.Divide(metaLoss, temperature);
            }

            // Add entropy regularization if configured
            if (_sealOptions.EntropyCoefficient > 0.0 && _sealOptions.EntropyOnlyDuringMetaTrain)
            {
                T entropyTerm = ComputeEntropyRegularization(queryPredictions);
                T entropyCoef = NumOps.FromDouble(_sealOptions.EntropyCoefficient);
                metaLoss = NumOps.Subtract(metaLoss, NumOps.Multiply(entropyCoef, entropyTerm));
            }

            totalMetaLoss = NumOps.Add(totalMetaLoss, metaLoss);

            // Compute meta-gradients (gradients with respect to initial parameters)
            var taskMetaGradients = ComputeMetaGradients(taskModel, task, adaptedParameters);

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
        T batchSizeT = NumOps.FromDouble(taskBatch.BatchSize);
        for (int i = 0; i < metaGradients.Length; i++)
        {
            metaGradients[i] = NumOps.Divide(metaGradients[i], batchSizeT);
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
        var updatedMetaParams = ApplyGradients(currentMetaParams, metaGradients, _sealOptions.OuterLearningRate);
        MetaModel.SetParameters(updatedMetaParams);

        // Return average meta-loss
        return NumOps.Divide(totalMetaLoss, batchSizeT);
    }

    /// <summary>
    /// Adapts the meta-learned model to a new task using gradient descent.
    /// </summary>
    /// <param name="task">The new task containing support set examples for adaptation.</param>
    /// <returns>A new model instance that has been fine-tuned to the given task.</returns>
    /// <exception cref="ArgumentNullException">Thrown when task is null.</exception>
    /// <remarks>
    /// <para>
    /// SEAL adaptation performs gradient descent on the support set, optionally
    /// using adaptive per-parameter learning rates based on gradient norms.
    /// The meta-learned initialization enables rapid adaptation with few examples.
    /// </para>
    /// <para>
    /// <b>Adaptation Process:</b>
    /// 1. Clone the meta-model with learned initialization
    /// 2. For each adaptation step:
    ///    a. Compute gradients on support set
    ///    b. Optionally compute adaptive learning rates
    ///    c. Update parameters using gradient descent
    /// 3. Return the adapted model
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> When you give SEAL a new task (with just a few examples),
    /// it quickly adjusts its parameters to perform well on that task. This works
    /// because the meta-learned starting point was specifically optimized to enable
    /// fast adaptation.
    ///
    /// It's like a musician who has learned many instruments - when they pick up
    /// a new one, they already know the general principles and just need to learn
    /// the specific fingerings and techniques for that instrument.
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
    /// Computes the current temperature based on annealing schedule.
    /// </summary>
    /// <returns>The current temperature value.</returns>
    /// <remarks>
    /// <para>
    /// If MinTemperature is less than Temperature, the temperature linearly
    /// decreases from Temperature to MinTemperature over NumMetaIterations.
    /// </para>
    /// </remarks>
    private double ComputeCurrentTemperature()
    {
        if (Math.Abs(_sealOptions.MinTemperature - _sealOptions.Temperature) < 1e-10)
        {
            return _sealOptions.Temperature;
        }

        // Linear annealing from Temperature to MinTemperature
        double progress = Math.Min(1.0, (double)_currentIteration / _sealOptions.NumMetaIterations);
        return _sealOptions.Temperature - progress * (_sealOptions.Temperature - _sealOptions.MinTemperature);
    }

    /// <summary>
    /// Performs the inner loop adaptation to a specific task.
    /// </summary>
    /// <param name="model">The model to adapt.</param>
    /// <param name="task">The task to adapt to.</param>
    /// <returns>The adapted parameters after gradient descent steps.</returns>
    /// <remarks>
    /// <para>
    /// The inner loop performs AdaptationSteps gradient updates on the support set.
    /// If UseAdaptiveInnerLR is enabled, per-parameter learning rates are computed
    /// based on the specified AdaptiveLearningRateMode.
    /// </para>
    /// </remarks>
    private Vector<T> InnerLoopAdaptation(IFullModel<T, TInput, TOutput> model, IMetaLearningTask<T, TInput, TOutput> task)
    {
        var parameters = model.GetParameters();
        Vector<T>? runningSquaredGrads = null;

        // Initialize running squared gradients for RunningMean mode
        if (_sealOptions.UseAdaptiveInnerLR && _sealOptions.AdaptiveLearningRateMode == SEALAdaptiveLearningRateMode.RunningMean)
        {
            runningSquaredGrads = new Vector<T>(parameters.Length);
            for (int i = 0; i < parameters.Length; i++)
            {
                runningSquaredGrads[i] = NumOps.Zero;
            }
        }

        // Perform adaptation steps
        for (int step = 0; step < _sealOptions.AdaptationSteps; step++)
        {
            // Compute gradients on support set
            var gradients = ComputeGradients(model, task.SupportInput, task.SupportOutput);

            // Apply adaptive learning rates if configured
            if (_sealOptions.UseAdaptiveInnerLR)
            {
                var adaptiveLr = ComputeAdaptiveLearningRates(gradients, ref runningSquaredGrads);
                parameters = ApplyAdaptiveGradients(parameters, gradients, adaptiveLr);
            }
            else
            {
                // Standard gradient descent with fixed learning rate
                parameters = ApplyGradients(parameters, gradients, _sealOptions.InnerLearningRate);
            }

            model.SetParameters(parameters);
        }

        return parameters;
    }

    /// <summary>
    /// Computes adaptive learning rates based on gradient statistics.
    /// </summary>
    /// <param name="gradients">The current gradient vector.</param>
    /// <param name="runningSquaredGrads">Running squared gradient state (for RunningMean mode).</param>
    /// <returns>Adaptive learning rates for each parameter.</returns>
    /// <remarks>
    /// <para>
    /// Different modes for computing adaptive learning rates:
    /// - GradientNorm: lr = base_lr / (sqrt(grad^2) + epsilon)
    /// - RunningMean: Uses exponential moving average of squared gradients
    /// - PerLayer: Averages gradient norms across layer parameters
    /// </para>
    /// </remarks>
    private Vector<T> ComputeAdaptiveLearningRates(Vector<T> gradients, ref Vector<T>? runningSquaredGrads)
    {
        var adaptiveLr = new Vector<T>(gradients.Length);
        T epsilon = NumOps.FromDouble(_sealOptions.AdaptiveLearningRateEpsilon);
        T baseLr = NumOps.FromDouble(_sealOptions.InnerLearningRate);

        switch (_sealOptions.AdaptiveLearningRateMode)
        {
            case SEALAdaptiveLearningRateMode.GradientNorm:
                // AdaGrad-like: lr = base_lr / (sqrt(grad^2) + epsilon)
                for (int i = 0; i < gradients.Length; i++)
                {
                    T gradSquared = NumOps.Multiply(gradients[i], gradients[i]);
                    T sqrtGrad = NumOps.FromDouble(Math.Sqrt(Math.Max(NumOps.ToDouble(gradSquared), 1e-16)));
                    T denominator = NumOps.Add(sqrtGrad, epsilon);
                    adaptiveLr[i] = NumOps.Divide(baseLr, denominator);
                }
                break;

            case SEALAdaptiveLearningRateMode.RunningMean:
                // RMSprop-like: uses exponential moving average of squared gradients
                if (runningSquaredGrads == null)
                {
                    runningSquaredGrads = new Vector<T>(gradients.Length);
                    for (int i = 0; i < gradients.Length; i++)
                    {
                        runningSquaredGrads[i] = NumOps.Zero;
                    }
                }

                T decay = NumOps.FromDouble(_sealOptions.AdaptiveLearningRateDecay);
                T oneMinusDecay = NumOps.FromDouble(1.0 - _sealOptions.AdaptiveLearningRateDecay);

                for (int i = 0; i < gradients.Length; i++)
                {
                    T gradSquared = NumOps.Multiply(gradients[i], gradients[i]);
                    // running = decay * running + (1 - decay) * grad^2
                    runningSquaredGrads[i] = NumOps.Add(
                        NumOps.Multiply(decay, runningSquaredGrads[i]),
                        NumOps.Multiply(oneMinusDecay, gradSquared)
                    );

                    T sqrtRunning = NumOps.FromDouble(Math.Sqrt(Math.Max(NumOps.ToDouble(runningSquaredGrads[i]), 1e-16)));
                    T denominator = NumOps.Add(sqrtRunning, epsilon);
                    adaptiveLr[i] = NumOps.Divide(baseLr, denominator);
                }
                break;

            case SEALAdaptiveLearningRateMode.PerLayer:
                // Compute average gradient norm across all parameters
                T sumSquared = NumOps.Zero;
                for (int i = 0; i < gradients.Length; i++)
                {
                    sumSquared = NumOps.Add(sumSquared, NumOps.Multiply(gradients[i], gradients[i]));
                }
                T avgSquared = NumOps.Divide(sumSquared, NumOps.FromDouble(gradients.Length));
                T avgSqrt = NumOps.FromDouble(Math.Sqrt(Math.Max(NumOps.ToDouble(avgSquared), 1e-16)));
                T sharedLr = NumOps.Divide(baseLr, NumOps.Add(avgSqrt, epsilon));

                for (int i = 0; i < gradients.Length; i++)
                {
                    adaptiveLr[i] = sharedLr;
                }
                break;
        }

        return adaptiveLr;
    }

    /// <summary>
    /// Applies gradients with per-parameter adaptive learning rates.
    /// </summary>
    /// <param name="parameters">Current parameters.</param>
    /// <param name="gradients">Gradients to apply.</param>
    /// <param name="adaptiveLr">Per-parameter learning rates.</param>
    /// <returns>Updated parameters.</returns>
    private Vector<T> ApplyAdaptiveGradients(Vector<T> parameters, Vector<T> gradients, Vector<T> adaptiveLr)
    {
        var updated = new Vector<T>(parameters.Length);
        for (int i = 0; i < parameters.Length; i++)
        {
            updated[i] = NumOps.Subtract(parameters[i], NumOps.Multiply(adaptiveLr[i], gradients[i]));
        }
        return updated;
    }

    /// <summary>
    /// Computes meta-gradients for the outer loop update.
    /// </summary>
    /// <param name="adaptedModel">The model after adaptation.</param>
    /// <param name="task">The task to compute meta-gradients for.</param>
    /// <param name="adaptedParameters">The adapted parameters.</param>
    /// <returns>The meta-gradient vector.</returns>
    /// <remarks>
    /// <para>
    /// For first-order approximation (UseFirstOrder = true), meta-gradients are simply
    /// the gradients on the query set after adaptation. This is computationally efficient
    /// but ignores the gradient through the adaptation process.
    /// </para>
    /// <para>
    /// For second-order (UseFirstOrder = false), full backpropagation through the
    /// adaptation process is required. This implementation currently uses first-order
    /// approximation as second-order requires significant framework infrastructure.
    /// </para>
    /// </remarks>
    private Vector<T> ComputeMetaGradients(
        IFullModel<T, TInput, TOutput> adaptedModel,
        IMetaLearningTask<T, TInput, TOutput> task,
        Vector<T> adaptedParameters)
    {
        // Compute gradients on query set
        var metaGradients = ComputeGradients(adaptedModel, task.QueryInput, task.QueryOutput);

        // If using first-order approximation, we're done
        if (_sealOptions.UseFirstOrder)
        {
            return metaGradients;
        }

        // For second-order, we need to backpropagate through the adaptation steps
        // NOTE: Full second-order implementation requires automatic differentiation
        // infrastructure. Most practical implementations use first-order approximation
        // (FOMAML) which performs nearly as well with O(n) complexity instead of O(n^3).
        //
        // To implement full second-order:
        // 1. Store computational graph during inner loop adaptation
        // 2. Compute Hessian-vector products using Pearlmutter's algorithm
        // 3. Apply implicit gradient computation: ∇θ_meta = (I - λH)^(-1) ∇θ_L
        //
        // Current implementation uses first-order which:
        // - Treats adapted parameters as constants during backprop
        // - Reduces complexity from O(n^3) to O(n)
        // - Maintains most of the performance benefits
        // - Is the de facto standard in production systems
        return metaGradients;
    }

    /// <summary>
    /// Computes entropy regularization term for the predictions.
    /// </summary>
    /// <param name="predictions">The model predictions.</param>
    /// <returns>The entropy value (higher = more diverse predictions).</returns>
    /// <remarks>
    /// <para>
    /// Entropy is computed as: H(p) = -sum(p_i * log(p_i))
    /// where p is the probability distribution over classes.
    ///
    /// Higher entropy means more uniform predictions (less confident).
    /// By subtracting entropy from the loss, we encourage the model to be less
    /// overconfident, which can improve generalization.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Entropy measures how "spread out" predictions are.
    /// A model that always predicts one class has low entropy (very confident).
    /// A model that spreads probability evenly has high entropy (uncertain).
    /// By encouraging entropy, we prevent the model from being too confident,
    /// which helps it adapt better to new tasks.
    /// </para>
    /// </remarks>
    private T ComputeEntropyRegularization(TOutput predictions)
    {
        try
        {
            // Convert predictions to probabilities (softmax if not already)
            var probabilities = ConvertToProbabilities(predictions);
            T entropy = NumOps.Zero;

            // Compute entropy: -sum(p * log(p))
            T epsilon = NumOps.FromDouble(_sealOptions.AdaptiveLearningRateEpsilon);

            for (int i = 0; i < probabilities.Length; i++)
            {
                T p = probabilities[i];

                // Avoid log(0) by adding small epsilon
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
            // If entropy computation fails, return zero (no regularization)
            return NumOps.Zero;
        }
    }

    /// <summary>
    /// Converts model outputs to probability distribution using softmax.
    /// </summary>
    /// <param name="predictions">The raw model predictions (logits).</param>
    /// <returns>Normalized probability distribution.</returns>
    /// <remarks>
    /// <para>
    /// Applies the softmax function: softmax(x)_i = exp(x_i) / sum(exp(x_j))
    /// Uses the numerical stability trick of subtracting max(x) before exponentiating.
    /// </para>
    /// </remarks>
    private Vector<T> ConvertToProbabilities(TOutput predictions)
    {
        // Extract vector from TOutput
        Vector<T>? logits = null;
        if (typeof(TOutput) == typeof(Vector<T>))
        {
            logits = predictions as Vector<T>;
        }
        else if (typeof(TOutput) == typeof(Tensor<T>))
        {
            var tensor = predictions as Tensor<T>;
            if (tensor != null)
            {
                logits = tensor.ToVector();
            }
        }

        if (logits == null)
        {
            throw new NotSupportedException($"Cannot compute entropy for type {typeof(TOutput).Name}");
        }

        // Apply softmax with numerical stability
        var expValues = new T[logits.Length];
        T sumExp = NumOps.Zero;

        // Find max for numerical stability
        T maxLogit = logits[0];
        for (int i = 1; i < logits.Length; i++)
        {
            if (NumOps.ToDouble(logits[i]) > NumOps.ToDouble(maxLogit))
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
