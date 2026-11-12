using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Data.Structures;

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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("How to Train Your MAML",
    "https://arxiv.org/abs/1810.09502",
    Year = 2019,
    Authors = "Antoniou, A., Edwards, H., & Storkey, A.")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class SEALAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{

    private readonly SEALOptions<T, TInput, TOutput> _sealOptions;
    [Buffer]
    private readonly Dictionary<string, Vector<T>>? _adaptiveLearningRateState;

    /// <summary>
    /// Initializes a new instance of the SEALAlgorithm class.
    /// </summary>
    /// <param name="options">SEAL configuration options containing the model and all hyperparameters.</param>
    /// <exception cref="ArgumentNullException">Thrown when options is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when required components are not set in options.</exception>
    /// <example>
    /// <code>
    /// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(inputFeatures: 4, outputSize: 2);
    /// var myNeuralNetwork = new NeuralNetwork&lt;double&gt;(architecture);
    /// // Create SEAL with minimal configuration
    /// var options = new SEALOptions&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(myNeuralNetwork);
    /// var seal = new SEALAlgorithm&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(options);
    ///
    /// // Create SEAL with entropy regularization
    /// var options2 = new SEALOptions&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(myNeuralNetwork)
    /// {
    ///     EntropyCoefficient = 0.01,
    ///     Temperature = 1.5,
    ///     UseAdaptiveInnerLR = true,
    ///     AdaptiveLearningRateMode = SEALAdaptiveLearningRateMode.RunningMean
    /// };
    /// var seal2 = new SEALAlgorithm&lt;double, Tensor&lt;double&gt;, Tensor&lt;double&gt;&gt;(options2);
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
        RequirePlainSgdInnerSteps(options.UseAdaptiveInnerLR, "adaptive per-parameter inner learning rates");

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

        // Compute current temperature (with optional annealing)
        double currentTemperature = ComputeCurrentTemperature();

        // Accumulate meta-gradients across all tasks in the batch
        Vector<T>? metaGradients = null;
        var metaLosses = new List<T>(taskBatch.BatchSize);

        foreach (var task in taskBatch.Tasks)
        {
            // Clone the meta model for this task
            var taskModel = CloneModel();

            // Inner loop: Adapt to the task using support set
            var adaptedParameters = InnerLoopAdaptation(taskModel, task);
            InterfaceGuard.Parameterizable(taskModel).SetParameters(adaptedParameters);

            // Evaluate on query set to get meta-loss
            var queryPredictions = taskModel.Predict(task.QueryInput);
            metaLosses.Add(RegularizedMetaLoss(
                ComputeLossFromOutput(queryPredictions, task.QueryOutput), queryPredictions, currentTemperature));

            // Compute meta-gradients (gradients with respect to initial parameters)
            var taskMetaGradients = ComputeMetaGradients(taskModel, task, adaptedParameters);

            // Clip gradients if threshold is set
            if (_sealOptions.GradientClipThreshold.HasValue)
            {
                taskMetaGradients = ClipGradients(taskMetaGradients, _sealOptions.GradientClipThreshold.Value);
            }

            // Accumulate meta-gradients
            metaGradients = metaGradients == null ? taskMetaGradients : Engine.Add(metaGradients, taskMetaGradients);
        }

        if (metaGradients == null)
        {
            throw new InvalidOperationException("Failed to compute meta-gradients.");
        }

        // Average the meta-gradients
        T batchSizeT = NumOps.FromDouble(taskBatch.BatchSize);
        metaGradients = Engine.Divide(metaGradients, batchSizeT);

        // Apply weight decay if configured: gradient += weight_decay * parameters
        if (_sealOptions.WeightDecay > 0.0)
        {
            metaGradients = Engine.Add(
                metaGradients, Engine.Multiply(ParamModel.GetParameters(), NumOps.FromDouble(_sealOptions.WeightDecay)));
        }

        // Outer loop: Update meta-parameters
        var currentMetaParams = ParamModel.GetParameters();
        var updatedMetaParams = ApplyOuterGradients(currentMetaParams, metaGradients, _sealOptions.OuterLearningRate);
        ParamModel.SetParameters(updatedMetaParams);

        // Return average meta-loss
        return NumOps.Divide(Engine.Sum(new Vector<T>(metaLosses.ToArray())), batchSizeT);
    }

    /// <summary>
    /// The query loss with temperature scaling (loss / temperature) and, when configured, the entropy
    /// bonus subtracted: loss - entropy_coef * H(predictions).
    /// </summary>
    private T RegularizedMetaLoss(T metaLoss, TOutput queryPredictions, double currentTemperature)
    {
        if (Math.Abs(currentTemperature - 1.0) > 1e-10)
        {
            metaLoss = NumOps.Divide(metaLoss, NumOps.FromDouble(currentTemperature));
        }

        if (_sealOptions.EntropyCoefficient > 0.0 && _sealOptions.EntropyOnlyDuringMetaTrain)
        {
            T entropyTerm = ComputeEntropyRegularization(queryPredictions);
            metaLoss = NumOps.Subtract(metaLoss, NumOps.Multiply(NumOps.FromDouble(_sealOptions.EntropyCoefficient), entropyTerm));
        }

        return metaLoss;
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
        InterfaceGuard.Parameterizable(adaptedModel).SetParameters(adaptedParameters);

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
        var parameters = InterfaceGuard.Parameterizable(model).GetParameters();
        Vector<T>? runningSquaredGrads = null;

        // Initialize running squared gradients for RunningMean mode
        if (_sealOptions.UseAdaptiveInnerLR && _sealOptions.AdaptiveLearningRateMode == SEALAdaptiveLearningRateMode.RunningMean)
        {
            runningSquaredGrads = Engine.Fill(parameters.Length, NumOps.Zero);
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
                parameters = ApplyInnerGradients(parameters, gradients, _sealOptions.InnerLearningRate, step == 0);
            }

            InterfaceGuard.Parameterizable(model).SetParameters(parameters);
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
    internal Vector<T> ComputeAdaptiveLearningRates(Vector<T> gradients, ref Vector<T>? runningSquaredGrads)
    {
        T epsilon = NumOps.FromDouble(_sealOptions.AdaptiveLearningRateEpsilon);
        T baseLr = NumOps.FromDouble(_sealOptions.InnerLearningRate);

        switch (_sealOptions.AdaptiveLearningRateMode)
        {
            case SEALAdaptiveLearningRateMode.GradientNorm:
                // AdaGrad-like: lr = base_lr / (sqrt(grad^2) + epsilon). sqrt(max(grad^2, 1e-16)) = max(|grad|, 1e-8),
                // taken without squaring so a gradient past sqrt(MaxValue) (256 in Half) does not overflow to inf.
                var magnitude = Engine.TensorMax(Engine.TensorAbs(Tensor<T>.FromVector(gradients)), NumOps.FromDouble(1e-8));
                return ScaledInverse(magnitude, baseLr, epsilon);

            case SEALAdaptiveLearningRateMode.RunningMean:
                // RMSprop-like: running = decay * running + (1 - decay) * grad^2, lr = base_lr / (sqrt(running) + epsilon)
                var previous = runningSquaredGrads ?? Engine.Fill(gradients.Length, NumOps.Zero);
                runningSquaredGrads = Engine.Lerp(
                    Engine.Multiply(gradients, gradients), previous, NumOps.FromDouble(_sealOptions.AdaptiveLearningRateDecay));
                return ScaledInverse(
                    Engine.TensorSqrt(Engine.TensorMax(Tensor<T>.FromVector(runningSquaredGrads), NumOps.FromDouble(1e-16))),
                    baseLr, epsilon);

            case SEALAdaptiveLearningRateMode.PerLayer:
                // One shared rate from the root-mean-square gradient across all parameters. The statistic is taken in
                // double (one conversion of the gradient), so neither the squares, their sum nor the parameter count
                // overflow a narrow T, and the 1e-8 floor and epsilon do not round to zero.
                if (gradients.Length == 0) return new Vector<T>(0);
                var wide = Tensor<T>.FromVector(gradients).Cast<double>();
                double rms = Math.Sqrt(Engine.TensorMean(Engine.TensorMultiply(wide, wide)));

                double sharedRate = _sealOptions.InnerLearningRate / (Math.Max(rms, 1e-8) + _sealOptions.AdaptiveLearningRateEpsilon);
                return Engine.Fill(gradients.Length, NumOps.FromDouble(Math.Min(sharedRate, NumOps.ToDouble(NumOps.MaxValue))));

            default:
                return new Vector<T>(gradients.Length);
        }
    }

    /// <summary>
    /// <c>base_lr / (root + epsilon)</c>, elementwise, capped at the largest finite T. In double the cap never binds
    /// (base_lr / 1e-8 is far below double.MaxValue); in Half the 1e-8 floor and epsilon round to 0, so a zero
    /// gradient would give 1 / 0 = inf and the update inf * 0 = NaN. The cap keeps that update at exactly 0.
    /// </summary>
    private Vector<T> ScaledInverse(Tensor<T> root, T baseLr, T epsilon)
    {
        var rates = Engine.TensorMultiplyScalar(Engine.TensorReciprocal(Engine.TensorAddScalar(root, epsilon)), baseLr);
        return Engine.TensorClamp(rates, NumOps.Zero, NumOps.MaxValue).ToVector();
    }

    /// <summary>
    /// Applies gradients with per-parameter adaptive learning rates.
    /// </summary>
    /// <param name="parameters">Current parameters.</param>
    /// <param name="gradients">Gradients to apply.</param>
    /// <param name="adaptiveLr">Per-parameter learning rates.</param>
    /// <returns>Updated parameters.</returns>
    private Vector<T> ApplyAdaptiveGradients(Vector<T> parameters, Vector<T> gradients, Vector<T> adaptiveLr)
        => Engine.Subtract(parameters, Engine.Multiply(adaptiveLr, gradients));

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
    internal T ComputeEntropyRegularization(TOutput predictions)
    {
        Vector<T> probabilities;
        try
        {
            // Convert predictions to probabilities (softmax if not already)
            probabilities = ConvertToProbabilities(predictions);
        }
        catch (NotSupportedException)
        {
            // Outputs that are not a probability vector carry no entropy term (no regularization).
            return NumOps.Zero;
        }

        // Avoid log(0) by adding a small epsilon, then renormalize: p' = (p + eps) / sum(p + eps).
        var shifted = Engine.TensorAddScalar(
            Tensor<T>.FromVector(probabilities), NumOps.FromDouble(_sealOptions.AdaptiveLearningRateEpsilon));
        var p = Engine.TensorDivideScalar(shifted, Engine.TensorSum(shifted));

        // H = -sum(p' * log(p')) with 0 log 0 = 0. In double the epsilon shift keeps every p' > 0; in Half epsilon
        // rounds to 0, an underflowed p' is exactly 0, and log is taken of 1 there so the term is 0 instead of NaN.
        var safe = Engine.TensorWhere(Engine.TensorGreaterThan(p, NumOps.Zero), p, Engine.TensorOnesLike(p));
        return NumOps.Negate(Engine.TensorSum(Engine.TensorMultiply(p, Engine.TensorLog(safe))));
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

        // SIMD softmax via Engine
        return Softmax(logits);
    }
}
