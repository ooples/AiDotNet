using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Models;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Tensors;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of Body Only Inner Loop (BOIL) meta-learning algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// BOIL is the opposite of ANIL - it only adapts the feature extractor (body) during
/// inner-loop adaptation while keeping the classification head frozen. This explores
/// whether learning task-specific representations is more important than task-specific classifiers.
/// </para>
/// <para>
/// <b>Key Insight:</b> ANIL showed that adapting only the head works well, suggesting
/// the body learns general features. BOIL tests the complementary hypothesis: what if
/// we need to adapt HOW we see things (features) rather than HOW we decide (classifier)?
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of a neural network as having two jobs:
/// </para>
/// <list type="number">
/// <item><b>Feature Extraction (Body):</b> "What do I see in this image?"</item>
/// <item><b>Classification (Head):</b> "Given what I see, which class is it?"</item>
/// </list>
/// <para>
/// BOIL says: "The classifier is general enough - we just need to learn to SEE things
/// differently for each task!" So it only updates how the network extracts features,
/// while the decision-making layer stays fixed.
/// </para>
/// <para>
/// <b>Algorithm (MAML-style with body-only adaptation):</b>
/// <code>
/// For each task batch:
///   For each task:
///     1. Clone body parameters, keep head parameters frozen
///     2. For each adaptation step:
///        a. Forward pass through (adaptable body) + (frozen head)
///        b. Compute loss on support set
///        c. Compute gradients for BODY ONLY
///        d. Update body parameters
///     3. Evaluate adapted body on query set
///   Meta-update: head + body initialization
/// </code>
/// </para>
/// <para>
/// Reference: Oh, J., Yoo, H., Kim, C., &amp; Yun, S. Y. (2021).
/// BOIL: Towards Representation Change for Few-shot Learning.
/// </para>
/// </remarks>
public class BOILAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly BOILOptions<T, TInput, TOutput> _boilOptions;

    // Head parameters (frozen during inner loop)
    private Vector<T> _headWeights;
    private Vector<T>? _headBias;

    // Body parameters (adapted per-task)
    private int _bodyParameterCount;
    private int _headParameterCount;

    /// <summary>
    /// Initializes a new instance of the BOILAlgorithm class.
    /// </summary>
    /// <param name="options">BOIL configuration options containing the model and all hyperparameters.</param>
    /// <exception cref="ArgumentNullException">Thrown when options is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when required components are not set in options.</exception>
    /// <example>
    /// <code>
    /// // Create BOIL with minimal configuration
    /// var options = new BOILOptions&lt;double, Tensor, Tensor&gt;(myNeuralNetwork);
    /// var boil = new BOILAlgorithm&lt;double, Tensor, Tensor&gt;(options);
    ///
    /// // Create BOIL with custom configuration
    /// var options = new BOILOptions&lt;double, Tensor, Tensor&gt;(myNeuralNetwork)
    /// {
    ///     AdaptationSteps = 5,
    ///     InnerLearningRate = 0.01,
    ///     NumClasses = 5,
    ///     BodyAdaptationFraction = 0.5
    /// };
    /// var boil = new BOILAlgorithm&lt;double, Tensor, Tensor&gt;(options);
    /// </code>
    /// </example>
    public BOILAlgorithm(BOILOptions<T, TInput, TOutput> options)
        : base(
            options?.MetaModel ?? throw new ArgumentNullException(nameof(options), "MetaModel must be set in options."),
            options.LossFunction ?? options.MetaModel.DefaultLossFunction,
            options,
            options.DataLoader,
            options.MetaOptimizer,
            options.InnerOptimizer)
    {
        _boilOptions = options;

        // Initialize head weights
        _headWeights = new Vector<T>(options.FeatureDimension * options.NumClasses);

        // Initialize head and body parameters
        InitializeParameters();
    }

    /// <summary>
    /// Gets the algorithm type identifier for this meta-learner.
    /// </summary>
    /// <value>Returns <see cref="MetaLearningAlgorithmType.BOIL"/>.</value>
    /// <remarks>
    /// <para>
    /// This property identifies the algorithm as BOIL (Body Only Inner Loop),
    /// which adapts only the feature extractor during inner-loop adaptation.
    /// </para>
    /// </remarks>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.BOIL;

    /// <summary>
    /// Performs one meta-training step using BOIL's body-only adaptation approach.
    /// </summary>
    /// <param name="taskBatch">A batch of tasks to meta-train on, each containing support and query sets.</param>
    /// <returns>The average meta-loss across all tasks in the batch (evaluated on query sets).</returns>
    /// <exception cref="ArgumentException">Thrown when the task batch is null or empty.</exception>
    /// <exception cref="InvalidOperationException">Thrown when meta-gradient computation fails.</exception>
    /// <remarks>
    /// <para>
    /// BOIL meta-training adapts only the body parameters in the inner loop:
    /// </para>
    /// <para>
    /// <b>BOIL Inner Loop (per task):</b>
    /// <code>
    /// 1. Clone body parameters from meta-learned initialization
    /// 2. Keep head parameters FROZEN
    /// 3. For each adaptation step:
    ///    a. Forward: (adaptable body) → features → (frozen head) → output
    ///    b. Compute loss on support set
    ///    c. Compute gradients for body only
    ///    d. Update body parameters
    /// 4. Evaluate on query set with adapted body
    /// </code>
    /// </para>
    /// <para>
    /// <b>BOIL Outer Loop:</b>
    /// <code>
    /// 1. Accumulate gradients from all tasks' query losses
    /// 2. Update body initialization to provide better starting point
    /// 3. Update head weights (frozen during inner loop, but updated in outer loop)
    /// </code>
    /// </para>
    /// <para>
    /// <b>Key Difference from ANIL:</b>
    /// - ANIL: Adapts head only, freezes body
    /// - BOIL: Adapts body only, freezes head
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> BOIL learns two things:
    /// 1. A good classifier (head) that works for all tasks (fixed during adaptation)
    /// 2. A good feature extractor (body) that can be quickly adjusted per task
    ///
    /// During adaptation, only the "seeing" part changes - the "decision" stays fixed.
    /// This tests whether task-specific vision is more important than task-specific decisions.
    /// </para>
    /// </remarks>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null || taskBatch.BatchSize == 0)
        {
            throw new ArgumentException("Task batch cannot be null or empty.", nameof(taskBatch));
        }

        // Accumulate gradients for body and head
        Vector<T>? accumulatedBodyGradients = null;
        Vector<T>? accumulatedHeadGradients = null;
        T totalLoss = NumOps.Zero;

        foreach (var task in taskBatch.Tasks)
        {
            // Clone body parameters for this task
            var taskBodyParams = CloneBodyParameters();

            // Inner loop: adapt body on support set (head is frozen)
            for (int step = 0; step < _boilOptions.AdaptationSteps; step++)
            {
                // Forward pass with current body + frozen head
                var supportPredictions = ForwardWithBody(task.SupportInput, taskBodyParams);

                // Compute loss on support set
                T supportLoss = ComputeLossFromOutput(supportPredictions, task.SupportOutput);

                // Add L2 regularization if configured
                if (_boilOptions.BodyL2Regularization > 0)
                {
                    T l2Penalty = ComputeL2Penalty(taskBodyParams);
                    supportLoss = NumOps.Add(supportLoss,
                        NumOps.Multiply(NumOps.FromDouble(_boilOptions.BodyL2Regularization), l2Penalty));
                }

                // Compute gradients for BODY ONLY
                var bodyGradients = ComputeBodyGradients(task.SupportInput, task.SupportOutput, taskBodyParams);

                // Apply layer-wise learning rates if configured
                if (_boilOptions.UseLayerwiseLearningRates)
                {
                    ApplyLayerwiseLearningRates(bodyGradients);
                }

                // Update body parameters
                double effectiveLR = _boilOptions.InnerLearningRate;
                taskBodyParams = ApplyGradients(taskBodyParams, bodyGradients, effectiveLR);
            }

            // Evaluate on query set with adapted body
            var queryPredictions = ForwardWithBody(task.QueryInput, taskBodyParams);
            T taskLoss = ComputeLossFromOutput(queryPredictions, task.QueryOutput);
            totalLoss = NumOps.Add(totalLoss, taskLoss);

            // Compute meta-gradients
            var (bodyGrads, headGrads) = ComputeMetaGradients(task, taskBodyParams, taskLoss);

            // Accumulate gradients
            if (accumulatedBodyGradients == null)
            {
                accumulatedBodyGradients = bodyGrads;
                accumulatedHeadGradients = headGrads;
            }
            else
            {
                for (int i = 0; i < accumulatedBodyGradients.Length; i++)
                {
                    accumulatedBodyGradients[i] = NumOps.Add(accumulatedBodyGradients[i], bodyGrads[i]);
                }
                if (accumulatedHeadGradients != null)
                {
                    for (int i = 0; i < accumulatedHeadGradients.Length; i++)
                    {
                        accumulatedHeadGradients[i] = NumOps.Add(accumulatedHeadGradients[i], headGrads[i]);
                    }
                }
            }
        }

        if (accumulatedBodyGradients == null || accumulatedHeadGradients == null)
        {
            throw new InvalidOperationException("Failed to compute meta-gradients.");
        }

        // Average gradients
        T batchSizeT = NumOps.FromDouble(taskBatch.BatchSize);
        for (int i = 0; i < accumulatedBodyGradients.Length; i++)
        {
            accumulatedBodyGradients[i] = NumOps.Divide(accumulatedBodyGradients[i], batchSizeT);
        }
        for (int i = 0; i < accumulatedHeadGradients.Length; i++)
        {
            accumulatedHeadGradients[i] = NumOps.Divide(accumulatedHeadGradients[i], batchSizeT);
        }

        // Clip gradients if configured
        if (_boilOptions.GradientClipThreshold.HasValue && _boilOptions.GradientClipThreshold.Value > 0)
        {
            accumulatedBodyGradients = ClipGradients(accumulatedBodyGradients, _boilOptions.GradientClipThreshold.Value);
            accumulatedHeadGradients = ClipGradients(accumulatedHeadGradients, _boilOptions.GradientClipThreshold.Value);
        }

        // Update body initialization (outer loop)
        UpdateBodyInitialization(accumulatedBodyGradients);

        // Update head parameters (outer loop only - frozen during inner loop)
        _headWeights = ApplyGradients(_headWeights, accumulatedHeadGradients, _boilOptions.OuterLearningRate);

        return NumOps.Divide(totalLoss, batchSizeT);
    }

    /// <summary>
    /// Adapts the meta-learned model to a new task by only updating the feature extractor (body).
    /// </summary>
    /// <param name="task">The new task containing support set examples for adaptation.</param>
    /// <returns>A new model instance that has been adapted to the given task with an updated body.</returns>
    /// <exception cref="ArgumentNullException">Thrown when task is null.</exception>
    /// <remarks>
    /// <para>
    /// BOIL adaptation updates only the body (feature extractor) while keeping the head frozen:
    /// </para>
    /// <list type="number">
    /// <item>Clone the meta-learned body parameters (or reinitialize if configured)</item>
    /// <item>Keep head parameters frozen</item>
    /// <item>For each adaptation step:</item>
    ///   <item>Forward pass: body → features → head → output</item>
    ///   <item>Compute loss on support set</item>
    ///   <item>Update body parameters with gradient descent</item>
    /// <item>Return model with adapted body + frozen head</item>
    /// </list>
    /// <para>
    /// <b>For Beginners:</b> When you give BOIL a new task:
    /// 1. It keeps its "decision maker" (classifier head) exactly the same
    /// 2. It only retrains the "feature extractor" (how it sees inputs)
    /// 3. This is like teaching someone who already knows what categories exist,
    ///    just teaching them what to look for in the new domain
    /// </para>
    /// <para>
    /// <b>Use Case:</b> BOIL might work better when:
    /// - Different tasks require seeing different patterns in the input
    /// - The classification boundaries are similar across tasks
    /// - You have a robust meta-learned classifier
    /// </para>
    /// </remarks>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        if (task == null)
        {
            throw new ArgumentNullException(nameof(task));
        }

        // Clone or reinitialize body parameters
        var adaptedBodyParams = _boilOptions.ReinitializeBody
            ? InitializeBodyParameters()
            : CloneBodyParameters();

        // Inner loop: adapt body on support set
        for (int step = 0; step < _boilOptions.AdaptationSteps; step++)
        {
            // Forward pass with current body + frozen head
            var supportPredictions = ForwardWithBody(task.SupportInput, adaptedBodyParams);

            // Compute loss on support set
            T supportLoss = ComputeLossFromOutput(supportPredictions, task.SupportOutput);

            // Add L2 regularization if configured
            if (_boilOptions.BodyL2Regularization > 0)
            {
                T l2Penalty = ComputeL2Penalty(adaptedBodyParams);
                supportLoss = NumOps.Add(supportLoss,
                    NumOps.Multiply(NumOps.FromDouble(_boilOptions.BodyL2Regularization), l2Penalty));
            }

            // Compute gradients for BODY ONLY
            var bodyGradients = ComputeBodyGradients(task.SupportInput, task.SupportOutput, adaptedBodyParams);

            // Apply layer-wise learning rates if configured
            if (_boilOptions.UseLayerwiseLearningRates)
            {
                ApplyLayerwiseLearningRates(bodyGradients);
            }

            // Update body parameters
            adaptedBodyParams = ApplyGradients(adaptedBodyParams, bodyGradients, _boilOptions.InnerLearningRate);
        }

        // Create BOIL model with adapted body and frozen head
        return new BOILModel<T, TInput, TOutput>(
            CloneModel(),
            adaptedBodyParams,
            _headWeights,
            _headBias,
            _boilOptions);
    }

    #region Parameter Management

    /// <summary>
    /// Initializes head and body parameter counts.
    /// </summary>
    private void InitializeParameters()
    {
        var totalParams = MetaModel.GetParameters();

        // Estimate head size
        _headParameterCount = _boilOptions.FeatureDimension * _boilOptions.NumClasses;
        if (_headParameterCount > totalParams.Length)
        {
            _headParameterCount = (int)(totalParams.Length * 0.1);
        }
        _bodyParameterCount = totalParams.Length - _headParameterCount;

        // Initialize head weights
        _headWeights = InitializeHeadWeights();
        if (_boilOptions.NumClasses > 0)
        {
            _headBias = InitializeHeadBias();
        }
    }

    /// <summary>
    /// Initializes head weights using Xavier initialization.
    /// </summary>
    private Vector<T> InitializeHeadWeights()
    {
        int size = _boilOptions.FeatureDimension * _boilOptions.NumClasses;
        var weights = new Vector<T>(size);
        double scale = Math.Sqrt(2.0 / _boilOptions.FeatureDimension);

        for (int i = 0; i < size; i++)
        {
            double value = (RandomGenerator.NextDouble() * 2 - 1) * scale;
            weights[i] = NumOps.FromDouble(value);
        }

        return weights;
    }

    /// <summary>
    /// Initializes head bias to zeros.
    /// </summary>
    private Vector<T> InitializeHeadBias()
    {
        var bias = new Vector<T>(_boilOptions.NumClasses);
        for (int i = 0; i < bias.Length; i++)
        {
            bias[i] = NumOps.Zero;
        }
        return bias;
    }

    /// <summary>
    /// Initializes body parameters.
    /// </summary>
    private Vector<T> InitializeBodyParameters()
    {
        var totalParams = MetaModel.GetParameters();
        var bodyParams = new Vector<T>(_bodyParameterCount);

        double scale = Math.Sqrt(2.0 / _boilOptions.FeatureDimension);
        for (int i = 0; i < _bodyParameterCount; i++)
        {
            if (i < totalParams.Length)
            {
                bodyParams[i] = totalParams[i];
            }
            else
            {
                bodyParams[i] = NumOps.FromDouble((RandomGenerator.NextDouble() * 2 - 1) * scale);
            }
        }

        return bodyParams;
    }

    /// <summary>
    /// Clones body parameters from the current model.
    /// </summary>
    private Vector<T> CloneBodyParameters()
    {
        var totalParams = MetaModel.GetParameters();
        var bodyParams = new Vector<T>(_bodyParameterCount);

        int copyLen = Math.Min(_bodyParameterCount, totalParams.Length);
        for (int i = 0; i < copyLen; i++)
        {
            bodyParams[i] = totalParams[i];
        }

        return bodyParams;
    }

    /// <summary>
    /// Clones a vector.
    /// </summary>
    private Vector<T> CloneVector(Vector<T> source)
    {
        var clone = new Vector<T>(source.Length);
        for (int i = 0; i < source.Length; i++)
        {
            clone[i] = source[i];
        }
        return clone;
    }

    #endregion

    #region Forward Pass

    /// <summary>
    /// Performs forward pass with given body parameters and frozen head.
    /// </summary>
    private TOutput ForwardWithBody(TInput input, Vector<T> bodyParams)
    {
        // Apply body parameters to model
        var currentParams = MetaModel.GetParameters();
        var tempParams = new Vector<T>(currentParams.Length);

        // Copy body parameters
        int copyLen = Math.Min(bodyParams.Length, currentParams.Length);
        for (int i = 0; i < copyLen; i++)
        {
            tempParams[i] = bodyParams[i];
        }
        // Keep head parameters as is
        for (int i = copyLen; i < currentParams.Length; i++)
        {
            tempParams[i] = currentParams[i];
        }

        MetaModel.SetParameters(tempParams);

        // Extract features
        var features = MetaModel.Predict(input);
        var featureVec = ConvertToVector(features);

        if (featureVec == null)
        {
            featureVec = new Vector<T>(_boilOptions.FeatureDimension);
        }

        // Apply frozen head
        var logits = ComputeLogits(featureVec, _headWeights, _headBias);

        // Restore original parameters
        MetaModel.SetParameters(currentParams);

        return ConvertFromVector(logits);
    }

    /// <summary>
    /// Computes logits from features using head parameters.
    /// </summary>
    private Vector<T> ComputeLogits(Vector<T> features, Vector<T> headWeights, Vector<T>? headBias)
    {
        var logits = new Vector<T>(_boilOptions.NumClasses);
        int featureDim = Math.Min(features.Length, _boilOptions.FeatureDimension);

        for (int c = 0; c < _boilOptions.NumClasses; c++)
        {
            T sum = NumOps.Zero;
            for (int f = 0; f < featureDim; f++)
            {
                int weightIdx = c * _boilOptions.FeatureDimension + f;
                if (weightIdx < headWeights.Length)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(features[f], headWeights[weightIdx]));
                }
            }
            if (headBias != null && c < headBias.Length)
            {
                sum = NumOps.Add(sum, headBias[c]);
            }
            logits[c] = sum;
        }

        return logits;
    }

    /// <summary>
    /// Converts a vector to the output type.
    /// </summary>
    private TOutput ConvertFromVector(Vector<T> vector)
    {
        if (typeof(TOutput) == typeof(Vector<T>))
        {
            return (TOutput)(object)vector;
        }

        if (typeof(TOutput) == typeof(Tensor<T>))
        {
            return (TOutput)(object)Tensor<T>.FromVector(vector);
        }

        if (typeof(TOutput) == typeof(T[]))
        {
            return (TOutput)(object)vector.ToArray();
        }

        throw new InvalidOperationException(
            $"Cannot convert Vector<{typeof(T).Name}> to {typeof(TOutput).Name}. " +
            $"Supported types: Vector<T>, Tensor<T>, T[]");
    }

    #endregion

    #region Gradient Computation

    /// <summary>
    /// Computes gradients for body parameters only.
    /// </summary>
    private Vector<T> ComputeBodyGradients(TInput input, TOutput expectedOutput, Vector<T> bodyParams)
    {
        double epsilon = 1e-5;
        var gradients = new Vector<T>(bodyParams.Length);

        // Compute baseline loss
        var basePred = ForwardWithBody(input, bodyParams);
        T baseLoss = ComputeLossFromOutput(basePred, expectedOutput);

        // Compute gradients using finite differences (sample subset for efficiency)
        int sampleCount = Math.Min(100, bodyParams.Length);
        // Scale factor for unbiased gradient estimation when subsampling
        double scaleFactor = (double)bodyParams.Length / sampleCount;
        double fraction = _boilOptions.BodyAdaptationFraction;

        for (int s = 0; s < sampleCount; s++)
        {
            int i = (int)(s * bodyParams.Length / (double)sampleCount);

            // Skip if not in adaptation fraction
            double position = i / (double)bodyParams.Length;
            if (position > fraction)
            {
                continue;
            }

            // Perturb parameter
            T original = bodyParams[i];
            bodyParams[i] = NumOps.Add(original, NumOps.FromDouble(epsilon));

            var perturbedPred = ForwardWithBody(input, bodyParams);
            T perturbedLoss = ComputeLossFromOutput(perturbedPred, expectedOutput);

            double grad = (NumOps.ToDouble(perturbedLoss) - NumOps.ToDouble(baseLoss)) / epsilon;
            gradients[i] = NumOps.FromDouble(grad * scaleFactor);

            // Restore
            bodyParams[i] = original;
        }

        return gradients;
    }

    /// <summary>
    /// Computes meta-gradients for body and head.
    /// </summary>
    private (Vector<T> bodyGrads, Vector<T> headGrads) ComputeMetaGradients(
        IMetaLearningTask<T, TInput, TOutput> task,
        Vector<T> adaptedBodyParams,
        T queryLoss)
    {
        if (_boilOptions.UseFirstOrder)
        {
            return ComputeFirstOrderMetaGradients(task, adaptedBodyParams);
        }
        return ComputeSecondOrderMetaGradients(task, adaptedBodyParams, queryLoss);
    }

    /// <summary>
    /// Computes second-order meta-gradients by differentiating through the inner adaptation loop.
    /// </summary>
    private (Vector<T> bodyGrads, Vector<T> headGrads) ComputeSecondOrderMetaGradients(
        IMetaLearningTask<T, TInput, TOutput> task,
        Vector<T> adaptedBodyParams,
        T queryLoss)
    {
        double epsilon = 1e-5;
        var initialBodyParams = CloneBodyParameters();
        var bodyGrads = new Vector<T>(initialBodyParams.Length);

        // Compute second-order gradients by measuring how perturbations to initial params
        // affect the final query loss after adaptation
        int sampleCount = Math.Min(100, initialBodyParams.Length);
        double scaleFactor = (double)initialBodyParams.Length / sampleCount;

        for (int i = 0; i < sampleCount; i++)
        {
            int idx = sampleCount > 0 ? (i * initialBodyParams.Length / sampleCount) : i;

            // Perturb initial body parameter
            T original = initialBodyParams[idx];
            initialBodyParams[idx] = NumOps.Add(original, NumOps.FromDouble(epsilon));

            // Re-run adaptation with perturbed initial params
            var perturbedAdapted = new Vector<T>(initialBodyParams.Length);
            for (int j = 0; j < initialBodyParams.Length; j++)
            {
                perturbedAdapted[j] = initialBodyParams[j];
            }

            for (int step = 0; step < _boilOptions.AdaptationSteps; step++)
            {
                var grads = ComputeBodyGradients(task.SupportInput, task.SupportOutput, perturbedAdapted);
                perturbedAdapted = ApplyGradients(perturbedAdapted, grads, _boilOptions.InnerLearningRate);
            }

            // Compute query loss with perturbed adapted params
            var perturbedPred = ForwardWithBody(task.QueryInput, perturbedAdapted);
            T perturbedLoss = ComputeLossFromOutput(perturbedPred, task.QueryOutput);

            // Gradient w.r.t. initial parameter = (perturbedLoss - queryLoss) / epsilon
            double grad = (NumOps.ToDouble(perturbedLoss) - NumOps.ToDouble(queryLoss)) / epsilon;
            bodyGrads[idx] = NumOps.FromDouble(grad * scaleFactor);

            // Restore
            initialBodyParams[idx] = original;
        }

        // Head gradients are computed at the adapted point (same as first-order)
        var headGrads = ComputeHeadGradients(task.QueryInput, task.QueryOutput, adaptedBodyParams);

        return (bodyGrads, headGrads);
    }

    /// <summary>
    /// Computes first-order meta-gradients.
    /// </summary>
    private (Vector<T> bodyGrads, Vector<T> headGrads) ComputeFirstOrderMetaGradients(
        IMetaLearningTask<T, TInput, TOutput> task,
        Vector<T> adaptedBodyParams)
    {
        // Body gradients
        var bodyGrads = ComputeBodyGradients(task.QueryInput, task.QueryOutput, adaptedBodyParams);

        // Head gradients (compute on query set)
        var headGrads = ComputeHeadGradients(task.QueryInput, task.QueryOutput, adaptedBodyParams);

        return (bodyGrads, headGrads);
    }

    /// <summary>
    /// Computes gradients for head parameters.
    /// </summary>
    private Vector<T> ComputeHeadGradients(TInput input, TOutput expectedOutput, Vector<T> bodyParams)
    {
        double epsilon = 1e-5;
        var gradients = new Vector<T>(_headWeights.Length);

        // Compute baseline loss
        var basePred = ForwardWithBody(input, bodyParams);
        T baseLoss = ComputeLossFromOutput(basePred, expectedOutput);

        // Compute gradients using finite differences
        for (int i = 0; i < _headWeights.Length; i++)
        {
            T original = _headWeights[i];
            _headWeights[i] = NumOps.Add(original, NumOps.FromDouble(epsilon));

            var perturbedPred = ForwardWithBody(input, bodyParams);
            T perturbedLoss = ComputeLossFromOutput(perturbedPred, expectedOutput);

            double grad = (NumOps.ToDouble(perturbedLoss) - NumOps.ToDouble(baseLoss)) / epsilon;
            gradients[i] = NumOps.FromDouble(grad);

            _headWeights[i] = original;
        }

        return gradients;
    }

    /// <summary>
    /// Updates body initialization using gradients.
    /// </summary>
    private void UpdateBodyInitialization(Vector<T> gradients)
    {
        var currentParams = MetaModel.GetParameters();
        var updatedParams = new Vector<T>(currentParams.Length);

        // Update body parameters
        int bodyLen = Math.Min(_bodyParameterCount, currentParams.Length);
        int gradLen = Math.Min(gradients.Length, bodyLen);

        for (int i = 0; i < gradLen; i++)
        {
            T update = NumOps.Multiply(NumOps.FromDouble(_boilOptions.OuterLearningRate), gradients[i]);
            updatedParams[i] = NumOps.Subtract(currentParams[i], update);
        }

        // Keep head parameters in model unchanged
        for (int i = bodyLen; i < currentParams.Length; i++)
        {
            updatedParams[i] = currentParams[i];
        }

        MetaModel.SetParameters(updatedParams);
    }

    #endregion

    #region Utility Methods

    /// <summary>
    /// Applies layer-wise learning rate scaling to gradients.
    /// </summary>
    private void ApplyLayerwiseLearningRates(Vector<T> gradients)
    {
        // Apply smaller learning rates to earlier layers
        double earlyMultiplier = _boilOptions.EarlyLayerLrMultiplier;

        // Assume earlier layers are in the first half of parameters
        int earlyLayerEnd = gradients.Length / 2;

        for (int i = 0; i < earlyLayerEnd; i++)
        {
            gradients[i] = NumOps.Multiply(gradients[i], NumOps.FromDouble(earlyMultiplier));
        }
    }

    /// <summary>
    /// Computes L2 penalty for body parameters.
    /// </summary>
    private T ComputeL2Penalty(Vector<T> bodyParams)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < bodyParams.Length; i++)
        {
            sum = NumOps.Add(sum, NumOps.Multiply(bodyParams[i], bodyParams[i]));
        }
        return NumOps.Multiply(NumOps.FromDouble(0.5), sum);
    }

    #endregion
}
