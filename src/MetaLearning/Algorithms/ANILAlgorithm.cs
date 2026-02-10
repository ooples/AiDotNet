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
/// Implementation of Almost No Inner Loop (ANIL) meta-learning algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// ANIL is a simplified version of MAML that only adapts the classification head
/// during inner-loop adaptation while keeping the feature extractor (body) frozen.
/// This significantly reduces computation while often maintaining competitive performance.
/// </para>
/// <para>
/// <b>Key Insight:</b> Most of the "learning to learn" ability in MAML comes from
/// learning a good feature representation, not from adapting the entire network.
/// By only adapting the final classification layer, ANIL achieves:
/// </para>
/// <list type="bullet">
/// <item>Much faster adaptation (fewer parameters to update)</item>
/// <item>Lower memory usage (no gradients stored for the body)</item>
/// <item>Comparable performance to full MAML in many scenarios</item>
/// </list>
/// <para>
/// <b>For Beginners:</b> Think of a neural network as having two parts:
/// </para>
/// <para>
/// 1. <b>Body (Feature Extractor):</b> Like learning to see and understand images
/// 2. <b>Head (Classifier):</b> Like learning which button to press for each category
/// </para>
/// <para>
/// ANIL says: "The 'seeing' part is general enough - we just need to learn
/// which button to press for each new task!" So it only updates the button-pressing
/// part (head) and keeps the seeing part (body) fixed during adaptation.
/// </para>
/// <para>
/// <b>Algorithm (MAML-style with head-only adaptation):</b>
/// <code>
/// For each task batch:
///   For each task:
///     1. Clone the classification head parameters
///     2. Freeze body, only compute gradients for head
///     3. For each adaptation step:
///        a. Forward pass through body (frozen) + head
///        b. Compute loss on support set
///        c. Update ONLY head parameters
///     4. Evaluate adapted head on query set
///   Meta-update: body (outer loop) + head initialization (inner loop starting point)
/// </code>
/// </para>
/// <para>
/// Reference: Raghu, A., Raghu, M., Bengio, S., &amp; Vinyals, O. (2020).
/// Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML.
/// </para>
/// </remarks>
public class ANILAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly ANILOptions<T, TInput, TOutput> _anilOptions;

    // Head parameters (adapted per-task)
    private Vector<T> _headWeights;
    private Vector<T>? _headBias;

    // Body parameters (frozen during inner loop, updated in outer loop)
    private int _bodyParameterCount;
    private int _headParameterCount;

    /// <summary>
    /// Initializes a new instance of the ANILAlgorithm class.
    /// </summary>
    /// <param name="options">ANIL configuration options containing the model and all hyperparameters.</param>
    /// <exception cref="ArgumentNullException">Thrown when options is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when required components are not set in options.</exception>
    /// <example>
    /// <code>
    /// // Create ANIL with minimal configuration
    /// var options = new ANILOptions&lt;double, Tensor, Tensor&gt;(myNeuralNetwork);
    /// var anil = new ANILAlgorithm&lt;double, Tensor, Tensor&gt;(options);
    ///
    /// // Create ANIL with custom configuration
    /// var options = new ANILOptions&lt;double, Tensor, Tensor&gt;(myNeuralNetwork)
    /// {
    ///     AdaptationSteps = 5,
    ///     InnerLearningRate = 0.01,
    ///     NumClasses = 5,
    ///     FeatureDimension = 512
    /// };
    /// var anil = new ANILAlgorithm&lt;double, Tensor, Tensor&gt;(options);
    /// </code>
    /// </example>
    public ANILAlgorithm(ANILOptions<T, TInput, TOutput> options)
        : base(
            options?.MetaModel ?? throw new ArgumentNullException(nameof(options), "MetaModel must be set in options."),
            options.LossFunction ?? options.MetaModel.DefaultLossFunction,
            options,
            options.DataLoader,
            options.MetaOptimizer,
            options.InnerOptimizer)
    {
        _anilOptions = options;

        // Initialize head weights - must be done before InitializeHeadParameters
        // to satisfy the compiler's non-nullable field requirement
        _headWeights = new Vector<T>(options.FeatureDimension * options.NumClasses);

        // Initialize head parameters with proper values
        InitializeHeadParameters();
    }

    /// <summary>
    /// Gets the algorithm type identifier for this meta-learner.
    /// </summary>
    /// <value>Returns <see cref="MetaLearningAlgorithmType.ANIL"/>.</value>
    /// <remarks>
    /// <para>
    /// This property identifies the algorithm as ANIL (Almost No Inner Loop),
    /// a simplified MAML variant that only adapts the classification head.
    /// </para>
    /// </remarks>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.ANIL;

    /// <summary>
    /// Performs one meta-training step using ANIL's head-only adaptation approach.
    /// </summary>
    /// <param name="taskBatch">A batch of tasks to meta-train on, each containing support and query sets.</param>
    /// <returns>The average meta-loss across all tasks in the batch (evaluated on query sets).</returns>
    /// <exception cref="ArgumentException">Thrown when the task batch is null or empty.</exception>
    /// <exception cref="InvalidOperationException">Thrown when meta-gradient computation fails.</exception>
    /// <remarks>
    /// <para>
    /// ANIL meta-training is a simplified version of MAML:
    /// </para>
    /// <para>
    /// <b>ANIL Inner Loop (per task):</b>
    /// 1. Clone the head parameters from the meta-learned initialization
    /// 2. Keep body parameters frozen (no gradients computed for body)
    /// 3. Perform gradient descent on head parameters using support set
    /// 4. Evaluate adapted model on query set
    /// </para>
    /// <para>
    /// <b>ANIL Outer Loop:</b>
    /// 1. Accumulate gradients from all tasks' query losses
    /// 2. Update body parameters to improve feature extraction
    /// 3. Update head initialization to provide better starting point
    /// </para>
    /// <para>
    /// <b>Key Differences from MAML:</b>
    /// - Only head parameters are updated in inner loop
    /// - Body parameters are only updated in outer loop
    /// - First-order approximation is typically used
    /// - Much faster per-iteration (fewer parameters to track)
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> ANIL learns two things:
    /// 1. A good feature extractor that works for all tasks (updated in outer loop)
    /// 2. A good starting point for the classifier head (adapted per task)
    ///
    /// During adaptation, only the classifier changes - the "vision system" stays fixed.
    /// This is much faster because the classifier is much smaller than the full network.
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
            // Clone head parameters for this task
            var taskHeadWeights = CloneVector(_headWeights);
            var taskHeadBias = _headBias != null ? CloneVector(_headBias) : null;

            // Inner loop: adapt head on support set
            for (int step = 0; step < _anilOptions.AdaptationSteps; step++)
            {
                // Forward pass with frozen body + current head
                var supportPredictions = ForwardWithHead(task.SupportInput, taskHeadWeights, taskHeadBias);

                // Compute loss on support set
                T supportLoss = ComputeLossFromOutput(supportPredictions, task.SupportOutput);

                // Compute gradients for HEAD ONLY
                var headGradients = ComputeHeadGradients(
                    task.SupportInput, task.SupportOutput, taskHeadWeights, taskHeadBias);

                // Update head parameters
                taskHeadWeights = ApplyGradients(
                    taskHeadWeights, headGradients.weightGradients, _anilOptions.InnerLearningRate);

                if (taskHeadBias != null && headGradients.biasGradients != null)
                {
                    taskHeadBias = ApplyGradients(
                        taskHeadBias, headGradients.biasGradients, _anilOptions.InnerLearningRate);
                }
            }

            // Evaluate on query set with adapted head
            var queryPredictions = ForwardWithHead(task.QueryInput, taskHeadWeights, taskHeadBias);
            T taskLoss = ComputeLossFromOutput(queryPredictions, task.QueryOutput);
            totalLoss = NumOps.Add(totalLoss, taskLoss);

            // Compute meta-gradients
            var (bodyGradients, headMetaGradients) = ComputeMetaGradients(
                task, taskHeadWeights, taskHeadBias, taskLoss);

            // Accumulate gradients
            if (accumulatedBodyGradients == null)
            {
                accumulatedBodyGradients = bodyGradients;
                accumulatedHeadGradients = headMetaGradients;
            }
            else
            {
                for (int i = 0; i < accumulatedBodyGradients.Length; i++)
                {
                    accumulatedBodyGradients[i] = NumOps.Add(accumulatedBodyGradients[i], bodyGradients[i]);
                }
                if (accumulatedHeadGradients != null)
                {
                    for (int i = 0; i < accumulatedHeadGradients.Length; i++)
                    {
                        accumulatedHeadGradients[i] = NumOps.Add(accumulatedHeadGradients[i], headMetaGradients[i]);
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
        if (_anilOptions.GradientClipThreshold.HasValue && _anilOptions.GradientClipThreshold.Value > 0)
        {
            accumulatedBodyGradients = ClipGradients(accumulatedBodyGradients, _anilOptions.GradientClipThreshold.Value);
            accumulatedHeadGradients = ClipGradients(accumulatedHeadGradients, _anilOptions.GradientClipThreshold.Value);
        }

        // Update body parameters (outer loop)
        UpdateBodyParameters(accumulatedBodyGradients);

        // Update head initialization (outer loop)
        _headWeights = ApplyGradients(_headWeights, accumulatedHeadGradients, _anilOptions.OuterLearningRate);

        return NumOps.Divide(totalLoss, batchSizeT);
    }

    /// <summary>
    /// Adapts the meta-learned model to a new task by only updating the classification head.
    /// </summary>
    /// <param name="task">The new task containing support set examples for adaptation.</param>
    /// <returns>A new model instance that has been adapted to the given task with an updated head.</returns>
    /// <exception cref="ArgumentNullException">Thrown when task is null.</exception>
    /// <remarks>
    /// <para>
    /// ANIL adaptation is significantly faster than full MAML adaptation because
    /// only the classification head parameters are updated.
    /// </para>
    /// <para>
    /// <b>Adaptation Process:</b>
    /// 1. Clone the meta-learned head parameters
    /// 2. Optionally reinitialize head (if configured)
    /// 3. For each adaptation step:
    ///    a. Forward pass: frozen body + current head
    ///    b. Compute loss on support set
    ///    c. Update head parameters with gradient descent
    /// 4. Return model with frozen body + adapted head
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> When you give ANIL a new task:
    /// 1. It keeps its "vision" (feature extractor) exactly the same
    /// 2. It only retrains the "decision maker" (classifier head)
    /// 3. This is like teaching someone who already knows how to see,
    ///    just teaching them what labels to assign to things
    /// </para>
    /// <para>
    /// <b>Speed Advantage:</b> If your network has 1 million parameters and the head
    /// has only 5000, ANIL updates 200x fewer parameters per adaptation step!
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

        // Clone head parameters for adaptation
        var adaptedHeadWeights = _anilOptions.ReinitializeHead
            ? InitializeHeadWeights()
            : CloneVector(_headWeights);
        var adaptedHeadBias = _headBias != null && !_anilOptions.ReinitializeHead
            ? CloneVector(_headBias)
            : (_anilOptions.UseHeadBias ? InitializeHeadBias() : null);

        // Inner loop: adapt head on support set
        for (int step = 0; step < _anilOptions.AdaptationSteps; step++)
        {
            // Forward pass with frozen body + current head
            var supportPredictions = ForwardWithHead(task.SupportInput, adaptedHeadWeights, adaptedHeadBias);

            // Compute loss on support set
            T supportLoss = ComputeLossFromOutput(supportPredictions, task.SupportOutput);

            // Add L2 regularization if configured
            if (_anilOptions.HeadL2Regularization > 0)
            {
                T l2Penalty = ComputeL2Penalty(adaptedHeadWeights);
                supportLoss = NumOps.Add(supportLoss,
                    NumOps.Multiply(NumOps.FromDouble(_anilOptions.HeadL2Regularization), l2Penalty));
            }

            // Compute gradients for HEAD ONLY
            var headGradients = ComputeHeadGradients(
                task.SupportInput, task.SupportOutput, adaptedHeadWeights, adaptedHeadBias);

            // Update head parameters
            adaptedHeadWeights = ApplyGradients(
                adaptedHeadWeights, headGradients.weightGradients, _anilOptions.InnerLearningRate);

            if (adaptedHeadBias != null && headGradients.biasGradients != null)
            {
                adaptedHeadBias = ApplyGradients(
                    adaptedHeadBias, headGradients.biasGradients, _anilOptions.InnerLearningRate);
            }
        }

        // Create ANIL model with adapted head
        return new ANILModel<T, TInput, TOutput>(
            adaptedModel,
            adaptedHeadWeights,
            adaptedHeadBias,
            _anilOptions);
    }

    #region Head Parameter Management

    /// <summary>
    /// Initializes the classification head parameters.
    /// </summary>
    private void InitializeHeadParameters()
    {
        // Compute body and head parameter counts
        var totalParams = MetaModel.GetParameters();

        // Estimate head size based on options
        _headParameterCount = _anilOptions.FeatureDimension * _anilOptions.NumClasses;
        if (_anilOptions.UseHeadBias)
        {
            _headParameterCount += _anilOptions.NumClasses;
        }

        _bodyParameterCount = totalParams.Length - _headParameterCount;
        if (_bodyParameterCount < 0)
        {
            // If model is smaller than expected head, adjust
            _bodyParameterCount = (int)(totalParams.Length * 0.9);
            _headParameterCount = totalParams.Length - _bodyParameterCount;
        }

        // Initialize head weights
        _headWeights = InitializeHeadWeights();

        if (_anilOptions.UseHeadBias)
        {
            _headBias = InitializeHeadBias();
        }
    }

    /// <summary>
    /// Initializes head weights using Xavier/He initialization.
    /// </summary>
    private Vector<T> InitializeHeadWeights()
    {
        int size = _anilOptions.FeatureDimension * _anilOptions.NumClasses;
        var weights = new Vector<T>(size);

        // He initialization (suitable for ReLU activations in body)
        double scale = Math.Sqrt(2.0 / _anilOptions.FeatureDimension);

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
        var bias = new Vector<T>(_anilOptions.NumClasses);
        // Initialize to zeros
        for (int i = 0; i < bias.Length; i++)
        {
            bias[i] = NumOps.Zero;
        }
        return bias;
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
    /// Performs forward pass with custom head parameters.
    /// </summary>
    /// <param name="input">The input data.</param>
    /// <param name="headWeights">The head weight parameters.</param>
    /// <param name="headBias">The head bias parameters (optional).</param>
    /// <returns>The model output.</returns>
    private TOutput ForwardWithHead(TInput input, Vector<T> headWeights, Vector<T>? headBias)
    {
        // Get features from body (frozen)
        var features = ExtractFeatures(input);

        // Apply head (linear layer + softmax for classification)
        var logits = ComputeLogits(features, headWeights, headBias);

        return ConvertFromVector(logits);
    }

    /// <summary>
    /// Extracts features using the frozen body of the model.
    /// </summary>
    private Vector<T> ExtractFeatures(TInput input)
    {
        // Use the meta model to extract features
        // In a full implementation, this would use a feature extraction layer
        var output = MetaModel.Predict(input);
        var features = ConvertToVector(output);

        if (features == null)
        {
            // Return a feature vector based on expected dimension
            features = new Vector<T>(_anilOptions.FeatureDimension);
        }

        return features;
    }

    /// <summary>
    /// Computes logits from features using the head parameters.
    /// </summary>
    private Vector<T> ComputeLogits(Vector<T> features, Vector<T> headWeights, Vector<T>? headBias)
    {
        var logits = new Vector<T>(_anilOptions.NumClasses);

        // Linear transformation: logits = features * W + b
        int featureDim = Math.Min(features.Length, _anilOptions.FeatureDimension);

        for (int c = 0; c < _anilOptions.NumClasses; c++)
        {
            T sum = NumOps.Zero;

            for (int f = 0; f < featureDim; f++)
            {
                int weightIdx = c * _anilOptions.FeatureDimension + f;
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

        // Handle Tensor<T>
        if (typeof(TOutput) == typeof(Tensor<T>))
        {
            return (TOutput)(object)Tensor<T>.FromVector(vector);
        }

        // Handle T[]
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
    /// Computes gradients for head parameters only.
    /// </summary>
    private (Vector<T> weightGradients, Vector<T>? biasGradients) ComputeHeadGradients(
        TInput input, TOutput expectedOutput, Vector<T> headWeights, Vector<T>? headBias)
    {
        // Use finite differences for gradient computation
        double epsilon = 1e-5;

        // Compute baseline loss
        var basePredictions = ForwardWithHead(input, headWeights, headBias);
        T baseLoss = ComputeLossFromOutput(basePredictions, expectedOutput);

        // Compute weight gradients
        var weightGradients = new Vector<T>(headWeights.Length);
        for (int i = 0; i < headWeights.Length; i++)
        {
            // Perturb weight
            T original = headWeights[i];
            headWeights[i] = NumOps.Add(original, NumOps.FromDouble(epsilon));

            var perturbedPred = ForwardWithHead(input, headWeights, headBias);
            T perturbedLoss = ComputeLossFromOutput(perturbedPred, expectedOutput);

            double grad = (NumOps.ToDouble(perturbedLoss) - NumOps.ToDouble(baseLoss)) / epsilon;
            weightGradients[i] = NumOps.FromDouble(grad);

            // Restore
            headWeights[i] = original;
        }

        // Compute bias gradients if applicable
        Vector<T>? biasGradients = null;
        if (headBias != null)
        {
            biasGradients = new Vector<T>(headBias.Length);
            for (int i = 0; i < headBias.Length; i++)
            {
                T original = headBias[i];
                headBias[i] = NumOps.Add(original, NumOps.FromDouble(epsilon));

                var perturbedPred = ForwardWithHead(input, headWeights, headBias);
                T perturbedLoss = ComputeLossFromOutput(perturbedPred, expectedOutput);

                double grad = (NumOps.ToDouble(perturbedLoss) - NumOps.ToDouble(baseLoss)) / epsilon;
                biasGradients[i] = NumOps.FromDouble(grad);

                headBias[i] = original;
            }
        }

        return (weightGradients, biasGradients);
    }

    /// <summary>
    /// Computes meta-gradients for body and head initialization.
    /// </summary>
    private (Vector<T> bodyGradients, Vector<T> headGradients) ComputeMetaGradients(
        IMetaLearningTask<T, TInput, TOutput> task,
        Vector<T> adaptedHeadWeights,
        Vector<T>? adaptedHeadBias,
        T queryLoss)
    {
        if (_anilOptions.UseFirstOrder)
        {
            // First-order approximation: use query gradients directly
            return ComputeFirstOrderMetaGradients(task, adaptedHeadWeights, adaptedHeadBias);
        }

        // Second-order would require differentiating through the adaptation process
        // For simplicity, we use first-order here
        return ComputeFirstOrderMetaGradients(task, adaptedHeadWeights, adaptedHeadBias);
    }

    /// <summary>
    /// Computes first-order meta-gradients.
    /// </summary>
    private (Vector<T> bodyGradients, Vector<T> headGradients) ComputeFirstOrderMetaGradients(
        IMetaLearningTask<T, TInput, TOutput> task,
        Vector<T> adaptedHeadWeights,
        Vector<T>? adaptedHeadBias)
    {
        // Compute body gradients using the adapted head
        var bodyGradients = ComputeBodyGradients(task.QueryInput, task.QueryOutput, adaptedHeadWeights, adaptedHeadBias);

        // Compute head initialization gradients
        var (headGradients, _) = ComputeHeadGradients(task.QueryInput, task.QueryOutput, adaptedHeadWeights, adaptedHeadBias);

        return (bodyGradients, headGradients);
    }

    /// <summary>
    /// Computes gradients for body parameters using the adapted head.
    /// </summary>
    private Vector<T> ComputeBodyGradients(TInput input, TOutput expectedOutput, Vector<T> adaptedHeadWeights, Vector<T>? adaptedHeadBias)
    {
        // Set the adapted head parameters in the model before computing body gradients
        // This ensures body gradients are computed with respect to the adapted classifier head
        var currentParams = MetaModel.GetParameters();
        var paramsWithAdaptedHead = new Vector<T>(currentParams.Length);

        // Copy body parameters as-is
        int bodyLen = Math.Min(_bodyParameterCount, currentParams.Length);
        for (int i = 0; i < bodyLen; i++)
        {
            paramsWithAdaptedHead[i] = currentParams[i];
        }

        // Set adapted head weights
        int headWeightsLen = Math.Min(adaptedHeadWeights.Length, currentParams.Length - bodyLen);
        for (int i = 0; i < headWeightsLen; i++)
        {
            paramsWithAdaptedHead[bodyLen + i] = adaptedHeadWeights[i];
        }

        // Set adapted head bias if present
        if (adaptedHeadBias != null)
        {
            int biasOffset = bodyLen + headWeightsLen;
            int biasLen = Math.Min(adaptedHeadBias.Length, currentParams.Length - biasOffset);
            for (int i = 0; i < biasLen; i++)
            {
                paramsWithAdaptedHead[biasOffset + i] = adaptedHeadBias[i];
            }
        }

        // Temporarily set adapted head parameters
        MetaModel.SetParameters(paramsWithAdaptedHead);

        // Use the base model's gradient computation with adapted head
        var fullGradients = ComputeGradients(MetaModel, input, expectedOutput);

        // Restore original parameters
        MetaModel.SetParameters(currentParams);

        // Extract only body gradients (first _bodyParameterCount parameters)
        var bodyGradients = new Vector<T>(_bodyParameterCount);
        int copyLen = Math.Min(_bodyParameterCount, fullGradients.Length);
        for (int i = 0; i < copyLen; i++)
        {
            bodyGradients[i] = fullGradients[i];
        }

        return bodyGradients;
    }

    /// <summary>
    /// Updates body parameters using gradients.
    /// </summary>
    private void UpdateBodyParameters(Vector<T> gradients)
    {
        var currentParams = MetaModel.GetParameters();
        var updatedParams = new Vector<T>(currentParams.Length);

        // Update body parameters (first _bodyParameterCount)
        int bodyLen = Math.Min(_bodyParameterCount, currentParams.Length);
        int gradLen = Math.Min(gradients.Length, bodyLen);

        for (int i = 0; i < gradLen; i++)
        {
            T update = NumOps.Multiply(NumOps.FromDouble(_anilOptions.OuterLearningRate), gradients[i]);
            updatedParams[i] = NumOps.Subtract(currentParams[i], update);
        }

        // Keep head parameters unchanged in the main model
        for (int i = bodyLen; i < currentParams.Length; i++)
        {
            updatedParams[i] = currentParams[i];
        }

        MetaModel.SetParameters(updatedParams);
    }

    #endregion

    #region Utility Methods

    /// <summary>
    /// Computes L2 penalty for head weights.
    /// </summary>
    private T ComputeL2Penalty(Vector<T> weights)
    {
        T sum = NumOps.Zero;
        for (int i = 0; i < weights.Length; i++)
        {
            sum = NumOps.Add(sum, NumOps.Multiply(weights[i], weights[i]));
        }
        return NumOps.Multiply(NumOps.FromDouble(0.5), sum);
    }

    #endregion
}
