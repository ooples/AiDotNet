using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of Conditional Neural Adaptive Processes (CNAP) for meta-learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// CNAP extends Neural Processes by conditioning on task-specific context points
/// and learning to produce fast adaptation weights for each task. Unlike MAML,
/// which adapts through gradient descent, CNAP learns to directly generate
/// task-specific weight modifications from context examples.
/// </para>
/// <para>
/// <b>Key Innovation:</b> Instead of computing gradients on support sets, CNAP:
/// 1. Encodes support set examples into a task representation
/// 2. Uses an adaptation network to generate task-specific fast weights
/// 3. Applies these fast weights to modify the base model for that task
/// </para>
/// <para>
/// <b>For Beginners:</b> CNAP is like having a teacher who can instantly understand
/// a new subject from a few examples:
/// </para>
/// <para>
/// - MAML: Learns how to learn quickly (by finding a good starting point)
/// - CNAP: Learns to directly generate solutions (by understanding the task)
///
/// Imagine showing someone a few examples of a new font. CNAP doesn't need to
/// practice writing in that font (like MAML would). Instead, it understands
/// the pattern from examples and directly modifies how it writes.
/// </para>
/// <para>
/// <b>Architecture:</b>
/// - <b>Encoder:</b> Processes each context (input, output) pair into embeddings
/// - <b>Aggregator:</b> Combines embeddings into a single task representation
/// - <b>Adaptation Network:</b> Generates fast weights from task representation
/// - <b>Base Model:</b> Modified by fast weights to perform the task
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <code>
/// For each task batch:
///   For each task:
///     1. Encode context points: z_i = encoder(x_i, y_i)
///     2. Aggregate to task representation: z = aggregate(z_1, ..., z_k)
///     3. Generate fast weights: α = adaptation_network(z)
///     4. Apply fast weights to base model: model' = apply_fast_weights(model, α)
///     5. Evaluate on query set: loss = query_loss(model', query_data)
///   Meta-update all networks based on query losses
/// </code>
/// </para>
/// <para>
/// Reference: Requeima, J., Gordon, J., Bronskill, J., Nowozin, S., &amp; Turner, R. E. (2019).
/// Fast and Flexible Multi-Task Classification Using Conditional Neural Adaptive Processes.
/// </para>
/// </remarks>
public class CNAPAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly CNAPOptions<T, TInput, TOutput> _cnapOptions;

    // Task representation state
    private Vector<T>? _currentTaskRepresentation;
    private Vector<T>? _currentFastWeights;

    // Adaptation network parameters (learned during meta-training)
    private Vector<T> _encoderWeights;
    private Vector<T> _adaptationNetworkWeights;

    /// <summary>
    /// Initializes a new instance of the CNAPAlgorithm class.
    /// </summary>
    /// <param name="options">CNAP configuration options containing the model and all hyperparameters.</param>
    /// <exception cref="ArgumentNullException">Thrown when options is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when required components are not set in options.</exception>
    /// <example>
    /// <code>
    /// // Create CNAP with minimal configuration
    /// var options = new CNAPOptions&lt;double, Tensor, Tensor&gt;(myNeuralNetwork);
    /// var cnap = new CNAPAlgorithm&lt;double, Tensor, Tensor&gt;(options);
    ///
    /// // Create CNAP with custom configuration
    /// var options = new CNAPOptions&lt;double, Tensor, Tensor&gt;(myNeuralNetwork)
    /// {
    ///     RepresentationDimension = 512,
    ///     UseAttention = true,
    ///     NumAttentionHeads = 8,
    ///     FastWeightMode = FastWeightApplicationMode.FiLM
    /// };
    /// var cnap = new CNAPAlgorithm&lt;double, Tensor, Tensor&gt;(options);
    /// </code>
    /// </example>
    public CNAPAlgorithm(CNAPOptions<T, TInput, TOutput> options)
        : base(
            options?.MetaModel ?? throw new ArgumentNullException(nameof(options), "MetaModel must be set in options."),
            options.LossFunction ?? options.MetaModel.DefaultLossFunction,
            options,
            options.DataLoader,
            options.MetaOptimizer,
            options.InnerOptimizer)
    {
        _cnapOptions = options;

        // Initialize encoder and adaptation network weights
        // Encoder: maps (input, output) pairs to representation dimension
        // Simple MLP-based encoder for now
        int encoderInputDim = EstimateInputOutputDimension();
        int encoderSize = encoderInputDim * _cnapOptions.HiddenDimension +
                         _cnapOptions.HiddenDimension * _cnapOptions.RepresentationDimension;
        _encoderWeights = InitializeWeights(encoderSize);

        // Adaptation network: maps representation to fast weights
        int modelParamCount = MetaModel.GetParameters().Length;
        int adaptationSize = _cnapOptions.RepresentationDimension * _cnapOptions.HiddenDimension +
                            _cnapOptions.HiddenDimension * modelParamCount;
        _adaptationNetworkWeights = InitializeWeights(adaptationSize);
    }

    /// <summary>
    /// Gets the algorithm type identifier for this meta-learner.
    /// </summary>
    /// <value>Returns <see cref="MetaLearningAlgorithmType.CNAP"/>.</value>
    /// <remarks>
    /// <para>
    /// This property identifies the algorithm as CNAP (Conditional Neural Adaptive Processes),
    /// a feed-forward meta-learning algorithm that generates task-specific weights
    /// from context examples without gradient-based adaptation.
    /// </para>
    /// </remarks>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.CNAP;

    /// <summary>
    /// Performs one meta-training step using CNAP's feed-forward adaptation approach.
    /// </summary>
    /// <param name="taskBatch">A batch of tasks to meta-train on, each containing support and query sets.</param>
    /// <returns>The average meta-loss across all tasks in the batch (evaluated on query sets).</returns>
    /// <exception cref="ArgumentException">Thrown when the task batch is null or empty.</exception>
    /// <exception cref="InvalidOperationException">Thrown when meta-gradient computation fails.</exception>
    /// <remarks>
    /// <para>
    /// CNAP meta-training differs fundamentally from gradient-based methods like MAML:
    /// </para>
    /// <para>
    /// <b>CNAP Approach (Feed-Forward):</b>
    /// 1. Encode context points from support set into embeddings
    /// 2. Aggregate embeddings into a single task representation
    /// 3. Generate fast weights directly from the task representation
    /// 4. Apply fast weights to modify base model
    /// 5. Evaluate on query set and compute loss
    /// 6. Update encoder and adaptation networks based on query loss
    /// </para>
    /// <para>
    /// <b>Key Differences from MAML:</b>
    /// - No gradient steps during inner loop (feed-forward only)
    /// - Fast weights are generated, not computed through gradients
    /// - More efficient at test time (no gradient computation needed)
    /// - Can be more sample-efficient for similar task distributions
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> While MAML learns "how to learn quickly," CNAP learns
    /// "how to directly understand tasks." CNAP looks at the support set examples
    /// and generates a task-specific modification in one shot, without iterating.
    /// This is faster at test time but requires the task distribution to be
    /// well-structured enough for the adaptation network to learn patterns.
    /// </para>
    /// </remarks>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null || taskBatch.BatchSize == 0)
        {
            throw new ArgumentException("Task batch cannot be null or empty.", nameof(taskBatch));
        }

        // Accumulate gradients for encoder and adaptation network
        Vector<T>? accumulatedEncoderGradients = null;
        Vector<T>? accumulatedAdaptationGradients = null;
        T totalLoss = NumOps.Zero;

        foreach (var task in taskBatch.Tasks)
        {
            // Step 1: Encode support set into task representation
            var taskRepresentation = EncodeTask(task.SupportInput, task.SupportOutput);

            // Step 2: Generate fast weights from task representation
            var fastWeights = GenerateFastWeights(taskRepresentation);

            // Step 3: Clone base model and apply fast weights
            var taskModel = CloneModel();
            ApplyFastWeights(taskModel, fastWeights);

            // Step 4: Evaluate on query set
            var queryPredictions = taskModel.Predict(task.QueryInput);
            T taskLoss = ComputeLossFromOutput(queryPredictions, task.QueryOutput);
            totalLoss = NumOps.Add(totalLoss, taskLoss);

            // Step 5: Compute gradients for encoder and adaptation network
            var (encoderGradients, adaptationGradients) = ComputeNetworkGradients(
                task, taskRepresentation, fastWeights, taskModel, taskLoss);

            // Accumulate gradients
            if (accumulatedEncoderGradients == null)
            {
                accumulatedEncoderGradients = encoderGradients;
                accumulatedAdaptationGradients = adaptationGradients;
            }
            else
            {
                for (int i = 0; i < accumulatedEncoderGradients.Length; i++)
                {
                    accumulatedEncoderGradients[i] = NumOps.Add(accumulatedEncoderGradients[i], encoderGradients[i]);
                }
                if (accumulatedAdaptationGradients != null)
                {
                    for (int i = 0; i < accumulatedAdaptationGradients.Length; i++)
                    {
                        accumulatedAdaptationGradients[i] = NumOps.Add(accumulatedAdaptationGradients[i], adaptationGradients[i]);
                    }
                }
            }
        }

        if (accumulatedEncoderGradients == null || accumulatedAdaptationGradients == null)
        {
            throw new InvalidOperationException("Failed to compute network gradients.");
        }

        // Average and apply gradients
        T batchSizeT = NumOps.FromDouble(taskBatch.BatchSize);
        for (int i = 0; i < accumulatedEncoderGradients.Length; i++)
        {
            accumulatedEncoderGradients[i] = NumOps.Divide(accumulatedEncoderGradients[i], batchSizeT);
        }
        for (int i = 0; i < accumulatedAdaptationGradients.Length; i++)
        {
            accumulatedAdaptationGradients[i] = NumOps.Divide(accumulatedAdaptationGradients[i], batchSizeT);
        }

        // Clip gradients if configured
        if (_cnapOptions.GradientClipThreshold.HasValue && _cnapOptions.GradientClipThreshold.Value > 0)
        {
            accumulatedEncoderGradients = ClipGradients(accumulatedEncoderGradients, _cnapOptions.GradientClipThreshold.Value);
            accumulatedAdaptationGradients = ClipGradients(accumulatedAdaptationGradients, _cnapOptions.GradientClipThreshold.Value);
        }

        // Update encoder and adaptation network weights
        _encoderWeights = ApplyGradients(_encoderWeights, accumulatedEncoderGradients, _cnapOptions.OuterLearningRate);
        _adaptationNetworkWeights = ApplyGradients(_adaptationNetworkWeights, accumulatedAdaptationGradients, _cnapOptions.OuterLearningRate);

        // Also update base model if gradients are available
        var baseModelGradients = ComputeBaseModelGradients(taskBatch);
        if (baseModelGradients != null)
        {
            var currentParams = MetaModel.GetParameters();
            var updatedParams = ApplyGradients(currentParams, baseModelGradients, _cnapOptions.OuterLearningRate);
            MetaModel.SetParameters(updatedParams);
        }

        return NumOps.Divide(totalLoss, batchSizeT);
    }

    /// <summary>
    /// Adapts the meta-learned model to a new task using CNAP's feed-forward approach.
    /// </summary>
    /// <param name="task">The new task containing support set examples for adaptation.</param>
    /// <returns>A new model instance that has been adapted to the given task using generated fast weights.</returns>
    /// <exception cref="ArgumentNullException">Thrown when task is null.</exception>
    /// <remarks>
    /// <para>
    /// CNAP adaptation is extremely efficient compared to gradient-based methods:
    /// </para>
    /// <para>
    /// <b>Adaptation Process:</b>
    /// 1. Encode all support set examples into embeddings
    /// 2. Aggregate embeddings into a task representation vector
    /// 3. Generate fast weights from the task representation
    /// 4. Apply fast weights to the base model
    /// 5. Return the adapted model (no gradient steps needed!)
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> When adapting to a new task, CNAP works like instant
    /// recognition. You show it a few examples, it immediately "understands" the task
    /// pattern, and modifies its behavior accordingly - all in a single forward pass.
    /// </para>
    /// <para>
    /// <b>Advantages over MAML at test time:</b>
    /// - No gradient computation required (just forward passes)
    /// - Constant time regardless of number of adaptation steps
    /// - More memory efficient (no computational graph needed)
    /// - Better suited for real-time applications
    /// </para>
    /// <para>
    /// <b>Trade-offs:</b>
    /// - May be less flexible than MAML for out-of-distribution tasks
    /// - Requires task distribution to be learnable by the adaptation network
    /// - Fast weight generation adds parameters that need to be trained
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

        // Encode task from support set
        var taskRepresentation = EncodeTask(task.SupportInput, task.SupportOutput);

        // Generate fast weights
        var fastWeights = GenerateFastWeights(taskRepresentation);

        // Apply fast weights to the model
        ApplyFastWeights(adaptedModel, fastWeights);

        // Store for potential further use
        _currentTaskRepresentation = taskRepresentation;
        _currentFastWeights = fastWeights;

        return adaptedModel;
    }

    /// <summary>
    /// Encodes support set data into a task representation vector.
    /// </summary>
    /// <param name="supportInput">The support set inputs.</param>
    /// <param name="supportOutput">The support set outputs.</param>
    /// <returns>A vector representing the task's characteristics.</returns>
    /// <remarks>
    /// <para>
    /// The encoder processes each (input, output) pair and aggregates them into
    /// a single task representation using either mean pooling or attention.
    /// </para>
    /// </remarks>
    private Vector<T> EncodeTask(TInput supportInput, TOutput supportOutput)
    {
        // Simplified encoding: use model to extract features and aggregate
        // In a full implementation, this would use a dedicated encoder network

        var representation = new Vector<T>(_cnapOptions.RepresentationDimension);

        // Get predictions from support input to understand task structure
        var predictions = MetaModel.Predict(supportInput);

        // Compute simple task representation from prediction-target differences
        // This captures what the task requires the model to learn
        var predictionVector = ConvertToVector(predictions);
        var targetVector = ConvertToVector(supportOutput);

        if (predictionVector != null && targetVector != null)
        {
            int minLen = Math.Min(Math.Min(predictionVector.Length, targetVector.Length), _cnapOptions.RepresentationDimension);

            for (int i = 0; i < minLen; i++)
            {
                // Encode difference between prediction and target
                T diff = NumOps.Subtract(targetVector[i], predictionVector[i]);
                representation[i] = diff;
            }

            // Fill remaining dimensions with aggregated statistics
            if (minLen < _cnapOptions.RepresentationDimension)
            {
                T mean = ComputeMeanPartial(representation, minLen);
                for (int i = minLen; i < _cnapOptions.RepresentationDimension; i++)
                {
                    representation[i] = mean;
                }
            }
        }

        // Apply learned encoder transformation
        representation = ApplyEncoderTransform(representation);

        return representation;
    }

    /// <summary>
    /// Generates fast weights from the task representation.
    /// </summary>
    /// <param name="taskRepresentation">The encoded task representation.</param>
    /// <returns>Fast weights to be applied to the base model.</returns>
    private Vector<T> GenerateFastWeights(Vector<T> taskRepresentation)
    {
        int numModelParams = MetaModel.GetParameters().Length;
        var fastWeights = new Vector<T>(numModelParams);

        // Apply adaptation network to generate fast weights
        // Simplified: linear projection with learned weights
        int weightsPerParam = _cnapOptions.RepresentationDimension;

        for (int i = 0; i < numModelParams; i++)
        {
            T sum = NumOps.Zero;

            // Compute weighted sum of representation dimensions
            int startIdx = (i * weightsPerParam) % _adaptationNetworkWeights.Length;
            for (int j = 0; j < Math.Min(weightsPerParam, taskRepresentation.Length); j++)
            {
                int weightIdx = (startIdx + j) % _adaptationNetworkWeights.Length;
                sum = NumOps.Add(sum, NumOps.Multiply(_adaptationNetworkWeights[weightIdx], taskRepresentation[j]));
            }

            // Apply scaling based on fast weight mode
            switch (_cnapOptions.FastWeightMode)
            {
                case FastWeightApplicationMode.Additive:
                    fastWeights[i] = NumOps.Multiply(NumOps.FromDouble(_cnapOptions.FastWeightScale), sum);
                    break;

                case FastWeightApplicationMode.Multiplicative:
                    // Generate scaling factors around 1.0
                    fastWeights[i] = NumOps.Add(NumOps.One, NumOps.Multiply(NumOps.FromDouble(_cnapOptions.FastWeightScale * 0.1), sum));
                    break;

                case FastWeightApplicationMode.FiLM:
                    // FiLM style: will be interpreted as (gamma, beta) pairs
                    fastWeights[i] = sum;
                    break;
            }
        }

        // Normalize fast weights if configured
        if (_cnapOptions.NormalizeFastWeights)
        {
            fastWeights = NormalizeFastWeights(fastWeights);
        }

        return fastWeights;
    }

    /// <summary>
    /// Applies fast weights to modify the model parameters.
    /// </summary>
    /// <param name="model">The model to modify.</param>
    /// <param name="fastWeights">The fast weights to apply.</param>
    private void ApplyFastWeights(IFullModel<T, TInput, TOutput> model, Vector<T> fastWeights)
    {
        var currentParams = model.GetParameters();

        if (currentParams.Length != fastWeights.Length)
        {
            // If sizes don't match, scale fast weights appropriately
            fastWeights = ResizeFastWeights(fastWeights, currentParams.Length);
        }

        var modifiedParams = new Vector<T>(currentParams.Length);

        switch (_cnapOptions.FastWeightMode)
        {
            case FastWeightApplicationMode.Additive:
                for (int i = 0; i < currentParams.Length; i++)
                {
                    modifiedParams[i] = NumOps.Add(currentParams[i], fastWeights[i]);
                }
                break;

            case FastWeightApplicationMode.Multiplicative:
                for (int i = 0; i < currentParams.Length; i++)
                {
                    modifiedParams[i] = NumOps.Multiply(currentParams[i], fastWeights[i]);
                }
                break;

            case FastWeightApplicationMode.FiLM:
                // For FiLM, interpret fast weights as alternating (gamma, beta)
                for (int i = 0; i < currentParams.Length; i++)
                {
                    if (i % 2 == 0)
                    {
                        // Gamma (scaling)
                        T gamma = NumOps.Add(NumOps.One, fastWeights[i]);
                        modifiedParams[i] = NumOps.Multiply(currentParams[i], gamma);
                    }
                    else
                    {
                        // Beta (shift)
                        modifiedParams[i] = NumOps.Add(currentParams[i], fastWeights[i]);
                    }
                }
                break;
        }

        model.SetParameters(modifiedParams);
    }

    /// <summary>
    /// Normalizes fast weights to prevent explosion.
    /// </summary>
    private Vector<T> NormalizeFastWeights(Vector<T> weights)
    {
        // Compute L2 norm
        T normSquared = NumOps.Zero;
        for (int i = 0; i < weights.Length; i++)
        {
            normSquared = NumOps.Add(normSquared, NumOps.Multiply(weights[i], weights[i]));
        }

        double norm = Math.Sqrt(Math.Max(NumOps.ToDouble(normSquared), 1e-8));
        double maxNorm = _cnapOptions.FastWeightScale * Math.Sqrt(weights.Length);

        if (norm > maxNorm)
        {
            T scaleFactor = NumOps.FromDouble(maxNorm / norm);
            var normalized = new Vector<T>(weights.Length);
            for (int i = 0; i < weights.Length; i++)
            {
                normalized[i] = NumOps.Multiply(weights[i], scaleFactor);
            }
            return normalized;
        }

        return weights;
    }

    /// <summary>
    /// Computes gradients for the encoder and adaptation networks using finite differences.
    /// </summary>
    private (Vector<T> encoderGradients, Vector<T> adaptationGradients) ComputeNetworkGradients(
        IMetaLearningTask<T, TInput, TOutput> task,
        Vector<T> taskRepresentation,
        Vector<T> fastWeights,
        IFullModel<T, TInput, TOutput> taskModel,
        T currentLoss)
    {
        // Use finite differences for gradient computation
        double epsilon = 1e-5;

        // Compute encoder gradients with scale factor for unbiased estimation
        var encoderGradients = new Vector<T>(_encoderWeights.Length);
        int encoderSampleCount = Math.Min(_encoderWeights.Length, 100);
        double encoderScaleFactor = (double)_encoderWeights.Length / encoderSampleCount;
        for (int i = 0; i < encoderSampleCount; i++)
        {
            int idx = encoderSampleCount > 0 ? (i * _encoderWeights.Length / encoderSampleCount) : i;

            // Perturb encoder weight
            T original = _encoderWeights[idx];
            _encoderWeights[idx] = NumOps.Add(original, NumOps.FromDouble(epsilon));

            // Recompute with perturbed encoder
            var perturbedRep = EncodeTask(task.SupportInput, task.SupportOutput);
            var perturbedWeights = GenerateFastWeights(perturbedRep);
            var perturbedModel = CloneModel();
            ApplyFastWeights(perturbedModel, perturbedWeights);
            var perturbedPred = perturbedModel.Predict(task.QueryInput);
            T perturbedLoss = ComputeLossFromOutput(perturbedPred, task.QueryOutput);

            // Compute gradient with scale factor
            double grad = (NumOps.ToDouble(perturbedLoss) - NumOps.ToDouble(currentLoss)) / epsilon;
            encoderGradients[idx] = NumOps.FromDouble(grad * encoderScaleFactor);

            // Restore
            _encoderWeights[idx] = original;
        }

        // Compute adaptation network gradients with scale factor for unbiased estimation
        var adaptationGradients = new Vector<T>(_adaptationNetworkWeights.Length);
        int adaptSampleCount = Math.Min(_adaptationNetworkWeights.Length, 100);
        double adaptScaleFactor = (double)_adaptationNetworkWeights.Length / adaptSampleCount;
        for (int i = 0; i < adaptSampleCount; i++)
        {
            int idx = adaptSampleCount > 0 ? (i * _adaptationNetworkWeights.Length / adaptSampleCount) : i;

            // Perturb adaptation weight
            T original = _adaptationNetworkWeights[idx];
            _adaptationNetworkWeights[idx] = NumOps.Add(original, NumOps.FromDouble(epsilon));

            // Recompute with perturbed adaptation network
            var perturbedWeights = GenerateFastWeights(taskRepresentation);
            var perturbedModel = CloneModel();
            ApplyFastWeights(perturbedModel, perturbedWeights);
            var perturbedPred = perturbedModel.Predict(task.QueryInput);
            T perturbedLoss = ComputeLossFromOutput(perturbedPred, task.QueryOutput);

            // Compute gradient with scale factor
            double grad = (NumOps.ToDouble(perturbedLoss) - NumOps.ToDouble(currentLoss)) / epsilon;
            adaptationGradients[idx] = NumOps.FromDouble(grad * adaptScaleFactor);

            // Restore
            _adaptationNetworkWeights[idx] = original;
        }

        return (encoderGradients, adaptationGradients);
    }

    /// <summary>
    /// Computes gradients for the base model parameters.
    /// </summary>
    private Vector<T>? ComputeBaseModelGradients(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        Vector<T>? accumulatedGradients = null;

        foreach (var task in taskBatch.Tasks)
        {
            var taskModel = CloneModel();
            var taskRep = EncodeTask(task.SupportInput, task.SupportOutput);
            var fastWeights = GenerateFastWeights(taskRep);
            ApplyFastWeights(taskModel, fastWeights);

            // Compute gradients on query set
            var gradients = ComputeGradients(taskModel, task.QueryInput, task.QueryOutput);

            if (accumulatedGradients == null)
            {
                accumulatedGradients = gradients;
            }
            else
            {
                for (int i = 0; i < accumulatedGradients.Length; i++)
                {
                    accumulatedGradients[i] = NumOps.Add(accumulatedGradients[i], gradients[i]);
                }
            }
        }

        if (accumulatedGradients != null)
        {
            T batchSize = NumOps.FromDouble(taskBatch.BatchSize);
            for (int i = 0; i < accumulatedGradients.Length; i++)
            {
                accumulatedGradients[i] = NumOps.Divide(accumulatedGradients[i], batchSize);
            }
        }

        return accumulatedGradients;
    }

    /// <summary>
    /// Applies learned encoder transformation to the representation.
    /// </summary>
    private Vector<T> ApplyEncoderTransform(Vector<T> input)
    {
        var output = new Vector<T>(input.Length);

        // Simple linear transformation with encoder weights
        for (int i = 0; i < input.Length; i++)
        {
            T sum = input[i];

            // Apply learned weights (simplified)
            int weightIdx = i % _encoderWeights.Length;
            sum = NumOps.Multiply(sum, _encoderWeights[weightIdx]);

            // Apply activation (tanh)
            double val = NumOps.ToDouble(sum);
            output[i] = NumOps.FromDouble(Math.Tanh(val));
        }

        return output;
    }

    /// <summary>
    /// Estimates the input/output dimension for the encoder.
    /// </summary>
    private int EstimateInputOutputDimension()
    {
        // Use model parameter count as a proxy
        return Math.Max(128, MetaModel.GetParameters().Length / 10);
    }

    /// <summary>
    /// Initializes weights using Xavier/He initialization.
    /// </summary>
    private Vector<T> InitializeWeights(int size)
    {
        var weights = new Vector<T>(size);
        double scale = Math.Sqrt(2.0 / size);

        for (int i = 0; i < size; i++)
        {
            // Xavier initialization
            double value = (RandomGenerator.NextDouble() * 2 - 1) * scale;
            weights[i] = NumOps.FromDouble(value);
        }

        return weights;
    }

    /// <summary>
    /// Resizes fast weights to match model parameter count.
    /// </summary>
    private Vector<T> ResizeFastWeights(Vector<T> fastWeights, int targetSize)
    {
        var resized = new Vector<T>(targetSize);

        for (int i = 0; i < targetSize; i++)
        {
            // Interpolate or repeat
            int srcIdx = (i * fastWeights.Length) / targetSize;
            resized[i] = fastWeights[srcIdx];
        }

        return resized;
    }

    /// <summary>
    /// Computes mean of first n elements of a vector.
    /// </summary>
    private T ComputeMeanPartial(Vector<T> vec, int n)
    {
        if (n <= 0) return NumOps.Zero;

        T sum = NumOps.Zero;
        for (int i = 0; i < n; i++)
        {
            sum = NumOps.Add(sum, vec[i]);
        }

        return NumOps.Divide(sum, NumOps.FromDouble(n));
    }
}
