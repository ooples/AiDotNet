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
/// Implementation of Task-Dependent Adaptive Metric (TADAM) algorithm for few-shot learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// TADAM extends Prototypical Networks by incorporating:
/// 1. Task Conditioning (TC) using FiLM layers to modulate features
/// 2. Metric Scaling to learn per-dimension distance weights
/// 3. Auxiliary Co-Training for improved feature learning
/// </para>
/// <para><b>For Beginners:</b> TADAM improves on ProtoNets by making the feature
/// extractor "aware" of the current task:
///
/// **How it works:**
/// 1. Extract features from support set examples
/// 2. Compute a "task embedding" summarizing what the task is about
/// 3. Use this task embedding to adjust (condition) how features are extracted
/// 4. Compute prototypes from the conditioned features
/// 5. Classify queries using scaled distances to prototypes
///
/// **Key insight:** Different tasks may require focusing on different features.
/// TADAM learns to adjust what the network pays attention to based on the task.
/// </para>
/// <para><b>Algorithm - TADAM:</b>
/// <code>
/// # Components
/// f_theta = feature_encoder with FiLM layers
/// g_phi = task_encoder that produces task embedding
/// alpha = learnable metric scaling parameters
/// tau = learnable temperature
///
/// # Episode training
/// for each episode:
///     # 1. Compute task embedding from support set
///     support_features = f_theta(support_set)  # Initial features
///     task_embedding = g_phi(mean(support_features))
///
///     # 2. Apply task conditioning via FiLM
///     gamma, beta = FiLM_generator(task_embedding)
///     conditioned_features = gamma * features + beta
///
///     # 3. Compute class prototypes
///     for each class c:
///         prototype_c = mean(conditioned_features[class == c])
///
///     # 4. Classify queries with scaled distance
///     query_features = f_theta(query_set, task_conditioning=True)
///     for each query q:
///         for each class c:
///             dist = sum(alpha * (query_features[q] - prototype_c)^2)
///         p(y=c|q) = softmax(-dist / tau)
///
///     # 5. Compute loss and update
///     loss = cross_entropy(p, true_labels)
///     if use_auxiliary:
///         loss += aux_weight * auxiliary_loss
///     update(f_theta, g_phi, alpha, tau)
/// </code>
/// </para>
/// <para><b>Key Innovations:</b>
///
/// 1. **Task Conditioning (TC)**: FiLM layers modulate feature maps based on task context.
///    gamma and beta parameters are generated from the task embedding.
///
/// 2. **Metric Scaling**: Learns per-dimension weights (alpha) for the distance metric,
///    allowing the model to emphasize or de-emphasize different feature dimensions.
///
/// 3. **Learnable Temperature**: The temperature tau controls softmax sharpness and
///    is learned along with other parameters.
///
/// 4. **Auxiliary Co-Training**: Optional auxiliary classification loss on base classes
///    to improve feature learning.
/// </para>
/// </remarks>
public class TADAMAlgorithm<T, TInput, TOutput> : MetaLearnerBase<T, TInput, TOutput>
{
    private readonly TADAMOptions<T, TInput, TOutput> _tadamOptions;

    // Learnable metric scaling parameters (alpha)
    private Vector<T> _metricScale;

    // Learnable temperature parameter
    private T _temperature;

    // Task embedding for current episode
    private Tensor<T>? _currentTaskEmbedding;

    /// <summary>
    /// Initializes a new instance of the TADAMAlgorithm class.
    /// </summary>
    /// <param name="options">The configuration options for TADAM.</param>
    /// <exception cref="ArgumentNullException">Thrown when options or required components are null.</exception>
    /// <exception cref="ArgumentException">Thrown when configuration validation fails.</exception>
    public TADAMAlgorithm(TADAMOptions<T, TInput, TOutput> options)
        : base(
            options?.MetaModel ?? throw new ArgumentNullException(nameof(options), "MetaModel must be set in options."),
            options.LossFunction ?? options.MetaModel.DefaultLossFunction,
            options,
            options.DataLoader,
            options.MetaOptimizer,
            options.InnerOptimizer)
    {
        _tadamOptions = options;

        // Initialize metric scaling parameters
        _metricScale = new Vector<T>(_tadamOptions.EmbeddingDimension);
        for (int i = 0; i < _metricScale.Length; i++)
        {
            _metricScale[i] = NumOps.One; // Initialize to 1 (no scaling initially)
        }

        // Initialize temperature
        _temperature = NumOps.FromDouble(_tadamOptions.InitialTemperature);

        // Validate configuration
        if (!_tadamOptions.IsValid())
        {
            throw new ArgumentException("TADAM configuration is invalid. Check all parameters.", nameof(options));
        }
    }

    /// <inheritdoc/>
    public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.TADAM;

    /// <inheritdoc/>
    public override T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        if (taskBatch == null || taskBatch.BatchSize == 0)
        {
            throw new ArgumentException("Task batch cannot be null or empty.", nameof(taskBatch));
        }

        T totalLoss = NumOps.Zero;

        foreach (var task in taskBatch.Tasks)
        {
            // Train on this episode
            T episodeLoss = TrainEpisode(task);
            totalLoss = NumOps.Add(totalLoss, episodeLoss);
        }

        // Return average loss
        return NumOps.Divide(totalLoss, NumOps.FromDouble(taskBatch.BatchSize));
    }

    /// <inheritdoc/>
    public override IModel<TInput, TOutput, ModelMetadata<T>> Adapt(IMetaLearningTask<T, TInput, TOutput> task)
    {
        if (task == null)
        {
            throw new ArgumentNullException(nameof(task));
        }

        // Step 1: Compute task embedding from support set
        _currentTaskEmbedding = ComputeTaskEmbedding(task.SupportInput, task.SupportOutput);

        // Step 2: Compute prototypes with task-conditioned features
        var prototypes = ComputeConditionedPrototypes(task.SupportInput, task.SupportOutput);

        // Create adapted model that stores prototypes
        var adaptedModel = new TADAMModel<T, TInput, TOutput>(
            MetaModel,
            prototypes,
            _metricScale,
            _temperature,
            _tadamOptions);

        return adaptedModel;
    }

    /// <summary>
    /// Trains on a single episode using TADAM.
    /// </summary>
    private T TrainEpisode(IMetaLearningTask<T, TInput, TOutput> task)
    {
        // Step 1: Compute task embedding from support set
        var taskEmbedding = ComputeTaskEmbedding(task.SupportInput, task.SupportOutput);
        _currentTaskEmbedding = taskEmbedding;

        // Step 2: Apply task conditioning and compute prototypes
        var prototypes = ComputeConditionedPrototypes(task.SupportInput, task.SupportOutput);

        // Step 3: Encode query examples with task conditioning
        var queryFeatures = EncodeWithTaskConditioning(task.QueryInput, taskEmbedding);

        // Step 4: Compute scaled distances to prototypes
        var distances = ComputeScaledDistances(queryFeatures, prototypes);

        // Step 5: Apply softmax with learnable temperature
        var probabilities = ApplySoftmaxWithTemperature(distances);

        // Step 6: Compute cross-entropy loss
        var loss = ComputeCrossEntropyLoss(probabilities, task.QueryOutput);

        // Step 7: Add auxiliary co-training loss if enabled
        if (_tadamOptions.UseAuxiliaryCoTraining)
        {
            var auxLoss = ComputeAuxiliaryLoss(task);
            T auxWeight = NumOps.FromDouble(_tadamOptions.AuxiliaryLossWeight);
            loss = NumOps.Add(loss, NumOps.Multiply(auxWeight, auxLoss));
        }

        // Step 8: Add regularization
        if (_tadamOptions.L2Regularization > 0)
        {
            loss = AddL2Regularization(loss);
        }

        // Step 9: Update parameters using gradients
        UpdateParameters(task, loss);

        return loss;
    }

    /// <summary>
    /// Updates all learnable parameters using gradient descent.
    /// </summary>
    private void UpdateParameters(IMetaLearningTask<T, TInput, TOutput> task, T loss)
    {
        // Compute gradients for feature encoder (MetaModel)
        var modelGradients = ComputeGradients(MetaModel, task.QueryInput, task.QueryOutput);

        // Apply gradient clipping if configured
        if (_options.GradientClipThreshold.HasValue && _options.GradientClipThreshold.Value > 0)
        {
            modelGradients = ClipGradients(modelGradients, _options.GradientClipThreshold.Value);
        }

        // Update MetaModel parameters
        var currentParams = MetaModel.GetParameters();
        var updatedParams = ApplyGradients(currentParams, modelGradients, _options.OuterLearningRate);
        MetaModel.SetParameters(updatedParams);

        // Update metric scale parameters using finite differences
        UpdateMetricScale(task, loss);

        // Update temperature parameter
        UpdateTemperature(task, loss);
    }

    /// <summary>
    /// Updates metric scaling parameters using gradient descent with efficient batch computation.
    /// </summary>
    /// <remarks>
    /// Instead of O(n) separate forward passes for each metric scale parameter,
    /// this method computes all gradients in a single batched operation using
    /// analytical gradients from the scaled distance computation.
    /// </remarks>
    private void UpdateMetricScale(IMetaLearningTask<T, TInput, TOutput> task, T baseLoss)
    {
        double learningRate = _options.OuterLearningRate * 0.1; // Use smaller LR for scale params

        // Compute prototypes and features once (not per-parameter)
        var prototypes = ComputeConditionedPrototypes(task.SupportInput, task.SupportOutput);
        Tensor<T> queryFeatures;
        if (_currentTaskEmbedding != null)
        {
            queryFeatures = EncodeWithTaskConditioning(task.QueryInput, _currentTaskEmbedding);
        }
        else
        {
            queryFeatures = EncodeExamples(task.QueryInput);
        }

        // Compute all gradients analytically from the distance computation
        // d_loss/d_alpha_i = sum over queries and prototypes of contribution from dimension i
        var gradients = ComputeMetricScaleGradients(queryFeatures, prototypes, task.QueryOutput);

        // Update all metric scale parameters in one pass
        for (int i = 0; i < _metricScale.Length && i < gradients.Length; i++)
        {
            // Update metric scale using computed gradient
            _metricScale[i] = NumOps.Subtract(_metricScale[i],
                NumOps.Multiply(gradients[i], NumOps.FromDouble(learningRate)));

            // Ensure metric scale stays positive
            if (NumOps.ToDouble(_metricScale[i]) < 1e-6)
            {
                _metricScale[i] = NumOps.FromDouble(1e-6);
            }
        }
    }

    /// <summary>
    /// Computes analytical gradients for all metric scale parameters in a single pass.
    /// </summary>
    /// <remarks>
    /// For scaled squared Euclidean distance: d = sum_i(alpha_i * (q_i - p_i)^2)
    /// The gradient w.r.t. alpha_i is: (q_i - p_i)^2 weighted by softmax derivatives.
    /// </remarks>
    private Vector<T> ComputeMetricScaleGradients(Tensor<T> queryFeatures,
        Dictionary<int, Tensor<T>> prototypes, TOutput trueLabels)
    {
        int featureDim = _metricScale.Length;
        var gradients = new Vector<T>(featureDim);

        int numQueries = queryFeatures.Shape.Length > 0 ? queryFeatures.Shape[0] : 1;
        int featureSize = queryFeatures.Shape.Length > 1 ? queryFeatures.Shape[1] :
            (queryFeatures.Shape.Length > 0 ? queryFeatures.Shape[0] : 1);

        // Limit feature size to metric scale length
        int effectiveFeatureSize = Math.Min(featureSize, featureDim);

        // Compute distances and softmax probabilities for gradient computation
        var distances = ComputeScaledDistances(queryFeatures, prototypes);
        var probabilities = ApplySoftmaxWithTemperature(distances);

        // Get ordered class labels to match probability matrix columns
        var classLabels = prototypes.Keys.OrderBy(k => k).ToList();
        int numClasses = classLabels.Count;

        // For each query, compute contribution to gradient
        for (int q = 0; q < numQueries; q++)
        {
            int trueClass = GetClassLabel(trueLabels, q);

            for (int c = 0; c < numClasses; c++)
            {
                int classLabel = classLabels[c];
                var prototype = prototypes[classLabel];

                // Get probability for this class from matrix [query, class]
                double prob = (q < probabilities.Rows && c < probabilities.Columns)
                    ? NumOps.ToDouble(probabilities[q, c])
                    : 0.0;

                // Gradient contribution: (prob - indicator) * squared_diff
                double indicator = (classLabel == trueClass) ? 1.0 : 0.0;
                double probDiff = prob - indicator;

                // Compute squared difference for each dimension
                for (int d = 0; d < effectiveFeatureSize; d++)
                {
                    T queryVal = queryFeatures.GetFlat(q * featureSize + d);
                    T protoVal = d < prototype.Shape[0] ? prototype[d] : NumOps.Zero;
                    T diff = NumOps.Subtract(queryVal, protoVal);
                    T squaredDiff = NumOps.Multiply(diff, diff);

                    // Gradient contribution for this dimension
                    T gradContrib = NumOps.Multiply(
                        NumOps.FromDouble(probDiff),
                        squaredDiff);

                    gradients[d] = NumOps.Add(gradients[d], gradContrib);
                }
            }
        }

        // Average gradients over queries
        if (numQueries > 0)
        {
            T invNumQueries = NumOps.FromDouble(1.0 / numQueries);
            for (int d = 0; d < featureDim; d++)
            {
                gradients[d] = NumOps.Multiply(gradients[d], invNumQueries);
            }
        }

        return gradients;
    }

    /// <summary>
    /// Updates temperature parameter using gradient descent.
    /// </summary>
    private void UpdateTemperature(IMetaLearningTask<T, TInput, TOutput> task, T baseLoss)
    {
        double epsilon = 1e-5;
        double learningRate = _options.OuterLearningRate * 0.01; // Use smaller LR for temperature

        // Perturb temperature
        T originalTemp = _temperature;
        _temperature = NumOps.Add(originalTemp, NumOps.FromDouble(epsilon));

        // Recompute loss with perturbed temperature
        T perturbedLoss = ComputeMetricLoss(task);

        // Compute gradient
        T gradient = NumOps.Divide(
            NumOps.Subtract(perturbedLoss, baseLoss),
            NumOps.FromDouble(epsilon));

        // Update temperature
        _temperature = NumOps.Subtract(originalTemp,
            NumOps.Multiply(gradient, NumOps.FromDouble(learningRate)));

        // Ensure temperature stays in valid range
        double tempVal = NumOps.ToDouble(_temperature);
        if (tempVal < 0.01)
        {
            _temperature = NumOps.FromDouble(0.01);
        }
        else if (tempVal > 10.0)
        {
            _temperature = NumOps.FromDouble(10.0);
        }
    }

    /// <summary>
    /// Computes metric-based loss for gradient computation.
    /// </summary>
    private T ComputeMetricLoss(IMetaLearningTask<T, TInput, TOutput> task)
    {
        // Ensure task embedding exists before computing metric loss
        if (_currentTaskEmbedding == null)
        {
            // Compute task embedding if not already set
            _currentTaskEmbedding = ComputeTaskEmbedding(task.SupportInput, task.SupportOutput);
        }

        var prototypes = ComputeConditionedPrototypes(task.SupportInput, task.SupportOutput);
        var queryFeatures = EncodeWithTaskConditioning(task.QueryInput, _currentTaskEmbedding);
        var distances = ComputeScaledDistances(queryFeatures, prototypes);
        var probabilities = ApplySoftmaxWithTemperature(distances);
        return ComputeCrossEntropyLoss(probabilities, task.QueryOutput);
    }

    /// <summary>
    /// Computes task embedding from support set statistics.
    /// </summary>
    private Tensor<T> ComputeTaskEmbedding(TInput supportInput, TOutput supportOutput)
    {
        // Extract features from support set
        var supportFeatures = EncodeExamples(supportInput);

        // Compute mean feature as task embedding
        var taskEmbedding = ComputeFeatureMean(supportFeatures);

        return taskEmbedding;
    }

    /// <summary>
    /// Computes prototypes using task-conditioned features.
    /// </summary>
    private Dictionary<int, Tensor<T>> ComputeConditionedPrototypes(TInput supportInput, TOutput supportOutput)
    {
        // Apply task conditioning if enabled
        var features = _tadamOptions.UseTaskConditioning && _currentTaskEmbedding != null
            ? EncodeWithTaskConditioning(supportInput, _currentTaskEmbedding)
            : EncodeExamples(supportInput);

        // Group features by class and compute prototypes
        var prototypes = GroupAndComputePrototypes(features, supportOutput);

        // Normalize prototypes if enabled
        if (_tadamOptions.NormalizeEmbeddings)
        {
            foreach (var key in prototypes.Keys.ToList())
            {
                prototypes[key] = NormalizeTensor(prototypes[key]);
            }
        }

        return prototypes;
    }

    /// <summary>
    /// Encodes examples with FiLM-style task conditioning.
    /// </summary>
    private Tensor<T> EncodeWithTaskConditioning(TInput input, Tensor<T> taskEmbedding)
    {
        // Get base features
        var features = EncodeExamples(input);

        if (!_tadamOptions.UseTaskConditioning)
        {
            return features;
        }

        // Apply FiLM modulation: gamma * features + beta
        // For simplicity, we use task embedding to generate simple scale and shift
        var gamma = ComputeFiLMGamma(taskEmbedding);
        var beta = ComputeFiLMBeta(taskEmbedding);

        return ApplyFiLM(features, gamma, beta);
    }

    /// <summary>
    /// Computes FiLM gamma (scale) parameters from task embedding.
    /// </summary>
    private Tensor<T> ComputeFiLMGamma(Tensor<T> taskEmbedding)
    {
        // Simple linear projection from task embedding to scaling factors
        int targetSize = _tadamOptions.EmbeddingDimension;
        var gamma = new Tensor<T>(new int[] { targetSize });

        int srcSize = 1;
        for (int i = 0; i < taskEmbedding.Shape.Length; i++)
        {
            srcSize *= taskEmbedding.Shape[i];
        }

        // Initialize with sigmoid of task embedding values
        for (int i = 0; i < targetSize; i++)
        {
            int srcIdx = i % srcSize;
            double val = NumOps.ToDouble(taskEmbedding.GetFlat(srcIdx));
            // Sigmoid to keep scale in reasonable range [0.5, 1.5]
            double scaledVal = 0.5 + 1.0 / (1.0 + Math.Exp(-val));
            gamma[i] = NumOps.FromDouble(scaledVal);
        }

        return gamma;
    }

    /// <summary>
    /// Computes FiLM beta (shift) parameters from task embedding.
    /// </summary>
    private Tensor<T> ComputeFiLMBeta(Tensor<T> taskEmbedding)
    {
        int targetSize = _tadamOptions.EmbeddingDimension;
        var beta = new Tensor<T>(new int[] { targetSize });

        int srcSize = 1;
        for (int i = 0; i < taskEmbedding.Shape.Length; i++)
        {
            srcSize *= taskEmbedding.Shape[i];
        }

        // Initialize with tanh of task embedding values
        for (int i = 0; i < targetSize; i++)
        {
            int srcIdx = i % srcSize;
            double val = NumOps.ToDouble(taskEmbedding.GetFlat(srcIdx));
            // Tanh to keep shift in reasonable range
            double shiftVal = 0.1 * Math.Tanh(val);
            beta[i] = NumOps.FromDouble(shiftVal);
        }

        return beta;
    }

    /// <summary>
    /// Applies FiLM modulation: gamma * features + beta.
    /// </summary>
    private Tensor<T> ApplyFiLM(Tensor<T> features, Tensor<T> gamma, Tensor<T> beta)
    {
        int totalSize = 1;
        for (int i = 0; i < features.Shape.Length; i++)
        {
            totalSize *= features.Shape[i];
        }

        var result = new Tensor<T>(features.Shape);
        int featureDim = gamma.Shape.Length > 0 ? gamma.Shape[0] : 1;

        for (int i = 0; i < totalSize; i++)
        {
            int dimIdx = i % featureDim;
            T val = features.GetFlat(i);
            T g = gamma[dimIdx];
            T b = beta[dimIdx];

            // gamma * val + beta
            T scaled = NumOps.Multiply(g, val);
            result.SetFlat(i, NumOps.Add(scaled, b));
        }

        return result;
    }

    /// <summary>
    /// Encodes input examples to feature representations using the MetaModel.
    /// </summary>
    /// <remarks>
    /// This method uses the MetaModel's forward pass to extract feature embeddings
    /// from input examples. For TADAM, these features are then used for prototype
    /// computation and task conditioning.
    /// </remarks>
    private Tensor<T> EncodeExamples(TInput inputs)
    {
        // If input is already a tensor, use it directly for encoding
        if (inputs is Tensor<T> inputTensor)
        {
            // If we have a MetaModel that can produce embeddings, use it
            if (MetaModel != null)
            {
                try
                {
                    // Forward pass through the model to get features
                    var output = MetaModel.Predict(inputs);

                    // Convert output to tensor if possible
                    if (output is Tensor<T> outputTensor)
                    {
                        return outputTensor;
                    }
                    else if (output is Vector<T> outputVector)
                    {
                        // Convert vector to 2D tensor [1, featureSize]
                        var resultTensor = new Tensor<T>(new int[] { 1, outputVector.Length });
                        for (int i = 0; i < outputVector.Length; i++)
                        {
                            resultTensor[new[] { 0, i }] = outputVector[i];
                        }
                        return resultTensor;
                    }
                }
                catch
                {
                    // If model forward pass fails, fall back to using input directly
                    // This can happen during initialization or with incompatible input shapes
                }
            }

            // If no model encoding available, use input features directly
            // This is valid for pre-extracted features
            return inputTensor;
        }

        // Handle Matrix<T> input
        if (inputs is Matrix<T> inputMatrix)
        {
            // Convert matrix to tensor [numExamples, featureDim]
            var resultTensor = new Tensor<T>(new int[] { inputMatrix.Rows, inputMatrix.Columns });
            for (int i = 0; i < inputMatrix.Rows; i++)
            {
                for (int j = 0; j < inputMatrix.Columns; j++)
                {
                    resultTensor[new[] { i, j }] = inputMatrix[i, j];
                }
            }
            return resultTensor;
        }

        // Handle Vector<T> input - single example
        if (inputs is Vector<T> inputVector)
        {
            // Convert vector to tensor [1, featureDim]
            var resultTensor = new Tensor<T>(new int[] { 1, inputVector.Length });
            for (int i = 0; i < inputVector.Length; i++)
            {
                resultTensor[new[] { 0, i }] = inputVector[i];
            }
            return resultTensor;
        }

        // Fallback: create empty tensor with expected embedding dimension
        return new Tensor<T>(new int[] { 1, _tadamOptions.EmbeddingDimension });
    }

    /// <summary>
    /// Computes mean feature vector.
    /// </summary>
    private Tensor<T> ComputeFeatureMean(Tensor<T> features)
    {
        if (features.Shape.Length < 2)
        {
            return features;
        }

        int numExamples = features.Shape[0];
        int featureSize = 1;
        for (int i = 1; i < features.Shape.Length; i++)
        {
            featureSize *= features.Shape[i];
        }

        var mean = new Tensor<T>(new int[] { featureSize });

        for (int f = 0; f < featureSize; f++)
        {
            T sum = NumOps.Zero;
            for (int n = 0; n < numExamples; n++)
            {
                sum = NumOps.Add(sum, features.GetFlat(n * featureSize + f));
            }
            mean[f] = NumOps.Divide(sum, NumOps.FromDouble(numExamples));
        }

        return mean;
    }

    /// <summary>
    /// Groups features by class and computes prototypes.
    /// </summary>
    private Dictionary<int, Tensor<T>> GroupAndComputePrototypes(Tensor<T> features, TOutput labels)
    {
        var classFeatures = new Dictionary<int, List<Tensor<T>>>();

        int numExamples = features.Shape.Length > 0 ? features.Shape[0] : 0;
        int featureSize = features.Shape.Length > 1 ? features.Shape[1] : 1;

        for (int i = 0; i < numExamples; i++)
        {
            int classLabel = GetClassLabel(labels, i);

            var feature = new Tensor<T>(new int[] { featureSize });
            for (int f = 0; f < featureSize; f++)
            {
                feature[f] = features.GetFlat(i * featureSize + f);
            }

            if (!classFeatures.ContainsKey(classLabel))
            {
                classFeatures[classLabel] = new List<Tensor<T>>();
            }
            classFeatures[classLabel].Add(feature);
        }

        // Compute prototype for each class
        var prototypes = new Dictionary<int, Tensor<T>>();
        foreach (var kvp in classFeatures)
        {
            prototypes[kvp.Key] = ComputePrototype(kvp.Value);
        }

        return prototypes;
    }

    /// <summary>
    /// Computes prototype (mean) of a list of feature tensors.
    /// </summary>
    private Tensor<T> ComputePrototype(List<Tensor<T>> features)
    {
        if (features.Count == 0)
        {
            return new Tensor<T>(new int[] { _tadamOptions.EmbeddingDimension });
        }

        int featureSize = features[0].Shape.Length > 0 ? features[0].Shape[0] : 0;
        var prototype = new Tensor<T>(new int[] { featureSize });

        for (int f = 0; f < featureSize; f++)
        {
            T sum = NumOps.Zero;
            foreach (var feature in features)
            {
                sum = NumOps.Add(sum, feature[f]);
            }
            prototype[f] = NumOps.Divide(sum, NumOps.FromDouble(features.Count));
        }

        return prototype;
    }

    /// <summary>
    /// Computes scaled squared distances from queries to prototypes.
    /// </summary>
    private Matrix<T> ComputeScaledDistances(Tensor<T> queryFeatures, Dictionary<int, Tensor<T>> prototypes)
    {
        int numQueries = queryFeatures.Shape.Length > 0 ? queryFeatures.Shape[0] : 0;
        int numClasses = prototypes.Count;
        int featureSize = queryFeatures.Shape.Length > 1 ? queryFeatures.Shape[1] : 1;

        var distances = new Matrix<T>(numQueries, numClasses);
        var classLabels = prototypes.Keys.OrderBy(k => k).ToList();

        for (int q = 0; q < numQueries; q++)
        {
            for (int c = 0; c < numClasses; c++)
            {
                int classLabel = classLabels[c];
                var prototype = prototypes[classLabel];

                T dist = NumOps.Zero;
                int scaleSize = _metricScale.Length;

                for (int f = 0; f < featureSize; f++)
                {
                    T queryVal = queryFeatures.GetFlat(q * featureSize + f);
                    T protoVal = prototype[f];
                    T diff = NumOps.Subtract(queryVal, protoVal);
                    T diffSquared = NumOps.Multiply(diff, diff);

                    // Apply metric scaling
                    if (_tadamOptions.UseMetricScaling && f < scaleSize)
                    {
                        diffSquared = NumOps.Multiply(diffSquared, _metricScale[f]);
                    }

                    dist = NumOps.Add(dist, diffSquared);
                }

                distances[q, c] = dist;
            }
        }

        return distances;
    }

    /// <summary>
    /// Applies softmax with learnable temperature.
    /// </summary>
    private Matrix<T> ApplySoftmaxWithTemperature(Matrix<T> distances)
    {
        int numQueries = distances.Rows;
        int numClasses = distances.Columns;
        var probabilities = new Matrix<T>(numQueries, numClasses);

        for (int q = 0; q < numQueries; q++)
        {
            // Find max for numerical stability
            T maxNegDist = NumOps.Negate(distances[q, 0]);
            for (int c = 1; c < numClasses; c++)
            {
                T negDist = NumOps.Negate(distances[q, c]);
                if (NumOps.ToDouble(negDist) > NumOps.ToDouble(maxNegDist))
                {
                    maxNegDist = negDist;
                }
            }

            // Compute exp and sum
            var expValues = new T[numClasses];
            T sumExp = NumOps.Zero;

            for (int c = 0; c < numClasses; c++)
            {
                T negDist = NumOps.Negate(distances[q, c]);
                T shifted = NumOps.Subtract(negDist, maxNegDist);
                T scaled = NumOps.Divide(shifted, _temperature);
                expValues[c] = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(scaled)));
                sumExp = NumOps.Add(sumExp, expValues[c]);
            }

            // Normalize
            for (int c = 0; c < numClasses; c++)
            {
                probabilities[q, c] = NumOps.Divide(expValues[c], sumExp);
            }
        }

        return probabilities;
    }

    /// <summary>
    /// Computes cross-entropy loss.
    /// </summary>
    private T ComputeCrossEntropyLoss(Matrix<T> probabilities, TOutput trueLabels)
    {
        T totalLoss = NumOps.Zero;
        int numQueries = probabilities.Rows;

        for (int i = 0; i < numQueries; i++)
        {
            int trueClass = GetClassLabel(trueLabels, i);

            if (trueClass >= 0 && trueClass < probabilities.Columns)
            {
                T predictedProb = probabilities[i, trueClass];
                predictedProb = NumOps.Add(predictedProb, NumOps.FromDouble(1e-8));
                T logProb = NumOps.FromDouble(Math.Log(NumOps.ToDouble(predictedProb)));
                totalLoss = NumOps.Subtract(totalLoss, logProb);
            }
        }

        return numQueries > 0 ? NumOps.Divide(totalLoss, NumOps.FromDouble(numQueries)) : NumOps.Zero;
    }

    /// <summary>
    /// Computes auxiliary co-training loss.
    /// </summary>
    private T ComputeAuxiliaryLoss(IMetaLearningTask<T, TInput, TOutput> task)
    {
        // Simplified auxiliary loss: classification on support set
        var supportFeatures = EncodeExamples(task.SupportInput);
        var prototypes = GroupAndComputePrototypes(supportFeatures, task.SupportOutput);
        var distances = ComputeScaledDistances(supportFeatures, prototypes);
        var probs = ApplySoftmaxWithTemperature(distances);
        return ComputeCrossEntropyLoss(probs, task.SupportOutput);
    }

    /// <summary>
    /// Adds L2 regularization to the loss.
    /// </summary>
    private T AddL2Regularization(T baseLoss)
    {
        T regLoss = NumOps.Zero;

        // Regularize metric scaling parameters
        for (int i = 0; i < _metricScale.Length; i++)
        {
            T val = _metricScale[i];
            regLoss = NumOps.Add(regLoss, NumOps.Multiply(val, val));
        }

        // Regularize MetaModel parameters (feature encoder)
        if (MetaModel != null)
        {
            var modelParams = MetaModel.GetParameters();
            for (int i = 0; i < modelParams.Length; i++)
            {
                T val = modelParams[i];
                regLoss = NumOps.Add(regLoss, NumOps.Multiply(val, val));
            }
        }

        T regWeight = NumOps.FromDouble(_tadamOptions.L2Regularization);
        return NumOps.Add(baseLoss, NumOps.Multiply(regWeight, regLoss));
    }

    /// <summary>
    /// Normalizes a tensor to unit length.
    /// </summary>
    private Tensor<T> NormalizeTensor(Tensor<T> tensor)
    {
        int size = 1;
        for (int i = 0; i < tensor.Shape.Length; i++)
        {
            size *= tensor.Shape[i];
        }

        T sumSquares = NumOps.Zero;
        for (int i = 0; i < size; i++)
        {
            T val = tensor.GetFlat(i);
            sumSquares = NumOps.Add(sumSquares, NumOps.Multiply(val, val));
        }

        double norm = Math.Sqrt(NumOps.ToDouble(sumSquares));
        if (norm < 1e-8) return tensor;

        var normalized = new Tensor<T>(tensor.Shape);
        for (int i = 0; i < size; i++)
        {
            normalized.SetFlat(i, NumOps.Divide(tensor.GetFlat(i), NumOps.FromDouble(norm)));
        }

        return normalized;
    }

    /// <summary>
    /// Gets class label from output at specified index.
    /// </summary>
    private int GetClassLabel(TOutput output, int index)
    {
        if (output is Tensor<T> tensor)
        {
            if (tensor.Shape.Length == 1)
            {
                return (int)NumOps.ToDouble(tensor[index]);
            }
            else if (tensor.Shape.Length >= 2)
            {
                int numClasses = tensor.Shape[1];
                int maxClass = 0;
                T maxProb = tensor.GetFlat(index * numClasses);

                for (int i = 1; i < numClasses; i++)
                {
                    var prob = tensor.GetFlat(index * numClasses + i);
                    if (NumOps.ToDouble(prob) > NumOps.ToDouble(maxProb))
                    {
                        maxProb = prob;
                        maxClass = i;
                    }
                }
                return maxClass;
            }
        }

        return 0;
    }
}
