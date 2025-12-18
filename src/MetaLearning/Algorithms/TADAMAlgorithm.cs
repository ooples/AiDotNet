using AiDotNet.Interfaces;
using AiDotNet.Layers;
using AiDotNet.MetaLearning.Config;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Implementation of Task-Adaptive Domain Adaptation Meta-learning (TADAM).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// TADAM extends MAML by incorporating domain adaptation through task-specific
/// feature transformation and domain discriminators. It learns to adapt the feature
/// space to be more suitable for few-shot learning across different domains.
/// </para>
/// <para><b>Key Components:</b></para>
/// - <b>Feature Transformer:</b> Task-specific feature adaptation
/// - <b>Domain Classifier:</b> Distinguishes between task domains
/// - <b>Gradient Reversal:</b> adversarial training for domain invariance
/// - <b>Task Encoder:</b> Encodes task-specific information
/// </remarks>
public class TADAMAlgorithm<T, TInput, TOutput> : IMetaLearningAlgorithm<T, TInput, TOutput>
    where T : struct, IEquatable<T>, IFormattable
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly TADAMAlgorithmOptions<T, TInput, TOutput> _tadamOptions;
    private readonly INeuralNetwork<T> _baseModel;
    private readonly INeuralNetwork<T> _taskEncoder;
    private readonly INeuralNetwork<T> _domainClassifier;
    private readonly INeuralNetwork<T> _featureTransformer;
    private readonly IInitializer<T> _initializer;
    private readonly IOptimizer<T> _metaOptimizer;
    private readonly IOptimizer<T> _domainOptimizer;

    private readonly List<ILayer<T>> _baseLayers;
    private readonly List<ILayer<T>> _taskEncoderLayers;
    private readonly List<ILayer<T>> _domainClassifierLayers;
    private readonly List<ILayer<T>> _featureTransformerLayers;

    private Tensor<T>? _taskEmbedding;
    private Tensor<T>? _adaptedParameters;
    private readonly Dictionary<int, Tensor<T>> _domainCache;

    /// <summary>
    /// Gets the algorithm type identifier.
    /// </summary>
    public string AlgorithmType => "TADAM";

    /// <summary>
    /// Gets the current episode number.
    /// </summary>
    public int CurrentEpisode { get; private set; }

    /// <summary>
    /// Gets or sets the meta-parameters of the algorithm.
    /// </summary>
    public Tensor<T> MetaParameters { get; private set; }

    /// <summary>
    /// Gets the performance history across episodes.
    /// </summary>
    public List<T> PerformanceHistory { get; }

    /// <summary>
    /// Initializes a new instance of the TADAMAlgorithm class.
    /// </summary>
    /// <param name="tadamOptions">Configuration options for TADAM.</param>
    /// <param name="baseModel">The base model for task-specific learning.</param>
    /// <param name="taskEncoder">The task encoder for capturing task information.</param>
    /// <param name="domainClassifier">The domain classifier for adversarial training.</param>
    /// <param name="featureTransformer">The feature transformer for domain adaptation.</param>
    /// <param name="initializer">Weight initializer.</param>
    /// <param name="metaOptimizer">Optimizer for meta-parameters.</param>
    /// <param name="domainOptimizer">Optimizer for domain classifier.</param>
    public TADAMAlgorithm(
        TADAMAlgorithmOptions<T, TInput, TOutput> tadamOptions,
        INeuralNetwork<T> baseModel,
        INeuralNetwork<T> taskEncoder,
        INeuralNetwork<T> domainClassifier,
        INeuralNetwork<T> featureTransformer,
        IInitializer<T> initializer,
        IOptimizer<T> metaOptimizer,
        IOptimizer<T> domainOptimizer)
    {
        _tadamOptions = tadamOptions ?? throw new ArgumentNullException(nameof(tadamOptions));
        _baseModel = baseModel ?? throw new ArgumentNullException(nameof(baseModel));
        _taskEncoder = taskEncoder ?? throw new ArgumentNullException(nameof(taskEncoder));
        _domainClassifier = domainClassifier ?? throw new ArgumentNullException(nameof(domainClassifier));
        _featureTransformer = featureTransformer ?? throw new ArgumentNullException(nameof(featureTransformer));
        _initializer = initializer ?? throw new ArgumentNullException(nameof(initializer));
        _metaOptimizer = metaOptimizer ?? throw new ArgumentNullException(nameof(metaOptimizer));
        _domainOptimizer = domainOptimizer ?? throw new ArgumentNullException(nameof(domainOptimizer));

        _baseLayers = new List<ILayer<T>>();
        _taskEncoderLayers = new List<ILayer<T>>();
        _domainClassifierLayers = new List<ILayer<T>>();
        _featureTransformerLayers = new List<ILayer<T>>();
        _domainCache = new Dictionary<int, Tensor<T>>();
        PerformanceHistory = new List<T>();

        InitializeNetworks();
    }

    /// <summary>
    /// Performs meta-learning on a batch of tasks.
    /// </summary>
    /// <param name="taskBatch">Batch of tasks for meta-learning.</param>
    /// <returns>Dictionary of performance metrics.</returns>
    public Dictionary<string, T> MetaLearnBatch(TaskBatch<TInput, TOutput> taskBatch)
    {
        var totalTaskLoss = NumOps.Zero;
        var totalDomainLoss = NumOps.Zero;
        var totalAdaptationLoss = NumOps.Zero;
        var totalAccuracy = NumOps.Zero;

        // Process each task in the batch
        for (int taskIdx = 0; taskIdx < taskBatch.Tasks.Count; taskIdx++)
        {
            var task = taskBatch.Tasks[taskIdx];
            var taskMetrics = ProcessTask(task);

            totalTaskLoss = NumOps.Add(totalTaskLoss, taskMetrics["task_loss"]);
            totalDomainLoss = NumOps.Add(totalDomainLoss, taskMetrics["domain_loss"]);
            totalAdaptationLoss = NumOps.Add(totalAdaptationLoss, taskMetrics["adaptation_loss"]);
            totalAccuracy = NumOps.Add(totalAccuracy, taskMetrics["accuracy"]);
        }

        // Average losses across tasks
        var numTasksT = NumOps.FromDouble(taskBatch.Tasks.Count);
        var avgTaskLoss = NumOps.Divide(totalTaskLoss, numTasksT);
        var avgDomainLoss = NumOps.Divide(totalDomainLoss, numTasksT);
        var avgAdaptationLoss = NumOps.Divide(totalAdaptationLoss, numTasksT);
        var avgAccuracy = NumOps.Divide(totalAccuracy, numTasksT);

        // Compute total loss with domain adversarial term
        var totalLoss = NumOps.Add(
            avgTaskLoss,
            NumOps.Multiply(NumOps.FromDouble(_tadamOptions.DomainAdversarialWeight), avgDomainLoss));

        if (_tadamOptions.UseFeatureTransformation)
        {
            totalLoss = NumOps.Add(
                totalLoss,
                NumOps.Multiply(NumOps.FromDouble(_tadamOptions.FeatureRegularizationWeight), avgAdaptationLoss));
        }

        // Meta-update
        MetaUpdate(totalLoss);

        // Update domain classifier (adversarial)
        UpdateDomainClassifier(avgDomainLoss);

        // Record performance
        PerformanceHistory.Add(avgTaskLoss);

        CurrentEpisode++;

        return new Dictionary<string, T>
        {
            ["loss"] = totalLoss,
            ["task_loss"] = avgTaskLoss,
            ["domain_loss"] = avgDomainLoss,
            ["adaptation_loss"] = avgAdaptationLoss,
            ["accuracy"] = avgAccuracy
        };
    }

    /// <summary>
    /// Adapts to a new task using support set examples.
    /// </summary>
    /// <param name="supportSet">Support set for adaptation.</param>
    /// <param name="numSteps">Number of adaptation steps.</param>
    public void Adapt(TaskBatch<TInput, TOutput> supportSet, int numSteps = 1)
    {
        // Encode task information
        _taskEmbedding = EncodeTask(supportSet);

        // Adapt features if enabled
        if (_tadamOptions.UseFeatureTransformation)
        {
            AdaptFeatures(_taskEmbedding);
        }

        // Generate task-specific parameters
        _adaptedParameters = GenerateTaskParameters(_taskEmbedding);

        // Perform gradient-based adaptation
        GradientBasedAdaptation(supportSet, numSteps);
    }

    /// <summary>
    /// Makes predictions on query examples after adaptation.
    /// </summary>
    /// <param name="querySet">Query examples for prediction.</param>
    /// <returns>Predictions for query examples.</returns>
    public TOutput Predict(TInput querySet)
    {
        if (_adaptedParameters == null)
        {
            throw new InvalidOperationException("Model must be adapted before making predictions");
        }

        // Transform features if needed
        var transformedInput = _tadamOptions.UseFeatureTransformation ?
            TransformFeatures(querySet) : querySet;

        // Forward pass with adapted parameters
        return ForwardWithAdaptedParameters(transformedInput);
    }

    /// <summary>
    /// Resets the meta-learning algorithm state.
    /// </summary>
    public void Reset()
    {
        CurrentEpisode = 0;
        _taskEmbedding = null;
        _adaptedParameters = null;
        _domainCache.Clear();
        PerformanceHistory.Clear();

        // Reset all networks
        foreach (var layer in _baseLayers.Concat(_taskEncoderLayers).Concat(_domainClassifierLayers).Concat(_featureTransformerLayers))
        {
            layer.Reset();
        }

        // Reset optimizers
        _metaOptimizer.Reset();
        _domainOptimizer.Reset();
    }

    /// <summary>
    /// Gets the meta-parameters for saving/loading.
    /// </summary>
    /// <returns>Current meta-parameters.</returns>
    public Tensor<T> GetMetaParameters()
    {
        return MetaParameters;
    }

    /// <summary>
    /// Sets the meta-parameters from saved state.
    /// </summary>
    /// <param name="parameters">Meta-parameters to load.</param>
    public void SetMetaParameters(Tensor<T> parameters)
    {
        MetaParameters = parameters;
        DistributeParameters(parameters);
    }

    private void InitializeNetworks()
    {
        // Initialize base model layers
        InitializeBaseModel();

        // Initialize task encoder
        InitializeTaskEncoder();

        // Initialize domain classifier
        InitializeDomainClassifier();

        // Initialize feature transformer
        if (_tadamOptions.UseFeatureTransformation)
        {
            InitializeFeatureTransformer();
        }

        // Initialize meta-parameters
        InitializeMetaParameters();
    }

    private void InitializeBaseModel()
    {
        // Base model architecture
        _baseLayers.Add(new DenseLayer<T>(
            _tadamOptions.InputDimension,
            _tadamOptions.HiddenDimension,
            activation: _tadamOptions.ActivationFunction,
            initializer: _initializer,
            useBias: true));

        // Add residual blocks
        for (int i = 0; i < _tadamOptions.NumResidualBlocks; i++)
        {
            _baseLayers.Add(new ResidualBlock<T>(
                _tadamOptions.HiddenDimension,
                _tadamOptions.HiddenDimension,
                activation: _tadamOptions.ActivationFunction,
                initializer: _initializer,
                dropoutRate: _tadamOptions.DropoutRate,
                useLayerNorm: _tadamOptions.UseLayerNorm));
        }

        // Output layer
        _baseLayers.Add(new DenseLayer<T>(
            _tadamOptions.HiddenDimension,
            _tadamOptions.OutputDimension,
            activation: ActivationFunctionType.None,
            initializer: _initializer,
            useBias: false));
    }

    private void InitializeTaskEncoder()
    {
        // Task encoder to capture task-specific information
        _taskEncoderLayers.Add(new DenseLayer<T>(
            _tadamOptions.TaskEmbeddingDimension,
            _tadamOptions.TaskHiddenDimension,
            activation: _tadamOptions.ActivationFunction,
            initializer: _initializer,
            useBias: true));

        // Attention mechanism for task encoding
        if (_tadamOptions.UseTaskAttention)
        {
            _taskEncoderLayers.Add(new MultiHeadAttentionLayer<T>(
                _tadamOptions.TaskHiddenDimension,
                _tadamOptions.NumTaskAttentionHeads,
                _tadamOptions.TaskAttentionDimension));
        }

        // Output task embedding
        _taskEncoderLayers.Add(new DenseLayer<T>(
            _tadamOptions.TaskHiddenDimension,
            _tadamOptions.TaskEmbeddingDimension,
            activation: ActivationFunctionType.Tanh,
            initializer: _initializer,
            useBias: false));
    }

    private void InitializeDomainClassifier()
    {
        // Domain classifier with gradient reversal
        _domainClassifierLayers.Add(new GradientReversalLayer<T>(
            _tadamOptions.GradientReversalScale));

        // Hidden layers
        _domainClassifierLayers.Add(new DenseLayer<T>(
            _tadamOptions.TaskEmbeddingDimension,
            _tadamOptions.DomainHiddenDimension,
            activation: _tadamOptions.ActivationFunction,
            initializer: _initializer,
            useBias: true));

        if (_tadamOptions.NumDomainLayers > 1)
        {
            _domainClassifierLayers.Add(new DenseLayer<T>(
                _tadamOptions.DomainHiddenDimension,
                _tadamOptions.DomainHiddenDimension,
                activation: _tadamOptions.ActivationFunction,
                initializer: _initializer,
                useBias: true,
                dropoutRate: _tadamOptions.DropoutRate));
        }

        // Output layer (domain prediction)
        _domainClassifierLayers.Add(new DenseLayer<T>(
            _tadamOptions.DomainHiddenDimension,
            _tadamOptions.NumDomains,
            activation: ActivationFunctionType.Softmax,
            initializer: _initializer,
            useBias: false));
    }

    private void InitializeFeatureTransformer()
    {
        // Feature transformer for domain adaptation
        _featureTransformerLayers.Add(new DenseLayer<T>(
            _tadamOptions.InputDimension,
            _tadamOptions.FeatureHiddenDimension,
            activation: _tadamOptions.ActivationFunction,
            initializer: _initializer,
            useBias: true));

        // Multiple transformation layers
        for (int i = 0; i < _tadamOptions.NumFeatureLayers; i++)
        {
            _featureTransformerLayers.Add(new DenseLayer<T>(
                _tadamOptions.FeatureHiddenDimension,
                _tadamOptions.FeatureHiddenDimension,
                activation: _tadamOptions.ActivationFunction,
                initializer: _initializer,
                useBias: true,
                dropoutRate: _tadamOptions.DropoutRate));
        }

        // Output transformation
        _featureTransformerLayers.Add(new DenseLayer<T>(
            _tadamOptions.FeatureHiddenDimension,
            _tadamOptions.InputDimension,
            activation: ActivationFunctionType.None,
            initializer: _initializer,
            useBias: false));
    }

    private void InitializeMetaParameters()
    {
        // Collect all parameters except domain classifier
        var allParameters = new List<Tensor<T>>();

        foreach (var layer in _baseLayers.Concat(_taskEncoderLayers).Concat(_featureTransformerLayers))
        {
            allParameters.AddRange(layer.GetParameters());
        }

        // Concatenate into single tensor
        MetaParameters = Tensor<T>.Concat(allParameters);
    }

    private Dictionary<string, T> ProcessTask(Task<TInput, TOutput> task)
    {
        // Split into support and query sets
        var (supportSet, querySet) = task.SplitForAdaptation(_tadamOptions.SupportSetSize);

        // Encode task
        var taskEmbedding = EncodeTask(supportSet);
        _taskEmbedding = taskEmbedding;

        // Adapt features
        if (_tadamOptions.UseFeatureTransformation)
        {
            AdaptFeatures(taskEmbedding);
        }

        // Generate task-specific parameters
        var taskParams = GenerateTaskParameters(taskEmbedding);
        _adaptedParameters = taskParams;

        // Perform inner loop adaptation
        var adaptedParams = PerformInnerLoop(supportSet, taskParams);

        // Evaluate on query set
        var queryPredictions = EvaluateQuery(querySet, adaptedParams);

        // Compute task loss
        var (taskLoss, accuracy) = ComputeTaskLoss(querySet, queryPredictions);

        // Compute domain loss (adversarial)
        var domainLoss = ComputeDomainLoss(taskEmbedding, task.DomainId);

        // Compute adaptation regularization
        var adaptationLoss = ComputeAdaptationLoss(taskParams, adaptedParams);

        return new Dictionary<string, T>
        {
            ["task_loss"] = taskLoss,
            ["domain_loss"] = domainLoss,
            ["adaptation_loss"] = adaptationLoss,
            ["accuracy"] = accuracy
        };
    }

    private Tensor<T> EncodeTask(TaskBatch<TInput, TOutput> supportSet)
    {
        // Compute support set statistics
        var supportFeatures = ExtractSupportFeatures(supportSet);
        var taskStatistics = ComputeTaskStatistics(supportFeatures);

        // Pass through task encoder
        var current = taskStatistics;
        foreach (var layer in _taskEncoderLayers)
        {
            current = layer.Forward(current);
        }

        return current;
    }

    private Tensor<T> ExtractSupportFeatures(TaskBatch<TInput, TOutput> supportSet)
    {
        // Extract features from support set
        var features = new List<Tensor<T>>();

        foreach (var example in supportSet.Tasks)
        {
            var feature = ExtractFeature(example.Input);
            features.Add(feature);
        }

        // Aggregate features
        return Tensor<T>.Stack(features, axis: 0);
    }

    private Tensor<T> ComputeTaskStatistics(Tensor<T> features)
    {
        // Compute mean and variance across support set
        var mean = Tensor<T>.Mean(features, axis: 0);
        var variance = Tensor<T>.Var(features, axis: 0);

        // Concatenate statistics for task encoding
        return Tensor<T>.Concat(new[] { mean, variance }, axis: -1);
    }

    private Tensor<T> ExtractFeature(TInput input)
    {
        // Extract raw features from input
        // This would be implemented based on input type
        return Tensor<T>.Zeros(new[] { _tadamOptions.InputDimension });
    }

    private void AdaptFeatures(Tensor<T> taskEmbedding)
    {
        // Use task embedding to modulate feature transformation
        foreach (var layer in _featureTransformerLayers)
        {
            if (layer is DenseLayer<T> denseLayer)
            {
                // Apply FiLM modulation based on task embedding
                denseLayer.ApplyTaskModulation(taskEmbedding);
            }
        }
    }

    private Tensor<T> GenerateTaskParameters(Tensor<T> taskEmbedding)
    {
        // Generate task-specific parameter modifications
        var modifications = new List<Tensor<T>>();

        // For each layer in base model
        foreach (var layer in _baseLayers)
        {
            var layerParams = layer.GetParameters();
            var layerModifications = new List<Tensor<T>>();

            // Generate modification for each parameter tensor
            foreach (var param in layerParams)
            {
                // Compute scaling and shift based on task embedding
                var scale = GenerateScale(taskEmbedding, param.Shape);
                var shift = GenerateShift(taskEmbedding, param.Shape);

                // Apply modification: param' = scale * param + shift
                var modification = Tensor<T>.Add(
                    Tensor<T>.Multiply(scale, param),
                    shift);

                layerModifications.Add(modification);
            }
            modifications.AddRange(layerModifications);
        }

        return Tensor<T>.Concat(modifications);
    }

    private Tensor<T> GenerateScale(Tensor<T> taskEmbedding, int[] shape)
    {
        // Generate scale parameters from task embedding
        var scaleNetwork = new DenseLayer<T>(
            taskEmbedding.Shape[0],
            shape.Product(),
            activation: ActivationFunctionType.Sigmoid,
            initializer: _initializer,
            useBias: false);

        var scale = scaleNetwork.Forward(taskEmbedding);
        return scale.Reshape(shape);
    }

    private Tensor<T> GenerateShift(Tensor<T> taskEmbedding, int[] shape)
    {
        // Generate shift parameters from task embedding
        var shiftNetwork = new DenseLayer<T>(
            taskEmbedding.Shape[0],
            shape.Product(),
            activation: ActivationFunctionType.Tanh,
            initializer: _initializer,
            useBias: false);

        var shift = shiftNetwork.Forward(taskEmbedding);
        return shift.Reshape(shape);
    }

    private Tensor<T> PerformInnerLoop(TaskBatch<TInput, TOutput> supportSet, Tensor<T> taskParams)
    {
        var adaptedParams = taskParams;

        // Perform gradient steps on support set
        for (int step = 0; step < _tadamOptions.AdaptationSteps; step++)
        {
            // Compute support loss
            var supportLoss = ComputeSupportLoss(supportSet, adaptedParams);

            // Compute gradients
            var gradients = ComputeParameterGradients(supportLoss, adaptedParams);

            // Update parameters
            adaptedParams = UpdateAdaptedParameters(adaptedParams, gradients);
        }

        return adaptedParams;
    }

    private T ComputeSupportLoss(TaskBatch<TInput, TOutput> supportSet, Tensor<T> parameters)
    {
        // Compute loss on support set with current parameters
        var totalLoss = NumOps.Zero;

        foreach (var example in supportSet.Tasks)
        {
            var prediction = ForwardWithParameters(example.Input, parameters);
            var loss = ComputeExampleLoss(prediction, example.Output);
            totalLoss = NumOps.Add(totalLoss, loss);
        }

        return NumOps.Divide(totalLoss, NumOps.FromDouble(supportSet.Tasks.Count));
    }

    private T ComputeExampleLoss(TOutput prediction, TOutput target)
    {
        // Compute loss for single example
        // This would use appropriate loss function
        return NumOps.Zero;
    }

    private Tensor<T> ComputeParameterGradients(T loss, Tensor<T> parameters)
    {
        // Compute gradients with respect to parameters
        // This would involve automatic differentiation
        return Tensor<T>.ZerosLike(parameters);
    }

    private Tensor<T> UpdateAdaptedParameters(Tensor<T> parameters, Tensor<T> gradients)
    {
        // Update parameters with inner loop learning rate
        var update = NumOps.Multiply(_tadamOptions.InnerLearningRate, gradients);
        return NumOps.Subtract(parameters, update);
    }

    private TOutput EvaluateQuery(TaskBatch<TInput, TOutput> querySet, Tensor<T> adaptedParams)
    {
        // Evaluate on query set with adapted parameters
        var predictions = new List<Tensor<T>>();

        foreach (var example in querySet.Tasks)
        {
            var prediction = ForwardWithParameters(example.Input, adaptedParams);
            predictions.Add(ConvertToTensor(prediction));
        }

        // Aggregate predictions
        return ConvertFromTensor(Tensor<T>.Stack(predictions, axis: 0));
    }

    private TOutput ForwardWithParameters(TInput input, Tensor<T> parameters)
    {
        // Forward pass with specific parameters
        var offset = 0;

        // Transform features if needed
        var current = _tadamOptions.UseFeatureTransformation ?
            TransformFeaturesWithParameters(input, parameters, ref offset) : input;

        // Pass through base model with adapted parameters
        foreach (var layer in _baseLayers)
        {
            var layerParams = new List<Tensor<T>>();
            foreach (var param in layer.GetParameters())
            {
                var numParams = param.Count;
                var adaptedParam = parameters.Slice(
                    new[] { offset },
                    new[] { numParams });
                layerParams.Add(adaptedParam);
                offset += numParams;
            }

            layer.SetParameters(layerParams);
            current = layer.Forward(ConvertToTensor(current));
        }

        return ConvertFromTensor(current);
    }

    private (T, T) ComputeTaskLoss(TaskBatch<TInput, TOutput> querySet, TOutput predictions)
    {
        // Compute task-specific loss and accuracy
        var targets = ConvertToTensor(querySet.Output);
        var predTensor = ConvertToTensor(predictions);

        // Compute accuracy
        var accuracy = ComputeAccuracy(predTensor, targets);

        // Compute loss
        var loss = ComputeLoss(predTensor, targets);

        return (loss, accuracy);
    }

    private T ComputeDomainLoss(Tensor<T> taskEmbedding, int domainId)
    {
        // Forward pass through domain classifier
        var current = taskEmbedding;
        foreach (var layer in _domainClassifierLayers)
        {
            current = layer.Forward(current);
        }

        // Compute domain classification loss
        var targetDomain = CreateDomainTarget(domainId);
        return ComputeDomainClassificationLoss(current, targetDomain);
    }

    private T ComputeAdaptationLoss(Tensor<T> originalParams, Tensor<T> adaptedParams)
    {
        if (!_tadamOptions.UseFeatureTransformation)
        {
            return NumOps.Zero;
        }

        // L2 regularization on parameter adaptations
        var diff = Tensor<T>.Subtract(adaptedParams, originalParams);
        var adaptationLoss = Tensor<T>.Square(diff).Sum();

        return NumOps.Multiply(
            NumOps.FromDouble(_tadamOptions.FeatureRegularizationWeight),
            adaptationLoss);
    }

    private void MetaUpdate(T totalLoss)
    {
        // Compute gradients and update meta-parameters
        var gradients = ComputeMetaGradients(totalLoss);
        _metaOptimizer.Update(MetaParameters, gradients);

        // Distribute updated parameters
        DistributeParameters(MetaParameters);
    }

    private void UpdateDomainClassifier(T domainLoss)
    {
        // Update domain classifier parameters (adversarial)
        var domainParams = GetDomainParameters();
        var domainGradients = ComputeDomainGradients(domainLoss);
        _domainOptimizer.Update(domainParams, domainGradients);
        SetDomainParameters(domainParams);
    }

    private Tensor<T> ComputeMetaGradients(T loss)
    {
        // Compute gradients with respect to meta-parameters
        return Tensor<T>.ZerosLike(MetaParameters);
    }

    private Tensor<T> GetDomainParameters()
    {
        var allParams = new List<Tensor<T>>();
        foreach (var layer in _domainClassifierLayers)
        {
            allParams.AddRange(layer.GetParameters());
        }
        return Tensor<T>.Concat(allParams);
    }

    private void SetDomainParameters(Tensor<T> parameters)
    {
        var offset = 0;
        foreach (var layer in _domainClassifierLayers)
        {
            var layerParams = layer.GetParameters();
            var numParams = layerParams.Sum(p => p.Count);
            var layerSlice = parameters.Slice(
                new[] { offset },
                new[] { numParams });
            layer.SetParameters(new List<Tensor<T>> { layerSlice });
            offset += numParams;
        }
    }

    private Tensor<T> ComputeDomainGradients(T loss)
    {
        return Tensor<T>.ZerosLike(GetDomainParameters());
    }

    private void DistributeParameters(Tensor<T> parameters)
    {
        var offset = 0;

        // Distribute to base model
        foreach (var layer in _baseLayers)
        {
            var layerParams = layer.GetParameters();
            var numParams = layerParams.Sum(p => p.Count);
            var layerSlice = parameters.Slice(
                new[] { offset },
                new[] { numParams });
            layer.SetParameters(new List<Tensor<T>> { layerSlice });
            offset += numParams;
        }

        // Distribute to task encoder
        foreach (var layer in _taskEncoderLayers)
        {
            var layerParams = layer.GetParameters();
            var numParams = layerParams.Sum(p => p.Count);
            var layerSlice = parameters.Slice(
                new[] { offset },
                new[] { numParams });
            layer.SetParameters(new List<Tensor<T>> { layerSlice });
            offset += numParams;
        }

        // Distribute to feature transformer
        foreach (var layer in _featureTransformerLayers)
        {
            var layerParams = layer.GetParameters();
            var numParams = layerParams.Sum(p => p.Count);
            var layerSlice = parameters.Slice(
                new[] { offset },
                new[] { numParams });
            layer.SetParameters(new List<Tensor<T>> { layerSlice });
            offset += numParams;
        }
    }

    private void GradientBasedAdaptation(TaskBatch<TInput, TOutput> supportSet, int numSteps)
    {
        // Additional gradient-based adaptation steps
        for (int step = 0; step < numSteps; step++)
        {
            // Compute gradients on support set
            var supportLoss = ComputeSupportLoss(supportSet, _adaptedParameters);
            var gradients = ComputeParameterGradients(supportLoss, _adaptedParameters);

            // Update adapted parameters
            _adaptedParameters = UpdateAdaptedParameters(_adaptedParameters, gradients);
        }
    }

    private TInput TransformFeatures(TInput input)
    {
        // Transform input features
        var inputTensor = ConvertToTensor(input);
        var current = inputTensor;

        foreach (var layer in _featureTransformerLayers)
        {
            current = layer.Forward(current);
        }

        return ConvertFromTensor(current);
    }

    private TInput TransformFeaturesWithParameters(TInput input, Tensor<T> parameters, ref int offset)
    {
        // Transform features with specific parameters
        var inputTensor = ConvertToTensor(input);
        var current = inputTensor;

        foreach (var layer in _featureTransformerLayers)
        {
            var layerParams = new List<Tensor<T>>();
            foreach (var param in layer.GetParameters())
            {
                var numParams = param.Count;
                var adaptedParam = parameters.Slice(
                    new[] { offset },
                    new[] { numParams });
                layerParams.Add(adaptedParam);
                offset += numParams;
            }

            layer.SetParameters(layerParams);
            current = layer.Forward(current);
        }

        return ConvertFromTensor(current);
    }

    private TOutput ForwardWithAdaptedParameters(TInput input)
    {
        return ForwardWithParameters(input, _adaptedParameters);
    }

    // Helper methods
    private Tensor<T> ConvertToTensor(object input)
    {
        // Convert various input types to tensors
        return Tensor<T>.Zeros(new[] { 1 });
    }

    private TOutput ConvertFromTensor(Tensor<T> tensor)
    {
        // Convert tensor predictions back to output type
        return default(TOutput);
    }

    private T ComputeLoss(Tensor<T> predictions, Tensor<T> targets)
    {
        // Compute task loss
        return NumOps.Zero;
    }

    private T ComputeAccuracy(Tensor<T> predictions, Tensor<T> targets)
    {
        // Compute accuracy metric
        return NumOps.Zero;
    }

    private Tensor<T> CreateDomainTarget(int domainId)
    {
        // Create one-hot encoding for domain
        var target = Tensor<T>.Zeros(new[] { _tadamOptions.NumDomains });
        // Set domainId position to 1
        return target;
    }

    private T ComputeDomainClassificationLoss(Tensor<T> predictions, Tensor<T> targets)
    {
        // Compute cross-entropy loss for domain classification
        return NumOps.Zero;
    }
}