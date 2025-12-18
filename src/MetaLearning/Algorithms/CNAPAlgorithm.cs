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
/// Implementation of Conditional Neural Adaptive Processes (CNAPs) for meta-learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector&lt;T&gt;, Tensor&lt;T&gt;).</typeparam>
/// <remarks>
/// <para>
/// CNAPs extend Neural Processes by conditioning on task-specific context points
/// and learning to adapt to new tasks with a single gradient step. The model learns
/// to produce fast adaptation weights for each task.
/// </para>
/// <para><b>Key Components:</b></para>
/// - <b>Encoder:</b> Processes context points into task representations
/// - <b>Decoder:</b> Generates predictions conditioned on representations
/// - <b>Adaptation Network:</b> Generates task-specific fast weights
/// - <b>Aggregator:</b> Combines information from multiple context points
/// </remarks>
public class CNAPAlgorithm<T, TInput, TOutput> : IMetaLearningAlgorithm<T, TInput, TOutput>
    where T : struct, IEquatable<T>, IFormattable
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly CNAPAlgorithmOptions<T, TInput, TOutput> _cnapOptions;
    private readonly INeuralNetwork<T> _encoder;
    private readonly INeuralNetwork<T> _decoder;
    private readonly INeuralNetwork<T> _adaptationNetwork;
    private readonly INeuralNetwork<T> _aggregator;
    private readonly IInitializer<T> _initializer;
    private readonly List<ILayer<T>> _encoderLayers;
    private readonly List<ILayer<T>> _decoderLayers;
    private readonly List<ILayer<T>> _adaptationLayers;
    private readonly List<ILayer<T>> _aggregatorLayers;
    private readonly IOptimizer<T> _metaOptimizer;

    private Tensor<T>? _fastWeights;
    private Tensor<T>? _taskRepresentation;
    private readonly Dictionary<int, Tensor<T>> _taskCache;

    /// <summary>
    /// Gets the algorithm type identifier.
    /// </summary>
    public string AlgorithmType => "CNAP";

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
    /// Initializes a new instance of the CNAPAlgorithm class.
    /// </summary>
    /// <param name="cnapOptions">Configuration options for CNAP.</param>
    /// <param name="encoder">The encoder network for processing context points.</param>
    /// <param name="decoder">The decoder network for generating predictions.</param>
    /// <param name="adaptationNetwork">The adaptation network for fast weights.</param>
    /// <param name="aggregator">The aggregator network for combining context information.</param>
    /// <param name="initializer">Weight initializer.</param>
    /// <param name="metaOptimizer">Optimizer for meta-parameters.</param>
    public CNAPAlgorithm(
        CNAPAlgorithmOptions<T, TInput, TOutput> cnapOptions,
        INeuralNetwork<T> encoder,
        INeuralNetwork<T> decoder,
        INeuralNetwork<T> adaptationNetwork,
        INeuralNetwork<T> aggregator,
        IInitializer<T> initializer,
        IOptimizer<T> metaOptimizer)
    {
        _cnapOptions = cnapOptions ?? throw new ArgumentNullException(nameof(cnapOptions));
        _encoder = encoder ?? throw new ArgumentNullException(nameof(encoder));
        _decoder = decoder ?? throw new ArgumentNullException(nameof(decoder));
        _adaptationNetwork = adaptationNetwork ?? throw new ArgumentNullException(nameof(adaptationNetwork));
        _aggregator = aggregator ?? throw new ArgumentNullException(nameof(aggregator));
        _initializer = initializer ?? throw new ArgumentNullException(nameof(initializer));
        _metaOptimizer = metaOptimizer ?? throw new ArgumentNullException(nameof(metaOptimizer));

        _encoderLayers = new List<ILayer<T>>();
        _decoderLayers = new List<ILayer<T>>();
        _adaptationLayers = new List<ILayer<T>>();
        _aggregatorLayers = new List<ILayer<T>>();
        _taskCache = new Dictionary<int, Tensor<T>>();
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
        var totalLoss = NumOps.Zero;
        var totalAccuracy = NumOps.Zero;
        var totalAdaptationLoss = NumOps.Zero;
        var totalUncertaintyLoss = NumOps.Zero;

        // Process each task in the batch
        for (int taskIdx = 0; taskIdx < taskBatch.Tasks.Count; taskIdx++)
        {
            var task = taskBatch.Tasks[taskIdx];
            var taskMetrics = ProcessTask(task);

            totalLoss = NumOps.Add(totalLoss, taskMetrics["loss"]);
            totalAccuracy = NumOps.Add(totalAccuracy, taskMetrics["accuracy"]);
            totalAdaptationLoss = NumOps.Add(totalAdaptationLoss, taskMetrics["adaptation_loss"]);
            totalUncertaintyLoss = NumOps.Add(totalUncertaintyLoss, taskMetrics["uncertainty_loss"]);
        }

        // Average losses across tasks
        var numTasksT = NumOps.FromDouble(taskBatch.Tasks.Count);
        var avgLoss = NumOps.Divide(totalLoss, numTasksT);
        var avgAccuracy = NumOps.Divide(totalAccuracy, numTasksT);
        var avgAdaptationLoss = NumOps.Divide(totalAdaptationLoss, numTasksT);
        var avgUncertaintyLoss = NumOps.Divide(totalUncertaintyLoss, numTasksT);

        // Meta-update: backpropagate through task processing
        MetaUpdate(avgLoss);

        // Record performance
        PerformanceHistory.Add(avgLoss);

        CurrentEpisode++;

        return new Dictionary<string, T>
        {
            ["loss"] = avgLoss,
            ["accuracy"] = avgAccuracy,
            ["adaptation_loss"] = avgAdaptationLoss,
            ["uncertainty_loss"] = avgUncertaintyLoss
        };
    }

    /// <summary>
    /// Adapts to a new task using support set examples.
    /// </summary>
    /// <param name="supportSet">Support set for adaptation.</param>
    /// <param name="numSteps">Number of adaptation steps.</param>
    public void Adapt(TaskBatch<TInput, TOutput> supportSet, int numSteps = 1)
    {
        // Process support set to get task representation
        _taskRepresentation = EncodeTask(supportSet);

        // Generate fast weights for adaptation
        _fastWeights = GenerateFastWeights(_taskRepresentation);

        // Fine-tune with support set if needed
        for (int step = 0; step < numSteps; step++)
        {
            FineTuneOnSupport(supportSet);
        }
    }

    /// <summary>
    /// Makes predictions on query examples after adaptation.
    /// </summary>
    /// <param name="querySet">Query examples for prediction.</param>
    /// <returns>Predictions for query examples.</returns>
    public TOutput Predict(TInput querySet)
    {
        if (_fastWeights == null || _taskRepresentation == null)
        {
            throw new InvalidOperationException("Model must be adapted before making predictions");
        }

        // Encode query with task representation
        var queryEncoding = EncodeQuery(querySet, _taskRepresentation);

        // Decode to prediction using fast weights
        return DecodeWithFastWeights(queryEncoding, _fastWeights);
    }

    /// <summary>
    /// Resets the meta-learning algorithm state.
    /// </summary>
    public void Reset()
    {
        CurrentEpisode = 0;
        _fastWeights = null;
        _taskRepresentation = null;
        _taskCache.Clear();
        PerformanceHistory.Clear();

        // Reset all networks
        foreach (var layer in _encoderLayers.Concat(_decoderLayers).Concat(_adaptationLayers).Concat(_aggregatorLayers))
        {
            layer.Reset();
        }

        // Reset optimizer
        _metaOptimizer.Reset();
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

        // Distribute parameters to networks
        DistributeParameters(parameters);
    }

    private void InitializeNetworks()
    {
        // Initialize encoder layers
        InitializeEncoder();

        // Initialize decoder layers
        InitializeDecoder();

        // Initialize adaptation network
        InitializeAdaptationNetwork();

        // Initialize aggregator
        InitializeAggregator();

        // Initialize meta-parameters
        InitializeMetaParameters();
    }

    private void InitializeEncoder()
    {
        // Add input embedding layer
        _encoderLayers.Add(new DenseLayer<T>(
            _cnapOptions.InputDimension,
            _cnapOptions.HiddenDimension,
            activation: _cnapOptions.ActivationFunction,
            initializer: _initializer,
            useBias: true,
            inputSize: _cnapOptions.ContextSize * _cnapOptions.InputDimension));

        // Add attention mechanism for context points
        if (_cnapOptions.UseAttention)
        {
            _encoderLayers.Add(new MultiHeadAttentionLayer<T>(
                _cnapOptions.HiddenDimension,
                _cnapOptions.NumAttentionHeads,
                _cnapOptions.AttentionDimension));
        }

        // Add transformer blocks
        for (int i = 0; i < _cnapOptions.NumTransformerBlocks; i++)
        {
            _encoderLayers.Add(new TransformerBlock<T>(
                _cnapOptions.HiddenDimension,
                _cnapOptions.FeedForwardDimension,
                _cnapOptions.NumAttentionHeads,
                dropoutRate: _cnapOptions.DropoutRate,
                layerNorm: _cnapOptions.UseLayerNorm));
        }

        // Add output projection
        _encoderLayers.Add(new DenseLayer<T>(
            _cnapOptions.HiddenDimension,
            _cnapOptions.RepresentationDimension,
            activation: ActivationFunctionType.None,
            initializer: _initializer,
            useBias: false));
    }

    private void InitializeDecoder()
    {
        // Add input layer (query + representation)
        _decoderLayers.Add(new DenseLayer<T>(
            _cnapOptions.InputDimension + _cnapOptions.RepresentationDimension,
            _cnapOptions.HiddenDimension,
            activation: _cnapOptions.ActivationFunction,
            initializer: _initializer,
            useBias: true));

        // Add hidden layers
        for (int i = 0; i < _cnapOptions.NumDecoderLayers; i++)
        {
            _decoderLayers.Add(new DenseLayer<T>(
                _cnapOptions.HiddenDimension,
                _cnapOptions.HiddenDimension,
                activation: _cnapOptions.ActivationFunction,
                initializer: _initializer,
                useBias: true,
                dropoutRate: _cnapOptions.DropoutRate));
        }

        // Add uncertainty head
        if (_cnapOptions.PredictUncertainty)
        {
            _decoderLayers.Add(new DenseLayer<T>(
                _cnapOptions.HiddenDimension,
                2 * _cnapOptions.OutputDimension,  // mean and variance
                activation: ActivationFunctionType.None,
                initializer: _initializer,
                useBias: false));
        }
        else
        {
            _decoderLayers.Add(new DenseLayer<T>(
                _cnapOptions.HiddenDimension,
                _cnapOptions.OutputDimension,
                activation: ActivationFunctionType.None,
                initializer: _initializer,
                useBias: false));
        }
    }

    private void InitializeAdaptationNetwork()
    {
        // Input: task representation
        _adaptationLayers.Add(new DenseLayer<T>(
            _cnapOptions.RepresentationDimension,
            _cnapOptions.AdaptationHiddenDimension,
            activation: _cnapOptions.ActivationFunction,
            initializer: _initializer,
            useBias: true));

        // Hidden layers
        for (int i = 0; i < _cnapOptions.NumAdaptationLayers - 1; i++)
        {
            _adaptationLayers.Add(new DenseLayer<T>(
                _cnapOptions.AdaptationHiddenDimension,
                _cnapOptions.AdaptationHiddenDimension,
                activation: _cnapOptions.ActivationFunction,
                initializer: _initializer,
                useBias: true,
                dropoutRate: _cnapOptions.DropoutRate));
        }

        // Output: fast weights for each parameter
        var numParameters = _encoderLayers.Concat(_decoderLayers)
            .Sum(l => l.GetParameters().Count);

        _adaptationLayers.Add(new DenseLayer<T>(
            _cnapOptions.AdaptationHiddenDimension,
            numParameters,
            activation: ActivationFunctionType.None,
            initializer: _initializer,
            useBias: false));
    }

    private void InitializeAggregator()
    {
        if (!_cnapOptions.UseAttention)
        {
            // Simple mean pooling aggregator
            _aggregatorLayers.Add(new DenseLayer<T>(
                _cnapOptions.RepresentationDimension,
                _cnapOptions.RepresentationDimension,
                activation: _cnapOptions.ActivationFunction,
                initializer: _initializer,
                useBias: true));
        }
    }

    private void InitializeMetaParameters()
    {
        // Collect all parameters
        var allParameters = new List<Tensor<T>>();

        foreach (var layer in _encoderLayers.Concat(_decoderLayers).Concat(_adaptationLayers).Concat(_aggregatorLayers))
        {
            allParameters.AddRange(layer.GetParameters());
        }

        // Concatenate into single tensor
        MetaParameters = Tensor<T>.Concat(allParameters);
    }

    private Dictionary<string, T> ProcessTask(Task<TInput, TOutput> task)
    {
        // Split into support and query sets
        var (supportSet, querySet) = task.SplitForAdaptation(_cnapOptions.SupportSetSize);

        // Encode task from support set
        var taskRep = EncodeTask(supportSet);

        // Generate fast weights
        var fastWeights = GenerateFastWeights(taskRep);

        // Make predictions on query set
        var predictions = PredictOnQuery(querySet, taskRep, fastWeights);

        // Compute losses
        var (taskLoss, accuracy) = ComputeTaskLoss(querySet, predictions);
        var adaptationLoss = ComputeAdaptationLoss(fastWeights);
        var uncertaintyLoss = _cnapOptions.PredictUncertainty ?
            ComputeUncertaintyLoss(predictions) : NumOps.Zero;

        // Total loss
        var totalLoss = NumOps.Add(
            taskLoss,
            NumOps.Multiply(NumOps.FromDouble(_cnapOptions.AdaptationWeight), adaptationLoss));

        if (_cnapOptions.PredictUncertainty)
        {
            totalLoss = NumOps.Add(
                totalLoss,
                NumOps.Multiply(NumOps.FromDouble(_cnapOptions.UncertaintyWeight), uncertaintyLoss));
        }

        return new Dictionary<string, T>
        {
            ["loss"] = totalLoss,
            ["accuracy"] = accuracy,
            ["adaptation_loss"] = adaptationLoss,
            ["uncertainty_loss"] = uncertaintyLoss
        };
    }

    private Tensor<T> EncodeTask(TaskBatch<TInput, TOutput> supportSet)
    {
        // Encode each context point
        var contextEncodings = new List<Tensor<T>>();

        for (int i = 0; i < supportSet.Tasks.Count; i++)
        {
            var context = supportSet.Tasks[i];
            var encoding = EncodeContextPoint(context);
            contextEncodings.Add(encoding);
        }

        // Aggregate context encodings
        if (_cnapOptions.UseAttention)
        {
            return AggregateWithAttention(contextEncodings);
        }
        else
        {
            return AggregateWithPooling(contextEncodings);
        }
    }

    private Tensor<T> EncodeContextPoint(Task<TInput, TOutput> context)
    {
        // Combine input and output
        var inputTensor = ConvertToTensor(context.Input);
        var outputTensor = ConvertToTensor(context.Output);
        var contextTensor = Tensor<T>.Concat(new[] { inputTensor, outputTensor }, axis: -1);

        // Pass through encoder
        var current = contextTensor;
        foreach (var layer in _encoderLayers)
        {
            current = layer.Forward(current);
        }

        return current;
    }

    private Tensor<T> AggregateWithAttention(List<Tensor<T>> encodings)
    {
        // Stack encodings
        var stacked = Tensor<T>.Stack(encodings, axis: 0);

        // Apply self-attention
        var attentionLayer = _encoderLayers.OfType<MultiHeadAttentionLayer<T>>().First();
        var attended = attentionLayer.Forward(stacked, stacked, stacked);

        // Mean pool
        return Tensor<T>.Mean(attended, axis: 0);
    }

    private Tensor<T> AggregateWithPooling(List<Tensor<T>> encodings)
    {
        // Pass through aggregator
        var aggregated = NumOps.Zero;

        foreach (var encoding in encodings)
        {
            var current = encoding;
            foreach (var layer in _aggregatorLayers)
            {
                current = layer.Forward(current);
            }
            aggregated = NumOps.Add(aggregated, current);
        }

        return NumOps.Divide(aggregated, NumOps.FromDouble(encodings.Count));
    }

    private Tensor<T> GenerateFastWeights(Tensor<T> taskRepresentation)
    {
        // Pass task representation through adaptation network
        var current = taskRepresentation;
        foreach (var layer in _adaptationLayers)
        {
            current = layer.Forward(current);
        }

        // Apply normalization and constraints
        if (_cnapOptions.NormalizeFastWeights)
        {
            current = NormalizeFastWeights(current);
        }

        return current;
    }

    private Tensor<T> NormalizeFastWeights(Tensor<T> weights)
    {
        // Apply weight normalization
        var norms = Tensor<T>.Norm(weights, axis: -1, keepDims: true);
        var scale = NumOps.Divide(weights, NumOps.Maximum(norms, NumOps.FromDouble(1e-8)));

        // Scale by learned magnitude
        return NumOps.Multiply(scale, NumOps.FromDouble(_cnapOptions.FastWeightScale));
    }

    private TOutput PredictOnQuery(TaskBatch<TInput, TOutput> querySet, Tensor<T> taskRep, Tensor<T> fastWeights)
    {
        // Encode queries with task representation
        var queryEncodings = new List<Tensor<T>>();
        foreach (var query in querySet.Tasks)
        {
            var encoding = EncodeQuery(query.Input, taskRep);
            queryEncodings.Add(encoding);
        }

        // Decode with fast weights
        var predictions = DecodeWithFastWeightsBatch(queryEncodings, fastWeights);

        return ConvertFromTensor(predictions);
    }

    private Tensor<T> EncodeQuery(TInput query, Tensor<T> taskRepresentation)
    {
        // Convert query to tensor
        var queryTensor = ConvertToTensor(query);

        // Repeat task representation for each query element
        var taskRepRepeated = Tensor<T>.Repeat(taskRepresentation, repeats: queryTensor.Shape[0], axis: 0);

        // Concatenate query and task representation
        var combined = Tensor<T>.Concat(new[] { queryTensor, taskRepRepeated }, axis: -1);

        return combined;
    }

    private TOutput DecodeWithFastWeights(Tensor<T> encoding, Tensor<T> fastWeights)
    {
        // Apply decoder with fast weights
        var current = encoding;
        var weightIndex = 0;

        foreach (var layer in _decoderLayers)
        {
            // Apply fast weight modification if available
            if (layer is DenseLayer<T> denseLayer && fastWeights != null)
            {
                var layerParams = denseLayer.GetParameters();
                var numParams = layerParams.Count;

                // Extract fast weights for this layer
                var layerFastWeights = fastWeights.Slice(
                    new[] { weightIndex },
                    new[] { numParams });

                // Apply fast weight modification
                denseLayer.ApplyFastWeights(layerFastWeights);
                weightIndex += numParams;
            }

            current = layer.Forward(current);
        }

        return ConvertFromTensor(current);
    }

    private Tensor<T> DecodeWithFastWeightsBatch(List<Tensor<T>> encodings, Tensor<T> fastWeights)
    {
        var predictions = new List<Tensor<T>>();

        foreach (var encoding in encodings)
        {
            var prediction = DecodeWithFastWeights(encoding, fastWeights);
            predictions.Add(prediction);
        }

        return Tensor<T>.Stack(predictions, axis: 0);
    }

    private (T, T) ComputeTaskLoss(TaskBatch<TInput, TOutput> querySet, Tensor<T> predictions)
    {
        // Compute task-specific loss (e.g., cross-entropy for classification)
        var targets = ConvertToTensor(querySet.Output);

        // Compute accuracy if classification
        var accuracy = ComputeAccuracy(predictions, targets);

        // Compute primary loss
        var loss = ComputePrimaryLoss(predictions, targets);

        return (loss, accuracy);
    }

    private T ComputeAdaptationLoss(Tensor<T> fastWeights)
    {
        // Regularization on fast weights
        if (_cnapOptions.FastWeightRegularization == 0.0)
        {
            return NumOps.Zero;
        }

        // L2 regularization on fast weights
        var weightNorm = Tensor<T>.Square(fastWeights).Sum();
        return NumOps.Multiply(
            NumOps.FromDouble(_cnapOptions.FastWeightRegularization),
            weightNorm);
    }

    private T ComputeUncertaintyLoss(Tensor<T> predictions)
    {
        if (!_cnapOptions.PredictUncertainty)
        {
            return NumOps.Zero;
        }

        // Extract mean and variance from predictions
        var halfDim = predictions.Shape[predictions.Shape.Length - 1] / 2;
        var means = predictions.Slice(new[] { 0 }, new[] { halfDim });
        var logVars = predictions.Slice(new[] { halfDim }, new[] { halfDim });

        // Compute negative log likelihood under Gaussian
        var variances = Tensor<T>.Exp(logVars);
        var nll = NumOps.Add(
            Tensor<T>.Divide(Tensor<T>.Square(means), NumOps.Multiply(NumOps.FromDouble(2.0), variances)).Sum(),
            NumOps.Multiply(NumOps.FromDouble(0.5), Tensor<T>.Log(variances).Sum())
        );

        return nll;
    }

    private void MetaUpdate(T totalLoss)
    {
        // Backpropagate through all networks
        var gradients = ComputeGradients(totalLoss);

        // Update meta-parameters
        _metaOptimizer.Update(MetaParameters, gradients);
    }

    private Tensor<T> ComputeGradients(T loss)
    {
        // Compute gradients of loss with respect to meta-parameters
        // This would involve automatic differentiation
        // For now, return dummy gradients
        return Tensor<T>.ZerosLike(MetaParameters);
    }

    private void FineTuneOnSupport(TaskBatch<TInput, TOutput> supportSet)
    {
        // Optional fine-tuning on support set with fast weights
        if (_cnapOptions.FastAdaptationSteps == 0)
        {
            return;
        }

        for (int step = 0; step < _cnapOptions.FastAdaptationSteps; step++)
        {
            // Compute gradient on support set
            var supportLoss = ComputeSupportLoss(supportSet);
            var supportGradients = ComputeSupportGradients(supportLoss);

            // Update fast weights
            UpdateFastWeights(supportGradients);
        }
    }

    private T ComputeSupportLoss(TaskBatch<TInput, TOutput> supportSet)
    {
        // Compute loss on support set
        // This would involve forward pass and loss computation
        return NumOps.Zero;
    }

    private Tensor<T> ComputeSupportGradients(T loss)
    {
        // Compute gradients for fast weight updates
        return Tensor<T>.ZerosLike(_fastWeights);
    }

    private void UpdateFastWeights(Tensor<T> gradients)
    {
        // Update fast weights with gradient step
        var update = NumOps.Multiply(
            NumOps.FromDouble(_cnapOptions.FastLearningRate),
            gradients);
        _fastWeights = NumOps.Subtract(_fastWeights, update);
    }

    private void DistributeParameters(Tensor<T> parameters)
    {
        // Distribute concatenated parameters to individual layers
        var offset = 0;

        foreach (var layer in _encoderLayers.Concat(_decoderLayers).Concat(_adaptationLayers).Concat(_aggregatorLayers))
        {
            var layerParams = layer.GetParameters();
            var numParams = layerParams.Sum(p => p.Count);

            var layerSlice = parameters.Slice(new[] { offset }, new[] { numParams });
            layer.SetParameters(new List<Tensor<T>> { layerSlice });

            offset += numParams;
        }
    }

    // Helper methods
    private Tensor<T> ConvertToTensor(object input)
    {
        // Convert various input types to tensors
        // This would be implemented based on specific input types
        return Tensor<T>.Zeros(new[] { 1 });
    }

    private TOutput ConvertFromTensor(Tensor<T> tensor)
    {
        // Convert tensor predictions back to output type
        // This would be implemented based on specific output types
        return default(TOutput);
    }

    private T ComputePrimaryLoss(Tensor<T> predictions, Tensor<T> targets)
    {
        // Compute the primary task loss (e.g., cross-entropy, MSE)
        // This would use the appropriate loss function based on the task
        return NumOps.Zero;
    }

    private T ComputeAccuracy(Tensor<T> predictions, Tensor<T> targets)
    {
        // Compute accuracy for classification tasks
        // This would find the predicted class and compare to targets
        return NumOps.Zero;
    }
}