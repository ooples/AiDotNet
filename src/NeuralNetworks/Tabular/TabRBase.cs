using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Distributions;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.Attributes;
using AiDotNet.Models.Parameters;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// Base implementation of TabR, a retrieval-augmented model for tabular data.
/// </summary>
/// <remarks>
/// <para>
/// TabR combines neural networks with instance-based learning. It encodes features,
/// retrieves similar training samples, and uses attention to aggregate neighbor
/// information for making predictions.
/// </para>
/// <para>
/// <b>For Beginners:</b> TabR is like having a photographic memory for training data.
///
/// Architecture overview:
/// 1. **Feature Encoder**: MLP that converts raw features to embeddings
/// 2. **Retrieval Index**: Stores embeddings of all training samples
/// 3. **K-NN Search**: Finds the K most similar training samples
/// 4. **Context Encoder**: Uses attention to aggregate neighbor information
/// 5. **Prediction Head**: Makes final prediction using combined information
///
/// Why this approach works:
/// - Tabular data often has "local" structure (similar inputs → similar outputs)
/// - Neural networks alone may struggle with rare patterns
/// - Retrieval provides explicit "memory" of similar past examples
/// - Attention learns which neighbors are most relevant
///
/// Think of it as:
/// - Traditional k-NN: Just average neighbors (simple but limited)
/// - TabR: Learn to encode, then learn to attend to neighbors (powerful)
/// </para>
/// <para>
/// The retrieval module follows Gorishniy et al. (2024), section 3: the target and every candidate are
/// encoded, <c>k = W_K(x~)</c>; the m candidates with the smallest <c>||k - k_i||^2</c> are retrieved; their
/// weights are <c>softmax(-||k - k_i||^2)</c>; each neighbour contributes the value
/// <c>W_Y(y_i) + T(k - k_i)</c>, with <c>T = Linear(ReLU(Linear(.)))</c> and <c>W_Y</c> the label embedding;
/// and the predictor sees <c>x~ + sum_i w_i V_i</c>. Candidates are re-encoded on every pass, so the encoder
/// and <c>W_K</c> also learn through the retrieved neighbours. The neighbour labels enter only when the index
/// was built with labels and <see cref="TabROptions{T}.IncludeNeighborTargets"/> is on.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract partial class TabRBase<T> : IParameterSource<T>
{
    /// <summary>
    /// Provides access to the hardware-accelerated tensor engine.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Numeric operations helper for type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;

    /// <summary>
    /// The model configuration options.
    /// </summary>
    protected readonly TabROptions<T> Options;

    /// <summary>
    /// Number of input features.
    /// </summary>
    protected readonly int NumFeatures;

    // Feature encoder layers
    private readonly List<FullyConnectedLayer<T>> _encoderLayers;
    private readonly LayerNormalizationLayer<T>? _encoderNorm;

    // Retrieval index (stores training sample embeddings)
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T>? _indexEmbeddings;       // [numTrainSamples, embeddingDim]
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T>? _indexFeatures;         // [numTrainSamples, numFeatures]
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T>? _indexLabels;           // [numTrainSamples, labelDimension] (one-hot or targets)
    private int _numIndexedSamples;

    // Label embedding W_Y of the retrieved neighbours' labels (null when labels are not used)
    private readonly FullyConnectedLayer<T>? _labelEmbedding;
    private readonly int _labelDimension;

    // Context encoder (processes retrieved neighbors)
    private readonly List<FullyConnectedLayer<T>> _contextLayers;
    private readonly LayerNormalizationLayer<T>? _contextNorm;

    // Retrieval: the key projection W_K, and the two linear maps of the value module T(k - k_i)
    private readonly FullyConnectedLayer<T> _keyProjection;
    private readonly FullyConnectedLayer<T> _valueProjection;
    private readonly FullyConnectedLayer<T> _outputProjection;

    // Cache for backward pass
    [Scratch]
    private Tensor<T>? _attentionWeightsCache;
    [Scratch]
    private Matrix<int>? _neighborIndicesCache;
    [Scratch]
    [KeepsActivation("Read after the pass by PredictWithConfidence; holds labels, not activations.")]
    private Tensor<T>? _neighborLabelsCache;   // [batch, m, labelDimension]

    /// <summary>
    /// Gets the embedding dimension.
    /// </summary>
    public int EmbeddingDimension => Options.EmbeddingDimension;

    /// <summary>
    /// Gets the number of nearest neighbors to retrieve.
    /// </summary>
    public int NumNeighbors => Options.NumNeighbors;

    /// <summary>
    /// Gets whether the retrieval index has been built.
    /// </summary>
    public bool IsIndexBuilt => _indexEmbeddings != null && _numIndexedSamples > 0;

    /// <summary>
    /// Gets the number of samples in the retrieval index.
    /// </summary>
    public int NumIndexedSamples => _numIndexedSamples;

    /// <summary>
    /// Trainable layers a subclass adds on top of the backbone, walked after it by every
    /// parameter surface below.
    /// </summary>
    /// <remarks>
    /// The task head. A classifier and a regressor share this whole backbone and differ only by a
    /// final projection, and each re-implemented the surfaces just to append that one layer. The
    /// subclass now states WHAT it adds and the folds below decide where it goes, so the count and
    /// the vector cannot place it differently. Mirrors NeuralNetworkBase.GetExtraTrainableLayers,
    /// which solves the same problem for models that derive from it -- this base has no base at
    /// all, so it cannot inherit that one.
    /// </remarks>
    protected virtual IEnumerable<ILayer<T>> GetExtraTrainableLayers()
        => GeneratedParameterDiscovery.EnumerateDerivedLayers<T>(this, typeof(TabRBase<T>));

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    /// <summary>Built once on first parameter access, then reused.</summary>
    private ParameterComponentRegistry<T>? _parameterRegistry;

    /// <summary>
    /// The single ordered traversal of this model's parameter-bearing components.
    /// </summary>
    /// <remarks>
    /// Count, read and restore all derive from THIS. Before, the count and the read were two
    /// separate walks restating the same component order, and there was no restore at all -- so the
    /// vector this model produced could be saved and never loaded back. Two walks agree until
    /// someone adds a component to one of them, and nothing reports the disagreement because the
    /// lengths still look plausible; one enumeration makes that unrepresentable.
    /// </remarks>
    private ParameterComponentRegistry<T> ParameterRegistry
    {
        get
        {
            if (_parameterRegistry is not null) return _parameterRegistry;

            var registry = new ParameterComponentRegistry<T>();

            for (int i = 0; i < _encoderLayers.Count; i++)
                registry.Register($"00000000/{i:D8}", _encoderLayers[i]);

            registry.Register("00000001/encoderNorm", _encoderNorm);

            for (int i = 0; i < _contextLayers.Count; i++)
                registry.Register($"00000002/{i:D8}", _contextLayers[i]);

            registry.Register("00000003/contextNorm", _contextNorm);

            registry.Register("00000005/key", _keyProjection);
            registry.Register("00000006/value", _valueProjection);
            registry.Register("00000007/output", _outputProjection);
            registry.Register("00000008/labelEmbedding", _labelEmbedding);

            int extraIndex = 0;
            foreach (var extra in GetExtraTrainableLayers())
            {
                if (extra is not null) registry.Register($"00009000/{extraIndex++:D8}", extra);
            }

            _parameterRegistry = registry;
            return registry;
        }
    }

    /// <inheritdoc cref="GetParameters"/>
    public virtual long ParameterCount => ParameterRegistry.ParameterCount;

    /// <summary>
    /// Initializes a new instance of the TabRBase class.
    /// </summary>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="options">Model configuration options.</param>
    /// <param name="labelDimension">
    /// Width of one encoded label: the class count for a classifier (one-hot), the output dimension for a
    /// regressor. Zero means the model never embeds neighbour labels.
    /// </param>
    protected TabRBase(int numFeatures, TabROptions<T>? options = null, int labelDimension = 0)
    {
        NumOps = MathHelper.GetNumericOperations<T>();
        Options = options ?? new TabROptions<T>();
        NumFeatures = numFeatures;
        if (labelDimension < 0)
            throw new ArgumentOutOfRangeException(nameof(labelDimension), "Label dimension cannot be negative.");
        _labelDimension = labelDimension;

        // Validate configuration
        if (Options.EmbeddingDimension % Options.NumAttentionHeads != 0)
        {
            throw new ArgumentException(
                $"EmbeddingDimension ({Options.EmbeddingDimension}) must be divisible by NumAttentionHeads ({Options.NumAttentionHeads})");
        }

        // Initialize feature encoder
        _encoderLayers = new List<FullyConnectedLayer<T>>();
        int prevDim = numFeatures;

        for (int i = 0; i < Options.NumLayers; i++)
        {
            // prevDim was already being threaded here and then discarded, so every encoder layer
            // was built deferred and the model reported no parameters at all until a forward pass.
            var layer = prevDim > 0
                ? new FullyConnectedLayer<T>(
                    prevDim,
                    Options.EmbeddingDimension,
                    new ReLUActivation<T>() as IActivationFunction<T>)
                : new FullyConnectedLayer<T>(
                    Options.EmbeddingDimension,
                    new ReLUActivation<T>() as IActivationFunction<T>);
            _encoderLayers.Add(layer);
            prevDim = Options.EmbeddingDimension;
        }

        if (Options.UseLayerNorm)
        {
            _encoderNorm = new LayerNormalizationLayer<T>(Options.EmbeddingDimension);
        }

        // Initialize context encoder
        _contextLayers = new List<FullyConnectedLayer<T>>();
        int contextInputDim = Options.EmbeddingDimension;  // x~ + retrieved context (paper eq. 6)

        for (int i = 0; i < Options.NumContextLayers; i++)
        {
            int inputDim = i == 0 ? contextInputDim : Options.EmbeddingDimension;
            var layer = new FullyConnectedLayer<T>(
                inputDim,
                Options.EmbeddingDimension,
                new ReLUActivation<T>() as IActivationFunction<T>);
            _contextLayers.Add(layer);
        }

        if (Options.UseLayerNorm)
        {
            _contextNorm = new LayerNormalizationLayer<T>(Options.EmbeddingDimension);
        }

        // Retrieval projections. Square maps over the embedding, so both dimensions are fixed here.
        int embDim = Options.EmbeddingDimension;
        _keyProjection = new FullyConnectedLayer<T>(embDim, embDim, (IActivationFunction<T>?)null);
        _valueProjection = new FullyConnectedLayer<T>(embDim, embDim, (IActivationFunction<T>?)null);
        _outputProjection = new FullyConnectedLayer<T>(embDim, embDim, (IActivationFunction<T>?)null);

        if (Options.IncludeNeighborTargets && labelDimension > 0)
        {
            _labelEmbedding = new FullyConnectedLayer<T>(labelDimension, embDim, (IActivationFunction<T>?)null);
        }
    }

    /// <summary>
    /// Encodes input features to embeddings.
    /// </summary>
    /// <param name="features">Input features [batch_size, num_features].</param>
    /// <returns>Embeddings [batch_size, embedding_dim].</returns>
    protected Tensor<T> EncodeFeatures(Tensor<T> features)
    {
        var x = features;

        // Pass through encoder layers
        foreach (var layer in _encoderLayers)
        {
            x = layer.Forward(x);
        }

        // Apply layer normalization
        if (_encoderNorm != null)
        {
            x = _encoderNorm.Forward(x);
        }

        // Optionally normalize to unit length for retrieval
        if (Options.NormalizeEmbeddings)
        {
            x = NormalizeEmbeddings(x);
        }

        return x;
    }

    /// <summary>
    /// Normalizes embeddings to unit length.
    /// </summary>
    private Tensor<T> NormalizeEmbeddings(Tensor<T> embeddings)
    {
        int batchSize = embeddings.Shape[0];
        int dim = embeddings.Shape[1];
        var epsilon = NumOps.FromDouble(1e-12);

        // L2 norm of every row (with epsilon inside the square root), then divide the row by it
        var rows = embeddings.Reshape(batchSize, dim);
        var normSq = Engine.ReduceSum(Engine.TensorMultiply(rows, rows), new[] { 1 }, true);
        var norm = Engine.TensorSqrt(TensorTapeOps.TapeAddScalar(Engine, normSq, epsilon));
        var normalized = Engine.TensorDivide(rows, norm).Reshape(embeddings._shape);

        return normalized;
    }

    /// <summary>
    /// Builds the retrieval index from training data.
    /// </summary>
    /// <param name="features">Training features [num_samples, num_features].</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method creates the "memory" of training samples.
    /// Call this once after training or when you want to update the index.
    ///
    /// The index stores:
    /// - Encoded embeddings of all training samples
    /// - Original features (for reference)
    ///
    /// During prediction, we search this index to find similar samples.
    /// </para>
    /// </remarks>
    public void BuildIndex(Tensor<T> features)
    {
        _numIndexedSamples = features.Shape[0];
        _indexFeatures = features;
        _indexLabels = null;

        // Encode all training samples; the index outlives this call, so it holds its own copy of the engine result
        _indexEmbeddings = TabularMetrics.KeptCopy(EncodeFeatures(features));
    }

    /// <summary>
    /// Builds the retrieval index from training data and the encoded labels of every candidate, which the
    /// label embedding <c>W_Y</c> turns into part of each neighbour's value.
    /// </summary>
    /// <param name="features">Training features [num_samples, num_features].</param>
    /// <param name="labelEncodings">Encoded labels [num_samples, labelDimension].</param>
    protected void BuildIndexWithLabels(Tensor<T> features, Tensor<T> labelEncodings)
    {
        if (features is null) throw new ArgumentNullException(nameof(features));
        if (labelEncodings is null) throw new ArgumentNullException(nameof(labelEncodings));
        int samples = features.Shape[0];
        if (_labelDimension == 0 || labelEncodings.Length != samples * _labelDimension)
        {
            throw new ArgumentException(
                $"Labels must hold {_labelDimension} values for each of the {samples} indexed samples.",
                nameof(labelEncodings));
        }

        BuildIndex(features);
        _indexLabels = TabularMetrics.KeptCopy(labelEncodings).Reshape(samples, _labelDimension);
    }

    /// <summary>
    /// Retrieves the K nearest indexed samples for query samples, by squared L2 distance in embedding space.
    /// </summary>
    /// <param name="queryEmbeddings">Query embeddings [batch_size, embedding_dim].</param>
    /// <param name="excludeIndices">Indices to exclude from retrieval (for leave-one-out during training).</param>
    /// <returns>Tuple of (neighbor indices matrix, neighbor embeddings, squared distances).</returns>
    protected (Matrix<int> Indices, Tensor<T> Embeddings, Tensor<T> Distances) RetrieveNeighbors(
        Tensor<T> queryEmbeddings,
        Vector<int>? excludeIndices = null)
    {
        if (_indexEmbeddings == null)
        {
            throw new InvalidOperationException("Retrieval index has not been built. Call BuildIndex first.");
        }

        int embDim = queryEmbeddings.Shape[1];
        var (nearest, neighbors, distances) = RetrieveNearest(
            queryEmbeddings.Reshape(queryEmbeddings.Shape[0], embDim),
            _indexEmbeddings.Reshape(_numIndexedSamples, embDim),
            excludeIndices);
        return (nearest.ToMatrix(), neighbors, distances);
    }

    /// <summary>
    /// The m candidates nearest to each query by squared L2 distance <c>||q - c_i||^2</c>, in stable ascending
    /// order of distance, with each query's excluded candidate (leave-one-out) never retrieved.
    /// </summary>
    /// <returns>The neighbour indices [batch, m], the neighbour rows [batch, m, dim] and their distances.</returns>
    private (Tensor<int> Nearest, Tensor<T> Neighbors, Tensor<T> Distances) RetrieveNearest(
        Tensor<T> queries,
        Tensor<T> candidates,
        Vector<int>? excludeIndices)
    {
        int batchSize = queries.Shape[0];
        int dim = queries.Shape[1];

        // The target is never its own candidate (paper section 3.1). An excluded column is pushed to the end of
        // the ranking, so it must also fall outside the k retrieved, or it would be weighted like a neighbour.
        bool excludes = excludeIndices != null && excludeIndices.Length > 0;
        int k = Math.Min(Options.NumNeighbors, _numIndexedSamples - (excludes ? 1 : 0));
        if (k < 1)
        {
            throw new InvalidOperationException(
                "Retrieval needs at least one candidate besides the excluded sample; index more samples.");
        }

        // ||q||^2 + ||c||^2 - 2 q.c for every (query, candidate) pair
        var queryNormSq = Engine.ReduceSum(Engine.TensorMultiply(queries, queries), new[] { 1 }, true);
        var candidateNormSq = Engine.ReduceSum(Engine.TensorMultiply(candidates, candidates), new[] { 1 }, false)
            .Reshape(1, _numIndexedSamples);
        var cross = Engine.TensorMatMul(queries, Engine.TensorTranspose(candidates));
        var allDistances = Engine.TensorSubtract(
            Engine.TensorAdd(queryNormSq, candidateNormSq),
            TensorTapeOps.TapeMultiplyScalar(Engine, cross, NumOps.FromDouble(2.0)));

        // Skip excluded indices (for leave-one-out): row b excludes column excludeIndices[b] when b is covered
        if (excludes && excludeIndices != null)
        {
            int covered = Math.Min(batchSize, excludeIndices.Length);
            var excluded = Tensor<int>.FromVector(excludeIndices.GetSubVector(0, covered));
            if (covered < batchSize)
            {
                excluded = Engine.TensorConcatenate(
                    new[] { excluded, Engine.TensorFullLike(new Tensor<int>(new[] { batchSize - covered }), -1) }, 0);
            }

            var excludedMask = Engine.TensorOneHot<T>(excluded, _numIndexedSamples).Reshape(batchSize, _numIndexedSamples);
            allDistances = Engine.TensorWhere(
                excludedMask, Engine.TensorFullLike(allDistances, NumOps.MaxValue), allDistances);
        }

        // k nearest in stable ascending order of distance
        var nearest = TieBreakArgMax.StableSmallestK(allDistances, k);
        var distances = Engine.TensorTakeAlongDim(allDistances, nearest, 1);
        var neighbors = Engine.TensorIndexSelect(candidates, nearest.Reshape(batchSize * k), 0)
            .Reshape(batchSize, k, dim);

        return (nearest, neighbors, distances);
    }

    /// <summary>
    /// The retrieval module of Gorishniy et al. (2024), eq. 5: weights <c>softmax(-||k - k_i||^2 / temperature)</c>
    /// over the retrieved neighbours, values <c>W_Y(y_i) + T(k - k_i)</c>, and their weighted sum.
    /// </summary>
    /// <param name="queryKeys">Target keys k [batch_size, embedding_dim].</param>
    /// <param name="neighborKeys">Retrieved keys k_i [batch_size, m, embedding_dim].</param>
    /// <param name="neighborLabels">Encoded neighbour labels [batch_size * m, labelDimension], or null.</param>
    /// <returns>Aggregated context [batch_size, embedding_dim].</returns>
    private Tensor<T> AggregateNeighbors(Tensor<T> queryKeys, Tensor<T> neighborKeys, Tensor<T>? neighborLabels)
    {
        int batchSize = queryKeys.Shape[0];
        int embDim = queryKeys.Shape[1];
        int k = neighborKeys.Shape[1];

        // k - k_i for every neighbour, on the tape
        var difference = Engine.TensorSubtract(queryKeys.Reshape(batchSize, 1, embDim), neighborKeys);

        // S = -||k - k_i||^2, divided by the retrieval temperature (1 in the paper)
        var squaredDistance = Engine.ReduceSum(Engine.TensorMultiply(difference, difference), new[] { 2 }, false);
        var scores = TensorTapeOps.TapeMultiplyScalar(
            Engine, squaredDistance, NumOps.FromDouble(-1.0 / Options.RetrievalTemperature));
        var attentionWeights = Engine.Softmax(scores, -1);
        _attentionWeightsCache = TabularMetrics.KeptCopy(attentionWeights);

        // V_i = T(k - k_i) + W_Y(y_i), with T = Linear(ReLU(Linear(.)))
        var differenceRows = difference.Reshape(batchSize * k, embDim);
        var values = _outputProjection.Forward(Engine.ReLU(_valueProjection.Forward(differenceRows)));
        if (_labelEmbedding != null && neighborLabels != null)
        {
            values = Engine.TensorAdd(values, _labelEmbedding.Forward(neighborLabels));
        }

        // sum_i w_i V_i
        return Engine.ReduceSum(
            Engine.TensorMultiply(attentionWeights.Reshape(batchSize, k, 1), values.Reshape(batchSize, k, embDim)),
            new[] { 1 },
            false);
    }

    /// <summary>
    /// Performs the forward pass through the TabR backbone.
    /// </summary>
    /// <param name="features">Input features [batch_size, num_features].</param>
    /// <param name="excludeIndices">Indices to exclude from retrieval (for training).</param>
    /// <returns>Combined representation [batch_size, embedding_dim].</returns>
    protected Tensor<T> ForwardBackbone(Tensor<T> features, Vector<int>? excludeIndices = null)
    {
        if (_indexFeatures == null || _numIndexedSamples == 0)
        {
            throw new InvalidOperationException("Retrieval index has not been built. Call BuildIndex first.");
        }

        int batchSize = features.Shape[0];
        int embDim = Options.EmbeddingDimension;

        // Encode the targets and, with the same encoder, every candidate; keys k = W_K(x~)
        var queryEmbedding = EncodeFeatures(features).Reshape(batchSize, embDim);
        var candidateEmbedding = EncodeFeatures(_indexFeatures).Reshape(_numIndexedSamples, embDim);
        _indexEmbeddings = TabularMetrics.KeptCopy(candidateEmbedding);
        var queryKeys = _keyProjection.Forward(queryEmbedding);
        var candidateKeys = _keyProjection.Forward(candidateEmbedding);

        // Retrieve the m nearest candidates by key distance
        var (nearest, neighborKeys, _) = RetrieveNearest(queryKeys, candidateKeys, excludeIndices);
        _neighborIndicesCache = nearest.ToMatrix();

        int k = nearest.Shape[1];
        Tensor<T>? neighborLabels = null;
        _neighborLabelsCache = null;
        if (_indexLabels != null)
        {
            neighborLabels = Engine.TensorIndexSelect(_indexLabels, nearest.Reshape(batchSize * k), 0);
            _neighborLabelsCache = TabularMetrics.KeptCopy(neighborLabels).Reshape(batchSize, k, _labelDimension);
        }

        // x = x~ + sum_i w_i V_i, then the predictor
        var context = Engine.TensorAdd(queryEmbedding, AggregateNeighbors(queryKeys, neighborKeys, neighborLabels));
        foreach (var layer in _contextLayers)
        {
            context = layer.Forward(context);
        }

        if (_contextNorm != null)
        {
            context = _contextNorm.Forward(context);
        }

        return context;
    }

    /// <summary>
    /// The live tensors a tape training step updates: the encoder, the retrieval projections, the label
    /// embedding, the predictor and the task head a subclass adds.
    /// </summary>
    protected IEnumerable<Tensor<T>> GetTrainableTensors()
    {
        var layers = new List<ILayer<T>>();
        layers.AddRange(_encoderLayers);
        if (_encoderNorm != null) layers.Add(_encoderNorm);
        layers.AddRange(_contextLayers);
        if (_contextNorm != null) layers.Add(_contextNorm);
        layers.Add(_keyProjection);
        layers.Add(_valueProjection);
        layers.Add(_outputProjection);
        if (_labelEmbedding != null) layers.Add(_labelEmbedding);
        layers.AddRange(GetExtraTrainableLayers());
        return TabularTraining.LayerTensors(layers);
    }

    /// <summary>
    /// One tape training step: gradients of <paramref name="objective"/> over every trainable tensor, then SGD.
    /// </summary>
    /// <returns>The loss before the update.</returns>
    protected T TrainWithTape(Func<Tensor<T>> objective, T learningRate)
    {
        try
        {
            return TabularTraining.SgdStep(GetTrainableTensors, objective, learningRate);
        }
        finally
        {
            ResetState();
        }
    }

    /// <summary>
    /// Updates all parameters using the calculated gradients.
    /// </summary>
    public virtual void UpdateParameters(T learningRate)
    {
        // Update encoder layers
        foreach (var layer in _encoderLayers)
        {
            layer.UpdateParameters(learningRate);
        }
        _encoderNorm?.UpdateParameters(learningRate);

        // Update context layers
        foreach (var layer in _contextLayers)
        {
            layer.UpdateParameters(learningRate);
        }
        _contextNorm?.UpdateParameters(learningRate);

        // Update retrieval projections and the label embedding
        _keyProjection.UpdateParameters(learningRate);
        _valueProjection.UpdateParameters(learningRate);
        _outputProjection.UpdateParameters(learningRate);
        _labelEmbedding?.UpdateParameters(learningRate);
    }

    /// <summary>
    /// Gets all trainable parameters as a single vector.
    /// </summary>
    public virtual Vector<T> GetParameters() => ParameterRegistry.GetParameters();

    /// <summary>Restores every parameter, in the order <see cref="GetParameters"/> emitted them.</summary>
    /// <remarks>
    /// This model had no restore path at all. Its vector could be produced and saved, and there was
    /// no way to load it back -- so a checkpoint of a TabR model was write-only.
    /// </remarks>
    public virtual void SetParameters(Vector<T> parameters)
        => ParameterRegistry.SetParameters(parameters);

    /// <summary>
    /// Resets internal state.
    /// </summary>
    public virtual void ResetState()
    {
        _attentionWeightsCache = null;
        _neighborIndicesCache = null;
        _neighborLabelsCache = null;

        foreach (var layer in _encoderLayers)
            layer.ResetState();
        _encoderNorm?.ResetState();

        foreach (var layer in _contextLayers)
            layer.ResetState();
        _contextNorm?.ResetState();

        _keyProjection.ResetState();
        _valueProjection.ResetState();
        _outputProjection.ResetState();
        _labelEmbedding?.ResetState();
    }

    /// <summary>
    /// Gets the attention weights from the last forward pass.
    /// </summary>
    /// <returns>Attention weights [batch_size, num_neighbors] or null if not available.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Attention weights show how much the model focuses on each neighbor:
    /// - Higher weights = neighbor is more influential in the prediction
    /// - Useful for interpretability (understanding why a prediction was made)
    /// - Can identify which training examples are most relevant
    /// </para>
    /// </remarks>
    public Tensor<T>? GetAttentionWeights() => _attentionWeightsCache;

    /// <summary>
    /// Gets the retrieved neighbor indices from the last forward pass.
    /// </summary>
    /// <returns>Neighbor indices matrix [batch_size, num_neighbors] or null if not available.</returns>
    public Matrix<int>? GetRetrievedNeighborIndices() => _neighborIndicesCache;

    /// <summary>
    /// The encoded labels [batch, m, labelDimension] of the neighbours the last forward pass retrieved, or null
    /// when the index holds no labels.
    /// </summary>
    protected Tensor<T>? GetNeighborLabels() => _neighborLabelsCache;
}
