using AiDotNet.Models.Options;
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
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class TabRBase<T>
{
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
    private Tensor<T>? _indexEmbeddings;       // [numTrainSamples, embeddingDim]
    private Tensor<T>? _indexFeatures;         // [numTrainSamples, numFeatures]
    private int _numIndexedSamples;

    // Context encoder (processes retrieved neighbors)
    private readonly List<FullyConnectedLayer<T>> _contextLayers;
    private readonly LayerNormalizationLayer<T>? _contextNorm;

    // Attention for neighbor aggregation
    private readonly FullyConnectedLayer<T> _queryProjection;
    private readonly FullyConnectedLayer<T> _keyProjection;
    private readonly FullyConnectedLayer<T> _valueProjection;
    private readonly FullyConnectedLayer<T> _outputProjection;

    // Cache for backward pass
    private Tensor<T>? _queryEmbeddingCache;
    private Tensor<T>? _neighborEmbeddingsCache;
    private Tensor<T>? _attentionWeightsCache;
    private Tensor<T>? _contextCache;
    private Matrix<int>? _neighborIndicesCache;

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
    /// Gets the total number of trainable parameters.
    /// </summary>
    public virtual int ParameterCount
    {
        get
        {
            int count = 0;
            foreach (var layer in _encoderLayers)
                count += layer.ParameterCount;
            if (_encoderNorm != null) count += _encoderNorm.ParameterCount;

            foreach (var layer in _contextLayers)
                count += layer.ParameterCount;
            if (_contextNorm != null) count += _contextNorm.ParameterCount;

            count += _queryProjection.ParameterCount;
            count += _keyProjection.ParameterCount;
            count += _valueProjection.ParameterCount;
            count += _outputProjection.ParameterCount;

            return count;
        }
    }

    /// <summary>
    /// Initializes a new instance of the TabRBase class.
    /// </summary>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="options">Model configuration options.</param>
    protected TabRBase(int numFeatures, TabROptions<T>? options = null)
    {
        NumOps = MathHelper.GetNumericOperations<T>();
        Options = options ?? new TabROptions<T>();
        NumFeatures = numFeatures;

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
            var layer = new FullyConnectedLayer<T>(
                prevDim,
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
        int contextInputDim = Options.EmbeddingDimension * 2;  // Query + neighbor context

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

        // Initialize attention projections
        int headDim = Options.EmbeddingDimension / Options.NumAttentionHeads;
        _queryProjection = new FullyConnectedLayer<T>(
            Options.EmbeddingDimension, Options.EmbeddingDimension, (IActivationFunction<T>?)null);
        _keyProjection = new FullyConnectedLayer<T>(
            Options.EmbeddingDimension, Options.EmbeddingDimension, (IActivationFunction<T>?)null);
        _valueProjection = new FullyConnectedLayer<T>(
            Options.EmbeddingDimension, Options.EmbeddingDimension, (IActivationFunction<T>?)null);
        _outputProjection = new FullyConnectedLayer<T>(
            Options.EmbeddingDimension, Options.EmbeddingDimension, (IActivationFunction<T>?)null);
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
        var normalized = new Tensor<T>(embeddings.Shape);
        var epsilon = NumOps.FromDouble(1e-12);

        for (int b = 0; b < batchSize; b++)
        {
            // Compute L2 norm
            var normSq = NumOps.Zero;
            for (int d = 0; d < dim; d++)
            {
                var val = embeddings[b * dim + d];
                normSq = NumOps.Add(normSq, NumOps.Multiply(val, val));
            }
            var norm = NumOps.Sqrt(NumOps.Add(normSq, epsilon));

            // Normalize
            for (int d = 0; d < dim; d++)
            {
                normalized[b * dim + d] = NumOps.Divide(embeddings[b * dim + d], norm);
            }
        }

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

        // Encode all training samples
        _indexEmbeddings = EncodeFeatures(features);
    }

    /// <summary>
    /// Retrieves the K nearest neighbors for query samples.
    /// </summary>
    /// <param name="queryEmbeddings">Query embeddings [batch_size, embedding_dim].</param>
    /// <param name="excludeIndices">Indices to exclude from retrieval (for leave-one-out during training).</param>
    /// <returns>Tuple of (neighbor indices matrix, neighbor embeddings, distances).</returns>
    protected (Matrix<int> Indices, Tensor<T> Embeddings, Tensor<T> Distances) RetrieveNeighbors(
        Tensor<T> queryEmbeddings,
        Vector<int>? excludeIndices = null)
    {
        if (_indexEmbeddings == null)
        {
            throw new InvalidOperationException("Retrieval index has not been built. Call BuildIndex first.");
        }

        int batchSize = queryEmbeddings.Shape[0];
        int embDim = queryEmbeddings.Shape[1];
        int k = Math.Min(Options.NumNeighbors, _numIndexedSamples);

        var indices = new Matrix<int>(batchSize, k);
        var distances = new Tensor<T>([batchSize, k]);
        var neighborEmbeddings = new Tensor<T>([batchSize, k, embDim]);

        for (int b = 0; b < batchSize; b++)
        {
            // Compute distances to all indexed samples
            var dists = new (int Index, T Distance)[_numIndexedSamples];

            for (int i = 0; i < _numIndexedSamples; i++)
            {
                // Skip excluded indices (for leave-one-out)
                if (excludeIndices != null && b < excludeIndices.Length && excludeIndices[b] == i)
                {
                    dists[i] = (i, NumOps.FromDouble(double.MaxValue));
                    continue;
                }

                // Compute negative cosine similarity (for normalized embeddings, this equals L2 distance)
                var dotProduct = NumOps.Zero;
                for (int d = 0; d < embDim; d++)
                {
                    dotProduct = NumOps.Add(dotProduct,
                        NumOps.Multiply(queryEmbeddings[b * embDim + d], _indexEmbeddings[i * embDim + d]));
                }
                // Use negative similarity so smaller = more similar
                dists[i] = (i, NumOps.Negate(dotProduct));
            }

            // Sort to find k nearest
            Array.Sort(dists, (a, b) => NumOps.Compare(a.Distance, b.Distance));

            for (int j = 0; j < k; j++)
            {
                int neighborIdx = dists[j].Index;
                indices[b, j] = neighborIdx;
                distances[b * k + j] = dists[j].Distance;

                // Copy neighbor embedding
                for (int d = 0; d < embDim; d++)
                {
                    neighborEmbeddings[b * k * embDim + j * embDim + d] =
                        _indexEmbeddings[neighborIdx * embDim + d];
                }
            }
        }

        return (indices, neighborEmbeddings, distances);
    }

    /// <summary>
    /// Aggregates neighbor information using attention.
    /// </summary>
    /// <param name="queryEmbedding">Query embedding [batch_size, embedding_dim].</param>
    /// <param name="neighborEmbeddings">Neighbor embeddings [batch_size, k, embedding_dim].</param>
    /// <returns>Aggregated context [batch_size, embedding_dim].</returns>
    protected Tensor<T> AggregateNeighbors(Tensor<T> queryEmbedding, Tensor<T> neighborEmbeddings)
    {
        int batchSize = queryEmbedding.Shape[0];
        int embDim = queryEmbedding.Shape[1];
        int k = neighborEmbeddings.Shape[1];
        int numHeads = Options.NumAttentionHeads;
        int headDim = embDim / numHeads;

        // Project query, keys, values
        var queries = _queryProjection.Forward(queryEmbedding);  // [batch, embDim]

        // Reshape neighbors for projection
        var neighborsFlat = neighborEmbeddings.Reshape(batchSize * k, embDim);
        var keys = _keyProjection.Forward(neighborsFlat).Reshape(batchSize, k, embDim);
        var values = _valueProjection.Forward(neighborsFlat).Reshape(batchSize, k, embDim);

        // Compute attention scores: Q @ K^T / sqrt(headDim)
        var attentionWeights = new Tensor<T>([batchSize, k]);
        var scale = NumOps.FromDouble(1.0 / Math.Sqrt(headDim));

        for (int b = 0; b < batchSize; b++)
        {
            // Find max for numerical stability
            var maxScore = NumOps.FromDouble(double.MinValue);
            var scores = new T[k];

            for (int j = 0; j < k; j++)
            {
                var score = NumOps.Zero;
                for (int d = 0; d < embDim; d++)
                {
                    score = NumOps.Add(score,
                        NumOps.Multiply(queries[b * embDim + d], keys[b * k * embDim + j * embDim + d]));
                }
                score = NumOps.Multiply(score, scale);
                score = NumOps.Divide(score, NumOps.FromDouble(Options.RetrievalTemperature));
                scores[j] = score;

                if (NumOps.Compare(score, maxScore) > 0)
                {
                    maxScore = score;
                }
            }

            // Softmax
            var sumExp = NumOps.Zero;
            for (int j = 0; j < k; j++)
            {
                var expScore = NumOps.Exp(NumOps.Subtract(scores[j], maxScore));
                attentionWeights[b * k + j] = expScore;
                sumExp = NumOps.Add(sumExp, expScore);
            }

            for (int j = 0; j < k; j++)
            {
                attentionWeights[b * k + j] = NumOps.Divide(attentionWeights[b * k + j], sumExp);
            }
        }

        _attentionWeightsCache = attentionWeights;

        // Apply attention to values: softmax(scores) @ V
        var context = new Tensor<T>([batchSize, embDim]);
        for (int b = 0; b < batchSize; b++)
        {
            for (int d = 0; d < embDim; d++)
            {
                var weighted = NumOps.Zero;
                for (int j = 0; j < k; j++)
                {
                    weighted = NumOps.Add(weighted,
                        NumOps.Multiply(attentionWeights[b * k + j], values[b * k * embDim + j * embDim + d]));
                }
                context[b * embDim + d] = weighted;
            }
        }

        // Project output
        return _outputProjection.Forward(context);
    }

    /// <summary>
    /// Performs the forward pass through the TabR backbone.
    /// </summary>
    /// <param name="features">Input features [batch_size, num_features].</param>
    /// <param name="excludeIndices">Indices to exclude from retrieval (for training).</param>
    /// <returns>Combined representation [batch_size, embedding_dim].</returns>
    protected Tensor<T> ForwardBackbone(Tensor<T> features, Vector<int>? excludeIndices = null)
    {
        // Encode query features
        var queryEmbedding = EncodeFeatures(features);
        _queryEmbeddingCache = queryEmbedding;

        // Retrieve similar neighbors
        var (neighborIndices, neighborEmbeddings, _) = RetrieveNeighbors(queryEmbedding, excludeIndices);
        _neighborIndicesCache = neighborIndices;
        _neighborEmbeddingsCache = neighborEmbeddings;

        // Aggregate neighbor information
        var neighborContext = AggregateNeighbors(queryEmbedding, neighborEmbeddings);

        // Combine query embedding with neighbor context
        int batchSize = features.Shape[0];
        int embDim = Options.EmbeddingDimension;
        var combined = new Tensor<T>([batchSize, embDim * 2]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int d = 0; d < embDim; d++)
            {
                combined[b * embDim * 2 + d] = queryEmbedding[b * embDim + d];
                combined[b * embDim * 2 + embDim + d] = neighborContext[b * embDim + d];
            }
        }

        // Process through context encoder
        var context = combined;
        foreach (var layer in _contextLayers)
        {
            context = layer.Forward(context);
        }

        if (_contextNorm != null)
        {
            context = _contextNorm.Forward(context);
        }

        _contextCache = context;
        return context;
    }

    /// <summary>
    /// Performs the backward pass through the TabR backbone.
    /// </summary>
    /// <param name="outputGradient">Gradient from prediction head [batch_size, embedding_dim].</param>
    /// <returns>Gradient with respect to input features [batch_size, num_features].</returns>
    protected Tensor<T> BackwardBackbone(Tensor<T> outputGradient)
    {
        // Backward through context layers (reverse order)
        var grad = outputGradient;

        if (_contextNorm != null)
        {
            grad = _contextNorm.Backward(grad);
        }

        for (int i = _contextLayers.Count - 1; i >= 0; i--)
        {
            grad = _contextLayers[i].Backward(grad);
        }

        // Split gradient for query and context parts
        int batchSize = grad.Shape[0];
        int embDim = Options.EmbeddingDimension;

        var queryGrad = new Tensor<T>([batchSize, embDim]);
        var contextGrad = new Tensor<T>([batchSize, embDim]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int d = 0; d < embDim; d++)
            {
                queryGrad[b * embDim + d] = grad[b * embDim * 2 + d];
                contextGrad[b * embDim + d] = grad[b * embDim * 2 + embDim + d];
            }
        }

        // Backward through attention and projections (simplified - full implementation would be more complex)
        // For now, just propagate query gradient through encoder

        // Backward through encoder layers
        if (_encoderNorm != null)
        {
            queryGrad = _encoderNorm.Backward(queryGrad);
        }

        for (int i = _encoderLayers.Count - 1; i >= 0; i--)
        {
            queryGrad = _encoderLayers[i].Backward(queryGrad);
        }

        return queryGrad;
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

        // Update attention projections
        _queryProjection.UpdateParameters(learningRate);
        _keyProjection.UpdateParameters(learningRate);
        _valueProjection.UpdateParameters(learningRate);
        _outputProjection.UpdateParameters(learningRate);
    }

    /// <summary>
    /// Gets all trainable parameters as a single vector.
    /// </summary>
    public virtual Vector<T> GetParameters()
    {
        var allParams = new List<T>();

        foreach (var layer in _encoderLayers)
        {
            var p = layer.GetParameters();
            for (int i = 0; i < p.Length; i++) allParams.Add(p[i]);
        }
        if (_encoderNorm != null)
        {
            var p = _encoderNorm.GetParameters();
            for (int i = 0; i < p.Length; i++) allParams.Add(p[i]);
        }

        foreach (var layer in _contextLayers)
        {
            var p = layer.GetParameters();
            for (int i = 0; i < p.Length; i++) allParams.Add(p[i]);
        }
        if (_contextNorm != null)
        {
            var p = _contextNorm.GetParameters();
            for (int i = 0; i < p.Length; i++) allParams.Add(p[i]);
        }

        var qp = _queryProjection.GetParameters();
        for (int i = 0; i < qp.Length; i++) allParams.Add(qp[i]);
        var kp = _keyProjection.GetParameters();
        for (int i = 0; i < kp.Length; i++) allParams.Add(kp[i]);
        var vp = _valueProjection.GetParameters();
        for (int i = 0; i < vp.Length; i++) allParams.Add(vp[i]);
        var op = _outputProjection.GetParameters();
        for (int i = 0; i < op.Length; i++) allParams.Add(op[i]);

        return new Vector<T>([.. allParams]);
    }

    /// <summary>
    /// Resets internal state.
    /// </summary>
    public virtual void ResetState()
    {
        _queryEmbeddingCache = null;
        _neighborEmbeddingsCache = null;
        _attentionWeightsCache = null;
        _contextCache = null;
        _neighborIndicesCache = null;

        foreach (var layer in _encoderLayers)
            layer.ResetState();
        _encoderNorm?.ResetState();

        foreach (var layer in _contextLayers)
            layer.ResetState();
        _contextNorm?.ResetState();

        _queryProjection.ResetState();
        _keyProjection.ResetState();
        _valueProjection.ResetState();
        _outputProjection.ResetState();
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
}
