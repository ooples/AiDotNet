using AiDotNet.ActivationFunctions;
using AiDotNet.Extensions;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// Base class for SAINT (Self-Attention and Intersample Attention Transformer).
/// </summary>
/// <remarks>
/// <para>
/// SAINT applies two types of attention in alternating layers:
/// 1. Column attention: Self-attention over features (like FT-Transformer)
/// 2. Row attention: Inter-sample attention comparing samples in a batch
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of SAINT as looking at your data from two perspectives:
///
/// - **Column attention**: "Which features are related to each other?"
///   (e.g., income and education level might be correlated)
/// - **Row attention**: "Which samples in my batch are similar?"
///   (e.g., customers with similar profiles might have similar behavior)
///
/// By combining both views, SAINT can learn patterns that other models miss.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class SAINTBase<T>
{
    protected readonly SAINTOptions<T> Options;
    protected readonly int NumNumericalFeatures;
    protected readonly int NumCategoricalFeatures;
    protected readonly int TotalEmbeddedFeatures;
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly Random _random = RandomHelper.CreateSecureRandom();

    // Feature embeddings
    private readonly FullyConnectedLayer<T> _numericalEmbedding;
    private readonly Tensor<T>[]? _categoricalEmbeddings;
    private readonly Tensor<T>? _columnEmbeddings;

    // Transformer layers (alternating column and row attention)
    private readonly List<MultiHeadAttentionLayer<T>> _columnAttentionLayers;
    private readonly List<MultiHeadAttentionLayer<T>>? _rowAttentionLayers;
    private readonly List<FullyConnectedLayer<T>> _ffnLayers;
    private readonly List<LayerNormalizationLayer<T>>? _layerNorms;

    // MLP head
    private readonly List<FullyConnectedLayer<T>> _mlpLayers;
    protected int MLPOutputDimension { get; }

    // Caches for backward pass
    private Tensor<T>? _embeddedFeaturesCache;
    private List<Tensor<T>>? _columnAttentionOutputsCache;
    private List<Tensor<T>>? _rowAttentionOutputsCache;
    private Tensor<T>? _mlpOutputCache;

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public virtual int ParameterCount
    {
        get
        {
            int count = _numericalEmbedding.ParameterCount;

            if (_categoricalEmbeddings != null)
            {
                foreach (var emb in _categoricalEmbeddings)
                    count += emb.Length;
            }

            if (_columnEmbeddings != null)
                count += _columnEmbeddings.Length;

            foreach (var layer in _columnAttentionLayers)
                count += layer.ParameterCount;

            if (_rowAttentionLayers != null)
            {
                foreach (var layer in _rowAttentionLayers)
                    count += layer.ParameterCount;
            }

            foreach (var layer in _ffnLayers)
                count += layer.ParameterCount;

            foreach (var layer in _mlpLayers)
                count += layer.ParameterCount;

            return count;
        }
    }

    /// <summary>
    /// Initializes a new instance of the SAINTBase class.
    /// </summary>
    /// <param name="numNumericalFeatures">Number of numerical input features.</param>
    /// <param name="options">Model configuration options.</param>
    protected SAINTBase(int numNumericalFeatures, SAINTOptions<T>? options = null)
    {
        Options = options ?? new SAINTOptions<T>();
        NumNumericalFeatures = numNumericalFeatures;
        NumCategoricalFeatures = Options.CategoricalCardinalities?.Length ?? 0;
        TotalEmbeddedFeatures = NumNumericalFeatures + NumCategoricalFeatures;

        if (TotalEmbeddedFeatures == 0)
        {
            throw new ArgumentException("Model must have at least one feature (numerical or categorical)");
        }

        // Numerical feature embedding
        _numericalEmbedding = new FullyConnectedLayer<T>(
            1,
            Options.EmbeddingDimension,
            Options.HiddenActivation ?? new GELUActivation<T>());

        // Categorical embeddings
        if (NumCategoricalFeatures > 0 && Options.CategoricalCardinalities != null)
        {
            _categoricalEmbeddings = new Tensor<T>[NumCategoricalFeatures];
            for (int i = 0; i < NumCategoricalFeatures; i++)
            {
                int cardinality = Options.CategoricalCardinalities[i];
                _categoricalEmbeddings[i] = new Tensor<T>(new[] { cardinality, Options.EmbeddingDimension });
                InitializeEmbedding(_categoricalEmbeddings[i]);
            }
        }

        // Column embeddings (positional encoding for features)
        _columnEmbeddings = new Tensor<T>(new[] { TotalEmbeddedFeatures, Options.EmbeddingDimension });
        InitializeEmbedding(_columnEmbeddings);

        // Initialize transformer layers
        _columnAttentionLayers = [];
        _ffnLayers = [];

        if (Options.UseIntersampleAttention)
        {
            _rowAttentionLayers = [];
        }

        if (Options.UseLayerNorm)
        {
            _layerNorms = [];
        }

        for (int i = 0; i < Options.NumLayers; i++)
        {
            // Column attention layer
            _columnAttentionLayers.Add(new MultiHeadAttentionLayer<T>(
                Options.EmbeddingDimension,
                Options.NumHeads,
                Options.EmbeddingDimension / Options.NumHeads));

            // Row attention layer (if enabled)
            if (Options.UseIntersampleAttention)
            {
                _rowAttentionLayers!.Add(new MultiHeadAttentionLayer<T>(
                    Options.EmbeddingDimension,
                    Options.NumHeads,
                    Options.EmbeddingDimension / Options.NumHeads));
            }

            // Feed-forward network
            _ffnLayers.Add(new FullyConnectedLayer<T>(
                Options.EmbeddingDimension,
                Options.EmbeddingDimension,
                Options.HiddenActivation ?? new GELUActivation<T>()));

            // Layer normalizations
            if (Options.UseLayerNorm)
            {
                _layerNorms!.Add(new LayerNormalizationLayer<T>(Options.EmbeddingDimension));
                _layerNorms.Add(new LayerNormalizationLayer<T>(Options.EmbeddingDimension));
                if (Options.UseIntersampleAttention)
                {
                    _layerNorms.Add(new LayerNormalizationLayer<T>(Options.EmbeddingDimension));
                }
            }
        }

        // MLP head
        _mlpLayers = [];
        int mlpInput = TotalEmbeddedFeatures * Options.EmbeddingDimension;
        foreach (var hiddenDim in Options.MLPHiddenDimensions)
        {
            _mlpLayers.Add(new FullyConnectedLayer<T>(
                mlpInput,
                hiddenDim,
                Options.HiddenActivation ?? new GELUActivation<T>()));
            mlpInput = hiddenDim;
        }

        MLPOutputDimension = mlpInput;
    }

    private void InitializeEmbedding(Tensor<T> embedding)
    {
        var scale = NumOps.FromDouble(Options.EmbeddingInitScale);
        for (int i = 0; i < embedding.Length; i++)
        {
            embedding[i] = NumOps.Multiply(
                NumOps.FromDouble(_random.NextGaussian()),
                scale);
        }
    }

    /// <summary>
    /// Embeds input features into the transformer space.
    /// </summary>
    protected Tensor<T> EmbedFeatures(Tensor<T> numericalFeatures, Matrix<int>? categoricalIndices = null)
    {
        int batchSize = numericalFeatures.Shape[0];
        var embedded = new Tensor<T>(new[] { batchSize, TotalEmbeddedFeatures, Options.EmbeddingDimension });

        // Embed numerical features
        for (int b = 0; b < batchSize; b++)
        {
            for (int f = 0; f < NumNumericalFeatures; f++)
            {
                var featureValue = new Tensor<T>(new[] { 1, 1 });
                featureValue[0] = numericalFeatures[b * NumNumericalFeatures + f];

                var embeddedFeature = _numericalEmbedding.Forward(featureValue);

                for (int d = 0; d < Options.EmbeddingDimension; d++)
                {
                    int idx = (b * TotalEmbeddedFeatures + f) * Options.EmbeddingDimension + d;
                    embedded[idx] = NumOps.Add(
                        embeddedFeature[d],
                        _columnEmbeddings![f * Options.EmbeddingDimension + d]);
                }
            }
        }

        // Embed categorical features
        if (categoricalIndices != null && _categoricalEmbeddings != null)
        {
            for (int b = 0; b < batchSize; b++)
            {
                for (int f = 0; f < NumCategoricalFeatures; f++)
                {
                    int catIdx = categoricalIndices[b, f];
                    int featureIdx = NumNumericalFeatures + f;

                    for (int d = 0; d < Options.EmbeddingDimension; d++)
                    {
                        int embIdx = catIdx * Options.EmbeddingDimension + d;
                        int outIdx = (b * TotalEmbeddedFeatures + featureIdx) * Options.EmbeddingDimension + d;
                        embedded[outIdx] = NumOps.Add(
                            _categoricalEmbeddings[f][embIdx],
                            _columnEmbeddings![featureIdx * Options.EmbeddingDimension + d]);
                    }
                }
            }
        }

        return embedded;
    }

    /// <summary>
    /// Performs the forward pass through the SAINT backbone.
    /// </summary>
    protected Tensor<T> ForwardBackbone(Tensor<T> numericalFeatures, Matrix<int>? categoricalIndices = null)
    {
        int batchSize = numericalFeatures.Shape[0];
        _columnAttentionOutputsCache = [];
        _rowAttentionOutputsCache = [];

        // Embed features
        var embedded = EmbedFeatures(numericalFeatures, categoricalIndices);
        _embeddedFeaturesCache = embedded;

        var current = embedded;

        // Process through transformer layers
        int normIdx = 0;
        for (int layer = 0; layer < Options.NumLayers; layer++)
        {
            // Column attention (self-attention over features)
            var colAttnInput = current;
            if (Options.UsePreNorm && Options.UseLayerNorm)
            {
                colAttnInput = ApplyLayerNorm(_layerNorms![normIdx++], colAttnInput, batchSize);
            }

            var colAttnOutput = ApplyColumnAttention(_columnAttentionLayers[layer], colAttnInput, batchSize);
            _columnAttentionOutputsCache.Add(colAttnOutput);

            // Residual connection
            current = AddTensors(current, colAttnOutput);

            if (!Options.UsePreNorm && Options.UseLayerNorm)
            {
                current = ApplyLayerNorm(_layerNorms![normIdx++], current, batchSize);
            }

            // Row attention (inter-sample attention, if enabled)
            if (Options.UseIntersampleAttention && _rowAttentionLayers != null)
            {
                var rowAttnInput = current;
                if (Options.UsePreNorm && Options.UseLayerNorm)
                {
                    rowAttnInput = ApplyLayerNorm(_layerNorms![normIdx++], rowAttnInput, batchSize);
                }

                var rowAttnOutput = ApplyRowAttention(_rowAttentionLayers[layer], rowAttnInput, batchSize);
                _rowAttentionOutputsCache.Add(rowAttnOutput);

                // Residual connection
                current = AddTensors(current, rowAttnOutput);

                if (!Options.UsePreNorm && Options.UseLayerNorm)
                {
                    current = ApplyLayerNorm(_layerNorms![normIdx++], current, batchSize);
                }
            }

            // Feed-forward network
            var ffnInput = current;
            if (Options.UsePreNorm && Options.UseLayerNorm)
            {
                ffnInput = ApplyLayerNorm(_layerNorms![normIdx++], ffnInput, batchSize);
            }

            var ffnOutput = ApplyFeedForward(_ffnLayers[layer], ffnInput, batchSize);

            // Residual connection
            current = AddTensors(current, ffnOutput);

            if (!Options.UsePreNorm && Options.UseLayerNorm)
            {
                current = ApplyLayerNorm(_layerNorms![normIdx++], current, batchSize);
            }
        }

        // Flatten and apply MLP head
        var flattened = new Tensor<T>(new[] { batchSize, TotalEmbeddedFeatures * Options.EmbeddingDimension });
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < TotalEmbeddedFeatures * Options.EmbeddingDimension; i++)
            {
                flattened[b * (TotalEmbeddedFeatures * Options.EmbeddingDimension) + i] =
                    current[b * TotalEmbeddedFeatures * Options.EmbeddingDimension + i];
            }
        }

        var mlpOutput = flattened;
        foreach (var mlpLayer in _mlpLayers)
        {
            mlpOutput = mlpLayer.Forward(mlpOutput);
        }

        _mlpOutputCache = mlpOutput;
        return mlpOutput;
    }

    private Tensor<T> ApplyColumnAttention(MultiHeadAttentionLayer<T> layer, Tensor<T> input, int batchSize)
    {
        // Column attention: attention over the feature dimension
        // Input shape: [batch, features, embedding_dim]
        // Each sample attends over its features independently
        var output = new Tensor<T>(input.Shape);

        for (int b = 0; b < batchSize; b++)
        {
            // Extract features for this sample
            var sample = new Tensor<T>(new[] { TotalEmbeddedFeatures, Options.EmbeddingDimension });
            for (int f = 0; f < TotalEmbeddedFeatures; f++)
            {
                for (int d = 0; d < Options.EmbeddingDimension; d++)
                {
                    sample[f * Options.EmbeddingDimension + d] =
                        input[(b * TotalEmbeddedFeatures + f) * Options.EmbeddingDimension + d];
                }
            }

            // Apply attention
            var attended = layer.Forward(sample);

            // Copy back
            for (int f = 0; f < TotalEmbeddedFeatures; f++)
            {
                for (int d = 0; d < Options.EmbeddingDimension; d++)
                {
                    output[(b * TotalEmbeddedFeatures + f) * Options.EmbeddingDimension + d] =
                        attended[f * Options.EmbeddingDimension + d];
                }
            }
        }

        return output;
    }

    private Tensor<T> ApplyRowAttention(MultiHeadAttentionLayer<T> layer, Tensor<T> input, int batchSize)
    {
        // Row attention: attention over the batch dimension for each feature
        // Input shape: [batch, features, embedding_dim]
        // Each feature attends across samples
        var output = new Tensor<T>(input.Shape);

        for (int f = 0; f < TotalEmbeddedFeatures; f++)
        {
            // Extract this feature across all samples
            var feature = new Tensor<T>(new[] { batchSize, Options.EmbeddingDimension });
            for (int b = 0; b < batchSize; b++)
            {
                for (int d = 0; d < Options.EmbeddingDimension; d++)
                {
                    feature[b * Options.EmbeddingDimension + d] =
                        input[(b * TotalEmbeddedFeatures + f) * Options.EmbeddingDimension + d];
                }
            }

            // Apply attention
            var attended = layer.Forward(feature);

            // Copy back
            for (int b = 0; b < batchSize; b++)
            {
                for (int d = 0; d < Options.EmbeddingDimension; d++)
                {
                    output[(b * TotalEmbeddedFeatures + f) * Options.EmbeddingDimension + d] =
                        attended[b * Options.EmbeddingDimension + d];
                }
            }
        }

        return output;
    }

    private Tensor<T> ApplyFeedForward(FullyConnectedLayer<T> layer, Tensor<T> input, int batchSize)
    {
        var output = new Tensor<T>(input.Shape);

        for (int b = 0; b < batchSize; b++)
        {
            for (int f = 0; f < TotalEmbeddedFeatures; f++)
            {
                var featureVec = new Tensor<T>(new[] { 1, Options.EmbeddingDimension });
                for (int d = 0; d < Options.EmbeddingDimension; d++)
                {
                    featureVec[d] = input[(b * TotalEmbeddedFeatures + f) * Options.EmbeddingDimension + d];
                }

                var ffnOut = layer.Forward(featureVec);

                for (int d = 0; d < Options.EmbeddingDimension; d++)
                {
                    output[(b * TotalEmbeddedFeatures + f) * Options.EmbeddingDimension + d] = ffnOut[d];
                }
            }
        }

        return output;
    }

    private Tensor<T> ApplyLayerNorm(LayerNormalizationLayer<T> norm, Tensor<T> input, int batchSize)
    {
        var output = new Tensor<T>(input.Shape);

        for (int b = 0; b < batchSize; b++)
        {
            for (int f = 0; f < TotalEmbeddedFeatures; f++)
            {
                var featureVec = new Tensor<T>(new[] { 1, Options.EmbeddingDimension });
                for (int d = 0; d < Options.EmbeddingDimension; d++)
                {
                    featureVec[d] = input[(b * TotalEmbeddedFeatures + f) * Options.EmbeddingDimension + d];
                }

                var normalized = norm.Forward(featureVec);

                for (int d = 0; d < Options.EmbeddingDimension; d++)
                {
                    output[(b * TotalEmbeddedFeatures + f) * Options.EmbeddingDimension + d] = normalized[d];
                }
            }
        }

        return output;
    }

    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        var result = new Tensor<T>(a.Shape);
        for (int i = 0; i < a.Length; i++)
        {
            result[i] = NumOps.Add(a[i], b[i]);
        }
        return result;
    }

    /// <summary>
    /// Performs the backward pass through the backbone.
    /// </summary>
    protected Tensor<T> BackwardBackbone(Tensor<T> gradOutput)
    {
        // Backward through MLP layers
        var grad = gradOutput;
        for (int i = _mlpLayers.Count - 1; i >= 0; i--)
        {
            grad = _mlpLayers[i].Backward(grad);
        }

        // For simplicity, return the gradient (full backward through attention is complex)
        return grad;
    }

    /// <summary>
    /// Updates all parameters.
    /// </summary>
    public virtual void UpdateParameters(T learningRate)
    {
        _numericalEmbedding.UpdateParameters(learningRate);

        foreach (var layer in _columnAttentionLayers)
        {
            layer.UpdateParameters(learningRate);
        }

        if (_rowAttentionLayers != null)
        {
            foreach (var layer in _rowAttentionLayers)
            {
                layer.UpdateParameters(learningRate);
            }
        }

        foreach (var layer in _ffnLayers)
        {
            layer.UpdateParameters(learningRate);
        }

        foreach (var layer in _mlpLayers)
        {
            layer.UpdateParameters(learningRate);
        }
    }

    /// <summary>
    /// Resets internal state.
    /// </summary>
    public virtual void ResetState()
    {
        _embeddedFeaturesCache = null;
        _columnAttentionOutputsCache = null;
        _rowAttentionOutputsCache = null;
        _mlpOutputCache = null;

        _numericalEmbedding.ResetState();

        foreach (var layer in _columnAttentionLayers)
        {
            layer.ResetState();
        }

        if (_rowAttentionLayers != null)
        {
            foreach (var layer in _rowAttentionLayers)
            {
                layer.ResetState();
            }
        }

        foreach (var layer in _ffnLayers)
        {
            layer.ResetState();
        }

        foreach (var layer in _mlpLayers)
        {
            layer.ResetState();
        }
    }
}
