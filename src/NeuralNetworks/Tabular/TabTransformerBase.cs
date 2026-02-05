using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// Base implementation of TabTransformer for tabular data.
/// </summary>
/// <remarks>
/// <para>
/// TabTransformer applies transformer self-attention to categorical features while
/// passing numerical features through directly. This captures complex relationships
/// between categorical features that simple embeddings might miss.
/// </para>
/// <para>
/// <b>For Beginners:</b> TabTransformer is like FT-Transformer but treats categories specially:
///
/// Architecture:
/// 1. **Categorical Path**: Embedding → Column Embedding → Transformer → Flatten
/// 2. **Numerical Path**: Pass through unchanged
/// 3. **Concatenation**: Combine both paths
/// 4. **MLP Head**: Final prediction layers
///
/// Key insight: Categorical features often have interactions that matter
/// (e.g., "New York" + "Finance" vs "New York" + "Farming"). The transformer
/// learns these relationships through self-attention.
///
/// Example flow:
/// Categories [batch, num_cat] → Embeddings [batch, num_cat, embed_dim]
///                             → Transformer [batch, num_cat, embed_dim]
///                             → Flatten [batch, num_cat * embed_dim]
///                             ↘
/// Numericals [batch, num_num] → Concat [batch, num_cat * embed_dim + num_num]
///                             → MLP → Prediction
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class TabTransformerBase<T>
{
    /// <summary>
    /// Numeric operations helper for type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;

    /// <summary>
    /// The model configuration options.
    /// </summary>
    protected readonly TabTransformerOptions<T> Options;

    /// <summary>
    /// Number of numerical features.
    /// </summary>
    protected readonly int NumNumericalFeatures;

    /// <summary>
    /// Number of categorical features.
    /// </summary>
    protected readonly int NumCategoricalFeatures;

    // Categorical embeddings (one per categorical feature)
    private readonly List<Tensor<T>> _categoricalEmbeddings;  // [numCat][cardinality, embDim]
    private readonly List<Tensor<T>?> _categoricalEmbeddingsGrad;

    // Column embeddings (learned position for each categorical feature)
    private Tensor<T>? _columnEmbeddings;  // [numCat, embDim]
    private Tensor<T>? _columnEmbeddingsGrad;

    // Transformer encoder layers
    private readonly List<TransformerEncoderLayer<T>> _encoderLayers;

    // Final layer norm
    private readonly LayerNormalizationLayer<T>? _finalLayerNorm;

    // MLP layers for combined features
    private readonly List<FullyConnectedLayer<T>> _mlpLayers;

    // Cache for backward pass
    private Tensor<T>? _numericalFeaturesCache;
    private Matrix<int>? _categoricalIndicesCache;
    private Tensor<T>? _embeddedCategoricalsCache;
    private Tensor<T>? _transformedCategoricalsCache;
    private Tensor<T>? _concatenatedCache;

    /// <summary>
    /// Gets the embedding dimension.
    /// </summary>
    public int EmbeddingDimension => Options.EmbeddingDimension;

    /// <summary>
    /// Gets the number of transformer layers.
    /// </summary>
    public int NumLayers => Options.NumLayers;

    /// <summary>
    /// Gets the combined feature dimension after concatenation.
    /// </summary>
    public int CombinedDimension => NumCategoricalFeatures * Options.EmbeddingDimension + NumNumericalFeatures;

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public virtual int ParameterCount
    {
        get
        {
            int count = 0;

            // Categorical embeddings
            foreach (var emb in _categoricalEmbeddings)
                count += emb.Length;

            // Column embeddings
            if (_columnEmbeddings != null)
                count += _columnEmbeddings.Length;

            // Transformer layers
            foreach (var layer in _encoderLayers)
                count += layer.ParameterCount;

            // Layer norm
            if (_finalLayerNorm != null)
                count += _finalLayerNorm.ParameterCount;

            // MLP layers
            foreach (var layer in _mlpLayers)
                count += layer.ParameterCount;

            return count;
        }
    }

    /// <summary>
    /// Initializes a new instance of the TabTransformerBase class.
    /// </summary>
    /// <param name="numNumericalFeatures">Number of numerical input features.</param>
    /// <param name="options">Model configuration options.</param>
    protected TabTransformerBase(int numNumericalFeatures, TabTransformerOptions<T>? options = null)
    {
        NumOps = MathHelper.GetNumericOperations<T>();
        Options = options ?? new TabTransformerOptions<T>();

        NumNumericalFeatures = numNumericalFeatures;
        NumCategoricalFeatures = Options.NumCategoricalFeatures;

        // Validate configuration
        if (Options.EmbeddingDimension % Options.NumHeads != 0)
        {
            throw new ArgumentException(
                $"EmbeddingDimension ({Options.EmbeddingDimension}) must be divisible by NumHeads ({Options.NumHeads})");
        }

        var random = RandomHelper.CreateSecureRandom();

        // Initialize categorical embeddings
        _categoricalEmbeddings = new List<Tensor<T>>();
        _categoricalEmbeddingsGrad = new List<Tensor<T>?>();

        if (Options.CategoricalCardinalities != null)
        {
            foreach (int cardinality in Options.CategoricalCardinalities)
            {
                var embedding = new Tensor<T>([cardinality, Options.EmbeddingDimension]);
                InitializeNormal(embedding, Options.EmbeddingInitScale, random);
                _categoricalEmbeddings.Add(embedding);
                _categoricalEmbeddingsGrad.Add(null);
            }
        }

        // Initialize column embeddings
        if (Options.UseColumnEmbedding && NumCategoricalFeatures > 0)
        {
            _columnEmbeddings = new Tensor<T>([NumCategoricalFeatures, Options.EmbeddingDimension]);
            InitializeNormal(_columnEmbeddings, Options.EmbeddingInitScale, random);
        }

        // Initialize transformer encoder layers
        _encoderLayers = new List<TransformerEncoderLayer<T>>();
        for (int i = 0; i < Options.NumLayers; i++)
        {
            var encoderLayer = new TransformerEncoderLayer<T>(
                Options.EmbeddingDimension,
                Options.NumHeads,
                Options.FeedForwardDimension);
            _encoderLayers.Add(encoderLayer);
        }

        // Final layer normalization
        if (Options.UseLayerNorm)
        {
            _finalLayerNorm = new LayerNormalizationLayer<T>(Options.EmbeddingDimension);
        }

        // Initialize MLP layers
        _mlpLayers = new List<FullyConnectedLayer<T>>();
        int prevDim = CombinedDimension;

        foreach (int hiddenDim in Options.MLPHiddenDimensions)
        {
            var layer = new FullyConnectedLayer<T>(
                prevDim,
                hiddenDim,
                new ReLUActivation<T>() as IActivationFunction<T>);
            _mlpLayers.Add(layer);
            prevDim = hiddenDim;
        }
    }

    /// <summary>
    /// Initializes a tensor with normal distribution.
    /// </summary>
    private void InitializeNormal(Tensor<T> tensor, double scale, Random random)
    {
        for (int i = 0; i < tensor.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = 1.0 - random.NextDouble();
            double normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            tensor[i] = NumOps.FromDouble(normal * scale);
        }
    }

    /// <summary>
    /// Gets the MLP output dimension (last hidden dimension or combined if no MLP layers).
    /// </summary>
    protected int MLPOutputDimension =>
        Options.MLPHiddenDimensions.Length > 0
            ? Options.MLPHiddenDimensions[^1]
            : CombinedDimension;

    /// <summary>
    /// Embeds categorical features.
    /// </summary>
    /// <param name="categoricalIndices">Categorical indices matrix [batch_size, num_categorical].</param>
    /// <returns>Embedded categoricals [batch_size, num_categorical, embedding_dim].</returns>
    protected Tensor<T> EmbedCategoricals(Matrix<int> categoricalIndices)
    {
        int batchSize = categoricalIndices.Rows;
        int numCat = NumCategoricalFeatures;
        int embDim = Options.EmbeddingDimension;

        var embedded = new Tensor<T>([batchSize, numCat, embDim]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < numCat; c++)
            {
                int catIdx = categoricalIndices[b, c];
                var embTable = _categoricalEmbeddings[c];

                for (int d = 0; d < embDim; d++)
                {
                    var value = embTable[catIdx * embDim + d];

                    // Add column embedding if enabled
                    if (_columnEmbeddings != null)
                    {
                        value = NumOps.Add(value, _columnEmbeddings[c * embDim + d]);
                    }

                    embedded[b * numCat * embDim + c * embDim + d] = value;
                }
            }
        }

        return embedded;
    }

    /// <summary>
    /// Performs the forward pass through the TabTransformer backbone.
    /// </summary>
    /// <param name="numericalFeatures">Numerical features [batch_size, num_numerical].</param>
    /// <param name="categoricalIndices">Categorical indices matrix [batch_size, num_categorical].</param>
    /// <returns>MLP output [batch_size, mlp_output_dim].</returns>
    protected Tensor<T> ForwardBackbone(Tensor<T> numericalFeatures, Matrix<int>? categoricalIndices)
    {
        int batchSize = numericalFeatures.Shape[0];
        _numericalFeaturesCache = numericalFeatures;
        _categoricalIndicesCache = categoricalIndices;

        Tensor<T> combinedFeatures;

        if (categoricalIndices != null && NumCategoricalFeatures > 0)
        {
            // Step 1: Embed categorical features
            var embeddedCat = EmbedCategoricals(categoricalIndices);
            _embeddedCategoricalsCache = embeddedCat;

            // Step 2: Pass through transformer layers
            var transformedCat = embeddedCat;
            foreach (var layer in _encoderLayers)
            {
                transformedCat = layer.Forward(transformedCat);
            }

            // Step 3: Apply final layer norm
            if (_finalLayerNorm != null)
            {
                transformedCat = _finalLayerNorm.Forward(transformedCat);
            }

            _transformedCategoricalsCache = transformedCat;

            // Step 4: Flatten transformed categoricals
            int flatCatDim = NumCategoricalFeatures * Options.EmbeddingDimension;
            var flattenedCat = transformedCat.Reshape(batchSize, flatCatDim);

            // Step 5: Concatenate with numerical features
            combinedFeatures = new Tensor<T>([batchSize, CombinedDimension]);
            for (int b = 0; b < batchSize; b++)
            {
                // Copy flattened categorical embeddings
                for (int i = 0; i < flatCatDim; i++)
                {
                    combinedFeatures[b * CombinedDimension + i] = flattenedCat[b * flatCatDim + i];
                }
                // Copy numerical features
                for (int i = 0; i < NumNumericalFeatures; i++)
                {
                    combinedFeatures[b * CombinedDimension + flatCatDim + i] =
                        numericalFeatures[b * NumNumericalFeatures + i];
                }
            }
        }
        else
        {
            // No categorical features, just use numerical
            combinedFeatures = numericalFeatures;
        }

        _concatenatedCache = combinedFeatures;

        // Step 6: Pass through MLP layers
        var mlpOutput = combinedFeatures;
        foreach (var layer in _mlpLayers)
        {
            mlpOutput = layer.Forward(mlpOutput);
        }

        return mlpOutput;
    }

    /// <summary>
    /// Performs the backward pass through the TabTransformer backbone.
    /// </summary>
    /// <param name="outputGradient">Gradient from prediction head.</param>
    /// <returns>Gradient with respect to numerical input.</returns>
    protected Tensor<T> BackwardBackbone(Tensor<T> outputGradient)
    {
        if (_numericalFeaturesCache == null || _concatenatedCache == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        int batchSize = _numericalFeaturesCache.Shape[0];

        // Backward through MLP layers
        var grad = outputGradient;
        for (int i = _mlpLayers.Count - 1; i >= 0; i--)
        {
            grad = _mlpLayers[i].Backward(grad);
        }

        // Extract numerical gradient from concatenated gradient
        var numericalGrad = new Tensor<T>([batchSize, NumNumericalFeatures]);
        int flatCatDim = NumCategoricalFeatures * Options.EmbeddingDimension;

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < NumNumericalFeatures; i++)
            {
                numericalGrad[b * NumNumericalFeatures + i] =
                    grad[b * CombinedDimension + flatCatDim + i];
            }
        }

        // Backward through transformer and embeddings (if categorical features exist)
        if (_transformedCategoricalsCache != null && _categoricalIndicesCache != null)
        {
            // Extract categorical gradient
            var catGrad = new Tensor<T>([batchSize, NumCategoricalFeatures, Options.EmbeddingDimension]);
            for (int b = 0; b < batchSize; b++)
            {
                for (int c = 0; c < NumCategoricalFeatures; c++)
                {
                    for (int d = 0; d < Options.EmbeddingDimension; d++)
                    {
                        catGrad[b * NumCategoricalFeatures * Options.EmbeddingDimension + c * Options.EmbeddingDimension + d] =
                            grad[b * CombinedDimension + c * Options.EmbeddingDimension + d];
                    }
                }
            }

            // Backward through layer norm
            if (_finalLayerNorm != null)
            {
                catGrad = _finalLayerNorm.Backward(catGrad);
            }

            // Backward through transformer layers
            for (int i = _encoderLayers.Count - 1; i >= 0; i--)
            {
                catGrad = _encoderLayers[i].Backward(catGrad);
            }

            // Update embedding gradients
            for (int c = 0; c < NumCategoricalFeatures; c++)
            {
                _categoricalEmbeddingsGrad[c] = new Tensor<T>(_categoricalEmbeddings[c].Shape);
                _categoricalEmbeddingsGrad[c]!.Fill(NumOps.Zero);
            }

            if (_columnEmbeddings != null)
            {
                _columnEmbeddingsGrad = new Tensor<T>(_columnEmbeddings.Shape);
                _columnEmbeddingsGrad.Fill(NumOps.Zero);
            }

            for (int b = 0; b < batchSize; b++)
            {
                for (int c = 0; c < NumCategoricalFeatures; c++)
                {
                    int catIdx = _categoricalIndicesCache[b, c];
                    for (int d = 0; d < Options.EmbeddingDimension; d++)
                    {
                        var g = catGrad[b * NumCategoricalFeatures * Options.EmbeddingDimension + c * Options.EmbeddingDimension + d];
                        _categoricalEmbeddingsGrad[c]![catIdx * Options.EmbeddingDimension + d] =
                            NumOps.Add(_categoricalEmbeddingsGrad[c]![catIdx * Options.EmbeddingDimension + d], g);

                        if (_columnEmbeddingsGrad != null)
                        {
                            _columnEmbeddingsGrad[c * Options.EmbeddingDimension + d] =
                                NumOps.Add(_columnEmbeddingsGrad[c * Options.EmbeddingDimension + d], g);
                        }
                    }
                }
            }
        }

        return numericalGrad;
    }

    /// <summary>
    /// Updates all parameters using the calculated gradients.
    /// </summary>
    public virtual void UpdateParameters(T learningRate)
    {
        // Update categorical embeddings
        for (int c = 0; c < _categoricalEmbeddings.Count; c++)
        {
            if (_categoricalEmbeddingsGrad[c] != null)
            {
                for (int i = 0; i < _categoricalEmbeddings[c].Length; i++)
                {
                    _categoricalEmbeddings[c][i] = NumOps.Subtract(
                        _categoricalEmbeddings[c][i],
                        NumOps.Multiply(learningRate, _categoricalEmbeddingsGrad[c]![i]));
                }
            }
        }

        // Update column embeddings
        if (_columnEmbeddings != null && _columnEmbeddingsGrad != null)
        {
            for (int i = 0; i < _columnEmbeddings.Length; i++)
            {
                _columnEmbeddings[i] = NumOps.Subtract(
                    _columnEmbeddings[i],
                    NumOps.Multiply(learningRate, _columnEmbeddingsGrad[i]));
            }
        }

        // Update transformer layers
        foreach (var layer in _encoderLayers)
        {
            layer.UpdateParameters(learningRate);
        }

        // Update layer norm
        _finalLayerNorm?.UpdateParameters(learningRate);

        // Update MLP layers
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
        _numericalFeaturesCache = null;
        _categoricalIndicesCache = null;
        _embeddedCategoricalsCache = null;
        _transformedCategoricalsCache = null;
        _concatenatedCache = null;

        for (int c = 0; c < _categoricalEmbeddingsGrad.Count; c++)
        {
            _categoricalEmbeddingsGrad[c] = null;
        }
        _columnEmbeddingsGrad = null;

        foreach (var layer in _encoderLayers)
            layer.ResetState();
        _finalLayerNorm?.ResetState();
        foreach (var layer in _mlpLayers)
            layer.ResetState();
    }
}
