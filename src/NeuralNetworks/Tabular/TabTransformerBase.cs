using AiDotNet.Engines;
using AiDotNet.Attributes;
using System.Collections.Generic;
using AiDotNet.Models.Parameters;
using AiDotNet.LinearAlgebra;
using AiDotNet.Interfaces;
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
public abstract class TabTransformerBase<T> : IParameterSource<T>
{
    /// <summary>
    /// Numeric operations helper for type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;

    /// <summary>
    /// Hardware-accelerated engine for tensor operations.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

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
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T>? _columnEmbeddings;  // [numCat, embDim]
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T>? _columnEmbeddingsGrad;

    // Transformer encoder layers
    private readonly List<TransformerEncoderLayer<T>> _encoderLayers;

    // Final layer norm
    private readonly LayerNormalizationLayer<T>? _finalLayerNorm;

    // MLP layers for combined features
    private readonly List<FullyConnectedLayer<T>> _mlpLayers;

    // Cache for backward pass
    [Scratch]
    private Tensor<T>? _numericalFeaturesCache;
    [Scratch]
    private Matrix<int>? _categoricalIndicesCache;
    [Scratch]
    private Tensor<T>? _embeddedCategoricalsCache;
    [Scratch]
    private Tensor<T>? _transformedCategoricalsCache;
    [Scratch]
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

    /// <summary>Built once on first parameter access, then reused.</summary>
    private ParameterComponentRegistry<T>? _parameterRegistry;

    /// <summary>Extra trainable layers a subclass contributes after the shared backbone.</summary>
    protected virtual IEnumerable<IParameterSource<T>> GetExtraTrainableLayers()
        => GeneratedParameterDiscovery.EnumerateDerivedSources<T>(this, typeof(TabTransformerBase<T>));

    /// <summary>The single ordered traversal of this model's parameter-bearing components.</summary>
    private ParameterComponentRegistry<T> ParameterRegistry
    {
        get
        {
            if (_parameterRegistry is not null) return _parameterRegistry;

            // The encoder width is fixed by EmbeddingDimension. Resolve its nested attention,
            // feed-forward, and normalization tensors before freezing the registry layout.
            foreach (var layer in _encoderLayers)
                layer.MaterializeParameters();

            var registry = new ParameterComponentRegistry<T>();
            registry.Register("00000000/categorical",
                new TensorCollectionParameterSource<T>(() => _categoricalEmbeddings));
            // Only when present. This is an OPTIONAL component -- the constructor allocates it just
            // for Options.UseColumnEmbedding, and the old count said the same thing with
            // "if (_columnEmbeddings != null)". Registering it unconditionally would hand the
            // registry a null-valued tensor source, which reports ShapeDeferred, and the whole model
            // would then refuse to be counted at all rather than report the size it genuinely has.
            // Absent is not the same as not-yet-sized.
            if (_columnEmbeddings != null)
            {
                registry.Register("00000001/column",
                    new TensorFieldParameterSource<T>(() => _columnEmbeddings));
            }

            for (int i = 0; i < _encoderLayers.Count; i++)
                registry.Register($"00000002/{i:D8}", _encoderLayers[i]);

            registry.Register("00000003/finalNorm", _finalLayerNorm);

            for (int i = 0; i < _mlpLayers.Count; i++)
                registry.Register($"00000004/{i:D8}", _mlpLayers[i]);

            int extraIndex = 0;
            foreach (var extra in GetExtraTrainableLayers())
                if (extra is not null) registry.Register($"00009000/{extraIndex++:D8}", extra);

            _parameterRegistry = registry;
            return registry;
        }
    }

    public virtual long ParameterCount => ParameterRegistry.ParameterCount;

    public virtual Vector<T> GetParameters() => ParameterRegistry.GetParameters();

    public virtual void SetParameters(Vector<T> parameters) => ParameterRegistry.SetParameters(parameters);

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
                Options.NumHeads,
                Options.FeedForwardDimension,
                Options.EmbeddingDimension);
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
    /// Updates all parameters using the calculated gradients.
    /// </summary>
    public virtual void UpdateParameters(T learningRate)
    {
        // Update categorical embeddings
        for (int c = 0; c < _categoricalEmbeddings.Count; c++)
        {
            if (_categoricalEmbeddingsGrad[c] is { } catGradTensor)
            {
                _categoricalEmbeddings[c] = Engine.TensorSubtract(_categoricalEmbeddings[c],
                    Engine.TensorMultiplyScalar(catGradTensor, learningRate));
            }
        }

        // Update column embeddings
        if (_columnEmbeddings != null && _columnEmbeddingsGrad != null)
        {
            _columnEmbeddings = Engine.TensorSubtract(_columnEmbeddings,
                Engine.TensorMultiplyScalar(_columnEmbeddingsGrad, learningRate));
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
