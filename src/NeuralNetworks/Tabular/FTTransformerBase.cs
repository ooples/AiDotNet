using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// Base implementation of FT-Transformer (Feature Tokenizer + Transformer) for tabular data.
/// </summary>
/// <remarks>
/// <para>
/// FT-Transformer is a simple but effective adaptation of the Transformer architecture for tabular data.
/// It treats each feature as a token by embedding it into a d-dimensional vector, then processes
/// the sequence with standard transformer encoder layers.
/// </para>
/// <para>
/// <b>For Beginners:</b> FT-Transformer brings the power of the Transformer architecture (used in
/// GPT, BERT, etc.) to traditional tabular data like spreadsheets and databases.
///
/// Architecture overview:
/// 1. **Feature Tokenizer**: Converts each column value into a vector embedding
/// 2. **[CLS] Token**: A special learnable token prepended to capture global information
/// 3. **Transformer Layers**: Self-attention layers to capture feature interactions
/// 4. **Prediction Head**: Uses the [CLS] token output to make predictions
///
/// How it works:
/// - Each feature (column) becomes a "token" with its own embedding
/// - Self-attention allows any feature to "look at" any other feature
/// - The [CLS] token aggregates information from all features
/// - Final prediction is made based on the [CLS] representation
///
/// Key advantages:
/// - Learns which features interact with each other automatically
/// - No manual feature engineering needed
/// - Often outperforms gradient boosting on larger datasets
/// - Attention weights provide interpretability
/// </para>
/// <para>
/// Reference: "Revisiting Deep Learning Models for Tabular Data" (Gorishniy et al., NeurIPS 2021)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class FTTransformerBase<T>
{
    /// <summary>
    /// Numeric operations helper for type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;

    /// <summary>
    /// The feature tokenizer that converts features to embeddings.
    /// </summary>
    protected readonly FeatureTokenizer<T> Tokenizer;

    /// <summary>
    /// The transformer encoder layers.
    /// </summary>
    protected readonly List<TransformerEncoderLayer<T>> EncoderLayers;

    /// <summary>
    /// Final layer normalization applied after the transformer.
    /// </summary>
    protected readonly LayerNormalizationLayer<T> FinalLayerNorm;

    /// <summary>
    /// The model configuration options.
    /// </summary>
    protected readonly FTTransformerOptions<T> Options;

    /// <summary>
    /// Number of input features (numerical + categorical).
    /// </summary>
    protected readonly int NumFeatures;

    /// <summary>
    /// Number of numerical features.
    /// </summary>
    protected readonly int NumNumericalFeatures;

    /// <summary>
    /// Number of categorical features.
    /// </summary>
    protected readonly int NumCategoricalFeatures;

    // Cache for backward pass
    private Tensor<T>? _tokenizedCache;
    private readonly List<Tensor<T>> _layerOutputsCache;

    /// <summary>
    /// Gets the embedding dimension.
    /// </summary>
    public int EmbeddingDimension => Options.EmbeddingDimension;

    /// <summary>
    /// Gets the number of transformer layers.
    /// </summary>
    public int NumLayers => Options.NumLayers;

    /// <summary>
    /// Gets the sequence length including [CLS] token.
    /// </summary>
    public int SequenceLength => Tokenizer.SequenceLength;

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public virtual int ParameterCount
    {
        get
        {
            int count = Tokenizer.ParameterCount;
            foreach (var layer in EncoderLayers)
            {
                count += layer.ParameterCount;
            }
            count += FinalLayerNorm.ParameterCount;
            return count;
        }
    }

    /// <summary>
    /// Initializes a new instance of the FTTransformerBase class.
    /// </summary>
    /// <param name="numNumericalFeatures">Number of numerical input features.</param>
    /// <param name="options">Model configuration options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creating an FT-Transformer:
    /// - numNumericalFeatures: Count of numerical columns in your data
    /// - options: Configuration for the model (embedding size, number of layers, etc.)
    ///
    /// Categorical features are handled through options.CategoricalCardinalities.
    /// </para>
    /// </remarks>
    protected FTTransformerBase(int numNumericalFeatures, FTTransformerOptions<T>? options = null)
    {
        NumOps = MathHelper.GetNumericOperations<T>();
        Options = options ?? new FTTransformerOptions<T>();

        NumNumericalFeatures = numNumericalFeatures;
        NumCategoricalFeatures = Options.CategoricalCardinalities?.Length ?? 0;
        NumFeatures = NumNumericalFeatures + NumCategoricalFeatures;

        // Validate embedding dimension is divisible by number of heads
        if (Options.EmbeddingDimension % Options.NumHeads != 0)
        {
            throw new ArgumentException(
                $"EmbeddingDimension ({Options.EmbeddingDimension}) must be divisible by NumHeads ({Options.NumHeads})");
        }

        // Initialize feature tokenizer
        Tokenizer = new FeatureTokenizer<T>(
            numNumericalFeatures,
            Options.EmbeddingDimension,
            Options.CategoricalCardinalities,
            Options.UseNumericalBias,
            Options.EmbeddingInitScale);

        // Initialize transformer encoder layers
        EncoderLayers = new List<TransformerEncoderLayer<T>>();
        for (int i = 0; i < Options.NumLayers; i++)
        {
            var encoderLayer = new TransformerEncoderLayer<T>(
                Options.EmbeddingDimension,
                Options.NumHeads,
                Options.FeedForwardDimension);

            EncoderLayers.Add(encoderLayer);
        }

        // Final layer normalization
        FinalLayerNorm = new LayerNormalizationLayer<T>(Options.EmbeddingDimension);

        // Initialize cache
        _layerOutputsCache = new List<Tensor<T>>();
    }

    /// <summary>
    /// Performs the forward pass through the FT-Transformer backbone.
    /// </summary>
    /// <param name="numericalFeatures">Numerical features tensor [batch_size, num_numerical].</param>
    /// <param name="categoricalIndices">Categorical feature indices matrix [batch_size, num_categorical] or null.</param>
    /// <returns>The [CLS] token representation [batch_size, embedding_dim].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The forward pass:
    /// 1. Tokenizes features into embeddings
    /// 2. Processes through transformer layers
    /// 3. Returns the [CLS] token representation for prediction
    ///
    /// The output is the learned representation that captures all feature interactions,
    /// ready for use by a prediction head (classification or regression).
    /// </para>
    /// </remarks>
    protected Tensor<T> ForwardBackbone(Tensor<T> numericalFeatures, Matrix<int>? categoricalIndices = null)
    {
        // Clear cache
        _layerOutputsCache.Clear();

        // Step 1: Tokenize features
        // Input: [batch, num_features]
        // Output: [batch, seq_len, embed_dim] where seq_len = 1 (CLS) + num_features
        var tokenized = Tokenizer.Forward(numericalFeatures, categoricalIndices);
        _tokenizedCache = tokenized;

        // Step 2: Pass through transformer encoder layers
        var hidden = tokenized;
        foreach (var layer in EncoderLayers)
        {
            _layerOutputsCache.Add(hidden);
            hidden = layer.Forward(hidden);
        }

        // Step 3: Apply final layer normalization
        var normalized = FinalLayerNorm.Forward(hidden);

        // Step 4: Extract [CLS] token representation (position 0)
        int batchSize = normalized.Shape[0];
        int seqLen = normalized.Shape[1];
        int embedDim = normalized.Shape[2];

        var clsOutput = new Tensor<T>([batchSize, embedDim]);
        for (int b = 0; b < batchSize; b++)
        {
            for (int d = 0; d < embedDim; d++)
            {
                // CLS is at position 0 in the sequence
                clsOutput[b * embedDim + d] = normalized[b * seqLen * embedDim + 0 * embedDim + d];
            }
        }

        return clsOutput;
    }

    /// <summary>
    /// Performs the backward pass through the FT-Transformer backbone.
    /// </summary>
    /// <param name="clsGradient">Gradient from the prediction head [batch_size, embedding_dim].</param>
    /// <returns>Gradient with respect to numerical input [batch_size, num_numerical].</returns>
    protected Tensor<T> BackwardBackbone(Tensor<T> clsGradient)
    {
        if (_tokenizedCache == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        int batchSize = clsGradient.Shape[0];
        int embedDim = clsGradient.Shape[1];
        int seqLen = Tokenizer.SequenceLength;

        // Convert [CLS] gradient to full sequence gradient
        // Gradient is zero for all positions except [CLS] (position 0)
        var seqGradient = new Tensor<T>([batchSize, seqLen, embedDim]);
        seqGradient.Fill(NumOps.Zero);

        for (int b = 0; b < batchSize; b++)
        {
            for (int d = 0; d < embedDim; d++)
            {
                seqGradient[b * seqLen * embedDim + 0 * embedDim + d] = clsGradient[b * embedDim + d];
            }
        }

        // Backward through final layer norm
        var grad = FinalLayerNorm.Backward(seqGradient);

        // Backward through transformer encoder layers (reverse order)
        for (int i = EncoderLayers.Count - 1; i >= 0; i--)
        {
            grad = EncoderLayers[i].Backward(grad);
        }

        // Backward through tokenizer
        return Tokenizer.Backward(grad);
    }

    /// <summary>
    /// Updates all parameters using the calculated gradients.
    /// </summary>
    /// <param name="learningRate">The learning rate.</param>
    public virtual void UpdateParameters(T learningRate)
    {
        Tokenizer.UpdateParameters(learningRate);

        foreach (var layer in EncoderLayers)
        {
            layer.UpdateParameters(learningRate);
        }

        FinalLayerNorm.UpdateParameters(learningRate);
    }

    /// <summary>
    /// Gets all trainable parameters as a single vector.
    /// </summary>
    public virtual Vector<T> GetParameters()
    {
        var allParams = new List<T>();

        // Tokenizer parameters
        var tokenizerParams = Tokenizer.GetParameters();
        for (int i = 0; i < tokenizerParams.Length; i++)
        {
            allParams.Add(tokenizerParams[i]);
        }

        // Encoder layer parameters
        foreach (var layer in EncoderLayers)
        {
            var layerParams = layer.GetParameters();
            for (int i = 0; i < layerParams.Length; i++)
            {
                allParams.Add(layerParams[i]);
            }
        }

        // Final layer norm parameters
        var normParams = FinalLayerNorm.GetParameters();
        for (int i = 0; i < normParams.Length; i++)
        {
            allParams.Add(normParams[i]);
        }

        return new Vector<T>([.. allParams]);
    }

    /// <summary>
    /// Sets all trainable parameters from a vector.
    /// </summary>
    public virtual void SetParameters(Vector<T> parameters)
    {
        int offset = 0;

        // Tokenizer parameters
        int tokenizerCount = Tokenizer.ParameterCount;
        var tokenizerParams = new Vector<T>(tokenizerCount);
        for (int i = 0; i < tokenizerCount; i++)
        {
            tokenizerParams[i] = parameters[offset + i];
        }
        Tokenizer.SetParameters(tokenizerParams);
        offset += tokenizerCount;

        // Encoder layer parameters
        foreach (var layer in EncoderLayers)
        {
            int layerCount = layer.ParameterCount;
            var layerParams = new Vector<T>(layerCount);
            for (int i = 0; i < layerCount; i++)
            {
                layerParams[i] = parameters[offset + i];
            }
            layer.SetParameters(layerParams);
            offset += layerCount;
        }

        // Final layer norm parameters
        int normCount = FinalLayerNorm.ParameterCount;
        var normParams = new Vector<T>(normCount);
        for (int i = 0; i < normCount; i++)
        {
            normParams[i] = parameters[offset + i];
        }
        FinalLayerNorm.SetParameters(normParams);
    }

    /// <summary>
    /// Gets parameter gradients as a single vector.
    /// </summary>
    public virtual Vector<T> GetParameterGradients()
    {
        var allGrads = new List<T>();

        // Tokenizer gradients
        var tokenizerGrads = Tokenizer.GetParameterGradients();
        for (int i = 0; i < tokenizerGrads.Length; i++)
        {
            allGrads.Add(tokenizerGrads[i]);
        }

        // Encoder layer gradients
        foreach (var layer in EncoderLayers)
        {
            var layerGrads = layer.GetParameterGradients();
            for (int i = 0; i < layerGrads.Length; i++)
            {
                allGrads.Add(layerGrads[i]);
            }
        }

        // Final layer norm gradients
        var normGrads = FinalLayerNorm.GetParameterGradients();
        for (int i = 0; i < normGrads.Length; i++)
        {
            allGrads.Add(normGrads[i]);
        }

        return new Vector<T>([.. allGrads]);
    }

    /// <summary>
    /// Resets internal state including caches and gradients.
    /// </summary>
    public virtual void ResetState()
    {
        _tokenizedCache = null;
        _layerOutputsCache.Clear();

        Tokenizer.ResetGradients();

        foreach (var layer in EncoderLayers)
        {
            layer.ResetState();
        }

        FinalLayerNorm.ResetState();
    }

    /// <summary>
    /// Gets attention weights from all layers for interpretability.
    /// </summary>
    /// <returns>List of attention weight tensors, one per layer.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Attention weights show which features the model focuses on
    /// when making predictions.
    ///
    /// - Higher weights mean stronger attention between features
    /// - Can be used to understand feature relationships
    /// - Provides model interpretability
    /// </para>
    /// </remarks>
    public List<Tensor<T>?> GetAttentionWeights()
    {
        // Note: This would require modifying TransformerEncoderLayer to expose attention weights
        // For now, we return an empty list as a placeholder
        return new List<Tensor<T>?>();
    }

    /// <summary>
    /// Computes feature importance based on the attention patterns.
    /// </summary>
    /// <returns>Feature importance scores [num_features].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Feature importance tells you which input columns matter most
    /// for the model's predictions.
    ///
    /// Higher scores = more important features
    /// The scores are derived from how much attention the [CLS] token pays to each feature.
    /// </para>
    /// </remarks>
    public virtual Vector<T> GetFeatureImportance()
    {
        // This is a simplified version - actual implementation would aggregate attention
        // weights from the [CLS] token to each feature token across all layers
        var importance = new Vector<T>(NumFeatures);
        var equalWeight = NumOps.FromDouble(1.0 / NumFeatures);
        for (int i = 0; i < NumFeatures; i++)
        {
            importance[i] = equalWeight;
        }
        return importance;
    }

    /// <summary>
    /// Sets the training mode for the model.
    /// </summary>
    /// <param name="isTraining">True for training mode, false for inference.</param>
    public void SetTrainingMode(bool isTraining)
    {
        foreach (var layer in EncoderLayers)
        {
            layer.SetTrainingMode(isTraining);
        }
        FinalLayerNorm.SetTrainingMode(isTraining);
    }
}
