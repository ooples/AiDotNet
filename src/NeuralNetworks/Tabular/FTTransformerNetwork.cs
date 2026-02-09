using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// FT-Transformer (Feature Tokenizer + Transformer) neural network for tabular data.
/// </summary>
/// <remarks>
/// <para>
/// FT-Transformer applies the transformer architecture to tabular data by treating each feature
/// as a token. It tokenizes numerical and categorical features into embeddings and processes
/// them with standard transformer encoder layers.
/// This implementation follows the standard neural network architecture pattern with:
/// - Proper inheritance from NeuralNetworkBase
/// - Layer-based architecture using ILayer components
/// - Engine-based tensor operations for CPU/GPU acceleration
/// - Full autodiff and JIT compilation support
/// </para>
/// <para>
/// <b>For Beginners:</b> FT-Transformer brings NLP's Transformer to tables:
///
/// Architecture:
/// 1. **Feature Tokenization**: Each column becomes a "token" (like words in text)
/// 2. **[CLS] Token**: Special token to aggregate information for prediction
/// 3. **Transformer Encoder**: Self-attention learns feature relationships
/// 4. **Classification Head**: MLP on [CLS] token for final prediction
///
/// Key insight: Just as transformers revolutionized NLP by learning word relationships,
/// FT-Transformer learns feature relationships through attention. Each feature can
/// "attend" to other features to capture complex interactions.
///
/// This often outperforms gradient boosting on larger datasets and automatically
/// learns which feature combinations matter.
/// </para>
/// <para>
/// Reference: "Revisiting Deep Learning Models for Tabular Data" (Gorishniy et al., NeurIPS 2021)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class FTTransformerNetwork<T> : NeuralNetworkBase<T>
{
    private FTTransformerOptions<T> _options;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private ILossFunction<T> _lossFunction;

    /// <summary>
    /// Gets the FT-Transformer-specific options.
    /// </summary>
    public FTTransformerOptions<T> Options => _options;

    /// <summary>
    /// Gets the embedding dimension.
    /// </summary>
    public int EmbeddingDimension => _options.EmbeddingDimension;

    /// <summary>
    /// Gets the number of attention heads.
    /// </summary>
    public int NumHeads => _options.NumHeads;

    /// <summary>
    /// Gets the number of transformer layers.
    /// </summary>
    public int NumLayers => _options.NumLayers;

    /// <summary>
    /// Initializes a new FT-Transformer network with the specified architecture.
    /// </summary>
    public FTTransformerNetwork(
        NeuralNetworkArchitecture<T> architecture,
        FTTransformerOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0)
        : base(architecture, lossFunction ??= NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), maxGradNorm)
    {
        _options = options ?? new FTTransformerOptions<T>();
        _lossFunction = lossFunction;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        if (_options.NumHeads <= 0)
        {
            throw new ArgumentException(
                $"NumHeads ({_options.NumHeads}) must be positive.");
        }

        if (_options.EmbeddingDimension % _options.NumHeads != 0)
        {
            throw new ArgumentException(
                $"EmbeddingDimension ({_options.EmbeddingDimension}) must be divisible by NumHeads ({_options.NumHeads})");
        }

        InitializeLayers();
    }

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultFTTransformerLayers(
                Architecture,
                numFeatures: Architecture.CalculatedInputSize,
                embeddingDimension: _options.EmbeddingDimension,
                numHeads: _options.NumHeads,
                numLayers: _options.NumLayers,
                numClasses: Architecture.OutputSize,
                dropoutRate: _options.DropoutRate));
        }
    }

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        Tensor<T> currentOutput = input;
        foreach (var layer in Layers)
        {
            currentOutput = layer.Forward(currentOutput);
        }

        return currentOutput;
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        Tensor<T> prediction = Predict(input);
        var predVector = prediction.ToVector();
        var expectedVector = expectedOutput.ToVector();
        LastLoss = _lossFunction.CalculateLoss(predVector, expectedVector);
        var gradientVector = _lossFunction.CalculateDerivative(predVector, expectedVector);
        var error = Tensor<T>.FromVector(gradientVector);
        BackpropagateError(error);
        UpdateNetworkParameters();
    }

    private void BackpropagateError(Tensor<T> error)
    {
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            error = Layers[i].Backward(error);
        }
    }

    private void UpdateNetworkParameters()
    {
        _optimizer.UpdateParameters(Layers);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int startIndex = 0;
        foreach (var layer in Layers)
        {
            int layerParameterCount = layer.ParameterCount;
            if (layerParameterCount > 0)
            {
                Vector<T> layerParameters = parameters.SubVector(startIndex, layerParameterCount);
                layer.UpdateParameters(layerParameters);
                startIndex += layerParameterCount;
            }
        }
    }

    /// <inheritdoc/>
    public override Dictionary<string, T> GetFeatureImportance()
    {
        var importance = new Dictionary<string, T>();
        int numFeatures = Architecture.CalculatedInputSize;

        if (numFeatures == 0)
        {
            return importance;
        }

        // For FT-Transformer, importance could be derived from attention weights
        var uniformValue = NumOps.FromDouble(1.0 / numFeatures);
        for (int f = 0; f < numFeatures; f++)
        {
            importance[$"feature_{f}"] = uniformValue;
        }

        return importance;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "Architecture", "FT-Transformer" },
                { "NumFeatures", Architecture.CalculatedInputSize },
                { "OutputDim", Architecture.OutputSize },
                { "EmbeddingDimension", _options.EmbeddingDimension },
                { "NumHeads", _options.NumHeads },
                { "NumLayers", _options.NumLayers },
                { "FeedForwardMultiplier", _options.FeedForwardMultiplier },
                { "UsePreLayerNorm", _options.UsePreLayerNorm },
                { "UseReGLU", _options.UseReGLU },
                { "LayerCount", Layers.Count },
                { "LayerTypes", Layers.Select(l => l.GetType().Name).ToArray() }
            },
            ModelData = this.Serialize()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_options.EmbeddingDimension);
        writer.Write(_options.NumHeads);
        writer.Write(_options.NumLayers);
        writer.Write(_options.FeedForwardMultiplier);
        writer.Write(_options.DropoutRate);
        writer.Write(_options.AttentionDropoutRate);
        writer.Write(_options.ResidualDropoutRate);
        writer.Write(_options.UsePreLayerNorm);
        writer.Write(_options.LayerNormEpsilon);
        writer.Write(_options.EmbeddingInitScale);
        writer.Write(_options.UseNumericalBias);
        writer.Write(_options.EnableGradientClipping);
        writer.Write(_options.MaxGradientNorm);
        writer.Write(_options.WeightDecay);
        writer.Write(_options.UseReGLU);

        if (_options.CategoricalCardinalities != null)
        {
            writer.Write(_options.CategoricalCardinalities.Length);
            foreach (var card in _options.CategoricalCardinalities)
            {
                writer.Write(card);
            }
        }
        else
        {
            writer.Write(0);
        }
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        int embDim = reader.ReadInt32();
        int numHeads = reader.ReadInt32();
        if (embDim <= 0)
        {
            throw new InvalidOperationException(
                $"Deserialized EmbeddingDimension ({embDim}) must be positive. Data may be corrupted.");
        }
        if (numHeads <= 0)
        {
            throw new InvalidOperationException(
                $"Deserialized NumHeads ({numHeads}) must be positive. Data may be corrupted.");
        }
        if (embDim % numHeads != 0)
        {
            throw new InvalidOperationException(
                $"Deserialized EmbeddingDimension ({embDim}) is not divisible by NumHeads ({numHeads}). Data may be corrupted.");
        }
        _options.EmbeddingDimension = embDim;
        _options.NumHeads = numHeads;
        _options.NumLayers = reader.ReadInt32();
        _options.FeedForwardMultiplier = reader.ReadInt32();
        _options.DropoutRate = reader.ReadDouble();
        _options.AttentionDropoutRate = reader.ReadDouble();
        _options.ResidualDropoutRate = reader.ReadDouble();
        _options.UsePreLayerNorm = reader.ReadBoolean();
        _options.LayerNormEpsilon = reader.ReadDouble();
        _options.EmbeddingInitScale = reader.ReadDouble();
        _options.UseNumericalBias = reader.ReadBoolean();
        _options.EnableGradientClipping = reader.ReadBoolean();
        _options.MaxGradientNorm = reader.ReadDouble();
        _options.WeightDecay = reader.ReadDouble();
        _options.UseReGLU = reader.ReadBoolean();

        int cardCount = reader.ReadInt32();
        const int MaxReasonableCardinalities = 100_000;
        if (cardCount < 0 || cardCount > MaxReasonableCardinalities)
        {
            throw new InvalidOperationException(
                $"Deserialized CategoricalCardinalities count ({cardCount}) is out of valid range [0, {MaxReasonableCardinalities}]. Data may be corrupted.");
        }
        if (cardCount > 0)
        {
            _options.CategoricalCardinalities = new int[cardCount];
            for (int i = 0; i < cardCount; i++)
            {
                _options.CategoricalCardinalities[i] = reader.ReadInt32();
            }
        }
        else
        {
            _options.CategoricalCardinalities = null;
        }
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        // Create with cloned options and a fresh optimizer to avoid shared mutable state
        return new FTTransformerNetwork<T>(
            Architecture,
            _options.Clone(),
            optimizer: null,
            _lossFunction);
    }
}
