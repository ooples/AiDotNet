using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// AutoInt (Automatic Feature Interaction Learning) neural network for tabular data.
/// </summary>
/// <remarks>
/// <para>
/// AutoInt uses multi-head self-attention to automatically learn high-order
/// feature interactions without manual feature engineering.
/// This implementation follows the standard neural network architecture pattern with:
/// - Proper inheritance from NeuralNetworkBase
/// - Layer-based architecture using ILayer components
/// - Engine-based tensor operations for CPU/GPU acceleration
/// - Full autodiff and JIT compilation support
/// </para>
/// <para>
/// <b>For Beginners:</b> AutoInt discovers feature interactions automatically:
///
/// Architecture:
/// 1. **Feature Embeddings**: Each feature gets a learned vector representation
/// 2. **Self-Attention Layers**: Features attend to each other to learn interactions
/// 3. **Residual Connections**: Preserve individual feature information
/// 4. **MLP Head**: Final prediction from interaction-enhanced features
///
/// Key insight: Feature interactions (e.g., "age + income" or "city + job")
/// are often important for predictions. AutoInt learns these automatically
/// through attention, capturing which features should be combined.
///
/// Example: In click prediction, "user_age + product_category" might have
/// a strong interaction that AutoInt discovers without manual feature engineering.
/// </para>
/// <para>
/// Reference: "AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks" (2018)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class AutoIntNetwork<T> : NeuralNetworkBase<T>
{
    private AutoIntOptions<T> _options;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private ILossFunction<T> _lossFunction;

    /// <summary>
    /// Gets the AutoInt-specific options.
    /// </summary>
    public new AutoIntOptions<T> Options => _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// Gets the embedding dimension.
    /// </summary>
    public int EmbeddingDimension => _options.EmbeddingDimension;

    /// <summary>
    /// Gets the number of attention heads.
    /// </summary>
    public int NumHeads => _options.NumHeads;

    /// <summary>
    /// Gets the number of interacting layers.
    /// </summary>
    public int NumLayers => _options.NumLayers;

    /// <summary>
    /// Initializes a new AutoInt network with the specified architecture.
    /// </summary>
    public AutoIntNetwork(
        NeuralNetworkArchitecture<T> architecture,
        AutoIntOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), maxGradNorm)
    {
        _options = options ?? new AutoIntOptions<T>();
        // Reuse the same resolved instance that was passed to base(...)
        _lossFunction = LossFunction;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

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
            Layers.AddRange(LayerHelper<T>.CreateDefaultAutoIntLayers(
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
        // Always use layer-by-layer forward pass during training to ensure
        // each layer's forward cache is populated for Backward().
        // Do NOT call Predict() here as it may take a GPU-optimized path
        // that skips per-layer cache population.
        Tensor<T> currentOutput = input;
        foreach (var layer in Layers)
        {
            currentOutput = layer.Forward(currentOutput);
        }

        LastLoss = _lossFunction.CalculateLoss(currentOutput.ToVector(), expectedOutput.ToVector());
        Vector<T> lossGrad = _lossFunction.CalculateDerivative(currentOutput.ToVector(), expectedOutput.ToVector());
        Tensor<T> error = Tensor<T>.FromVector(lossGrad, currentOutput.Shape);
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
                { "Architecture", "AutoInt" },
                { "NumFeatures", Architecture.CalculatedInputSize },
                { "OutputDim", Architecture.OutputSize },
                { "EmbeddingDimension", _options.EmbeddingDimension },
                { "NumHeads", _options.NumHeads },
                { "NumLayers", _options.NumLayers },
                { "AttentionDimension", _options.AttentionDimension },
                { "UseResidual", _options.UseResidual },
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
        writer.Write(_options.NumLayers);
        writer.Write(_options.NumHeads);
        writer.Write(_options.AttentionDimension);
        writer.Write(_options.DropoutRate);
        writer.Write(_options.UseResidual);
        writer.Write(_options.UseLayerNorm);
        writer.Write(_options.EmbeddingInitScale);

        writer.Write(_options.MLPHiddenDimensions.Length);
        foreach (var dim in _options.MLPHiddenDimensions)
        {
            writer.Write(dim);
        }

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
        var options = new AutoIntOptions<T>
        {
            EmbeddingDimension = reader.ReadInt32(),
            NumLayers = reader.ReadInt32(),
            NumHeads = reader.ReadInt32(),
            AttentionDimension = reader.ReadInt32(),
            DropoutRate = reader.ReadDouble(),
            UseResidual = reader.ReadBoolean(),
            UseLayerNorm = reader.ReadBoolean(),
            EmbeddingInitScale = reader.ReadDouble()
        };

        int mlpDimCount = reader.ReadInt32();
        var mlpDims = new int[mlpDimCount];
        for (int i = 0; i < mlpDimCount; i++)
        {
            mlpDims[i] = reader.ReadInt32();
        }
        options.MLPHiddenDimensions = mlpDims;

        int catCount = reader.ReadInt32();
        if (catCount > 0)
        {
            var cardinalities = new int[catCount];
            for (int i = 0; i < catCount; i++)
            {
                cardinalities[i] = reader.ReadInt32();
            }
            options.CategoricalCardinalities = cardinalities;
        }

        _options = options;
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new AutoIntNetwork<T>(
            Architecture,
            _options,
            null,
            _lossFunction);
    }
}
