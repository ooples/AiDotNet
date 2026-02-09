using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// TabPFN (Prior-Fitted Network) neural network for tabular data.
/// </summary>
/// <remarks>
/// <para>
/// TabPFN is a transformer model trained on synthetic data that can perform
/// in-context learning on tabular classification tasks. It approximates
/// Bayesian inference through attention mechanisms.
/// This implementation follows the standard neural network architecture pattern with:
/// - Proper inheritance from NeuralNetworkBase
/// - Layer-based architecture using ILayer components
/// - Engine-based tensor operations for CPU/GPU acceleration
/// - Full autodiff and JIT compilation support
/// </para>
/// <para>
/// <b>For Beginners:</b> TabPFN is unique because it learns from synthetic data:
///
/// Architecture:
/// 1. **Input Projection**: Convert features to embeddings
/// 2. **Deep Transformer**: Many attention layers (typically 12)
/// 3. **In-Context Learning**: Training data provided as context
/// 4. **Output Prediction**: Classification directly from attention
///
/// Key insight: TabPFN was trained on millions of synthetic classification
/// tasks, learning to "learn" from examples. At inference, you provide
/// training data as context and it predicts test labels without fine-tuning.
///
/// This makes TabPFN extremely fast for small datasets since there's no
/// traditional training loop - just a forward pass with context.
/// </para>
/// <para>
/// Reference: "TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second" (2022)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TabPFNNetwork<T> : NeuralNetworkBase<T>
{
    private readonly TabPFNOptions<T> _options;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private ILossFunction<T> _lossFunction;

    /// <summary>
    /// Gets the TabPFN-specific options.
    /// </summary>
    public TabPFNOptions<T> Options => _options;

    /// <summary>
    /// Gets the embedding dimension.
    /// </summary>
    public int EmbeddingDimension => _options.EmbeddingDimension;

    /// <summary>
    /// Gets the number of transformer layers.
    /// </summary>
    public int NumLayers => _options.NumLayers;

    /// <summary>
    /// Initializes a new TabPFN network with the specified architecture.
    /// </summary>
    public TabPFNNetwork(
        NeuralNetworkArchitecture<T> architecture,
        TabPFNOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), maxGradNorm)
    {
        _options = options ?? new TabPFNOptions<T>();
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

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
            Layers.AddRange(LayerHelper<T>.CreateDefaultTabPFNLayers(
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
        LastLoss = _lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());
        Tensor<T> error = prediction.Subtract(expectedOutput);
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
                { "Architecture", "TabPFN" },
                { "NumFeatures", Architecture.CalculatedInputSize },
                { "OutputDim", Architecture.OutputSize },
                { "EmbeddingDimension", _options.EmbeddingDimension },
                { "NumHeads", _options.NumHeads },
                { "NumLayers", _options.NumLayers },
                { "MaxContextSamples", _options.MaxContextSamples },
                { "MaxClasses", _options.MaxClasses },
                { "UseEnsemble", _options.UseEnsemble },
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
        writer.Write(_options.FeedForwardMultiplier);
        writer.Write(_options.DropoutRate);
        writer.Write(_options.MaxFeatures);
        writer.Write(_options.MaxContextSamples);
        writer.Write(_options.MaxClasses);
        writer.Write(_options.UsePositionalEncoding);
        writer.Write(_options.UsePreNorm);
        writer.Write(_options.InitScale);
        writer.Write(_options.UseEnsemble);
        writer.Write(_options.NumEnsembles);

        writer.Write(_options.OutputHeadDimensions.Length);
        foreach (var dim in _options.OutputHeadDimensions)
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
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new TabPFNNetwork<T>(
            Architecture,
            _options,
            _optimizer,
            _lossFunction);
    }
}
