using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// Mambular (State Space Model for Tabular Data) neural network.
/// </summary>
/// <remarks>
/// <para>
/// Mambular applies the Mamba state space model architecture to tabular data,
/// treating features as a sequence and using selective state spaces for processing.
/// This implementation follows the standard neural network architecture pattern with:
/// - Proper inheritance from NeuralNetworkBase
/// - Layer-based architecture using ILayer components
/// - Engine-based tensor operations for CPU/GPU acceleration
/// - Full autodiff and JIT compilation support
/// </para>
/// <para>
/// <b>For Beginners:</b> Mambular treats features like a sequence:
///
/// Architecture:
/// 1. **Feature Embedding**: Convert features to learned representations
/// 2. **State Space Layers**: Process features sequentially with memory
/// 3. **Selective Mechanism**: Learn which features to remember/forget
/// 4. **MLP Head**: Final prediction from processed features
///
/// Key insight: State space models (like Mamba) are more efficient than
/// transformers for long sequences. For tabular data with many features,
/// this can provide both better scaling and learned sequential relationships.
/// </para>
/// <para>
/// Reference: "Mambular: A Sequential Model for Tabular Deep Learning" (2024)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class MambularNetwork<T> : NeuralNetworkBase<T>
{
    private readonly MambularOptions<T> _options;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private ILossFunction<T> _lossFunction;

    /// <summary>
    /// Gets the Mambular-specific options.
    /// </summary>
    public new MambularOptions<T> Options => _options;

    /// <summary>
    /// Gets the embedding dimension.
    /// </summary>
    public int EmbeddingDimension => _options.EmbeddingDimension;

    /// <summary>
    /// Gets the state dimension for the SSM.
    /// </summary>
    public int StateDimension => _options.StateDimension;

    /// <summary>
    /// Gets the number of layers.
    /// </summary>
    public int NumLayers => _options.NumLayers;

    /// <summary>
    /// Initializes a new Mambular network with the specified architecture.
    /// </summary>
    public MambularNetwork(
        NeuralNetworkArchitecture<T> architecture,
        MambularOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), maxGradNorm)
    {
        _options = options ?? new MambularOptions<T>();
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultMambularLayers(
                Architecture,
                numFeatures: Architecture.CalculatedInputSize,
                embeddingDimension: _options.EmbeddingDimension,
                stateDimension: _options.StateDimension,
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
                { "Architecture", "Mambular" },
                { "NumFeatures", Architecture.CalculatedInputSize },
                { "OutputDim", Architecture.OutputSize },
                { "EmbeddingDimension", _options.EmbeddingDimension },
                { "StateDimension", _options.StateDimension },
                { "NumLayers", _options.NumLayers },
                { "ExpansionFactor", _options.ExpansionFactor },
                { "UseBidirectional", _options.UseBidirectional },
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
        writer.Write(_options.StateDimension);
        writer.Write(_options.NumLayers);
        writer.Write(_options.ExpansionFactor);
        writer.Write(_options.ConvKernelSize);
        writer.Write(_options.DropoutRate);
        writer.Write(_options.InitScale);
        writer.Write(_options.DeltaMin);
        writer.Write(_options.DeltaMax);
        writer.Write(_options.UseBidirectional);

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
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new MambularNetwork<T>(
            Architecture,
            _options,
            _optimizer,
            _lossFunction);
    }
}
