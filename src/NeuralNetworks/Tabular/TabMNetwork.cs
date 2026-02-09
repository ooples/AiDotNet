using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// TabM (Parameter-Efficient Ensemble) neural network for tabular data.
/// </summary>
/// <remarks>
/// <para>
/// TabM uses BatchEnsemble-style parameter sharing to create multiple ensemble members
/// with minimal parameter overhead. Each member shares base weights but has its own
/// small rank vectors that modulate the shared weights.
/// This implementation follows the standard neural network architecture pattern with:
/// - Proper inheritance from NeuralNetworkBase
/// - Layer-based architecture using ILayer components
/// - Engine-based tensor operations for CPU/GPU acceleration
/// - Full autodiff and JIT compilation support
/// </para>
/// <para>
/// <b>For Beginners:</b> TabM gets ensemble benefits efficiently:
///
/// Architecture:
/// 1. **Shared Base Weights**: Core network weights shared by all ensemble members
/// 2. **Rank Vectors**: Small per-member vectors that customize each member
/// 3. **Weight Modulation**: Rank vectors scale the shared weights
/// 4. **Ensemble Aggregation**: Average or concatenate member predictions
///
/// Key insight: Traditional ensembles need N times the parameters for N models.
/// TabM only adds about 1-5% extra parameters per member while getting similar
/// benefits. This is like having multiple experts share most of their knowledge
/// but each having their own "style" or perspective.
/// </para>
/// <para>
/// Reference: "TabM: Advancing Tabular Deep Learning With Parameter-Efficient Ensembling" (2024)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class TabMNetwork<T> : NeuralNetworkBase<T>
{
    private readonly TabMOptions<T> _options;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private ILossFunction<T> _lossFunction;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// Gets the number of ensemble members.
    /// </summary>
    public int NumEnsembleMembers => _options.NumEnsembleMembers;

    /// <summary>
    /// Initializes a new TabM network with the specified architecture.
    /// </summary>
    public TabMNetwork(
        NeuralNetworkArchitecture<T> architecture,
        TabMOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), maxGradNorm)
    {
        _options = options ?? new TabMOptions<T>();
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultTabMLayers(
                Architecture,
                numFeatures: Architecture.CalculatedInputSize,
                hiddenDimensions: _options.HiddenDimensions,
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
                { "Architecture", "TabM" },
                { "NumFeatures", Architecture.CalculatedInputSize },
                { "OutputDim", Architecture.OutputSize },
                { "NumEnsembleMembers", _options.NumEnsembleMembers },
                { "HiddenDimensions", _options.HiddenDimensions },
                { "UseLayerNorm", _options.UseLayerNorm },
                { "AverageEnsemble", _options.AverageEnsemble },
                { "UseFeatureEmbeddings", _options.UseFeatureEmbeddings },
                { "LayerCount", Layers.Count },
                { "LayerTypes", Layers.Select(l => l.GetType().Name).ToArray() }
            },
            ModelData = this.Serialize()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_options.NumEnsembleMembers);
        writer.Write(_options.DropoutRate);
        writer.Write(_options.UseLayerNorm);
        writer.Write(_options.RankInitScale);
        writer.Write(_options.UseBias);
        writer.Write(_options.ActivationType);
        writer.Write(_options.AverageEnsemble);
        writer.Write(_options.UseFeatureEmbeddings);
        writer.Write(_options.FeatureEmbeddingDimension);
        writer.Write(_options.EnableGradientClipping);
        writer.Write(_options.MaxGradientNorm);
        writer.Write(_options.WeightDecay);

        writer.Write(_options.HiddenDimensions.Length);
        foreach (var dim in _options.HiddenDimensions)
        {
            writer.Write(dim);
        }
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new TabMNetwork<T>(
            Architecture,
            _options,
            _optimizer,
            _lossFunction);
    }
}
