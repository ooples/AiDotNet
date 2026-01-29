using System.IO;
using AiDotNet.Finance.Interfaces;
using ModelOptions = AiDotNet.Models.Options;
using AiDotNet.Finance.Base;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;

namespace AiDotNet.Finance.NLP;

/// <summary>
/// FinMA (Financial Multi-Agent) neural network model for collaborative financial task solving.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class FinMA<T> : FinancialNLPModelBase<T>
{
    #region Native Mode Fields

    private ILayer<T>? _tokenEmbedding;
    private ILayer<T>? _positionEmbedding;
    private readonly List<ILayer<T>> _decoderLayers = [];
    private ILayer<T>? _finalNorm;
    private ILayer<T>? _outputHead;

    #endregion

    #region Shared Fields

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly double _dropout;
    private readonly int _numAgents;

    #endregion

    #region Interface Properties

    /// <inheritdoc/>

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a FinMA network using a pretrained ONNX model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinMA model, FinMA sets up the architecture and options. This prepares the model for training or inference.
    /// </para>
    /// </remarks>
    public FinMA(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        ModelOptions.FinMAOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, onnxModelPath, 
               options?.MaxSequenceLength ?? 512, 
               options?.VocabularySize ?? 32000,
               options?.HiddenDimension ?? 768)
    {
        options ??= new ModelOptions.FinMAOptions<T>();
        options.Validate();

        _dropout = options.DropoutRate;
        _numAgents = options.NumAgents;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    /// <summary>
    /// Creates a FinMA network in native mode for training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinMA model, FinMA sets up the architecture and options. This prepares the model for training or inference.
    /// </para>
    /// </remarks>
    public FinMA(
        NeuralNetworkArchitecture<T> architecture,
        ModelOptions.FinMAOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, 
               options?.MaxSequenceLength ?? 512, 
               options?.VocabularySize ?? 32000,
               options?.HiddenDimension ?? 768,
               3,
               lossFunction)
    {
        options ??= new ModelOptions.FinMAOptions<T>();
        options.Validate();

        _dropout = options.DropoutRate;
        _numAgents = options.NumAgents;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Executes InitializeLayers for the FinMA.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinMA model, InitializeLayers builds and wires up model components. This sets up the FinMA architecture before use.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else if (UseNativeMode)
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultFinMALayers(
                Architecture, MaxSequenceLength, VocabularySize, _numAgents,
                HiddenDimension, 12, 12, _dropout));

            ExtractLayerReferences();
        }
    }

    /// <summary>
    /// Executes ExtractLayerReferences for the FinMA.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinMA model, ExtractLayerReferences performs a supporting step in the workflow. It keeps the FinMA architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private void ExtractLayerReferences()
    {
        int idx = 0;
        if (Layers.Count > idx) _tokenEmbedding = Layers[idx++];
        if (Layers.Count > idx) _positionEmbedding = Layers[idx++];
        idx++; // skip dropout

        _decoderLayers.Clear();
        for (int i = 0; i < 12; i++)
        {
            if (idx < Layers.Count) _decoderLayers.Add(Layers[idx++]);
            if (idx < Layers.Count) _decoderLayers.Add(Layers[idx++]);
            if (idx < Layers.Count) _decoderLayers.Add(Layers[idx++]);
            if (idx < Layers.Count) _decoderLayers.Add(Layers[idx++]);
            if (idx < Layers.Count) _decoderLayers.Add(Layers[idx++]);
            if (idx < Layers.Count) _decoderLayers.Add(Layers[idx++]);
        }

        if (idx < Layers.Count) _finalNorm = Layers[idx++];
        if (idx < Layers.Count) _outputHead = Layers[idx];
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <summary>
    /// Executes TrainCore for the FinMA.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinMA model, TrainCore performs a training step. This updates the FinMA architecture so it learns from data.
    /// </para>
    /// </remarks>
    protected override void TrainCore(Tensor<T> input, Tensor<T> target, Tensor<T> output)
    {
        SetTrainingMode(true);
        var grad = LossFunction.CalculateDerivative(output.ToVector(), target.ToVector());
        Backward(Tensor<T>.FromVector(grad, output.Shape));
        _optimizer.UpdateParameters(Layers);
        SetTrainingMode(false);
    }

    /// <summary>
    /// Executes Backward for the FinMA.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinMA model, Backward propagates gradients backward. This teaches the FinMA architecture how to adjust its weights.
    /// </para>
    /// </remarks>
    private void Backward(Tensor<T> outputGradient)
    {
        var grad = outputGradient;
        for (int i = Layers.Count - 1; i >= 0; i--) grad = Layers[i].Backward(grad);
    }

    /// <summary>
    /// Executes UpdateParameters for the FinMA.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinMA model, UpdateParameters updates internal parameters or state. This keeps the FinMA architecture aligned with the latest values.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int offset = 0;
        foreach (var layer in Layers)
        {
            var layerParams = layer.GetParameters();
            layer.SetParameters(parameters.Slice(offset, layerParams.Length));
            offset += layerParams.Length;
        }
    }

    /// <summary>
    /// Executes CreateNewInstance for the FinMA.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinMA model, CreateNewInstance builds and wires up model components. This sets up the FinMA architecture before use.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var options = new ModelOptions.FinMAOptions<T>
        {
            MaxSequenceLength = MaxSequenceLength,
            NumAgents = _numAgents,
            VocabularySize = VocabularySize,
            HiddenDimension = HiddenDimension
        };
        return new FinMA<T>(Architecture, options, _optimizer, LossFunction);
    }

    /// <summary>
    /// Executes SerializeModelSpecificData for the FinMA.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinMA model, SerializeModelSpecificData saves or restores model-specific settings. This lets the FinMA architecture be reused later.
    /// </para>
    /// </remarks>
    protected override void SerializeModelSpecificData(BinaryWriter writer)
    {
        writer.Write(_dropout);
        writer.Write(_numAgents);
    }

    /// <summary>
    /// Executes DeserializeModelSpecificData for the FinMA.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinMA model, DeserializeModelSpecificData saves or restores model-specific settings. This lets the FinMA architecture be reused later.
    /// </para>
    /// </remarks>
    protected override void DeserializeModelSpecificData(BinaryReader reader)
    {
        _ = reader.ReadDouble();
        _ = reader.ReadInt32();
    }

    /// <summary>
    /// Executes ForecastNative for the FinMA.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinMA model, ForecastNative produces predictions from input data. This is the main inference step of the FinMA architecture.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ForecastNative(Tensor<T> input, double[]? quantiles)
    {
        SetTrainingMode(false);
        var current = input;
        foreach (var layer in Layers) current = layer.Forward(current);
        return current;
    }

    /// <summary>
    /// Executes ValidateInputShape for the FinMA.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinMA model, ValidateInputShape checks inputs and configuration. This protects the FinMA architecture from mismatches and errors.
    /// </para>
    /// </remarks>
    protected override void ValidateInputShape(Tensor<T> input)
    {
        if (input.Rank < 2) throw new ArgumentException("Input must be at least 2D.");
    }

    #endregion
}
