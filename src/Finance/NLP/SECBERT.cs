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
/// SEC-BERT neural network model for domain-specific financial language processing.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// SEC-BERT is a BERT-based model specifically pretrained on SEC filings (10-K, 10-Q, etc.).
/// </para>
/// <para>
/// Reference: Loukas et al., "SEC-BERT: A Pre-trained Financial Language Model", 2022.
/// </para>
/// </remarks>
public class SECBERT<T> : FinancialNLPModelBase<T>
{
    #region Native Mode Fields

    private ILayer<T>? _wordEmbedding;
    private ILayer<T>? _positionEmbedding;
    private ILayer<T>? _typeEmbedding;
    private readonly List<ILayer<T>> _transformerLayers = [];
    private ILayer<T>? _pooler;
    private ILayer<T>? _taskHead;

    #endregion

    #region Shared Fields

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly double _dropout;

    #endregion

    #region Interface Properties

    /// <inheritdoc/>

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a SEC-BERT network using a pretrained ONNX model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the SECBERT model, SECBERT sets up the architecture and options. This prepares the model for training or inference.
    /// </para>
    /// </remarks>
    public SECBERT(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        ModelOptions.SECBERTOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, onnxModelPath, 
               options?.MaxSequenceLength ?? 512, 
               options?.VocabularySize ?? 30522,
               options?.HiddenSize ?? 768)
    {
        options ??= new ModelOptions.SECBERTOptions<T>();
        options.Validate();

        _dropout = options.DropoutProbability;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    /// <summary>
    /// Creates a SEC-BERT network in native mode for training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the SECBERT model, SECBERT sets up the architecture and options. This prepares the model for training or inference.
    /// </para>
    /// </remarks>
    public SECBERT(
        NeuralNetworkArchitecture<T> architecture,
        ModelOptions.SECBERTOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, 
               options?.MaxSequenceLength ?? 512, 
               options?.VocabularySize ?? 30522,
               options?.HiddenSize ?? 768,
               3, // numSentimentClasses
               lossFunction)
    {
        options ??= new ModelOptions.SECBERTOptions<T>();
        options.Validate();

        _dropout = options.DropoutProbability;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Executes InitializeLayers for the SECBERT.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the SECBERT model, InitializeLayers builds and wires up model components. This sets up the SECBERT architecture before use.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultSECBERTLayers(
                Architecture, _maxSequenceLength, _vocabularySize, _hiddenDimension,
                12, 12, _dropout)); // Default heads/layers

            ExtractLayerReferences();
        }
    }

    /// <summary>
    /// Executes ExtractLayerReferences for the SECBERT.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the SECBERT model, ExtractLayerReferences performs a supporting step in the workflow. It keeps the SECBERT architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private void ExtractLayerReferences()
    {
        int idx = 0;
        if (Layers.Count > idx) _wordEmbedding = Layers[idx++];
        if (Layers.Count > idx) _positionEmbedding = Layers[idx++];
        if (Layers.Count > idx) _typeEmbedding = Layers[idx++];
        idx += 2; // skip norm/dropout

        _transformerLayers.Clear();
        for (int i = 0; i < 12; i++)
        {
            if (idx < Layers.Count) _transformerLayers.Add(Layers[idx++]);
            if (idx < Layers.Count) _transformerLayers.Add(Layers[idx++]);
            if (idx < Layers.Count) _transformerLayers.Add(Layers[idx++]);
            if (idx < Layers.Count) _transformerLayers.Add(Layers[idx++]);
            if (idx < Layers.Count) _transformerLayers.Add(Layers[idx++]);
            if (idx < Layers.Count) _transformerLayers.Add(Layers[idx++]);
        }

        if (idx < Layers.Count) _pooler = Layers[idx++];
        if (idx < Layers.Count) _taskHead = Layers[idx];
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <summary>
    /// Executes TrainCore for the SECBERT.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the SECBERT model, TrainCore performs a training step. This updates the SECBERT architecture so it learns from data.
    /// </para>
    /// </remarks>
    protected override void TrainCore(Tensor<T> input, Tensor<T> target, Tensor<T> output)
    {
        SetTrainingMode(true);
        var grad = LossFunction.CalculateDerivative(output.ToVector(), target.ToVector());
        Backward(Tensor<T>.FromVector(grad));
        _optimizer.UpdateParameters(Layers);
        SetTrainingMode(false);
    }

    /// <summary>
    /// Executes Backward for the SECBERT.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the SECBERT model, Backward propagates gradients backward. This teaches the SECBERT architecture how to adjust its weights.
    /// </para>
    /// </remarks>
    private void Backward(Tensor<T> outputGradient)
    {
        var grad = outputGradient;
        for (int i = Layers.Count - 1; i >= 0; i--) grad = Layers[i].Backward(grad);
    }

    /// <summary>
    /// Executes UpdateParameters for the SECBERT.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the SECBERT model, UpdateParameters updates internal parameters or state. This keeps the SECBERT architecture aligned with the latest values.
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
    /// Executes CreateNewInstance for the SECBERT.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the SECBERT model, CreateNewInstance builds and wires up model components. This sets up the SECBERT architecture before use.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var options = new ModelOptions.SECBERTOptions<T> { MaxSequenceLength = _maxSequenceLength, VocabularySize = _vocabularySize, HiddenSize = _hiddenDimension };
        return new SECBERT<T>(Architecture, options, _optimizer, LossFunction);
    }

    /// <summary>
    /// Executes SerializeModelSpecificData for the SECBERT.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the SECBERT model, SerializeModelSpecificData saves or restores model-specific settings. This lets the SECBERT architecture be reused later.
    /// </para>
    /// </remarks>
    protected override void SerializeModelSpecificData(BinaryWriter writer)
    {
        writer.Write(_dropout);
    }

    /// <summary>
    /// Executes DeserializeModelSpecificData for the SECBERT.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the SECBERT model, DeserializeModelSpecificData saves or restores model-specific settings. This lets the SECBERT architecture be reused later.
    /// </para>
    /// </remarks>
    protected override void DeserializeModelSpecificData(BinaryReader reader)
    {
        _ = reader.ReadDouble();
    }

    /// <summary>
    /// Executes ForecastNative for the SECBERT.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the SECBERT model, ForecastNative produces predictions from input data. This is the main inference step of the SECBERT architecture.
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
    /// Executes ValidateInputShape for the SECBERT.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the SECBERT model, ValidateInputShape checks inputs and configuration. This protects the SECBERT architecture from mismatches and errors.
    /// </para>
    /// </remarks>
    protected override void ValidateInputShape(Tensor<T> input)
    {
        if (input.Rank < 2) throw new ArgumentException("Input must be at least 2D.");
    }

    #endregion
}
