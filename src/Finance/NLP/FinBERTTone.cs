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
/// FinBERT-tone neural network model specialized for financial sentiment analysis.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class FinBERTTone<T> : FinancialNLPModelBase<T>
{
    #region Native Mode Fields

    private ILayer<T>? _wordEmbedding;
    private ILayer<T>? _positionEmbedding;
    private ILayer<T>? _typeEmbedding;
    private readonly List<ILayer<T>> _transformerLayers = [];
    private ILayer<T>? _pooler;
    private ILayer<T>? _sentimentHead;

    #endregion

    #region Shared Fields

    private readonly ModelOptions.FinBERTToneOptions<T> _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private double _dropout;

    /// <inheritdoc/>
    public override AiDotNet.Models.Options.ModelOptions GetOptions() => _options;

    #endregion

    #region Interface Properties

    /// <inheritdoc/>

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a FinBERT-tone network using a pretrained ONNX model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinBERTTone model, FinBERTTone sets up the architecture and options. This prepares the model for training or inference.
    /// </para>
    /// </remarks>
    public FinBERTTone(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        ModelOptions.FinBERTToneOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, onnxModelPath, 
               options?.MaxSequenceLength ?? 512, 
               options?.VocabularySize ?? 30522,
               options?.HiddenDimension ?? 768,
               options?.NumToneClasses ?? 5)
    {
        options ??= new ModelOptions.FinBERTToneOptions<T>();
        _options = options;
        Options = _options;
        options.Validate();

        _dropout = options.DropoutRate;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    /// <summary>
    /// Creates a FinBERT-tone network in native mode for training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinBERTTone model, FinBERTTone sets up the architecture and options. This prepares the model for training or inference.
    /// </para>
    /// </remarks>
    public FinBERTTone(
        NeuralNetworkArchitecture<T> architecture,
        ModelOptions.FinBERTToneOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture,
               options?.MaxSequenceLength ?? 512,
               options?.VocabularySize ?? 30522,
               options?.HiddenDimension ?? 768,
               options?.NumToneClasses ?? 5,
               lossFunction)
    {
        options ??= new ModelOptions.FinBERTToneOptions<T>();
        _options = options;
        Options = _options;
        options.Validate();

        _dropout = options.DropoutRate;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Executes InitializeLayers for the FinBERTTone.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinBERTTone model, InitializeLayers builds and wires up model components. This sets up the FinBERTTone architecture before use.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultFinBERTToneLayers(
                Architecture, MaxSequenceLength, VocabularySize, NumSentimentClasses, 
                HiddenDimension, 12, 12, _dropout));

            ExtractLayerReferences();
        }
    }

    /// <summary>
    /// Executes ExtractLayerReferences for the FinBERTTone.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinBERTTone model, ExtractLayerReferences performs a supporting step in the workflow. It keeps the FinBERTTone architecture pipeline consistent.
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
        if (idx < Layers.Count) _sentimentHead = Layers[idx];
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <summary>
    /// Executes TrainCore for the FinBERTTone.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinBERTTone model, TrainCore performs a training step. This updates the FinBERTTone architecture so it learns from data.
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
    /// Executes Backward for the FinBERTTone.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinBERTTone model, Backward propagates gradients backward. This teaches the FinBERTTone architecture how to adjust its weights.
    /// </para>
    /// </remarks>
    private void Backward(Tensor<T> outputGradient)
    {
        var grad = outputGradient;
        for (int i = Layers.Count - 1; i >= 0; i--) grad = Layers[i].Backward(grad);
    }

    /// <summary>
    /// Executes UpdateParameters for the FinBERTTone.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinBERTTone model, UpdateParameters updates internal parameters or state. This keeps the FinBERTTone architecture aligned with the latest values.
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
    /// Executes CreateNewInstance for the FinBERTTone.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinBERTTone model, CreateNewInstance builds and wires up model components. This sets up the FinBERTTone architecture before use.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var options = new ModelOptions.FinBERTToneOptions<T>
        {
            MaxSequenceLength = MaxSequenceLength,
            VocabularySize = VocabularySize,
            HiddenDimension = HiddenDimension,
            NumToneClasses = NumSentimentClasses
        };
        return new FinBERTTone<T>(Architecture, options, _optimizer, LossFunction);
    }

    /// <summary>
    /// Executes SerializeModelSpecificData for the FinBERTTone.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinBERTTone model, SerializeModelSpecificData saves or restores model-specific settings. This lets the FinBERTTone architecture be reused later.
    /// </para>
    /// </remarks>
    protected override void SerializeModelSpecificData(BinaryWriter writer)
    {
        writer.Write(_dropout);
    }

    /// <summary>
    /// Executes DeserializeModelSpecificData for the FinBERTTone.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinBERTTone model, DeserializeModelSpecificData saves or restores model-specific settings. This lets the FinBERTTone architecture be reused later.
    /// </para>
    /// </remarks>
    protected override void DeserializeModelSpecificData(BinaryReader reader)
    {
        _dropout = reader.ReadDouble();
    }

    /// <summary>
    /// Executes ForecastNative for the FinBERTTone.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinBERTTone model, ForecastNative produces predictions from input data. This is the main inference step of the FinBERTTone architecture.
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
    /// Executes ValidateInputShape for the FinBERTTone.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FinBERTTone model, ValidateInputShape checks inputs and configuration. This protects the FinBERTTone architecture from mismatches and errors.
    /// </para>
    /// </remarks>
    protected override void ValidateInputShape(Tensor<T> input)
    {
        if (input.Rank < 2) throw new ArgumentException("Input must be at least 2D.");
    }

    #endregion
}
