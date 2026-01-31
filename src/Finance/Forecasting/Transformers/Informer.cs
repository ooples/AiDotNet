using System.IO;
using AiDotNet.Finance.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

using AiDotNet.Finance.Base;
namespace AiDotNet.Finance.Forecasting.Transformers;

/// <summary>
/// Informer (Efficient Transformer for Long Sequence Forecasting) neural network.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// Informer was the first transformer specifically designed for long sequence time series
/// forecasting. It achieved AAAI 2021 Best Paper for its innovations in efficient attention.
/// </para>
/// <para>
/// <b>For Beginners:</b> In regular transformers, every position looks at every other position
/// (O(n²) complexity), which is slow for long sequences. Informer is clever - it figures out
/// which positions are most "active" (have high variance in their attention scores) and only
/// computes attention for those.
///
/// Key innovations:
/// - <b>ProbSparse Attention:</b> Only compute attention for top-k important queries
/// - <b>Self-attention Distilling:</b> Progressively reduce sequence length with max-pooling
/// - <b>Generative Decoder:</b> Predict all future values at once, not step-by-step
/// </para>
/// <para>
/// <b>Reference:</b> Zhou et al., "Informer: Beyond Efficient Transformer for Long Sequence
/// Time-Series Forecasting", AAAI 2021 (Best Paper). https://arxiv.org/abs/2012.07436
/// </para>
/// </remarks>
public class Informer<T> : ForecastingModelBase<T>
{
    #region Execution Mode

    /// <summary>
    /// Indicates whether this network uses native layers (true) or ONNX model (false).
    /// </summary>
    private readonly bool _useNativeMode;

    #endregion

    
    #region Native Mode Fields

    /// <summary>
    /// Input embedding layer.
    /// </summary>
    private ILayer<T>? _inputEmbedding;

    /// <summary>
    /// Encoder layers with ProbSparse attention.
    /// </summary>
    private readonly List<ILayer<T>> _encoderLayers = [];

    /// <summary>
    /// Decoder layers.
    /// </summary>
    private readonly List<ILayer<T>> _decoderLayers = [];

    /// <summary>
    /// Final layer normalization.
    /// </summary>
    private ILayer<T>? _finalNorm;

    /// <summary>
    /// Output projection layer.
    /// </summary>
    private ILayer<T>? _outputProjection;

    /// <summary>
    /// Instance normalization mean (for RevIN).
    /// </summary>
    private Tensor<T>? _instanceMean;

    /// <summary>
    /// Instance normalization standard deviation (for RevIN).
    /// </summary>
    private Tensor<T>? _instanceStd;

    #endregion

    #region Shared Fields

    /// <summary>
    /// The optimizer for training the model.
    /// </summary>
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <summary>
    /// The loss function for computing prediction errors.
    /// </summary>
    private readonly ILossFunction<T> _lossFunction;

    private int _sequenceLength;
    private int _labelLength;
    private int _predictionHorizon;
    private int _numFeatures;
    private int _numEncoderLayers;
    private int _numDecoderLayers;
    private int _numHeads;
    private int _modelDimension;
    private int _feedForwardDimension;
    private int _distillingFactor;
    private double _dropout;
    private bool _useInstanceNormalization;

    #endregion

    #region IForecastingModel Properties

    /// <inheritdoc/>
    public override int SequenceLength => _sequenceLength;

    /// <inheritdoc/>
    public override int PredictionHorizon => _predictionHorizon;

    /// <inheritdoc/>
    public override int NumFeatures => _numFeatures;

    /// <inheritdoc/>
    public override int PatchSize => 1;

    /// <inheritdoc/>
    public override int Stride => 1;

    /// <inheritdoc/>
    public override bool IsChannelIndependent => false;

    /// <inheritdoc/>
    public override bool UseNativeMode => _useNativeMode;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates an Informer network using a pretrained ONNX model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Informer model, Informer sets up the architecture and options. This prepares the model for training or inference.
    /// </para>
    /// </remarks>
    public Informer(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        InformerOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentNullException(nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model file not found: {onnxModelPath}", onnxModelPath);

        options ??= new InformerOptions<T>();

        _useNativeMode = false;
        OnnxSession = new InferenceSession(onnxModelPath);
        OnnxModelPath = onnxModelPath;

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        _sequenceLength = options.LookbackWindow;
        _labelLength = options.LookbackWindow / 2;
        _predictionHorizon = options.ForecastHorizon;
        _numFeatures = architecture.InputSize;
        _numEncoderLayers = options.NumEncoderLayers;
        _numDecoderLayers = options.NumDecoderLayers;
        _numHeads = options.NumAttentionHeads;
        _modelDimension = options.EmbeddingDim;
        _feedForwardDimension = options.EmbeddingDim * 4;
        _distillingFactor = options.DistillingFactor;
        _dropout = options.DropoutRate;
        _useInstanceNormalization = true;
    }

    /// <summary>
    /// Creates an Informer network in native mode for training from scratch.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Informer model, Informer sets up the architecture and options. This prepares the model for training or inference.
    /// </para>
    /// </remarks>
    public Informer(
        NeuralNetworkArchitecture<T> architecture,
        InformerOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new InformerOptions<T>();

        _useNativeMode = true;
        OnnxSession = null;
        OnnxModelPath = null;

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        _sequenceLength = options.LookbackWindow;
        _labelLength = options.LookbackWindow / 2;
        _predictionHorizon = options.ForecastHorizon;
        _numFeatures = architecture.InputSize;
        _numEncoderLayers = options.NumEncoderLayers;
        _numDecoderLayers = options.NumDecoderLayers;
        _numHeads = options.NumAttentionHeads;
        _modelDimension = options.EmbeddingDim;
        _feedForwardDimension = options.EmbeddingDim * 4;
        _distillingFactor = options.DistillingFactor;
        _dropout = options.DropoutRate;
        _useInstanceNormalization = true;

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes the layers for native mode operation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Informer model, InitializeLayers builds and wires up model components. This sets up the Informer architecture before use.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else if (_useNativeMode)
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultInformerLayers(
                Architecture,
                _sequenceLength,
                _predictionHorizon,
                _numFeatures,
                _numEncoderLayers,
                _numDecoderLayers,
                _numHeads,
                _modelDimension,
                _feedForwardDimension));

            ExtractLayerReferences();
        }
    }

    /// <summary>
    /// Extracts references to specific layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Informer model, ExtractLayerReferences performs a supporting step in the workflow. It keeps the Informer architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private void ExtractLayerReferences()
    {
        if (Layers.Count == 0)
            return;

        _inputEmbedding = Layers[0];

        int encoderStartIndex = 1;
        int encoderEndIndex = encoderStartIndex + _numEncoderLayers;
        for (int i = encoderStartIndex; i < encoderEndIndex && i < Layers.Count; i++)
        {
            _encoderLayers.Add(Layers[i]);
        }

        int decoderStartIndex = encoderEndIndex;
        int decoderEndIndex = decoderStartIndex + _numDecoderLayers;
        for (int i = decoderStartIndex; i < decoderEndIndex && i < Layers.Count; i++)
        {
            _decoderLayers.Add(Layers[i]);
        }

        int normIndex = decoderEndIndex;
        if (normIndex < Layers.Count)
            _finalNorm = Layers[normIndex];

        int projIndex = normIndex + 1;
        if (projIndex < Layers.Count)
            _outputProjection = Layers[projIndex];
    }

    /// <summary>
    /// Validates custom layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Informer model, ValidateCustomLayers checks inputs and configuration. This protects the Informer architecture from mismatches and errors.
    /// </para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);

        if (layers.Count < 3)
            throw new ArgumentException("Informer requires at least 3 layers: embedding, encoder, and output.");
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <inheritdoc/>
    public override bool SupportsTraining => _useNativeMode;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Informer model, Predict produces predictions from input data. This is the main inference step of the Informer architecture.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return Forecast(input);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Informer model, Train performs a training step. This updates the Informer architecture so it learns from data.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is not supported in ONNX mode.");

        SetTrainingMode(true);

        var prediction = Forward(input);
        LastLoss = _lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());

        var outputGradient = _lossFunction.CalculateDerivative(prediction.ToVector(), expectedOutput.ToVector());
        Backward(Tensor<T>.FromVector(outputGradient, prediction.Shape));

        _optimizer.UpdateParameters(Layers);

        SetTrainingMode(false);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Informer model, UpdateParameters updates internal parameters or state. This keeps the Informer architecture aligned with the latest values.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (parameters is null)
            throw new ArgumentNullException(nameof(parameters));

        int offset = 0;
        foreach (var layer in Layers)
        {
            var layerParams = layer.GetParameters();
            var newParams = parameters.Slice(offset, layerParams.Length);
            layer.SetParameters(newParams);
            offset += layerParams.Length;
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Informer model, GetModelMetadata performs a supporting step in the workflow. It keeps the Informer architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "Informer" },
                { "SequenceLength", _sequenceLength },
                { "LabelLength", _labelLength },
                { "PredictionHorizon", _predictionHorizon },
                { "NumFeatures", _numFeatures },
                { "NumEncoderLayers", _numEncoderLayers },
                { "NumDecoderLayers", _numDecoderLayers },
                { "NumHeads", _numHeads },
                { "ModelDimension", _modelDimension },
                { "DistillingFactor", _distillingFactor },
                { "UseNativeMode", _useNativeMode },
                { "ParameterCount", GetParameterCount() }
            },
            ModelData = _useNativeMode ? this.Serialize() : Array.Empty<byte>()
        };
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Informer model, CreateNewInstance builds and wires up model components. This sets up the Informer architecture before use.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var options = new InformerOptions<T>
        {
            LookbackWindow = _sequenceLength,
            ForecastHorizon = _predictionHorizon,
            NumEncoderLayers = _numEncoderLayers,
            NumDecoderLayers = _numDecoderLayers,
            NumAttentionHeads = _numHeads,
            EmbeddingDim = _modelDimension,
            DistillingFactor = _distillingFactor,
            DropoutRate = _dropout
        };

        return _useNativeMode
            ? new Informer<T>(Architecture, options, _optimizer, _lossFunction)
            : new Informer<T>(Architecture, OnnxModelPath!, options, _optimizer, _lossFunction);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Informer model, SerializeNetworkSpecificData saves or restores model-specific settings. This lets the Informer architecture be reused later.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_sequenceLength);
        writer.Write(_labelLength);
        writer.Write(_predictionHorizon);
        writer.Write(_numFeatures);
        writer.Write(_numEncoderLayers);
        writer.Write(_numDecoderLayers);
        writer.Write(_numHeads);
        writer.Write(_modelDimension);
        writer.Write(_feedForwardDimension);
        writer.Write(_distillingFactor);
        writer.Write(_dropout);
        writer.Write(_useInstanceNormalization);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Informer model, DeserializeNetworkSpecificData saves or restores model-specific settings. This lets the Informer architecture be reused later.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _sequenceLength = reader.ReadInt32();
        _labelLength = reader.ReadInt32();
        _predictionHorizon = reader.ReadInt32();
        _numFeatures = reader.ReadInt32();
        _numEncoderLayers = reader.ReadInt32();
        _numDecoderLayers = reader.ReadInt32();
        _numHeads = reader.ReadInt32();
        _modelDimension = reader.ReadInt32();
        _feedForwardDimension = reader.ReadInt32();
        _distillingFactor = reader.ReadInt32();
        _dropout = reader.ReadDouble();
        _useInstanceNormalization = reader.ReadBoolean();
    }

    #endregion

    #region IForecastingModel Implementation

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Informer model, Forecast produces predictions from input data. This is the main inference step of the Informer architecture.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forecast(Tensor<T> input, double[]? quantiles = null)
    {
        if (input is null)
            throw new ArgumentNullException(nameof(input));

        return _useNativeMode ? ForecastNative(input, quantiles) : ForecastOnnx(input);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Informer model, AutoregressiveForecast produces predictions from input data. This is the main inference step of the Informer architecture.
    /// </para>
    /// </remarks>
    public override Tensor<T> AutoregressiveForecast(Tensor<T> input, int steps)
    {
        if (input is null)
            throw new ArgumentNullException(nameof(input));
        if (steps < 1)
            throw new ArgumentOutOfRangeException(nameof(steps), "Steps must be at least 1.");

        var currentInput = input;
        var allPredictions = new List<Tensor<T>>();

        int stepsRemaining = steps;
        while (stepsRemaining > 0)
        {
            var forecast = Forecast(currentInput);
            int stepsToUse = Math.Min(stepsRemaining, _predictionHorizon);
            allPredictions.Add(forecast);
            stepsRemaining -= stepsToUse;

            if (stepsRemaining > 0)
            {
                currentInput = ShiftInputWithPredictions(currentInput, forecast, stepsToUse);
            }
        }

        return ConcatenatePredictions(allPredictions, steps);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Informer model, Evaluate performs a supporting step in the workflow. It keeps the Informer architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> Evaluate(Tensor<T> inputs, Tensor<T> targets)
    {
        if (inputs is null)
            throw new ArgumentNullException(nameof(inputs));
        if (targets is null)
            throw new ArgumentNullException(nameof(targets));

        var predictions = Forecast(inputs);
        var metrics = new Dictionary<string, T>();

        T maeSum = NumOps.Zero;
        int count = predictions.Length;
        for (int i = 0; i < count; i++)
        {
            T diff = NumOps.Subtract(predictions.Data.Span[i], targets.Data.Span[i]);
            maeSum = NumOps.Add(maeSum, NumOps.Abs(diff));
        }
        metrics["MAE"] = NumOps.Divide(maeSum, NumOps.FromDouble(count));

        T mseSum = NumOps.Zero;
        for (int i = 0; i < count; i++)
        {
            T diff = NumOps.Subtract(predictions.Data.Span[i], targets.Data.Span[i]);
            mseSum = NumOps.Add(mseSum, NumOps.Multiply(diff, diff));
        }
        metrics["MSE"] = NumOps.Divide(mseSum, NumOps.FromDouble(count));
        metrics["RMSE"] = NumOps.Sqrt(metrics["MSE"]);

        return metrics;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Informer model, ApplyInstanceNormalization performs a supporting step in the workflow. It keeps the Informer architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
    {
        if (!_useInstanceNormalization)
            return input;

        return ApplyRevIN(input, normalize: true);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Informer model, GetFinancialMetrics calculates evaluation metrics. This summarizes how the Informer architecture is performing.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> GetFinancialMetrics()
    {
        return new Dictionary<string, T>
        {
            ["SequenceLength"] = NumOps.FromDouble(_sequenceLength),
            ["PredictionHorizon"] = NumOps.FromDouble(_predictionHorizon),
            ["NumFeatures"] = NumOps.FromDouble(_numFeatures),
            ["ParameterCount"] = NumOps.FromDouble(GetParameterCount())
        };
    }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// Performs the forward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Informer model, Forward runs the forward pass through the layers. This moves data through the Informer architecture to compute outputs.
    /// </para>
    /// </remarks>
    private Tensor<T> Forward(Tensor<T> input)
    {
        var current = _useInstanceNormalization ? ApplyRevIN(input, normalize: true) : input;

        if (_inputEmbedding is not null)
        {
            current = _inputEmbedding.Forward(current);
        }

        foreach (var layer in _encoderLayers)
        {
            current = layer.Forward(current);
            current = ApplyDistilling(current);
        }

        var encoderOutput = current;
        var decoderInput = PrepareDecoderInput(input, encoderOutput);

        current = decoderInput;
        foreach (var layer in _decoderLayers)
        {
            current = layer.Forward(current);
        }

        if (_finalNorm is not null)
        {
            current = _finalNorm.Forward(current);
        }

        if (_outputProjection is not null)
        {
            current = _outputProjection.Forward(current);
        }

        current = ExtractPrediction(current);

        if (_useInstanceNormalization)
        {
            current = ApplyRevIN(current, normalize: false);
        }

        return current;
    }

    /// <summary>
    /// Performs the backward pass.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Informer model, Backward propagates gradients backward. This teaches the Informer architecture how to adjust its weights.
    /// </para>
    /// </remarks>
    private Tensor<T> Backward(Tensor<T> outputGradient)
    {
        var gradient = outputGradient;

        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            gradient = Layers[i].Backward(gradient);
        }

        return gradient;
    }

    /// <summary>
    /// Performs native mode forecasting.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Informer model, ForecastNative produces predictions from input data. This is the main inference step of the Informer architecture.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ForecastNative(Tensor<T> input, double[]? quantiles)
    {
        SetTrainingMode(false);
        return Forward(input);
    }

    /// <summary>
    /// Performs ONNX mode forecasting.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Informer model, ForecastOnnx produces predictions from input data. This is the main inference step of the Informer architecture.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ForecastOnnx(Tensor<T> input)
    {
        if (OnnxSession is null)
            throw new InvalidOperationException("ONNX session is not initialized.");

        var inputName = OnnxSession.InputMetadata.Keys.First();
        var inputShape = input.Shape.Select(d => (long)d).ToArray();
        var onnxInput = new OnnxTensors.DenseTensor<float>(
            input.ToArray().Select(x => Convert.ToSingle(x)).ToArray(),
            inputShape.Select(d => (int)d).ToArray());

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, onnxInput)
        };

        using var results = OnnxSession.Run(inputs);
        var output = results.First().AsTensor<float>();
        var outputArray = output.ToArray().Select(x => NumOps.FromDouble(x)).ToArray();
        return new Tensor<T>(outputArray, output.Dimensions.ToArray());
    }

    #endregion

    #region Model-Specific Processing

    /// <summary>
    /// Applies RevIN normalization/denormalization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Informer model, ApplyRevIN performs a supporting step in the workflow. It keeps the Informer architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private Tensor<T> ApplyRevIN(Tensor<T> input, bool normalize)
    {
        if (normalize)
        {
            int batchSize = input.Shape[0];
            int seqLen = input.Shape[1];
            int features = input.Shape[2];

            _instanceMean = new Tensor<T>(new[] { batchSize, 1, features });
            _instanceStd = new Tensor<T>(new[] { batchSize, 1, features });

            var normalized = new Tensor<T>(input.Shape);
            T epsilon = NumOps.FromDouble(1e-5);

            for (int b = 0; b < batchSize; b++)
            {
                for (int f = 0; f < features; f++)
                {
                    T sum = NumOps.Zero;
                    for (int t = 0; t < seqLen; t++)
                    {
                        sum = NumOps.Add(sum, input[b, t, f]);
                    }
                    T mean = NumOps.Divide(sum, NumOps.FromDouble(seqLen));
                    _instanceMean[b, 0, f] = mean;

                    T varSum = NumOps.Zero;
                    for (int t = 0; t < seqLen; t++)
                    {
                        var diff = NumOps.Subtract(input[b, t, f], mean);
                        varSum = NumOps.Add(varSum, NumOps.Multiply(diff, diff));
                    }
                    T std = NumOps.Sqrt(NumOps.Add(NumOps.Divide(varSum, NumOps.FromDouble(seqLen)), epsilon));
                    _instanceStd[b, 0, f] = std;

                    for (int t = 0; t < seqLen; t++)
                    {
                        normalized[b, t, f] = NumOps.Divide(NumOps.Subtract(input[b, t, f], mean), std);
                    }
                }
            }

            return normalized;
        }
        else
        {
            if (_instanceMean is null || _instanceStd is null)
                return input;

            var denormalized = new Tensor<T>(input.Shape);
            int batchSize = input.Shape[0];
            int horizonLen = input.Shape[1];
            int features = input.Shape[2];

            for (int b = 0; b < batchSize; b++)
            {
                for (int t = 0; t < horizonLen; t++)
                {
                    for (int f = 0; f < features; f++)
                    {
                        var scaled = NumOps.Multiply(input[b, t, f], _instanceStd[b, 0, f]);
                        denormalized[b, t, f] = NumOps.Add(scaled, _instanceMean[b, 0, f]);
                    }
                }
            }

            return denormalized;
        }
    }

    /// <summary>
    /// Applies self-attention distilling.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Informer model, ApplyDistilling performs a supporting step in the workflow. It keeps the Informer architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private Tensor<T> ApplyDistilling(Tensor<T> input)
    {
        int batchSize = input.Shape[0];
        int seqLen = input.Shape[1];
        int features = input.Shape[2];

        int newSeqLen = (seqLen + 1) / _distillingFactor;
        var distilled = new Tensor<T>(new[] { batchSize, newSeqLen, features });

        for (int b = 0; b < batchSize; b++)
        {
            for (int t = 0; t < newSeqLen; t++)
            {
                int idx1 = t * _distillingFactor;
                int idx2 = Math.Min(idx1 + 1, seqLen - 1);

                for (int f = 0; f < features; f++)
                {
                    T val1 = input[b, idx1, f];
                    T val2 = input[b, idx2, f];
                    distilled[b, t, f] = NumOps.GreaterThan(val1, val2) ? val1 : val2;
                }
            }
        }

        return distilled;
    }

    /// <summary>
    /// Prepares decoder input.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Informer model, PrepareDecoderInput performs a supporting step in the workflow. It keeps the Informer architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private Tensor<T> PrepareDecoderInput(Tensor<T> input, Tensor<T> encoderOutput)
    {
        int batchSize = input.Shape[0];
        int features = encoderOutput.Shape[2];
        int decoderLen = _labelLength + _predictionHorizon;

        var decoderInput = new Tensor<T>(new[] { batchSize, decoderLen, features });

        int encoderLen = encoderOutput.Shape[1];
        int labelStart = Math.Max(0, encoderLen - _labelLength);
        int copyLen = Math.Min(_labelLength, encoderLen);

        for (int b = 0; b < batchSize; b++)
        {
            for (int t = 0; t < copyLen; t++)
            {
                for (int f = 0; f < features; f++)
                {
                    decoderInput[b, t, f] = encoderOutput[b, labelStart + t, f];
                }
            }
        }

        return decoderInput;
    }

    /// <summary>
    /// Extracts prediction portion from decoder output.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Informer model, ExtractPrediction produces predictions from input data. This is the main inference step of the Informer architecture.
    /// </para>
    /// </remarks>
    private Tensor<T> ExtractPrediction(Tensor<T> decoderOutput)
    {
        int batchSize = decoderOutput.Shape[0];
        int decoderLen = decoderOutput.Shape[1];
        int features = decoderOutput.Shape[2];

        int predStart = Math.Max(0, decoderLen - _predictionHorizon);
        int predLen = Math.Min(_predictionHorizon, decoderLen);

        var prediction = new Tensor<T>(new[] { batchSize, _predictionHorizon, features });

        for (int b = 0; b < batchSize; b++)
        {
            for (int t = 0; t < predLen; t++)
            {
                for (int f = 0; f < features; f++)
                {
                    prediction[b, t, f] = decoderOutput[b, predStart + t, f];
                }
            }
        }

        return prediction;
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Shifts input and appends predictions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Informer model, ShiftInputWithPredictions produces predictions from input data. This is the main inference step of the Informer architecture.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ShiftInputWithPredictions(Tensor<T> input, Tensor<T> predictions, int steps)
    {
        int batchSize = input.Shape[0];
        int features = input.Shape[2];
        var newInput = new Tensor<T>(new[] { batchSize, _sequenceLength, features });

        int keepLen = _sequenceLength - steps;
        for (int b = 0; b < batchSize; b++)
        {
            for (int t = 0; t < keepLen; t++)
            {
                for (int f = 0; f < features; f++)
                {
                    newInput[b, t, f] = input[b, t + steps, f];
                }
            }

            for (int t = 0; t < steps; t++)
            {
                for (int f = 0; f < features; f++)
                {
                    newInput[b, keepLen + t, f] = predictions[b, t, f];
                }
            }
        }

        return newInput;
    }

    /// <summary>
    /// Concatenates prediction tensors.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Informer model, ConcatenatePredictions produces predictions from input data. This is the main inference step of the Informer architecture.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ConcatenatePredictions(List<Tensor<T>> predictions, int totalSteps)
    {
        if (predictions.Count == 0)
            return new Tensor<T>(new[] { 1, totalSteps, _numFeatures });

        int batchSize = predictions[0].Shape[0];
        var result = new Tensor<T>(new[] { batchSize, totalSteps, _numFeatures });

        int currentStep = 0;
        foreach (var pred in predictions)
        {
            int stepsToAdd = Math.Min(pred.Shape[1], totalSteps - currentStep);
            for (int b = 0; b < batchSize; b++)
            {
                for (int t = 0; t < stepsToAdd; t++)
                {
                    for (int f = 0; f < _numFeatures; f++)
                    {
                        result[b, currentStep + t, f] = pred[b, t, f];
                    }
                }
            }
            currentStep += stepsToAdd;
        }

        return result;
    }

    #endregion

    #region IDisposable

    /// <summary>
    /// Releases resources.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Informer model, Dispose performs a supporting step in the workflow. It keeps the Informer architecture pipeline consistent.
    /// </para>
    /// </remarks>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            OnnxSession?.Dispose();
        }

        base.Dispose(disposing);
    }

    #endregion
}


