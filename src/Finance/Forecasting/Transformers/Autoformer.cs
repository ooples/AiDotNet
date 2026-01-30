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
/// Autoformer (Decomposition Transformers with Auto-Correlation) neural network for time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// Autoformer replaces the traditional attention mechanism with auto-correlation, which finds
/// period-based dependencies efficiently using FFT. It also progressively decomposes the series
/// into trend and seasonal components at each layer.
/// </para>
/// <para>
/// <b>For Beginners:</b> Autoformer is like a music analyst who finds repeating patterns
/// by checking if the melody matches itself at different time delays. Instead of comparing
/// every note to every other note (O(L²)), it uses FFT to find these patterns in O(L log L) time.
///
/// Key innovations:
/// - <b>Auto-Correlation:</b> Finds periodic patterns using FFT instead of attention
/// - <b>Series Decomposition:</b> Separates trend from seasonal patterns at each layer
/// - <b>Progressive Decomposition:</b> Decomposition happens multiple times for accuracy
/// </para>
/// <para>
/// <b>Reference:</b> Wu et al., "Autoformer: Decomposition Transformers with Auto-Correlation",
/// NeurIPS 2021. https://arxiv.org/abs/2106.13008
/// </para>
/// </remarks>
public class Autoformer<T> : ForecastingModelBase<T>
{
    #region Execution Mode

    /// <summary>
    /// Indicates whether this network uses native layers (true) or ONNX model (false).
    /// </summary>
    private readonly bool _useNativeMode;

    #endregion

    
    #region Native Mode Fields

    /// <summary>
    /// Input embedding layer that projects features to model dimension.
    /// </summary>
    private ILayer<T>? _inputEmbedding;

    /// <summary>
    /// Encoder layers with auto-correlation.
    /// </summary>
    private readonly List<ILayer<T>> _encoderLayers = [];

    /// <summary>
    /// Decoder layers for generating predictions.
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

    /// <summary>
    /// Trend component from decomposition.
    /// </summary>
    private Tensor<T>? _trendComponent;

    /// <summary>
    /// Seasonal component from decomposition.
    /// </summary>
    private Tensor<T>? _seasonalComponent;

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
    private int _predictionHorizon;
    private int _numFeatures;
    private int _numEncoderLayers;
    private int _numDecoderLayers;
    private int _numHeads;
    private int _modelDimension;
    private int _feedForwardDimension;
    private int _movingAverageKernel;
    private int _topKFactor;
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
    /// Creates an Autoformer network using a pretrained ONNX model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor when you have a pretrained Autoformer model.
    /// </para>
    /// </remarks>
    public Autoformer(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        AutoformerOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentNullException(nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model file not found: {onnxModelPath}", onnxModelPath);

        options ??= new AutoformerOptions<T>();

        _useNativeMode = false;
        OnnxSession = new InferenceSession(onnxModelPath);
        OnnxModelPath = onnxModelPath;

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        _sequenceLength = options.LookbackWindow;
        _predictionHorizon = options.ForecastHorizon;
        _numFeatures = architecture.InputSize;
        _numEncoderLayers = options.NumEncoderLayers;
        _numDecoderLayers = options.NumDecoderLayers;
        _numHeads = options.NumAttentionHeads;
        _modelDimension = options.EmbeddingDim;
        _feedForwardDimension = options.EmbeddingDim * 4;
        _movingAverageKernel = options.MovingAverageKernel;
        _topKFactor = options.AutoCorrelationFactor;
        _dropout = options.DropoutRate;
        _useInstanceNormalization = true;
    }

    /// <summary>
    /// Creates an Autoformer network in native mode for training from scratch.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor to train a new Autoformer model.
    /// </para>
    /// </remarks>
    public Autoformer(
        NeuralNetworkArchitecture<T> architecture,
        AutoformerOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new AutoformerOptions<T>();

        _useNativeMode = true;
        OnnxSession = null;
        OnnxModelPath = null;

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        _sequenceLength = options.LookbackWindow;
        _predictionHorizon = options.ForecastHorizon;
        _numFeatures = architecture.InputSize;
        _numEncoderLayers = options.NumEncoderLayers;
        _numDecoderLayers = options.NumDecoderLayers;
        _numHeads = options.NumAttentionHeads;
        _modelDimension = options.EmbeddingDim;
        _feedForwardDimension = options.EmbeddingDim * 4;
        _movingAverageKernel = options.MovingAverageKernel;
        _topKFactor = options.AutoCorrelationFactor;
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
    /// <b>For Beginners:</b> In the Autoformer model, InitializeLayers builds and wires up model components. This sets up the Autoformer architecture before use.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultAutoformerLayers(
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
    /// <b>For Beginners:</b> In the Autoformer model, ExtractLayerReferences performs a supporting step in the workflow. It keeps the Autoformer architecture pipeline consistent.
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
    /// Validates custom layers provided by the user.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Autoformer model, ValidateCustomLayers checks inputs and configuration. This protects the Autoformer architecture from mismatches and errors.
    /// </para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);

        if (layers.Count < 3)
            throw new ArgumentException("Autoformer requires at least 3 layers: embedding, encoder, and output.");
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <inheritdoc/>
    public override bool SupportsTraining => _useNativeMode;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Autoformer model, Predict produces predictions from input data. This is the main inference step of the Autoformer architecture.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return Forecast(input);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Autoformer model, Train performs a training step. This updates the Autoformer architecture so it learns from data.
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
    /// <b>For Beginners:</b> In the Autoformer model, UpdateParameters updates internal parameters or state. This keeps the Autoformer architecture aligned with the latest values.
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
    /// <b>For Beginners:</b> In the Autoformer model, GetModelMetadata performs a supporting step in the workflow. It keeps the Autoformer architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "Autoformer" },
                { "SequenceLength", _sequenceLength },
                { "PredictionHorizon", _predictionHorizon },
                { "NumFeatures", _numFeatures },
                { "NumEncoderLayers", _numEncoderLayers },
                { "NumDecoderLayers", _numDecoderLayers },
                { "NumHeads", _numHeads },
                { "ModelDimension", _modelDimension },
                { "MovingAverageKernel", _movingAverageKernel },
                { "UseNativeMode", _useNativeMode },
                { "ParameterCount", GetParameterCount() }
            },
            ModelData = _useNativeMode ? this.Serialize() : Array.Empty<byte>()
        };
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Autoformer model, CreateNewInstance builds and wires up model components. This sets up the Autoformer architecture before use.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var options = new AutoformerOptions<T>
        {
            LookbackWindow = _sequenceLength,
            ForecastHorizon = _predictionHorizon,
            NumEncoderLayers = _numEncoderLayers,
            NumDecoderLayers = _numDecoderLayers,
            NumAttentionHeads = _numHeads,
            EmbeddingDim = _modelDimension,
            MovingAverageKernel = _movingAverageKernel,
            AutoCorrelationFactor = _topKFactor,
            DropoutRate = _dropout
        };

        return _useNativeMode
            ? new Autoformer<T>(Architecture, options, _optimizer, _lossFunction)
            : new Autoformer<T>(Architecture, OnnxModelPath!, options, _optimizer, _lossFunction);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Autoformer model, SerializeNetworkSpecificData saves or restores model-specific settings. This lets the Autoformer architecture be reused later.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_sequenceLength);
        writer.Write(_predictionHorizon);
        writer.Write(_numFeatures);
        writer.Write(_numEncoderLayers);
        writer.Write(_numDecoderLayers);
        writer.Write(_numHeads);
        writer.Write(_modelDimension);
        writer.Write(_feedForwardDimension);
        writer.Write(_movingAverageKernel);
        writer.Write(_topKFactor);
        writer.Write(_dropout);
        writer.Write(_useInstanceNormalization);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Autoformer model, DeserializeNetworkSpecificData saves or restores model-specific settings. This lets the Autoformer architecture be reused later.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _sequenceLength = reader.ReadInt32();
        _predictionHorizon = reader.ReadInt32();
        _numFeatures = reader.ReadInt32();
        _numEncoderLayers = reader.ReadInt32();
        _numDecoderLayers = reader.ReadInt32();
        _numHeads = reader.ReadInt32();
        _modelDimension = reader.ReadInt32();
        _feedForwardDimension = reader.ReadInt32();
        _movingAverageKernel = reader.ReadInt32();
        _topKFactor = reader.ReadInt32();
        _dropout = reader.ReadDouble();
        _useInstanceNormalization = reader.ReadBoolean();
    }

    #endregion

    #region IForecastingModel Implementation

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Autoformer model, Forecast produces predictions from input data. This is the main inference step of the Autoformer architecture.
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
    /// <b>For Beginners:</b> In the Autoformer model, AutoregressiveForecast produces predictions from input data. This is the main inference step of the Autoformer architecture.
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
    /// <b>For Beginners:</b> In the Autoformer model, Evaluate performs a supporting step in the workflow. It keeps the Autoformer architecture pipeline consistent.
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
    /// <b>For Beginners:</b> In the Autoformer model, ApplyInstanceNormalization performs a supporting step in the workflow. It keeps the Autoformer architecture pipeline consistent.
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
    /// <b>For Beginners:</b> In the Autoformer model, GetFinancialMetrics calculates evaluation metrics. This summarizes how the Autoformer architecture is performing.
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
    /// Performs the forward pass through all layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Autoformer model, Forward runs the forward pass through the layers. This moves data through the Autoformer architecture to compute outputs.
    /// </para>
    /// </remarks>
    private Tensor<T> Forward(Tensor<T> input)
    {
        var current = _useInstanceNormalization ? ApplyRevIN(input, normalize: true) : input;

        if (_inputEmbedding is not null)
        {
            current = _inputEmbedding.Forward(current);
        }

        (_trendComponent, _seasonalComponent) = InitializeDecomposition(current);

        foreach (var layer in _encoderLayers)
        {
            current = layer.Forward(current);
            var (trend, seasonal) = SeriesDecomposition(current);
            _trendComponent = AddTensors(_trendComponent, trend);
            _seasonalComponent = seasonal;
            current = _seasonalComponent;
        }

        foreach (var layer in _decoderLayers)
        {
            current = layer.Forward(current);
            var (trend, seasonal) = SeriesDecomposition(current);
            _trendComponent = AddTensors(_trendComponent, trend);
            current = seasonal;
        }

        current = AddTensors(current, _trendComponent!);

        if (_finalNorm is not null)
        {
            current = _finalNorm.Forward(current);
        }

        if (_outputProjection is not null)
        {
            current = _outputProjection.Forward(current);
        }

        current = AdjustToPredictionHorizon(current);

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
    /// <b>For Beginners:</b> In the Autoformer model, Backward propagates gradients backward. This teaches the Autoformer architecture how to adjust its weights.
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
    /// <b>For Beginners:</b> In the Autoformer model, ForecastNative produces predictions from input data. This is the main inference step of the Autoformer architecture.
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
    /// <b>For Beginners:</b> In the Autoformer model, ForecastOnnx produces predictions from input data. This is the main inference step of the Autoformer architecture.
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
    /// Initializes decomposition components.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Autoformer model, private performs a supporting step in the workflow. It keeps the Autoformer architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private (Tensor<T> trend, Tensor<T> seasonal) InitializeDecomposition(Tensor<T> input)
    {
        var trend = new Tensor<T>(input.Shape);
        var seasonal = input.Clone();
        return (trend, seasonal);
    }

    /// <summary>
    /// Performs series decomposition using moving average.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Autoformer model, private performs a supporting step in the workflow. It keeps the Autoformer architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private (Tensor<T> trend, Tensor<T> seasonal) SeriesDecomposition(Tensor<T> input)
    {
        var trend = MovingAverage(input, _movingAverageKernel);
        var seasonal = SubtractTensors(input, trend);
        return (trend, seasonal);
    }

    /// <summary>
    /// Computes moving average.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Autoformer model, MovingAverage performs a supporting step in the workflow. It keeps the Autoformer architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private Tensor<T> MovingAverage(Tensor<T> input, int kernelSize)
    {
        var result = new Tensor<T>(input.Shape);
        int seqLen = input.Shape[1];
        int halfKernel = kernelSize / 2;

        for (int b = 0; b < input.Shape[0]; b++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                int start = Math.Max(0, t - halfKernel);
                int end = Math.Min(seqLen, t + halfKernel + 1);
                int count = end - start;

                for (int f = 0; f < input.Shape[2]; f++)
                {
                    T sum = NumOps.Zero;
                    for (int i = start; i < end; i++)
                    {
                        sum = NumOps.Add(sum, input[b, i, f]);
                    }
                    result[b, t, f] = NumOps.Divide(sum, NumOps.FromDouble(count));
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Applies RevIN normalization/denormalization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Autoformer model, ApplyRevIN performs a supporting step in the workflow. It keeps the Autoformer architecture pipeline consistent.
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
    /// Adjusts output to prediction horizon.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Autoformer model, AdjustToPredictionHorizon produces predictions from input data. This is the main inference step of the Autoformer architecture.
    /// </para>
    /// </remarks>
    private Tensor<T> AdjustToPredictionHorizon(Tensor<T> output)
    {
        int currentLen = output.Shape[1];
        if (currentLen == _predictionHorizon)
            return output;

        var adjusted = new Tensor<T>(new[] { output.Shape[0], _predictionHorizon, output.Shape[2] });
        int copyLen = Math.Min(currentLen, _predictionHorizon);

        for (int b = 0; b < output.Shape[0]; b++)
        {
            for (int t = 0; t < copyLen; t++)
            {
                for (int f = 0; f < output.Shape[2]; f++)
                {
                    adjusted[b, t, f] = output[b, t, f];
                }
            }
        }

        return adjusted;
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Adds two tensors element-wise.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Autoformer model, AddTensors performs a supporting step in the workflow. It keeps the Autoformer architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private Tensor<T> AddTensors(Tensor<T>? a, Tensor<T>? b)
    {
        if (a is null)
            return b ?? new Tensor<T>(new[] { 1 });
        if (b is null)
            return a;

        var result = new Tensor<T>(a.Shape);
        for (int i = 0; i < a.Length; i++)
        {
            result[i] = NumOps.Add(a[i], b[i]);
        }
        return result;
    }

    /// <summary>
    /// Subtracts tensor b from tensor a.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Autoformer model, SubtractTensors performs a supporting step in the workflow. It keeps the Autoformer architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private Tensor<T> SubtractTensors(Tensor<T> a, Tensor<T> b)
    {
        var result = new Tensor<T>(a.Shape);
        for (int i = 0; i < a.Length; i++)
        {
            result[i] = NumOps.Subtract(a[i], b[i]);
        }
        return result;
    }

    /// <summary>
    /// Shifts input and appends predictions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Autoformer model, ShiftInputWithPredictions produces predictions from input data. This is the main inference step of the Autoformer architecture.
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
    /// <b>For Beginners:</b> In the Autoformer model, ConcatenatePredictions produces predictions from input data. This is the main inference step of the Autoformer architecture.
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
    /// <b>For Beginners:</b> In the Autoformer model, Dispose performs a supporting step in the workflow. It keeps the Autoformer architecture pipeline consistent.
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


