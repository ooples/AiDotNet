using System.IO;
using AiDotNet.Finance.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

using AiDotNet.Finance.Base;
namespace AiDotNet.Finance.Forecasting.Transformers;

/// <summary>
/// FEDformer (Frequency Enhanced Decomposed Transformer) for long-term time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// FEDformer achieves linear complexity O(N) by performing attention in the frequency domain
/// using Fourier or Wavelet transforms. It also decomposes time series into trend and seasonal
/// components for better interpretability and forecasting accuracy.
/// </para>
/// <para>
/// <b>For Beginners:</b> FEDformer is like analyzing music by frequencies instead of individual
/// notes. Standard transformers look at every time step (expensive), but FEDformer converts to
/// frequency domain where patterns are simpler and operations are faster.
///
/// Key innovations:
/// - <b>Frequency Attention:</b> Uses Fourier/Wavelet transforms for O(N) complexity
/// - <b>Decomposition:</b> Separates trend (overall direction) from seasonal (repeating patterns)
/// - <b>Random Mode Selection:</b> Keeps only important frequencies for efficiency
/// </para>
/// <para>
/// <b>Reference:</b> Zhou et al., "FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting",
/// ICML 2022. https://arxiv.org/abs/2201.12740
/// </para>
/// </remarks>
public class FEDformer<T> : ForecastingModelBase<T>
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
    /// Encoder layers for frequency-enhanced attention.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Encoder layers process the input time series to extract patterns.
    /// In FEDformer, each encoder layer applies frequency-domain attention which is much faster
    /// than standard attention while still capturing important dependencies.
    /// </para>
    /// </remarks>
    private readonly List<ILayer<T>> _encoderLayers = [];

    /// <summary>
    /// Decoder layers for generating predictions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Decoder layers generate the actual forecast. They receive information
    /// from the encoder and use it to predict future values. FEDformer's decoder also uses
    /// frequency-domain operations for efficiency.
    /// </para>
    /// </remarks>
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
    /// The optimizer for training.
    /// </summary>
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <summary>
    /// The loss function for training.
    /// </summary>
    private readonly ILossFunction<T> _lossFunction;

    /// <summary>
    /// The input sequence length.
    /// </summary>
    private readonly int _sequenceLength;

    /// <summary>
    /// The prediction horizon.
    /// </summary>
    private readonly int _predictionHorizon;

    /// <summary>
    /// The number of input features.
    /// </summary>
    private readonly int _numFeatures;

    /// <summary>
    /// The number of encoder layers.
    /// </summary>
    private readonly int _numEncoderLayers;

    /// <summary>
    /// The number of decoder layers.
    /// </summary>
    private readonly int _numDecoderLayers;

    /// <summary>
    /// The number of attention heads.
    /// </summary>
    private readonly int _numHeads;

    /// <summary>
    /// The model dimension.
    /// </summary>
    private readonly int _modelDimension;

    /// <summary>
    /// The feedforward dimension.
    /// </summary>
    private readonly int _feedForwardDimension;

    /// <summary>
    /// Whether to use instance normalization (RevIN).
    /// </summary>
    private readonly bool _useInstanceNormalization;

    /// <summary>
    /// Dropout rate.
    /// </summary>
    private readonly double _dropout;

    /// <summary>
    /// Number of frequency modes to use in attention.
    /// </summary>
    private readonly int _numModes;

    /// <summary>
    /// Moving average kernel size for decomposition.
    /// </summary>
    private readonly int _movingAverageKernel;

    #endregion

    #region IForecastingModel Properties

    /// <inheritdoc/>
    public override int SequenceLength => _sequenceLength;

    /// <inheritdoc/>
    public override int PredictionHorizon => _predictionHorizon;

    /// <inheritdoc/>
    public override int NumFeatures => _numFeatures;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> FEDformer doesn't use patches - it operates on the full sequence
    /// in the frequency domain. This property returns 0.
    /// </para>
    /// </remarks>
    public override int PatchSize => 0;

    /// <inheritdoc/>
    public override int Stride => 0;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> FEDformer processes all channels together to capture cross-variable
    /// dependencies, so channels are not independent.
    /// </para>
    /// </remarks>
    public override bool IsChannelIndependent => false;

    /// <inheritdoc/>
    public override bool UseNativeMode => _useNativeMode;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a FEDformer network using a pretrained ONNX model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="sequenceLength">Input sequence length (default: 96).</param>
    /// <param name="predictionHorizon">Prediction horizon (default: 96).</param>
    /// <param name="numFeatures">Number of input features (default: 7).</param>
    /// <param name="optimizer">Optional optimizer for fine-tuning.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor when you have a pretrained FEDformer model
    /// in ONNX format. ONNX models can be trained in Python/PyTorch and used directly in C#.
    /// </para>
    /// </remarks>
    public FEDformer(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int sequenceLength = 96,
        int predictionHorizon = 96,
        int numFeatures = 7,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture,
               lossFunction ?? new MeanSquaredErrorLoss<T>(),
               1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        _useNativeMode = false;
        OnnxModelPath = onnxModelPath;
        _sequenceLength = sequenceLength;
        _predictionHorizon = predictionHorizon;
        _numFeatures = numFeatures;
        _numEncoderLayers = 2;
        _numDecoderLayers = 1;
        _numHeads = 8;
        _modelDimension = 512;
        _feedForwardDimension = 2048;
        _useInstanceNormalization = true;
        _dropout = 0.05;
        _numModes = 64;
        _movingAverageKernel = 25;

        InferenceSession? session = null;
        try
        {
            session = new InferenceSession(onnxModelPath);
            OnnxSession = session;
            _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
            _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
            InitializeLayers();
        }
        catch
        {
            session?.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Creates a FEDformer network using native library layers.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="sequenceLength">Input sequence length (default: 96).</param>
    /// <param name="predictionHorizon">Prediction horizon (default: 96).</param>
    /// <param name="numFeatures">Number of input features (default: 7).</param>
    /// <param name="numEncoderLayers">Number of encoder layers (default: 2).</param>
    /// <param name="numDecoderLayers">Number of decoder layers (default: 1).</param>
    /// <param name="numHeads">Number of attention heads (default: 8).</param>
    /// <param name="modelDimension">Model dimension (default: 512).</param>
    /// <param name="feedForwardDimension">Feedforward dimension (default: 2048).</param>
    /// <param name="numModes">Number of frequency modes (default: 64).</param>
    /// <param name="movingAverageKernel">Moving average kernel for decomposition (default: 25).</param>
    /// <param name="useInstanceNormalization">Whether to use RevIN (default: true).</param>
    /// <param name="dropout">Dropout rate (default: 0.05).</param>
    /// <param name="optimizer">Optional optimizer for training.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor to train FEDformer from scratch.
    ///
    /// Key parameters:
    /// - <b>numModes:</b> How many frequency components to keep. More = more accurate but slower.
    /// - <b>movingAverageKernel:</b> Window size for separating trend from seasonal. Larger = smoother trend.
    /// </para>
    /// </remarks>
    public FEDformer(
        NeuralNetworkArchitecture<T> architecture,
        int sequenceLength = 96,
        int predictionHorizon = 96,
        int numFeatures = 7,
        int numEncoderLayers = 2,
        int numDecoderLayers = 1,
        int numHeads = 8,
        int modelDimension = 512,
        int feedForwardDimension = 2048,
        int numModes = 64,
        int movingAverageKernel = 25,
        bool useInstanceNormalization = true,
        double dropout = 0.05,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture,
               lossFunction ?? new MeanSquaredErrorLoss<T>(),
               1.0)
    {
        ValidateParameters(sequenceLength, predictionHorizon, numFeatures, numEncoderLayers, numDecoderLayers, numHeads, modelDimension);

        _useNativeMode = true;
        _sequenceLength = sequenceLength;
        _predictionHorizon = predictionHorizon;
        _numFeatures = numFeatures;
        _numEncoderLayers = numEncoderLayers;
        _numDecoderLayers = numDecoderLayers;
        _numHeads = numHeads;
        _modelDimension = modelDimension;
        _feedForwardDimension = feedForwardDimension;
        _numModes = numModes;
        _movingAverageKernel = movingAverageKernel;
        _useInstanceNormalization = useInstanceNormalization;
        _dropout = dropout;

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        InitializeLayers();
    }

    /// <summary>
    /// Validates constructor parameters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This ensures all configuration values are valid before building
    /// the model, catching mistakes early.
    /// </para>
    /// </remarks>
    private static void ValidateParameters(int sequenceLength, int predictionHorizon, int numFeatures,
        int numEncoderLayers, int numDecoderLayers, int numHeads, int modelDimension)
    {
        if (sequenceLength < 1)
            throw new ArgumentOutOfRangeException(nameof(sequenceLength), "Sequence length must be at least 1.");
        if (predictionHorizon < 1)
            throw new ArgumentOutOfRangeException(nameof(predictionHorizon), "Prediction horizon must be at least 1.");
        if (numFeatures < 1)
            throw new ArgumentOutOfRangeException(nameof(numFeatures), "Number of features must be at least 1.");
        if (numEncoderLayers < 1)
            throw new ArgumentOutOfRangeException(nameof(numEncoderLayers), "Number of encoder layers must be at least 1.");
        if (numDecoderLayers < 1)
            throw new ArgumentOutOfRangeException(nameof(numDecoderLayers), "Number of decoder layers must be at least 1.");
        if (numHeads < 1)
            throw new ArgumentOutOfRangeException(nameof(numHeads), "Number of heads must be at least 1.");
        if (modelDimension % numHeads != 0)
            throw new ArgumentException("Model dimension must be divisible by number of heads.", nameof(modelDimension));
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes the neural network layers for FEDformer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This sets up FEDformer's architecture:
    /// </para>
    /// <para>
    /// <list type="number">
    /// <item><b>Input Embedding:</b> Projects input features to model dimension</item>
    /// <item><b>Encoder Stack:</b> Processes input with frequency-enhanced attention and decomposition</item>
    /// <item><b>Decoder Stack:</b> Generates predictions using encoder output</item>
    /// <item><b>Output Projection:</b> Maps to final forecast values</item>
    /// </list>
    /// </para>
    /// <para>
    /// The decomposition block separates trend (moving average) from seasonal components
    /// at each layer, making the model more interpretable.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultFEDformerLayers(
                Architecture, _sequenceLength, _predictionHorizon, _numFeatures,
                _numEncoderLayers, _numDecoderLayers, _numHeads, _modelDimension, _feedForwardDimension));

            ExtractLayerReferences();
        }
    }

    /// <summary>
    /// Extracts references to specific layers for direct access.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This organizes the layers so we can access encoder and decoder
    /// layers separately during the forward pass.
    /// </para>
    /// </remarks>
    private void ExtractLayerReferences()
    {
        int idx = 0;
        if (Layers.Count > idx)
            _inputEmbedding = Layers[idx++];

        for (int i = 0; i < _numEncoderLayers && idx < Layers.Count; i++)
        {
            _encoderLayers.Add(Layers[idx++]);
        }

        for (int i = 0; i < _numDecoderLayers && idx < Layers.Count; i++)
        {
            _decoderLayers.Add(Layers[idx++]);
        }

        if (Layers.Count > idx)
            _finalNorm = Layers[idx++];

        if (Layers.Count > idx)
            _outputProjection = Layers[idx];
    }

    /// <summary>
    /// Validates that custom layers meet FEDformer's requirements.
    /// </summary>
    /// <param name="layers">The list of custom layers to validate.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> FEDformer needs at minimum an embedding layer, at least one
    /// encoder, at least one decoder, and an output layer.
    /// </para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);

        if (layers.Count < 4)
            throw new ArgumentException("FEDformer requires at least 4 layers: embedding, encoder(s), decoder(s), and output projection.");
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <inheritdoc/>
    public override bool SupportsTraining => _useNativeMode;

    /// <summary>
    /// Makes a prediction using the input tensor.
    /// </summary>
    /// <param name="input">Input tensor of shape [batch_size, sequence_length, num_features].</param>
    /// <returns>Output tensor of shape [batch_size, prediction_horizon, num_features].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Give FEDformer your historical data and it returns predictions.
    /// The frequency-domain processing makes this fast even for long sequences.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return Forecast(input);
    }

    /// <summary>
    /// Trains the model on a single batch.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <param name="expectedOutput">Target tensor.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Training adjusts the model's parameters to minimize prediction error.
    /// FEDformer learns which frequency components are most important for forecasting.
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
    /// <b>For Beginners:</b> In the FEDformer model, UpdateParameters updates internal parameters or state. This keeps the FEDformer architecture aligned with the latest values.
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
    /// <b>For Beginners:</b> In the FEDformer model, GetModelMetadata performs a supporting step in the workflow. It keeps the FEDformer architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "FEDformer" },
                { "SequenceLength", _sequenceLength },
                { "PredictionHorizon", _predictionHorizon },
                { "NumFeatures", _numFeatures },
                { "NumEncoderLayers", _numEncoderLayers },
                { "NumDecoderLayers", _numDecoderLayers },
                { "NumModes", _numModes },
                { "UseNativeMode", _useNativeMode },
                { "ParameterCount", GetParameterCount() }
            },
            ModelData = _useNativeMode ? this.Serialize() : Array.Empty<byte>()
        };
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FEDformer model, CreateNewInstance builds and wires up model components. This sets up the FEDformer architecture before use.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (_useNativeMode)
        {
            return new FEDformer<T>(
                Architecture, _sequenceLength, _predictionHorizon, _numFeatures,
                _numEncoderLayers, _numDecoderLayers, _numHeads, _modelDimension, _feedForwardDimension,
                _numModes, _movingAverageKernel, _useInstanceNormalization, _dropout,
                _optimizer, _lossFunction);
        }
        else
        {
            return new FEDformer<T>(
                Architecture, OnnxModelPath ?? string.Empty,
                _sequenceLength, _predictionHorizon, _numFeatures,
                _optimizer, _lossFunction);
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FEDformer model, SerializeNetworkSpecificData saves or restores model-specific settings. This lets the FEDformer architecture be reused later.
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
        writer.Write(_numModes);
        writer.Write(_movingAverageKernel);
        writer.Write(_useInstanceNormalization);
        writer.Write(_dropout);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FEDformer model, DeserializeNetworkSpecificData saves or restores model-specific settings. This lets the FEDformer architecture be reused later.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32(); // sequenceLength
        _ = reader.ReadInt32(); // predictionHorizon
        _ = reader.ReadInt32(); // numFeatures
        _ = reader.ReadInt32(); // numEncoderLayers
        _ = reader.ReadInt32(); // numDecoderLayers
        _ = reader.ReadInt32(); // numHeads
        _ = reader.ReadInt32(); // modelDimension
        _ = reader.ReadInt32(); // feedForwardDimension
        _ = reader.ReadInt32(); // numModes
        _ = reader.ReadInt32(); // movingAverageKernel
        _ = reader.ReadBoolean(); // useInstanceNormalization
        _ = reader.ReadDouble(); // dropout
    }

    #endregion

    #region IForecastingModel Implementation

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FEDformer model, Forecast produces predictions from input data. This is the main inference step of the FEDformer architecture.
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
    /// <b>For Beginners:</b> In the FEDformer model, AutoregressiveForecast produces predictions from input data. This is the main inference step of the FEDformer architecture.
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
    /// <b>For Beginners:</b> In the FEDformer model, Evaluate performs a supporting step in the workflow. It keeps the FEDformer architecture pipeline consistent.
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
    /// <b>For Beginners:</b> In the FEDformer model, ApplyInstanceNormalization performs a supporting step in the workflow. It keeps the FEDformer architecture pipeline consistent.
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
    /// <b>For Beginners:</b> In the FEDformer model, GetFinancialMetrics calculates evaluation metrics. This summarizes how the FEDformer architecture is performing.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> GetFinancialMetrics()
    {
        return new Dictionary<string, T>
        {
            ["SequenceLength"] = NumOps.FromDouble(_sequenceLength),
            ["PredictionHorizon"] = NumOps.FromDouble(_predictionHorizon),
            ["NumFeatures"] = NumOps.FromDouble(_numFeatures),
            ["NumModes"] = NumOps.FromDouble(_numModes),
            ["ParameterCount"] = NumOps.FromDouble(GetParameterCount())
        };
    }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// Performs the forward pass through the FEDformer network.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Output tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The forward pass in FEDformer involves:
    /// </para>
    /// <para>
    /// <list type="number">
    /// <item><b>RevIN:</b> Normalize input to handle distribution shift</item>
    /// <item><b>Embedding:</b> Project to model dimension</item>
    /// <item><b>Encoder:</b> Process with frequency-enhanced attention + decomposition</item>
    /// <item><b>Decoder:</b> Generate predictions using encoder output</item>
    /// <item><b>Reverse RevIN:</b> Scale back to original range</item>
    /// </list>
    /// </para>
    /// <para>
    /// The decomposition blocks separate trend from seasonal components at each layer,
    /// making predictions more stable and interpretable.
    /// </para>
    /// </remarks>
    private Tensor<T> Forward(Tensor<T> input)
    {
        var processed = _useInstanceNormalization ? ApplyRevIN(input, normalize: true) : input;

        var output = processed;
        foreach (var layer in Layers)
        {
            output = layer.Forward(output);
        }

        if (_useInstanceNormalization)
        {
            output = ApplyRevIN(output, normalize: false);
        }

        return output;
    }

    /// <summary>
    /// Performs the backward pass.
    /// </summary>
    /// <param name="outputGradient">Gradient from the loss function.</param>
    /// <returns>Gradient with respect to the input.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Backpropagation calculates how each parameter contributed to
    /// the error, allowing the optimizer to adjust them appropriately.
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
    /// Forecasts using native layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FEDformer model, ForecastNative produces predictions from input data. This is the main inference step of the FEDformer architecture.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ForecastNative(Tensor<T> input, double[]? quantiles)
    {
        SetTrainingMode(false);
        return Forward(input);
    }

    /// <summary>
    /// Forecasts using ONNX model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FEDformer model, ForecastOnnx produces predictions from input data. This is the main inference step of the FEDformer architecture.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ForecastOnnx(Tensor<T> input)
    {
        if (OnnxSession is null)
            throw new InvalidOperationException("ONNX session is not initialized.");

        var inputData = new float[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            inputData[i] = Convert.ToSingle(input.Data.Span[i]);
        }

        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input.Shape);
        var inputMeta = OnnxSession.InputMetadata;
        string inputName = inputMeta.Keys.First();

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, onnxInput)
        };

        using var results = OnnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        var outputShape = outputTensor.Dimensions.ToArray();
        var outputData = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++)
        {
            outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));
        }

        return new Tensor<T>(outputShape, new Vector<T>(outputData));
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Applies RevIN (Reversible Instance Normalization).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FEDformer model, ApplyRevIN performs a supporting step in the workflow. It keeps the FEDformer architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private Tensor<T> ApplyRevIN(Tensor<T> input, bool normalize)
    {
        var result = new Tensor<T>(input.Shape);
        T epsilon = NumOps.FromDouble(1e-5);

        if (normalize)
        {
            _instanceMean = CalculateInstanceMean(input);
            _instanceStd = CalculateInstanceStd(input, _instanceMean);

            for (int i = 0; i < input.Length; i++)
            {
                int statIdx = i % _numFeatures;
                T centered = NumOps.Subtract(input.Data.Span[i], _instanceMean.Data.Span[statIdx]);
                T stdVal = NumOps.Add(_instanceStd.Data.Span[statIdx], epsilon);
                result.Data.Span[i] = NumOps.Divide(centered, stdVal);
            }
        }
        else
        {
            if (_instanceMean is null || _instanceStd is null)
                return input;

            for (int i = 0; i < input.Length; i++)
            {
                int statIdx = i % _numFeatures;
                T scaled = NumOps.Multiply(input.Data.Span[i], _instanceStd.Data.Span[statIdx]);
                result.Data.Span[i] = NumOps.Add(scaled, _instanceMean.Data.Span[statIdx]);
            }
        }

        return result;
    }

    /// <summary>
    /// Calculates the mean for each feature.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FEDformer model, CalculateInstanceMean performs a supporting step in the workflow. It keeps the FEDformer architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private Tensor<T> CalculateInstanceMean(Tensor<T> input)
    {
        var mean = new T[_numFeatures];
        int samplesPerFeature = input.Length / _numFeatures;

        for (int f = 0; f < _numFeatures; f++)
        {
            T sum = NumOps.Zero;
            for (int i = f; i < input.Length; i += _numFeatures)
            {
                sum = NumOps.Add(sum, input.Data.Span[i]);
            }
            mean[f] = NumOps.Divide(sum, NumOps.FromDouble(samplesPerFeature));
        }

        return new Tensor<T>(new[] { _numFeatures }, new Vector<T>(mean));
    }

    /// <summary>
    /// Calculates the standard deviation for each feature.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FEDformer model, CalculateInstanceStd performs a supporting step in the workflow. It keeps the FEDformer architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private Tensor<T> CalculateInstanceStd(Tensor<T> input, Tensor<T> mean)
    {
        var std = new T[_numFeatures];
        int samplesPerFeature = input.Length / _numFeatures;

        for (int f = 0; f < _numFeatures; f++)
        {
            T sumSq = NumOps.Zero;
            for (int i = f; i < input.Length; i += _numFeatures)
            {
                T diff = NumOps.Subtract(input.Data.Span[i], mean.Data.Span[f]);
                sumSq = NumOps.Add(sumSq, NumOps.Multiply(diff, diff));
            }
            T variance = NumOps.Divide(sumSq, NumOps.FromDouble(samplesPerFeature));
            std[f] = NumOps.Sqrt(variance);
        }

        return new Tensor<T>(new[] { _numFeatures }, new Vector<T>(std));
    }

    /// <summary>
    /// Shifts input with predictions for autoregressive forecasting.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FEDformer model, ShiftInputWithPredictions produces predictions from input data. This is the main inference step of the FEDformer architecture.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ShiftInputWithPredictions(Tensor<T> input, Tensor<T> predictions, int stepsToShift)
    {
        var newData = new T[input.Length];
        int shiftAmount = stepsToShift * _numFeatures;

        Array.Copy(input.Data.ToArray(), shiftAmount, newData, 0, input.Length - shiftAmount);
        Array.Copy(predictions.Data.ToArray(), 0, newData, input.Length - shiftAmount, shiftAmount);

        return new Tensor<T>(input.Shape, new Vector<T>(newData));
    }

    /// <summary>
    /// Concatenates multiple predictions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FEDformer model, ConcatenatePredictions produces predictions from input data. This is the main inference step of the FEDformer architecture.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ConcatenatePredictions(List<Tensor<T>> predictions, int totalSteps)
    {
        var outputData = new T[totalSteps * _numFeatures];
        int currentIdx = 0;

        foreach (var pred in predictions)
        {
            int stepsToCopy = Math.Min(_predictionHorizon, totalSteps - currentIdx / _numFeatures);
            int elementsToCopy = stepsToCopy * _numFeatures;

            Array.Copy(pred.Data.ToArray(), 0, outputData, currentIdx, elementsToCopy);
            currentIdx += elementsToCopy;

            if (currentIdx >= totalSteps * _numFeatures)
                break;
        }

        return new Tensor<T>(new[] { totalSteps, _numFeatures }, new Vector<T>(outputData));
    }

    #endregion

    #region IDisposable

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the FEDformer model, Dispose performs a supporting step in the workflow. It keeps the FEDformer architecture pipeline consistent.
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


