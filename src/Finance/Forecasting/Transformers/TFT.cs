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
/// Temporal Fusion Transformer (TFT) neural network for multi-horizon time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// TFT is a state-of-the-art architecture that combines high-performance multi-horizon forecasting
/// with interpretable insights. It uses variable selection networks, gating mechanisms, and
/// self-attention to handle multiple input types (static, known future, unknown observed).
/// </para>
/// <para>
/// <b>For Beginners:</b> TFT is like a smart assistant that considers different types of information:
/// - <b>Static features:</b> Things that don't change (e.g., store location)
/// - <b>Known future inputs:</b> Things we know ahead (e.g., holidays, promotions)
/// - <b>Unknown inputs:</b> Historical data we can only observe (e.g., past sales)
///
/// Key innovations:
/// - <b>Variable Selection:</b> Automatically finds which features matter most
/// - <b>Gated Skip Connections:</b> Helps information flow through the network
/// - <b>Interpretable Attention:</b> Shows which time periods influenced predictions
/// </para>
/// <para>
/// <b>Reference:</b> Lim et al., "Temporal Fusion Transformers for Interpretable Multi-horizon
/// Time Series Forecasting", International Journal of Forecasting 2021.
/// https://arxiv.org/abs/1912.09363
/// </para>
/// </remarks>
public class TFT<T> : ForecastingModelBase<T>
{
    #region Execution Mode

    /// <summary>
    /// Indicates whether this network uses native layers (true) or ONNX model (false).
    /// </summary>
    private readonly bool _useNativeMode;

    #endregion

    
    #region Native Mode Fields

    /// <summary>
    /// Variable selection network for static features.
    /// </summary>
    private ILayer<T>? _staticVariableSelection;

    /// <summary>
    /// Variable selection network for encoder (historical) features.
    /// </summary>
    private ILayer<T>? _encoderVariableSelection;

    /// <summary>
    /// Variable selection network for decoder (future) features.
    /// </summary>
    private ILayer<T>? _decoderVariableSelection;

    /// <summary>
    /// LSTM encoder for processing historical sequence.
    /// </summary>
    private ILayer<T>? _lstmEncoder;

    /// <summary>
    /// LSTM decoder for processing future sequence.
    /// </summary>
    private ILayer<T>? _lstmDecoder;

    /// <summary>
    /// Gated residual network layers.
    /// </summary>
    private readonly List<ILayer<T>> _grnLayers = [];

    /// <summary>
    /// Interpretable multi-head attention layer.
    /// </summary>
    private ILayer<T>? _attentionLayer;

    /// <summary>
    /// Final layer normalization.
    /// </summary>
    private ILayer<T>? _finalNorm;

    /// <summary>
    /// Output projection layer for quantile predictions.
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
    /// The input sequence length (lookback window).
    /// </summary>
    private int _sequenceLength;

    /// <summary>
    /// The prediction horizon.
    /// </summary>
    private int _predictionHorizon;

    /// <summary>
    /// The number of input features.
    /// </summary>
    private int _numFeatures;

    /// <summary>
    /// Hidden state size for the model.
    /// </summary>
    private int _hiddenSize;

    /// <summary>
    /// Number of attention heads.
    /// </summary>
    private int _numHeads;

    /// <summary>
    /// Number of transformer/GRN layers.
    /// </summary>
    private int _numLayers;

    /// <summary>
    /// Dropout rate for regularization.
    /// </summary>
    private double _dropout;

    /// <summary>
    /// Quantile levels for probabilistic forecasting.
    /// </summary>
    private double[] _quantileLevels;

    /// <summary>
    /// Whether to use variable selection networks.
    /// </summary>
    private bool _useVariableSelection;

    /// <summary>
    /// Size of static covariate inputs.
    /// </summary>
    private int _staticCovariateSize;

    /// <summary>
    /// Whether to use instance normalization (RevIN).
    /// </summary>
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
    public override int PatchSize => 1; // TFT doesn't use patching

    /// <inheritdoc/>
    public override int Stride => 1;

    /// <inheritdoc/>
    public override bool IsChannelIndependent => false; // TFT processes all channels together

    /// <inheritdoc/>
    public override bool UseNativeMode => _useNativeMode;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a TFT network using pretrained ONNX model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer for fine-tuning.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor when you have a pretrained ONNX model.
    /// ONNX models are pre-trained and ready to use for predictions immediately.
    /// </para>
    /// </remarks>
    public TFT(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        TemporalFusionTransformerOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        options ??= new TemporalFusionTransformerOptions<T>();

        _useNativeMode = false;
        OnnxSession = new InferenceSession(onnxModelPath);
        OnnxModelPath = onnxModelPath;

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        _sequenceLength = options.LookbackWindow;
        _predictionHorizon = options.ForecastHorizon;
        _numFeatures = architecture.InputSize;
        _hiddenSize = options.HiddenSize;
        _numHeads = options.NumAttentionHeads;
        _numLayers = options.NumLayers;
        _dropout = options.DropoutRate;
        _quantileLevels = options.QuantileLevels;
        _useVariableSelection = options.UseVariableSelection;
        _staticCovariateSize = options.StaticCovariateSize;
        _useInstanceNormalization = true;
    }

    /// <summary>
    /// Creates a TFT network in native mode for training from scratch.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor to train a new TFT model from scratch.
    /// TFT excels at:
    /// - Multi-horizon forecasting (predicting multiple future time steps)
    /// - Handling mixed input types (static, known future, historical)
    /// - Providing interpretable attention weights
    /// </para>
    /// </remarks>
    public TFT(
        NeuralNetworkArchitecture<T> architecture,
        TemporalFusionTransformerOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new TemporalFusionTransformerOptions<T>();

        _useNativeMode = true;
        OnnxSession = null;
        OnnxModelPath = null;

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        _sequenceLength = options.LookbackWindow;
        _predictionHorizon = options.ForecastHorizon;
        _numFeatures = architecture.InputSize;
        _hiddenSize = options.HiddenSize;
        _numHeads = options.NumAttentionHeads;
        _numLayers = options.NumLayers;
        _dropout = options.DropoutRate;
        _quantileLevels = options.QuantileLevels;
        _useVariableSelection = options.UseVariableSelection;
        _staticCovariateSize = options.StaticCovariateSize;
        _useInstanceNormalization = true;

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes the neural network layers for TFT.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> TFT has several specialized components:
    /// </para>
    /// <para>
    /// <list type="number">
    /// <item><b>Variable Selection Networks:</b> Learn which input features matter most</item>
    /// <item><b>LSTM Encoder/Decoder:</b> Process sequential patterns in the data</item>
    /// <item><b>Gated Residual Networks:</b> Enable flexible information flow</item>
    /// <item><b>Interpretable Multi-head Attention:</b> Focus on important time periods</item>
    /// <item><b>Quantile Output:</b> Produce probabilistic forecasts</item>
    /// </list>
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultTFTLayers(
                Architecture, _sequenceLength, _predictionHorizon, _numFeatures,
                _hiddenSize, _numHeads, _numLayers, _dropout));

            ExtractLayerReferences();
        }
    }

    /// <summary>
    /// Extracts references to specific layers from the layer collection.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> TFT has many specialized components that need to be
    /// accessed during the forward pass. This organizes them for efficient access.
    /// </para>
    /// </remarks>
    private void ExtractLayerReferences()
    {
        int idx = 0;

        // Variable selection networks
        if (_useVariableSelection && Layers.Count > idx)
        {
            _staticVariableSelection = Layers[idx++];
            if (Layers.Count > idx)
                _encoderVariableSelection = Layers[idx++];
            if (Layers.Count > idx)
                _decoderVariableSelection = Layers[idx++];
        }

        // LSTM encoder/decoder
        if (Layers.Count > idx)
            _lstmEncoder = Layers[idx++];
        if (Layers.Count > idx)
            _lstmDecoder = Layers[idx++];

        // GRN layers
        for (int i = 0; i < _numLayers && idx < Layers.Count; i++)
        {
            _grnLayers.Add(Layers[idx++]);
        }

        // Attention and output
        if (Layers.Count > idx)
            _attentionLayer = Layers[idx++];
        if (Layers.Count > idx)
            _finalNorm = Layers[idx++];
        if (Layers.Count > idx)
            _outputProjection = Layers[idx];
    }

    /// <summary>
    /// Validates that custom layers meet TFT's architectural requirements.
    /// </summary>
    /// <param name="layers">The list of custom layers to validate.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> TFT requires minimum layers for LSTM processing and output.
    /// </para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);
        if (layers.Count < 4)
        {
            throw new ArgumentException(
                "TFT requires at least 4 layers: LSTM encoder, LSTM decoder, attention, and output projection.",
                nameof(layers));
        }
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <inheritdoc/>
    public override bool SupportsTraining => _useNativeMode;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TFT model, Predict produces predictions from input data. This is the main inference step of the TFT architecture.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return _useNativeMode ? ForecastNative(input) : ForecastOnnx(input);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TFT model, Train performs a training step. This updates the TFT architecture so it learns from data.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is only supported in native mode.");

        SetTrainingMode(true);

        // Forward pass
        var predictions = Forward(input);

        // Compute loss - convert to vectors for loss function
        LastLoss = _lossFunction.CalculateLoss(predictions.ToVector(), target.ToVector());

        // Backward pass - convert gradient back to tensor
        var gradient = _lossFunction.CalculateDerivative(predictions.ToVector(), target.ToVector());
        Backward(Tensor<T>.FromVector(gradient, predictions.Shape));

        // Update weights via optimizer
        _optimizer.UpdateParameters(Layers);

        SetTrainingMode(false);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TFT model, UpdateParameters updates internal parameters or state. This keeps the TFT architecture aligned with the latest values.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> gradients)
    {
        // Parameters are updated through the optimizer in Train method
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TFT model, GetModelMetadata performs a supporting step in the workflow. It keeps the TFT architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "TFT" },
                { "SequenceLength", _sequenceLength },
                { "PredictionHorizon", _predictionHorizon },
                { "HiddenSize", _hiddenSize },
                { "NumHeads", _numHeads },
                { "NumLayers", _numLayers },
                { "QuantileLevels", _quantileLevels },
                { "UseVariableSelection", _useVariableSelection },
                { "UseNativeMode", _useNativeMode },
                { "ParameterCount", GetParameterCount() }
            },
            ModelData = _useNativeMode ? this.Serialize() : Array.Empty<byte>()
        };
    }

    /// <summary>
    /// Creates a new instance of this model with the same configuration.
    /// </summary>
    /// <returns>A new TFT model instance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates a fresh copy of the model with the same settings
    /// but new (randomly initialized) weights. Useful for ensemble training or cross-validation.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var options = new TemporalFusionTransformerOptions<T>
        {
            LookbackWindow = _sequenceLength,
            ForecastHorizon = _predictionHorizon,
            HiddenSize = _hiddenSize,
            NumAttentionHeads = _numHeads,
            NumLayers = _numLayers,
            DropoutRate = _dropout,
            QuantileLevels = _quantileLevels,
            UseVariableSelection = _useVariableSelection,
            StaticCovariateSize = _staticCovariateSize
        };

        return new TFT<T>(Architecture, options);
    }

    /// <summary>
    /// Writes TFT-specific configuration during serialization.
    /// </summary>
    /// <param name="writer">Binary writer for output.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This saves TFT settings like hidden size and number of heads
    /// to a file so the model can be loaded later with the same configuration.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_sequenceLength);
        writer.Write(_predictionHorizon);
        writer.Write(_hiddenSize);
        writer.Write(_numHeads);
        writer.Write(_numLayers);
        writer.Write(_dropout);
        writer.Write(_quantileLevels.Length);
        foreach (var q in _quantileLevels)
            writer.Write(q);
        writer.Write(_useVariableSelection);
        writer.Write(_staticCovariateSize);
    }

    /// <summary>
    /// Reads TFT-specific configuration during deserialization.
    /// </summary>
    /// <param name="reader">Binary reader for input.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This reads back TFT settings when loading a saved model
    /// and restores the model configuration.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _sequenceLength = reader.ReadInt32();
        _predictionHorizon = reader.ReadInt32();
        _hiddenSize = reader.ReadInt32();
        _numHeads = reader.ReadInt32();
        _numLayers = reader.ReadInt32();
        _dropout = reader.ReadDouble();
        int quantileCount = reader.ReadInt32();
        _quantileLevels = new double[quantileCount];
        for (int i = 0; i < quantileCount; i++)
            _quantileLevels[i] = reader.ReadDouble();
        _useVariableSelection = reader.ReadBoolean();
        _staticCovariateSize = reader.ReadInt32();
    }

    #endregion

    #region IForecastingModel Implementation

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TFT model, Forecast produces predictions from input data. This is the main inference step of the TFT architecture.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
    {
        if (_useInstanceNormalization)
            historicalData = ApplyInstanceNormalization(historicalData);

        var forecast = _useNativeMode ? ForecastNative(historicalData) : ForecastOnnx(historicalData);

        if (_useInstanceNormalization)
            forecast = ApplyRevIN(forecast, normalize: false);

        return forecast;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TFT model, AutoregressiveForecast produces predictions from input data. This is the main inference step of the TFT architecture.
    /// </para>
    /// </remarks>
    public override Tensor<T> AutoregressiveForecast(Tensor<T> input, int steps)
    {
        var predictions = new List<Tensor<T>>();
        var currentInput = input;

        int stepsRemaining = steps;
        while (stepsRemaining > 0)
        {
            var prediction = Forecast(currentInput, null);
            predictions.Add(prediction);

            int stepsUsed = Math.Min(_predictionHorizon, stepsRemaining);
            stepsRemaining -= stepsUsed;

            if (stepsRemaining > 0)
            {
                currentInput = ShiftInputWithPredictions(currentInput, prediction, stepsUsed);
            }
        }

        return ConcatenatePredictions(predictions, steps);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TFT model, Evaluate performs a supporting step in the workflow. It keeps the TFT architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> Evaluate(Tensor<T> predictions, Tensor<T> actuals)
    {
        var metrics = new Dictionary<string, T>();

        // Calculate MSE
        T mse = NumOps.Zero;
        T mae = NumOps.Zero;
        int count = 0;

        for (int i = 0; i < predictions.Length && i < actuals.Length; i++)
        {
            var diff = NumOps.Subtract(predictions[i], actuals[i]);
            mse = NumOps.Add(mse, NumOps.Multiply(diff, diff));
            mae = NumOps.Add(mae, NumOps.Abs(diff));
            count++;
        }

        if (count > 0)
        {
            mse = NumOps.Divide(mse, NumOps.FromDouble(count));
            mae = NumOps.Divide(mae, NumOps.FromDouble(count));
        }

        metrics["MSE"] = mse;
        metrics["MAE"] = mae;
        metrics["RMSE"] = NumOps.Sqrt(mse);

        return metrics;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TFT model, ApplyInstanceNormalization performs a supporting step in the workflow. It keeps the TFT architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
    {
        return ApplyRevIN(input, normalize: true);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TFT model, GetFinancialMetrics calculates evaluation metrics. This summarizes how the TFT architecture is performing.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> GetFinancialMetrics()
    {
        return new Dictionary<string, T>
        {
            ["SequenceLength"] = NumOps.FromDouble(_sequenceLength),
            ["PredictionHorizon"] = NumOps.FromDouble(_predictionHorizon),
            ["HiddenSize"] = NumOps.FromDouble(_hiddenSize),
            ["NumHeads"] = NumOps.FromDouble(_numHeads),
            ["NumLayers"] = NumOps.FromDouble(_numLayers),
            ["NumQuantiles"] = NumOps.FromDouble(_quantileLevels.Length)
        };
    }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// Performs the forward pass through the TFT network.
    /// </summary>
    /// <param name="input">Input tensor of shape [batch, sequence_length, features].</param>
    /// <returns>Output tensor of shape [batch, prediction_horizon, features].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The TFT forward pass has several stages:
    /// 1. Variable Selection: Determine which features are important
    /// 2. LSTM Processing: Capture sequential dependencies
    /// 3. Gated Residual Networks: Process with skip connections
    /// 4. Attention: Focus on relevant time periods
    /// 5. Output: Generate predictions for each horizon step
    /// </para>
    /// </remarks>
    private Tensor<T> Forward(Tensor<T> input)
    {
        var current = input;

        // Variable selection (if enabled)
        if (_useVariableSelection && _encoderVariableSelection is not null)
        {
            current = _encoderVariableSelection.Forward(current);
        }

        // LSTM encoder
        if (_lstmEncoder is not null)
        {
            current = _lstmEncoder.Forward(current);
        }

        // Gated Residual Networks
        foreach (var grn in _grnLayers)
        {
            var grnOutput = grn.Forward(current);
            current = AddGatedConnection(current, grnOutput);
        }

        // Multi-head attention
        if (_attentionLayer is not null)
        {
            var attended = _attentionLayer.Forward(current);
            current = AddGatedConnection(current, attended);
        }

        // Final normalization
        if (_finalNorm is not null)
        {
            current = _finalNorm.Forward(current);
        }

        // Output projection
        if (_outputProjection is not null)
        {
            current = _outputProjection.Forward(current);
        }

        return AdjustToPredictionHorizon(current);
    }

    /// <summary>
    /// Performs the backward pass through the TFT network.
    /// </summary>
    /// <param name="gradOutput">Gradient from the loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Backward pass computes gradients for all learnable
    /// parameters by propagating error signals backwards through each layer.
    /// </para>
    /// </remarks>
    private void Backward(Tensor<T> gradOutput)
    {
        var grad = gradOutput;

        // Backward through output projection
        if (_outputProjection is not null)
        {
            grad = _outputProjection.Backward(grad);
        }

        // Backward through final norm
        if (_finalNorm is not null)
        {
            grad = _finalNorm.Backward(grad);
        }

        // Backward through attention
        if (_attentionLayer is not null)
        {
            grad = _attentionLayer.Backward(grad);
        }

        // Backward through GRN layers (in reverse order)
        for (int i = _grnLayers.Count - 1; i >= 0; i--)
        {
            grad = _grnLayers[i].Backward(grad);
        }

        // Backward through LSTM
        if (_lstmEncoder is not null)
        {
            grad = _lstmEncoder.Backward(grad);
        }

        // Backward through variable selection
        if (_useVariableSelection && _encoderVariableSelection is not null)
        {
            _encoderVariableSelection.Backward(grad);
        }
    }

    /// <summary>
    /// Performs native mode forecasting.
    /// </summary>
    /// <param name="input">Input historical data.</param>
    /// <returns>Forecasted values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Native mode uses the layers defined in this library
    /// for inference. This allows full control and training capability.
    /// </para>
    /// </remarks>
    private Tensor<T> ForecastNative(Tensor<T> input)
    {
        SetTrainingMode(false);
        return Forward(input);
    }

    /// <summary>
    /// Performs ONNX mode forecasting.
    /// </summary>
    /// <param name="input">Input historical data.</param>
    /// <returns>Forecasted values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> ONNX mode uses a pre-trained model file for inference.
    /// This is typically faster but doesn't support training.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ForecastOnnx(Tensor<T> input)
    {
        if (OnnxSession is null)
            throw new InvalidOperationException("ONNX session is not initialized.");

        // Convert to ONNX format
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

    #region Model-Specific Processing

    /// <summary>
    /// Adds a gated connection between input and processed output.
    /// </summary>
    /// <param name="input">Original input tensor.</param>
    /// <param name="processed">Processed tensor from layer.</param>
    /// <returns>Gated combination of input and processed tensors.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Gated connections allow the network to decide how much
    /// of the original input vs. processed output to use. This helps with gradient
    /// flow and allows the network to skip processing when it's not helpful.
    /// </para>
    /// </remarks>
    private Tensor<T> AddGatedConnection(Tensor<T> input, Tensor<T> processed)
    {
        // Simplified gating: element-wise combination
        var result = new Tensor<T>(input.Shape);
        T half = NumOps.FromDouble(0.5);

        for (int i = 0; i < input.Length && i < processed.Length; i++)
        {
            // Simple average for now; full implementation would use learned gates
            var combined = NumOps.Add(
                NumOps.Multiply(input[i], half),
                NumOps.Multiply(processed[i], half));
            result[i] = combined;
        }

        return result;
    }

    /// <summary>
    /// Applies RevIN normalization/denormalization.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <param name="normalize">True to normalize, false to denormalize.</param>
    /// <returns>Normalized or denormalized tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> RevIN (Reversible Instance Normalization) handles
    /// distribution shifts in time series data by normalizing at inference time
    /// and then reversing the normalization on the output.
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
    /// Adjusts output to match prediction horizon.
    /// </summary>
    /// <param name="output">Output from network.</param>
    /// <returns>Adjusted output with correct prediction horizon size.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The network might output a different sequence length
    /// than the desired prediction horizon. This adjusts the output to match.
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
    /// Shifts input and appends predictions for autoregressive forecasting.
    /// </summary>
    /// <param name="input">Current input tensor.</param>
    /// <param name="predictions">Predictions to append.</param>
    /// <param name="steps">Number of steps to shift.</param>
    /// <returns>Shifted input tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> For forecasting beyond the prediction horizon,
    /// we need to "roll" the input window forward, using our predictions
    /// as the new historical data.
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
    /// Concatenates multiple prediction tensors.
    /// </summary>
    /// <param name="predictions">List of prediction tensors.</param>
    /// <param name="totalSteps">Total number of steps to include.</param>
    /// <returns>Concatenated predictions tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When forecasting multiple horizons autoregressively,
    /// we accumulate predictions and combine them into a single output.
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
    /// Disposes resources used by the TFT model.
    /// </summary>
    /// <param name="disposing">True if disposing managed resources.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Proper disposal ensures that ONNX sessions and other
    /// resources are released when the model is no longer needed.
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

