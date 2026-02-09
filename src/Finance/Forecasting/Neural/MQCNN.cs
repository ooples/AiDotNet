using System.IO;
using AiDotNet.Enums;
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
namespace AiDotNet.Finance.Forecasting.Neural;

/// <summary>
/// MQCNN (Multi-Quantile Convolutional Neural Network) for probabilistic time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// MQCNN is a probabilistic forecasting model that predicts multiple quantiles simultaneously,
/// providing uncertainty estimates along with point forecasts. It uses an encoder-decoder architecture
/// with dilated causal convolutions for temporal modeling.
/// </para>
/// <para>
/// <b>For Beginners:</b> MQCNN gives you confidence intervals with your predictions:
///
/// <b>What are Quantiles?</b>
/// Quantiles are percentiles of the predicted distribution:
/// - 10th percentile (P10): 10% of values fall below this
/// - 50th percentile (P50): The median (50% above, 50% below)
/// - 90th percentile (P90): 90% of values fall below this
///
/// <b>Why Predict Multiple Quantiles?</b>
/// Instead of just saying "tomorrow's price will be $100":
/// - P10: $95 (likely lower bound - 90% chance actual is above this)
/// - P50: $100 (median prediction)
/// - P90: $105 (likely upper bound - 90% chance actual is below this)
///
/// <b>The Architecture:</b>
/// 1. <b>Encoder:</b> Dilated convolutions extract temporal patterns from history
/// 2. <b>Context:</b> Compressed representation of the encoded sequence
/// 3. <b>Decoder:</b> Produces quantile predictions from the context
///
/// <b>Quantile Loss (Pinball Loss):</b>
/// Unlike MSE which penalizes all errors equally, quantile loss:
/// - For P90: Penalizes under-predictions more than over-predictions
/// - For P10: Penalizes over-predictions more than under-predictions
/// - For P50: Equal penalty (reduces to MAE)
///
/// <b>Example Use Cases:</b>
/// - Stock price prediction with confidence bounds
/// - Demand forecasting with safety stock levels
/// - Energy load forecasting with peak/valley estimates
/// </para>
/// <para>
/// <b>Reference:</b> Wen et al., "A Multi-Horizon Quantile Recurrent Forecaster", 2017.
/// https://arxiv.org/abs/1711.11053
/// </para>
/// </remarks>
public class MQCNN<T> : ForecastingModelBase<T>
{
    #region Execution Mode

    /// <summary>
    /// Indicates whether this network uses native layers (true) or ONNX model (false).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Native mode allows training from scratch, while ONNX mode
    /// runs a pretrained model for inference only.
    /// </para>
    /// </remarks>
    private readonly bool _useNativeMode;

    #endregion

    
    #region Native Mode Fields

    /// <summary>
    /// Encoder input projection layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The first layer that transforms raw input features
    /// into the encoder's internal representation space.
    /// </para>
    /// </remarks>
    private ILayer<T>? _encoderInputProjection;

    /// <summary>
    /// Encoder convolution layers for temporal pattern extraction.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> These layers process the sequence through dilated convolutions,
    /// each with increasing receptive field to capture patterns at different time scales.
    /// </para>
    /// </remarks>
    private readonly List<ILayer<T>> _encoderLayers = [];

    /// <summary>
    /// Context projection layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Compresses the encoder output into a fixed-size context
    /// representation that summarizes the historical patterns.
    /// </para>
    /// </remarks>
    private ILayer<T>? _contextLayer;

    /// <summary>
    /// Decoder layers for quantile prediction.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> These layers transform the context into predictions
    /// for each quantile at each future time step.
    /// </para>
    /// </remarks>
    private readonly List<ILayer<T>> _decoderLayers = [];

    /// <summary>
    /// Final output layer producing quantile predictions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Outputs a tensor of shape [forecastHorizon * numQuantiles],
    /// where each time step has predictions for all quantiles (e.g., P10, P50, P90).
    /// </para>
    /// </remarks>
    private ILayer<T>? _outputLayer;

    #endregion

    #region Shared Fields

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly MQCNNOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private int _lookbackWindow;
    private int _forecastHorizon;
    private int _numFeatures;
    private double[] _quantiles;
    private int _encoderChannels;
    private int _decoderChannels;
    private int _numEncoderLayers;
    private int _numDecoderLayers;
    private double _dropout;

    #endregion

    #region IForecastingModel Properties

    /// <inheritdoc/>
    public override int SequenceLength => _lookbackWindow;

    /// <inheritdoc/>
    public override int PredictionHorizon => _forecastHorizon;

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

    /// <summary>
    /// Gets the number of quantiles this model predicts.
    /// </summary>
    /// <value>The number of quantiles.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How many percentiles the model outputs. Default is 3
    /// (P10, P50, P90), but you can configure more for finer uncertainty estimation.
    /// </para>
    /// </remarks>
    public int NumQuantiles => _quantiles.Length;

    /// <summary>
    /// Gets the quantile levels being predicted.
    /// </summary>
    /// <value>Array of quantile levels (values between 0 and 1).</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The specific percentiles being predicted. Default [0.1, 0.5, 0.9]
    /// gives you the 10th, 50th, and 90th percentiles.
    /// </para>
    /// </remarks>
    public double[] Quantiles => _quantiles;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates an MQCNN using pretrained ONNX model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer for fine-tuning.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor to load a pretrained MQCNN model
    /// for inference. The ONNX model should output [forecastHorizon * numQuantiles] values.
    /// </para>
    /// </remarks>
    public MQCNN(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        MQCNNOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        options ??= new MQCNNOptions<T>();
        _options = options;
        Options = _options;
        ValidateOptions(options);

        _useNativeMode = false;
        OnnxSession = new InferenceSession(onnxModelPath);
        OnnxModelPath = onnxModelPath;

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        _lookbackWindow = options.LookbackWindow;
        _forecastHorizon = options.ForecastHorizon;
        _numFeatures = architecture.InputSize > 0 ? architecture.InputSize : 1;
        _quantiles = options.Quantiles;
        _encoderChannels = options.EncoderChannels;
        _decoderChannels = options.DecoderChannels;
        _numEncoderLayers = options.NumEncoderLayers;
        _numDecoderLayers = options.NumDecoderLayers;
        _dropout = options.DropoutRate;
    }

    /// <summary>
    /// Creates an MQCNN in native mode for training from scratch.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor to train a new MQCNN model.
    /// MQCNN is ideal when you need:
    /// - Uncertainty estimates with your forecasts
    /// - Prediction intervals for risk management
    /// - Probabilistic forecasts for decision making under uncertainty
    /// </para>
    /// </remarks>
    public MQCNN(
        NeuralNetworkArchitecture<T> architecture,
        MQCNNOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new MQCNNOptions<T>();
        _options = options;
        Options = _options;
        ValidateOptions(options);

        _useNativeMode = true;
        OnnxSession = null;
        OnnxModelPath = null;

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        _lookbackWindow = options.LookbackWindow;
        _forecastHorizon = options.ForecastHorizon;
        _numFeatures = architecture.InputSize > 0 ? architecture.InputSize : 1;
        _quantiles = options.Quantiles;
        _encoderChannels = options.EncoderChannels;
        _decoderChannels = options.DecoderChannels;
        _numEncoderLayers = options.NumEncoderLayers;
        _numDecoderLayers = options.NumDecoderLayers;
        _dropout = options.DropoutRate;

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes the neural network layers for MQCNN.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This sets up the MQCNN architecture:
    /// 1. Encoder input projection
    /// 2. Encoder convolution layers (with dropout)
    /// 3. Context compression layer
    /// 4. Decoder layers (with dropout)
    /// 5. Output layer for quantile predictions
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultMQCNNLayers(
                Architecture, _lookbackWindow, _forecastHorizon, _numFeatures,
                _quantiles.Length, _encoderChannels, _decoderChannels,
                _numEncoderLayers, _numDecoderLayers, _dropout));

            ExtractLayerReferences();
        }
    }

    /// <summary>
    /// Extracts references to specific layers from the layer collection.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Organizes layers into encoder, context, decoder, and output
    /// components for the forward pass. The encoder extracts patterns, the decoder
    /// produces quantile predictions.
    /// </para>
    /// </remarks>
    private void ExtractLayerReferences()
    {
        int idx = 0;

        // Encoder input projection
        if (idx < Layers.Count)
            _encoderInputProjection = Layers[idx++];

        // Encoder layers (with optional dropout)
        _encoderLayers.Clear();
        int encoderLayersWithDropout = _dropout > 0 ? _numEncoderLayers * 2 : _numEncoderLayers;
        for (int i = 0; i < encoderLayersWithDropout && idx < Layers.Count - 2 - (_dropout > 0 ? _numDecoderLayers * 2 : _numDecoderLayers); i++)
        {
            _encoderLayers.Add(Layers[idx++]);
        }

        // Context layer
        if (idx < Layers.Count)
            _contextLayer = Layers[idx++];

        // Decoder layers (with optional dropout)
        _decoderLayers.Clear();
        int decoderLayersWithDropout = _dropout > 0 ? _numDecoderLayers * 2 : _numDecoderLayers;
        for (int i = 0; i < decoderLayersWithDropout && idx < Layers.Count - 1; i++)
        {
            _decoderLayers.Add(Layers[idx++]);
        }

        // Output layer
        if (idx < Layers.Count)
            _outputLayer = Layers[idx];
    }

    /// <summary>
    /// Validates that custom layers meet MQCNN architectural requirements.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Ensures you have at least encoder, context, decoder, and output layers.
    /// </para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);
        if (layers.Count < 4)
        {
            throw new ArgumentException(
                "MQCNN requires at least 4 layers: encoder input, context, decoder, and output.",
                nameof(layers));
        }
    }

    /// <summary>
    /// Validates the MQCNN options.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Checks that all configuration values are valid:
    /// - Window sizes must be positive
    /// - Quantiles must be between 0 and 1
    /// - Channel counts must be positive
    /// </para>
    /// </remarks>
    private static void ValidateOptions(MQCNNOptions<T> options)
    {
        var errors = new List<string>();

        if (options.LookbackWindow < 1)
            errors.Add("LookbackWindow must be at least 1.");
        if (options.ForecastHorizon < 1)
            errors.Add("ForecastHorizon must be at least 1.");
        if (options.Quantiles == null || options.Quantiles.Length == 0)
            errors.Add("Quantiles must have at least one value.");
        else
        {
            foreach (var q in options.Quantiles)
            {
                if (q <= 0 || q >= 1)
                {
                    errors.Add($"Quantile {q} must be between 0 and 1 (exclusive).");
                    break;
                }
            }
        }
        if (options.EncoderChannels < 1)
            errors.Add("EncoderChannels must be at least 1.");
        if (options.DecoderChannels < 1)
            errors.Add("DecoderChannels must be at least 1.");
        if (options.NumEncoderLayers < 1)
            errors.Add("NumEncoderLayers must be at least 1.");
        if (options.NumDecoderLayers < 1)
            errors.Add("NumDecoderLayers must be at least 1.");
        if (options.DropoutRate < 0 || options.DropoutRate >= 1)
            errors.Add("DropoutRate must be between 0 and 1 (exclusive).");

        if (errors.Count > 0)
            throw new ArgumentException($"Invalid options: {string.Join(", ", errors)}");
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <inheritdoc/>
    public override bool SupportsTraining => _useNativeMode;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MQCNN model, Predict produces predictions from input data. This is the main inference step of the MQCNN architecture.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return _useNativeMode ? ForecastNative(input) : ForecastOnnx(input);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Training MQCNN uses quantile loss (pinball loss),
    /// which penalizes predictions differently based on whether they're above or
    /// below the actual value, and based on which quantile is being predicted.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is only supported in native mode.");

        SetTrainingMode(true);

        var predictions = Forward(input);

        // Calculate quantile loss
        LastLoss = CalculateQuantileLoss(predictions, target);

        var gradient = CalculateQuantileLossGradient(predictions, target);
        Backward(gradient);

        _optimizer.UpdateParameters(Layers);

        SetTrainingMode(false);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MQCNN model, UpdateParameters updates internal parameters or state. This keeps the MQCNN architecture aligned with the latest values.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> gradients)
    {
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MQCNN model, GetModelMetadata performs a supporting step in the workflow. It keeps the MQCNN architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "MQCNN" },
                { "LookbackWindow", _lookbackWindow },
                { "ForecastHorizon", _forecastHorizon },
                { "Quantiles", _quantiles },
                { "NumQuantiles", _quantiles.Length },
                { "EncoderChannels", _encoderChannels },
                { "DecoderChannels", _decoderChannels },
                { "NumEncoderLayers", _numEncoderLayers },
                { "NumDecoderLayers", _numDecoderLayers },
                { "UseNativeMode", _useNativeMode },
                { "ParameterCount", GetParameterCount() }
            },
            ModelData = _useNativeMode ? this.Serialize() : Array.Empty<byte>()
        };
    }

    /// <summary>
    /// Creates a new instance of this model with the same configuration.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a fresh copy of the model architecture,
    /// useful for ensemble methods or hyperparameter search.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var options = new MQCNNOptions<T>
        {
            LookbackWindow = _lookbackWindow,
            ForecastHorizon = _forecastHorizon,
            Quantiles = _quantiles,
            EncoderChannels = _encoderChannels,
            DecoderChannels = _decoderChannels,
            NumEncoderLayers = _numEncoderLayers,
            NumDecoderLayers = _numDecoderLayers,
            DropoutRate = _dropout
        };

        return new MQCNN<T>(Architecture, options);
    }

    /// <summary>
    /// Writes MQCNN-specific configuration during serialization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Saves all the configuration needed to reconstruct this model.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_lookbackWindow);
        writer.Write(_forecastHorizon);
        writer.Write(_numFeatures);
        writer.Write(_quantiles.Length);
        foreach (var q in _quantiles)
            writer.Write(q);
        writer.Write(_encoderChannels);
        writer.Write(_decoderChannels);
        writer.Write(_numEncoderLayers);
        writer.Write(_numDecoderLayers);
        writer.Write(_dropout);
    }

    /// <summary>
    /// Reads MQCNN-specific configuration during deserialization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads the configuration that was saved during serialization.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _lookbackWindow = reader.ReadInt32();
        _forecastHorizon = reader.ReadInt32();
        _numFeatures = reader.ReadInt32();
        int numQuantiles = reader.ReadInt32();
        _quantiles = new double[numQuantiles];
        for (int i = 0; i < numQuantiles; i++)
            _quantiles[i] = reader.ReadDouble();
        _encoderChannels = reader.ReadInt32();
        _decoderChannels = reader.ReadInt32();
        _numEncoderLayers = reader.ReadInt32();
        _numDecoderLayers = reader.ReadInt32();
        _dropout = reader.ReadDouble();
    }

    #endregion

    #region IForecastingModel Implementation

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MQCNN model, Forecast produces predictions from input data. This is the main inference step of the MQCNN architecture.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
    {
        return _useNativeMode ? ForecastNative(historicalData) : ForecastOnnx(historicalData);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> For multi-step forecasting beyond the horizon,
    /// MQCNN rolls forward using the median (P50) prediction as the "point forecast"
    /// for the next step's input.
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

            // Extract median (P50) for rolling forward
            var medianPrediction = ExtractQuantilePrediction(prediction, 0.5);
            predictions.Add(prediction);

            int stepsUsed = Math.Min(_forecastHorizon, stepsRemaining);
            stepsRemaining -= stepsUsed;

            if (stepsRemaining > 0)
            {
                currentInput = ShiftInputWithPredictions(currentInput, medianPrediction, stepsUsed);
            }
        }

        return ConcatenatePredictions(predictions, steps);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MQCNN model, Evaluate performs a supporting step in the workflow. It keeps the MQCNN architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> Evaluate(Tensor<T> predictions, Tensor<T> actuals)
    {
        var metrics = new Dictionary<string, T>();

        // Calculate overall quantile loss
        T totalQuantileLoss = CalculateQuantileLoss(predictions, actuals);
        metrics["QuantileLoss"] = totalQuantileLoss;

        // Calculate MSE and MAE for median prediction
        var medianPred = ExtractQuantilePrediction(predictions, 0.5);
        T mse = NumOps.Zero;
        T mae = NumOps.Zero;
        int count = 0;

        for (int i = 0; i < medianPred.Length && i < actuals.Length; i++)
        {
            var diff = NumOps.Subtract(medianPred[i], actuals[i]);
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

        // Calculate coverage (percentage of actuals within P10-P90)
        metrics["Coverage"] = CalculateCoverage(predictions, actuals, 0.1, 0.9);

        return metrics;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MQCNN model, ApplyInstanceNormalization performs a supporting step in the workflow. It keeps the MQCNN architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
    {
        return input;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MQCNN model, GetFinancialMetrics calculates evaluation metrics. This summarizes how the MQCNN architecture is performing.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> GetFinancialMetrics()
    {
        T lastLoss = LastLoss is not null ? LastLoss : NumOps.Zero;

        return new Dictionary<string, T>
        {
            ["LookbackWindow"] = NumOps.FromDouble(_lookbackWindow),
            ["ForecastHorizon"] = NumOps.FromDouble(_forecastHorizon),
            ["NumQuantiles"] = NumOps.FromDouble(_quantiles.Length),
            ["EncoderChannels"] = NumOps.FromDouble(_encoderChannels),
            ["DecoderChannels"] = NumOps.FromDouble(_decoderChannels),
            ["LastLoss"] = lastLoss
        };
    }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// Performs the forward pass through MQCNN.
    /// </summary>
    /// <param name="input">Input tensor of shape [batch, lookback_window * features].</param>
    /// <returns>Output tensor of shape [batch, forecast_horizon * num_quantiles].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The MQCNN forward pass:
    ///
    /// 1. <b>Encoder Input</b>: Project raw features to encoder dimension
    ///
    /// 2. <b>Encoder Layers</b>: Apply dilated convolutions for temporal patterns
    ///
    /// 3. <b>Context</b>: Compress encoder output to context representation
    ///
    /// 4. <b>Decoder Layers</b>: Process context for quantile prediction
    ///
    /// 5. <b>Output</b>: Generate [forecastHorizon * numQuantiles] predictions
    ///
    /// The output is arranged as: [t1_q1, t1_q2, t1_q3, t2_q1, t2_q2, t2_q3, ...]
    /// where t=time step and q=quantile.
    /// </para>
    /// </remarks>
    private Tensor<T> Forward(Tensor<T> input)
    {
        if (!_useNativeMode)
            return ForecastOnnx(input);

        var current = input;

        // Encoder input projection
        if (_encoderInputProjection is not null)
            current = _encoderInputProjection.Forward(current);

        // Encoder layers
        foreach (var layer in _encoderLayers)
        {
            current = layer.Forward(current);
        }

        // Context layer
        if (_contextLayer is not null)
            current = _contextLayer.Forward(current);

        // Decoder layers
        foreach (var layer in _decoderLayers)
        {
            current = layer.Forward(current);
        }

        // Output layer
        if (_outputLayer is not null)
            current = _outputLayer.Forward(current);

        return current;
    }

    /// <summary>
    /// Performs the backward pass through MQCNN.
    /// </summary>
    /// <param name="gradOutput">Gradient from the loss function.</param>
    /// <returns>Gradient with respect to the input.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Backpropagation computes how each parameter contributed
    /// to the prediction error. For MQCNN, the gradient is computed using quantile loss,
    /// which has different gradients depending on whether predictions were above or below actual values.
    /// </para>
    /// </remarks>
    private Tensor<T> Backward(Tensor<T> gradOutput)
    {
        var current = gradOutput;

        // Output layer backward
        if (_outputLayer is not null)
            current = _outputLayer.Backward(current);

        // Decoder layers backward
        for (int i = _decoderLayers.Count - 1; i >= 0; i--)
        {
            current = _decoderLayers[i].Backward(current);
        }

        // Context layer backward
        if (_contextLayer is not null)
            current = _contextLayer.Backward(current);

        // Encoder layers backward
        for (int i = _encoderLayers.Count - 1; i >= 0; i--)
        {
            current = _encoderLayers[i].Backward(current);
        }

        // Encoder input projection backward
        if (_encoderInputProjection is not null)
            current = _encoderInputProjection.Backward(current);

        return current;
    }

    #endregion

    #region Quantile Loss Methods

    /// <summary>
    /// Calculates the quantile loss (pinball loss) for the predictions.
    /// </summary>
    /// <param name="predictions">Predicted values arranged as [t1_q1, t1_q2, ..., t2_q1, t2_q2, ...].</param>
    /// <param name="actuals">Actual values (one per time step).</param>
    /// <returns>The average quantile loss across all time steps and quantiles.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Quantile loss (also called pinball loss) penalizes
    /// predictions differently based on the quantile level:
    ///
    /// For quantile q and error e = actual - predicted:
    /// - If e >= 0 (under-prediction): loss = q * e
    /// - If e &lt; 0 (over-prediction): loss = (q - 1) * e = (1 - q) * |e|
    ///
    /// Example for q = 0.9 (90th percentile):
    /// - Under-predict by 10: loss = 0.9 * 10 = 9
    /// - Over-predict by 10: loss = 0.1 * 10 = 1
    /// This makes the model prefer over-predicting for high quantiles.
    /// </para>
    /// </remarks>
    private T CalculateQuantileLoss(Tensor<T> predictions, Tensor<T> actuals)
    {
        T totalLoss = NumOps.Zero;
        int count = 0;

        for (int t = 0; t < _forecastHorizon && t < actuals.Length; t++)
        {
            T actual = actuals[t];

            for (int q = 0; q < _quantiles.Length; q++)
            {
                int predIdx = t * _quantiles.Length + q;
                if (predIdx >= predictions.Length)
                    continue;

                T predicted = predictions[predIdx];
                T error = NumOps.Subtract(actual, predicted);
                double quantile = _quantiles[q];

                // Quantile loss: q * max(error, 0) + (1-q) * max(-error, 0)
                T loss;
                if (NumOps.ToDouble(error) >= 0)
                {
                    loss = NumOps.Multiply(NumOps.FromDouble(quantile), error);
                }
                else
                {
                    loss = NumOps.Multiply(NumOps.FromDouble(quantile - 1), error);
                }

                totalLoss = NumOps.Add(totalLoss, loss);
                count++;
            }
        }

        return count > 0 ? NumOps.Divide(totalLoss, NumOps.FromDouble(count)) : NumOps.Zero;
    }

    /// <summary>
    /// Calculates the gradient of quantile loss for backpropagation.
    /// </summary>
    /// <param name="predictions">Predicted values.</param>
    /// <param name="actuals">Actual values.</param>
    /// <returns>Gradient tensor for backpropagation.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The gradient of quantile loss is:
    /// - If actual >= predicted: gradient = -q (we should have predicted higher)
    /// - If actual &lt; predicted: gradient = 1-q (we should have predicted lower)
    ///
    /// This asymmetric gradient is what makes the model learn different quantiles.
    /// </para>
    /// </remarks>
    private Tensor<T> CalculateQuantileLossGradient(Tensor<T> predictions, Tensor<T> actuals)
    {
        var gradient = new Tensor<T>(predictions.Shape);
        int totalElements = _forecastHorizon * _quantiles.Length;

        for (int t = 0; t < _forecastHorizon && t < actuals.Length; t++)
        {
            T actual = actuals[t];

            for (int q = 0; q < _quantiles.Length; q++)
            {
                int predIdx = t * _quantiles.Length + q;
                if (predIdx >= predictions.Length || predIdx >= totalElements)
                    continue;

                T predicted = predictions[predIdx];
                double quantile = _quantiles[q];

                // Gradient of quantile loss
                T grad;
                if (NumOps.ToDouble(NumOps.Subtract(actual, predicted)) >= 0)
                {
                    grad = NumOps.FromDouble(-quantile); // Under-prediction
                }
                else
                {
                    grad = NumOps.FromDouble(1 - quantile); // Over-prediction
                }

                gradient.Data.Span[predIdx] = NumOps.Divide(grad, NumOps.FromDouble(totalElements));
            }
        }

        return gradient;
    }

    /// <summary>
    /// Extracts predictions for a specific quantile level.
    /// </summary>
    /// <param name="predictions">Full prediction tensor with all quantiles.</param>
    /// <param name="targetQuantile">The quantile level to extract (e.g., 0.5 for median).</param>
    /// <returns>Tensor containing only the predictions for the specified quantile.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Since MQCNN outputs all quantiles interleaved,
    /// this extracts just one quantile (like the median) for analysis or rolling forecasts.
    /// </para>
    /// </remarks>
    private Tensor<T> ExtractQuantilePrediction(Tensor<T> predictions, double targetQuantile)
    {
        // Find the index of the closest quantile
        int qIdx = 0;
        double minDiff = double.MaxValue;
        for (int q = 0; q < _quantiles.Length; q++)
        {
            double diff = Math.Abs(_quantiles[q] - targetQuantile);
            if (diff < minDiff)
            {
                minDiff = diff;
                qIdx = q;
            }
        }

        // Extract predictions for this quantile
        var result = new Tensor<T>(new[] { _forecastHorizon });
        for (int t = 0; t < _forecastHorizon; t++)
        {
            int predIdx = t * _quantiles.Length + qIdx;
            if (predIdx < predictions.Length)
            {
                result.Data.Span[t] = predictions.Data.Span[predIdx];
            }
        }

        return result;
    }

    /// <summary>
    /// Calculates the coverage percentage (fraction of actuals within prediction interval).
    /// </summary>
    /// <param name="predictions">Predicted quantiles.</param>
    /// <param name="actuals">Actual values.</param>
    /// <param name="lowerQuantile">Lower bound quantile (e.g., 0.1).</param>
    /// <param name="upperQuantile">Upper bound quantile (e.g., 0.9).</param>
    /// <returns>Fraction of actuals within the prediction interval.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Coverage measures how well the prediction intervals
    /// capture the actual values. For an 80% interval (P10 to P90), we expect
    /// about 80% of actuals to fall within this range.
    /// </para>
    /// </remarks>
    private T CalculateCoverage(Tensor<T> predictions, Tensor<T> actuals, double lowerQuantile, double upperQuantile)
    {
        var lowerPred = ExtractQuantilePrediction(predictions, lowerQuantile);
        var upperPred = ExtractQuantilePrediction(predictions, upperQuantile);

        int covered = 0;
        int total = 0;

        for (int t = 0; t < _forecastHorizon && t < actuals.Length; t++)
        {
            T actual = actuals[t];
            T lower = lowerPred[t];
            T upper = upperPred[t];

            if (NumOps.ToDouble(actual) >= NumOps.ToDouble(lower) &&
                NumOps.ToDouble(actual) <= NumOps.ToDouble(upper))
            {
                covered++;
            }
            total++;
        }

        return total > 0 ? NumOps.FromDouble((double)covered / total) : NumOps.Zero;
    }

    #endregion

    #region Inference Methods

    /// <summary>
    /// Performs native mode forecasting.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Uses the trained neural network layers to produce
    /// quantile predictions from the input sequence.
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
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Uses a pretrained ONNX model for fast inference.
    /// ONNX Runtime provides optimized execution on CPUs and GPUs.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ForecastOnnx(Tensor<T> input)
    {
        if (OnnxSession is null)
            throw new InvalidOperationException("ONNX session is not initialized.");

        // Convert input to float array
        var inputData = new float[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            inputData[i] = Convert.ToSingle(NumOps.ToDouble(input.Data.Span[i]));
        }

        // Create ONNX tensor using input shape
        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input.Shape);
        var inputMeta = OnnxSession.InputMetadata;
        string inputName = inputMeta.Keys.First();

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, onnxInput)
        };

        // Run inference
        using var results = OnnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        // Convert back to Tensor<T>
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
    /// Shifts input tensor by incorporating predictions for autoregressive forecasting.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> For multi-step forecasting, we slide the input window forward
    /// and add our predictions as new "history" for the next forecast.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ShiftInputWithPredictions(Tensor<T> input, Tensor<T> predictions, int stepsUsed)
    {
        int totalElements = _lookbackWindow * _numFeatures;
        var newInput = new Tensor<T>(input.Shape);

        // Shift old values left
        int shift = stepsUsed * _numFeatures;
        for (int i = 0; i < totalElements - shift; i++)
        {
            newInput.Data.Span[i] = input.Data.Span[i + shift];
        }

        // Add predictions at the end
        for (int i = 0; i < stepsUsed && i < predictions.Length; i++)
        {
            int targetIdx = totalElements - shift + i * _numFeatures;
            if (targetIdx < totalElements)
            {
                newInput.Data.Span[targetIdx] = predictions[i];
            }
        }

        return newInput;
    }

    /// <summary>
    /// Concatenates multiple prediction tensors for extended horizons.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When forecasting beyond the model's horizon,
    /// we combine multiple prediction batches into one result.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ConcatenatePredictions(List<Tensor<T>> predictions, int totalSteps)
    {
        int totalOutputSize = totalSteps * _quantiles.Length;
        var result = new Tensor<T>(new[] { totalOutputSize });

        int resultIdx = 0;
        int stepsAdded = 0;

        foreach (var pred in predictions)
        {
            int stepsToAdd = Math.Min(_forecastHorizon, totalSteps - stepsAdded);
            int elementsToAdd = stepsToAdd * _quantiles.Length;

            for (int i = 0; i < elementsToAdd && resultIdx < totalOutputSize; i++)
            {
                if (i < pred.Length)
                    result.Data.Span[resultIdx++] = pred[i];
            }

            stepsAdded += stepsToAdd;
            if (stepsAdded >= totalSteps)
                break;
        }

        return result;
    }

    #endregion

    #region IDisposable

    /// <summary>
    /// Disposes resources used by the MQCNN model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Releases the ONNX session and other resources when the model
    /// is no longer needed.
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

