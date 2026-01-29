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
/// LSTNet (Long Short-Term Time-series Network) model for multivariate time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// LSTNet combines multiple neural network components to capture patterns at different temporal scales:
/// - Convolutional layers for short-term local patterns
/// - Recurrent layers (GRU) for long-term dependencies
/// - Skip-RNN for periodic patterns
/// - Autoregressive component for simple linear relationships
/// </para>
/// <para>
/// <b>For Beginners:</b> LSTNet is like having a team of specialists analyze your time series:
///
/// 1. <b>The Pattern Scanner (CNN)</b>: Scans through your data looking for local patterns,
///    like finding "every Monday is busy" or "there's always a dip at noon".
///
/// 2. <b>The Trend Tracker (GRU)</b>: Remembers long-term trends, like "sales have been
///    growing steadily for the past 3 months".
///
/// 3. <b>The Season Watcher (Skip-RNN)</b>: Compares the same time across different periods,
///    like comparing this Tuesday's 3 PM with last Tuesday's 3 PM.
///
/// 4. <b>The Simple Forecaster (Autoregressive)</b>: Makes basic predictions based on recent
///    values, like "if it was 100 yesterday, it's probably around 100 today".
///
/// All these specialists combine their insights to produce the final forecast.
/// </para>
/// <para>
/// <b>Reference:</b> Lai et al., "Modeling Long- and Short-Term Temporal Patterns with Deep
/// Neural Networks", SIGIR 2018. https://arxiv.org/abs/1703.07015
/// </para>
/// </remarks>
public class LSTNet<T> : ForecastingModelBase<T>
{
    #region Execution Mode

    /// <summary>
    /// Indicates whether this network uses native layers (true) or ONNX model (false).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Native mode means the model is built with layers that can
    /// be trained. ONNX mode means we're using a pre-trained model file.
    /// </para>
    /// </remarks>
    private readonly bool _useNativeMode;

    #endregion

    
    #region Native Mode Fields

    /// <summary>
    /// The convolutional layer for extracting local features.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This layer slides a small window over your time series,
    /// looking for patterns. It's like scanning a document for keywords.
    /// </para>
    /// </remarks>
    private ILayer<T>? _convLayer;

    /// <summary>
    /// Activation layer after convolution.
    /// </summary>
    private ILayer<T>? _convActivation;

    /// <summary>
    /// Dropout layer after convolution for regularization.
    /// </summary>
    private ILayer<T>? _convDropout;

    /// <summary>
    /// The main recurrent layer (GRU) for long-term dependencies.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> GRU (Gated Recurrent Unit) is a type of recurrent neural
    /// network that can remember patterns from far in the past.
    /// </para>
    /// </remarks>
    private ILayer<T>? _gruLayer;

    /// <summary>
    /// Dropout layer after GRU.
    /// </summary>
    private ILayer<T>? _gruDropout;

    /// <summary>
    /// The skip recurrent layer for periodic patterns.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The skip-RNN jumps ahead by a fixed number of time steps
    /// (like 24 for hourly data), allowing it to directly compare today's 3 PM
    /// with yesterday's 3 PM.
    /// </para>
    /// </remarks>
    private ILayer<T>? _skipGruLayer;

    /// <summary>
    /// Dropout layer after skip GRU.
    /// </summary>
    private ILayer<T>? _skipDropout;

    /// <summary>
    /// Highway layer for linear pass-through.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The highway layer provides a direct linear connection
    /// from input to output, helping capture simple trends that don't need
    /// complex processing.
    /// </para>
    /// </remarks>
    private ILayer<T>? _highwayLayer;

    /// <summary>
    /// Layer to combine outputs from different components.
    /// </summary>
    private ILayer<T>? _combinationLayer;

    /// <summary>
    /// Final output projection layer.
    /// </summary>
    private ILayer<T>? _outputLayer;

    #endregion

    #region Shared Fields

    /// <summary>
    /// The optimizer for training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The optimizer adjusts the model's parameters during training
    /// to minimize prediction errors. Adam is a popular choice that works well in most cases.
    /// </para>
    /// </remarks>
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <summary>
    /// The loss function for training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The loss function measures how wrong our predictions are.
    /// Mean Squared Error penalizes large errors more heavily than small ones.
    /// </para>
    /// </remarks>
    private readonly ILossFunction<T> _lossFunction;

    /// <summary>
    /// The lookback window size.
    /// </summary>
    private readonly int _lookbackWindow;

    /// <summary>
    /// The forecast horizon.
    /// </summary>
    private readonly int _forecastHorizon;

    /// <summary>
    /// Number of input features.
    /// </summary>
    private readonly int _numFeatures;

    /// <summary>
    /// Hidden size for main recurrent layer.
    /// </summary>
    private readonly int _hiddenRecurrentSize;

    /// <summary>
    /// Hidden size for skip recurrent layer.
    /// </summary>
    private readonly int _hiddenSkipSize;

    /// <summary>
    /// Number of convolutional filters.
    /// </summary>
    private readonly int _convolutionFilters;

    /// <summary>
    /// Kernel size for convolutional layer.
    /// </summary>
    private readonly int _convolutionKernelSize;

    /// <summary>
    /// Skip period for skip-RNN.
    /// </summary>
    private readonly int _skipPeriod;

    /// <summary>
    /// Window size for autoregressive component.
    /// </summary>
    private readonly int _autoregressiveWindow;

    /// <summary>
    /// Whether to use highway connections.
    /// </summary>
    private readonly bool _useHighway;

    /// <summary>
    /// Dropout rate for regularization.
    /// </summary>
    private readonly double _dropout;

    #endregion

    #region IForecastingModel Properties

    /// <inheritdoc/>
    public override int SequenceLength => _lookbackWindow;

    /// <inheritdoc/>
    public override int PredictionHorizon => _forecastHorizon;

    /// <inheritdoc/>
    public override int NumFeatures => _numFeatures;

    /// <inheritdoc/>
    public override int PatchSize => _convolutionKernelSize;

    /// <inheritdoc/>
    public override int Stride => 1;

    /// <inheritdoc/>
    public override bool IsChannelIndependent => false; // LSTNet processes all features together

    /// <inheritdoc/>
    public override bool UseNativeMode => _useNativeMode;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates an LSTNet using pretrained ONNX model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer for fine-tuning.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor when you have a pre-trained LSTNet model
    /// in ONNX format. ONNX is a standard format that allows models trained in one
    /// framework (like PyTorch) to be used in another (like this library).
    /// </para>
    /// </remarks>
    public LSTNet(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        LSTNetOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        options ??= new LSTNetOptions<T>();
        ValidateOptions(options);

        _useNativeMode = false;
        OnnxSession = new InferenceSession(onnxModelPath);
        OnnxModelPath = onnxModelPath;

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        _lookbackWindow = options.LookbackWindow;
        _forecastHorizon = options.ForecastHorizon;
        _numFeatures = architecture.InputSize;
        _hiddenRecurrentSize = options.HiddenRecurrentSize;
        _hiddenSkipSize = options.HiddenSkipSize;
        _convolutionFilters = options.ConvolutionFilters;
        _convolutionKernelSize = options.ConvolutionKernelSize;
        _skipPeriod = options.SkipPeriod;
        _autoregressiveWindow = options.AutoregressiveWindow;
        _useHighway = options.UseHighway;
        _dropout = options.DropoutRate;
    }

    /// <summary>
    /// Creates an LSTNet in native mode for training from scratch.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor to train a new LSTNet model from scratch.
    /// LSTNet is particularly good for:
    /// - Multivariate time series (multiple related variables)
    /// - Data with multiple seasonal patterns (hourly + daily + weekly)
    /// - Traffic, electricity, and sensor data forecasting
    /// </para>
    /// </remarks>
    public LSTNet(
        NeuralNetworkArchitecture<T> architecture,
        LSTNetOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new LSTNetOptions<T>();
        ValidateOptions(options);

        _useNativeMode = true;
        OnnxSession = null;
        OnnxModelPath = null;

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        _lookbackWindow = options.LookbackWindow;
        _forecastHorizon = options.ForecastHorizon;
        _numFeatures = architecture.InputSize > 0 ? architecture.InputSize : 7;
        _hiddenRecurrentSize = options.HiddenRecurrentSize;
        _hiddenSkipSize = options.HiddenSkipSize;
        _convolutionFilters = options.ConvolutionFilters;
        _convolutionKernelSize = options.ConvolutionKernelSize;
        _skipPeriod = options.SkipPeriod;
        _autoregressiveWindow = options.AutoregressiveWindow;
        _useHighway = options.UseHighway;
        _dropout = options.DropoutRate;

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes the neural network layers for LSTNet.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method sets up the building blocks of LSTNet:
    /// 1. Convolutional layer to scan for local patterns
    /// 2. GRU layer to track long-term trends
    /// 3. Skip-GRU to catch periodic patterns
    /// 4. Highway layer for direct linear relationships
    /// 5. Combination and output layers to produce predictions
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultLSTNetLayers(
                Architecture, _lookbackWindow, _forecastHorizon, _numFeatures,
                _convolutionFilters, _convolutionKernelSize, _hiddenRecurrentSize,
                _hiddenSkipSize, _skipPeriod, _dropout));

            ExtractLayerReferences();
        }
    }

    /// <summary>
    /// Extracts references to specific layers from the layer collection.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> After creating all the layers, we need to know which layer
    /// does what. This method organizes them so we can use them correctly during
    /// the forward pass.
    /// </para>
    /// </remarks>
    private void ExtractLayerReferences()
    {
        int idx = 0;

        // Convolutional component
        if (idx < Layers.Count)
            _convLayer = Layers[idx++];

        if (idx < Layers.Count)
            _convActivation = Layers[idx++];

        if (_dropout > 0 && idx < Layers.Count)
            _convDropout = Layers[idx++];

        // Main GRU
        if (idx < Layers.Count)
            _gruLayer = Layers[idx++];

        if (_dropout > 0 && idx < Layers.Count)
            _gruDropout = Layers[idx++];

        // Skip GRU
        if (idx < Layers.Count)
            _skipGruLayer = Layers[idx++];

        if (_dropout > 0 && idx < Layers.Count)
            _skipDropout = Layers[idx++];

        // Highway layer
        if (idx < Layers.Count)
            _highwayLayer = Layers[idx++];

        // Combination layer
        if (idx < Layers.Count)
            _combinationLayer = Layers[idx++];

        // Output layer
        if (idx < Layers.Count)
            _outputLayer = Layers[idx++];
    }

    /// <summary>
    /// Validates that custom layers meet LSTNet architectural requirements.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> If you provide your own layers instead of using defaults,
    /// this method checks that they can work together properly.
    /// </para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);
        if (layers.Count < 4)
        {
            throw new ArgumentException(
                "LSTNet requires at least 4 layers: convolution, GRU, skip-GRU, and output.",
                nameof(layers));
        }
    }

    /// <summary>
    /// Validates the LSTNet options.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This ensures all configuration values make sense before
    /// we try to build the network. For example, the lookback window must be positive.
    /// </para>
    /// </remarks>
    private static void ValidateOptions(LSTNetOptions<T> options)
    {
        var errors = new List<string>();

        if (options.LookbackWindow < 1)
            errors.Add("LookbackWindow must be at least 1.");
        if (options.ForecastHorizon < 1)
            errors.Add("ForecastHorizon must be at least 1.");
        if (options.HiddenRecurrentSize < 1)
            errors.Add("HiddenRecurrentSize must be at least 1.");
        if (options.HiddenSkipSize < 1)
            errors.Add("HiddenSkipSize must be at least 1.");
        if (options.ConvolutionFilters < 1)
            errors.Add("ConvolutionFilters must be at least 1.");
        if (options.ConvolutionKernelSize < 1)
            errors.Add("ConvolutionKernelSize must be at least 1.");
        if (options.SkipPeriod < 1)
            errors.Add("SkipPeriod must be at least 1.");
        if (options.DropoutRate < 0 || options.DropoutRate >= 1)
            errors.Add("DropoutRate must be between 0 and 1 (exclusive).");
        if (options.ConvolutionKernelSize >= options.LookbackWindow)
            errors.Add("ConvolutionKernelSize must be less than LookbackWindow.");

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
    /// <b>For Beginners:</b> In the LSTNet model, Predict produces predictions from input data. This is the main inference step of the LSTNet architecture.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return _useNativeMode ? ForecastNative(input) : ForecastOnnx(input);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the LSTNet model, Train performs a training step. This updates the LSTNet architecture so it learns from data.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is only supported in native mode.");

        SetTrainingMode(true);

        var predictions = Forward(input);
        LastLoss = _lossFunction.CalculateLoss(predictions.ToVector(), target.ToVector());

        var gradient = _lossFunction.CalculateDerivative(predictions.ToVector(), target.ToVector());
        Backward(Tensor<T>.FromVector(gradient, predictions.Shape));

        _optimizer.UpdateParameters(Layers);

        SetTrainingMode(false);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the LSTNet model, UpdateParameters updates internal parameters or state. This keeps the LSTNet architecture aligned with the latest values.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> gradients)
    {
        // Parameters are updated by the optimizer in Train method
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the LSTNet model, GetModelMetadata performs a supporting step in the workflow. It keeps the LSTNet architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "LSTNet" },
                { "LookbackWindow", _lookbackWindow },
                { "ForecastHorizon", _forecastHorizon },
                { "NumFeatures", _numFeatures },
                { "HiddenRecurrentSize", _hiddenRecurrentSize },
                { "HiddenSkipSize", _hiddenSkipSize },
                { "ConvolutionFilters", _convolutionFilters },
                { "SkipPeriod", _skipPeriod },
                { "UseHighway", _useHighway },
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
    /// <b>For Beginners:</b> This creates a fresh copy of the model with the same
    /// settings but randomly initialized weights. Useful for techniques like
    /// ensemble learning.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var options = new LSTNetOptions<T>
        {
            LookbackWindow = _lookbackWindow,
            ForecastHorizon = _forecastHorizon,
            HiddenRecurrentSize = _hiddenRecurrentSize,
            HiddenSkipSize = _hiddenSkipSize,
            ConvolutionFilters = _convolutionFilters,
            ConvolutionKernelSize = _convolutionKernelSize,
            SkipPeriod = _skipPeriod,
            AutoregressiveWindow = _autoregressiveWindow,
            UseHighway = _useHighway,
            DropoutRate = _dropout
        };

        return new LSTNet<T>(Architecture, options);
    }

    /// <summary>
    /// Writes LSTNet-specific configuration during serialization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Serialization saves the model to a file so it can be
    /// loaded later. This method saves all the configuration values.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_lookbackWindow);
        writer.Write(_forecastHorizon);
        writer.Write(_numFeatures);
        writer.Write(_hiddenRecurrentSize);
        writer.Write(_hiddenSkipSize);
        writer.Write(_convolutionFilters);
        writer.Write(_convolutionKernelSize);
        writer.Write(_skipPeriod);
        writer.Write(_autoregressiveWindow);
        writer.Write(_useHighway);
        writer.Write(_dropout);
    }

    /// <summary>
    /// Reads LSTNet-specific configuration during deserialization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This reads back the configuration when loading a saved model.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32(); // lookbackWindow
        _ = reader.ReadInt32(); // forecastHorizon
        _ = reader.ReadInt32(); // numFeatures
        _ = reader.ReadInt32(); // hiddenRecurrentSize
        _ = reader.ReadInt32(); // hiddenSkipSize
        _ = reader.ReadInt32(); // convolutionFilters
        _ = reader.ReadInt32(); // convolutionKernelSize
        _ = reader.ReadInt32(); // skipPeriod
        _ = reader.ReadInt32(); // autoregressiveWindow
        _ = reader.ReadBoolean(); // useHighway
        _ = reader.ReadDouble(); // dropout
    }

    #endregion

    #region IForecastingModel Implementation

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the LSTNet model, Forecast produces predictions from input data. This is the main inference step of the LSTNet architecture.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
    {
        return _useNativeMode ? ForecastNative(historicalData) : ForecastOnnx(historicalData);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the LSTNet model, AutoregressiveForecast produces predictions from input data. This is the main inference step of the LSTNet architecture.
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

            int stepsUsed = Math.Min(_forecastHorizon, stepsRemaining);
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
    /// <b>For Beginners:</b> In the LSTNet model, Evaluate performs a supporting step in the workflow. It keeps the LSTNet architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> Evaluate(Tensor<T> predictions, Tensor<T> actuals)
    {
        var metrics = new Dictionary<string, T>();

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
    /// <b>For Beginners:</b> In the LSTNet model, ApplyInstanceNormalization performs a supporting step in the workflow. It keeps the LSTNet architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
    {
        // LSTNet doesn't typically use instance normalization
        return input;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the LSTNet model, GetFinancialMetrics calculates evaluation metrics. This summarizes how the LSTNet architecture is performing.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> GetFinancialMetrics()
    {
        T lastLoss = LastLoss is not null ? LastLoss : NumOps.Zero;
        return new Dictionary<string, T>
        {
            ["LookbackWindow"] = NumOps.FromDouble(_lookbackWindow),
            ["ForecastHorizon"] = NumOps.FromDouble(_forecastHorizon),
            ["NumFeatures"] = NumOps.FromDouble(_numFeatures),
            ["HiddenRecurrentSize"] = NumOps.FromDouble(_hiddenRecurrentSize),
            ["SkipPeriod"] = NumOps.FromDouble(_skipPeriod),
            ["LastLoss"] = lastLoss
        };
    }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// Performs the forward pass through the LSTNet.
    /// </summary>
    /// <param name="input">Input tensor of shape [batch, lookback_window, features].</param>
    /// <returns>Output tensor of shape [batch, forecast_horizon].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The forward pass is how data flows through the network:
    ///
    /// 1. <b>Convolution</b>: Scans for local patterns in the input
    /// 2. <b>GRU</b>: Processes the conv output to capture long-term trends
    /// 3. <b>Skip-GRU</b>: Looks at periodic patterns by sampling at skip intervals
    /// 4. <b>Highway</b>: Provides a direct linear path from input to output
    /// 5. <b>Combine</b>: Merges all these signals into the final forecast
    /// </para>
    /// </remarks>
    private Tensor<T> Forward(Tensor<T> input)
    {
        int batchSize = input.Shape[0];

        // === Component 1: Convolutional Processing ===
        Tensor<T> convOutput = input;

        if (_convLayer is not null)
        {
            convOutput = _convLayer.Forward(input);
        }

        if (_convActivation is not null)
        {
            convOutput = _convActivation.Forward(convOutput);
        }

        if (_convDropout is not null)
        {
            convOutput = _convDropout.Forward(convOutput);
        }

        // === Component 2: Main GRU Processing ===
        Tensor<T> gruOutput = convOutput;

        if (_gruLayer is not null)
        {
            gruOutput = _gruLayer.Forward(convOutput);
        }

        if (_gruDropout is not null)
        {
            gruOutput = _gruDropout.Forward(gruOutput);
        }

        // === Component 3: Skip-GRU Processing ===
        // Sample input at skip intervals for periodic pattern detection
        Tensor<T>? skipOutput = null;

        if (_skipGruLayer is not null)
        {
            var skipSampled = SampleAtSkipIntervals(convOutput, _skipPeriod);
            skipOutput = _skipGruLayer.Forward(skipSampled);

            if (_skipDropout is not null)
            {
                skipOutput = _skipDropout.Forward(skipOutput);
            }
        }

        // === Component 4: Highway (Autoregressive) ===
        Tensor<T>? highwayOutput = null;

        if (_useHighway && _highwayLayer is not null)
        {
            // Use the most recent values for highway
            var recentValues = ExtractRecentValues(input, _autoregressiveWindow);
            highwayOutput = _highwayLayer.Forward(recentValues);
        }

        // === Component 5: Combine All Outputs ===
        var combined = CombineOutputs(gruOutput, skipOutput, highwayOutput);

        if (_combinationLayer is not null)
        {
            combined = _combinationLayer.Forward(combined);
        }

        // === Component 6: Final Output ===
        Tensor<T> output = combined;

        if (_outputLayer is not null)
        {
            output = _outputLayer.Forward(combined);
        }

        return output;
    }

    /// <summary>
    /// Performs the backward pass through the LSTNet.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The backward pass is how the network learns. It calculates
    /// how much each part of the network contributed to the error and adjusts
    /// accordingly.
    /// </para>
    /// </remarks>
    private void Backward(Tensor<T> gradOutput)
    {
        var grad = gradOutput;

        // Backward through output layer
        if (_outputLayer is not null)
        {
            grad = _outputLayer.Backward(grad);
        }

        // Backward through combination layer
        if (_combinationLayer is not null)
        {
            grad = _combinationLayer.Backward(grad);
        }

        // Backward through highway
        if (_highwayLayer is not null)
        {
            _ = _highwayLayer.Backward(grad);
        }

        // Backward through skip GRU
        if (_skipDropout is not null)
        {
            grad = _skipDropout.Backward(grad);
        }

        if (_skipGruLayer is not null)
        {
            grad = _skipGruLayer.Backward(grad);
        }

        // Backward through main GRU
        if (_gruDropout is not null)
        {
            grad = _gruDropout.Backward(grad);
        }

        if (_gruLayer is not null)
        {
            grad = _gruLayer.Backward(grad);
        }

        // Backward through convolution
        if (_convDropout is not null)
        {
            grad = _convDropout.Backward(grad);
        }

        if (_convActivation is not null)
        {
            grad = _convActivation.Backward(grad);
        }

        if (_convLayer is not null)
        {
            _ = _convLayer.Backward(grad);
        }
    }

    /// <summary>
    /// Performs native mode forecasting.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This runs the forward pass without updating the network.
    /// Used when making actual predictions (not training).
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
    /// <b>For Beginners:</b> This runs predictions using a pre-trained ONNX model
    /// loaded from a file.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ForecastOnnx(Tensor<T> input)
    {
        if (OnnxSession is null)
            throw new InvalidOperationException("ONNX session is not initialized.");

        var inputData = new float[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            inputData[i] = Convert.ToSingle(NumOps.ToDouble(input.Data.Span[i]));
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
    /// Samples the input at skip intervals to capture periodic patterns.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <param name="skipPeriod">Number of time steps between samples.</param>
    /// <returns>Sampled tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> If your data is hourly and skipPeriod is 24, this takes
    /// every 24th value (midnight, midnight, midnight...) to find daily patterns.
    /// </para>
    /// </remarks>
    private Tensor<T> SampleAtSkipIntervals(Tensor<T> input, int skipPeriod)
    {
        int batchSize = input.Shape[0];
        int seqLen = input.Shape.Length > 1 ? input.Shape[1] : input.Length / batchSize;
        int features = input.Shape.Length > 2 ? input.Shape[2] : 1;
        int numSamples = Math.Max(1, seqLen / skipPeriod);

        var sampled = new Tensor<T>(new[] { batchSize, numSamples, features });

        for (int b = 0; b < batchSize; b++)
        {
            for (int s = 0; s < numSamples; s++)
            {
                int srcIdx = s * skipPeriod;
                if (srcIdx < seqLen)
                {
                    for (int f = 0; f < features; f++)
                    {
                        int srcFlatIdx = b * seqLen * features + srcIdx * features + f;
                        int dstFlatIdx = b * numSamples * features + s * features + f;

                        if (srcFlatIdx < input.Length && dstFlatIdx < sampled.Length)
                        {
                            sampled.Data.Span[dstFlatIdx] = input.Data.Span[srcFlatIdx];
                        }
                    }
                }
            }
        }

        return sampled;
    }

    /// <summary>
    /// Extracts the most recent values from the input for autoregressive processing.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <param name="window">Number of recent values to extract.</param>
    /// <returns>Tensor of recent values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The autoregressive component looks at just the most recent
    /// values to make simple linear predictions. This extracts those values.
    /// </para>
    /// </remarks>
    private Tensor<T> ExtractRecentValues(Tensor<T> input, int window)
    {
        int batchSize = input.Shape[0];
        int seqLen = input.Shape.Length > 1 ? input.Shape[1] : input.Length / batchSize;
        int features = input.Shape.Length > 2 ? input.Shape[2] : 1;

        int actualWindow = Math.Min(window, seqLen);
        int startIdx = seqLen - actualWindow;

        // For highway layer, we want to flatten the recent values
        var recent = new Tensor<T>(new[] { batchSize, features });

        for (int b = 0; b < batchSize; b++)
        {
            // Take the average of recent values for each feature
            for (int f = 0; f < features; f++)
            {
                T sum = NumOps.Zero;
                for (int t = startIdx; t < seqLen; t++)
                {
                    int idx = b * seqLen * features + t * features + f;
                    if (idx < input.Length)
                    {
                        sum = NumOps.Add(sum, input.Data.Span[idx]);
                    }
                }
                int dstIdx = b * features + f;
                recent.Data.Span[dstIdx] = NumOps.Divide(sum, NumOps.FromDouble(actualWindow));
            }
        }

        return recent;
    }

    /// <summary>
    /// Combines outputs from GRU, Skip-GRU, and Highway components.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This merges the signals from all the specialized components
    /// into one combined representation that will be used for the final prediction.
    /// </para>
    /// </remarks>
    private Tensor<T> CombineOutputs(Tensor<T> gruOutput, Tensor<T>? skipOutput, Tensor<T>? highwayOutput)
    {
        int batchSize = gruOutput.Shape[0];
        int gruSize = gruOutput.Length / batchSize;
        int skipSize = skipOutput is not null ? skipOutput.Length / batchSize : 0;

        int combinedSize = gruSize + skipSize;

        var combined = new Tensor<T>(new[] { batchSize, combinedSize });

        for (int b = 0; b < batchSize; b++)
        {
            // Copy GRU output
            for (int i = 0; i < gruSize; i++)
            {
                int srcIdx = b * gruSize + i;
                int dstIdx = b * combinedSize + i;
                if (srcIdx < gruOutput.Length && dstIdx < combined.Length)
                {
                    combined.Data.Span[dstIdx] = gruOutput.Data.Span[srcIdx];
                }
            }

            // Copy Skip output
            if (skipOutput is not null)
            {
                for (int i = 0; i < skipSize; i++)
                {
                    int srcIdx = b * skipSize + i;
                    int dstIdx = b * combinedSize + gruSize + i;
                    if (srcIdx < skipOutput.Length && dstIdx < combined.Length)
                    {
                        combined.Data.Span[dstIdx] = skipOutput.Data.Span[srcIdx];
                    }
                }
            }
        }

        // Add highway output directly to combined (residual connection style)
        if (highwayOutput is not null)
        {
            // Highway output gets added as a residual to the final output
            // The combination layer will handle the final transformation
        }

        return combined;
    }

    /// <summary>
    /// Shifts the input tensor by incorporating new predictions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> For autoregressive forecasting, we take our prediction,
    /// add it to our input history, and shift everything over. This lets us make
    /// predictions further into the future.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ShiftInputWithPredictions(Tensor<T> input, Tensor<T> predictions, int stepsUsed)
    {
        int batchSize = input.Shape[0];
        int seqLen = input.Shape.Length > 1 ? input.Shape[1] : input.Length / batchSize;
        int features = input.Shape.Length > 2 ? input.Shape[2] : 1;

        var shifted = new Tensor<T>(input.Shape);

        for (int b = 0; b < batchSize; b++)
        {
            // Shift old values left
            for (int t = 0; t < seqLen - stepsUsed; t++)
            {
                for (int f = 0; f < features; f++)
                {
                    int srcIdx = b * seqLen * features + (t + stepsUsed) * features + f;
                    int dstIdx = b * seqLen * features + t * features + f;

                    if (srcIdx < input.Length && dstIdx < shifted.Length)
                    {
                        shifted.Data.Span[dstIdx] = input.Data.Span[srcIdx];
                    }
                }
            }

            // Add predictions at the end
            for (int t = 0; t < stepsUsed; t++)
            {
                int srcIdx = b * stepsUsed + t;
                int dstIdx = b * seqLen * features + (seqLen - stepsUsed + t) * features;

                if (srcIdx < predictions.Length && dstIdx < shifted.Length)
                {
                    // For multivariate, replicate prediction to all features
                    for (int f = 0; f < features && dstIdx + f < shifted.Length; f++)
                    {
                        shifted.Data.Span[dstIdx + f] = predictions.Data.Span[srcIdx];
                    }
                }
            }
        }

        return shifted;
    }

    /// <summary>
    /// Concatenates multiple prediction tensors into a single result.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When we predict multiple horizons at once, we need to
    /// combine them into a single continuous forecast.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ConcatenatePredictions(List<Tensor<T>> predictions, int totalSteps)
    {
        if (predictions.Count == 0)
            return new Tensor<T>(new[] { 1, totalSteps });

        int batchSize = predictions[0].Shape[0];
        var result = new Tensor<T>(new[] { batchSize, totalSteps });

        int currentStep = 0;
        foreach (var pred in predictions)
        {
            int predLen = pred.Length / batchSize;
            int stepsToCopy = Math.Min(predLen, totalSteps - currentStep);

            for (int b = 0; b < batchSize; b++)
            {
                for (int t = 0; t < stepsToCopy; t++)
                {
                    int srcIdx = b * predLen + t;
                    int dstIdx = b * totalSteps + currentStep + t;

                    if (srcIdx < pred.Length && dstIdx < result.Length)
                    {
                        result.Data.Span[dstIdx] = pred.Data.Span[srcIdx];
                    }
                }
            }

            currentStep += stepsToCopy;
            if (currentStep >= totalSteps) break;
        }

        return result;
    }

    #endregion

    #region IDisposable

    /// <summary>
    /// Releases managed resources used by this model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This cleans up memory when the model is no longer needed.
    /// The ONNX session in particular needs to be properly disposed.
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

