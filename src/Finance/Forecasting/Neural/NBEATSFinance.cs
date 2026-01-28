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

namespace AiDotNet.Finance.Forecasting.Neural;

/// <summary>
/// N-BEATS (Neural Basis Expansion Analysis for Time Series) model for financial forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// N-BEATS is a deep neural architecture that uses basis expansion to decompose time series
/// into interpretable components. It achieves state-of-the-art performance while providing
/// the ability to decompose forecasts into trend and seasonality components.
/// </para>
/// <para>
/// <b>For Beginners:</b> N-BEATS works by stacking multiple "blocks" that each try to explain
/// part of the time series. Each block:
/// - Looks at the input
/// - Produces a "backcast" (its explanation of the past)
/// - Produces a "forecast" (its prediction for the future)
/// - Passes the "residual" (what it couldn't explain) to the next block
///
/// This hierarchical approach allows N-BEATS to decompose complex patterns into simpler components,
/// similar to how you might break down a sound wave into different frequency components.
///
/// Key benefits:
/// - <b>Interpretable:</b> Can separate trend from seasonality
/// - <b>No feature engineering:</b> Works directly on raw time series
/// - <b>State-of-the-art accuracy:</b> Competitive with or better than traditional methods
/// </para>
/// <para>
/// <b>Reference:</b> Oreshkin et al., "N-BEATS: Neural basis expansion analysis for
/// interpretable time series forecasting", ICLR 2020. https://arxiv.org/abs/1905.10437
/// </para>
/// </remarks>
public class NBEATSFinance<T> : NeuralNetworkBase<T>, IForecastingModel<T>
{
    #region Execution Mode

    /// <summary>
    /// Indicates whether this network uses native layers (true) or ONNX model (false).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Native mode allows training and full control,
    /// while ONNX mode uses pre-trained models for fast inference.
    /// </para>
    /// </remarks>
    private readonly bool _useNativeMode;

    #endregion

    #region ONNX Mode Fields

    /// <summary>
    /// The ONNX inference session for running pretrained models.
    /// </summary>
    private readonly InferenceSession? _onnxSession;

    /// <summary>
    /// Path to the ONNX model file.
    /// </summary>
    private readonly string? _onnxModelPath;

    #endregion

    #region Native Mode Fields

    /// <summary>
    /// Groups of layers representing each N-BEATS block.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Each block contains multiple fully connected layers
    /// followed by backcast and forecast output layers.
    /// </para>
    /// </remarks>
    private readonly List<List<ILayer<T>>> _blocks = [];

    /// <summary>
    /// Final output projection layer.
    /// </summary>
    private ILayer<T>? _outputProjection;

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
    /// The lookback window size.
    /// </summary>
    private readonly int _lookbackWindow;

    /// <summary>
    /// The forecast horizon.
    /// </summary>
    private readonly int _forecastHorizon;

    /// <summary>
    /// Number of stacks.
    /// </summary>
    private readonly int _numStacks;

    /// <summary>
    /// Number of blocks per stack.
    /// </summary>
    private readonly int _numBlocksPerStack;

    /// <summary>
    /// Size of hidden layers.
    /// </summary>
    private readonly int _hiddenSize;

    /// <summary>
    /// Number of hidden layers per block.
    /// </summary>
    private readonly int _numHiddenLayers;

    /// <summary>
    /// Polynomial degree for trend basis.
    /// </summary>
    private readonly int _polynomialDegree;

    /// <summary>
    /// Whether to use interpretable basis functions.
    /// </summary>
    private readonly bool _useInterpretableBasis;

    /// <summary>
    /// Whether to share weights within stacks.
    /// </summary>
    private readonly bool _shareWeightsInStack;

    #endregion

    #region IForecastingModel Properties

    /// <inheritdoc/>
    public int SequenceLength => _lookbackWindow;

    /// <inheritdoc/>
    public int PredictionHorizon => _forecastHorizon;

    /// <inheritdoc/>
    public int NumFeatures => 1; // N-BEATS works on univariate series

    /// <inheritdoc/>
    public int PatchSize => 1; // N-BEATS doesn't use patching

    /// <inheritdoc/>
    public int Stride => 1;

    /// <inheritdoc/>
    public bool IsChannelIndependent => true; // N-BEATS processes each channel independently

    /// <inheritdoc/>
    public bool UseNativeMode => _useNativeMode;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates an N-BEATS network using pretrained ONNX model.
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
    public NBEATSFinance(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        NBEATSModelOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        options ??= new NBEATSModelOptions<T>();
        ValidateOptions(options);

        _useNativeMode = false;
        _onnxSession = new InferenceSession(onnxModelPath);
        _onnxModelPath = onnxModelPath;

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        _lookbackWindow = options.LookbackWindow;
        _forecastHorizon = options.ForecastHorizon;
        _numStacks = options.NumStacks;
        _numBlocksPerStack = options.NumBlocksPerStack;
        _hiddenSize = options.HiddenLayerSize;
        _numHiddenLayers = options.NumHiddenLayers;
        _polynomialDegree = options.PolynomialDegree;
        _useInterpretableBasis = options.UseInterpretableBasis;
        _shareWeightsInStack = options.ShareWeightsInStack;
    }

    /// <summary>
    /// Creates an N-BEATS network in native mode for training from scratch.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor to train a new N-BEATS model from scratch.
    /// N-BEATS excels at:
    /// - Univariate time series forecasting
    /// - Decomposing series into trend and seasonality
    /// - Achieving high accuracy without feature engineering
    /// </para>
    /// </remarks>
    public NBEATSFinance(
        NeuralNetworkArchitecture<T> architecture,
        NBEATSModelOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new NBEATSModelOptions<T>();
        ValidateOptions(options);

        _useNativeMode = true;
        _onnxSession = null;
        _onnxModelPath = null;

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        _lookbackWindow = options.LookbackWindow;
        _forecastHorizon = options.ForecastHorizon;
        _numStacks = options.NumStacks;
        _numBlocksPerStack = options.NumBlocksPerStack;
        _hiddenSize = options.HiddenLayerSize;
        _numHiddenLayers = options.NumHiddenLayers;
        _polynomialDegree = options.PolynomialDegree;
        _useInterpretableBasis = options.UseInterpretableBasis;
        _shareWeightsInStack = options.ShareWeightsInStack;

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes the neural network layers for N-BEATS.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> N-BEATS has a hierarchical structure:
    /// </para>
    /// <para>
    /// <list type="number">
    /// <item><b>Stacks:</b> Groups of blocks, each focusing on different patterns</item>
    /// <item><b>Blocks:</b> Each produces backcast (past) and forecast (future)</item>
    /// <item><b>Residual connections:</b> Pass unexplained signal to next block</item>
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultNBEATSLayers(
                Architecture, _lookbackWindow, _forecastHorizon, _numStacks,
                _numBlocksPerStack, _hiddenSize, _numHiddenLayers));

            ExtractLayerReferences();
        }
    }

    /// <summary>
    /// Extracts references to specific layers from the layer collection.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This organizes the layers into logical blocks for easier
    /// access during the forward pass.
    /// </para>
    /// </remarks>
    private void ExtractLayerReferences()
    {
        int idx = 0;
        int layersPerBlock = _numHiddenLayers + 2; // hidden layers + backcast + forecast

        _blocks.Clear();

        for (int stack = 0; stack < _numStacks; stack++)
        {
            for (int block = 0; block < _numBlocksPerStack; block++)
            {
                var blockLayers = new List<ILayer<T>>();
                for (int i = 0; i < layersPerBlock && idx < Layers.Count - 1; i++)
                {
                    blockLayers.Add(Layers[idx++]);
                }
                _blocks.Add(blockLayers);
            }
        }

        // Final output projection
        if (idx < Layers.Count)
            _outputProjection = Layers[idx];
    }

    /// <summary>
    /// Validates that custom layers meet N-BEATS architectural requirements.
    /// </summary>
    /// <param name="layers">The list of custom layers to validate.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> N-BEATS requires at least one block with hidden layers
    /// and backcast/forecast outputs.
    /// </para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);
        if (layers.Count < 3)
        {
            throw new ArgumentException(
                "N-BEATS requires at least 3 layers: hidden, backcast, and forecast.",
                nameof(layers));
        }
    }

    /// <summary>
    /// Validates the N-BEATS options.
    /// </summary>
    /// <param name="options">The options to validate.</param>
    /// <exception cref="ArgumentException">Thrown when options are invalid.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This ensures all configuration values are reasonable
    /// before attempting to build the model.
    /// </para>
    /// </remarks>
    private static void ValidateOptions(NBEATSModelOptions<T> options)
    {
        var errors = new List<string>();

        if (options.LookbackWindow < 1)
            errors.Add("LookbackWindow must be at least 1.");
        if (options.ForecastHorizon < 1)
            errors.Add("ForecastHorizon must be at least 1.");
        if (options.NumStacks < 1)
            errors.Add("NumStacks must be at least 1.");
        if (options.NumBlocksPerStack < 1)
            errors.Add("NumBlocksPerStack must be at least 1.");
        if (options.HiddenLayerSize < 1)
            errors.Add("HiddenLayerSize must be at least 1.");
        if (options.NumHiddenLayers < 1)
            errors.Add("NumHiddenLayers must be at least 1.");

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
    /// <b>For Beginners:</b> In the NBEATSFinance model, Predict produces predictions from input data. This is the main inference step of the NBEATSFinance architecture.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return _useNativeMode ? ForecastNative(input) : ForecastOnnx(input);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the NBEATSFinance model, Train performs a training step. This updates the NBEATSFinance architecture so it learns from data.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is only supported in native mode.");

        SetTrainingMode(true);

        // Forward pass
        var predictions = Forward(input);

        // Compute loss
        LastLoss = _lossFunction.CalculateLoss(predictions.ToVector(), target.ToVector());

        // Backward pass
        var gradient = _lossFunction.CalculateDerivative(predictions.ToVector(), target.ToVector());
        Backward(Tensor<T>.FromVector(gradient));

        // Update weights via optimizer
        _optimizer.UpdateParameters(Layers);

        SetTrainingMode(false);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the NBEATSFinance model, UpdateParameters updates internal parameters or state. This keeps the NBEATSFinance architecture aligned with the latest values.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> gradients)
    {
        // Parameters are updated through the optimizer in Train method
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the NBEATSFinance model, GetModelMetadata performs a supporting step in the workflow. It keeps the NBEATSFinance architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "N-BEATS" },
                { "LookbackWindow", _lookbackWindow },
                { "ForecastHorizon", _forecastHorizon },
                { "NumStacks", _numStacks },
                { "NumBlocksPerStack", _numBlocksPerStack },
                { "HiddenSize", _hiddenSize },
                { "UseInterpretableBasis", _useInterpretableBasis },
                { "UseNativeMode", _useNativeMode },
                { "ParameterCount", GetParameterCount() }
            },
            ModelData = _useNativeMode ? this.Serialize() : Array.Empty<byte>()
        };
    }

    /// <summary>
    /// Creates a new instance of this model with the same configuration.
    /// </summary>
    /// <returns>A new N-BEATS model instance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This creates a fresh copy of the model with the same settings
    /// but new (randomly initialized) weights.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var options = new NBEATSModelOptions<T>
        {
            LookbackWindow = _lookbackWindow,
            ForecastHorizon = _forecastHorizon,
            NumStacks = _numStacks,
            NumBlocksPerStack = _numBlocksPerStack,
            HiddenLayerSize = _hiddenSize,
            NumHiddenLayers = _numHiddenLayers,
            PolynomialDegree = _polynomialDegree,
            UseInterpretableBasis = _useInterpretableBasis,
            ShareWeightsInStack = _shareWeightsInStack
        };

        return new NBEATSFinance<T>(Architecture, options);
    }

    /// <summary>
    /// Writes N-BEATS-specific configuration during serialization.
    /// </summary>
    /// <param name="writer">Binary writer for output.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This saves N-BEATS settings to a file so the model
    /// can be loaded later with the same configuration.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_lookbackWindow);
        writer.Write(_forecastHorizon);
        writer.Write(_numStacks);
        writer.Write(_numBlocksPerStack);
        writer.Write(_hiddenSize);
        writer.Write(_numHiddenLayers);
        writer.Write(_polynomialDegree);
        writer.Write(_useInterpretableBasis);
        writer.Write(_shareWeightsInStack);
    }

    /// <summary>
    /// Reads N-BEATS-specific configuration during deserialization.
    /// </summary>
    /// <param name="reader">Binary reader for input.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This reads back N-BEATS settings when loading a saved model.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32();   // lookbackWindow
        _ = reader.ReadInt32();   // forecastHorizon
        _ = reader.ReadInt32();   // numStacks
        _ = reader.ReadInt32();   // numBlocksPerStack
        _ = reader.ReadInt32();   // hiddenSize
        _ = reader.ReadInt32();   // numHiddenLayers
        _ = reader.ReadInt32();   // polynomialDegree
        _ = reader.ReadBoolean(); // useInterpretableBasis
        _ = reader.ReadBoolean(); // shareWeightsInStack
    }

    #endregion

    #region IForecastingModel Implementation

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the NBEATSFinance model, Forecast produces predictions from input data. This is the main inference step of the NBEATSFinance architecture.
    /// </para>
    /// </remarks>
    public Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
    {
        return _useNativeMode ? ForecastNative(historicalData) : ForecastOnnx(historicalData);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the NBEATSFinance model, AutoregressiveForecast produces predictions from input data. This is the main inference step of the NBEATSFinance architecture.
    /// </para>
    /// </remarks>
    public Tensor<T> AutoregressiveForecast(Tensor<T> input, int steps)
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
    /// <b>For Beginners:</b> In the NBEATSFinance model, Evaluate performs a supporting step in the workflow. It keeps the NBEATSFinance architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public Dictionary<string, T> Evaluate(Tensor<T> predictions, Tensor<T> actuals)
    {
        var metrics = new Dictionary<string, T>();

        T mse = NumOps.Zero;
        T mae = NumOps.Zero;
        T smape = NumOps.Zero;
        int count = 0;

        for (int i = 0; i < predictions.Length && i < actuals.Length; i++)
        {
            var diff = NumOps.Subtract(predictions[i], actuals[i]);
            mse = NumOps.Add(mse, NumOps.Multiply(diff, diff));
            mae = NumOps.Add(mae, NumOps.Abs(diff));

            // sMAPE calculation
            var absSum = NumOps.Add(NumOps.Abs(predictions[i]), NumOps.Abs(actuals[i]));
            if (NumOps.Compare(absSum, NumOps.Zero) > 0)
            {
                var smapeContrib = NumOps.Divide(NumOps.Abs(diff), absSum);
                smape = NumOps.Add(smape, smapeContrib);
            }
            count++;
        }

        if (count > 0)
        {
            mse = NumOps.Divide(mse, NumOps.FromDouble(count));
            mae = NumOps.Divide(mae, NumOps.FromDouble(count));
            smape = NumOps.Multiply(NumOps.Divide(smape, NumOps.FromDouble(count)), NumOps.FromDouble(200)); // Convert to percentage
        }

        metrics["MSE"] = mse;
        metrics["MAE"] = mae;
        metrics["RMSE"] = NumOps.Sqrt(mse);
        metrics["sMAPE"] = smape;

        return metrics;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the NBEATSFinance model, ApplyInstanceNormalization performs a supporting step in the workflow. It keeps the NBEATSFinance architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
    {
        // N-BEATS typically doesn't use instance normalization
        return input;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the NBEATSFinance model, GetFinancialMetrics calculates evaluation metrics. This summarizes how the NBEATSFinance architecture is performing.
    /// </para>
    /// </remarks>
    public Dictionary<string, T> GetFinancialMetrics()
    {
        T lastLoss = LastLoss is not null ? LastLoss : NumOps.Zero;
        return new Dictionary<string, T>
        {
            ["LookbackWindow"] = NumOps.FromDouble(_lookbackWindow),
            ["ForecastHorizon"] = NumOps.FromDouble(_forecastHorizon),
            ["NumStacks"] = NumOps.FromDouble(_numStacks),
            ["NumBlocksPerStack"] = NumOps.FromDouble(_numBlocksPerStack),
            ["HiddenSize"] = NumOps.FromDouble(_hiddenSize),
            ["LastLoss"] = lastLoss
        };
    }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// Performs the forward pass through the N-BEATS network.
    /// </summary>
    /// <param name="input">Input tensor of shape [batch, lookback_window].</param>
    /// <returns>Output tensor of shape [batch, forecast_horizon].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The N-BEATS forward pass is unique:
    /// 1. Each block processes the residual from the previous block
    /// 2. Each block outputs a backcast (explanation of past) and forecast (prediction)
    /// 3. The residual = input - backcast is passed to the next block
    /// 4. All forecasts are summed to get the final prediction
    /// </para>
    /// </remarks>
    private Tensor<T> Forward(Tensor<T> input)
    {
        // Initialize residual as input
        var residual = input;
        var totalForecast = new Tensor<T>(new[] { input.Shape[0], _forecastHorizon });

        // Process each block
        foreach (var blockLayers in _blocks)
        {
            if (blockLayers.Count == 0) continue;

            // Forward through hidden layers
            var current = residual;
            int numHidden = blockLayers.Count - 2; // Exclude backcast and forecast layers

            for (int i = 0; i < numHidden && i < blockLayers.Count; i++)
            {
                current = blockLayers[i].Forward(current);
            }

            // Get backcast
            Tensor<T>? backcast = null;
            if (numHidden < blockLayers.Count)
            {
                backcast = blockLayers[numHidden].Forward(current);
            }

            // Get forecast
            Tensor<T>? forecast = null;
            if (numHidden + 1 < blockLayers.Count)
            {
                forecast = blockLayers[numHidden + 1].Forward(current);
            }

            // Update residual (subtract backcast)
            if (backcast is not null)
            {
                residual = SubtractTensors(residual, backcast);
            }

            // Accumulate forecast
            if (forecast is not null)
            {
                totalForecast = AddTensors(totalForecast, forecast);
            }
        }

        // Final output projection
        if (_outputProjection is not null)
        {
            totalForecast = _outputProjection.Forward(totalForecast);
        }

        return totalForecast;
    }

    /// <summary>
    /// Performs the backward pass through the N-BEATS network.
    /// </summary>
    /// <param name="gradOutput">Gradient from the loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Backward pass computes gradients for all learnable
    /// parameters by propagating error signals backwards through each block.
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

        // Backward through blocks in reverse order
        for (int i = _blocks.Count - 1; i >= 0; i--)
        {
            var blockLayers = _blocks[i];
            for (int j = blockLayers.Count - 1; j >= 0; j--)
            {
                grad = blockLayers[j].Backward(grad);
            }
        }
    }

    /// <summary>
    /// Performs native mode forecasting.
    /// </summary>
    /// <param name="input">Input historical data.</param>
    /// <returns>Forecasted values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the NBEATSFinance model, ForecastNative produces predictions from input data. This is the main inference step of the NBEATSFinance architecture.
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
    /// <b>For Beginners:</b> In the NBEATSFinance model, ForecastOnnx produces predictions from input data. This is the main inference step of the NBEATSFinance architecture.
    /// </para>
    /// </remarks>
    private Tensor<T> ForecastOnnx(Tensor<T> input)
    {
        if (_onnxSession is null)
            throw new InvalidOperationException("ONNX session is not initialized.");

        var inputData = new float[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            inputData[i] = Convert.ToSingle(NumOps.ToDouble(input.Data.Span[i]));
        }

        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input.Shape);
        var inputMeta = _onnxSession.InputMetadata;
        string inputName = inputMeta.Keys.First();

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, onnxInput)
        };

        using var results = _onnxSession.Run(inputs);
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
    /// Subtracts two tensors element-wise.
    /// </summary>
    /// <param name="a">First tensor.</param>
    /// <param name="b">Second tensor (subtracted from first).</param>
    /// <returns>Result tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This computes a - b for each element.
    /// Used to compute residuals in the N-BEATS architecture.
    /// </para>
    /// </remarks>
    private Tensor<T> SubtractTensors(Tensor<T> a, Tensor<T> b)
    {
        var result = new Tensor<T>(a.Shape);
        int length = Math.Min(a.Length, b.Length);
        for (int i = 0; i < length; i++)
        {
            result.Data.Span[i] = NumOps.Subtract(a.Data.Span[i], b.Data.Span[i]);
        }
        return result;
    }

    /// <summary>
    /// Adds two tensors element-wise.
    /// </summary>
    /// <param name="a">First tensor.</param>
    /// <param name="b">Second tensor.</param>
    /// <returns>Result tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This computes a + b for each element.
    /// Used to accumulate forecasts from all blocks.
    /// </para>
    /// </remarks>
    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        var result = new Tensor<T>(a.Shape);
        int length = Math.Min(a.Length, b.Length);
        for (int i = 0; i < length; i++)
        {
            result.Data.Span[i] = NumOps.Add(a.Data.Span[i], b.Data.Span[i]);
        }
        return result;
    }

    /// <summary>
    /// Shifts input by incorporating recent predictions.
    /// </summary>
    /// <param name="input">Current input tensor.</param>
    /// <param name="prediction">Recent prediction tensor.</param>
    /// <param name="stepsUsed">Number of prediction steps to incorporate.</param>
    /// <returns>Shifted input tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the NBEATSFinance model, ShiftInputWithPredictions produces predictions from input data. This is the main inference step of the NBEATSFinance architecture.
    /// </para>
    /// </remarks>
    private Tensor<T> ShiftInputWithPredictions(Tensor<T> input, Tensor<T> prediction, int stepsUsed)
    {
        int batchSize = input.Shape[0];
        int seqLen = input.Shape.Length > 1 ? input.Shape[1] : input.Length / batchSize;

        var shifted = new Tensor<T>(input.Shape);

        for (int b = 0; b < batchSize; b++)
        {
            // Shift old values
            for (int t = 0; t < seqLen - stepsUsed; t++)
            {
                int srcIdx = b * seqLen + t + stepsUsed;
                int dstIdx = b * seqLen + t;
                if (srcIdx < input.Length && dstIdx < shifted.Length)
                    shifted.Data.Span[dstIdx] = input.Data.Span[srcIdx];
            }

            // Add new predictions
            for (int t = seqLen - stepsUsed; t < seqLen; t++)
            {
                int predIdx = b * stepsUsed + (t - (seqLen - stepsUsed));
                int dstIdx = b * seqLen + t;
                if (predIdx < prediction.Length && dstIdx < shifted.Length)
                    shifted.Data.Span[dstIdx] = prediction.Data.Span[predIdx];
            }
        }

        return shifted;
    }

    /// <summary>
    /// Concatenates multiple predictions into a single tensor.
    /// </summary>
    /// <param name="predictions">List of prediction tensors.</param>
    /// <param name="totalSteps">Total number of steps to include.</param>
    /// <returns>Concatenated prediction tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the NBEATSFinance model, ConcatenatePredictions produces predictions from input data. This is the main inference step of the NBEATSFinance architecture.
    /// </para>
    /// </remarks>
    private Tensor<T> ConcatenatePredictions(List<Tensor<T>> predictions, int totalSteps)
    {
        if (predictions.Count == 0)
            return new Tensor<T>(new[] { 1, totalSteps });

        int batchSize = predictions[0].Shape[0];
        var result = new Tensor<T>(new[] { batchSize, totalSteps });
        int currentStep = 0;

        foreach (var pred in predictions)
        {
            int predSteps = pred.Shape.Length > 1 ? pred.Shape[1] : pred.Length / batchSize;
            int stepsToCopy = Math.Min(predSteps, totalSteps - currentStep);

            for (int b = 0; b < batchSize; b++)
            {
                for (int t = 0; t < stepsToCopy; t++)
                {
                    int srcIdx = b * predSteps + t;
                    int dstIdx = b * totalSteps + currentStep + t;
                    if (srcIdx < pred.Length && dstIdx < result.Length)
                        result.Data.Span[dstIdx] = pred.Data.Span[srcIdx];
                }
            }

            currentStep += stepsToCopy;
            if (currentStep >= totalSteps)
                break;
        }

        return result;
    }

    #endregion

    #region IDisposable

    /// <summary>
    /// Releases resources used by the N-BEATS model.
    /// </summary>
    /// <param name="disposing">True if called from Dispose(), false if from finalizer.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the NBEATSFinance model, Dispose performs a supporting step in the workflow. It keeps the NBEATSFinance architecture pipeline consistent.
    /// </para>
    /// </remarks>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _onnxSession?.Dispose();
        }
        base.Dispose(disposing);
    }

    #endregion
}
