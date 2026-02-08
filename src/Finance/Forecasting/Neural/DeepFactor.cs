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
/// DeepFactor (Deep Factor Model) for multivariate time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// DeepFactor combines classical factor models with deep learning. It decomposes time series
/// into global factors (shared patterns) and local components (series-specific behavior),
/// learning both through neural networks for improved forecasting accuracy.
/// </para>
/// <para>
/// <b>For Beginners:</b> DeepFactor is designed for forecasting many related time series:
///
/// <b>The Factor Model Idea:</b>
/// Many time series are driven by common underlying patterns:
/// - Stock prices: Market factors, sector factors, economic factors
/// - Retail sales: Holiday effects, weather, economic conditions
/// - Energy demand: Temperature, time of day, day of week
///
/// <b>The Decomposition:</b>
/// y_t = (factor_loadings * global_factors_t) + local_t + noise
///
/// - Global factors: Shared patterns learned from all series
/// - Factor loadings: How much each series is affected by each factor
/// - Local component: Series-specific patterns not captured by factors
///
/// <b>Why Deep Learning?</b>
/// Traditional factor models use linear relationships.
/// DeepFactor uses neural networks to:
/// - Learn non-linear factor dynamics (factors can evolve in complex ways)
/// - Automatically discover the right number and type of factors
/// - Capture complex interactions between factors
///
/// <b>Architecture:</b>
/// 1. <b>Factor Model</b>: RNN that generates global factor values over time
/// 2. <b>Loading Layer</b>: Maps factors to series-specific contributions
/// 3. <b>Local Model</b>: Smaller network for series-specific patterns
/// 4. <b>Combination</b>: Merges factor-based and local predictions
///
/// <b>Benefits:</b>
/// - Efficient: Shares computation across many series via factors
/// - Interpretable: Factors can be analyzed to understand shared patterns
/// - Robust: Less overfitting when series share common dynamics
/// </para>
/// <para>
/// <b>Reference:</b> Wang et al., "Deep Factors for Forecasting", 2019.
/// https://arxiv.org/abs/1905.12417
/// </para>
/// </remarks>
public class DeepFactor<T> : ForecastingModelBase<T>
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
    /// Input projection layer for the factor model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Transforms raw input to the factor model's hidden dimension.
    /// </para>
    /// </remarks>
    private ILayer<T>? _factorInputProjection;

    /// <summary>
    /// RNN layers for the global factor model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> These layers learn the dynamics of global factors
    /// that affect all time series in the dataset.
    /// </para>
    /// </remarks>
    private readonly List<ILayer<T>> _factorRnnLayers = [];

    /// <summary>
    /// Layer that generates factor values for the forecast horizon.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Outputs factor values for each time step in the forecast.
    /// Shape: [numFactors * forecastHorizon].
    /// </para>
    /// </remarks>
    private ILayer<T>? _factorGenerationLayer;

    /// <summary>
    /// Factor loading layer that maps factors to predictions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Learns how strongly each factor affects this particular series.
    /// This is what makes the model "multivariate" - different series have different loadings.
    /// </para>
    /// </remarks>
    private ILayer<T>? _factorLoadingLayer;

    /// <summary>
    /// Input projection for the local model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Transforms input for the series-specific local model.
    /// </para>
    /// </remarks>
    private ILayer<T>? _localInputProjection;

    /// <summary>
    /// Layers for the local (series-specific) model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Captures patterns unique to this series that aren't
    /// explained by the global factors.
    /// </para>
    /// </remarks>
    private readonly List<ILayer<T>> _localLayers = [];

    /// <summary>
    /// Local prediction layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Outputs the local component of the forecast.
    /// </para>
    /// </remarks>
    private ILayer<T>? _localPredictionLayer;

    /// <summary>
    /// Combination layer that merges factor and local predictions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Combines factor-based prediction with local prediction
    /// to produce the final forecast: final = combine(factor_pred, local_pred).
    /// </para>
    /// </remarks>
    private ILayer<T>? _combinationLayer;

    #endregion

    #region Shared Fields

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly DeepFactorOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private int _lookbackWindow;
    private int _forecastHorizon;
    private int _numFeatures;
    private int _numFactors;
    private int _factorHiddenDim;
    private int _localHiddenDim;
    private int _numFactorLayers;
    private int _numLocalLayers;
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
    /// Gets the number of latent factors in the model.
    /// </summary>
    /// <value>The number of factors.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How many global patterns the model learns.
    /// Each factor represents a shared dynamic that affects multiple series.
    /// </para>
    /// </remarks>
    public int NumFactors => _numFactors;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a DeepFactor model using pretrained ONNX model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer for fine-tuning.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor to load a pretrained DeepFactor model
    /// for inference.
    /// </para>
    /// </remarks>
    public DeepFactor(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        DeepFactorOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        options ??= new DeepFactorOptions<T>();
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
        _numFactors = options.NumFactors;
        _factorHiddenDim = options.FactorHiddenDimension;
        _localHiddenDim = options.LocalHiddenDimension;
        _numFactorLayers = options.NumFactorLayers;
        _numLocalLayers = options.NumLocalLayers;
        _dropout = options.DropoutRate;
    }

    /// <summary>
    /// Creates a DeepFactor model in native mode for training from scratch.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor to train a new DeepFactor model.
    /// DeepFactor is ideal when you have:
    /// - Multiple related time series to forecast
    /// - Shared patterns across series (e.g., market factors, seasonal effects)
    /// - Hierarchical data (stores in regions, products in categories)
    /// </para>
    /// </remarks>
    public DeepFactor(
        NeuralNetworkArchitecture<T> architecture,
        DeepFactorOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new DeepFactorOptions<T>();
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
        _numFactors = options.NumFactors;
        _factorHiddenDim = options.FactorHiddenDimension;
        _localHiddenDim = options.LocalHiddenDimension;
        _numFactorLayers = options.NumFactorLayers;
        _numLocalLayers = options.NumLocalLayers;
        _dropout = options.DropoutRate;

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes the neural network layers for DeepFactor.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This sets up two parallel paths:
    /// 1. Factor model path: Input -> RNN -> Factor generation -> Loading
    /// 2. Local model path: Input -> Dense layers -> Local prediction
    /// 3. Combination: Merge both paths for final forecast
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
            ExtractLayerReferences();
        }
        else if (_useNativeMode)
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultDeepFactorLayers(
                Architecture, _lookbackWindow, _forecastHorizon, _numFeatures,
                _numFactors, _factorHiddenDim, _localHiddenDim,
                _numFactorLayers, _numLocalLayers, _dropout));

            ExtractLayerReferences();
        }
    }

    /// <summary>
    /// Extracts references to specific layers from the layer collection.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Organizes layers into factor path and local path:
    /// - Factor path: Input projection -> RNN layers -> Factor gen -> Loading
    /// - Local path: Local input -> Dense layers -> Local prediction
    /// - Final: Combination layer
    /// </para>
    /// </remarks>
    private void ExtractLayerReferences()
    {
        int idx = 0;

        // Factor model: Input projection
        if (idx < Layers.Count)
            _factorInputProjection = Layers[idx++];

        // Factor model: RNN layers (with optional dropout)
        _factorRnnLayers.Clear();
        int factorRnnWithDropout = _dropout > 0 ? _numFactorLayers * 2 : _numFactorLayers;
        for (int i = 0; i < factorRnnWithDropout && idx < Layers.Count; i++)
        {
            _factorRnnLayers.Add(Layers[idx++]);
        }

        // Factor generation layer
        if (idx < Layers.Count)
            _factorGenerationLayer = Layers[idx++];

        // Factor loading layer
        if (idx < Layers.Count)
            _factorLoadingLayer = Layers[idx++];

        // Local model: Input projection
        if (idx < Layers.Count)
            _localInputProjection = Layers[idx++];

        // Local model: Dense layers (with optional dropout)
        _localLayers.Clear();
        int localWithDropout = _dropout > 0 ? _numLocalLayers * 2 : _numLocalLayers;
        for (int i = 0; i < localWithDropout && idx < Layers.Count - 2; i++)
        {
            _localLayers.Add(Layers[idx++]);
        }

        // Local prediction layer
        if (idx < Layers.Count)
            _localPredictionLayer = Layers[idx++];

        // Combination layer
        if (idx < Layers.Count)
            _combinationLayer = Layers[idx];
    }

    /// <summary>
    /// Validates that custom layers meet DeepFactor architectural requirements.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Ensures you have layers for both factor and local paths plus combination.
    /// </para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);
        if (layers.Count < 7)
        {
            throw new ArgumentException(
                "DeepFactor requires at least 7 layers for factor model, local model, and combination.",
                nameof(layers));
        }
    }

    /// <summary>
    /// Validates the DeepFactor options.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Checks that all configuration values are valid.
    /// </para>
    /// </remarks>
    private static void ValidateOptions(DeepFactorOptions<T> options)
    {
        var errors = new List<string>();

        if (options.LookbackWindow < 1)
            errors.Add("LookbackWindow must be at least 1.");
        if (options.ForecastHorizon < 1)
            errors.Add("ForecastHorizon must be at least 1.");
        if (options.NumFactors < 1)
            errors.Add("NumFactors must be at least 1.");
        if (options.FactorHiddenDimension < 1)
            errors.Add("FactorHiddenDimension must be at least 1.");
        if (options.LocalHiddenDimension < 1)
            errors.Add("LocalHiddenDimension must be at least 1.");
        if (options.NumFactorLayers < 1)
            errors.Add("NumFactorLayers must be at least 1.");
        if (options.NumLocalLayers < 1)
            errors.Add("NumLocalLayers must be at least 1.");
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
    /// <b>For Beginners:</b> In the DeepFactor model, Predict produces predictions from input data. This is the main inference step of the DeepFactor architecture.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return _useNativeMode ? ForecastNative(input) : ForecastOnnx(input);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Training updates both the factor model (global patterns)
    /// and local model (series-specific patterns) to minimize prediction error.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is only supported in native mode.");

        SetTrainingMode(true);
        try
        {
            var predictions = Forward(input);
            LastLoss = _lossFunction.CalculateLoss(predictions.ToVector(), target.ToVector());

            var gradient = _lossFunction.CalculateDerivative(predictions.ToVector(), target.ToVector());
            Backward(Tensor<T>.FromVector(gradient, predictions.Shape));

            _optimizer.UpdateParameters(Layers);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the DeepFactor model, UpdateParameters updates internal parameters or state. This keeps the DeepFactor architecture aligned with the latest values.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> gradients)
    {
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the DeepFactor model, GetModelMetadata performs a supporting step in the workflow. It keeps the DeepFactor architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "DeepFactor" },
                { "LookbackWindow", _lookbackWindow },
                { "ForecastHorizon", _forecastHorizon },
                { "NumFactors", _numFactors },
                { "FactorHiddenDimension", _factorHiddenDim },
                { "LocalHiddenDimension", _localHiddenDim },
                { "NumFactorLayers", _numFactorLayers },
                { "NumLocalLayers", _numLocalLayers },
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
    /// <b>For Beginners:</b> Creates a fresh copy of the model architecture.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var options = new DeepFactorOptions<T>
        {
            LookbackWindow = _lookbackWindow,
            ForecastHorizon = _forecastHorizon,
            NumFactors = _numFactors,
            FactorHiddenDimension = _factorHiddenDim,
            LocalHiddenDimension = _localHiddenDim,
            NumFactorLayers = _numFactorLayers,
            NumLocalLayers = _numLocalLayers,
            DropoutRate = _dropout
        };

        if (_useNativeMode)
        {
            return new DeepFactor<T>(Architecture, options);
        }
        else
        {
            // Use null-coalescing throw to satisfy null analysis across all framework targets
            string onnxPath = OnnxModelPath ?? throw new InvalidOperationException(
                "Cannot create new instance from ONNX mode when OnnxModelPath is not available.");
            if (onnxPath.Length == 0)
            {
                throw new InvalidOperationException(
                    "Cannot create new instance from ONNX mode when OnnxModelPath is empty.");
            }
            return new DeepFactor<T>(Architecture, onnxPath, options);
        }
    }

    /// <summary>
    /// Writes DeepFactor-specific configuration during serialization.
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
        writer.Write(_numFactors);
        writer.Write(_factorHiddenDim);
        writer.Write(_localHiddenDim);
        writer.Write(_numFactorLayers);
        writer.Write(_numLocalLayers);
        writer.Write(_dropout);
    }

    /// <summary>
    /// Reads DeepFactor-specific configuration during deserialization.
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
        _numFactors = reader.ReadInt32();
        _factorHiddenDim = reader.ReadInt32();
        _localHiddenDim = reader.ReadInt32();
        _numFactorLayers = reader.ReadInt32();
        _numLocalLayers = reader.ReadInt32();
        _dropout = reader.ReadDouble();
    }

    #endregion

    #region IForecastingModel Implementation

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the DeepFactor model, Forecast produces predictions from input data. This is the main inference step of the DeepFactor architecture.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
    {
        return _useNativeMode ? ForecastNative(historicalData) : ForecastOnnx(historicalData);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> For forecasting beyond the horizon, DeepFactor uses
    /// its predictions as new history and continues generating forecasts.
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
    /// <b>For Beginners:</b> In the DeepFactor model, Evaluate performs a supporting step in the workflow. It keeps the DeepFactor architecture pipeline consistent.
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
    /// <b>For Beginners:</b> In the DeepFactor model, ApplyInstanceNormalization performs a supporting step in the workflow. It keeps the DeepFactor architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
    {
        return input;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the DeepFactor model, GetFinancialMetrics calculates evaluation metrics. This summarizes how the DeepFactor architecture is performing.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> GetFinancialMetrics()
    {
        T lastLoss = LastLoss is not null ? LastLoss : NumOps.Zero;

        return new Dictionary<string, T>
        {
            ["LookbackWindow"] = NumOps.FromDouble(_lookbackWindow),
            ["ForecastHorizon"] = NumOps.FromDouble(_forecastHorizon),
            ["NumFactors"] = NumOps.FromDouble(_numFactors),
            ["FactorHiddenDim"] = NumOps.FromDouble(_factorHiddenDim),
            ["LocalHiddenDim"] = NumOps.FromDouble(_localHiddenDim),
            ["LastLoss"] = lastLoss
        };
    }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// Performs the forward pass through DeepFactor.
    /// </summary>
    /// <param name="input">Input tensor of shape [batch, lookback_window * features].</param>
    /// <returns>Output tensor of shape [batch, forecast_horizon].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The DeepFactor forward pass has two parallel paths:
    ///
    /// <b>Factor Path:</b>
    /// 1. Input projection to hidden dimension
    /// 2. RNN processes sequence, extracts temporal context
    /// 3. Factor generation: Output [numFactors * forecastHorizon] values
    /// 4. Factor loading: Map factors to series-specific prediction
    ///
    /// <b>Local Path:</b>
    /// 1. Separate input projection
    /// 2. Dense layers for series-specific patterns
    /// 3. Local prediction output
    ///
    /// <b>Combination:</b>
    /// Concatenate factor and local predictions, then combine with final layer.
    /// </para>
    /// </remarks>
    private Tensor<T> Forward(Tensor<T> input)
    {
        if (!_useNativeMode)
            return ForecastOnnx(input);

        // === Factor Path ===
        var factorCurrent = input;

        if (_factorInputProjection is not null)
            factorCurrent = _factorInputProjection.Forward(factorCurrent);

        foreach (var layer in _factorRnnLayers)
        {
            factorCurrent = layer.Forward(factorCurrent);
        }

        if (_factorGenerationLayer is not null)
            factorCurrent = _factorGenerationLayer.Forward(factorCurrent);

        Tensor<T>? factorPrediction = null;
        if (_factorLoadingLayer is not null)
            factorPrediction = _factorLoadingLayer.Forward(factorCurrent);

        // === Local Path ===
        var localCurrent = input;

        if (_localInputProjection is not null)
            localCurrent = _localInputProjection.Forward(input);

        foreach (var layer in _localLayers)
        {
            localCurrent = layer.Forward(localCurrent);
        }

        Tensor<T>? localPrediction = null;
        if (_localPredictionLayer is not null)
            localPrediction = _localPredictionLayer.Forward(localCurrent);

        // === Combination ===
        // Concatenate factor and local predictions
        var combined = ConcatenateTensors(
            factorPrediction ?? new Tensor<T>(new[] { _forecastHorizon }),
            localPrediction ?? new Tensor<T>(new[] { _forecastHorizon }));

        if (_combinationLayer is not null)
            combined = _combinationLayer.Forward(combined);

        return combined;
    }

    /// <summary>
    /// Concatenates two tensors along the last dimension.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Joins factor prediction and local prediction into one tensor
    /// for the combination layer.
    /// </para>
    /// </remarks>
    private static Tensor<T> ConcatenateTensors(Tensor<T> a, Tensor<T> b)
    {
        var result = new Tensor<T>(new[] { a.Length + b.Length });

        for (int i = 0; i < a.Length; i++)
        {
            result.Data.Span[i] = a.Data.Span[i];
        }

        for (int i = 0; i < b.Length; i++)
        {
            result.Data.Span[a.Length + i] = b.Data.Span[i];
        }

        return result;
    }

    /// <summary>
    /// Performs the backward pass through DeepFactor.
    /// </summary>
    /// <param name="gradOutput">Gradient from the loss function.</param>
    /// <returns>Gradient with respect to the input.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Backpropagation splits gradients between the factor path
    /// and local path, updating both to minimize prediction error.
    /// </para>
    /// </remarks>
    private Tensor<T> Backward(Tensor<T> gradOutput)
    {
        var current = gradOutput;

        // Combination layer backward
        if (_combinationLayer is not null)
            current = _combinationLayer.Backward(current);

        // Split gradient for factor and local paths
        // Account for multivariate outputs: chunk size = forecastHorizon * numFeatures
        int outputElements = _forecastHorizon * Math.Max(_numFeatures, 1);
        var factorGrad = new Tensor<T>(new[] { outputElements });
        var localGrad = new Tensor<T>(new[] { outputElements });

        for (int i = 0; i < outputElements && i < current.Length; i++)
        {
            factorGrad.Data.Span[i] = current.Data.Span[i];
        }
        for (int i = 0; i < outputElements && i + outputElements < current.Length; i++)
        {
            localGrad.Data.Span[i] = current.Data.Span[i + outputElements];
        }

        // === Factor Path Backward ===
        var factorCurrent = factorGrad;

        if (_factorLoadingLayer is not null)
            factorCurrent = _factorLoadingLayer.Backward(factorCurrent);

        if (_factorGenerationLayer is not null)
            factorCurrent = _factorGenerationLayer.Backward(factorCurrent);

        for (int i = _factorRnnLayers.Count - 1; i >= 0; i--)
        {
            factorCurrent = _factorRnnLayers[i].Backward(factorCurrent);
        }

        if (_factorInputProjection is not null)
            factorCurrent = _factorInputProjection.Backward(factorCurrent);

        // === Local Path Backward ===
        var localCurrent = localGrad;

        if (_localPredictionLayer is not null)
            localCurrent = _localPredictionLayer.Backward(localCurrent);

        for (int i = _localLayers.Count - 1; i >= 0; i--)
        {
            localCurrent = _localLayers[i].Backward(localCurrent);
        }

        if (_localInputProjection is not null)
            localCurrent = _localInputProjection.Backward(localCurrent);

        // Combine input gradients from both paths
        // Both paths receive the same input, so gradients should be summed
        int maxLen = Math.Max(factorCurrent.Length, localCurrent.Length);
        var combined = new Tensor<T>(new[] { maxLen });

        // Add factor path gradients
        for (int i = 0; i < factorCurrent.Length && i < maxLen; i++)
        {
            combined.Data.Span[i] = factorCurrent.Data.Span[i];
        }

        // Add local path gradients (summed with factor gradients)
        for (int i = 0; i < localCurrent.Length && i < maxLen; i++)
        {
            combined.Data.Span[i] = NumOps.Add(combined.Data.Span[i], localCurrent.Data.Span[i]);
        }

        return combined;
    }

    #endregion

    #region Inference Methods

    /// <summary>
    /// Performs native mode forecasting.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Uses the trained neural network layers to produce forecasts
    /// by combining global factor predictions with local predictions.
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
    /// Shifts input tensor by incorporating predictions for autoregressive forecasting.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> For multi-step forecasting, we slide the input window forward
    /// and add our predictions as new "history".
    /// </para>
    /// </remarks>
    protected override Tensor<T> ShiftInputWithPredictions(Tensor<T> input, Tensor<T> predictions, int stepsUsed)
    {
        int totalElements = _lookbackWindow * _numFeatures;
        var newInput = new Tensor<T>(input.Shape);

        int shift = stepsUsed * _numFeatures;
        for (int i = 0; i < totalElements - shift; i++)
        {
            newInput.Data.Span[i] = input.Data.Span[i + shift];
        }

        // Copy all features for each time step from predictions
        for (int step = 0; step < stepsUsed; step++)
        {
            for (int f = 0; f < _numFeatures; f++)
            {
                int predIdx = step * _numFeatures + f;
                int targetIdx = totalElements - shift + step * _numFeatures + f;
                if (predIdx < predictions.Length && targetIdx >= 0 && targetIdx < totalElements)
                {
                    newInput.Data.Span[targetIdx] = predictions.Data.Span[predIdx];
                }
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
        // Account for multivariate outputs
        int features = _numFeatures > 0 ? _numFeatures : 1;
        var result = new Tensor<T>(new[] { totalSteps, features });

        int stepsAdded = 0;

        foreach (var pred in predictions)
        {
            int stepsToAdd = Math.Min(_forecastHorizon, totalSteps - stepsAdded);
            // For 1D tensors, derive feature count from length/steps to preserve multivariate outputs
            int predFeatures = pred.Shape.Length > 1
                ? pred.Shape[^1]
                : (stepsToAdd > 0 ? Math.Max(1, pred.Length / stepsToAdd) : 1);
            int featuresToCopy = Math.Min(features, predFeatures);

            for (int i = 0; i < stepsToAdd; i++)
            {
                for (int f = 0; f < featuresToCopy; f++)
                {
                    int srcIdx = i * predFeatures + f;
                    int dstIdx = (stepsAdded + i) * features + f;
                    if (srcIdx < pred.Length && dstIdx < result.Length)
                        result.Data.Span[dstIdx] = pred.Data.Span[srcIdx];
                }
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
    /// Disposes resources used by the DeepFactor model.
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

