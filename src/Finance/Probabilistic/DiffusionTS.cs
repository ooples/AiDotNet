using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Finance.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

using AiDotNet.Finance.Base;
namespace AiDotNet.Finance.Probabilistic;

/// <summary>
/// DiffusionTS (Interpretable Diffusion for Time Series) for probabilistic forecasting with seasonal-trend decomposition.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// DiffusionTS is an interpretable diffusion model that uses seasonal-trend decomposition
/// to generate forecasts with clear interpretable components.
/// </para>
/// <para><b>For Beginners:</b> DiffusionTS makes diffusion models more interpretable
/// by decomposing time series into understandable components:
///
/// <b>The Key Insight:</b>
/// Time series often have clear structure (trends, seasonality) that gets lost in
/// "black box" models. DiffusionTS preserves this structure by generating each
/// component separately and combining them.
///
/// <b>How DiffusionTS Works:</b>
/// 1. <b>Decomposition:</b> Split time series into trend, seasonal, and residual
/// 2. <b>Component Diffusion:</b> Generate each component with specialized networks
/// 3. <b>Reconstruction:</b> Combine components to form final forecast
/// 4. <b>Interpretation:</b> Each component has clear meaning
///
/// <b>DiffusionTS Architecture:</b>
/// - Trend Network: Captures long-term movements (slow, smooth)
/// - Seasonal Network: Captures periodic patterns (daily, weekly, yearly)
/// - Residual Network: Captures irregular fluctuations
/// - Fusion Module: Combines components coherently
///
/// <b>Key Benefits:</b>
/// - Interpretable decomposition of forecasts
/// - Can enforce structural constraints (smooth trends, periodic seasons)
/// - Better uncertainty quantification per component
/// - Enables "what-if" analysis by modifying components
/// </para>
/// <para>
/// <b>Reference:</b> Yuan and Qiu, "Diffusion-TS: Interpretable Diffusion for General Time Series Generation", 2024.
/// https://arxiv.org/abs/2403.01742
/// </para>
/// </remarks>
public class DiffusionTS<T> : ForecastingModelBase<T>
{
    #region Execution Mode
    private readonly bool _useNativeMode;
    #endregion

    
    #region Native Mode Fields
    private DenseLayer<T>? _trendInputLayer;
    private DenseLayer<T>? _seasonalInputLayer;
    private DenseLayer<T>? _residualInputLayer;
    private DenseLayer<T>? _fusionLayer;
    private DenseLayer<T>? _outputLayer;
    private List<LayerNormalizationLayer<T>>? _layerNorms;
    #endregion

    #region Diffusion Fields
    private readonly double[] _betas;
    private readonly double[] _alphas;
    private readonly double[] _alphasCumprod;
    private readonly double[] _sqrtAlphasCumprod;
    private readonly double[] _sqrtOneMinusAlphasCumprod;
    private readonly Random _random;
    #endregion

    #region Shared Fields
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly DiffusionTSOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private int _sequenceLength;
    private int _forecastHorizon;
    private int _numFeatures;
    private int _hiddenDimension;
    private int _trendHiddenDim;
    private int _seasonalHiddenDim;
    private int _numDiffusionSteps;
    private int _numSamples;
    private int _decompositionPeriod;
    private int _trendKernelSize;
    private bool _useTrendComponent;
    private bool _useSeasonalComponent;
    private string _betaSchedule;
    #endregion

    #region IForecastingModel Properties

    /// <inheritdoc/>
    public override int SequenceLength => _sequenceLength;

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
    /// Gets the forecast horizon.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many future steps to predict.
    /// </para>
    /// </remarks>
    public int ForecastHorizon => _forecastHorizon;

    /// <summary>
    /// Gets whether the model supports training (native mode only).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> ONNX mode is inference-only.
    /// Native mode supports both training and inference.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => _useNativeMode;

    /// <summary>
    /// Gets the number of diffusion steps.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> More steps = higher quality but slower.
    /// </para>
    /// </remarks>
    public int NumDiffusionSteps => _numDiffusionSteps;

    /// <summary>
    /// Gets the number of samples for uncertainty estimation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many different forecasts to generate
    /// for uncertainty quantification.
    /// </para>
    /// </remarks>
    public int NumSamples => _numSamples;

    /// <summary>
    /// Gets the decomposition period for seasonal extraction.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The expected periodicity of seasonal patterns.
    /// </para>
    /// </remarks>
    public int DecompositionPeriod => _decompositionPeriod;

    /// <summary>
    /// Gets whether trend component is used.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When true, the model generates a separate
    /// trend component for smooth long-term movements.
    /// </para>
    /// </remarks>
    public bool UseTrendComponent => _useTrendComponent;

    /// <summary>
    /// Gets whether seasonal component is used.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When true, the model generates a separate
    /// seasonal component for periodic patterns.
    /// </para>
    /// </remarks>
    public bool UseSeasonalComponent => _useSeasonalComponent;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the DiffusionTS model in ONNX mode for inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to a pretrained ONNX model file.</param>
    /// <param name="options">DiffusionTS-specific options.</param>
    /// <param name="optimizer">Optional optimizer for fine-tuning.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to load a pretrained DiffusionTS model
    /// for fast interpretable forecasting. The ONNX model encapsulates the trained
    /// decomposition networks for trend, seasonal, and residual components.
    /// </para>
    /// </remarks>
    public DiffusionTS(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        DiffusionTSOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentNullException(nameof(onnxModelPath));
        if (!System.IO.File.Exists(onnxModelPath))
            throw new System.IO.FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        _useNativeMode = false;
        OnnxModelPath = onnxModelPath;
        OnnxSession = new InferenceSession(onnxModelPath);
        _options = options ?? new DiffusionTSOptions<T>();
        Options = _options;
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        _sequenceLength = _options.SequenceLength;
        _forecastHorizon = _options.ForecastHorizon;
        _numFeatures = _options.NumFeatures;
        _hiddenDimension = _options.HiddenDimension;
        _trendHiddenDim = _options.TrendHiddenDim;
        _seasonalHiddenDim = _options.SeasonalHiddenDim;
        _numDiffusionSteps = _options.NumDiffusionSteps;
        _numSamples = _options.NumSamples;
        _decompositionPeriod = _options.DecompositionPeriod;
        _trendKernelSize = _options.TrendKernelSize;
        _useTrendComponent = _options.UseTrendComponent;
        _useSeasonalComponent = _options.UseSeasonalComponent;
        _betaSchedule = _options.BetaSchedule;

        (_betas, _alphas, _alphasCumprod, _sqrtAlphasCumprod, _sqrtOneMinusAlphasCumprod) =
            InitializeDiffusionSchedule(_numDiffusionSteps, _options.BetaStart, _options.BetaEnd, _betaSchedule);
        _random = RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Initializes a new instance of the DiffusionTS model in native mode for training.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">DiffusionTS-specific options.</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="optimizer">Optional optimizer for training.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to create a DiffusionTS model
    /// that can be trained on your data. The model learns separate networks for
    /// generating trend, seasonal, and residual components, which are then combined.
    /// </para>
    /// </remarks>
    public DiffusionTS(
        NeuralNetworkArchitecture<T> architecture,
        DiffusionTSOptions<T>? options = null,
        int numFeatures = 1,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        _useNativeMode = true;
        _options = options ?? new DiffusionTSOptions<T>();
        Options = _options;
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        _sequenceLength = _options.SequenceLength;
        _forecastHorizon = _options.ForecastHorizon;
        _numFeatures = numFeatures > 0 ? numFeatures : _options.NumFeatures;
        _hiddenDimension = _options.HiddenDimension;
        _trendHiddenDim = _options.TrendHiddenDim;
        _seasonalHiddenDim = _options.SeasonalHiddenDim;
        _numDiffusionSteps = _options.NumDiffusionSteps;
        _numSamples = _options.NumSamples;
        _decompositionPeriod = _options.DecompositionPeriod;
        _trendKernelSize = _options.TrendKernelSize;
        _useTrendComponent = _options.UseTrendComponent;
        _useSeasonalComponent = _options.UseSeasonalComponent;
        _betaSchedule = _options.BetaSchedule;

        (_betas, _alphas, _alphasCumprod, _sqrtAlphasCumprod, _sqrtOneMinusAlphasCumprod) =
            InitializeDiffusionSchedule(_numDiffusionSteps, _options.BetaStart, _options.BetaEnd, _betaSchedule);
        _random = RandomHelper.CreateSecureRandom();

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes all layers for the DiffusionTS model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up three specialized sub-networks:
    ///
    /// <b>Layer Structure:</b>
    /// 1. Trend Network: Smoothing layers for capturing long-term movements
    /// 2. Seasonal Network: Periodic feature extractors with Fourier-like processing
    /// 3. Residual Network: Dense layers for irregular fluctuations
    /// 4. Fusion Module: Combines all component outputs
    ///
    /// Each network is designed for its specific purpose - trend networks have
    /// smoothing behavior, seasonal networks capture periodicity, and residual
    /// networks handle what's left over.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultDiffusionTSLayers(
                Architecture,
                _sequenceLength,
                _forecastHorizon,
                _numFeatures,
                _hiddenDimension,
                _trendHiddenDim,
                _seasonalHiddenDim));

            ExtractLayerReferences();
        }
    }

    /// <summary>
    /// Extracts references to key layers for efficient access during forward/backward passes.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Keeps direct references to the trend, seasonal,
    /// and residual network layers. This allows the model to process each component
    /// separately during generation and combine them in the fusion layer.
    /// </para>
    /// </remarks>
    private void ExtractLayerReferences()
    {
        var allDense = Layers.OfType<DenseLayer<T>>().ToList();
        _layerNorms = Layers.OfType<LayerNormalizationLayer<T>>().ToList();

        // Organize layers by component networks
        if (allDense.Count >= 5)
        {
            _trendInputLayer = allDense[0];
            _seasonalInputLayer = allDense.Count > 3 ? allDense[3] : null;
            _residualInputLayer = allDense.Count > 6 ? allDense[6] : null;
            _fusionLayer = allDense.Count > 9 ? allDense[9] : allDense[allDense.Count - 2];
            _outputLayer = allDense[allDense.Count - 1];
        }
    }

    /// <summary>
    /// Initializes the diffusion noise schedule.
    /// </summary>
    /// <param name="numSteps">Number of diffusion steps.</param>
    /// <param name="betaStart">Starting noise level.</param>
    /// <param name="betaEnd">Ending noise level.</param>
    /// <param name="schedule">Schedule type ("linear", "cosine", or "quadratic").</param>
    /// <returns>Tuple of precomputed diffusion values for efficient computation.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The noise schedule controls how quickly noise is added
    /// during forward diffusion and removed during reverse diffusion. Precomputing these
    /// values makes the diffusion process more efficient. Linear schedules add noise
    /// uniformly, cosine schedules are smoother, and quadratic schedules add more
    /// noise toward the end.
    /// </para>
    /// </remarks>
    private (double[] betas, double[] alphas, double[] alphasCumprod, double[] sqrtAlphasCumprod, double[] sqrtOneMinusAlphasCumprod)
        InitializeDiffusionSchedule(int numSteps, double betaStart, double betaEnd, string schedule)
    {
        var betas = new double[numSteps];

        if (schedule == "linear")
        {
            for (int i = 0; i < numSteps; i++)
            {
                betas[i] = betaStart + (betaEnd - betaStart) * i / (numSteps - 1);
            }
        }
        else if (schedule == "cosine")
        {
            double s = 0.008;
            for (int i = 0; i < numSteps; i++)
            {
                double t = (double)i / numSteps;
                double alphaBar = Math.Cos((t + s) / (1 + s) * Math.PI / 2);
                alphaBar = alphaBar * alphaBar;
                betas[i] = Math.Min(1 - alphaBar, 0.999);
            }
        }
        else // quadratic
        {
            for (int i = 0; i < numSteps; i++)
            {
                double t = (double)i / (numSteps - 1);
                betas[i] = betaStart + (betaEnd - betaStart) * t * t;
            }
        }

        var alphas = new double[numSteps];
        var alphasCumprod = new double[numSteps];
        var sqrtAlphasCumprod = new double[numSteps];
        var sqrtOneMinusAlphasCumprod = new double[numSteps];

        double cumprod = 1.0;
        for (int i = 0; i < numSteps; i++)
        {
            alphas[i] = 1.0 - betas[i];
            cumprod *= alphas[i];
            alphasCumprod[i] = cumprod;
            sqrtAlphasCumprod[i] = Math.Sqrt(cumprod);
            sqrtOneMinusAlphasCumprod[i] = Math.Sqrt(1.0 - cumprod);
        }

        return (betas, alphas, alphasCumprod, sqrtAlphasCumprod, sqrtOneMinusAlphasCumprod);
    }

    /// <summary>
    /// Validates custom layers provided by the user.
    /// </summary>
    /// <param name="layers">The list of custom layers to validate.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> When users provide their own layer configuration,
    /// this ensures the layers form a valid DiffusionTS architecture with separate
    /// trend, seasonal, and residual processing paths.
    /// </para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);

        if (layers.Count < 5)
            throw new ArgumentException("DiffusionTS requires at least 5 layers (trend, seasonal, residual, fusion, output).");
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <summary>
    /// Performs forward prediction on the input tensor.
    /// </summary>
    /// <param name="input">Input tensor containing historical time series data.</param>
    /// <returns>Output tensor containing the forecast with decomposed components.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This runs the interpretable diffusion process:
    /// 1. Decomposes input into trend, seasonal, and residual
    /// 2. Generates each component separately using diffusion
    /// 3. Combines components for final forecast
    /// The result is an interpretable forecast where you can see each component's contribution.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return _useNativeMode ? ForecastNative(input) : ForecastOnnx(input);
    }

    /// <summary>
    /// Trains the DiffusionTS model on a batch of input-target pairs.
    /// </summary>
    /// <param name="input">Input tensor containing the full time series for training.</param>
    /// <param name="target">Target tensor (same as input for generation training).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Training involves:
    /// 1. Decompose target into trend, seasonal, and residual components
    /// 2. Add noise to each component at a random timestep
    /// 3. Predict the noise using the specialized denoising networks
    /// 4. Minimize the difference between predicted and actual noise
    ///
    /// Each component network learns its specific role - trend networks learn
    /// smooth patterns, seasonal networks learn periodic patterns, and residual
    /// networks learn irregular fluctuations.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training requires native mode.");

        SetTrainingMode(true);

        // Sample random timestep
        int t = _random.Next(_numDiffusionSteps);

        // Decompose target into components
        var (trend, seasonal, residual) = DecomposeTimeSeries(target);

        // Add noise to each component
        var (noisyTrend, noiseTrend) = AddNoise(trend, t);
        var (noisySeasonal, noiseSeasonal) = AddNoise(seasonal, t);
        var (noisyResidual, noiseResidual) = AddNoise(residual, t);

        // Combine noisy components for forward pass
        var combined = CombineComponents(noisyTrend, noisySeasonal, noisyResidual);

        // Forward pass: predict noise
        var output = Forward(combined);

        // Combine target noise
        var targetNoise = CombineComponents(noiseTrend, noiseSeasonal, noiseResidual);

        // Ensure shapes match for loss calculation - use minimum length
        var outputVec = output.ToVector();
        var targetVec = targetNoise.ToVector();
        int minLength = Math.Min(outputVec.Length, targetVec.Length);

        // Create matching-length vectors for loss calculation
        var matchedOutput = new T[minLength];
        var matchedTarget = new T[minLength];
        for (int i = 0; i < minLength; i++)
        {
            matchedOutput[i] = outputVec[i];
            matchedTarget[i] = targetVec[i];
        }

        // Calculate loss using matched-size vectors
        var matchedOutputVec = new Vector<T>(matchedOutput);
        var matchedTargetVec = new Vector<T>(matchedTarget);
        LastLoss = _lossFunction.CalculateLoss(matchedOutputVec, matchedTargetVec);

        // Backward pass - use matched size for gradient computation
        var gradient = _lossFunction.CalculateDerivative(matchedOutputVec, matchedTargetVec);

        // Pad gradient back to output shape if needed
        var fullGradient = new T[output.Length];
        for (int i = 0; i < Math.Min(gradient.Length, fullGradient.Length); i++)
        {
            fullGradient[i] = gradient[i];
        }
        Backward(Tensor<T>.FromVector(new Vector<T>(fullGradient), output.Shape));

        _optimizer.UpdateParameters(Layers);

        SetTrainingMode(false);
    }

    /// <summary>
    /// Updates parameters using the provided gradients.
    /// </summary>
    /// <param name="gradients">Gradient vector for parameter updates.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Parameters are updated through the optimizer in Train().
    /// This method exists for interface compliance.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> gradients)
    {
        // Parameters are updated through the optimizer in Train()
    }

    /// <summary>
    /// Gets metadata about the DiffusionTS model.
    /// </summary>
    /// <returns>ModelMetadata containing comprehensive model information.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Returns all configuration details about the model
    /// including the decomposition settings, diffusion parameters, and training state.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "DiffusionTS" },
                { "SequenceLength", _sequenceLength },
                { "ForecastHorizon", _forecastHorizon },
                { "NumFeatures", _numFeatures },
                { "HiddenDimension", _hiddenDimension },
                { "TrendHiddenDim", _trendHiddenDim },
                { "SeasonalHiddenDim", _seasonalHiddenDim },
                { "NumDiffusionSteps", _numDiffusionSteps },
                { "NumSamples", _numSamples },
                { "DecompositionPeriod", _decompositionPeriod },
                { "TrendKernelSize", _trendKernelSize },
                { "UseTrendComponent", _useTrendComponent },
                { "UseSeasonalComponent", _useSeasonalComponent },
                { "BetaSchedule", _betaSchedule },
                { "UseNativeMode", _useNativeMode }
            }
        };
    }

    /// <summary>
    /// Creates a new instance with the same configuration.
    /// </summary>
    /// <returns>A new DiffusionTS instance with identical settings.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a fresh model with the same architecture
    /// and options but without trained weights. Useful for ensemble methods.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new DiffusionTS<T>(Architecture, _options, _numFeatures);
    }

    /// <summary>
    /// Serializes DiffusionTS-specific data for model persistence.
    /// </summary>
    /// <param name="writer">The binary writer for serialization.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Saves all the model configuration so it can
    /// be reconstructed later. This includes decomposition settings and
    /// diffusion parameters.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_sequenceLength);
        writer.Write(_forecastHorizon);
        writer.Write(_numFeatures);
        writer.Write(_hiddenDimension);
        writer.Write(_trendHiddenDim);
        writer.Write(_seasonalHiddenDim);
        writer.Write(_numDiffusionSteps);
        writer.Write(_numSamples);
        writer.Write(_decompositionPeriod);
        writer.Write(_trendKernelSize);
        writer.Write(_useTrendComponent);
        writer.Write(_useSeasonalComponent);
        writer.Write(_betaSchedule);
    }

    /// <summary>
    /// Deserializes DiffusionTS-specific data from a saved model.
    /// </summary>
    /// <param name="reader">The binary reader for deserialization.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Reads the saved model configuration to restore
    /// all settings. The values are read but not assigned since they were set
    /// in the constructor.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _sequenceLength = reader.ReadInt32();
        _forecastHorizon = reader.ReadInt32();
        _numFeatures = reader.ReadInt32();
        _hiddenDimension = reader.ReadInt32();
        _trendHiddenDim = reader.ReadInt32();
        _seasonalHiddenDim = reader.ReadInt32();
        _numDiffusionSteps = reader.ReadInt32();
        _numSamples = reader.ReadInt32();
        _decompositionPeriod = reader.ReadInt32();
        _trendKernelSize = reader.ReadInt32();
        _useTrendComponent = reader.ReadBoolean();
        _useSeasonalComponent = reader.ReadBoolean();
        _betaSchedule = reader.ReadString();
    }

    #endregion

    #region IForecastingModel Implementation

    /// <summary>
    /// Generates forecasts for the given input time series.
    /// </summary>
    /// <param name="historicalData">Input tensor containing historical data (context).</param>
    /// <param name="quantiles">Optional quantile levels for probabilistic forecasting (e.g., [0.1, 0.5, 0.9]).</param>
    /// <returns>Forecast tensor with predicted values.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Runs interpretable diffusion to generate forecasts.
    /// If quantiles are specified, generates multiple samples and computes percentiles.
    /// The forecast is a sum of trend, seasonal, and residual components.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
    {
        if (quantiles is not null && quantiles.Length > 0)
        {
            var samples = GenerateSamples(historicalData, _numSamples);
            return ComputeQuantiles(samples, quantiles);
        }

        return _useNativeMode ? ForecastNative(historicalData) : ForecastOnnx(historicalData);
    }

    /// <summary>
    /// Generates forecasts with prediction intervals.
    /// </summary>
    /// <param name="input">Input tensor containing historical data.</param>
    /// <param name="confidenceLevel">Confidence level for intervals (e.g., 0.95 for 95%).</param>
    /// <returns>Tuple of (forecast median, lower bound, upper bound).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Generates multiple samples using different
    /// random seeds and computes confidence intervals. Higher confidence levels
    /// produce wider intervals.
    /// </para>
    /// </remarks>
    public (Tensor<T> Forecast, Tensor<T> Lower, Tensor<T> Upper) ForecastWithIntervals(
        Tensor<T> input,
        double confidenceLevel = 0.95)
    {
        var samples = GenerateSamples(input, _numSamples);
        return ComputePredictionIntervals(samples, confidenceLevel);
    }

    /// <summary>
    /// Performs autoregressive forecasting for extended horizons.
    /// </summary>
    /// <param name="input">Initial input tensor.</param>
    /// <param name="steps">Number of forecast steps to generate.</param>
    /// <returns>Extended forecast tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Generates forecasts iteratively by using
    /// previous predictions as input for the next step. Useful for forecasting
    /// further than the model's native horizon.
    /// </para>
    /// </remarks>
    public override Tensor<T> AutoregressiveForecast(Tensor<T> input, int steps)
    {
        var predictions = new List<Tensor<T>>();
        var currentInput = input;

        for (int i = 0; i < steps; i++)
        {
            var prediction = Forecast(currentInput, null);
            predictions.Add(prediction);
            currentInput = ShiftInputWindow(currentInput, prediction);
        }

        return ConcatenatePredictions(predictions);
    }

    /// <summary>
    /// Evaluates forecast quality against actual values.
    /// </summary>
    /// <param name="predictions">Predicted values tensor.</param>
    /// <param name="actuals">Actual values tensor for comparison.</param>
    /// <returns>Dictionary of evaluation metrics (MSE, MAE, RMSE).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Computes standard forecasting metrics:
    /// - MSE: Mean Squared Error (penalizes large errors more)
    /// - MAE: Mean Absolute Error (robust to outliers)
    /// - RMSE: Root Mean Squared Error (same units as data)
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> Evaluate(Tensor<T> predictions, Tensor<T> actuals)
    {
        var metrics = new Dictionary<string, T>();

        T mse = NumOps.Zero;
        T mae = NumOps.Zero;
        int count = Math.Min(predictions.Data.Length, actuals.Data.Length);

        for (int i = 0; i < count; i++)
        {
            T diff = NumOps.Subtract(predictions.Data.Span[i], actuals.Data.Span[i]);
            mse = NumOps.Add(mse, NumOps.Multiply(diff, diff));
            mae = NumOps.Add(mae, NumOps.Abs(diff));
        }

        if (count > 0)
        {
            mse = NumOps.Divide(mse, NumOps.FromDouble(count));
            mae = NumOps.Divide(mae, NumOps.FromDouble(count));
        }

        T rmse = NumOps.Sqrt(mse);

        metrics["MSE"] = mse;
        metrics["MAE"] = mae;
        metrics["RMSE"] = rmse;

        return metrics;
    }

    /// <summary>
    /// Applies instance normalization (RevIN) to the input.
    /// </summary>
    /// <param name="input">Input tensor to normalize.</param>
    /// <returns>Normalized tensor (identity for DiffusionTS).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> DiffusionTS handles normalization internally
    /// through the decomposition process, so this returns the input unchanged.
    /// </para>
    /// </remarks>
    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
    {
        return input;
    }

    /// <summary>
    /// Gets financial-specific metrics for model evaluation.
    /// </summary>
    /// <returns>Dictionary of financial metrics.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Returns model statistics relevant for
    /// financial forecasting applications.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> GetFinancialMetrics()
    {
        T lastLoss = LastLoss is not null ? LastLoss : NumOps.Zero;

        return new Dictionary<string, T>
        {
            ["LastLoss"] = lastLoss,
            ["SequenceLength"] = NumOps.FromDouble(_sequenceLength),
            ["ForecastHorizon"] = NumOps.FromDouble(_forecastHorizon),
            ["NumDiffusionSteps"] = NumOps.FromDouble(_numDiffusionSteps),
            ["NumSamples"] = NumOps.FromDouble(_numSamples),
            ["DecompositionPeriod"] = NumOps.FromDouble(_decompositionPeriod)
        };
    }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// Performs the forward pass through all layers.
    /// </summary>
    /// <param name="input">Input tensor (combined noisy components).</param>
    /// <returns>Output tensor (predicted noise for all components).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The forward pass processes the combined
    /// noisy components through the specialized networks and predicts the
    /// noise to be removed for each component.
    /// </para>
    /// </remarks>
    public Tensor<T> Forward(Tensor<T> input)
    {
        var current = FlattenInput(input);

        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }

        return current;
    }

    /// <summary>
    /// Performs backpropagation through all layers.
    /// </summary>
    /// <param name="gradOutput">Gradient of loss with respect to output.</param>
    /// <returns>Gradient with respect to input.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Backpropagation computes how each parameter
    /// contributed to the error, allowing the optimizer to adjust them for
    /// better predictions.
    /// </para>
    /// </remarks>
    public Tensor<T> Backward(Tensor<T> gradOutput)
    {
        var grad = gradOutput;

        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            grad = Layers[i].Backward(grad);
        }

        return grad;
    }

    /// <summary>
    /// Flattens input tensor for dense layer processing.
    /// </summary>
    /// <param name="input">Multi-dimensional input tensor.</param>
    /// <returns>Flattened 1D tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Converts multi-dimensional time series data
    /// into a flat vector that dense layers can process.
    /// </para>
    /// </remarks>
    private Tensor<T> FlattenInput(Tensor<T> input)
    {
        int totalSize = 1;
        foreach (var dim in input.Shape)
        {
            totalSize *= dim;
        }

        var flattened = new Tensor<T>(new[] { totalSize });
        for (int i = 0; i < totalSize; i++)
        {
            flattened.Data.Span[i] = input.Data.Span[i];
        }

        return flattened;
    }

    #endregion

    #region Decomposition Methods

    /// <summary>
    /// Decomposes a time series into trend, seasonal, and residual components.
    /// </summary>
    /// <param name="timeSeries">Input time series tensor.</param>
    /// <returns>Tuple of (trend, seasonal, residual) component tensors.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Time series decomposition splits data into:
    /// - Trend: The long-term direction (up, down, or flat)
    /// - Seasonal: Repeating patterns (daily, weekly, yearly cycles)
    /// - Residual: What's left after removing trend and seasonality
    ///
    /// This makes the model interpretable because each component has clear meaning.
    /// </para>
    /// </remarks>
    private (Tensor<T> trend, Tensor<T> seasonal, Tensor<T> residual) DecomposeTimeSeries(Tensor<T> timeSeries)
    {
        var dataVec = timeSeries.ToVector();
        int len = dataVec.Length;

        var trendVec = new T[len];
        var seasonalVec = new T[len];
        var residualVec = new T[len];

        // Extract trend using moving average
        if (_useTrendComponent)
        {
            int halfKernel = _trendKernelSize / 2;
            for (int i = 0; i < len; i++)
            {
                double sum = 0;
                int count = 0;
                for (int j = Math.Max(0, i - halfKernel); j <= Math.Min(len - 1, i + halfKernel); j++)
                {
                    sum += NumOps.ToDouble(dataVec[j]);
                    count++;
                }
                trendVec[i] = NumOps.FromDouble(sum / count);
            }
        }
        else
        {
            // No trend - use zeros
            for (int i = 0; i < len; i++)
            {
                trendVec[i] = NumOps.Zero;
            }
        }

        // Extract seasonal component
        if (_useSeasonalComponent && _decompositionPeriod > 0 && len >= _decompositionPeriod)
        {
            // Compute seasonal averages for each position in the period
            var seasonalAverages = new double[_decompositionPeriod];
            var seasonalCounts = new int[_decompositionPeriod];

            for (int i = 0; i < len; i++)
            {
                int pos = i % _decompositionPeriod;
                double detrendedValue = NumOps.ToDouble(dataVec[i]) - NumOps.ToDouble(trendVec[i]);
                seasonalAverages[pos] += detrendedValue;
                seasonalCounts[pos]++;
            }

            // Normalize and center seasonal values
            double seasonalMean = 0;
            for (int i = 0; i < _decompositionPeriod; i++)
            {
                if (seasonalCounts[i] > 0)
                {
                    seasonalAverages[i] /= seasonalCounts[i];
                    seasonalMean += seasonalAverages[i];
                }
            }
            seasonalMean /= _decompositionPeriod;

            for (int i = 0; i < _decompositionPeriod; i++)
            {
                seasonalAverages[i] -= seasonalMean;
            }

            // Apply seasonal pattern
            for (int i = 0; i < len; i++)
            {
                int pos = i % _decompositionPeriod;
                seasonalVec[i] = NumOps.FromDouble(seasonalAverages[pos]);
            }
        }
        else
        {
            // No seasonality - use zeros
            for (int i = 0; i < len; i++)
            {
                seasonalVec[i] = NumOps.Zero;
            }
        }

        // Residual = original - trend - seasonal
        for (int i = 0; i < len; i++)
        {
            double original = NumOps.ToDouble(dataVec[i]);
            double trend = NumOps.ToDouble(trendVec[i]);
            double seasonal = NumOps.ToDouble(seasonalVec[i]);
            residualVec[i] = NumOps.FromDouble(original - trend - seasonal);
        }

        return (new Tensor<T>(timeSeries.Shape, new Vector<T>(trendVec)),
                new Tensor<T>(timeSeries.Shape, new Vector<T>(seasonalVec)),
                new Tensor<T>(timeSeries.Shape, new Vector<T>(residualVec)));
    }

    /// <summary>
    /// Combines trend, seasonal, and residual components into a single tensor.
    /// </summary>
    /// <param name="trend">Trend component tensor.</param>
    /// <param name="seasonal">Seasonal component tensor.</param>
    /// <param name="residual">Residual component tensor.</param>
    /// <returns>Combined tensor with all three components.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Concatenates the three decomposition components
    /// into a single tensor for processing. During reconstruction, these are
    /// added back together to form the final forecast.
    /// </para>
    /// </remarks>
    private Tensor<T> CombineComponents(Tensor<T> trend, Tensor<T> seasonal, Tensor<T> residual)
    {
        var trendVec = trend.ToVector();
        var seasonalVec = seasonal.ToVector();
        var residualVec = residual.ToVector();

        int len = trendVec.Length;
        var combined = new T[len * 3];

        for (int i = 0; i < len; i++)
        {
            combined[i] = trendVec[i];
            combined[len + i] = seasonalVec[i];
            combined[2 * len + i] = residualVec[i];
        }

        return new Tensor<T>(new[] { len * 3 }, new Vector<T>(combined));
    }

    /// <summary>
    /// Reconstructs the time series from its decomposed components.
    /// </summary>
    /// <param name="trend">Trend component tensor.</param>
    /// <param name="seasonal">Seasonal component tensor.</param>
    /// <param name="residual">Residual component tensor.</param>
    /// <returns>Reconstructed time series tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Simply adds together the trend, seasonal,
    /// and residual components to get the final forecast. This is the inverse
    /// of decomposition.
    /// </para>
    /// </remarks>
    private Tensor<T> ReconstructFromComponents(Tensor<T> trend, Tensor<T> seasonal, Tensor<T> residual)
    {
        var trendVec = trend.ToVector();
        var seasonalVec = seasonal.ToVector();
        var residualVec = residual.ToVector();

        int len = Math.Min(Math.Min(trendVec.Length, seasonalVec.Length), residualVec.Length);
        var reconstructed = new T[len];

        for (int i = 0; i < len; i++)
        {
            double t = NumOps.ToDouble(trendVec[i]);
            double s = NumOps.ToDouble(seasonalVec[i]);
            double r = NumOps.ToDouble(residualVec[i]);
            reconstructed[i] = NumOps.FromDouble(t + s + r);
        }

        return new Tensor<T>(new[] { len }, new Vector<T>(reconstructed));
    }

    #endregion

    #region Diffusion Methods

    /// <summary>
    /// Performs native mode forecasting with interpretable diffusion.
    /// </summary>
    /// <param name="context">Input context tensor containing historical data.</param>
    /// <returns>Forecast tensor with generated values.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Generates forecasts by:
    /// 1. Decomposing input context to estimate component patterns
    /// 2. Initializing noisy trend, seasonal, and residual
    /// 3. Running reverse diffusion separately for each component
    /// 4. Combining denoised components for final forecast
    ///
    /// The result is interpretable - you can see how much each component
    /// contributes to the forecast.
    /// </para>
    /// </remarks>
    private Tensor<T> ForecastNative(Tensor<T> context)
    {
        SetTrainingMode(false);

        int forecastSize = _forecastHorizon * _numFeatures;

        // Initialize each component with noise
        var trendSample = GenerateNoise(forecastSize);
        var seasonalSample = GenerateNoise(forecastSize);
        var residualSample = GenerateNoise(forecastSize);

        // Get context components for conditioning
        var (contextTrend, contextSeasonal, contextResidual) = DecomposeTimeSeries(context);

        // Reverse diffusion for each component
        for (int t = _numDiffusionSteps - 1; t >= 0; t--)
        {
            // Combine all components
            var combined = CombineComponents(trendSample, seasonalSample, residualSample);
            var contextCombined = CombineComponents(contextTrend, contextSeasonal, contextResidual);
            var fullInput = CombineTensors(contextCombined, combined);

            // Predict noise
            var predictedNoise = Forward(fullInput);

            // Denoise each component
            int componentSize = forecastSize;
            var trendNoise = ExtractComponent(predictedNoise, 0, componentSize);
            var seasonalNoise = ExtractComponent(predictedNoise, componentSize, componentSize);
            var residualNoise = ExtractComponent(predictedNoise, 2 * componentSize, componentSize);

            trendSample = DenoisingStep(trendSample, trendNoise, t);
            seasonalSample = DenoisingStep(seasonalSample, seasonalNoise, t);
            residualSample = DenoisingStep(residualSample, residualNoise, t);

            // Apply smoothness constraint to trend
            if (t > 0)
            {
                trendSample = ApplySmoothness(trendSample);
            }
        }

        // Reconstruct final forecast
        return ReconstructFromComponents(trendSample, seasonalSample, residualSample);
    }

    /// <summary>
    /// Performs ONNX mode forecasting using a pretrained model.
    /// </summary>
    /// <param name="input">Input tensor containing historical data.</param>
    /// <returns>Forecast tensor from ONNX inference.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Uses a pretrained ONNX model for fast inference.
    /// The ONNX model contains the trained decomposition networks.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ForecastOnnx(Tensor<T> input)
    {
        if (OnnxSession is null)
            throw new InvalidOperationException("ONNX session not initialized.");

        var flatInput = FlattenInput(input);
        var inputData = new float[flatInput.Data.Length];
        for (int i = 0; i < flatInput.Data.Length; i++)
        {
            inputData[i] = Convert.ToSingle(NumOps.ToDouble(flatInput.Data.Span[i]));
        }

        var inputTensor = new OnnxTensors.DenseTensor<float>(
            inputData,
            new[] { 1, _sequenceLength, _numFeatures });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input", inputTensor)
        };

        using var results = OnnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        var outputData = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++)
        {
            outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));
        }

        return new Tensor<T>(new[] { _forecastHorizon }, new Vector<T>(outputData));
    }

    /// <summary>
    /// Generates multiple forecast samples for uncertainty estimation.
    /// </summary>
    /// <param name="context">Input context tensor.</param>
    /// <param name="numSamples">Number of samples to generate.</param>
    /// <returns>List of forecast samples.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Diffusion is stochastic - different starting noise
    /// leads to different forecasts. This diversity captures uncertainty in
    /// each component (trend, seasonal, residual).
    /// </para>
    /// </remarks>
    private List<Tensor<T>> GenerateSamples(Tensor<T> context, int numSamples)
    {
        var samples = new List<Tensor<T>>();
        for (int i = 0; i < numSamples; i++)
        {
            samples.Add(ForecastNative(context));
        }
        return samples;
    }

    /// <summary>
    /// Adds noise to data at a specific diffusion timestep.
    /// </summary>
    /// <param name="data">Clean data tensor.</param>
    /// <param name="t">Diffusion timestep (0 = clean, max = pure noise).</param>
    /// <returns>Tuple of (noisy data, noise added).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The forward diffusion process adds Gaussian noise
    /// according to the schedule. More noise at higher timesteps progressively
    /// destroys the signal, which the model learns to reverse.
    /// </para>
    /// </remarks>
    private (Tensor<T> noisy, Tensor<T> noise) AddNoise(Tensor<T> data, int t)
    {
        var dataVec = data.ToVector();
        var noiseVec = new T[dataVec.Length];
        var noisyVec = new T[dataVec.Length];

        double sqrtAlpha = _sqrtAlphasCumprod[t];
        double sqrtOneMinusAlpha = _sqrtOneMinusAlphasCumprod[t];

        for (int i = 0; i < dataVec.Length; i++)
        {
            double u1 = _random.NextDouble();
            double u2 = _random.NextDouble();
            double noise = Math.Sqrt(-2.0 * Math.Log(Math.Max(u1, 1e-10))) * Math.Cos(2.0 * Math.PI * u2);
            noiseVec[i] = NumOps.FromDouble(noise);

            double dataVal = NumOps.ToDouble(dataVec[i]);
            double noisyVal = sqrtAlpha * dataVal + sqrtOneMinusAlpha * noise;
            noisyVec[i] = NumOps.FromDouble(noisyVal);
        }

        return (new Tensor<T>(data.Shape, new Vector<T>(noisyVec)),
                new Tensor<T>(data.Shape, new Vector<T>(noiseVec)));
    }

    /// <summary>
    /// Performs one step of the reverse diffusion process.
    /// </summary>
    /// <param name="current">Current noisy state tensor.</param>
    /// <param name="predictedNoise">Predicted noise to remove.</param>
    /// <param name="t">Current timestep.</param>
    /// <returns>Denoised state tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each denoising step removes a bit of the predicted
    /// noise, gradually revealing the clean signal. This is the core of diffusion
    /// models - learning to reverse the noise-adding process.
    /// </para>
    /// </remarks>
    private Tensor<T> DenoisingStep(Tensor<T> current, Tensor<T> predictedNoise, int t)
    {
        var currentVec = current.ToVector();
        var noiseVec = predictedNoise.ToVector();
        var resultVec = new T[currentVec.Length];

        double alpha = _alphas[t];
        double sqrtAlpha = Math.Sqrt(alpha);
        double sqrtOneMinusAlphaBar = _sqrtOneMinusAlphasCumprod[t];

        double coeff = (1 - alpha) / sqrtOneMinusAlphaBar;
        double sigma = t > 0 ? Math.Sqrt(_betas[t]) : 0;

        for (int i = 0; i < currentVec.Length; i++)
        {
            double x_t = NumOps.ToDouble(currentVec[i]);
            double eps = NumOps.ToDouble(noiseVec[i % noiseVec.Length]);

            double mean = (x_t - coeff * eps) / sqrtAlpha;

            double z = 0;
            if (t > 0)
            {
                double u1 = _random.NextDouble();
                double u2 = _random.NextDouble();
                z = Math.Sqrt(-2.0 * Math.Log(Math.Max(u1, 1e-10))) * Math.Cos(2.0 * Math.PI * u2);
            }

            resultVec[i] = NumOps.FromDouble(mean + sigma * z);
        }

        return new Tensor<T>(current.Shape, new Vector<T>(resultVec));
    }

    /// <summary>
    /// Applies smoothness constraint to the trend component.
    /// </summary>
    /// <param name="trend">Trend tensor to smooth.</param>
    /// <returns>Smoothed trend tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Trends should be smooth by definition - they capture
    /// long-term movements, not short-term fluctuations. This constraint helps
    /// ensure the trend component stays interpretable.
    /// </para>
    /// </remarks>
    private Tensor<T> ApplySmoothness(Tensor<T> trend)
    {
        var trendVec = trend.ToVector();
        var smoothed = new T[trendVec.Length];

        // Simple exponential smoothing
        double alpha = 0.3;
        smoothed[0] = trendVec[0];

        for (int i = 1; i < trendVec.Length; i++)
        {
            double prev = NumOps.ToDouble(smoothed[i - 1]);
            double curr = NumOps.ToDouble(trendVec[i]);
            smoothed[i] = NumOps.FromDouble(alpha * curr + (1 - alpha) * prev);
        }

        return new Tensor<T>(trend.Shape, new Vector<T>(smoothed));
    }

    /// <summary>
    /// Generates random Gaussian noise for initialization.
    /// </summary>
    /// <param name="size">Size of noise vector.</param>
    /// <returns>Noise tensor with standard normal values.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Diffusion starts from pure noise and gradually
    /// transforms it into a coherent forecast. This generates the starting noise.
    /// </para>
    /// </remarks>
    private Tensor<T> GenerateNoise(int size)
    {
        var noiseVec = new T[size];

        for (int i = 0; i < size; i++)
        {
            double u1 = _random.NextDouble();
            double u2 = _random.NextDouble();
            double noise = Math.Sqrt(-2.0 * Math.Log(Math.Max(u1, 1e-10))) * Math.Cos(2.0 * Math.PI * u2);
            noiseVec[i] = NumOps.FromDouble(noise);
        }

        return new Tensor<T>(new[] { size }, new Vector<T>(noiseVec));
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Combines two tensors by concatenation.
    /// </summary>
    /// <param name="first">First tensor.</param>
    /// <param name="second">Second tensor.</param>
    /// <returns>Combined tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Concatenates context and sample tensors
    /// for joint processing through the network.
    /// </para>
    /// </remarks>
    private Tensor<T> CombineTensors(Tensor<T> first, Tensor<T> second)
    {
        var firstVec = first.ToVector();
        var secondVec = second.ToVector();
        var combined = new T[firstVec.Length + secondVec.Length];

        for (int i = 0; i < firstVec.Length; i++)
        {
            combined[i] = firstVec[i];
        }
        for (int i = 0; i < secondVec.Length; i++)
        {
            combined[firstVec.Length + i] = secondVec[i];
        }

        return new Tensor<T>(new[] { combined.Length }, new Vector<T>(combined));
    }

    /// <summary>
    /// Extracts a component from a combined tensor.
    /// </summary>
    /// <param name="combined">Combined tensor with all components.</param>
    /// <param name="offset">Starting offset for extraction.</param>
    /// <param name="length">Length of component to extract.</param>
    /// <returns>Extracted component tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> After the network processes combined components,
    /// this extracts the predicted noise for each individual component.
    /// </para>
    /// </remarks>
    private Tensor<T> ExtractComponent(Tensor<T> combined, int offset, int length)
    {
        var combinedVec = combined.ToVector();
        var component = new T[length];

        // If offset is beyond the combined vector, just return zeros
        // This handles cases where the network output is smaller than expected
        if (offset >= combinedVec.Length)
        {
            for (int i = 0; i < length; i++)
            {
                component[i] = NumOps.Zero;
            }
            return new Tensor<T>(new[] { length }, new Vector<T>(component));
        }

        // Calculate safe copy length
        int availableLength = combinedVec.Length - offset;
        int actualLength = Math.Min(length, availableLength);

        for (int i = 0; i < actualLength; i++)
        {
            component[i] = combinedVec[offset + i];
        }

        // Fill remaining with zeros if needed
        for (int i = actualLength; i < length; i++)
        {
            component[i] = NumOps.Zero;
        }

        return new Tensor<T>(new[] { length }, new Vector<T>(component));
    }

    /// <summary>
    /// Shifts input window by appending new prediction.
    /// </summary>
    /// <param name="input">Current input tensor.</param>
    /// <param name="prediction">New prediction to append.</param>
    /// <returns>Shifted input tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> For autoregressive forecasting, removes oldest
    /// values and appends newest predictions to maintain the input window size.
    /// </para>
    /// </remarks>
    private Tensor<T> ShiftInputWindow(Tensor<T> input, Tensor<T> prediction)
    {
        var inputVec = input.ToVector();
        var predVec = prediction.ToVector();

        int windowLen = (_sequenceLength - _forecastHorizon) * _numFeatures;
        var shifted = new T[windowLen];

        int shiftAmount = Math.Min(predVec.Length, windowLen);
        for (int i = 0; i < windowLen - shiftAmount; i++)
        {
            shifted[i] = inputVec[i + shiftAmount];
        }
        for (int i = 0; i < shiftAmount; i++)
        {
            shifted[windowLen - shiftAmount + i] = predVec[i];
        }

        return new Tensor<T>(new[] { windowLen }, new Vector<T>(shifted));
    }

    /// <summary>
    /// Concatenates multiple prediction tensors.
    /// </summary>
    /// <param name="predictions">List of prediction tensors.</param>
    /// <returns>Concatenated tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Combines multiple forecast steps into one
    /// continuous forecast tensor.
    /// </para>
    /// </remarks>
        protected Tensor<T> ConcatenatePredictions(List<Tensor<T>> predictions)
    {
        int totalLen = predictions.Sum(p => p.ToVector().Length);
        var result = new T[totalLen];
        int offset = 0;

        foreach (var pred in predictions)
        {
            var predVec = pred.ToVector();
            for (int i = 0; i < predVec.Length; i++)
            {
                result[offset + i] = predVec[i];
            }
            offset += predVec.Length;
        }

        return new Tensor<T>(new[] { totalLen }, new Vector<T>(result));
    }

    /// <summary>
    /// Computes quantiles from multiple samples.
    /// </summary>
    /// <param name="samples">List of forecast samples.</param>
    /// <param name="quantiles">Quantile levels to compute (e.g., [0.1, 0.5, 0.9]).</param>
    /// <returns>Tensor with quantile values.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Sorts samples at each position and picks
    /// the values at specified percentiles for uncertainty estimation.
    /// </para>
    /// </remarks>
    private Tensor<T> ComputeQuantiles(List<Tensor<T>> samples, double[] quantiles)
    {
        if (samples.Count == 0)
            return new Tensor<T>(new[] { 0 });

        int len = samples[0].ToVector().Length;
        var result = new T[len * quantiles.Length];

        for (int pos = 0; pos < len; pos++)
        {
            var values = samples.Select(s => NumOps.ToDouble(s.ToVector()[pos])).OrderBy(v => v).ToList();

            for (int q = 0; q < quantiles.Length; q++)
            {
                int idx = (int)(quantiles[q] * (values.Count - 1));
                result[q * len + pos] = NumOps.FromDouble(values[idx]);
            }
        }

        return new Tensor<T>(new[] { quantiles.Length, len }, new Vector<T>(result));
    }

    /// <summary>
    /// Computes prediction intervals from multiple samples.
    /// </summary>
    /// <param name="samples">List of forecast samples.</param>
    /// <param name="confidenceLevel">Confidence level (e.g., 0.95).</param>
    /// <returns>Tuple of (median forecast, lower bound, upper bound).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Uses the sample distribution to compute
    /// confidence intervals. A 95% interval means 95% of samples fall within bounds.
    /// </para>
    /// </remarks>
    private (Tensor<T> Forecast, Tensor<T> Lower, Tensor<T> Upper) ComputePredictionIntervals(
        List<Tensor<T>> samples,
        double confidenceLevel)
    {
        double alpha = 1 - confidenceLevel;
        double[] quantiles = { alpha / 2, 0.5, 1 - alpha / 2 };

        var quantileResult = ComputeQuantiles(samples, quantiles);
        var resultVec = quantileResult.ToVector();

        int len = samples[0].ToVector().Length;
        var lowerVec = new T[len];
        var medianVec = new T[len];
        var upperVec = new T[len];

        for (int i = 0; i < len; i++)
        {
            lowerVec[i] = resultVec[i];
            medianVec[i] = resultVec[len + i];
            upperVec[i] = resultVec[2 * len + i];
        }

        var shape = new[] { len };
        return (new Tensor<T>(shape, new Vector<T>(medianVec)),
                new Tensor<T>(shape, new Vector<T>(lowerVec)),
                new Tensor<T>(shape, new Vector<T>(upperVec)));
    }

    #endregion

    #region IDisposable

    /// <summary>
    /// Disposes of managed and unmanaged resources.
    /// </summary>
    /// <param name="disposing">Whether to dispose managed resources.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Releases the ONNX session and other resources
    /// when the model is no longer needed.
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



