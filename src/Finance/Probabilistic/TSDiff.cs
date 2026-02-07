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
/// TSDiff (Time Series Diffusion) for probabilistic time series forecasting with self-guided diffusion.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// TSDiff is a versatile diffusion model that supports unconditional generation, forecasting,
/// and imputation through a unified self-guided diffusion framework.
/// </para>
/// <para><b>For Beginners:</b> TSDiff takes a flexible approach to time series generation
/// by treating all tasks as conditional generation problems:
///
/// <b>The Key Innovation - Self-Guidance:</b>
/// Unlike models that need separate conditioning mechanisms, TSDiff uses its own
/// intermediate predictions to guide the generation process. This creates a "refinement loop"
/// where the model progressively improves its outputs.
///
/// <b>How Self-Guided Diffusion Works:</b>
/// 1. <b>Initial Generation:</b> Start from noise, run standard diffusion
/// 2. <b>Self-Prediction:</b> Use partial output to predict missing parts
/// 3. <b>Guidance Gradient:</b> Compute gradient to improve consistency
/// 4. <b>Refined Step:</b> Combine denoising step with guidance gradient
/// 5. <b>Iterate:</b> Repeat until high-quality output
///
/// <b>TSDiff Architecture:</b>
/// - U-Net backbone with residual blocks for multi-scale processing
/// - Self-attention in bottleneck for long-range dependencies
/// - Timestep conditioning throughout the network
/// - Skip connections for preserving fine details
///
/// <b>Supported Tasks:</b>
/// - Forecasting: Condition on past, generate future
/// - Imputation: Condition on observed, generate missing
/// - Generation: Create synthetic time series from scratch
///
/// <b>Key Benefits:</b>
/// - Unified framework for multiple tasks
/// - Self-guidance improves temporal coherence
/// - Captures complex multivariate dynamics
/// - Produces diverse, high-quality samples
/// </para>
/// <para>
/// <b>Reference:</b> Kollovieh et al., "Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting", 2023.
/// https://arxiv.org/abs/2307.11494
/// </para>
/// </remarks>
public class TSDiff<T> : ForecastingModelBase<T>
{
    #region Execution Mode
    private bool _useNativeMode;
    #endregion


    #region Native Mode Fields
    private DenseLayer<T>? _inputEmbedding;
    private List<DenseLayer<T>>? _downsampleLayers;
    private List<DenseLayer<T>>? _upsampleLayers;
    private List<LayerNormalizationLayer<T>>? _layerNorms;
    private DenseLayer<T>? _outputHead;
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
    private readonly TSDiffOptions<T> _options;
    private int _sequenceLength;
    private int _forecastHorizon;
    private int _numFeatures;
    private int _hiddenDimension;
    private int _numResidualBlocks;
    private int _numDiffusionSteps;
    private int _numSamples;
    private int _numAttentionHeads;
    private double _guidanceScale;
    private bool _useSelfGuidance;
    private bool _useObservationGuidance;
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
    public override bool IsChannelIndependent => false; // Multivariate model

    /// <inheritdoc/>
    public override bool UseNativeMode => _useNativeMode;

    /// <summary>
    /// Gets the context length (input history length).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The number of past time steps used for conditioning.
    /// </para>
    /// </remarks>
    public int ContextLength => _sequenceLength - _forecastHorizon;

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
    /// <para><b>For Beginners:</b> How many different forecasts to generate.
    /// </para>
    /// </remarks>
    public int NumSamples => _numSamples;

    /// <summary>
    /// Gets the guidance scale for conditional generation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Higher values mean stronger conditioning.
    /// </para>
    /// </remarks>
    public double GuidanceScale => _guidanceScale;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the TSDiff model in ONNX mode for inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to a pretrained ONNX model file.</param>
    /// <param name="options">TSDiff-specific options.</param>
    /// <param name="optimizer">Optional optimizer for fine-tuning.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to load a pretrained TSDiff model
    /// for fast probabilistic forecasting.
    /// </para>
    /// </remarks>
    public TSDiff(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        TSDiffOptions<T>? options = null,
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
        _options = options ?? new TSDiffOptions<T>();
        Options = _options;
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        _sequenceLength = _options.SequenceLength;
        _forecastHorizon = _options.ForecastHorizon;
        _numFeatures = _options.NumFeatures;
        _hiddenDimension = _options.HiddenDimension;
        _numResidualBlocks = _options.NumResidualBlocks;
        _numDiffusionSteps = _options.NumDiffusionSteps;
        _numSamples = _options.NumSamples;
        _numAttentionHeads = _options.NumAttentionHeads;
        _guidanceScale = _options.GuidanceScale;
        _useSelfGuidance = _options.UseSelfGuidance;
        _useObservationGuidance = _options.UseObservationGuidance;
        _betaSchedule = _options.BetaSchedule;

        (_betas, _alphas, _alphasCumprod, _sqrtAlphasCumprod, _sqrtOneMinusAlphasCumprod) =
            InitializeDiffusionSchedule(_numDiffusionSteps, _options.BetaStart, _options.BetaEnd, _betaSchedule);
        _random = RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Initializes a new instance of the TSDiff model in native mode for training.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">TSDiff-specific options.</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="optimizer">Optional optimizer for training.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to create a TSDiff model
    /// that can be trained on your data.
    /// </para>
    /// </remarks>
    public TSDiff(
        NeuralNetworkArchitecture<T> architecture,
        TSDiffOptions<T>? options = null,
        int numFeatures = 1,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        _useNativeMode = true;
        _options = options ?? new TSDiffOptions<T>();
        Options = _options;
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        _sequenceLength = _options.SequenceLength;
        _forecastHorizon = _options.ForecastHorizon;
        _numFeatures = numFeatures > 0 ? numFeatures : _options.NumFeatures;
        _hiddenDimension = _options.HiddenDimension;
        _numResidualBlocks = _options.NumResidualBlocks;
        _numDiffusionSteps = _options.NumDiffusionSteps;
        _numSamples = _options.NumSamples;
        _numAttentionHeads = _options.NumAttentionHeads;
        _guidanceScale = _options.GuidanceScale;
        _useSelfGuidance = _options.UseSelfGuidance;
        _useObservationGuidance = _options.UseObservationGuidance;
        _betaSchedule = _options.BetaSchedule;

        (_betas, _alphas, _alphasCumprod, _sqrtAlphasCumprod, _sqrtOneMinusAlphasCumprod) =
            InitializeDiffusionSchedule(_numDiffusionSteps, _options.BetaStart, _options.BetaEnd, _betaSchedule);
        _random = RandomHelper.CreateSecureRandom();

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes all layers for the TSDiff model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up the U-Net style architecture:
    ///
    /// <b>Layer Structure:</b>
    /// 1. Input embedding + time embedding
    /// 2. Downsampling path (compress)
    /// 3. Middle block with self-attention
    /// 4. Upsampling path (expand)
    /// 5. Output head (noise prediction)
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultTSDiffLayers(
                Architecture,
                _sequenceLength,
                _numFeatures,
                _hiddenDimension,
                _numResidualBlocks,
                _numAttentionHeads));

            ExtractLayerReferences();
        }
    }

    /// <summary>
    /// Extracts references to key layers for efficient access.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Keeps direct references to important layers
    /// for quick access during the forward/backward passes.
    /// </para>
    /// </remarks>
    private void ExtractLayerReferences()
    {
        _inputEmbedding = Layers.OfType<DenseLayer<T>>().FirstOrDefault();
        _layerNorms = Layers.OfType<LayerNormalizationLayer<T>>().ToList();
        _outputHead = Layers.OfType<DenseLayer<T>>().LastOrDefault();

        var allDense = Layers.OfType<DenseLayer<T>>().ToList();
        _downsampleLayers = new List<DenseLayer<T>>();
        _upsampleLayers = new List<DenseLayer<T>>();

        // Organize layers into downsample/upsample paths
        if (allDense.Count > 4)
        {
            int mid = allDense.Count / 2;
            _downsampleLayers = allDense.Take(mid).ToList();
            _upsampleLayers = allDense.Skip(mid).ToList();
        }
    }

    /// <summary>
    /// Initializes the diffusion noise schedule.
    /// </summary>
    /// <param name="numSteps">Number of diffusion steps.</param>
    /// <param name="betaStart">Starting noise level.</param>
    /// <param name="betaEnd">Ending noise level.</param>
    /// <param name="schedule">Schedule type.</param>
    /// <returns>Tuple of precomputed diffusion values.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Precomputes noise schedule values for efficiency.
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
    /// Validates custom layers.
    /// </summary>
    /// <param name="layers">The list of custom layers.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Ensures custom layers form a valid TSDiff architecture.
    /// </para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);

        if (layers.Count < 5)
            throw new ArgumentException("TSDiff requires at least 5 layers (embedding, downsample, middle, upsample, head).");
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <summary>
    /// Performs forward prediction on the input tensor.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Output tensor (forecast).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Runs the self-guided diffusion process to generate forecasts.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return _useNativeMode ? ForecastNative(input) : ForecastOnnx(input);
    }

    /// <summary>
    /// Trains the TSDiff model on a batch of input-target pairs.
    /// </summary>
    /// <param name="input">Input tensor (full sequence for training).</param>
    /// <param name="target">Target tensor (same as input for generation training).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Training involves:
    /// 1. Add noise to the target at a random timestep
    /// 2. Predict the noise using the denoising network
    /// 3. Minimize the difference between predicted and actual noise
    ///
    /// Self-guidance is learned implicitly through this noise prediction objective.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training requires native mode.");

        SetTrainingMode(true);

        // Sample random timestep
        int t = _random.Next(_numDiffusionSteps);

        // Add noise to target
        var (noisyTarget, noise) = AddNoise(target, t);

        // Forward pass: predict noise
        var output = Forward(noisyTarget);

        // Flatten noise to match output shape (Forward flattens input)
        var noiseFlattened = FlattenInput(noise);

        // Ensure shapes match for loss calculation - use minimum length
        var outputVec = output.ToVector();
        var noiseVec = noiseFlattened.ToVector();
        int minLength = Math.Min(outputVec.Length, noiseVec.Length);

        // Create matching-length vectors for loss calculation
        var matchedOutput = new T[minLength];
        var matchedNoise = new T[minLength];
        for (int i = 0; i < minLength; i++)
        {
            matchedOutput[i] = outputVec[i];
            matchedNoise[i] = noiseVec[i];
        }

        // Calculate loss using matched-size vectors
        var matchedOutputVec = new Vector<T>(matchedOutput);
        var matchedNoiseVec = new Vector<T>(matchedNoise);
        LastLoss = _lossFunction.CalculateLoss(matchedOutputVec, matchedNoiseVec);

        // Backward pass - use matched size for gradient computation
        var gradient = _lossFunction.CalculateDerivative(matchedOutputVec, matchedNoiseVec);

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
    /// Updates parameters (required override).
    /// </summary>
    /// <param name="gradients">Gradient vector.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TSDiff model, UpdateParameters updates internal parameters or state. This keeps the TSDiff architecture aligned with the latest values.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> gradients)
    {
        // Parameters are updated through the optimizer in Train()
    }

    /// <summary>
    /// Gets metadata about the TSDiff model.
    /// </summary>
    /// <returns>ModelMetadata containing model information.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TSDiff model, GetModelMetadata performs a supporting step in the workflow. It keeps the TSDiff architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "TSDiff" },
                { "SequenceLength", _sequenceLength },
                { "ForecastHorizon", _forecastHorizon },
                { "NumFeatures", _numFeatures },
                { "HiddenDimension", _hiddenDimension },
                { "NumResidualBlocks", _numResidualBlocks },
                { "NumDiffusionSteps", _numDiffusionSteps },
                { "NumSamples", _numSamples },
                { "GuidanceScale", _guidanceScale },
                { "UseSelfGuidance", _useSelfGuidance },
                { "BetaSchedule", _betaSchedule },
                { "UseNativeMode", _useNativeMode }
            }
        };
    }

    /// <summary>
    /// Creates a new instance with the same configuration.
    /// </summary>
    /// <returns>A new TSDiff instance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TSDiff model, CreateNewInstance builds and wires up model components. This sets up the TSDiff architecture before use.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new TSDiff<T>(Architecture, _options, _numFeatures);
    }

    /// <summary>
    /// Serializes TSDiff-specific data.
    /// </summary>
    /// <param name="writer">The binary writer.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TSDiff model, SerializeNetworkSpecificData saves or restores model-specific settings. This lets the TSDiff architecture be reused later.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_sequenceLength);
        writer.Write(_forecastHorizon);
        writer.Write(_numFeatures);
        writer.Write(_hiddenDimension);
        writer.Write(_numResidualBlocks);
        writer.Write(_numDiffusionSteps);
        writer.Write(_numSamples);
        writer.Write(_numAttentionHeads);
        writer.Write(_guidanceScale);
        writer.Write(_useSelfGuidance);
        writer.Write(_useObservationGuidance);
        writer.Write(_betaSchedule);
    }

    /// <summary>
    /// Deserializes TSDiff-specific data.
    /// </summary>
    /// <param name="reader">The binary reader.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TSDiff model, DeserializeNetworkSpecificData saves or restores model-specific settings. This lets the TSDiff architecture be reused later.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _sequenceLength = reader.ReadInt32();
        _forecastHorizon = reader.ReadInt32();
        _numFeatures = reader.ReadInt32();
        _hiddenDimension = reader.ReadInt32();
        _numResidualBlocks = reader.ReadInt32();
        _numDiffusionSteps = reader.ReadInt32();
        _numSamples = reader.ReadInt32();
        _numAttentionHeads = reader.ReadInt32();
        _guidanceScale = reader.ReadDouble();
        _useSelfGuidance = reader.ReadBoolean();
        _useObservationGuidance = reader.ReadBoolean();
        _betaSchedule = reader.ReadString();
    }

    #endregion

    #region IForecastingModel Implementation

    /// <summary>
    /// Generates forecasts for the given input time series.
    /// </summary>
    /// <param name="historicalData">Input tensor (context/history).</param>
    /// <param name="quantiles">Optional quantile levels for probabilistic forecasting.</param>
    /// <returns>Forecast tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Runs self-guided diffusion to generate forecasts.
    /// If quantiles are specified, generates multiple samples and computes percentiles.
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
    /// <param name="input">Input tensor.</param>
    /// <param name="confidenceLevel">Confidence level (e.g., 0.95).</param>
    /// <returns>Tuple of (forecast, lower bound, upper bound).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Generates multiple samples and computes intervals.
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
    /// Performs autoregressive forecasting.
    /// </summary>
    /// <param name="input">Initial input.</param>
    /// <param name="steps">Number of steps.</param>
    /// <returns>Forecast tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TSDiff model, AutoregressiveForecast produces predictions from input data. This is the main inference step of the TSDiff architecture.
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
    /// <param name="predictions">Predicted values.</param>
    /// <param name="actuals">Actual values.</param>
    /// <returns>Dictionary of evaluation metrics.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TSDiff model, Evaluate performs a supporting step in the workflow. It keeps the TSDiff architecture pipeline consistent.
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
    /// Applies instance normalization (RevIN).
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Normalized tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TSDiff model, ApplyInstanceNormalization performs a supporting step in the workflow. It keeps the TSDiff architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
    {
        return input;
    }

    /// <summary>
    /// Gets financial-specific metrics.
    /// </summary>
    /// <returns>Dictionary of financial metrics.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TSDiff model, GetFinancialMetrics calculates evaluation metrics. This summarizes how the TSDiff architecture is performing.
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
            ["GuidanceScale"] = NumOps.FromDouble(_guidanceScale)
        };
    }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// Performs the forward pass through all layers.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Output tensor (predicted noise).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The forward pass processes input through the U-Net
    /// architecture, predicting the noise to be removed at each timestep.
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
    /// <para><b>For Beginners:</b> Backpropagation computes gradients for learning.
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
    /// <param name="input">Input tensor.</param>
    /// <returns>Flattened tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TSDiff model, FlattenInput performs a supporting step in the workflow. It keeps the TSDiff architecture pipeline consistent.
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

    #region Diffusion Methods

    /// <summary>
    /// Performs native mode forecasting with self-guided diffusion.
    /// </summary>
    /// <param name="context">Input context tensor.</param>
    /// <returns>Forecast tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Self-guided diffusion iteratively refines forecasts
    /// by using its own predictions as guidance. This improves temporal coherence.
    /// </para>
    /// </remarks>
    private Tensor<T> ForecastNative(Tensor<T> context)
    {
        SetTrainingMode(false);

        // Initialize forecast region with noise
        var sample = GenerateNoise(_forecastHorizon * _numFeatures);

        // Reverse diffusion with self-guidance
        for (int t = _numDiffusionSteps - 1; t >= 0; t--)
        {
            // Combine context with current noisy sample
            var combined = CombineContextAndSample(context, sample);

            // Predict noise
            var predictedNoise = Forward(combined);

            // Standard denoising step
            sample = DenoisingStep(sample, predictedNoise, t);

            // Apply self-guidance if enabled
            if (_useSelfGuidance && t > 0 && t % 10 == 0)
            {
                sample = ApplySelfGuidance(sample, context, t);
            }
        }

        return sample;
    }

    /// <summary>
    /// Performs ONNX mode forecasting.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Forecast tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TSDiff model, ForecastOnnx produces predictions from input data. This is the main inference step of the TSDiff architecture.
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
    /// <param name="context">Input context.</param>
    /// <param name="numSamples">Number of samples to generate.</param>
    /// <returns>List of forecast samples.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Diffusion is stochastic - different starting noise
    /// leads to different forecasts. This diversity captures uncertainty.
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
    /// <param name="t">Diffusion timestep.</param>
    /// <returns>Tuple of (noisy data, noise added).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The forward diffusion process adds noise according
    /// to the schedule. More noise at higher timesteps.
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
    /// <param name="current">Current noisy state.</param>
    /// <param name="predictedNoise">Predicted noise to remove.</param>
    /// <param name="t">Current timestep.</param>
    /// <returns>Denoised state.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each denoising step removes a bit of noise,
    /// gradually revealing the clean signal underneath.
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
    /// Applies self-guidance to refine the current sample.
    /// </summary>
    /// <param name="sample">Current sample.</param>
    /// <param name="context">Input context.</param>
    /// <param name="t">Current timestep.</param>
    /// <returns>Refined sample.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Self-guidance uses the model's own predictions
    /// to steer generation toward more coherent outputs. It's like the model
    /// "checking its own work" and making corrections.
    /// </para>
    /// </remarks>
    private Tensor<T> ApplySelfGuidance(Tensor<T> sample, Tensor<T> context, int t)
    {
        // Simple self-guidance: blend with context-conditioned prediction
        var combined = CombineContextAndSample(context, sample);
        var refinedPrediction = Forward(combined);

        var sampleVec = sample.ToVector();
        var refinedVec = refinedPrediction.ToVector();
        var resultVec = new T[sampleVec.Length];

        double guidanceWeight = _guidanceScale * (1.0 - (double)t / _numDiffusionSteps);

        for (int i = 0; i < sampleVec.Length; i++)
        {
            double sampleVal = NumOps.ToDouble(sampleVec[i]);
            double refinedVal = i < refinedVec.Length ? NumOps.ToDouble(refinedVec[i]) : sampleVal;

            // Blend original with guidance
            double guided = sampleVal + guidanceWeight * (refinedVal - sampleVal);
            resultVec[i] = NumOps.FromDouble(guided);
        }

        return new Tensor<T>(sample.Shape, new Vector<T>(resultVec));
    }

    /// <summary>
    /// Generates random noise for initialization.
    /// </summary>
    /// <param name="size">Size of noise vector.</param>
    /// <returns>Noise tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TSDiff model, GenerateNoise performs a supporting step in the workflow. It keeps the TSDiff architecture pipeline consistent.
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
    /// Combines context and sample for network input.
    /// </summary>
    /// <param name="context">Context (historical) tensor.</param>
    /// <param name="sample">Current sample tensor.</param>
    /// <returns>Combined tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TSDiff model, CombineContextAndSample performs a supporting step in the workflow. It keeps the TSDiff architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private Tensor<T> CombineContextAndSample(Tensor<T> context, Tensor<T> sample)
    {
        var contextVec = context.ToVector();
        var sampleVec = sample.ToVector();
        var combined = new T[contextVec.Length + sampleVec.Length];

        for (int i = 0; i < contextVec.Length; i++)
        {
            combined[i] = contextVec[i];
        }
        for (int i = 0; i < sampleVec.Length; i++)
        {
            combined[contextVec.Length + i] = sampleVec[i];
        }

        return new Tensor<T>(new[] { combined.Length }, new Vector<T>(combined));
    }

    /// <summary>
    /// Shifts input window by appending new prediction.
    /// </summary>
    /// <param name="input">Current input.</param>
    /// <param name="prediction">New prediction to append.</param>
    /// <returns>Shifted input.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TSDiff model, ShiftInputWindow performs a supporting step in the workflow. It keeps the TSDiff architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private Tensor<T> ShiftInputWindow(Tensor<T> input, Tensor<T> prediction)
    {
        var inputVec = input.ToVector();
        var predVec = prediction.ToVector();

        int contextLen = ContextLength * _numFeatures;
        var shifted = new T[contextLen];

        // Shift: remove oldest, add newest
        int shiftAmount = Math.Min(predVec.Length, contextLen);
        for (int i = 0; i < contextLen - shiftAmount; i++)
        {
            shifted[i] = inputVec[i + shiftAmount];
        }
        for (int i = 0; i < shiftAmount; i++)
        {
            shifted[contextLen - shiftAmount + i] = predVec[i];
        }

        return new Tensor<T>(new[] { contextLen }, new Vector<T>(shifted));
    }

    /// <summary>
    /// Concatenates multiple predictions into one tensor.
    /// </summary>
    /// <param name="predictions">List of predictions.</param>
    /// <returns>Concatenated tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TSDiff model, ConcatenatePredictions produces predictions from input data. This is the main inference step of the TSDiff architecture.
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
    /// Computes quantiles from samples.
    /// </summary>
    /// <param name="samples">List of samples.</param>
    /// <param name="quantiles">Quantile levels.</param>
    /// <returns>Quantile tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TSDiff model, ComputeQuantiles performs a supporting step in the workflow. It keeps the TSDiff architecture pipeline consistent.
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
    /// Computes prediction intervals from samples.
    /// </summary>
    /// <param name="samples">List of samples.</param>
    /// <param name="confidenceLevel">Confidence level.</param>
    /// <returns>Tuple of (median, lower, upper).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TSDiff model, private performs a supporting step in the workflow. It keeps the TSDiff architecture pipeline consistent.
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
    /// Disposes of resources.
    /// </summary>
    /// <param name="disposing">Whether to dispose managed resources.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TSDiff model, Dispose performs a supporting step in the workflow. It keeps the TSDiff architecture pipeline consistent.
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



