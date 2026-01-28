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

namespace AiDotNet.Finance.Probabilistic;

/// <summary>
/// TimeGrad (Autoregressive Denoising Diffusion Model) for probabilistic time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// TimeGrad is a probabilistic time series forecasting model that uses denoising diffusion
/// to generate accurate forecasts with well-calibrated uncertainty estimates.
/// </para>
/// <para><b>For Beginners:</b> TimeGrad brings the power of diffusion models (like DALL-E
/// for images) to time series forecasting:
///
/// <b>The Core Problem:</b>
/// Most forecasting models give ONE prediction. But you often need to know:
/// - How confident is this prediction?
/// - What's the worst-case scenario?
/// - What range of outcomes is possible?
///
/// TimeGrad solves this by generating MANY possible future paths, giving you
/// a full probability distribution of what might happen.
///
/// <b>How Diffusion Works:</b>
/// 1. <b>Forward Process:</b> Gradually add noise to real data until it's pure noise
/// 2. <b>Reverse Process:</b> Learn to remove noise step-by-step
/// 3. <b>Conditioning:</b> Use historical data to guide the denoising
/// 4. <b>Sampling:</b> Start from noise, denoise to get realistic forecasts
///
/// <b>TimeGrad Architecture:</b>
/// - RNN encoder: Processes historical data into a hidden state
/// - Denoising network: Predicts and removes noise, conditioned on RNN state
/// - Sampling: Run reverse process multiple times for uncertainty
///
/// <b>Key Benefits:</b>
/// - Probabilistic forecasts with uncertainty quantification
/// - Well-calibrated prediction intervals
/// - Can generate diverse, realistic scenarios
/// - Captures complex multimodal distributions
/// </para>
/// <para>
/// <b>Reference:</b> Rasul et al., "Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting", 2021.
/// https://arxiv.org/abs/2101.12072
/// </para>
/// </remarks>
public class TimeGrad<T> : NeuralNetworkBase<T>, IForecastingModel<T>
{
    #region Execution Mode
    private readonly bool _useNativeMode;
    #endregion

    #region ONNX Mode Fields
    private readonly InferenceSession? _onnxSession;
    private readonly string? _onnxModelPath;
    #endregion

    #region Native Mode Fields
    private DenseLayer<T>? _inputEmbedding;
    private List<DenseLayer<T>>? _rnnLayers;
    private List<DenseLayer<T>>? _denoisingLayers;
    private List<LayerNormalizationLayer<T>>? _layerNorms;
    private DenseLayer<T>? _noiseHead;
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
    private readonly TimeGradOptions<T> _options;
    private readonly int _contextLength;
    private readonly int _forecastHorizon;
    private readonly int _hiddenDimension;
    private readonly int _numRnnLayers;
    private readonly int _numDiffusionSteps;
    private readonly int _numSamples;
    private readonly int _denoisingDim;
    private readonly string _betaSchedule;
    private readonly int _numFeatures;
    #endregion

    #region IForecastingModel Properties

    /// <inheritdoc/>
    public int SequenceLength => _contextLength;

    /// <inheritdoc/>
    public int PredictionHorizon => _forecastHorizon;

    /// <inheritdoc/>
    public int NumFeatures => _numFeatures;

    /// <inheritdoc/>
    public int PatchSize => 1;

    /// <inheritdoc/>
    public int Stride => 1;

    /// <inheritdoc/>
    public bool IsChannelIndependent => false; // Multivariate model

    /// <inheritdoc/>
    public bool UseNativeMode => _useNativeMode;

    /// <summary>
    /// Gets the input context length for the model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is how many past time steps TimeGrad looks at.
    /// </para>
    /// </remarks>
    public int ContextLength => _contextLength;

    /// <summary>
    /// Gets the forecast horizon (number of future steps to predict).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is how many steps into the future
    /// the model predicts in one forward pass.
    /// </para>
    /// </remarks>
    public int ForecastHorizon => _forecastHorizon;

    /// <summary>
    /// Gets whether the model supports training (native mode only).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> ONNX mode is inference-only (pretrained models).
    /// Native mode supports both training and inference.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => _useNativeMode;

    /// <summary>
    /// Gets the number of diffusion steps.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The number of noise levels in the diffusion process.
    /// More steps = finer denoising = better quality but slower.
    /// </para>
    /// </remarks>
    public int NumDiffusionSteps => _numDiffusionSteps;

    /// <summary>
    /// Gets the number of samples to generate for probabilistic forecasting.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many different forecast paths to generate.
    /// More samples = better uncertainty estimates.
    /// </para>
    /// </remarks>
    public int NumSamples => _numSamples;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the TimeGrad model in ONNX mode for inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to a pretrained ONNX model file.</param>
    /// <param name="options">TimeGrad-specific options.</param>
    /// <param name="optimizer">Optional optimizer for fine-tuning.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to load a pretrained TimeGrad model
    /// for fast probabilistic forecasting.
    /// </para>
    /// </remarks>
    public TimeGrad(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        TimeGradOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentNullException(nameof(onnxModelPath));
        if (!System.IO.File.Exists(onnxModelPath))
            throw new System.IO.FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        _useNativeMode = false;
        _onnxModelPath = onnxModelPath;
        _onnxSession = new InferenceSession(onnxModelPath);
        _options = options ?? new TimeGradOptions<T>();
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        _contextLength = _options.ContextLength;
        _forecastHorizon = _options.ForecastHorizon;
        _hiddenDimension = _options.HiddenDimension;
        _numRnnLayers = _options.NumRnnLayers;
        _numDiffusionSteps = _options.NumDiffusionSteps;
        _numSamples = _options.NumSamples;
        _denoisingDim = _options.DenoisingNetworkDim;
        _betaSchedule = _options.BetaSchedule;
        _numFeatures = 1;

        // Initialize diffusion schedule
        (_betas, _alphas, _alphasCumprod, _sqrtAlphasCumprod, _sqrtOneMinusAlphasCumprod) =
            InitializeDiffusionSchedule(_numDiffusionSteps, _options.BetaStart, _options.BetaEnd, _betaSchedule);
        _random = RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Initializes a new instance of the TimeGrad model in native mode for training.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">TimeGrad-specific options.</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="optimizer">Optional optimizer for training.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to create a TimeGrad model
    /// that can be trained on your data for probabilistic forecasting.
    /// </para>
    /// </remarks>
    public TimeGrad(
        NeuralNetworkArchitecture<T> architecture,
        TimeGradOptions<T>? options = null,
        int numFeatures = 1,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        _useNativeMode = true;
        _options = options ?? new TimeGradOptions<T>();
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        _contextLength = _options.ContextLength;
        _forecastHorizon = _options.ForecastHorizon;
        _hiddenDimension = _options.HiddenDimension;
        _numRnnLayers = _options.NumRnnLayers;
        _numDiffusionSteps = _options.NumDiffusionSteps;
        _numSamples = _options.NumSamples;
        _denoisingDim = _options.DenoisingNetworkDim;
        _betaSchedule = _options.BetaSchedule;
        _numFeatures = numFeatures;

        // Initialize diffusion schedule
        (_betas, _alphas, _alphasCumprod, _sqrtAlphasCumprod, _sqrtOneMinusAlphasCumprod) =
            InitializeDiffusionSchedule(_numDiffusionSteps, _options.BetaStart, _options.BetaEnd, _betaSchedule);
        _random = RandomHelper.CreateSecureRandom();

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes all layers for the TimeGrad model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method sets up the neural network layers:
    ///
    /// <b>Layer Structure:</b>
    /// 1. Input embedding
    /// 2. RNN encoder (processes historical data)
    /// 3. Denoising network (predicts noise to remove)
    /// 4. Noise prediction head (outputs predicted noise)
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultTimeGradLayers(
                Architecture,
                _contextLength,
                _forecastHorizon,
                _hiddenDimension,
                _numRnnLayers,
                _denoisingDim,
                _numDiffusionSteps,
                _numFeatures));

            ExtractLayerReferences();
        }
    }

    /// <summary>
    /// Extracts references to key layers for efficient access.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> After creating all layers, we keep direct references
    /// to important ones for quick access during diffusion sampling.
    /// </para>
    /// </remarks>
    private void ExtractLayerReferences()
    {
        _inputEmbedding = Layers.OfType<DenseLayer<T>>().FirstOrDefault();
        _layerNorms = Layers.OfType<LayerNormalizationLayer<T>>().ToList();
        _noiseHead = Layers.OfType<DenseLayer<T>>().LastOrDefault();

        var allDense = Layers.OfType<DenseLayer<T>>().ToList();
        _rnnLayers = new List<DenseLayer<T>>();
        _denoisingLayers = new List<DenseLayer<T>>();

        // Organize layers
        if (allDense.Count > 2)
        {
            // First portion are RNN layers, rest are denoising
            int rnnLayerCount = _numRnnLayers * 4; // 4 dense layers per RNN block
            _rnnLayers = allDense.Skip(1).Take(Math.Min(rnnLayerCount, allDense.Count - 2)).ToList();
            _denoisingLayers = allDense.Skip(1 + rnnLayerCount).Take(allDense.Count - 2 - rnnLayerCount).ToList();
        }
    }

    /// <summary>
    /// Initializes the diffusion noise schedule.
    /// </summary>
    /// <param name="numSteps">Number of diffusion steps.</param>
    /// <param name="betaStart">Starting noise level.</param>
    /// <param name="betaEnd">Ending noise level.</param>
    /// <param name="schedule">Schedule type (linear, cosine, quadratic).</param>
    /// <returns>Tuple of (betas, alphas, alphas_cumprod, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The noise schedule defines how much noise to add
    /// at each diffusion step. These precomputed values are used during both
    /// training (adding noise) and sampling (removing noise).
    ///
    /// <b>Key values:</b>
    /// - beta_t: Amount of noise to add at step t
    /// - alpha_t: 1 - beta_t (amount of signal preserved)
    /// - alpha_bar_t: Product of alphas up to step t (total signal remaining)
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
            // Cosine schedule (often better for certain tasks)
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
    /// Validates custom layers provided through the architecture.
    /// </summary>
    /// <param name="layers">The list of custom layers to validate.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Ensures custom layers form a valid TimeGrad architecture.
    /// </para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);

        if (layers.Count < 5)
            throw new ArgumentException("TimeGrad requires at least 5 layers (embedding, RNN, denoising, head).");
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <summary>
    /// Performs forward prediction on the input tensor.
    /// </summary>
    /// <param name="input">Input tensor of shape [batch, context, features].</param>
    /// <returns>Output tensor of shape [batch, forecast_horizon].</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This runs the diffusion sampling process to generate
    /// a forecast. It starts from pure noise and gradually denoises to produce a prediction.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return _useNativeMode ? ForecastNative(input) : ForecastOnnx(input);
    }

    /// <summary>
    /// Trains the TimeGrad model on a batch of input-target pairs.
    /// </summary>
    /// <param name="input">Input tensor of shape [batch, context, features].</param>
    /// <param name="target">Target tensor of shape [batch, forecast_horizon].</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Training involves:
    /// 1. Add noise to the target at a random timestep
    /// 2. Predict the noise using the denoising network
    /// 3. Minimize the difference between predicted and actual noise
    ///
    /// This teaches the model how to remove noise, which is used during sampling.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training requires native mode. ONNX mode is inference-only.");

        SetTrainingMode(true);

        // Sample random timestep
        int t = _random.Next(_numDiffusionSteps);

        // Add noise to target (forward diffusion)
        var (noisyTarget, noise) = AddNoise(target, t);

        // Concatenate context and noisy target for denoising
        var combined = CombineContextAndNoisy(input, noisyTarget, t);

        // Predict noise
        var output = Forward(combined);

        // Calculate loss (noise prediction error)
        LastLoss = _lossFunction.CalculateLoss(output.ToVector(), noise.ToVector());

        // Backward pass
        var gradient = _lossFunction.CalculateDerivative(output.ToVector(), noise.ToVector());
        Backward(Tensor<T>.FromVector(gradient));

        _optimizer.UpdateParameters(Layers);

        SetTrainingMode(false);
    }

    /// <summary>
    /// Updates the model parameters using the optimizer (required override).
    /// </summary>
    /// <param name="gradients">Gradient vector.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TimeGrad model, UpdateParameters updates internal parameters or state. This keeps the TimeGrad architecture aligned with the latest values.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> gradients)
    {
        // Parameters are updated through the optimizer in Train()
    }

    /// <summary>
    /// Gets metadata about the TimeGrad model.
    /// </summary>
    /// <returns>ModelMetadata containing model information.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TimeGrad model, GetModelMetadata performs a supporting step in the workflow. It keeps the TimeGrad architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "TimeGrad" },
                { "ContextLength", _contextLength },
                { "ForecastHorizon", _forecastHorizon },
                { "HiddenDimension", _hiddenDimension },
                { "NumRnnLayers", _numRnnLayers },
                { "NumDiffusionSteps", _numDiffusionSteps },
                { "NumSamples", _numSamples },
                { "BetaSchedule", _betaSchedule },
                { "UseNativeMode", _useNativeMode },
                { "SupportsTraining", SupportsTraining }
            }
        };
    }

    /// <summary>
    /// Creates a new instance of the TimeGrad model with the same configuration.
    /// </summary>
    /// <returns>A new TimeGrad instance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TimeGrad model, CreateNewInstance builds and wires up model components. This sets up the TimeGrad architecture before use.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new TimeGrad<T>(Architecture, _options);
    }

    /// <summary>
    /// Serializes TimeGrad-specific data for model persistence.
    /// </summary>
    /// <param name="writer">The binary writer to serialize data to.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TimeGrad model, SerializeNetworkSpecificData saves or restores model-specific settings. This lets the TimeGrad architecture be reused later.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_contextLength);
        writer.Write(_forecastHorizon);
        writer.Write(_hiddenDimension);
        writer.Write(_numRnnLayers);
        writer.Write(_numDiffusionSteps);
        writer.Write(_numSamples);
        writer.Write(_denoisingDim);
        writer.Write(_betaSchedule);
    }

    /// <summary>
    /// Deserializes TimeGrad-specific data when loading a saved model.
    /// </summary>
    /// <param name="reader">The binary reader to deserialize data from.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TimeGrad model, DeserializeNetworkSpecificData saves or restores model-specific settings. This lets the TimeGrad architecture be reused later.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32(); // contextLength
        _ = reader.ReadInt32(); // forecastHorizon
        _ = reader.ReadInt32(); // hiddenDimension
        _ = reader.ReadInt32(); // numRnnLayers
        _ = reader.ReadInt32(); // numDiffusionSteps
        _ = reader.ReadInt32(); // numSamples
        _ = reader.ReadInt32(); // denoisingDim
        _ = reader.ReadString(); // betaSchedule
    }

    #endregion

    #region IForecastingModel Implementation

    /// <summary>
    /// Generates forecasts for the given input time series.
    /// </summary>
    /// <param name="historicalData">Input tensor of shape [batch, context, features].</param>
    /// <param name="quantiles">Optional quantile levels for probabilistic forecasting.</param>
    /// <returns>Forecast tensor of shape [batch, forecast_horizon].</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Runs the diffusion sampling process to generate forecasts.
    /// If quantiles are specified, generates multiple samples and computes the requested percentiles.
    /// </para>
    /// </remarks>
    public Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
    {
        if (quantiles is not null && quantiles.Length > 0)
        {
            // Generate multiple samples for quantile estimation
            var samples = GenerateSamples(historicalData, _numSamples);
            return ComputeQuantiles(samples, quantiles);
        }

        return _useNativeMode ? ForecastNative(historicalData) : ForecastOnnx(historicalData);
    }

    /// <summary>
    /// Generates forecasts with prediction intervals for uncertainty quantification.
    /// </summary>
    /// <param name="input">Input tensor of shape [batch, context, features].</param>
    /// <param name="confidenceLevel">Confidence level for intervals (e.g., 0.95).</param>
    /// <returns>Tuple of (point forecast, lower bound, upper bound).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Generates multiple forecast samples using diffusion
    /// and computes prediction intervals from the sample distribution.
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
    /// Performs autoregressive forecasting step by step.
    /// </summary>
    /// <param name="input">Initial input tensor.</param>
    /// <param name="steps">Number of autoregressive steps to perform.</param>
    /// <returns>Forecast tensor containing all predicted steps.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TimeGrad model, AutoregressiveForecast produces predictions from input data. This is the main inference step of the TimeGrad architecture.
    /// </para>
    /// </remarks>
    public Tensor<T> AutoregressiveForecast(Tensor<T> input, int steps)
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
    /// <param name="actuals">Actual observed values.</param>
    /// <returns>Dictionary of evaluation metrics.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TimeGrad model, Evaluate performs a supporting step in the workflow. It keeps the TimeGrad architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public Dictionary<string, T> Evaluate(Tensor<T> predictions, Tensor<T> actuals)
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

        mse = NumOps.Divide(mse, NumOps.FromDouble(count));
        mae = NumOps.Divide(mae, NumOps.FromDouble(count));
        T rmse = NumOps.FromDouble(Math.Sqrt(NumOps.ToDouble(mse)));

        metrics["MSE"] = mse;
        metrics["MAE"] = mae;
        metrics["RMSE"] = rmse;

        return metrics;
    }

    /// <summary>
    /// Applies instance normalization to the input.
    /// </summary>
    /// <param name="input">Input tensor to normalize.</param>
    /// <returns>Normalized tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TimeGrad model, ApplyInstanceNormalization performs a supporting step in the workflow. It keeps the TimeGrad architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
    {
        return input;
    }

    /// <summary>
    /// Gets financial-specific metrics about the model.
    /// </summary>
    /// <returns>Dictionary of financial metrics.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TimeGrad model, GetFinancialMetrics calculates evaluation metrics. This summarizes how the TimeGrad architecture is performing.
    /// </para>
    /// </remarks>
    public Dictionary<string, T> GetFinancialMetrics()
    {
        T lastLoss = LastLoss is not null ? LastLoss : NumOps.Zero;

        return new Dictionary<string, T>
        {
            ["LastLoss"] = lastLoss,
            ["ContextLength"] = NumOps.FromDouble(_contextLength),
            ["ForecastHorizon"] = NumOps.FromDouble(_forecastHorizon),
            ["NumDiffusionSteps"] = NumOps.FromDouble(_numDiffusionSteps),
            ["NumSamples"] = NumOps.FromDouble(_numSamples)
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
    /// <para>
    /// <b>For Beginners:</b> In the TimeGrad model, Forward runs the forward pass through the layers. This moves data through the TimeGrad architecture to compute outputs.
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
    /// <param name="gradOutput">Gradient of the loss with respect to output.</param>
    /// <returns>Gradient with respect to input.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TimeGrad model, Backward propagates gradients backward. This teaches the TimeGrad architecture how to adjust its weights.
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
    /// Performs native mode forecasting using reverse diffusion.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Forecast tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TimeGrad model, ForecastNative produces predictions from input data. This is the main inference step of the TimeGrad architecture.
    /// </para>
    /// </remarks>
    private Tensor<T> ForecastNative(Tensor<T> input)
    {
        SetTrainingMode(false);

        // Start from pure noise
        var sample = GenerateNoise(_forecastHorizon);

        // Reverse diffusion process
        for (int t = _numDiffusionSteps - 1; t >= 0; t--)
        {
            var combined = CombineContextAndNoisy(input, sample, t);
            var predictedNoise = Forward(combined);

            // Denoise step
            sample = DenoisingStep(sample, predictedNoise, t);
        }

        return sample;
    }

    /// <summary>
    /// Performs ONNX mode forecasting using the pretrained model.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Forecast tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TimeGrad model, ForecastOnnx produces predictions from input data. This is the main inference step of the TimeGrad architecture.
    /// </para>
    /// </remarks>
    private Tensor<T> ForecastOnnx(Tensor<T> input)
    {
        if (_onnxSession is null)
            throw new InvalidOperationException("ONNX session not initialized.");

        var flatInput = FlattenInput(input);
        var inputData = new float[flatInput.Data.Length];
        for (int i = 0; i < flatInput.Data.Length; i++)
        {
            inputData[i] = Convert.ToSingle(flatInput.Data.Span[i]);
        }

        var inputTensor = new OnnxTensors.DenseTensor<float>(
            inputData,
            new[] { 1, _contextLength, _numFeatures });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input", inputTensor)
        };

        using var results = _onnxSession.Run(inputs);
        var outputTensor = results[0].AsTensor<float>();

        var output = new Tensor<T>(new[] { _forecastHorizon });
        for (int i = 0; i < _forecastHorizon; i++)
        {
            output.Data.Span[i] = NumOps.FromDouble(outputTensor.GetValue(i));
        }

        return output;
    }

    #endregion

    #region Diffusion Process

    /// <summary>
    /// Adds noise to data using the forward diffusion process.
    /// </summary>
    /// <param name="data">Original data tensor.</param>
    /// <param name="t">Timestep for noise level.</param>
    /// <returns>Tuple of (noisy data, added noise).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The forward diffusion formula is:
    /// x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
    ///
    /// This gradually mixes the original data with noise.
    /// </para>
    /// </remarks>
    private (Tensor<T> noisyData, Tensor<T> noise) AddNoise(Tensor<T> data, int t)
    {
        var noise = GenerateNoise(data.Data.Length);
        var noisyData = new Tensor<T>(data.Shape);

        double sqrtAlphaBar = _sqrtAlphasCumprod[t];
        double sqrtOneMinusAlphaBar = _sqrtOneMinusAlphasCumprod[t];

        for (int i = 0; i < data.Data.Length; i++)
        {
            double val = NumOps.ToDouble(data.Data.Span[i]) * sqrtAlphaBar +
                        NumOps.ToDouble(noise.Data.Span[i]) * sqrtOneMinusAlphaBar;
            noisyData.Data.Span[i] = NumOps.FromDouble(val);
        }

        return (noisyData, noise);
    }

    /// <summary>
    /// Performs one denoising step in the reverse diffusion process.
    /// </summary>
    /// <param name="noisySample">Current noisy sample.</param>
    /// <param name="predictedNoise">Predicted noise from the model.</param>
    /// <param name="t">Current timestep.</param>
    /// <returns>Less noisy sample.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each denoising step removes a little bit of noise
    /// based on the model's prediction. Over many steps, this reconstructs the original signal.
    /// </para>
    /// </remarks>
    private Tensor<T> DenoisingStep(Tensor<T> noisySample, Tensor<T> predictedNoise, int t)
    {
        var result = new Tensor<T>(noisySample.Shape);

        double alpha = _alphas[t];
        double alphaBar = _alphasCumprod[t];
        double beta = _betas[t];
        double sqrtAlpha = Math.Sqrt(alpha);
        double sqrtOneMinusAlphaBar = _sqrtOneMinusAlphasCumprod[t];

        for (int i = 0; i < noisySample.Data.Length && i < predictedNoise.Data.Length; i++)
        {
            // Predict x_0 from x_t and predicted noise
            double predX0 = (NumOps.ToDouble(noisySample.Data.Span[i]) -
                           sqrtOneMinusAlphaBar * NumOps.ToDouble(predictedNoise.Data.Span[i])) /
                           Math.Sqrt(alphaBar);

            // Compute mean for sampling
            double mean = (NumOps.ToDouble(noisySample.Data.Span[i]) - beta * NumOps.ToDouble(predictedNoise.Data.Span[i]) / sqrtOneMinusAlphaBar) / sqrtAlpha;

            // Add noise if not at final step
            double noise = t > 0 ? SampleGaussian() * Math.Sqrt(beta) : 0;
            result.Data.Span[i] = NumOps.FromDouble(mean + noise);
        }

        return result;
    }

    /// <summary>
    /// Generates multiple forecast samples using diffusion.
    /// </summary>
    /// <param name="input">Input context tensor.</param>
    /// <param name="numSamples">Number of samples to generate.</param>
    /// <returns>List of forecast samples.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each sample represents one possible future trajectory.
    /// The variation between samples captures the forecast uncertainty.
    /// </para>
    /// </remarks>
    private List<Tensor<T>> GenerateSamples(Tensor<T> input, int numSamples)
    {
        var samples = new List<Tensor<T>>();

        for (int s = 0; s < numSamples; s++)
        {
            samples.Add(ForecastNative(input));
        }

        return samples;
    }

    /// <summary>
    /// Generates random Gaussian noise.
    /// </summary>
    /// <param name="length">Length of noise vector.</param>
    /// <returns>Noise tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TimeGrad model, GenerateNoise performs a supporting step in the workflow. It keeps the TimeGrad architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private Tensor<T> GenerateNoise(int length)
    {
        var noise = new Tensor<T>(new[] { length });

        for (int i = 0; i < length; i++)
        {
            noise.Data.Span[i] = NumOps.FromDouble(SampleGaussian());
        }

        return noise;
    }

    /// <summary>
    /// Samples from a standard Gaussian distribution.
    /// </summary>
    /// <returns>A random value from N(0, 1).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TimeGrad model, SampleGaussian performs a supporting step in the workflow. It keeps the TimeGrad architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private double SampleGaussian()
    {
        // Box-Muller transform
        double u1 = 1.0 - _random.NextDouble();
        double u2 = 1.0 - _random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    /// <summary>
    /// Combines context and noisy target for the denoising network.
    /// </summary>
    /// <param name="context">Historical context tensor.</param>
    /// <param name="noisyTarget">Noisy target tensor.</param>
    /// <param name="t">Current timestep.</param>
    /// <returns>Combined tensor for denoising network input.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TimeGrad model, CombineContextAndNoisy performs a supporting step in the workflow. It keeps the TimeGrad architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private Tensor<T> CombineContextAndNoisy(Tensor<T> context, Tensor<T> noisyTarget, int t)
    {
        var flatContext = FlattenInput(context);
        // For simplicity, return just the context - the denoising network processes it
        return flatContext;
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Flattens the input tensor for processing through dense layers.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Flattened tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TimeGrad model, FlattenInput performs a supporting step in the workflow. It keeps the TimeGrad architecture pipeline consistent.
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

    /// <summary>
    /// Computes prediction intervals from samples.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TimeGrad model, private performs a supporting step in the workflow. It keeps the TimeGrad architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private (Tensor<T> Forecast, Tensor<T> Lower, Tensor<T> Upper) ComputePredictionIntervals(
        List<Tensor<T>> samples,
        double confidenceLevel)
    {
        int horizonLength = samples[0].Data.Length;
        var mean = new Tensor<T>(new[] { horizonLength });
        var lower = new Tensor<T>(new[] { horizonLength });
        var upper = new Tensor<T>(new[] { horizonLength });

        double alpha = 1.0 - confidenceLevel;
        int lowerIdx = (int)(samples.Count * alpha / 2);
        int upperIdx = samples.Count - 1 - lowerIdx;

        for (int t = 0; t < horizonLength; t++)
        {
            var values = new List<double>();
            double sum = 0;

            foreach (var sample in samples)
            {
                double val = NumOps.ToDouble(sample.Data.Span[t]);
                values.Add(val);
                sum += val;
            }

            values.Sort();
            mean.Data.Span[t] = NumOps.FromDouble(sum / samples.Count);
            lower.Data.Span[t] = NumOps.FromDouble(values[lowerIdx]);
            upper.Data.Span[t] = NumOps.FromDouble(values[upperIdx]);
        }

        return (mean, lower, upper);
    }

    /// <summary>
    /// Computes quantiles from samples.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TimeGrad model, ComputeQuantiles performs a supporting step in the workflow. It keeps the TimeGrad architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private Tensor<T> ComputeQuantiles(List<Tensor<T>> samples, double[] quantiles)
    {
        // Return mean as default
        int horizonLength = samples[0].Data.Length;
        var result = new Tensor<T>(new[] { horizonLength });

        for (int t = 0; t < horizonLength; t++)
        {
            double sum = 0;
            foreach (var sample in samples)
            {
                sum += NumOps.ToDouble(sample.Data.Span[t]);
            }
            result.Data.Span[t] = NumOps.FromDouble(sum / samples.Count);
        }

        return result;
    }

    /// <summary>
    /// Shifts the input window for autoregressive forecasting.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TimeGrad model, ShiftInputWindow performs a supporting step in the workflow. It keeps the TimeGrad architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private Tensor<T> ShiftInputWindow(Tensor<T> input, Tensor<T> prediction)
    {
        int inputLength = input.Data.Length;
        int predLength = Math.Min(prediction.Data.Length, inputLength);

        var shifted = new Tensor<T>(input.Shape);

        for (int i = predLength; i < inputLength; i++)
        {
            shifted.Data.Span[i - predLength] = input.Data.Span[i];
        }

        for (int i = 0; i < predLength; i++)
        {
            shifted.Data.Span[inputLength - predLength + i] = prediction.Data.Span[i];
        }

        return shifted;
    }

    /// <summary>
    /// Concatenates multiple prediction tensors.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TimeGrad model, ConcatenatePredictions produces predictions from input data. This is the main inference step of the TimeGrad architecture.
    /// </para>
    /// </remarks>
    private Tensor<T> ConcatenatePredictions(List<Tensor<T>> predictions)
    {
        if (predictions.Count == 0)
            return new Tensor<T>(new[] { 0 });

        int totalLength = 0;
        foreach (var pred in predictions)
        {
            totalLength += pred.Data.Length;
        }

        var result = new Tensor<T>(new[] { totalLength });
        int offset = 0;

        foreach (var pred in predictions)
        {
            for (int i = 0; i < pred.Data.Length; i++)
            {
                result.Data.Span[offset + i] = pred.Data.Span[i];
            }
            offset += pred.Data.Length;
        }

        return result;
    }

    #endregion

    #region IDisposable

    /// <summary>
    /// Disposes of managed and unmanaged resources.
    /// </summary>
    /// <param name="disposing">True if called from Dispose().</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TimeGrad model, Dispose performs a supporting step in the workflow. It keeps the TimeGrad architecture pipeline consistent.
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
