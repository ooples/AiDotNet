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
/// CSDI (Conditional Score-based Diffusion model for Imputation) for probabilistic time series imputation.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// CSDI is a probabilistic model for time series imputation that uses score-based diffusion
/// to fill in missing values with well-calibrated uncertainty estimates.
/// </para>
/// <para><b>For Beginners:</b> CSDI solves a common real-world problem: missing data.
/// Instead of simple interpolation, it generates plausible values that are consistent
/// with observed data.
///
/// <b>The Core Problem:</b>
/// Real-world time series often have missing values:
/// - Sensor failures
/// - Data transmission errors
/// - Irregular sampling
/// - Intentional gaps (surveys, experiments)
///
/// Simple methods (mean imputation, linear interpolation) ignore uncertainty.
/// CSDI gives you the FULL probability distribution of missing values.
///
/// <b>How Score-based Diffusion Works:</b>
/// 1. <b>Score Matching:</b> Learn the gradient of log probability (the "score")
/// 2. <b>Conditional Generation:</b> Keep observed values fixed, only modify missing ones
/// 3. <b>Reverse SDE:</b> Follow the score to transform noise into realistic values
/// 4. <b>Multiple Samples:</b> Generate diverse imputations for uncertainty
///
/// <b>CSDI Architecture:</b>
/// - Input: Values + Mask (indicating observed vs missing)
/// - Transformer blocks: Capture temporal and cross-feature dependencies
/// - Residual blocks: Process diffusion timestep and predict noise
/// - Conditional sampling: Only denoise missing positions
///
/// <b>Key Benefits:</b>
/// - Handles ANY missing pattern (not just regular gaps)
/// - Uncertainty quantification for imputed values
/// - Captures complex dependencies across time and features
/// - State-of-the-art imputation quality
/// </para>
/// <para>
/// <b>Reference:</b> Tashiro et al., "CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation", 2021.
/// https://arxiv.org/abs/2107.03502
/// </para>
/// </remarks>
public class CSDI<T> : ForecastingModelBase<T>
{
    #region Execution Mode
    private bool _useNativeMode;
    #endregion


    #region Native Mode Fields
    private DenseLayer<T>? _inputProjection;
    private List<DenseLayer<T>>? _transformerLayers;
    private List<DenseLayer<T>>? _residualLayers;
    private List<LayerNormalizationLayer<T>>? _layerNorms;
    private DenseLayer<T>? _scoreHead;
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
    private readonly CSDIOptions<T> _options;
    private int _sequenceLength;
    private int _numFeatures;
    private int _hiddenDimension;
    private int _numResidualLayers;
    private int _numDiffusionSteps;
    private int _numSamples;
    private int _numHeads;
    private int _timeEmbeddingDim;
    private int _featureEmbeddingDim;
    private string _betaSchedule;
    private bool _useAttention;
    #endregion

    #region IForecastingModel Properties

    /// <inheritdoc/>
    public override int SequenceLength => _sequenceLength;

    /// <inheritdoc/>
    public override int PredictionHorizon => _sequenceLength; // For imputation, horizon = sequence length

    /// <inheritdoc/>
    public override int NumFeatures => _numFeatures;

    /// <inheritdoc/>
    public override int PatchSize => 1;

    /// <inheritdoc/>
    public override int Stride => 1;

    /// <inheritdoc/>
    public override bool IsChannelIndependent => false; // Captures cross-feature dependencies

    /// <inheritdoc/>
    public override bool UseNativeMode => _useNativeMode;

    /// <summary>
    /// Gets the hidden dimension of the score network.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The internal representation size.
    /// Larger values can capture more complex patterns.
    /// </para>
    /// </remarks>
    public int HiddenDimension => _hiddenDimension;

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
    /// Gets the number of samples to generate for uncertainty estimation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many different imputations to generate.
    /// More samples = better uncertainty estimates.
    /// </para>
    /// </remarks>
    public int NumSamples => _numSamples;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the CSDI model in ONNX mode for inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to a pretrained ONNX model file.</param>
    /// <param name="options">CSDI-specific options.</param>
    /// <param name="optimizer">Optional optimizer for fine-tuning.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to load a pretrained CSDI model
    /// for fast probabilistic imputation.
    /// </para>
    /// </remarks>
    public CSDI(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        CSDIOptions<T>? options = null,
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
        _options = options ?? new CSDIOptions<T>();
        Options = _options;
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        _sequenceLength = _options.SequenceLength;
        _numFeatures = _options.NumFeatures;
        _hiddenDimension = _options.HiddenDimension;
        _numResidualLayers = _options.NumResidualLayers;
        _numDiffusionSteps = _options.NumDiffusionSteps;
        _numSamples = _options.NumSamples;
        _numHeads = _options.NumHeads;
        _timeEmbeddingDim = _options.TimeEmbeddingDim;
        _featureEmbeddingDim = _options.FeatureEmbeddingDim;
        _betaSchedule = _options.BetaSchedule;
        _useAttention = _options.UseAttention;

        // Initialize diffusion schedule
        (_betas, _alphas, _alphasCumprod, _sqrtAlphasCumprod, _sqrtOneMinusAlphasCumprod) =
            InitializeDiffusionSchedule(_numDiffusionSteps, _options.BetaStart, _options.BetaEnd, _betaSchedule);
        _random = RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Initializes a new instance of the CSDI model in native mode for training.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">CSDI-specific options.</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="optimizer">Optional optimizer for training.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to create a CSDI model
    /// that can be trained on your data for probabilistic imputation.
    /// </para>
    /// </remarks>
    public CSDI(
        NeuralNetworkArchitecture<T> architecture,
        CSDIOptions<T>? options = null,
        int numFeatures = 1,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        _useNativeMode = true;
        _options = options ?? new CSDIOptions<T>();
        Options = _options;
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        _sequenceLength = _options.SequenceLength;
        _numFeatures = numFeatures > 0 ? numFeatures : _options.NumFeatures;
        _hiddenDimension = _options.HiddenDimension;
        _numResidualLayers = _options.NumResidualLayers;
        _numDiffusionSteps = _options.NumDiffusionSteps;
        _numSamples = _options.NumSamples;
        _numHeads = _options.NumHeads;
        _timeEmbeddingDim = _options.TimeEmbeddingDim;
        _featureEmbeddingDim = _options.FeatureEmbeddingDim;
        _betaSchedule = _options.BetaSchedule;
        _useAttention = _options.UseAttention;

        // Initialize diffusion schedule
        (_betas, _alphas, _alphasCumprod, _sqrtAlphasCumprod, _sqrtOneMinusAlphasCumprod) =
            InitializeDiffusionSchedule(_numDiffusionSteps, _options.BetaStart, _options.BetaEnd, _betaSchedule);
        _random = RandomHelper.CreateSecureRandom();

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes all layers for the CSDI model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method sets up the neural network layers:
    ///
    /// <b>Layer Structure:</b>
    /// 1. Input projection (combines values and mask)
    /// 2. Time/feature embeddings
    /// 3. Transformer blocks (self-attention)
    /// 4. Residual diffusion blocks
    /// 5. Score prediction head (outputs noise estimate)
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultCSDILayers(
                Architecture,
                _sequenceLength,
                _numFeatures,
                _hiddenDimension,
                _numResidualLayers,
                _numHeads,
                _timeEmbeddingDim,
                _featureEmbeddingDim));

            ExtractLayerReferences();
        }
    }

    /// <summary>
    /// Extracts references to key layers for efficient access during imputation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> After creating all layers, we keep direct references
    /// to important ones for quick access during the diffusion sampling process.
    /// </para>
    /// </remarks>
    private void ExtractLayerReferences()
    {
        _inputProjection = Layers.OfType<DenseLayer<T>>().FirstOrDefault();
        _layerNorms = Layers.OfType<LayerNormalizationLayer<T>>().ToList();
        _scoreHead = Layers.OfType<DenseLayer<T>>().LastOrDefault();

        var allDense = Layers.OfType<DenseLayer<T>>().ToList();
        _transformerLayers = new List<DenseLayer<T>>();
        _residualLayers = new List<DenseLayer<T>>();

        // Organize layers: first few are input processing, then transformers, then residual
        if (allDense.Count > 2)
        {
            // Input processing layers (embedding, time, feature)
            int embeddingLayers = 3;

            // Transformer layers (4 dense per transformer block * 2 blocks)
            int transformerLayerCount = 4 * 2;

            _transformerLayers = allDense.Skip(embeddingLayers)
                .Take(Math.Min(transformerLayerCount, allDense.Count - embeddingLayers - 1))
                .ToList();

            // Residual layers are the rest (before the head)
            int residualStart = embeddingLayers + transformerLayerCount;
            _residualLayers = allDense.Skip(residualStart)
                .Take(Math.Max(0, allDense.Count - residualStart - 2))
                .ToList();
        }
    }

    /// <summary>
    /// Initializes the diffusion noise schedule for score-based sampling.
    /// </summary>
    /// <param name="numSteps">Number of diffusion steps.</param>
    /// <param name="betaStart">Starting noise level.</param>
    /// <param name="betaEnd">Ending noise level.</param>
    /// <param name="schedule">Schedule type (linear, cosine, quad).</param>
    /// <returns>Tuple of precomputed diffusion schedule values.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The noise schedule defines how noise is added/removed.
    /// CSDI typically uses a quadratic schedule which adds noise more aggressively
    /// than linear, helping with the imputation task.
    ///
    /// <b>Key values:</b>
    /// - beta_t: Noise variance at step t
    /// - alpha_bar_t: Total signal remaining at step t
    /// - sqrt values: Used for efficient noise addition/removal
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
        else // quad (default for CSDI)
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
    /// <para><b>For Beginners:</b> Ensures custom layers form a valid CSDI architecture
    /// with input processing, transformer blocks, residual blocks, and score head.
    /// </para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);

        if (layers.Count < 5)
            throw new ArgumentException("CSDI requires at least 5 layers (input projection, embeddings, transformers, residuals, head).");
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <summary>
    /// Performs forward prediction (imputation) on the input tensor.
    /// </summary>
    /// <param name="input">Input tensor containing values and mask.</param>
    /// <returns>Output tensor with imputed values.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This runs the diffusion sampling process to impute
    /// missing values. It starts from noise for missing positions and gradually
    /// denoises while keeping observed values fixed.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return _useNativeMode ? ImputeNative(input) : ImputeOnnx(input);
    }

    /// <summary>
    /// Trains the CSDI model on a batch of complete data (simulating missing values).
    /// </summary>
    /// <param name="input">Input tensor of complete data.</param>
    /// <param name="target">Target tensor (same as input for imputation).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Training involves:
    /// 1. Create a random mask (simulate missing values)
    /// 2. Add noise to the "missing" positions at a random timestep
    /// 3. Predict the noise using the score network
    /// 4. Minimize the difference between predicted and actual noise
    ///
    /// The key innovation: we only compute loss on MASKED (missing) positions,
    /// teaching the model to impute conditioned on observed values.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training requires native mode. ONNX mode is inference-only.");

        SetTrainingMode(true);

        // Generate random mask (simulate missing data)
        var mask = GenerateRandomMask(input.Shape);

        // Sample random diffusion timestep
        int t = _random.Next(_numDiffusionSteps);

        // Add noise to data (only to masked/missing positions conceptually)
        var (noisyData, noise) = AddNoiseConditional(input, mask, t);

        // Prepare input: concatenate noisy data with mask
        var combined = CombineDataAndMask(noisyData, mask, t);

        // Predict noise
        var output = Forward(combined);

        // Calculate loss only on masked positions
        var maskedOutput = ApplyMask(output, mask);
        var maskedNoise = ApplyMask(noise, mask);
        LastLoss = _lossFunction.CalculateLoss(maskedOutput.ToVector(), maskedNoise.ToVector());

        // Backward pass
        var gradient = _lossFunction.CalculateDerivative(maskedOutput.ToVector(), maskedNoise.ToVector());
        Backward(Tensor<T>.FromVector(gradient, maskedOutput.Shape));

        _optimizer.UpdateParameters(Layers);

        SetTrainingMode(false);
    }

    /// <summary>
    /// Updates the model parameters using the optimizer (required override).
    /// </summary>
    /// <param name="gradients">Gradient vector.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the CSDI model, UpdateParameters updates internal parameters or state. This keeps the CSDI architecture aligned with the latest values.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> gradients)
    {
        // Parameters are updated through the optimizer in Train()
    }

    /// <summary>
    /// Gets metadata about the CSDI model.
    /// </summary>
    /// <returns>ModelMetadata containing model information.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the CSDI model, GetModelMetadata performs a supporting step in the workflow. It keeps the CSDI architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "CSDI" },
                { "SequenceLength", _sequenceLength },
                { "NumFeatures", _numFeatures },
                { "HiddenDimension", _hiddenDimension },
                { "NumResidualLayers", _numResidualLayers },
                { "NumDiffusionSteps", _numDiffusionSteps },
                { "NumSamples", _numSamples },
                { "NumHeads", _numHeads },
                { "BetaSchedule", _betaSchedule },
                { "UseAttention", _useAttention },
                { "UseNativeMode", _useNativeMode },
                { "SupportsTraining", SupportsTraining }
            }
        };
    }

    /// <summary>
    /// Creates a new instance of the CSDI model with the same configuration.
    /// </summary>
    /// <returns>A new CSDI instance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the CSDI model, CreateNewInstance builds and wires up model components. This sets up the CSDI architecture before use.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new CSDI<T>(Architecture, _options, _numFeatures);
    }

    /// <summary>
    /// Serializes CSDI-specific data for model persistence.
    /// </summary>
    /// <param name="writer">The binary writer to serialize data to.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the CSDI model, SerializeNetworkSpecificData saves or restores model-specific settings. This lets the CSDI architecture be reused later.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_sequenceLength);
        writer.Write(_numFeatures);
        writer.Write(_hiddenDimension);
        writer.Write(_numResidualLayers);
        writer.Write(_numDiffusionSteps);
        writer.Write(_numSamples);
        writer.Write(_numHeads);
        writer.Write(_timeEmbeddingDim);
        writer.Write(_featureEmbeddingDim);
        writer.Write(_betaSchedule);
        writer.Write(_useAttention);
    }

    /// <summary>
    /// Deserializes CSDI-specific data when loading a saved model.
    /// </summary>
    /// <param name="reader">The binary reader to deserialize data from.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the CSDI model, DeserializeNetworkSpecificData saves or restores model-specific settings. This lets the CSDI architecture be reused later.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _sequenceLength = reader.ReadInt32();
        _numFeatures = reader.ReadInt32();
        _hiddenDimension = reader.ReadInt32();
        _numResidualLayers = reader.ReadInt32();
        _numDiffusionSteps = reader.ReadInt32();
        _numSamples = reader.ReadInt32();
        _numHeads = reader.ReadInt32();
        _timeEmbeddingDim = reader.ReadInt32();
        _featureEmbeddingDim = reader.ReadInt32();
        _betaSchedule = reader.ReadString();
        _useAttention = reader.ReadBoolean();
    }

    #endregion

    #region IForecastingModel Implementation

    /// <summary>
    /// Generates imputed values for the given time series with missing data.
    /// </summary>
    /// <param name="historicalData">Input tensor with values and mask concatenated.</param>
    /// <param name="quantiles">Optional quantile levels for probabilistic imputation.</param>
    /// <returns>Imputed tensor with missing values filled in.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Runs the diffusion sampling process to impute.
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

        return _useNativeMode ? ImputeNative(historicalData) : ImputeOnnx(historicalData);
    }

    /// <summary>
    /// Generates imputations with prediction intervals for uncertainty quantification.
    /// </summary>
    /// <param name="input">Input tensor with values and mask.</param>
    /// <param name="confidenceLevel">Confidence level for intervals (e.g., 0.95).</param>
    /// <returns>Tuple of (point imputation, lower bound, upper bound).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Generates multiple imputation samples using diffusion
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
    /// Performs autoregressive imputation (not typically used for CSDI).
    /// </summary>
    /// <param name="input">Initial input tensor.</param>
    /// <param name="steps">Number of steps to perform.</param>
    /// <returns>Imputed tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the CSDI model, AutoregressiveForecast produces predictions from input data. This is the main inference step of the CSDI architecture.
    /// </para>
    /// </remarks>
    public override Tensor<T> AutoregressiveForecast(Tensor<T> input, int steps)
    {
        // CSDI imputes all missing values at once, not autoregressively
        return Forecast(input, null);
    }

    /// <summary>
    /// Evaluates imputation quality against actual values.
    /// </summary>
    /// <param name="predictions">Imputed values.</param>
    /// <param name="actuals">True values.</param>
    /// <returns>Dictionary of evaluation metrics.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Computes standard regression metrics to evaluate
    /// how well the imputed values match the ground truth. Lower values are better.
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
    /// Applies instance normalization to the input tensor (RevIN for non-stationarity).
    /// </summary>
    /// <param name="input">Input tensor to normalize.</param>
    /// <returns>Normalized tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Instance normalization (RevIN) helps with non-stationary
    /// time series by normalizing each instance independently. For CSDI, this is typically
    /// handled internally during the imputation process.
    /// </para>
    /// </remarks>
    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
    {
        return input;
    }

    /// <summary>
    /// Gets financial-specific metrics about the model.
    /// </summary>
    /// <returns>Dictionary of financial metrics including model configuration.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Returns key metrics and configuration parameters
    /// that are useful for understanding the model's state and performance.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> GetFinancialMetrics()
    {
        T lastLoss = LastLoss is not null ? LastLoss : NumOps.Zero;

        return new Dictionary<string, T>
        {
            ["LastLoss"] = lastLoss,
            ["SequenceLength"] = NumOps.FromDouble(_sequenceLength),
            ["NumFeatures"] = NumOps.FromDouble(_numFeatures),
            ["NumDiffusionSteps"] = NumOps.FromDouble(_numDiffusionSteps),
            ["NumSamples"] = NumOps.FromDouble(_numSamples),
            ["HiddenDimension"] = NumOps.FromDouble(_hiddenDimension),
            ["NumResidualLayers"] = NumOps.FromDouble(_numResidualLayers)
        };
    }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// Performs the forward pass through all layers.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Output tensor (predicted noise/score).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The forward pass processes input through all layers
    /// sequentially. For CSDI, the input contains noisy data + mask + timestep info,
    /// and the output is the predicted noise to remove.
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
    /// <para><b>For Beginners:</b> Backpropagation computes gradients for each layer
    /// by propagating the loss gradient backwards. This is how the network learns:
    /// gradients tell each parameter how to change to reduce the loss.
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
    /// Flattens the input tensor for processing through dense layers.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Flattened tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Neural network layers often expect 1D input.
    /// This method flattens multi-dimensional tensors (like [batch, time, features])
    /// into a single vector for processing.
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

    #region Imputation Methods

    /// <summary>
    /// Imputes missing values using native mode (full diffusion sampling).
    /// </summary>
    /// <param name="input">Input tensor with values and mask.</param>
    /// <returns>Tensor with imputed values.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the core imputation algorithm:
    /// 1. Extract observed values and mask from input
    /// 2. Initialize missing positions with random noise
    /// 3. Iteratively denoise missing positions using the score network
    /// 4. Keep observed values fixed throughout
    /// 5. Return the final imputed sequence
    /// </para>
    /// </remarks>
    private Tensor<T> ImputeNative(Tensor<T> input)
    {
        // Parse input to extract data and mask
        var (data, mask) = ParseInputWithMask(input);

        // Initialize: start with observed values + noise for missing
        var current = InitializeWithNoise(data, mask);

        // Reverse diffusion: iterate from T-1 to 0
        for (int t = _numDiffusionSteps - 1; t >= 0; t--)
        {
            // Prepare network input
            var networkInput = CombineDataAndMask(current, mask, t);

            // Predict noise/score
            var predictedNoise = Forward(networkInput);

            // Denoise step (only update missing positions)
            current = DenoisingStepConditional(current, predictedNoise, mask, t);
        }

        // Ensure observed values are exactly preserved
        current = PreserveObservedValues(current, data, mask);

        return current;
    }

    /// <summary>
    /// Imputes missing values using ONNX mode.
    /// </summary>
    /// <param name="input">Input tensor with values and mask.</param>
    /// <returns>Tensor with imputed values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the CSDI model, ImputeOnnx performs a supporting step in the workflow. It keeps the CSDI architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private Tensor<T> ImputeOnnx(Tensor<T> input)
    {
        if (OnnxSession is null)
            throw new InvalidOperationException("ONNX session not initialized.");

        var inputVector = input.ToVector();
        var inputArray = new float[inputVector.Length];
        for (int i = 0; i < inputVector.Length; i++)
        {
            inputArray[i] = Convert.ToSingle(NumOps.ToDouble(inputVector[i]));
        }

        var onnxInput = new OnnxTensors.DenseTensor<float>(inputArray, input.Shape);
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input", onnxInput)
        };

        using var results = OnnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        var outputData = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++)
        {
            outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));
        }

        return new Tensor<T>(input.Shape, new Vector<T>(outputData));
    }

    #endregion

    #region Diffusion Methods

    /// <summary>
    /// Generates multiple imputation samples for uncertainty estimation.
    /// </summary>
    /// <param name="input">Input tensor with values and mask.</param>
    /// <param name="numSamples">Number of samples to generate.</param>
    /// <returns>List of imputed tensors representing different possible imputations.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Diffusion models are stochastic: different random
    /// starting points lead to different imputations. By generating many samples,
    /// we can estimate uncertainty: high variance positions are less certain.
    /// </para>
    /// </remarks>
    private List<Tensor<T>> GenerateSamples(Tensor<T> input, int numSamples)
    {
        var samples = new List<Tensor<T>>();
        for (int i = 0; i < numSamples; i++)
        {
            samples.Add(ImputeNative(input));
        }
        return samples;
    }

    /// <summary>
    /// Adds noise to data, only affecting masked (missing) positions.
    /// </summary>
    /// <param name="data">Clean data tensor.</param>
    /// <param name="mask">Mask tensor (1 = observed, 0 = missing).</param>
    /// <param name="t">Diffusion timestep.</param>
    /// <returns>Tuple of (noisy data, noise that was added).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> During training, we add noise to simulate
    /// the forward diffusion process. But in CSDI, we only care about the
    /// noise added to MISSING positions - observed values stay clean.
    /// </para>
    /// </remarks>
    private (Tensor<T> noisy, Tensor<T> noise) AddNoiseConditional(Tensor<T> data, Tensor<T> mask, int t)
    {
        var dataVec = data.ToVector();
        var maskVec = mask.ToVector();
        var noiseVec = new T[dataVec.Length];
        var noisyVec = new T[dataVec.Length];

        double sqrtAlpha = _sqrtAlphasCumprod[t];
        double sqrtOneMinusAlpha = _sqrtOneMinusAlphasCumprod[t];

        for (int i = 0; i < dataVec.Length; i++)
        {
            // Generate standard Gaussian noise
            double u1 = _random.NextDouble();
            double u2 = _random.NextDouble();
            double noise = Math.Sqrt(-2.0 * Math.Log(Math.Max(u1, 1e-10))) * Math.Cos(2.0 * Math.PI * u2);
            noiseVec[i] = NumOps.FromDouble(noise);

            double maskVal = NumOps.ToDouble(maskVec[i]);
            if (maskVal < 0.5) // Missing position
            {
                // Add noise only to missing positions
                double dataVal = NumOps.ToDouble(dataVec[i]);
                double noisyVal = sqrtAlpha * dataVal + sqrtOneMinusAlpha * noise;
                noisyVec[i] = NumOps.FromDouble(noisyVal);
            }
            else // Observed position
            {
                // Keep observed values unchanged
                noisyVec[i] = dataVec[i];
            }
        }

        return (new Tensor<T>(data.Shape, new Vector<T>(noisyVec)),
                new Tensor<T>(data.Shape, new Vector<T>(noiseVec)));
    }

    /// <summary>
    /// Performs one step of the reverse diffusion process, only updating missing positions.
    /// </summary>
    /// <param name="current">Current state.</param>
    /// <param name="predictedNoise">Predicted noise from the network.</param>
    /// <param name="mask">Mask tensor (1 = observed, 0 = missing).</param>
    /// <param name="t">Current diffusion timestep.</param>
    /// <returns>Denoised state for the next step.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The denoising step uses the predicted noise to
    /// reverse one step of the diffusion process. The key insight of CSDI:
    /// we ONLY update missing positions, keeping observed values exactly fixed.
    /// </para>
    /// </remarks>
    private Tensor<T> DenoisingStepConditional(Tensor<T> current, Tensor<T> predictedNoise, Tensor<T> mask, int t)
    {
        var currentVec = current.ToVector();
        var noiseVec = predictedNoise.ToVector();
        var maskVec = mask.ToVector();
        var resultVec = new T[currentVec.Length];

        double alpha = _alphas[t];
        double alphaBar = _alphasCumprod[t];
        double sqrtAlpha = Math.Sqrt(alpha);
        double sqrtOneMinusAlphaBar = _sqrtOneMinusAlphasCumprod[t];

        // Compute coefficient for noise prediction
        double coeff = (1 - alpha) / sqrtOneMinusAlphaBar;

        // Add stochastic noise if not at final step
        double sigma = t > 0 ? Math.Sqrt(_betas[t]) : 0;

        for (int i = 0; i < currentVec.Length; i++)
        {
            double maskVal = NumOps.ToDouble(maskVec[i]);
            if (maskVal < 0.5) // Missing position - apply denoising
            {
                double x_t = NumOps.ToDouble(currentVec[i]);
                double eps = NumOps.ToDouble(noiseVec[i]);

                // Mean prediction
                double mean = (x_t - coeff * eps) / sqrtAlpha;

                // Add noise for stochasticity (except at t=0)
                double z = 0;
                if (t > 0)
                {
                    double u1 = _random.NextDouble();
                    double u2 = _random.NextDouble();
                    z = Math.Sqrt(-2.0 * Math.Log(Math.Max(u1, 1e-10))) * Math.Cos(2.0 * Math.PI * u2);
                }

                resultVec[i] = NumOps.FromDouble(mean + sigma * z);
            }
            else // Observed position - keep unchanged
            {
                resultVec[i] = currentVec[i];
            }
        }

        return new Tensor<T>(current.Shape, new Vector<T>(resultVec));
    }

    /// <summary>
    /// Initializes missing positions with random noise, keeping observed values.
    /// </summary>
    /// <param name="data">Data with observed values.</param>
    /// <param name="mask">Mask tensor (1 = observed, 0 = missing).</param>
    /// <returns>Tensor with observed values preserved and missing positions filled with noise.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Before starting the reverse diffusion, we need
    /// initial values for missing positions. We use random Gaussian noise,
    /// which will be gradually refined through the denoising process.
    /// </para>
    /// </remarks>
    private Tensor<T> InitializeWithNoise(Tensor<T> data, Tensor<T> mask)
    {
        var dataVec = data.ToVector();
        var maskVec = mask.ToVector();
        var resultVec = new T[dataVec.Length];

        for (int i = 0; i < dataVec.Length; i++)
        {
            double maskVal = NumOps.ToDouble(maskVec[i]);
            if (maskVal < 0.5) // Missing position
            {
                // Initialize with random noise
                double u1 = _random.NextDouble();
                double u2 = _random.NextDouble();
                double noise = Math.Sqrt(-2.0 * Math.Log(Math.Max(u1, 1e-10))) * Math.Cos(2.0 * Math.PI * u2);
                resultVec[i] = NumOps.FromDouble(noise);
            }
            else // Observed position
            {
                resultVec[i] = dataVec[i];
            }
        }

        return new Tensor<T>(data.Shape, new Vector<T>(resultVec));
    }

    /// <summary>
    /// Ensures observed values are exactly preserved after imputation.
    /// </summary>
    /// <param name="imputed">Imputed tensor.</param>
    /// <param name="original">Original data with observed values.</param>
    /// <param name="mask">Mask tensor (1 = observed, 0 = missing).</param>
    /// <returns>Tensor with observed values guaranteed to match original.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Due to numerical precision, observed values might
    /// drift slightly during denoising. This final step ensures they're exactly preserved.
    /// </para>
    /// </remarks>
    private Tensor<T> PreserveObservedValues(Tensor<T> imputed, Tensor<T> original, Tensor<T> mask)
    {
        var imputedVec = imputed.ToVector();
        var originalVec = original.ToVector();
        var maskVec = mask.ToVector();
        var resultVec = new T[imputedVec.Length];

        for (int i = 0; i < imputedVec.Length; i++)
        {
            double maskVal = NumOps.ToDouble(maskVec[i]);
            resultVec[i] = maskVal >= 0.5 ? originalVec[i] : imputedVec[i];
        }

        return new Tensor<T>(imputed.Shape, new Vector<T>(resultVec));
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Generates a random mask simulating missing data patterns.
    /// </summary>
    /// <param name="shape">Shape of the data tensor.</param>
    /// <returns>Mask tensor (1 = observed, 0 = missing).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> During training, we simulate missing data by
    /// randomly masking out some positions. The model learns to impute these.
    /// Typical missing rates: 10-50% of data.
    /// </para>
    /// </remarks>
    private Tensor<T> GenerateRandomMask(int[] shape)
    {
        int totalSize = shape.Aggregate(1, (a, b) => a * b);
        var maskVec = new T[totalSize];

        double missingRate = 0.2; // 20% missing rate
        for (int i = 0; i < totalSize; i++)
        {
            maskVec[i] = _random.NextDouble() > missingRate
                ? NumOps.One
                : NumOps.Zero;
        }

        return new Tensor<T>(shape, new Vector<T>(maskVec));
    }

    /// <summary>
    /// Combines data tensor with mask for network input.
    /// </summary>
    /// <param name="data">Data tensor.</param>
    /// <param name="mask">Mask tensor.</param>
    /// <param name="t">Diffusion timestep.</param>
    /// <returns>Combined tensor ready for the score network.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The score network needs to know:
    /// 1. The current noisy values
    /// 2. Which positions are observed vs missing
    /// 3. The current diffusion timestep
    ///
    /// We concatenate these into a single input tensor.
    /// </para>
    /// </remarks>
    private Tensor<T> CombineDataAndMask(Tensor<T> data, Tensor<T> mask, int t)
    {
        var dataVec = data.ToVector();
        var maskVec = mask.ToVector();
        var combined = new T[dataVec.Length + maskVec.Length];

        // Copy data
        for (int i = 0; i < dataVec.Length; i++)
        {
            combined[i] = dataVec[i];
        }

        // Copy mask
        for (int i = 0; i < maskVec.Length; i++)
        {
            combined[dataVec.Length + i] = maskVec[i];
        }

        // Note: timestep embedding would typically be added inside the network
        return new Tensor<T>(new[] { combined.Length }, new Vector<T>(combined));
    }

    /// <summary>
    /// Parses input tensor that contains both data and mask.
    /// </summary>
    /// <param name="input">Combined input tensor.</param>
    /// <returns>Tuple of (data tensor, mask tensor).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The input format for CSDI is [data, mask] concatenated.
    /// This method splits them apart for processing.
    /// </para>
    /// </remarks>
    private (Tensor<T> data, Tensor<T> mask) ParseInputWithMask(Tensor<T> input)
    {
        var inputVec = input.ToVector();
        int halfLen = inputVec.Length / 2;

        var dataVec = new T[halfLen];
        var maskVec = new T[halfLen];

        for (int i = 0; i < halfLen; i++)
        {
            dataVec[i] = inputVec[i];
            maskVec[i] = inputVec[halfLen + i];
        }

        var dataShape = new[] { halfLen };
        return (new Tensor<T>(dataShape, new Vector<T>(dataVec)),
                new Tensor<T>(dataShape, new Vector<T>(maskVec)));
    }

    /// <summary>
    /// Applies mask to tensor, zeroing out observed positions for loss computation.
    /// </summary>
    /// <param name="tensor">Input tensor.</param>
    /// <param name="mask">Mask tensor (1 = observed, 0 = missing).</param>
    /// <returns>Tensor with observed positions zeroed.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> During training, we only compute loss on MISSING positions
    /// because those are what we want to learn to impute. Masking zeros out observed positions.
    /// </para>
    /// </remarks>
    private Tensor<T> ApplyMask(Tensor<T> tensor, Tensor<T> mask)
    {
        var tensorVec = tensor.ToVector();
        var maskVec = mask.ToVector();

        // Adjust for potential size mismatch
        int minLen = Math.Min(tensorVec.Length, maskVec.Length);
        var resultVec = new T[tensorVec.Length];

        for (int i = 0; i < minLen; i++)
        {
            double maskVal = NumOps.ToDouble(maskVec[i]);
            // Invert mask: we want MISSING positions (mask=0)
            resultVec[i] = maskVal < 0.5 ? tensorVec[i] : NumOps.Zero;
        }

        return new Tensor<T>(tensor.Shape, new Vector<T>(resultVec));
    }

    /// <summary>
    /// Computes quantiles from a list of samples.
    /// </summary>
    /// <param name="samples">List of sample tensors.</param>
    /// <param name="quantiles">Quantile levels (e.g., [0.1, 0.5, 0.9]).</param>
    /// <returns>Tensor containing quantile values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the CSDI model, ComputeQuantiles performs a supporting step in the workflow. It keeps the CSDI architecture pipeline consistent.
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
    /// <param name="samples">List of sample tensors.</param>
    /// <param name="confidenceLevel">Confidence level (e.g., 0.95).</param>
    /// <returns>Tuple of (median, lower bound, upper bound).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the CSDI model, private performs a supporting step in the workflow. It keeps the CSDI architecture pipeline consistent.
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
    /// Disposes of resources used by the CSDI model.
    /// </summary>
    /// <param name="disposing">Whether to dispose managed resources.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the CSDI model, Dispose performs a supporting step in the workflow. It keeps the CSDI architecture pipeline consistent.
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


