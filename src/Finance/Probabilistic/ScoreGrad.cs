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
/// ScoreGrad (Score-based Gradient Model) for probabilistic time series forecasting using score matching.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// ScoreGrad is a score-based generative model that learns the gradient of the log probability
/// density function (score) and uses it for sampling via Langevin dynamics.
/// </para>
/// <para><b>For Beginners:</b> ScoreGrad uses a principled probabilistic approach called "score matching":
///
/// <b>The Key Insight:</b>
/// Instead of directly generating samples or predicting noise, ScoreGrad learns the "score function" -
/// the direction in which probability increases. By following this direction (with some randomness),
/// the model can generate realistic time series from noise.
///
/// <b>What is the Score Function?</b>
/// The score is ∇_x log p(x) - the gradient of log probability with respect to the data x.
/// - If you're at point x, the score tells you which direction has higher probability
/// - Following the score uphill leads to likely data points
/// - It's like having a compass that points toward "good" data
///
/// <b>How ScoreGrad Works:</b>
/// 1. <b>Score Network:</b> Train a neural network to predict the score at different noise levels
/// 2. <b>Denoising Score Matching:</b> Learn scores by adding noise and learning to denoise
/// 3. <b>Langevin Dynamics:</b> Sample by iteratively following the score plus random noise
/// 4. <b>Annealing:</b> Start with high noise (easy to sample), gradually reduce for refinement
///
/// <b>ScoreGrad Architecture:</b>
/// - Score Network: Predicts ∇_x log p(x|σ) conditioned on noise level σ
/// - Noise Embedding: Sinusoidal encoding of current noise level
/// - Residual Blocks: Deep architecture for learning complex score functions
/// - Output: Gradient direction at each position in the forecast
///
/// <b>Key Benefits:</b>
/// - Principled probabilistic foundation
/// - Flexible sampling (adjust step size and iterations)
/// - Natural uncertainty quantification
/// - Works well for complex multivariate dynamics
/// </para>
/// <para>
/// <b>Reference:</b> Yan et al., "ScoreGrad: Multivariate Probabilistic Time Series Forecasting with Continuous Energy-based Generative Models", 2021.
/// https://arxiv.org/abs/2106.10121
/// </para>
/// </remarks>
public class ScoreGrad<T> : ForecastingModelBase<T>
{
    #region Execution Mode
    private readonly bool _useNativeMode;
    #endregion

    
    #region Native Mode Fields
    private DenseLayer<T>? _inputProjection;
    private List<DenseLayer<T>>? _scoreNetworkLayers;
    private List<LayerNormalizationLayer<T>>? _layerNorms;
    private DenseLayer<T>? _outputHead;
    #endregion

    #region Score/Noise Fields
    private readonly double[] _sigmas;
    private readonly Random _random;
    #endregion

    #region Shared Fields
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly ScoreGradOptions<T> _options;
    private readonly int _sequenceLength;
    private readonly int _forecastHorizon;
    private readonly int _numFeatures;
    private readonly int _hiddenDimension;
    private readonly int _numLayers;
    private readonly int _numNoiseScales;
    private readonly int _numLangevinSteps;
    private readonly int _numSamples;
    private readonly double _sigmaMin;
    private readonly double _sigmaMax;
    private readonly double _stepSize;
    private readonly bool _useAnnealing;
    private readonly double _annealingPower;
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
    /// Gets the context length (input history length).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The number of past time steps used as context for
    /// computing the score function.
    /// </para>
    /// </remarks>
    public int ContextLength => _sequenceLength - _forecastHorizon;

    /// <summary>
    /// Gets the forecast horizon.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How many future steps to predict using Langevin sampling.
    /// </para>
    /// </remarks>
    public int ForecastHorizon => _forecastHorizon;

    /// <summary>
    /// Gets whether the model supports training (native mode only).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> ONNX mode is inference-only. Native mode allows
    /// training the score network using denoising score matching.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => _useNativeMode;

    /// <summary>
    /// Gets the number of noise scales for multi-scale score matching.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> ScoreGrad learns scores at multiple noise levels.
    /// More levels give smoother sampling trajectories.
    /// </para>
    /// </remarks>
    public int NumNoiseScales => _numNoiseScales;

    /// <summary>
    /// Gets the number of Langevin steps per noise level.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> More steps give better samples but slower generation.
    /// </para>
    /// </remarks>
    public int NumLangevinSteps => _numLangevinSteps;

    /// <summary>
    /// Gets the number of samples for uncertainty estimation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each sample follows a different random Langevin
    /// trajectory, capturing uncertainty in the forecast.
    /// </para>
    /// </remarks>
    public int NumSamples => _numSamples;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the ScoreGrad model in ONNX mode for inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to a pretrained ONNX model file.</param>
    /// <param name="options">ScoreGrad-specific options.</param>
    /// <param name="optimizer">Optional optimizer for fine-tuning.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to load a pretrained ScoreGrad model.
    /// The ONNX model contains the trained score network that can predict scores for
    /// Langevin sampling.
    /// </para>
    /// </remarks>
    public ScoreGrad(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        ScoreGradOptions<T>? options = null,
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
        _options = options ?? new ScoreGradOptions<T>();
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        _sequenceLength = _options.SequenceLength;
        _forecastHorizon = _options.ForecastHorizon;
        _numFeatures = _options.NumFeatures;
        _hiddenDimension = _options.HiddenDimension;
        _numLayers = _options.NumLayers;
        _numNoiseScales = _options.NumNoiseScales;
        _numLangevinSteps = _options.NumLangevinSteps;
        _numSamples = _options.NumSamples;
        _sigmaMin = _options.SigmaMin;
        _sigmaMax = _options.SigmaMax;
        _stepSize = _options.StepSize;
        _useAnnealing = _options.UseAnnealing;
        _annealingPower = _options.AnnealingPower;

        _sigmas = InitializeNoiseSchedule(_numNoiseScales, _sigmaMin, _sigmaMax);
        _random = RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Initializes a new instance of the ScoreGrad model in native mode for training.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">ScoreGrad-specific options.</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="optimizer">Optional optimizer for training.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to create a ScoreGrad model for training.
    /// The model learns the score function using denoising score matching - adding noise to data
    /// and learning to predict the direction toward the original data.
    /// </para>
    /// </remarks>
    public ScoreGrad(
        NeuralNetworkArchitecture<T> architecture,
        ScoreGradOptions<T>? options = null,
        int numFeatures = 1,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        _useNativeMode = true;
        _options = options ?? new ScoreGradOptions<T>();
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        _sequenceLength = _options.SequenceLength;
        _forecastHorizon = _options.ForecastHorizon;
        _numFeatures = numFeatures > 0 ? numFeatures : _options.NumFeatures;
        _hiddenDimension = _options.HiddenDimension;
        _numLayers = _options.NumLayers;
        _numNoiseScales = _options.NumNoiseScales;
        _numLangevinSteps = _options.NumLangevinSteps;
        _numSamples = _options.NumSamples;
        _sigmaMin = _options.SigmaMin;
        _sigmaMax = _options.SigmaMax;
        _stepSize = _options.StepSize;
        _useAnnealing = _options.UseAnnealing;
        _annealingPower = _options.AnnealingPower;

        _sigmas = InitializeNoiseSchedule(_numNoiseScales, _sigmaMin, _sigmaMax);
        _random = RandomHelper.CreateSecureRandom();

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes all layers for the ScoreGrad model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up the score network architecture:
    ///
    /// <b>Layer Structure:</b>
    /// 1. Input projection: Combines context, noisy forecast, and noise embedding
    /// 2. Score network core: Residual blocks for learning complex score functions
    /// 3. Output head: Projects to forecast dimension (the score/gradient)
    ///
    /// The network learns to output ∇_x log p(x|σ) for any input x and noise level σ.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultScoreGradLayers(
                Architecture,
                _sequenceLength,
                _forecastHorizon,
                _numFeatures,
                _hiddenDimension,
                _numLayers));

            ExtractLayerReferences();
        }
    }

    /// <summary>
    /// Extracts references to key layers for efficient access.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Keeps direct references to important layers
    /// for quick access during forward/backward passes.
    /// </para>
    /// </remarks>
    private void ExtractLayerReferences()
    {
        var allDense = Layers.OfType<DenseLayer<T>>().ToList();
        _layerNorms = Layers.OfType<LayerNormalizationLayer<T>>().ToList();

        if (allDense.Count >= 2)
        {
            _inputProjection = allDense[0];
            _scoreNetworkLayers = allDense.Skip(1).Take(allDense.Count - 2).ToList();
            _outputHead = allDense[allDense.Count - 1];
        }
    }

    /// <summary>
    /// Initializes the geometric noise schedule (sigma levels).
    /// </summary>
    /// <param name="numScales">Number of noise levels.</param>
    /// <param name="sigmaMin">Minimum noise level.</param>
    /// <param name="sigmaMax">Maximum noise level.</param>
    /// <returns>Array of sigma values in geometric progression.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a geometric sequence of noise levels from
    /// sigmaMax down to sigmaMin. Geometric spacing works better than linear because
    /// perceptual differences in noise are logarithmic.
    /// </para>
    /// </remarks>
    private double[] InitializeNoiseSchedule(int numScales, double sigmaMin, double sigmaMax)
    {
        var sigmas = new double[numScales];
        double ratio = Math.Pow(sigmaMin / sigmaMax, 1.0 / (numScales - 1));

        for (int i = 0; i < numScales; i++)
        {
            sigmas[i] = sigmaMax * Math.Pow(ratio, i);
        }

        return sigmas;
    }

    /// <summary>
    /// Validates custom layers provided by the user.
    /// </summary>
    /// <param name="layers">The list of custom layers.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Ensures custom layers form a valid ScoreGrad
    /// architecture that can learn and output scores.
    /// </para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);

        if (layers.Count < 3)
            throw new ArgumentException("ScoreGrad requires at least 3 layers (input, score network, output).");
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <summary>
    /// Performs forward prediction on the input tensor.
    /// </summary>
    /// <param name="input">Input tensor containing historical time series data.</param>
    /// <returns>Output tensor containing the forecast.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This runs annealed Langevin dynamics to generate forecasts:
    /// 1. Start with random noise
    /// 2. At each noise level, iteratively follow the score
    /// 3. Gradually reduce noise level (annealing)
    /// 4. Return the refined sample as forecast
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return _useNativeMode ? ForecastNative(input) : ForecastOnnx(input);
    }

    /// <summary>
    /// Trains the ScoreGrad model using denoising score matching.
    /// </summary>
    /// <param name="input">Input tensor (context data).</param>
    /// <param name="target">Target tensor (data to learn scores for).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Training uses Denoising Score Matching (DSM):
    ///
    /// 1. Sample a noise level σ from the schedule
    /// 2. Add Gaussian noise with variance σ² to the target
    /// 3. The network predicts the score ∇_x log p(x|σ)
    /// 4. Loss = ||predicted_score - true_score||² where true_score = -(x_noisy - x) / σ²
    ///
    /// This works because the optimal score at noise level σ is the direction
    /// from noisy data toward clean data, scaled by 1/σ².
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training requires native mode.");

        SetTrainingMode(true);

        // Sample random noise level
        int sigmaIdx = _random.Next(_numNoiseScales);
        double sigma = _sigmas[sigmaIdx];

        // Add noise to target
        var (noisyTarget, noise) = AddNoise(target, sigma);

        // Compute true score: -noise / sigma^2
        var trueScore = ComputeTrueScore(noise, sigma);

        // Create noise embedding
        var sigmaEmbed = CreateSigmaEmbedding(sigma);

        // Combine inputs: context + noisy target + sigma embedding
        var combinedInput = CombineInputs(input, noisyTarget, sigmaEmbed);

        // Forward pass: predict score
        var predictedScore = Forward(combinedInput);

        // Calculate loss: ||predicted - true||^2
        LastLoss = _lossFunction.CalculateLoss(predictedScore.ToVector(), trueScore.ToVector());

        // Backward pass
        var gradient = _lossFunction.CalculateDerivative(predictedScore.ToVector(), trueScore.ToVector());
        Backward(Tensor<T>.FromVector(gradient, predictedScore.Shape));

        _optimizer.UpdateParameters(Layers);

        SetTrainingMode(false);
    }

    /// <summary>
    /// Updates parameters using the provided gradients.
    /// </summary>
    /// <param name="gradients">Gradient vector.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the ScoreGrad model, UpdateParameters updates internal parameters or state. This keeps the ScoreGrad architecture aligned with the latest values.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> gradients)
    {
        // Parameters are updated through the optimizer in Train()
    }

    /// <summary>
    /// Gets metadata about the ScoreGrad model.
    /// </summary>
    /// <returns>ModelMetadata containing model information.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the ScoreGrad model, GetModelMetadata performs a supporting step in the workflow. It keeps the ScoreGrad architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "ScoreGrad" },
                { "SequenceLength", _sequenceLength },
                { "ForecastHorizon", _forecastHorizon },
                { "NumFeatures", _numFeatures },
                { "HiddenDimension", _hiddenDimension },
                { "NumLayers", _numLayers },
                { "NumNoiseScales", _numNoiseScales },
                { "SigmaMin", _sigmaMin },
                { "SigmaMax", _sigmaMax },
                { "NumLangevinSteps", _numLangevinSteps },
                { "StepSize", _stepSize },
                { "UseAnnealing", _useAnnealing },
                { "AnnealingPower", _annealingPower },
                { "NumSamples", _numSamples },
                { "UseNativeMode", _useNativeMode }
            }
        };
    }

    /// <summary>
    /// Creates a new instance with the same configuration.
    /// </summary>
    /// <returns>A new ScoreGrad instance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the ScoreGrad model, CreateNewInstance builds and wires up model components. This sets up the ScoreGrad architecture before use.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new ScoreGrad<T>(Architecture, _options, _numFeatures);
    }

    /// <summary>
    /// Serializes ScoreGrad-specific data.
    /// </summary>
    /// <param name="writer">The binary writer.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the ScoreGrad model, SerializeNetworkSpecificData saves or restores model-specific settings. This lets the ScoreGrad architecture be reused later.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_sequenceLength);
        writer.Write(_forecastHorizon);
        writer.Write(_numFeatures);
        writer.Write(_hiddenDimension);
        writer.Write(_numLayers);
        writer.Write(_numNoiseScales);
        writer.Write(_sigmaMin);
        writer.Write(_sigmaMax);
        writer.Write(_numLangevinSteps);
        writer.Write(_stepSize);
        writer.Write(_useAnnealing);
        writer.Write(_annealingPower);
        writer.Write(_numSamples);
    }

    /// <summary>
    /// Deserializes ScoreGrad-specific data.
    /// </summary>
    /// <param name="reader">The binary reader.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the ScoreGrad model, DeserializeNetworkSpecificData saves or restores model-specific settings. This lets the ScoreGrad architecture be reused later.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32(); // sequenceLength
        _ = reader.ReadInt32(); // forecastHorizon
        _ = reader.ReadInt32(); // numFeatures
        _ = reader.ReadInt32(); // hiddenDimension
        _ = reader.ReadInt32(); // numLayers
        _ = reader.ReadInt32(); // numNoiseScales
        _ = reader.ReadDouble(); // sigmaMin
        _ = reader.ReadDouble(); // sigmaMax
        _ = reader.ReadInt32(); // numLangevinSteps
        _ = reader.ReadDouble(); // stepSize
        _ = reader.ReadBoolean(); // useAnnealing
        _ = reader.ReadDouble(); // annealingPower
        _ = reader.ReadInt32(); // numSamples
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
    /// <para><b>For Beginners:</b> Runs annealed Langevin dynamics to generate forecasts.
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
    /// <para>
    /// <b>For Beginners:</b> In the ScoreGrad model, public performs a supporting step in the workflow. It keeps the ScoreGrad architecture pipeline consistent.
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
    /// <param name="steps">Number of forecast steps.</param>
    /// <returns>Extended forecast tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the ScoreGrad model, AutoregressiveForecast produces predictions from input data. This is the main inference step of the ScoreGrad architecture.
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
    /// <b>For Beginners:</b> In the ScoreGrad model, Evaluate performs a supporting step in the workflow. It keeps the ScoreGrad architecture pipeline consistent.
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
    /// Applies instance normalization (identity for ScoreGrad).
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>The input tensor unchanged.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the ScoreGrad model, ApplyInstanceNormalization performs a supporting step in the workflow. It keeps the ScoreGrad architecture pipeline consistent.
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
    /// <b>For Beginners:</b> In the ScoreGrad model, GetFinancialMetrics calculates evaluation metrics. This summarizes how the ScoreGrad architecture is performing.
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
            ["NumNoiseScales"] = NumOps.FromDouble(_numNoiseScales),
            ["NumLangevinSteps"] = NumOps.FromDouble(_numLangevinSteps),
            ["NumSamples"] = NumOps.FromDouble(_numSamples)
        };
    }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// Performs the forward pass through all layers.
    /// </summary>
    /// <param name="input">Input tensor (combined context, noisy sample, sigma embedding).</param>
    /// <returns>Output tensor (predicted score).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The forward pass computes the score function
    /// ∇_x log p(x|σ) given the input and noise level embedding.
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
    /// <para>
    /// <b>For Beginners:</b> In the ScoreGrad model, Backward propagates gradients backward. This teaches the ScoreGrad architecture how to adjust its weights.
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
    /// <b>For Beginners:</b> In the ScoreGrad model, FlattenInput performs a supporting step in the workflow. It keeps the ScoreGrad architecture pipeline consistent.
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

    #region Langevin Dynamics

    /// <summary>
    /// Performs native mode forecasting with annealed Langevin dynamics.
    /// </summary>
    /// <param name="context">Input context tensor.</param>
    /// <returns>Forecast tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Annealed Langevin dynamics samples from the distribution by:
    /// 1. Starting from random noise
    /// 2. At each noise level σ, run multiple Langevin steps:
    ///    x_{t+1} = x_t + ε * score(x_t, σ) + sqrt(2ε) * noise
    /// 3. Gradually reduce σ (annealing) for refinement
    /// 4. The final x is the forecast
    ///
    /// The key insight is that following the score uphill leads to likely data points.
    /// </para>
    /// </remarks>
    private Tensor<T> ForecastNative(Tensor<T> context)
    {
        SetTrainingMode(false);

        int forecastSize = _forecastHorizon * _numFeatures;

        // Initialize with random noise scaled by max sigma
        var sample = GenerateNoise(forecastSize, _sigmaMax);

        // Annealed Langevin dynamics
        if (_useAnnealing)
        {
            // Iterate through noise levels from high to low
            for (int sigmaIdx = 0; sigmaIdx < _numNoiseScales; sigmaIdx++)
            {
                double sigma = _sigmas[sigmaIdx];
                double effectiveStepSize = _stepSize * Math.Pow(sigma / _sigmaMax, _annealingPower);

                // Langevin steps at this noise level
                for (int step = 0; step < _numLangevinSteps / _numNoiseScales; step++)
                {
                    sample = LangevinStep(sample, context, sigma, effectiveStepSize);
                }
            }
        }
        else
        {
            // Single noise level Langevin
            double sigma = _sigmaMin;
            for (int step = 0; step < _numLangevinSteps; step++)
            {
                sample = LangevinStep(sample, context, sigma, _stepSize);
            }
        }

        return sample;
    }

    /// <summary>
    /// Performs one step of Langevin dynamics.
    /// </summary>
    /// <param name="current">Current sample.</param>
    /// <param name="context">Context tensor.</param>
    /// <param name="sigma">Current noise level.</param>
    /// <param name="stepSize">Step size (epsilon).</param>
    /// <returns>Updated sample.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Langevin dynamics update rule:
    /// x_{t+1} = x_t + ε * score(x_t, σ) + sqrt(2ε) * z
    ///
    /// where:
    /// - ε is the step size
    /// - score() is the learned score function
    /// - z is standard Gaussian noise
    ///
    /// This is like gradient ascent on log probability, plus noise for exploration.
    /// </para>
    /// </remarks>
    private Tensor<T> LangevinStep(Tensor<T> current, Tensor<T> context, double sigma, double stepSize)
    {
        // Create sigma embedding
        var sigmaEmbed = CreateSigmaEmbedding(sigma);

        // Combine inputs
        var combinedInput = CombineInputs(context, current, sigmaEmbed);

        // Get score from network
        var score = Forward(combinedInput);

        var currentVec = current.ToVector();
        var scoreVec = score.ToVector();
        var resultVec = new T[currentVec.Length];

        double noiseScale = Math.Sqrt(2 * stepSize);

        for (int i = 0; i < currentVec.Length; i++)
        {
            double x = NumOps.ToDouble(currentVec[i]);
            double s = NumOps.ToDouble(scoreVec[i % scoreVec.Length]);

            // Sample random noise
            double u1 = _random.NextDouble();
            double u2 = _random.NextDouble();
            double z = Math.Sqrt(-2.0 * Math.Log(Math.Max(u1, 1e-10))) * Math.Cos(2.0 * Math.PI * u2);

            // Langevin update: x + stepSize * score + noiseScale * z
            double newX = x + stepSize * s + noiseScale * z;
            resultVec[i] = NumOps.FromDouble(newX);
        }

        return new Tensor<T>(current.Shape, new Vector<T>(resultVec));
    }

    /// <summary>
    /// Performs ONNX mode forecasting.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Forecast tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the ScoreGrad model, ForecastOnnx produces predictions from input data. This is the main inference step of the ScoreGrad architecture.
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
    /// <para><b>For Beginners:</b> Each sample follows a different random Langevin
    /// trajectory due to the stochastic noise in each step. This diversity
    /// captures the uncertainty in the forecast.
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

    #endregion

    #region Score Matching Helpers

    /// <summary>
    /// Adds Gaussian noise to data with specified standard deviation.
    /// </summary>
    /// <param name="data">Clean data tensor.</param>
    /// <param name="sigma">Noise standard deviation.</param>
    /// <returns>Tuple of (noisy data, noise added).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Adds Gaussian noise with variance σ² to the data.
    /// This is the first step in denoising score matching - corrupt the data,
    /// then learn to predict the direction back to clean data.
    /// </para>
    /// </remarks>
    private (Tensor<T> noisy, Tensor<T> noise) AddNoise(Tensor<T> data, double sigma)
    {
        var dataVec = data.ToVector();
        var noiseVec = new T[dataVec.Length];
        var noisyVec = new T[dataVec.Length];

        for (int i = 0; i < dataVec.Length; i++)
        {
            double u1 = _random.NextDouble();
            double u2 = _random.NextDouble();
            double noise = Math.Sqrt(-2.0 * Math.Log(Math.Max(u1, 1e-10))) * Math.Cos(2.0 * Math.PI * u2);
            noise *= sigma;
            noiseVec[i] = NumOps.FromDouble(noise);

            double dataVal = NumOps.ToDouble(dataVec[i]);
            noisyVec[i] = NumOps.FromDouble(dataVal + noise);
        }

        return (new Tensor<T>(data.Shape, new Vector<T>(noisyVec)),
                new Tensor<T>(data.Shape, new Vector<T>(noiseVec)));
    }

    /// <summary>
    /// Computes the true score for denoising score matching.
    /// </summary>
    /// <param name="noise">The noise that was added.</param>
    /// <param name="sigma">The noise standard deviation.</param>
    /// <returns>True score tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The true score at noisy point x_noisy is:
    /// score = -noise / σ² = -(x_noisy - x_clean) / σ²
    ///
    /// This points from the noisy data toward the clean data, which is exactly
    /// what we want the network to learn.
    /// </para>
    /// </remarks>
    private Tensor<T> ComputeTrueScore(Tensor<T> noise, double sigma)
    {
        var noiseVec = noise.ToVector();
        var scoreVec = new T[noiseVec.Length];

        double sigmaSq = sigma * sigma;
        for (int i = 0; i < noiseVec.Length; i++)
        {
            double n = NumOps.ToDouble(noiseVec[i]);
            scoreVec[i] = NumOps.FromDouble(-n / sigmaSq);
        }

        return new Tensor<T>(noise.Shape, new Vector<T>(scoreVec));
    }

    /// <summary>
    /// Creates a sinusoidal embedding for the noise level (sigma).
    /// </summary>
    /// <param name="sigma">The noise level.</param>
    /// <returns>Sigma embedding tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Sinusoidal embeddings represent the noise level
    /// in a way the network can easily learn from. Similar to positional encoding
    /// in transformers, it uses sine and cosine functions at different frequencies.
    /// </para>
    /// </remarks>
    private Tensor<T> CreateSigmaEmbedding(double sigma)
    {
        int embedDim = _hiddenDimension / 4;
        var embedding = new T[embedDim];

        double logSigma = Math.Log(sigma + 1e-10);

        for (int i = 0; i < embedDim; i++)
        {
            double freq = Math.Exp(-i * Math.Log(10000.0) / embedDim);
            if (i % 2 == 0)
            {
                embedding[i] = NumOps.FromDouble(Math.Sin(logSigma * freq));
            }
            else
            {
                embedding[i] = NumOps.FromDouble(Math.Cos(logSigma * freq));
            }
        }

        return new Tensor<T>(new[] { embedDim }, new Vector<T>(embedding));
    }

    /// <summary>
    /// Combines context, sample, and sigma embedding into network input.
    /// </summary>
    /// <param name="context">Context tensor.</param>
    /// <param name="sample">Current sample (noisy for training, iterating for inference).</param>
    /// <param name="sigmaEmbed">Sigma embedding tensor.</param>
    /// <returns>Combined input tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The score network needs all this information:
    /// - Context: Historical data that conditions the forecast
    /// - Sample: The current point where we're computing the score
    /// - Sigma embedding: The noise level, so the network knows which scale to work at
    /// </para>
    /// </remarks>
    private Tensor<T> CombineInputs(Tensor<T> context, Tensor<T> sample, Tensor<T> sigmaEmbed)
    {
        var contextVec = context.ToVector();
        var sampleVec = sample.ToVector();
        var embedVec = sigmaEmbed.ToVector();

        int totalLen = contextVec.Length + sampleVec.Length + embedVec.Length;
        var combined = new T[totalLen];

        int offset = 0;
        for (int i = 0; i < contextVec.Length; i++)
        {
            combined[offset++] = contextVec[i];
        }
        for (int i = 0; i < sampleVec.Length; i++)
        {
            combined[offset++] = sampleVec[i];
        }
        for (int i = 0; i < embedVec.Length; i++)
        {
            combined[offset++] = embedVec[i];
        }

        return new Tensor<T>(new[] { totalLen }, new Vector<T>(combined));
    }

    /// <summary>
    /// Generates random Gaussian noise scaled by sigma.
    /// </summary>
    /// <param name="size">Size of noise vector.</param>
    /// <param name="sigma">Noise scale (standard deviation).</param>
    /// <returns>Noise tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Generates starting noise for Langevin sampling.
    /// The noise is scaled by sigma so the initial sample is at the right scale
    /// for the highest noise level.
    /// </para>
    /// </remarks>
    private Tensor<T> GenerateNoise(int size, double sigma)
    {
        var noiseVec = new T[size];

        for (int i = 0; i < size; i++)
        {
            double u1 = _random.NextDouble();
            double u2 = _random.NextDouble();
            double noise = Math.Sqrt(-2.0 * Math.Log(Math.Max(u1, 1e-10))) * Math.Cos(2.0 * Math.PI * u2);
            noiseVec[i] = NumOps.FromDouble(noise * sigma);
        }

        return new Tensor<T>(new[] { size }, new Vector<T>(noiseVec));
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Shifts input window by appending new prediction.
    /// </summary>
    /// <param name="input">Current input.</param>
    /// <param name="prediction">New prediction to append.</param>
    /// <returns>Shifted input.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the ScoreGrad model, ShiftInputWindow performs a supporting step in the workflow. It keeps the ScoreGrad architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private Tensor<T> ShiftInputWindow(Tensor<T> input, Tensor<T> prediction)
    {
        var inputVec = input.ToVector();
        var predVec = prediction.ToVector();

        int contextLen = ContextLength * _numFeatures;
        var shifted = new T[contextLen];

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
    /// <b>For Beginners:</b> In the ScoreGrad model, ConcatenatePredictions produces predictions from input data. This is the main inference step of the ScoreGrad architecture.
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
    /// <b>For Beginners:</b> In the ScoreGrad model, ComputeQuantiles performs a supporting step in the workflow. It keeps the ScoreGrad architecture pipeline consistent.
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
    /// <b>For Beginners:</b> In the ScoreGrad model, private performs a supporting step in the workflow. It keeps the ScoreGrad architecture pipeline consistent.
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
    /// <b>For Beginners:</b> In the ScoreGrad model, Dispose performs a supporting step in the workflow. It keeps the ScoreGrad architecture pipeline consistent.
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



