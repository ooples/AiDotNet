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
namespace AiDotNet.Finance.Forecasting.Foundation;

/// <summary>
/// Lag-Llama foundation model for probabilistic time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// Lag-Llama adapts the Llama large language model architecture for time series forecasting.
/// It uses lag-based features to capture temporal patterns at multiple scales and outputs
/// probabilistic forecasts via distribution parameter prediction.
/// </para>
/// <para>
/// <b>For Beginners:</b> Lag-Llama brings LLM innovations to time series forecasting:
///
/// <b>The Lag Feature Innovation:</b>
/// Instead of just looking at recent values, Lag-Llama creates features from specific past points:
/// - Lag-1: Yesterday's value (immediate trend)
/// - Lag-7: Same day last week (weekly pattern)
/// - Lag-365: Same day last year (annual pattern)
///
/// This lets the model explicitly see patterns at different time scales without needing
/// a very long context window.
///
/// <b>Llama Architecture Adaptations:</b>
/// Lag-Llama adopts key Llama innovations:
/// - <b>RMSNorm</b>: Simpler, faster layer normalization
/// - <b>SwiGLU</b>: Improved MLP activation function
/// - <b>RoPE</b>: Rotary Position Embeddings for better position encoding
/// - <b>Causal Attention</b>: Each position only sees earlier positions
///
/// <b>Probabilistic Forecasting:</b>
/// Unlike models that output single values, Lag-Llama outputs distribution parameters:
/// - For Student-t: degrees of freedom (nu), location (mu), scale (sigma)
/// - Allows uncertainty quantification: "The forecast is 100 ± 15"
/// - Enables risk-aware decisions: "There's a 5% chance it exceeds 130"
///
/// <b>Why Student-t Distribution?</b>
/// - Has heavier tails than Normal (better for extreme events)
/// - Degrades gracefully to Normal as nu → infinity
/// - More robust to outliers in training data
///
/// <b>Zero-Shot Capability:</b>
/// Pre-trained on diverse time series, Lag-Llama can forecast new series without training.
/// </para>
/// <para>
/// <b>Reference:</b> Rasul et al., "Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting", 2024.
/// https://arxiv.org/abs/2310.08278
/// </para>
/// </remarks>
public class LagLlama<T> : ForecastingModelBase<T>
{
    #region Execution Mode

    /// <summary>
    /// Indicates whether this network uses native layers (true) or ONNX model (false).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> ONNX mode loads pretrained weights for immediate zero-shot forecasting.
    /// Native mode allows fine-tuning on your specific domain.
    /// </para>
    /// </remarks>
    private readonly bool _useNativeMode;

    #endregion

    
    #region Native Mode Fields

    /// <summary>
    /// Input embedding layer that projects lag features to hidden dimension.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Converts the raw lag features (current value + lagged values)
    /// into a rich hidden representation for the transformer.
    /// </para>
    /// </remarks>
    private ILayer<T>? _inputEmbedding;

    /// <summary>
    /// Transformer layers with Llama-style architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Multiple layers of attention and feed-forward processing.
    /// Each layer includes: RMSNorm → Attention → RMSNorm → FFN.
    /// </para>
    /// </remarks>
    private readonly List<ILayer<T>> _transformerLayers = [];

    /// <summary>
    /// Final layer normalization before output projection.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Stabilizes values before predicting distribution parameters.
    /// </para>
    /// </remarks>
    private ILayer<T>? _finalNorm;

    /// <summary>
    /// Distribution output head that predicts distribution parameters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Instead of predicting single values, outputs parameters
    /// of a probability distribution (e.g., mu, sigma, nu for Student-t).
    /// </para>
    /// </remarks>
    private ILayer<T>? _distributionHead;

    #endregion

    #region Shared Fields

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private int _contextLength;
    private int _forecastHorizon;
    private int _hiddenDimension;
    private int _numLayers;
    private int _numHeads;
    private int _intermediateSize;
    private int[] _lagIndices;
    private double _dropout;
    private string _distributionOutput;
    private bool _useRoPE;

    #endregion

    #region IForecastingModel Properties

    /// <inheritdoc/>
    public override int SequenceLength => _contextLength;

    /// <inheritdoc/>
    public override int PredictionHorizon => _forecastHorizon;

    /// <inheritdoc/>
    public override int NumFeatures => 1;

    /// <inheritdoc/>
    public override int PatchSize => 1;

    /// <inheritdoc/>
    public override int Stride => 1;

    /// <inheritdoc/>
    public override bool IsChannelIndependent => true;

    /// <inheritdoc/>
    public override bool UseNativeMode => _useNativeMode;

    /// <summary>
    /// Gets the lag indices used for feature extraction.
    /// </summary>
    /// <value>Array of lag indices.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The specific time lags used as features.
    /// Default is [1, 2, 3, 7, 14, 28, 365] to capture daily, weekly, monthly, and yearly patterns.
    /// </para>
    /// </remarks>
    public int[] LagIndices => _lagIndices;

    /// <summary>
    /// Gets the distribution type used for probabilistic output.
    /// </summary>
    /// <value>The distribution type (e.g., "StudentT", "Normal").</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Determines what kind of probability distribution is predicted.
    /// Student-t is the default as it handles outliers better than Normal.
    /// </para>
    /// </remarks>
    public string DistributionOutput => _distributionOutput;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a Lag-Llama model using pretrained ONNX model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer for fine-tuning.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the recommended way to use Lag-Llama:
    /// Load pretrained weights for immediate zero-shot probabilistic forecasting.
    /// </para>
    /// </remarks>
    public LagLlama(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        LagLlamaOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        options ??= new LagLlamaOptions<T>();
        ValidateOptions(options);

        _useNativeMode = false;
        OnnxSession = new InferenceSession(onnxModelPath);
        OnnxModelPath = onnxModelPath;

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        _contextLength = options.ContextLength;
        _forecastHorizon = options.ForecastHorizon;
        _hiddenDimension = options.HiddenDimension;
        _numLayers = options.NumLayers;
        _numHeads = options.NumHeads;
        _intermediateSize = options.IntermediateSize;
        _lagIndices = (int[])options.LagIndices.Clone();
        _dropout = options.DropoutRate;
        _distributionOutput = options.DistributionOutput;
        _useRoPE = options.UseRoPE;
    }

    /// <summary>
    /// Creates a Lag-Llama model in native mode for training or fine-tuning.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this for fine-tuning or training from scratch.
    /// Native mode supports:
    /// - Fine-tuning on domain-specific data
    /// - Custom lag configurations for your use case
    /// - Experimentation with different distributions
    /// </para>
    /// </remarks>
    public LagLlama(
        NeuralNetworkArchitecture<T> architecture,
        LagLlamaOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new LagLlamaOptions<T>();
        ValidateOptions(options);

        _useNativeMode = true;
        OnnxSession = null;
        OnnxModelPath = null;

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        _contextLength = options.ContextLength;
        _forecastHorizon = options.ForecastHorizon;
        _hiddenDimension = options.HiddenDimension;
        _numLayers = options.NumLayers;
        _numHeads = options.NumHeads;
        _intermediateSize = options.IntermediateSize;
        _lagIndices = (int[])options.LagIndices.Clone();
        _dropout = options.DropoutRate;
        _distributionOutput = options.DistributionOutput;
        _useRoPE = options.UseRoPE;

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes the neural network layers for Lag-Llama.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Sets up the complete Lag-Llama architecture:
    /// 1. Input embedding for lag features
    /// 2. Stack of Llama-style transformer layers
    /// 3. Final normalization
    /// 4. Distribution output head
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
            // Extract layer references so Forward/Backward work with custom layers
            ExtractLayerReferences();
        }
        else if (_useNativeMode)
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultLagLlamaLayers(
                Architecture, _contextLength, _forecastHorizon, 1,
                _lagIndices.Length, _hiddenDimension, _numLayers, _numHeads,
                _intermediateSize, _dropout));

            ExtractLayerReferences();
        }
    }

    /// <summary>
    /// Extracts references to specific layers from the layer collection.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Organizes layers by their role:
    /// - First: Input embedding
    /// - Middle: Transformer blocks (attention + FFN + norms)
    /// - Last two: Final norm and distribution head
    /// </para>
    /// </remarks>
    private void ExtractLayerReferences()
    {
        int idx = 0;

        // Input embedding
        if (idx < Layers.Count)
            _inputEmbedding = Layers[idx++];

        // Transformer layers
        // Each block has: norm, Q, K, V, O, (dropout), norm, gate, up, down, (dropout)
        _transformerLayers.Clear();
        int layersPerBlock = _dropout > 0 ? 10 : 8;
        int totalTransformerLayers = _numLayers * layersPerBlock;

        for (int i = 0; i < totalTransformerLayers && idx < Layers.Count - 2; i++)
        {
            _transformerLayers.Add(Layers[idx++]);
        }

        // Final normalization
        if (idx < Layers.Count)
            _finalNorm = Layers[idx++];

        // Distribution output head
        if (idx < Layers.Count)
            _distributionHead = Layers[idx];
    }

    /// <summary>
    /// Validates that custom layers meet Lag-Llama architectural requirements.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Ensures you have all required components for the model.
    /// </para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);
        if (layers.Count < 5)
        {
            throw new ArgumentException(
                "Lag-Llama requires at least 5 layers (input embed, attention, FFN, norm, distribution head).",
                nameof(layers));
        }
    }

    /// <summary>
    /// Validates the Lag-Llama options.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Checks that all configuration values are sensible.
    /// </para>
    /// </remarks>
    private static void ValidateOptions(LagLlamaOptions<T> options)
    {
        var errors = new List<string>();

        if (options.ContextLength < 1)
            errors.Add("ContextLength must be at least 1.");
        if (options.ForecastHorizon < 1)
            errors.Add("ForecastHorizon must be at least 1.");
        if (options.HiddenDimension < 1)
            errors.Add("HiddenDimension must be at least 1.");
        if (options.NumLayers < 1)
            errors.Add("NumLayers must be at least 1.");
        if (options.NumHeads < 1)
            errors.Add("NumHeads must be at least 1.");
        if (options.HiddenDimension % options.NumHeads != 0)
            errors.Add("HiddenDimension must be divisible by NumHeads.");
        if (options.IntermediateSize < 1)
            errors.Add("IntermediateSize must be at least 1.");
        if (options.LagIndices == null || options.LagIndices.Length == 0)
            errors.Add("LagIndices must contain at least one lag.");
        if (options.DropoutRate < 0 || options.DropoutRate >= 1)
            errors.Add("DropoutRate must be between 0 and 1 (exclusive).");

        // Validate distribution output - ExtractPointPredictions assumes 3 params per step (mu/sigma/nu)
        // for StudentT distribution. Other distributions require matching param count.
        var validDistributions = new[] { "StudentT", "Normal" };
        if (!string.IsNullOrEmpty(options.DistributionOutput) &&
            !validDistributions.Contains(options.DistributionOutput, StringComparer.OrdinalIgnoreCase))
        {
            errors.Add($"DistributionOutput must be one of: {string.Join(", ", validDistributions)}. Got: {options.DistributionOutput}");
        }

        // For StudentT (default), output should be 3 * ForecastHorizon (mu, sigma, nu per step)
        // For Normal, output should be 2 * ForecastHorizon (mu, sigma per step)
        // This validation ensures ExtractPointPredictions indexing is correct
        if (string.IsNullOrEmpty(options.DistributionOutput) ||
            string.Equals(options.DistributionOutput, "StudentT", StringComparison.OrdinalIgnoreCase))
        {
            // StudentT expected - requires 3 params per step (handled by default)
        }
        else if (string.Equals(options.DistributionOutput, "Normal", StringComparison.OrdinalIgnoreCase))
        {
            // Normal distribution - note: ExtractPointPredictions uses stride of 3,
            // so Normal would need separate handling or stride adjustment
            errors.Add("DistributionOutput 'Normal' requires 2 params per step but ExtractPointPredictions assumes 3 (StudentT). Use StudentT or adjust model output.");
        }

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
    /// <b>For Beginners:</b> In the LagLlama model, Predict produces predictions from input data. This is the main inference step of the LagLlama architecture.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return _useNativeMode ? ForecastNative(input) : ForecastOnnx(input);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Training Lag-Llama by default uses MSE loss on point predictions
    /// extracted from distribution parameters. For true probabilistic training with
    /// negative log-likelihood, provide a custom NLL loss function in the constructor.
    /// The model learns to predict distribution parameters (mu, sigma, nu for Student-t)
    /// that best fit the observed values.
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

            // Extract point predictions from distribution parameters (use mean)
            var pointPredictions = ExtractPointPredictions(predictions);
            var predVector = pointPredictions.ToVector();

            // Align target to match point predictions length
            // Target may be raw distribution params or point values of different size
            var targetVector = target.ToVector();
            Vector<T> alignedTarget;
            if (targetVector.Length == predVector.Length)
            {
                alignedTarget = targetVector;
            }
            else if (targetVector.Length >= predVector.Length)
            {
                // Take first N elements if target is larger
                alignedTarget = new Vector<T>(predVector.Length);
                for (int i = 0; i < predVector.Length; i++)
                {
                    alignedTarget[i] = targetVector[i];
                }
            }
            else
            {
                // Pad with last value if target is smaller
                alignedTarget = new Vector<T>(predVector.Length);
                for (int i = 0; i < targetVector.Length; i++)
                {
                    alignedTarget[i] = targetVector[i];
                }
                T lastVal = targetVector.Length > 0 ? targetVector[targetVector.Length - 1] : NumOps.Zero;
                for (int i = targetVector.Length; i < predVector.Length; i++)
                {
                    alignedTarget[i] = lastVal;
                }
            }

            LastLoss = _lossFunction.CalculateLoss(predVector, alignedTarget);

            // Calculate gradient w.r.t. point predictions (mu values)
            var pointGradient = _lossFunction.CalculateDerivative(predVector, alignedTarget);

            // Map the point-prediction gradient back to full distribution-parameter gradient
            // Forward outputs [mu, sigma, nu] per step (3 params), but loss is computed on mu only
            // We need to create a full-sized gradient matching predictions.Shape with:
            // - Gradient for mu positions = pointGradient values
            // - Gradient for sigma/nu positions = 0 (no direct loss contribution)
            int paramsPerStep = 3; // mu, sigma, nu for StudentT
            int fullGradientLength = predictions.Length;
            var fullGradient = new Vector<T>(fullGradientLength);

            // Insert point gradients at mu positions (every 3rd value starting at 0)
            for (int i = 0; i < pointGradient.Length; i++)
            {
                int muIdx = i * paramsPerStep;
                if (muIdx < fullGradientLength)
                {
                    fullGradient[muIdx] = pointGradient[i];
                }
                // sigma (idx+1) and nu (idx+2) remain zero - no direct loss gradient
            }

            Backward(Tensor<T>.FromVector(fullGradient, predictions.Shape));

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
    /// <b>For Beginners:</b> In the LagLlama model, UpdateParameters updates internal parameters or state. This keeps the LagLlama architecture aligned with the latest values.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> gradients)
    {
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the LagLlama model, GetModelMetadata performs a supporting step in the workflow. It keeps the LagLlama architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "LagLlama" },
                { "ContextLength", _contextLength },
                { "ForecastHorizon", _forecastHorizon },
                { "HiddenDimension", _hiddenDimension },
                { "NumLayers", _numLayers },
                { "NumHeads", _numHeads },
                { "IntermediateSize", _intermediateSize },
                { "LagIndices", string.Join(",", _lagIndices) },
                { "DistributionOutput", _distributionOutput },
                { "UseRoPE", _useRoPE },
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
    /// <b>For Beginners:</b> Creates a fresh copy of the Lag-Llama architecture.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var options = new LagLlamaOptions<T>
        {
            ContextLength = _contextLength,
            ForecastHorizon = _forecastHorizon,
            HiddenDimension = _hiddenDimension,
            NumLayers = _numLayers,
            NumHeads = _numHeads,
            IntermediateSize = _intermediateSize,
            LagIndices = (int[])_lagIndices.Clone(),
            DropoutRate = _dropout,
            DistributionOutput = _distributionOutput,
            UseRoPE = _useRoPE
        };

        // ONNX mode cloning is not supported - throw explicitly rather than silently
        // changing behavior by returning a native-mode clone
        if (!_useNativeMode && OnnxSession is not null)
        {
            throw new NotSupportedException(
                "CreateNewInstance is not supported for ONNX-backed LagLlama models. " +
                "ONNX sessions cannot be cloned. To create a new instance, load the model " +
                "from the original ONNX file using the ONNX constructor.");
        }

        return new LagLlama<T>(Architecture, options);
    }

    /// <summary>
    /// Writes Lag-Llama-specific configuration during serialization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Saves all the configuration needed to reconstruct this model.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_contextLength);
        writer.Write(_forecastHorizon);
        writer.Write(_hiddenDimension);
        writer.Write(_numLayers);
        writer.Write(_numHeads);
        writer.Write(_intermediateSize);
        writer.Write(_lagIndices.Length);
        foreach (var lag in _lagIndices)
            writer.Write(lag);
        writer.Write(_dropout);
        writer.Write(_distributionOutput);
        writer.Write(_useRoPE);
    }

    /// <summary>
    /// Reads Lag-Llama-specific configuration during deserialization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads the configuration that was saved during serialization.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _contextLength = reader.ReadInt32();
        _forecastHorizon = reader.ReadInt32();
        _hiddenDimension = reader.ReadInt32();
        _numLayers = reader.ReadInt32();
        _numHeads = reader.ReadInt32();
        _intermediateSize = reader.ReadInt32();
        int lagCount = reader.ReadInt32();
        _lagIndices = new int[lagCount];
        for (int i = 0; i < lagCount; i++)
            _lagIndices[i] = reader.ReadInt32();
        _dropout = reader.ReadDouble();
        _distributionOutput = reader.ReadString();
        _useRoPE = reader.ReadBoolean();
    }

    #endregion

    #region IForecastingModel Implementation

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the LagLlama model, Forecast produces predictions from input data. This is the main inference step of the LagLlama architecture.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
    {
        var output = _useNativeMode ? ForecastNative(historicalData) : ForecastOnnx(historicalData);

        // If quantiles requested, sample from distribution
        if (quantiles is not null && quantiles.Length > 0)
        {
            return SampleQuantiles(output, quantiles);
        }

        // Return point predictions (mean of distribution)
        return ExtractPointPredictions(output);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> For forecasting beyond the horizon, Lag-Llama:
    /// 1. Generates probabilistic forecasts for the first horizon
    /// 2. Uses the predicted means as new "history"
    /// 3. Repeats until the desired forecast length is reached
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
    /// <b>For Beginners:</b> In the LagLlama model, Evaluate performs a supporting step in the workflow. It keeps the LagLlama architecture pipeline consistent.
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
    /// <b>For Beginners:</b> In the LagLlama model, ApplyInstanceNormalization performs a supporting step in the workflow. It keeps the LagLlama architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
    {
        return input;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the LagLlama model, GetFinancialMetrics calculates evaluation metrics. This summarizes how the LagLlama architecture is performing.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> GetFinancialMetrics()
    {
        T lastLoss = LastLoss is not null ? LastLoss : NumOps.Zero;

        return new Dictionary<string, T>
        {
            ["ContextLength"] = NumOps.FromDouble(_contextLength),
            ["ForecastHorizon"] = NumOps.FromDouble(_forecastHorizon),
            ["HiddenDimension"] = NumOps.FromDouble(_hiddenDimension),
            ["NumLayers"] = NumOps.FromDouble(_numLayers),
            ["NumHeads"] = NumOps.FromDouble(_numHeads),
            ["NumLags"] = NumOps.FromDouble(_lagIndices.Length),
            ["LastLoss"] = lastLoss
        };
    }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// Performs the forward pass through Lag-Llama.
    /// </summary>
    /// <param name="input">Input tensor with lag features.</param>
    /// <returns>Output tensor with distribution parameters.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Lag-Llama forward pass:
    ///
    /// <b>Step 1: Lag Feature Processing</b>
    /// - Input includes current values and lagged values
    /// - Embedded into hidden dimension
    ///
    /// <b>Step 2: Transformer Processing</b>
    /// - Multiple Llama-style blocks
    /// - Each block: RMSNorm → Attention → RMSNorm → FFN
    /// - Causal attention (only look at past)
    ///
    /// <b>Step 3: Distribution Output</b>
    /// - Final norm
    /// - Project to distribution parameters
    /// - For Student-t: outputs [mu, sigma, nu] for each forecast step
    /// </para>
    /// </remarks>
    private Tensor<T> Forward(Tensor<T> input)
    {
        if (!_useNativeMode)
            return ForecastOnnx(input);

        var current = input;

        // Input embedding
        if (_inputEmbedding is not null)
            current = _inputEmbedding.Forward(current);

        // Process through transformer layers
        foreach (var layer in _transformerLayers)
        {
            current = layer.Forward(current);
        }

        // Final normalization
        if (_finalNorm is not null)
            current = _finalNorm.Forward(current);

        // Distribution output head
        if (_distributionHead is not null)
            current = _distributionHead.Forward(current);

        return current;
    }

    /// <summary>
    /// Performs the backward pass through Lag-Llama.
    /// </summary>
    /// <param name="gradOutput">Gradient from the loss function.</param>
    /// <returns>Gradient with respect to the input.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Backpropagation through Lag-Llama updates all learnable
    /// parameters: embeddings, attention weights, FFN weights, and distribution head.
    /// </para>
    /// </remarks>
    private Tensor<T> Backward(Tensor<T> gradOutput)
    {
        var current = gradOutput;

        // Distribution head backward
        if (_distributionHead is not null)
            current = _distributionHead.Backward(current);

        // Final norm backward
        if (_finalNorm is not null)
            current = _finalNorm.Backward(current);

        // Transformer layers backward (reverse order)
        for (int i = _transformerLayers.Count - 1; i >= 0; i--)
        {
            current = _transformerLayers[i].Backward(current);
        }

        // Input embedding backward
        if (_inputEmbedding is not null)
            current = _inputEmbedding.Backward(current);

        return current;
    }

    #endregion

    #region Inference Methods

    /// <summary>
    /// Performs native mode forecasting.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Uses the neural network layers to produce probabilistic forecasts.
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
    /// <b>For Beginners:</b> Uses the pretrained ONNX model for inference.
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

    #region Distribution Processing

    /// <summary>
    /// Extracts point predictions (means) from distribution parameters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The model outputs distribution parameters.
    /// For point predictions, we extract the mean (mu) parameter.
    /// For Student-t: output is [mu1, sigma1, nu1, mu2, sigma2, nu2, ...]
    /// </para>
    /// </remarks>
    private Tensor<T> ExtractPointPredictions(Tensor<T> distributionParams)
    {
        // Output format: [mu, sigma, nu] repeated for each forecast step (3 params per step)
        // Extract just the mu values (every 3rd value starting at index 0)
        // Preserve batch dimension if present to avoid downstream misindexing

        int paramsPerStep = 3; // mu, sigma, nu for StudentT
        int rank = distributionParams.Shape.Length;
        int batchSize;
        int totalParams;

        if (rank == 1)
        {
            // 1D: [totalParams] - single sample, no batch dimension
            batchSize = 1;
            totalParams = distributionParams.Shape[0];
        }
        else if (rank == 2)
        {
            // 2D: [batchSize, totalParams] - batched distribution params
            batchSize = distributionParams.Shape[0];
            totalParams = distributionParams.Shape[1];
        }
        else if (rank == 3)
        {
            // 3D: [batchSize, forecastHorizon, paramsPerStep]
            batchSize = distributionParams.Shape[0];
            // In this case, mu is at index 0 of the last dimension
            var result3D = new Tensor<T>(new[] { batchSize, _forecastHorizon });
            for (int b = 0; b < batchSize; b++)
            {
                for (int i = 0; i < _forecastHorizon && i < distributionParams.Shape[1]; i++)
                {
                    int srcIdx = b * distributionParams.Shape[1] * distributionParams.Shape[2] + i * distributionParams.Shape[2];
                    result3D.Data.Span[b * _forecastHorizon + i] = distributionParams.Data.Span[srcIdx];
                }
            }
            return result3D;
        }
        else
        {
            // Fallback for other ranks - treat as 1D
            batchSize = 1;
            totalParams = distributionParams.Length;
        }

        // For 1D/2D cases: extract mu values with batch preservation
        var result = new Tensor<T>(batchSize == 1 ? new[] { _forecastHorizon } : new[] { batchSize, _forecastHorizon });

        for (int b = 0; b < batchSize; b++)
        {
            int batchOffset = b * (paramsPerStep * _forecastHorizon);
            for (int i = 0; i < _forecastHorizon; i++)
            {
                int srcIdx = batchOffset + i * paramsPerStep; // mu is at positions 0, 3, 6, ...
                int dstIdx = b * _forecastHorizon + i;
                if (srcIdx < distributionParams.Length && dstIdx < result.Length)
                {
                    result.Data.Span[dstIdx] = distributionParams.Data.Span[srcIdx];
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Samples quantiles from the predicted distribution.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Given distribution parameters and desired quantiles,
    /// compute the values at those quantiles. For example, quantile 0.5 gives the median,
    /// and [0.1, 0.9] gives a 80% prediction interval.
    /// </para>
    /// </remarks>
    private Tensor<T> SampleQuantiles(Tensor<T> distributionParams, double[] quantiles)
    {
        // Determine batch size from distribution parameters
        int paramsPerStep = 3; // mu, sigma, nu for Student-t (or mu, sigma for Normal)
        int totalParams = distributionParams.Length;
        int expectedParamsPerBatch = paramsPerStep * _forecastHorizon;
        int batchSize = Math.Max(1, totalParams / expectedParamsPerBatch);

        // Create batch-aware result tensor
        var result = new Tensor<T>(batchSize == 1
            ? new[] { _forecastHorizon, quantiles.Length }
            : new[] { batchSize, _forecastHorizon, quantiles.Length });

        var pointPreds = ExtractPointPredictions(distributionParams);

        bool isStudentT = string.IsNullOrEmpty(_distributionOutput) ||
                          string.Equals(_distributionOutput, "StudentT", StringComparison.OrdinalIgnoreCase);
        bool isNormal = string.Equals(_distributionOutput, "Normal", StringComparison.OrdinalIgnoreCase);

        for (int b = 0; b < batchSize; b++)
        {
            int batchOffset = b * expectedParamsPerBatch;
            int pointBatchOffset = b * _forecastHorizon;

            for (int q = 0; q < quantiles.Length; q++)
            {
                double quantile = quantiles[q];

                for (int i = 0; i < _forecastHorizon; i++)
                {
                    // Extract mu (mean) for this step
                    int muIdx = pointBatchOffset + i;
                    T mu = muIdx < pointPreds.Length ? pointPreds.Data.Span[muIdx] : NumOps.Zero;

                    // Extract sigma (scale) for this step (index 1 in each 3-param group)
                    int sigmaIdx = batchOffset + i * paramsPerStep + 1;
                    T sigma = sigmaIdx < distributionParams.Length
                        ? distributionParams.Data.Span[sigmaIdx]
                        : NumOps.FromDouble(1.0);

                double adjustmentFactor;

                if (isStudentT)
                {
                    // Extract nu (degrees of freedom) for this step (index 2 in each 3-param group)
                    int nuIdx = batchOffset + i * paramsPerStep + 2;
                    double nu = nuIdx < distributionParams.Length
                        ? NumOps.ToDouble(distributionParams.Data.Span[nuIdx])
                        : 3.0; // Default nu = 3 for heavy tails

                    // Student-t quantile: use approximation based on normal with tail adjustment
                    // For large nu, Student-t approaches Normal; for small nu, tails are heavier
                    if (nu <= 2.0)
                    {
                        // Very heavy tails - variance is undefined, use normal approximation
                        adjustmentFactor = ApproximateInverseNormal(quantile);
                    }
                    else
                    {
                        // Approximate Student-t quantile using scaled normal
                        // t_q ≈ z_q * sqrt(nu / (nu - 2)) for moderate nu
                        double zScore = ApproximateInverseNormal(quantile);
                        double tailScale = Math.Sqrt(nu / (nu - 2));
                        adjustmentFactor = zScore * tailScale;
                    }
                }
                else if (isNormal)
                {
                    // Normal distribution: use standard inverse normal (z-score)
                    adjustmentFactor = ApproximateInverseNormal(quantile);
                }
                else
                {
                    // Unsupported distribution - throw clear error
                    throw new NotSupportedException(
                        $"SampleQuantiles does not support distribution type '{_distributionOutput}'. " +
                        $"Supported types are: StudentT, Normal. For StudentT, nu (degrees of freedom) " +
                        $"is expected at index 2 of each 3-param group in distributionParams.");
                }

                    // quantile_value = mu + adjustment * sigma
                    T adjustment = NumOps.Multiply(sigma, NumOps.FromDouble(adjustmentFactor));

                    // Batch-aware output index
                    int outIdx = batchSize == 1
                        ? i * quantiles.Length + q
                        : b * _forecastHorizon * quantiles.Length + i * quantiles.Length + q;
                    if (outIdx < result.Length)
                        result.Data.Span[outIdx] = NumOps.Add(mu, adjustment);
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Approximates the inverse standard normal CDF.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Converts a probability (like 0.95) to a z-score (like 1.645).
    /// This is a simplified approximation; full implementation would use proper quantile function.
    /// </para>
    /// </remarks>
    private static double ApproximateInverseNormal(double p)
    {
        // Rational approximation for inverse normal (Abramowitz and Stegun)
        if (p <= 0) return double.NegativeInfinity;
        if (p >= 1) return double.PositiveInfinity;
        if (p == 0.5) return 0;

        double t = p < 0.5 ? Math.Sqrt(-2 * Math.Log(p)) : Math.Sqrt(-2 * Math.Log(1 - p));
        double c0 = 2.515517, c1 = 0.802853, c2 = 0.010328;
        double d1 = 1.432788, d2 = 0.189269, d3 = 0.001308;

        double z = t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t);

        return p < 0.5 ? -z : z;
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Shifts input tensor by incorporating predictions for autoregressive forecasting.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> For multi-step forecasting, we slide the window and
    /// add predicted values as new "history" for the next iteration.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ShiftInputWithPredictions(Tensor<T> input, Tensor<T> predictions, int stepsUsed)
    {
        // For Lag-Llama, update lag features after shifting.
        var newInput = new Tensor<T>(input.Shape);

        int featureSize = 1 + _lagIndices.Length; // value + lags
        int batchSize = input.Shape.Length > 1 ? input.Shape[0] : 1;
        int seqLen = input.Shape.Length > 1 ? input.Shape[1] : input.Length / featureSize;

        int steps = Math.Min(stepsUsed, seqLen);
        int shift = steps * featureSize;

        for (int b = 0; b < batchSize; b++)
        {
            int baseOffset = b * seqLen * featureSize;

            // Shift existing data
            for (int i = 0; i < (seqLen * featureSize) - shift; i++)
            {
                int srcIdx = baseOffset + i + shift;
                int dstIdx = baseOffset + i;
                if (srcIdx < input.Length && dstIdx < newInput.Length)
                {
                    newInput.Data.Span[dstIdx] = input.Data.Span[srcIdx];
                }
            }

            // Add predictions at the end (value feature only)
            for (int i = 0; i < steps; i++)
            {
                int predIdx = b * steps + i;
                int targetIdx = baseOffset + (seqLen - steps + i) * featureSize;
                if (predIdx < predictions.Length && targetIdx < newInput.Length)
                {
                    newInput.Data.Span[targetIdx] = predictions.Data.Span[predIdx];
                }
            }

            // Recompute lag features based on updated values
            for (int t = 0; t < seqLen; t++)
            {
                int valueIdx = baseOffset + t * featureSize;

                for (int l = 0; l < _lagIndices.Length; l++)
                {
                    int lag = _lagIndices[l];
                    int lagFeatureIdx = valueIdx + 1 + l;
                    int sourceStep = t - lag;

                    if (lagFeatureIdx >= newInput.Length)
                        continue;

                    if (sourceStep >= 0)
                    {
                        int sourceIdx = baseOffset + sourceStep * featureSize;
                        newInput.Data.Span[lagFeatureIdx] = newInput.Data.Span[sourceIdx];
                    }
                    else
                    {
                        newInput.Data.Span[lagFeatureIdx] = NumOps.Zero;
                    }
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
    /// <b>For Beginners:</b> Combines predictions from multiple iterations into one result.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ConcatenatePredictions(List<Tensor<T>> predictions, int totalSteps)
    {
        var result = new Tensor<T>(new[] { totalSteps });

        int resultIdx = 0;
        int stepsAdded = 0;

        foreach (var pred in predictions)
        {
            int stepsToAdd = Math.Min(_forecastHorizon, totalSteps - stepsAdded);

            for (int i = 0; i < stepsToAdd && resultIdx < totalSteps; i++)
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
    /// Disposes resources used by the Lag-Llama model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Releases the ONNX session and other resources.
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

