using System.IO;
using AiDotNet.Attributes;
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
/// Timer (Generative Pre-Training for Time Series) implementation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// Timer is a generative pre-training approach for time series that uses
/// autoregressive generation to learn rich temporal representations from
/// diverse time series datasets, similar to GPT for language.
/// </para>
/// <para><b>For Beginners:</b> Timer brings GPT-style pre-training to time series:
///
/// <b>The Key Insight:</b>
/// Just like GPT learns language by predicting the next token, Timer learns
/// time series patterns by predicting future values. Pre-training on diverse
/// datasets enables strong zero-shot transfer.
///
/// <b>How It Works:</b>
/// 1. <b>Autoregressive Pre-training:</b> Learn to predict future from past
/// 2. <b>Masked Modeling:</b> Learn to reconstruct masked portions
/// 3. <b>Multi-scale Processing:</b> Handle different temporal granularities
/// 4. <b>Fine-tuning:</b> Adapt to specific domains with minimal data
///
/// <b>Advantages:</b>
/// - Strong zero-shot and few-shot performance
/// - Generalizes across domains and frequencies
/// - Efficient fine-tuning with minimal labeled data
/// - Handles variable sequence lengths
/// </para>
/// <para>
/// <b>Timer-XL Enhancements (2024):</b>
/// Timer-XL extends the original Timer with long-context support and a unified
/// framework for multiple forecasting tasks. Key improvements:
/// - Extended context length (up to 4096 time steps)
/// - Unified multi-task forecasting framework
/// - Improved long-horizon performance
///
/// <b>Reference:</b> Liu et al., "Timer: Generative Pre-Training of Time Series", 2024.
/// https://arxiv.org/abs/2402.02368
/// Timer-XL: "Timer-XL: Long-Context Transformers for Unified Time Series Forecasting", 2024.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Timer model for generative pre-training on time series
/// // Uses autoregressive generation like GPT to learn rich temporal representations
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 512, inputWidth: 1, inputDepth: 1, outputSize: 24);
///
/// // Training mode with autoregressive and masked modeling objectives
/// var model = new Timer&lt;double&gt;(architecture);
///
/// // ONNX inference mode with pre-trained model
/// var onnxModel = new Timer&lt;double&gt;(architecture, "timer_base.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Finance)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Forecasting)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Timer: Generative Pre-trained Transformers Are Large Time Series Models", "https://arxiv.org/abs/2402.02368", Year = 2024, Authors = "Yong Liu, Haoran Zhang, Chenyu Li, Xiangdong Huang, Jianmin Wang, Mingsheng Long")]
public class Timer<T> : TimeSeriesFoundationModelBase<T>
{
    #region Execution Mode

    /// <summary>
    /// Indicates whether the model is running in native mode (true) or ONNX mode (false).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Timer supports two execution modes:
    /// - Native mode: Train or fine-tune the model from scratch
    /// - ONNX mode: Inference using pretrained ONNX model
    /// </para>
    /// </remarks>
    private bool _useNativeMode;

    #endregion


    #region Native Mode Fields

    /// <summary>
    /// Reference to the patch embedding layer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The patch embedding converts raw time series
    /// patches into dense vector representations (tokens) for the transformer.
    /// </para>
    /// </remarks>
    private DenseLayer<T>? _patchEmbedding;

    /// <summary>
    /// References to the transformer decoder layers.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> These GPT-style decoder layers use causal
    /// self-attention to learn patterns from past to future.
    /// </para>
    /// </remarks>
    private List<MultiHeadAttentionLayer<T>>? _transformerLayers;

    /// <summary>
    /// Reference to the generation head layer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The generation head projects the transformer
    /// output to forecast values (like the LM head in GPT).
    /// </para>
    /// </remarks>
    private DenseLayer<T>? _generationHead;

    #endregion

    #region Shared Fields

    /// <summary>
    /// The optimizer used for training.
    /// </summary>
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <summary>
    /// The loss function used for training.
    /// </summary>
    private readonly ILossFunction<T> _lossFunction;
    private readonly TimerOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// Context length for the input sequence.
    /// </summary>
    private int _contextLength;

    /// <summary>
    /// Forecast horizon for predictions.
    /// </summary>
    private int _forecastHorizon;

    /// <summary>
    /// Patch length for tokenization.
    /// </summary>
    private int _patchLength;

    /// <summary>
    /// Patch stride.
    /// </summary>
    private int _patchStride;

    /// <summary>
    /// Hidden dimension size.
    /// </summary>
    private int _hiddenDimension;

    /// <summary>
    /// Number of transformer layers.
    /// </summary>
    private int _numLayers;

    /// <summary>
    /// Number of attention heads.
    /// </summary>
    private int _numHeads;

    /// <summary>
    /// Dropout rate.
    /// </summary>
    private double _dropout;

    /// <summary>
    /// Mask ratio for masked modeling.
    /// </summary>
    private double _maskRatio;

    /// <summary>
    /// Whether to use autoregressive decoding.
    /// </summary>
    private bool _useAutoregressiveDecoding;

    /// <summary>
    /// Temperature for sampling during generation.
    /// </summary>
    private double _generationTemperature;

    /// <summary>
    /// Number of input features.
    /// </summary>
    private int _numFeatures;

    /// <summary>
    /// RevIN (reversible instance normalization, Kim et al. 2022) statistics.
    /// Timer normalizes each input series before patch embedding and restores the
    /// level on the output so distinct input scales produce distinct forecasts.
    /// Keyed per instance (batch row).
    /// </summary>
    private Vector<T> _revinMean = new Vector<T>(0);
    private Vector<T> _revinStd = new Vector<T>(0);

    /// <summary>
    /// Number of patches.
    /// </summary>
    private int _numPatches;

    #endregion

    #region IForecastingModel Properties

    /// <inheritdoc/>
    public override int SequenceLength => _contextLength;

    /// <inheritdoc/>
    public override int PredictionHorizon => _forecastHorizon;

    /// <inheritdoc/>
    public override int NumFeatures => _numFeatures;

    /// <inheritdoc/>
    public override int PatchSize => _patchLength;

    /// <inheritdoc/>
    public override int Stride => _patchStride;

    /// <inheritdoc/>
    public override bool IsChannelIndependent => true;

    /// <inheritdoc/>
    public override bool UseNativeMode => _useNativeMode;

    /// <summary>
    /// Gets the mask ratio for masked modeling.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The fraction of patches masked during pre-training.
    /// </para>
    /// </remarks>
    public double MaskRatio => _maskRatio;

    /// <summary>
    /// Gets whether autoregressive decoding is used.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Autoregressive decoding generates one step at a time.
    /// </para>
    /// </remarks>
    public bool UseAutoregressiveDecoding => _useAutoregressiveDecoding;

    /// <summary>
    /// Gets the generation temperature.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Temperature controls randomness in generation.
    /// </para>
    /// </remarks>
    public double GenerationTemperature => _generationTemperature;

    /// <inheritdoc/>
    public override FoundationModelSize ModelSize => FoundationModelSize.Base;

    /// <inheritdoc/>
    public override int MaxContextLength => _contextLength;

    /// <inheritdoc/>
    public override int MaxPredictionHorizon => _forecastHorizon;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance using an ONNX pretrained model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="options">Configuration options for Timer.</param>
    /// <param name="optimizer">Optional optimizer.</param>
    /// <param name="lossFunction">Optional custom loss function.</param>
    /// <exception cref="ArgumentException">Thrown when onnxModelPath is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown when ONNX model file doesn't exist.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to load a pretrained Timer model
    /// for inference. The ONNX model contains all the weights learned during pre-training.
    /// </para>
    /// </remarks>
    public Timer(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        TimerOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        options ??= new TimerOptions<T>();
        _options = options;
        Options = _options;

        _useNativeMode = false;
        OnnxModelPath = onnxModelPath;
        OnnxSession = new InferenceSession(onnxModelPath);

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        _contextLength = options.ContextLength;
        _forecastHorizon = options.ForecastHorizon;
        _patchLength = options.PatchLength;
        _patchStride = options.PatchStride;
        _hiddenDimension = options.HiddenDimension;
        _numLayers = options.NumLayers;
        _numHeads = options.NumHeads;
        _dropout = options.DropoutRate;
        _maskRatio = options.MaskRatio;
        _useAutoregressiveDecoding = options.UseAutoregressiveDecoding;
        _generationTemperature = options.GenerationTemperature;
        _numFeatures = 1;

        if (_patchLength < 1)
            throw new ArgumentOutOfRangeException(nameof(options.PatchLength), "Patch length must be at least 1.");
        if (_patchStride < 1)
            throw new ArgumentOutOfRangeException(nameof(options.PatchStride), "Patch stride must be at least 1.");
        if (_patchLength > _contextLength)
            throw new ArgumentOutOfRangeException(nameof(options.PatchLength), "Patch length cannot exceed context length.");
        if (_maskRatio < 0 || _maskRatio >= 1)
            throw new ArgumentOutOfRangeException(nameof(options.MaskRatio), "Mask ratio must be between 0 and 1.");

        // Timer tokenizes the context into NON-OVERLAPPING patches (Liu et al.
        // 2024 S3 single-series format), so the patch count matches the
        // ReshapeLayer the layer helper builds: contextLength / patchLength.
        _numPatches = _contextLength / _patchLength;

        InitializeLayers();
    }

    /// <summary>
    /// Initializes a new instance in native mode for training.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">Configuration options for Timer.</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="optimizer">Optional optimizer.</param>
    /// <param name="lossFunction">Optional custom loss function.</param>
    /// <exception cref="ArgumentNullException">Thrown when architecture is null.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to train Timer from scratch or
    /// fine-tune on your specific data. Generative pre-training teaches the model to
    /// predict future values from past values.
    /// </para>
    /// </remarks>
    public Timer(
        NeuralNetworkArchitecture<T> architecture,
        TimerOptions<T>? options = null,
        int numFeatures = 1,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        // Validate numFeatures before any assignments to prevent invalid layer shapes
        if (numFeatures <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(numFeatures),
                numFeatures,
                "numFeatures must be greater than 0. Timer requires at least one input feature.");
        }

        options ??= new TimerOptions<T>();
        _options = options;
        Options = _options;

        _useNativeMode = true;

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        _contextLength = options.ContextLength;
        _forecastHorizon = options.ForecastHorizon;
        _patchLength = options.PatchLength;
        _patchStride = options.PatchStride;
        _hiddenDimension = options.HiddenDimension;
        _numLayers = options.NumLayers;
        _numHeads = options.NumHeads;
        _dropout = options.DropoutRate;
        _maskRatio = options.MaskRatio;
        _useAutoregressiveDecoding = options.UseAutoregressiveDecoding;
        _generationTemperature = options.GenerationTemperature;
        _numFeatures = numFeatures;

        if (_patchLength < 1)
            throw new ArgumentOutOfRangeException(nameof(options.PatchLength), "Patch length must be at least 1.");
        if (_patchStride < 1)
            throw new ArgumentOutOfRangeException(nameof(options.PatchStride), "Patch stride must be at least 1.");
        if (_patchLength > _contextLength)
            throw new ArgumentOutOfRangeException(nameof(options.PatchLength), "Patch length cannot exceed context length.");
        if (_maskRatio < 0 || _maskRatio >= 1)
            throw new ArgumentOutOfRangeException(nameof(options.MaskRatio), "Mask ratio must be between 0 and 1.");

        // Timer tokenizes the context into NON-OVERLAPPING patches (Liu et al.
        // 2024 S3 single-series format), so the patch count matches the
        // ReshapeLayer the layer helper builds: contextLength / patchLength.
        _numPatches = _contextLength / _patchLength;

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes the model layers based on configuration.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up the Timer architecture including
    /// patch embedding, GPT-style decoder stack, and generation head.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultTimerLayers(
                Architecture, _contextLength, _forecastHorizon, _numFeatures,
                _patchLength, _patchStride, _hiddenDimension, _numLayers, _numHeads, _dropout));

            ExtractLayerReferences();
        }
    }

    /// <summary>
    /// Extracts references to key layers for efficient access.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> After creating all layers, we keep direct references
    /// to important ones for quick access during computation.
    /// </para>
    /// </remarks>
    private void ExtractLayerReferences()
    {
        _patchEmbedding = Layers.OfType<DenseLayer<T>>().FirstOrDefault();
        _transformerLayers = Layers.OfType<MultiHeadAttentionLayer<T>>().ToList();
        _generationHead = Layers.OfType<DenseLayer<T>>().LastOrDefault();
    }

    /// <summary>
    /// Validates that custom layers meet Timer requirements.
    /// </summary>
    /// <param name="layers">The layers to validate.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> When using custom layers, we ensure they include
    /// the necessary components for the Timer architecture (transformer layers).
    /// </para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);

        var attentionCount = layers.OfType<MultiHeadAttentionLayer<T>>().Count();
        if (attentionCount < 1)
        {
            throw new ArgumentException(
                "Timer requires at least one MultiHeadAttentionLayer for the transformer decoder.");
        }
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <inheritdoc/>
    public override bool SupportsTraining => _useNativeMode;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Timer model, Predict produces predictions from input data. This is the main inference step of the Timer architecture.
    /// </para>
    /// </remarks>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        return _useNativeMode ? Forward(input) : ForecastOnnx(input);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Timer model, Train performs a training step. This updates the Timer architecture so it learns from data.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is only supported in native mode.");

        // Issue #1166: the old body computed a loss + gradient and then
        // called _optimizer.UpdateParameters(Layers) without a backward
        // pass, so every layer's UpdateParameters threw "Backward pass
        // must be called before updating parameters." Delegate to
        // FinancialModelBase.Train — it routes through the tape-based
        // NeuralNetworkBase.TrainWithTape flow (GradientTape forward +
        // tape.ComputeGradients + optimizer.Step) that every other
        // NeuralNetworkBase subclass uses.
        base.Train(input, target);
    }

    /// <summary>
    /// Timer training-mode forward. The tape training flow would otherwise feed
    /// the raw rank-1 context straight to the patch <c>ReshapeLayer</c> (which
    /// needs a leading batch axis) and skip RevIN; routing through
    /// <see cref="Forward"/> applies the same normalization, batch reshape,
    /// horizon slice and denormalization as inference while keeping training
    /// mode (dropout) active.
    /// </summary>
    protected override Tensor<T> ForwardNativeForTraining(Tensor<T> input)
    {
        return Forward(input);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Timer model, UpdateParameters updates internal parameters or state. This keeps the Timer architecture aligned with the latest values.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> gradients)
    {
        // Parameters are updated through the optimizer in Train()
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Timer model, GetModelMetadata performs a supporting step in the workflow. It keeps the Timer architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "Timer" },
                { "ContextLength", _contextLength },
                { "ForecastHorizon", _forecastHorizon },
                { "PatchLength", _patchLength },
                { "PatchStride", _patchStride },
                { "HiddenDimension", _hiddenDimension },
                { "NumLayers", _numLayers },
                { "NumHeads", _numHeads },
                { "MaskRatio", _maskRatio },
                { "UseAutoregressiveDecoding", _useAutoregressiveDecoding },
                { "GenerationTemperature", _generationTemperature },
                { "UseNativeMode", _useNativeMode },
                { "ParameterCount", GetParameterCount() }
            },
            ModelData = _useNativeMode ? this.Serialize() : Array.Empty<byte>()
        };
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Timer model, CreateNewInstance builds and wires up model components. This sets up the Timer architecture before use.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var options = new TimerOptions<T>
        {
            ContextLength = _contextLength,
            ForecastHorizon = _forecastHorizon,
            PatchLength = _patchLength,
            PatchStride = _patchStride,
            HiddenDimension = _hiddenDimension,
            NumLayers = _numLayers,
            NumHeads = _numHeads,
            DropoutRate = _dropout,
            MaskRatio = _maskRatio,
            UseAutoregressiveDecoding = _useAutoregressiveDecoding,
            GenerationTemperature = _generationTemperature
        };

        return new Timer<T>(Architecture, options, _numFeatures);
    }

    /// <summary>
    /// Writes Timer-specific configuration during serialization.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Saves all the configuration needed to reconstruct this model.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_contextLength);
        writer.Write(_forecastHorizon);
        writer.Write(_patchLength);
        writer.Write(_patchStride);
        writer.Write(_hiddenDimension);
        writer.Write(_numLayers);
        writer.Write(_numHeads);
        writer.Write(_dropout);
        writer.Write(_maskRatio);
        writer.Write(_useAutoregressiveDecoding);
        writer.Write(_generationTemperature);
        writer.Write(_numFeatures);
    }

    /// <summary>
    /// Reads Timer-specific configuration during deserialization.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Loads the configuration that was saved during serialization.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _contextLength = reader.ReadInt32();
        _forecastHorizon = reader.ReadInt32();
        _patchLength = reader.ReadInt32();
        _patchStride = reader.ReadInt32();
        _hiddenDimension = reader.ReadInt32();
        _numLayers = reader.ReadInt32();
        _numHeads = reader.ReadInt32();
        _dropout = reader.ReadDouble();
        _maskRatio = reader.ReadDouble();
        _useAutoregressiveDecoding = reader.ReadBoolean();
        _generationTemperature = reader.ReadDouble();
        _numFeatures = reader.ReadInt32();

        // Recompute _numPatches from deserialized values to keep derived field in sync
        // Use same logic as constructor: (_contextLength - _patchLength) / _patchStride + 1
        int computedPatches = (_contextLength - _patchLength) / _patchStride + 1;
        _numPatches = Math.Max(0, computedPatches); // Clamp to zero if negative
    }

    #endregion

    #region IForecastingModel Implementation

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Timer model, Forecast produces predictions from input data. This is the main inference step of the Timer architecture.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
    {
        var output = _useNativeMode ? Forward(historicalData) : ForecastOnnx(historicalData);

        // For quantile forecasts, we generate samples through temperature scaling
        // Only supported in native mode (requires MC dropout sampling)
        if (quantiles is not null && quantiles.Length > 0 && _useNativeMode)
        {
            return GenerateQuantilePredictions(historicalData, quantiles);
        }

        return output;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Timer model, AutoregressiveForecast produces predictions from input data. This is the main inference step of the Timer architecture.
    /// </para>
    /// </remarks>
    public override Tensor<T> AutoregressiveForecast(Tensor<T> input, int steps)
    {
        // Autoregressive decoding only works in native mode (uses Forward directly)
        if (_useAutoregressiveDecoding && _useNativeMode)
        {
            return AutoregressiveGenerate(input, steps);
        }

        // Fall back to standard multi-step forecasting (works in both modes)
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
    /// <b>For Beginners:</b> In the Timer model, Evaluate performs a supporting step in the workflow. It keeps the Timer architecture pipeline consistent.
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
    /// <b>For Beginners:</b> In the Timer model, ApplyInstanceNormalization performs a supporting step in the workflow. It keeps the Timer architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
    {
        // RevIN forward (Kim et al. 2022): subtract each instance's mean and
        // divide by its std so the decoder sees a normalized series. The
        // transformer's LayerNorm would otherwise discard the input's level,
        // making constant inputs of different magnitudes collapse to the same
        // forecast. Stats are taken over every non-batch element of each row so
        // this works for 1-D [seqLen], 2-D [batch, seqLen] and 3-D
        // [batch, seqLen, features] inputs alike.
        int batchSize = input.Shape.Length > 1 ? input.Shape[0] : 1;
        int instanceSize = batchSize > 0 ? input.Length / batchSize : input.Length;
        if (instanceSize <= 0)
            return input;

        var result = new Tensor<T>(input._shape);
        _revinMean = new Vector<T>(batchSize);
        _revinStd = new Vector<T>(batchSize);

        for (int b = 0; b < batchSize; b++)
        {
            int start = b * instanceSize;

            T mean = NumOps.Zero;
            for (int t = 0; t < instanceSize; t++)
                mean = NumOps.Add(mean, input[start + t]);
            mean = NumOps.Divide(mean, NumOps.FromDouble(instanceSize));

            T variance = NumOps.Zero;
            for (int t = 0; t < instanceSize; t++)
            {
                var diff = NumOps.Subtract(input[start + t], mean);
                variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
            }
            variance = NumOps.Divide(variance, NumOps.FromDouble(instanceSize));
            T std = NumOps.Sqrt(NumOps.Add(variance, NumOps.FromDouble(1e-5)));

            _revinMean[b] = mean;
            _revinStd[b] = std;

            for (int t = 0; t < instanceSize; t++)
                result.Data.Span[start + t] = NumOps.Divide(NumOps.Subtract(input[start + t], mean), std);
        }

        return result;
    }

    /// <summary>
    /// RevIN reverse step: restores each instance's mean/std to the forecast so it
    /// is expressed on the input's original scale (Kim et al. 2022). The forecast
    /// rows align with the instances normalized in <see cref="ApplyInstanceNormalization"/>.
    /// </summary>
    private Tensor<T> DenormalizeForecast(Tensor<T> forecast)
    {
        int batch = forecast.Shape.Length > 1 ? forecast.Shape[0] : 1;
        if (_revinMean.Length != batch || forecast.Length % batch != 0)
            return forecast;

        // Per-instance scale/shift as [batch, 1] constants that broadcast over the
        // forecast's trailing dimension. The multiply/add go through the Engine so
        // the forecast stays on the autodiff tape (a manual element fill would
        // detach it and starve the forecast head of gradients).
        var meanT = new Tensor<T>(new[] { batch, 1 });
        var stdT = new Tensor<T>(new[] { batch, 1 });
        for (int b = 0; b < batch; b++)
        {
            meanT.Data.Span[b] = _revinMean[b];
            stdT.Data.Span[b] = _revinStd[b];
        }

        bool reshaped = forecast.Rank != 2;
        var work = reshaped ? Engine.Reshape(forecast, new[] { batch, forecast.Length / batch }) : forecast;
        var scaled = Engine.TensorBroadcastMultiply(work, stdT);
        var shifted = Engine.TensorBroadcastAdd(scaled, meanT);
        return reshaped ? Engine.Reshape(shifted, forecast._shape) : shifted;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Timer model, GetFinancialMetrics calculates evaluation metrics. This summarizes how the Timer architecture is performing.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> GetFinancialMetrics()
    {
        T lastLoss = LastLoss is not null ? LastLoss : NumOps.Zero;

        return new Dictionary<string, T>
        {
            ["ContextLength"] = NumOps.FromDouble(_contextLength),
            ["ForecastHorizon"] = NumOps.FromDouble(_forecastHorizon),
            ["PatchLength"] = NumOps.FromDouble(_patchLength),
            ["NumPatches"] = NumOps.FromDouble(_numPatches),
            ["HiddenDimension"] = NumOps.FromDouble(_hiddenDimension),
            ["MaskRatio"] = NumOps.FromDouble(_maskRatio),
            ["LastLoss"] = lastLoss
        };
    }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// Performs the forward pass through the network.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Output tensor with forecast values.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The forward pass:
    /// 1. Embeds input patches into tokens
    /// 2. Processes through GPT-style decoder layers
    /// 3. Projects to forecast values
    /// </para>
    /// </remarks>
    /// <inheritdoc/>
    /// <remarks>
    /// Mirrors <see cref="Forward"/>'s preprocessing so the captured activations match the
    /// real forward pass: a bare rank-1 context is RevIN-normalized and given a leading batch
    /// axis before it reaches the patch <c>ReshapeLayer</c> (which would otherwise misread the
    /// context vector as a multi-row batch and fail to reshape into patches).
    /// </remarks>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        if (!_useNativeMode)
            return base.GetNamedLayerActivations(input);

        var activations = new Dictionary<string, Tensor<T>>();

        var current = ApplyInstanceNormalization(input);
        if (current.Rank == 1)
            current = current.Reshape(new[] { 1, current.Length });

        for (int i = 0; i < Layers.Count; i++)
        {
            current = Layers[i].Forward(current);
            activations[$"Layer_{i}_{Layers[i].GetType().Name}"] = current.Clone();
        }

        return activations;
    }

    private Tensor<T> Forward(Tensor<T> input)
    {
        // RevIN forward: normalize the context so the decoder sees a zero-mean
        // unit-std series, then restore the level after the head so distinct
        // input scales produce distinct forecasts (Kim et al. 2022).
        var current = ApplyInstanceNormalization(input);

        // The patch ReshapeLayer needs a leading batch axis; a bare rank-1
        // context [contextLength] would be misread as a contextLength-row batch.
        bool addedBatchDim = false;
        if (current.Rank == 1)
        {
            current = current.Reshape(new[] { 1, current.Length });
            addedBatchDim = true;
        }

        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }

        // The per-patch head + Flatten produce [B, numPatches * patchLength] =
        // the generated continuation. Take the last forecastHorizon values
        // (the most-recent patches predict the furthest future). Tape-aware
        // narrow keeps the head's gradient connected.
        int total = current.Shape.Length >= 2 ? current.Shape[current.Shape.Length - 1] : current.Length;
        if (total > _forecastHorizon)
        {
            int offset = total - _forecastHorizon;
            current = Engine.TensorNarrow(current, current.Rank - 1, offset, _forecastHorizon);
        }

        // RevIN reverse: restore the input's per-instance level/scale.
        current = DenormalizeForecast(current);

        if (addedBatchDim && current.Rank == 2 && current.Shape[0] == 1)
            current = Engine.Reshape(current, new[] { current.Shape[1] });

        return current;
    }

    /// <summary>
    /// Performs ONNX-based inference for forecasting.
    /// </summary>
    /// <param name="input">Input tensor with historical data.</param>
    /// <returns>Forecast tensor with predictions.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> ONNX mode uses the pretrained model for fast inference.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ForecastOnnx(Tensor<T> input)
    {
        if (OnnxSession is null)
        {
            throw new InvalidOperationException("ONNX session not initialized.");
        }

        var inputData = new float[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            inputData[i] = Convert.ToSingle(NumOps.ToDouble(input.Data.Span[i]));
        }

        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input._shape);
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

    #region Model-Specific Processing

    /// <summary>
    /// Performs autoregressive generation step by step.
    /// </summary>
    /// <param name="input">Input tensor with historical data.</param>
    /// <param name="steps">Number of steps to generate.</param>
    /// <returns>Generated forecast tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Like GPT generating text token by token, Timer
    /// can generate time series values one step at a time. Each generated value
    /// becomes input for predicting the next value.
    /// </para>
    /// </remarks>
    private Tensor<T> AutoregressiveGenerate(Tensor<T> input, int steps)
    {
        // Normalize input to 3D [1, seqLen, features] for Forward pass
        Tensor<T> normalizedInput;
        int seqLen;
        int features;

        if (input.Shape.Length == 1)
        {
            // 1D input [seqLen] - reshape to [1, seqLen, 1]
            seqLen = input.Shape[0];
            features = 1;
            normalizedInput = input.Reshape(new[] { 1, seqLen, features });
        }
        else if (input.Shape.Length == 2)
        {
            // 2D input [batch, seqLen] or [seqLen, features]
            if (input.Shape[0] == 1)
            {
                // [1, seqLen] - reshape to [1, seqLen, 1]
                seqLen = input.Shape[1];
                features = 1;
                normalizedInput = input.Reshape(new[] { 1, seqLen, features });
            }
            else
            {
                // [seqLen, features] - reshape to [1, seqLen, features]
                seqLen = input.Shape[0];
                features = input.Shape[1];
                normalizedInput = input.Reshape(new[] { 1, seqLen, features });
            }
        }
        else if (input.Shape.Length == 3 && input.Shape[0] == 1)
        {
            // Already 3D [1, seqLen, features]
            seqLen = input.Shape[1];
            features = input.Shape[2];
            normalizedInput = input;
        }
        else
        {
            throw new InvalidOperationException(
                $"Autoregressive generation currently supports a single univariate series with shape [context_length], " +
                $"[1, context_length], [context_length, features], or [1, context_length, features]. " +
                $"Got input shape [{string.Join(", ", input._shape)}].");
        }

        // Validate that normalized input sequence length matches _contextLength
        // The shifting loop uses _contextLength to index into currentInput.Data.Span
        int actualSeqLength = normalizedInput.Shape[1];
        if (actualSeqLength != _contextLength)
        {
            // Handle mismatch: pad/truncate to _contextLength or throw
            if (actualSeqLength < _contextLength)
            {
                // Pad with zeros at the beginning
                var paddedInput = new Tensor<T>(new[] { 1, _contextLength, features });
                int offset = _contextLength - actualSeqLength;
                for (int t = 0; t < actualSeqLength; t++)
                {
                    for (int f = 0; f < features; f++)
                    {
                        paddedInput.Data.Span[(offset + t) * features + f] = normalizedInput.Data.Span[t * features + f];
                    }
                }
                normalizedInput = paddedInput;
                seqLen = _contextLength;
            }
            else
            {
                // Truncate to last _contextLength elements
                var truncatedInput = new Tensor<T>(new[] { 1, _contextLength, features });
                int offset = actualSeqLength - _contextLength;
                for (int t = 0; t < _contextLength; t++)
                {
                    for (int f = 0; f < features; f++)
                    {
                        truncatedInput.Data.Span[t * features + f] = normalizedInput.Data.Span[(offset + t) * features + f];
                    }
                }
                normalizedInput = truncatedInput;
                seqLen = _contextLength;
            }
        }

        var result = new Tensor<T>(new[] { 1, steps, 1 });
        var currentInput = normalizedInput;
        var rand = RandomHelper.CreateSecureRandom();

        for (int step = 0; step < steps; step++)
        {
            // Get next value prediction - pass 3D input to Forward
            var prediction = Forward(currentInput);

            // Apply temperature scaling and sample
            T nextValue;
            if (_generationTemperature != 1.0 && _generationTemperature > 0)
            {
                // Scale by temperature and add some randomness
                double predValue = NumOps.ToDouble(prediction.Data.Span[0]);
                double noise = (rand.NextDouble() - 0.5) * _generationTemperature * 0.1;
                nextValue = NumOps.FromDouble(predValue + noise);
            }
            else
            {
                nextValue = prediction.Data.Span[0];
            }

            result.Data.Span[step] = nextValue;

            // Shift input and append generated value - currentInput is [1, seqLen, features]
            if (step < steps - 1)
            {
                var shifted = new Tensor<T>(currentInput._shape);
                int actualSeqLen = currentInput.Shape[1];
                int actualFeatures = currentInput.Shape[2];

                // Shift along time dimension: copy data from t+1 to t
                for (int t = 0; t < actualSeqLen - 1; t++)
                {
                    for (int f = 0; f < actualFeatures; f++)
                    {
                        int srcIdx = (t + 1) * actualFeatures + f;
                        int dstIdx = t * actualFeatures + f;
                        shifted.Data.Span[dstIdx] = currentInput.Data.Span[srcIdx];
                    }
                }

                // Append generated value at the last time step (first feature)
                int lastTimeIdx = (actualSeqLen - 1) * actualFeatures;
                shifted.Data.Span[lastTimeIdx] = nextValue;

                // Zero out remaining features at last time step (if any)
                for (int f = 1; f < actualFeatures; f++)
                {
                    shifted.Data.Span[lastTimeIdx + f] = NumOps.Zero;
                }

                currentInput = shifted;
            }
        }

        return result;
    }

    /// <summary>
    /// Generates quantile predictions through temperature-based sampling.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <param name="quantiles">Quantile levels to compute.</param>
    /// <returns>Quantile predictions tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> For uncertainty estimation, we generate multiple
    /// samples with different temperatures and compute quantiles from them.
    /// </para>
    /// </remarks>
    private Tensor<T> GenerateQuantilePredictions(Tensor<T> input, double[] quantiles)
    {
        // Validate quantiles input
        if (quantiles is null || quantiles.Length == 0)
        {
            throw new ArgumentException(
                "GenerateQuantilePredictions requires non-null, non-empty quantiles array.",
                nameof(quantiles));
        }

        // Validate each quantile is within [0, 1]
        for (int i = 0; i < quantiles.Length; i++)
        {
            if (quantiles[i] < 0.0 || quantiles[i] > 1.0)
            {
                throw new ArgumentException(
                    $"GenerateQuantilePredictions: quantile at index {i} is {quantiles[i]}, " +
                    $"but must be within [0, 1].",
                    nameof(quantiles));
            }
        }

        int numSamples = 100;
        var samples = new List<Tensor<T>>();
        var rand = RandomHelper.CreateSecureRandom();

        // Generate samples with temperature variation
        for (int s = 0; s < numSamples; s++)
        {
            // Vary temperature for diversity
            double temp = 0.5 + rand.NextDouble() * 1.5; // Range [0.5, 2.0]

            // Use dropout for additional diversity
            Tensor<T> sample;
            SetTrainingMode(true);
            try
            {
                sample = Forward(input);
            }
            finally
            {
                SetTrainingMode(false);
            }

            // Apply temperature scaling
            var scaled = new Tensor<T>(sample._shape);
            for (int i = 0; i < sample.Length; i++)
            {
                double val = NumOps.ToDouble(sample.Data.Span[i]);
                double noise = (rand.NextDouble() - 0.5) * temp * 0.1;
                scaled.Data.Span[i] = NumOps.FromDouble(val + noise);
            }

            samples.Add(scaled);
        }

        // Compute quantiles
        var result = new Tensor<T>(new[] { 1, _forecastHorizon, quantiles.Length });

        for (int t = 0; t < _forecastHorizon; t++)
        {
            var values = new List<double>();
            foreach (var sample in samples)
            {
                if (t < sample.Length)
                {
                    values.Add(NumOps.ToDouble(sample.Data.Span[t]));
                }
            }

            // Handle case where Forward returns fewer than _forecastHorizon values
            if (values.Count == 0)
            {
                // No samples available for this time step - fill with NaN/zero
                for (int q = 0; q < quantiles.Length; q++)
                {
                    result.Data.Span[t * quantiles.Length + q] = NumOps.Zero;
                }
                continue;
            }

            values.Sort();

            for (int q = 0; q < quantiles.Length; q++)
            {
                // Use safe clamping to avoid out-of-range indices
                int idx = (int)(quantiles[q] * values.Count);
                idx = Math.Max(0, Math.Min(idx, values.Count - 1));
                result.Data.Span[t * quantiles.Length + q] = NumOps.FromDouble(values[idx]);
            }
        }

        return result;
    }

    /// <summary>
    /// Performs masked modeling pre-training step.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <param name="mask">Mask tensor indicating which patches to mask.</param>
    /// <returns>Reconstructed tensor with masked patches predicted.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Masked modeling is one of Timer's pre-training objectives.
    /// Some patches are hidden (masked) and the model learns to predict them from
    /// the visible patches. This helps learn bidirectional context.
    /// </para>
    /// </remarks>
    public Tensor<T> MaskedPretraining(Tensor<T> input, Tensor<T>? mask = null)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Masked pre-training is only supported in native mode.");

        // Generate random mask if not provided
        if (mask is null)
        {
            mask = GenerateRandomMask(input);
        }

        // Apply mask to input
        var maskedInput = new Tensor<T>(input._shape);
        for (int i = 0; i < input.Length; i++)
        {
            var maskVal = i < mask.Length ? NumOps.ToDouble(mask.Data.Span[i]) : 1.0;
            maskedInput.Data.Span[i] = maskVal > 0.5 ? input.Data.Span[i] : NumOps.Zero;
        }

        // Forward pass through the network
        return Forward(maskedInput);
    }

    /// <summary>
    /// Generates a random mask tensor for masked pre-training.
    /// </summary>
    /// <param name="input">Input tensor to generate mask for.</param>
    /// <returns>Mask tensor with 1 for visible and 0 for masked positions.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a random mask where a fraction (mask ratio)
    /// of positions are set to 0 (masked) and the rest to 1 (visible).
    /// </para>
    /// </remarks>
    private Tensor<T> GenerateRandomMask(Tensor<T> input)
    {
        var mask = new Tensor<T>(input._shape);
        var rand = RandomHelper.CreateSecureRandom();

        for (int i = 0; i < input.Length; i++)
        {
            mask.Data.Span[i] = rand.NextDouble() > _maskRatio
                ? NumOps.FromDouble(1.0)
                : NumOps.FromDouble(0.0);
        }

        return mask;
    }

    /// <summary>
    /// Shifts input tensor by appending predictions.
    /// </summary>
    /// <param name="input">Original input tensor.</param>
    /// <param name="predictions">Predictions to append.</param>
    /// <param name="stepsUsed">Number of prediction steps to use.</param>
    /// <returns>Shifted input tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> For multi-step forecasting, we need to update
    /// the input with predictions so we can forecast further.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ShiftInputWithPredictions(Tensor<T> input, Tensor<T> predictions, int stepsUsed)
    {
        // Guard: ShiftInputWithPredictions currently supports single univariate series only
        if (input.Shape.Length != 1 || _numFeatures != 1)
        {
            throw new InvalidOperationException(
                $"ShiftInputWithPredictions currently supports a single univariate series with shape [context_length]. " +
                $"Got input shape [{string.Join(", ", input._shape)}] with numFeatures={_numFeatures}.");
        }

        // Validate input length matches expected _contextLength
        if (input.Length != _contextLength)
        {
            throw new InvalidOperationException(
                $"ShiftInputWithPredictions: input length ({input.Length}) does not match expected " +
                $"context length ({_contextLength}). Ensure input tensor has shape [_contextLength].");
        }

        var result = new Tensor<T>(input._shape);
        int contextLen = _contextLength;
        int steps = Math.Min(stepsUsed, contextLen);

        // Shift old values left
        for (int i = 0; i < contextLen - steps; i++)
        {
            result.Data.Span[i] = input.Data.Span[i + steps];
        }

        // Append predictions
        for (int i = 0; i < steps && i < predictions.Length; i++)
        {
            result.Data.Span[contextLen - steps + i] = predictions.Data.Span[i];
        }

        return result;
    }

    /// <summary>
    /// Concatenates multiple prediction tensors into a single tensor.
    /// </summary>
    /// <param name="predictions">List of prediction tensors.</param>
    /// <param name="totalSteps">Total number of steps to include.</param>
    /// <returns>Concatenated predictions tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> When doing forecasting in chunks, we combine
    /// all predictions into one final result.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ConcatenatePredictions(List<Tensor<T>> predictions, int totalSteps)
    {
        var result = new Tensor<T>(new[] { 1, totalSteps, 1 });
        int position = 0;

        foreach (var pred in predictions)
        {
            int toCopy = Math.Min(pred.Length, totalSteps - position);
            for (int i = 0; i < toCopy; i++)
            {
                result.Data.Span[position + i] = pred.Data.Span[i];
            }
            position += toCopy;
        }

        return result;
    }

    #endregion

    #region IDisposable

    /// <summary>
    /// Releases resources used by the model.
    /// </summary>
    /// <param name="disposing">True if called from Dispose method.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This ensures proper cleanup of resources,
    /// especially the ONNX session which uses native memory.
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


