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
/// TimesFM (Time Series Foundation Model) for zero-shot time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// TimesFM is Google's foundation model for time series forecasting. It uses a decoder-only
/// transformer architecture pre-trained on a massive dataset spanning diverse domains, enabling
/// zero-shot forecasting without task-specific training.
/// </para>
/// <para>
/// <b>For Beginners:</b> TimesFM is a revolutionary approach to time series forecasting:
///
/// <b>Foundation Model Concept:</b>
/// Just like GPT learns language patterns from vast text data, TimesFM learns time series
/// patterns from millions of diverse series:
/// - Weather data (temperature, precipitation, wind)
/// - Financial data (stock prices, exchange rates)
/// - Retail data (sales, inventory, demand)
/// - Energy data (consumption, production, prices)
///
/// <b>Zero-Shot Capability:</b>
/// The "zero-shot" term means no training required for new tasks:
/// - Traditional models: Train specifically for your data
/// - TimesFM: Works immediately on any time series
/// - Just provide history → get forecasts
///
/// <b>Patching Innovation:</b>
/// TimesFM groups consecutive time steps into "patches":
/// - Example: 512 time steps → 16 patches of 32 steps each
/// - Each patch becomes one token for the transformer
/// - Benefits: Longer context, faster processing, captures local patterns
///
/// <b>Decoder-Only Architecture:</b>
/// Like GPT, TimesFM uses causal (one-directional) attention:
/// - Each position only attends to earlier positions
/// - Naturally suited for autoregressive forecasting
/// - Generates predictions step by step
///
/// <b>When TimesFM Excels:</b>
/// - Quick prototyping without model training
/// - Cross-domain forecasting (same model for different data types)
/// - Limited historical data (leverages pre-training knowledge)
/// - Baseline comparisons for specialized models
/// </para>
/// <para>
/// <b>Reference:</b> Das et al., "A decoder-only foundation model for time-series forecasting", 2024.
/// https://arxiv.org/abs/2310.10688
/// </para>
/// </remarks>
public class TimesFM<T> : ForecastingModelBase<T>
{
    #region Execution Mode

    /// <summary>
    /// Indicates whether this network uses native layers (true) or ONNX model (false).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> ONNX mode loads pretrained weights for immediate use.
    /// Native mode allows fine-tuning or training from scratch (though pre-training is recommended).
    /// </para>
    /// </remarks>
    private bool _useNativeMode;

    #endregion


    #region Native Mode Fields

    /// <summary>
    /// Patch embedding layer that converts raw patches to hidden representations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Each patch (group of time steps) is projected to a
    /// high-dimensional vector that the transformer can process.
    /// </para>
    /// </remarks>
    private ILayer<T>? _patchEmbedding;

    /// <summary>
    /// Positional embedding layer for encoding patch positions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Transformers don't inherently know order, so we add
    /// position information. Without this, the model couldn't distinguish "first patch"
    /// from "last patch".
    /// </para>
    /// </remarks>
    private ILayer<T>? _positionEmbedding;

    /// <summary>
    /// Transformer decoder layers for processing the embedded sequence.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The core of TimesFM - multiple layers of:
    /// 1. Causal self-attention (each token looks at previous tokens)
    /// 2. Feed-forward network (processes each position)
    /// 3. Layer normalization and residual connections
    /// </para>
    /// </remarks>
    private readonly List<ILayer<T>> _transformerLayers = [];

    /// <summary>
    /// Layer normalization applied after all transformer layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Stabilizes the values before the output projection,
    /// ensuring consistent scale regardless of input magnitude.
    /// </para>
    /// </remarks>
    private ILayer<T>? _finalLayerNorm;

    /// <summary>
    /// Output projection layer that maps hidden states to forecasts.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Converts the transformer's internal representation
    /// to actual forecast values at each time step.
    /// </para>
    /// </remarks>
    private ILayer<T>? _outputProjection;

    #endregion

    #region Shared Fields

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly TimesFMOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private int _contextLength;
    private int _forecastHorizon;
    private int _patchLength;
    private int _hiddenDimension;
    private int _numLayers;
    private int _numHeads;
    private double _dropout;
    private bool _usePretrainedWeights;

    #endregion

    #region IForecastingModel Properties

    /// <inheritdoc/>
    public override int SequenceLength => _contextLength;

    /// <inheritdoc/>
    public override int PredictionHorizon => _forecastHorizon;

    /// <inheritdoc/>
    public override int NumFeatures => 1;

    /// <inheritdoc/>
    public override int PatchSize => _patchLength;

    /// <inheritdoc/>
    public override int Stride => _patchLength;

    /// <inheritdoc/>
    public override bool IsChannelIndependent => true;

    /// <inheritdoc/>
    public override bool UseNativeMode => _useNativeMode;

    /// <summary>
    /// Gets the number of patches derived from the context length and patch size.
    /// </summary>
    /// <value>The number of patches.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is ContextLength / PatchLength.
    /// With defaults (512/32 = 16), the transformer processes 16 tokens.
    /// </para>
    /// </remarks>
    public int NumPatches => _contextLength / _patchLength;

    /// <summary>
    /// Gets the number of attention heads in the transformer.
    /// </summary>
    /// <value>The number of attention heads.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Multi-head attention allows the model to focus on
    /// different aspects of the time series simultaneously.
    /// </para>
    /// </remarks>
    public int NumHeads => _numHeads;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a TimesFM model using pretrained ONNX model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer for fine-tuning.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the recommended way to use TimesFM:
    /// Load pretrained weights and use immediately for zero-shot forecasting.
    /// No training needed - just provide your time series and get predictions.
    /// </para>
    /// </remarks>
    public TimesFM(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        TimesFMOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        options ??= new TimesFMOptions<T>();
        _options = options;
        Options = _options;
        ValidateOptions(options);

        _useNativeMode = false;
        OnnxSession = new InferenceSession(onnxModelPath);
        OnnxModelPath = onnxModelPath;

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        _contextLength = options.ContextLength;
        _forecastHorizon = options.ForecastHorizon;
        _patchLength = options.PatchLength;
        _hiddenDimension = options.HiddenDimension;
        _numLayers = options.NumLayers;
        _numHeads = options.NumHeads;
        _dropout = options.DropoutRate;
        _usePretrainedWeights = options.UsePretrainedWeights;
    }

    /// <summary>
    /// Creates a TimesFM model in native mode for training or fine-tuning.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this to train TimesFM from scratch or fine-tune.
    /// Note that foundation models like TimesFM are designed to be used pretrained;
    /// training from scratch requires massive compute and diverse data.
    ///
    /// For most use cases, prefer the ONNX constructor with pretrained weights.
    /// This native mode is useful for:
    /// - Fine-tuning on domain-specific data
    /// - Research and experimentation
    /// - Custom architecture variations
    /// </para>
    /// </remarks>
    public TimesFM(
        NeuralNetworkArchitecture<T> architecture,
        TimesFMOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new TimesFMOptions<T>();
        _options = options;
        Options = _options;
        ValidateOptions(options);

        _useNativeMode = true;
        OnnxSession = null;
        OnnxModelPath = null;

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        _contextLength = options.ContextLength;
        _forecastHorizon = options.ForecastHorizon;
        _patchLength = options.PatchLength;
        _hiddenDimension = options.HiddenDimension;
        _numLayers = options.NumLayers;
        _numHeads = options.NumHeads;
        _dropout = options.DropoutRate;
        _usePretrainedWeights = options.UsePretrainedWeights;

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes the neural network layers for TimesFM.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Sets up the complete TimesFM architecture:
    /// 1. Patch embedding (converts time step groups to vectors)
    /// 2. Position embedding (adds location information)
    /// 3. Stack of transformer decoder layers (the brain of the model)
    /// 4. Final layer norm and output projection (produces forecasts)
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultTimesFMLayers(
                Architecture, _contextLength, _forecastHorizon, 1,
                _patchLength, _hiddenDimension, _numLayers, _numHeads, _dropout));

            ExtractLayerReferences();
        }
    }

    /// <summary>
    /// Extracts references to specific layers from the layer collection.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Organizes layers by their role in the architecture:
    /// - First layer: Patch embedding
    /// - Second layer: Position embedding
    /// - Middle layers: Transformer blocks (attention + MLP + normalization)
    /// - Final layers: Layer norm and output projection
    /// </para>
    /// </remarks>
    private void ExtractLayerReferences()
    {
        int idx = 0;

        // Patch embedding
        if (idx < Layers.Count)
            _patchEmbedding = Layers[idx++];

        // Position embedding (may be a dense layer for learned positions)
        if (idx < Layers.Count)
            _positionEmbedding = Layers[idx++];

        // Transformer layers (attention, feed-forward, norms, dropout)
        // Each transformer block has multiple sub-layers
        _transformerLayers.Clear();
        int transformerLayerCount = _numLayers * (_dropout > 0 ? 6 : 4);
        for (int i = 0; i < transformerLayerCount && idx < Layers.Count - 2; i++)
        {
            _transformerLayers.Add(Layers[idx++]);
        }

        // Final layer norm
        if (idx < Layers.Count)
            _finalLayerNorm = Layers[idx++];

        // Output projection
        if (idx < Layers.Count)
            _outputProjection = Layers[idx];
    }

    /// <summary>
    /// Validates that custom layers meet TimesFM architectural requirements.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Ensures you have all required components:
    /// patch embedding, position embedding, transformer layers, and output projection.
    /// </para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);
        if (layers.Count < 6)
        {
            throw new ArgumentException(
                "TimesFM requires at least 6 layers (patch embed, pos embed, " +
                "attention, feed-forward, layer norm, output projection).",
                nameof(layers));
        }
    }

    /// <summary>
    /// Validates the TimesFM options.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Checks that all configuration values are sensible:
    /// - Context length must be divisible by patch length
    /// - Hidden dimension must be divisible by number of heads
    /// - All sizes must be positive
    /// </para>
    /// </remarks>
    private static void ValidateOptions(TimesFMOptions<T> options)
    {
        var errors = new List<string>();

        if (options.ContextLength < 1)
            errors.Add("ContextLength must be at least 1.");
        if (options.ForecastHorizon < 1)
            errors.Add("ForecastHorizon must be at least 1.");
        if (options.PatchLength < 1)
            errors.Add("PatchLength must be at least 1.");
        if (options.ContextLength % options.PatchLength != 0)
            errors.Add("ContextLength must be divisible by PatchLength.");
        if (options.HiddenDimension < 1)
            errors.Add("HiddenDimension must be at least 1.");
        if (options.NumLayers < 1)
            errors.Add("NumLayers must be at least 1.");
        if (options.NumHeads < 1)
            errors.Add("NumHeads must be at least 1.");
        if (options.HiddenDimension % options.NumHeads != 0)
            errors.Add("HiddenDimension must be divisible by NumHeads.");
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
    /// <b>For Beginners:</b> In the TimesFM model, Predict produces predictions from input data. This is the main inference step of the TimesFM architecture.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return _useNativeMode ? ForecastNative(input) : ForecastOnnx(input);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Training TimesFM fine-tunes the pretrained weights
    /// for your specific domain. For foundation models, this is often called
    /// "transfer learning" - adapting broad knowledge to a specific task.
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
    /// <b>For Beginners:</b> In the TimesFM model, UpdateParameters updates internal parameters or state. This keeps the TimesFM architecture aligned with the latest values.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> gradients)
    {
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TimesFM model, GetModelMetadata performs a supporting step in the workflow. It keeps the TimesFM architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "TimesFM" },
                { "ContextLength", _contextLength },
                { "ForecastHorizon", _forecastHorizon },
                { "PatchLength", _patchLength },
                { "NumPatches", NumPatches },
                { "HiddenDimension", _hiddenDimension },
                { "NumLayers", _numLayers },
                { "NumHeads", _numHeads },
                { "UseNativeMode", _useNativeMode },
                { "UsePretrainedWeights", _usePretrainedWeights },
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
    /// <b>For Beginners:</b> Creates a fresh copy of the TimesFM architecture.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var options = new TimesFMOptions<T>
        {
            ContextLength = _contextLength,
            ForecastHorizon = _forecastHorizon,
            PatchLength = _patchLength,
            HiddenDimension = _hiddenDimension,
            NumLayers = _numLayers,
            NumHeads = _numHeads,
            DropoutRate = _dropout,
            UsePretrainedWeights = _usePretrainedWeights
        };

        return new TimesFM<T>(Architecture, options);
    }

    /// <summary>
    /// Writes TimesFM-specific configuration during serialization.
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
        writer.Write(_patchLength);
        writer.Write(_hiddenDimension);
        writer.Write(_numLayers);
        writer.Write(_numHeads);
        writer.Write(_dropout);
        writer.Write(_usePretrainedWeights);
    }

    /// <summary>
    /// Reads TimesFM-specific configuration during deserialization.
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
        _patchLength = reader.ReadInt32();
        _hiddenDimension = reader.ReadInt32();
        _numLayers = reader.ReadInt32();
        _numHeads = reader.ReadInt32();
        _dropout = reader.ReadDouble();
        _usePretrainedWeights = reader.ReadBoolean();
    }

    #endregion

    #region IForecastingModel Implementation

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TimesFM model, Forecast produces predictions from input data. This is the main inference step of the TimesFM architecture.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
    {
        return _useNativeMode ? ForecastNative(historicalData) : ForecastOnnx(historicalData);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> TimesFM is naturally autoregressive (like GPT).
    /// For longer forecasts, it generates predictions, shifts the context,
    /// and continues forecasting - much like how you'd read a book one page
    /// at a time, always remembering the previous context.
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
    /// <b>For Beginners:</b> In the TimesFM model, Evaluate performs a supporting step in the workflow. It keeps the TimesFM architecture pipeline consistent.
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
    /// <b>For Beginners:</b> In the TimesFM model, ApplyInstanceNormalization performs a supporting step in the workflow. It keeps the TimesFM architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
    {
        return input;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TimesFM model, GetFinancialMetrics calculates evaluation metrics. This summarizes how the TimesFM architecture is performing.
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
            ["NumPatches"] = NumOps.FromDouble(NumPatches),
            ["HiddenDimension"] = NumOps.FromDouble(_hiddenDimension),
            ["NumLayers"] = NumOps.FromDouble(_numLayers),
            ["NumHeads"] = NumOps.FromDouble(_numHeads),
            ["LastLoss"] = lastLoss
        };
    }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// Performs the forward pass through TimesFM.
    /// </summary>
    /// <param name="input">Input tensor of shape [batch, context_length].</param>
    /// <returns>Output tensor of shape [batch, forecast_horizon].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The TimesFM forward pass mimics GPT-style processing:
    ///
    /// <b>Step 1: Patching</b>
    /// - Input: 512 time steps
    /// - After patching: 16 patches of 32 steps each
    /// - Each patch becomes one "token" for the transformer
    ///
    /// <b>Step 2: Embedding</b>
    /// - Patch embedding: Each patch → hidden dimension vector
    /// - Position embedding: Add position information (which patch is which)
    ///
    /// <b>Step 3: Transformer Processing</b>
    /// - Multiple layers of self-attention
    /// - Each token attends only to earlier tokens (causal)
    /// - Captures long-range dependencies and patterns
    ///
    /// <b>Step 4: Output Projection</b>
    /// - Final layer norm for stability
    /// - Project to forecast horizon length
    /// </para>
    /// </remarks>
    private Tensor<T> Forward(Tensor<T> input)
    {
        if (!_useNativeMode)
            return ForecastOnnx(input);

        // Reshape input into patches conceptually (handled by patch embedding layer)
        var current = input;

        // Patch embedding: [batch, context] -> [batch, num_patches, hidden]
        if (_patchEmbedding is not null)
            current = _patchEmbedding.Forward(current);

        // Add positional embeddings
        if (_positionEmbedding is not null)
        {
            var posEmbed = _positionEmbedding.Forward(current);
            current = AddTensors(current, posEmbed);
        }

        // Process through transformer layers
        foreach (var layer in _transformerLayers)
        {
            current = layer.Forward(current);
        }

        // Final layer normalization
        if (_finalLayerNorm is not null)
            current = _finalLayerNorm.Forward(current);

        // Output projection to forecast horizon
        if (_outputProjection is not null)
            current = _outputProjection.Forward(current);

        return current;
    }

    /// <summary>
    /// Element-wise addition of two tensors.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Adds patch embeddings and position embeddings together.
    /// This is how transformers encode "what" (patch content) + "where" (position).
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

        // Copy remaining elements from a if lengths differ
        for (int i = length; i < a.Length; i++)
        {
            result.Data.Span[i] = a.Data.Span[i];
        }

        return result;
    }

    /// <summary>
    /// Performs the backward pass through TimesFM.
    /// </summary>
    /// <param name="gradOutput">Gradient from the loss function.</param>
    /// <returns>Gradient with respect to the input.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Backpropagation through transformers:
    /// 1. Gradient flows back through output projection
    /// 2. Then back through each transformer layer (in reverse order)
    /// 3. Finally through embeddings to the input
    ///
    /// This updates all weights to reduce prediction error.
    /// </para>
    /// </remarks>
    private Tensor<T> Backward(Tensor<T> gradOutput)
    {
        var current = gradOutput;

        // Output projection backward
        if (_outputProjection is not null)
            current = _outputProjection.Backward(current);

        // Final layer norm backward
        if (_finalLayerNorm is not null)
            current = _finalLayerNorm.Backward(current);

        // Transformer layers backward (reverse order)
        for (int i = _transformerLayers.Count - 1; i >= 0; i--)
        {
            current = _transformerLayers[i].Backward(current);
        }

        // Position embedding backward (simplified - skip for now as positions are fixed)
        // Note: In full implementation, learned position embeddings would be updated here

        // Patch embedding backward
        if (_patchEmbedding is not null)
            current = _patchEmbedding.Backward(current);

        return current;
    }

    #endregion

    #region Inference Methods

    /// <summary>
    /// Performs native mode forecasting.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Uses the neural network layers to produce forecasts.
    /// In native mode, all computation is done in C# with our tensor operations.
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
    /// ONNX (Open Neural Network Exchange) is an industry standard format
    /// that allows models trained in Python/PyTorch to run in C#.
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
    /// <b>For Beginners:</b> For multi-step forecasting beyond the horizon:
    /// 1. Take the model's predictions
    /// 2. Slide the context window forward
    /// 3. Add predictions as new "history"
    /// 4. Forecast again from the new position
    ///
    /// This is how GPT generates text - one token at a time, building on previous outputs.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ShiftInputWithPredictions(Tensor<T> input, Tensor<T> predictions, int stepsUsed)
    {
        int batchSize = input.Shape.Length > 0 ? input.Shape[0] : 1;
        if (batchSize > 1)
            throw new InvalidOperationException("TimesFM autoregressive helpers currently support batchSize = 1.");

        var newInput = new Tensor<T>(input.Shape);
        int steps = Math.Min(stepsUsed, _contextLength);

        // Shift existing data left by stepsUsed
        for (int i = 0; i < _contextLength - steps; i++)
        {
            if (i + steps < input.Length)
                newInput.Data.Span[i] = input.Data.Span[i + steps];
        }

        // Fill end with predictions
        for (int i = 0; i < steps && i < predictions.Length; i++)
        {
            int targetIdx = _contextLength - steps + i;
            if (targetIdx < _contextLength)
            {
                newInput.Data.Span[targetIdx] = predictions[i];
            }
        }

        return newInput;
    }

    /// <summary>
    /// Concatenates multiple prediction tensors for extended horizons.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When forecasting beyond the model's native horizon,
    /// we combine multiple prediction batches into one continuous result.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ConcatenatePredictions(List<Tensor<T>> predictions, int totalSteps)
    {
        if (predictions.Count > 0)
        {
            int batchSize = predictions[0].Shape.Length > 0 ? predictions[0].Shape[0] : 1;
            if (batchSize > 1)
                throw new InvalidOperationException("TimesFM autoregressive helpers currently support batchSize = 1.");
        }

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
    /// Disposes resources used by the TimesFM model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Releases the ONNX session and other resources when the model
    /// is no longer needed. Always dispose models when done to free memory.
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

