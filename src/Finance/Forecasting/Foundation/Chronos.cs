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
/// Chronos foundation model for time series forecasting using tokenization.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// Chronos is Amazon's foundation model that treats time series forecasting as a language
/// modeling problem. It tokenizes continuous time series values into discrete tokens,
/// uses a pretrained language model (T5-style) to predict future tokens, and then
/// converts tokens back to continuous values.
/// </para>
/// <para>
/// <b>For Beginners:</b> Chronos brings the power of language models to time series:
///
/// <b>The Key Insight:</b>
/// Language models like GPT are amazing at predicting the next word in a sequence.
/// Chronos asks: "What if we convert time series to 'words' and use the same approach?"
///
/// <b>How Tokenization Works:</b>
/// 1. <b>Scaling</b>: Normalize values (e.g., mean=0, std=1 or min-max to [-1, 1])
/// 2. <b>Quantization</b>: Divide range into bins (e.g., 4096 bins)
/// 3. <b>Token Assignment</b>: Each value gets the bin number as its "token"
///
/// Example: If value = 0.73 and bins are [-1 to 1] with 100 bins:
/// - 0.73 falls in bin 86 (because 0.73 is 86% of the way from -1 to 1)
/// - Token = 86
///
/// <b>Why This Works:</b>
/// - LLMs excel at pattern recognition in sequences
/// - Time series have patterns just like language
/// - Pretrained LLMs already understand "sequences" - just need different tokens
/// - Can leverage massive pretraining investments
///
/// <b>Sampling for Uncertainty:</b>
/// Like GPT sampling text, Chronos samples from predicted token probabilities:
/// - Take 20 samples → 20 different forecasts
/// - Median = point forecast
/// - Spread = uncertainty estimate
///
/// <b>Model Sizes:</b>
/// - Mini (20M params): Fast, good for experiments
/// - Small (46M params): Balanced
/// - Base (200M params): Strong general use
/// - Large (710M params): Best accuracy
/// </para>
/// <para>
/// <b>Reference:</b> Ansari et al., "Chronos: Learning the Language of Time Series", 2024.
/// https://arxiv.org/abs/2403.07815
/// </para>
/// </remarks>
public class Chronos<T> : ForecastingModelBase<T>
{
    #region Execution Mode

    /// <summary>
    /// Indicates whether this network uses native layers (true) or ONNX model (false).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> ONNX mode uses pretrained Amazon Chronos weights.
    /// Native mode allows training or fine-tuning (though pretraining is recommended).
    /// </para>
    /// </remarks>
    private readonly bool _useNativeMode;

    #endregion

    
    #region Native Mode Fields

    /// <summary>
    /// Token embedding layer that maps token IDs to vectors.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Like word embeddings in NLP, this converts discrete
    /// token IDs into continuous vectors the transformer can process.
    /// </para>
    /// </remarks>
    private ILayer<T>? _tokenEmbedding;

    /// <summary>
    /// Transformer layers for processing the embedded sequence.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The core language model - multiple layers of
    /// attention and feed-forward processing that learn time series patterns.
    /// </para>
    /// </remarks>
    private readonly List<ILayer<T>> _transformerLayers = [];

    /// <summary>
    /// Final layer normalization before the language model head.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Stabilizes values before predicting token probabilities.
    /// </para>
    /// </remarks>
    private ILayer<T>? _finalNorm;

    /// <summary>
    /// Language model head that predicts token probabilities.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Like the output layer in GPT that predicts the next word,
    /// this predicts which token (bin) each future time step belongs to.
    /// </para>
    /// </remarks>
    private ILayer<T>? _lmHead;

    #endregion

    #region Shared Fields

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private int _contextLength;
    private int _forecastHorizon;
    private int _numTokens;
    private int _hiddenDimension;
    private int _numLayers;
    private int _numHeads;
    private int _intermediateSize;
    private int _numSamples;
    private double _dropout;
    private double _temperature;
    private string _modelSize;
    private T _lastTokenMin = default!;
    private T _lastTokenRange = default!;
    private bool _hasTokenScale;

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
    /// Gets the number of discrete tokens used for quantization.
    /// </summary>
    /// <value>The number of tokens (bins).</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> How many distinct "words" the model vocabulary has.
    /// More tokens = finer value resolution.
    /// </para>
    /// </remarks>
    public int NumTokens => _numTokens;

    /// <summary>
    /// Gets the model size variant.
    /// </summary>
    /// <value>The model size (mini, small, base, large).</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Larger models have more capacity but require more compute.
    /// </para>
    /// </remarks>
    public string ModelSize => _modelSize;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a Chronos model using pretrained ONNX model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer for fine-tuning.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The recommended way to use Chronos - load pretrained
    /// weights for immediate zero-shot forecasting. Just provide your time series
    /// and get probabilistic predictions.
    /// </para>
    /// </remarks>
    public Chronos(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        ChronosFinanceOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        options ??= new ChronosFinanceOptions<T>();
        ValidateOptions(options);

        _useNativeMode = false;
        OnnxSession = new InferenceSession(onnxModelPath);
        OnnxModelPath = onnxModelPath;

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        _contextLength = options.ContextLength;
        _forecastHorizon = options.ForecastHorizon;
        _numTokens = options.NumTokens;
        _hiddenDimension = options.HiddenDimension;
        _numLayers = options.NumLayers;
        _numHeads = options.NumHeads;
        _intermediateSize = options.IntermediateSize;
        _numSamples = options.NumSamples;
        _dropout = options.DropoutRate;
        _temperature = options.Temperature;
        _modelSize = options.ModelSize;
    }

    /// <summary>
    /// Creates a Chronos model in native mode for training or fine-tuning.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this for fine-tuning on domain-specific data or
    /// experimentation. Note that Chronos was designed for pretrained use; training
    /// from scratch requires significant compute.
    /// </para>
    /// </remarks>
    public Chronos(
        NeuralNetworkArchitecture<T> architecture,
        ChronosFinanceOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new ChronosFinanceOptions<T>();
        ValidateOptions(options);

        _useNativeMode = true;
        OnnxSession = null;
        OnnxModelPath = null;

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        _contextLength = options.ContextLength;
        _forecastHorizon = options.ForecastHorizon;
        _numTokens = options.NumTokens;
        _hiddenDimension = options.HiddenDimension;
        _numLayers = options.NumLayers;
        _numHeads = options.NumHeads;
        _intermediateSize = options.IntermediateSize;
        _numSamples = options.NumSamples;
        _dropout = options.DropoutRate;
        _temperature = options.Temperature;
        _modelSize = options.ModelSize;

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes the neural network layers for Chronos.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Sets up the complete Chronos architecture:
    /// 1. Token embedding layer
    /// 2. Stack of transformer layers
    /// 3. Final normalization
    /// 4. Language model head for token prediction
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultChronosLayers(
                Architecture, _contextLength, _forecastHorizon, _numTokens,
                _hiddenDimension, _numLayers, _numHeads, _intermediateSize, _dropout));

            ExtractLayerReferences();
        }
    }

    /// <summary>
    /// Extracts references to specific layers from the layer collection.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Organizes layers by their role:
    /// - First: Token embedding
    /// - Middle: Transformer blocks
    /// - Last two: Final norm and LM head
    /// </para>
    /// </remarks>
    private void ExtractLayerReferences()
    {
        int idx = 0;

        // Token embedding
        if (idx < Layers.Count)
            _tokenEmbedding = Layers[idx++];

        // Transformer layers
        _transformerLayers.Clear();
        int layersPerBlock = _dropout > 0 ? 9 : 7;
        int totalTransformerLayers = _numLayers * layersPerBlock;

        for (int i = 0; i < totalTransformerLayers && idx < Layers.Count - 2; i++)
        {
            _transformerLayers.Add(Layers[idx++]);
        }

        // Final normalization
        if (idx < Layers.Count)
            _finalNorm = Layers[idx++];

        // Language model head
        if (idx < Layers.Count)
            _lmHead = Layers[idx];
    }

    /// <summary>
    /// Validates that custom layers meet Chronos architectural requirements.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Ensures you have all required components.
    /// </para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);
        if (layers.Count < 5)
        {
            throw new ArgumentException(
                "Chronos requires at least 5 layers (token embed, attention, FFN, norm, LM head).",
                nameof(layers));
        }
    }

    /// <summary>
    /// Validates the Chronos options.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Checks that all configuration values are sensible.
    /// </para>
    /// </remarks>
    private static void ValidateOptions(ChronosFinanceOptions<T> options)
    {
        var errors = new List<string>();

        if (options.ContextLength < 1)
            errors.Add("ContextLength must be at least 1.");
        if (options.ForecastHorizon < 1)
            errors.Add("ForecastHorizon must be at least 1.");
        if (options.NumTokens < 1)
            errors.Add("NumTokens must be at least 1.");
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
        if (options.NumSamples < 1)
            errors.Add("NumSamples must be at least 1.");
        if (options.DropoutRate < 0 || options.DropoutRate >= 1)
            errors.Add("DropoutRate must be between 0 and 1 (exclusive).");
        if (options.Temperature <= 0)
            errors.Add("Temperature must be positive.");

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
    /// <b>For Beginners:</b> In the Chronos model, Predict produces predictions from input data. This is the main inference step of the Chronos architecture.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return _useNativeMode ? ForecastNative(input) : ForecastOnnx(input);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Training Chronos uses cross-entropy loss on token predictions.
    /// The model learns to predict which token (bin) each future value belongs to.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is only supported in native mode.");

        SetTrainingMode(true);
        try
        {
            // Tokenize input and run through the model
            var tokenizedInput = Tokenize(input);
            var logits = Forward(tokenizedInput);

            // If target is already tokenized/logit-shaped (as in tests), use it directly.
            // Otherwise, tokenize the target to match logits for training.
            var targetTokens = target.Length == logits.Length ? target : Tokenize(target);

            LastLoss = _lossFunction.CalculateLoss(logits.ToVector(), targetTokens.ToVector());

            var gradient = _lossFunction.CalculateDerivative(logits.ToVector(), targetTokens.ToVector());
            Backward(Tensor<T>.FromVector(gradient, logits.Shape));

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
    /// <b>For Beginners:</b> In the Chronos model, UpdateParameters updates internal parameters or state. This keeps the Chronos architecture aligned with the latest values.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> gradients)
    {
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Chronos model, GetModelMetadata performs a supporting step in the workflow. It keeps the Chronos architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "Chronos" },
                { "ContextLength", _contextLength },
                { "ForecastHorizon", _forecastHorizon },
                { "NumTokens", _numTokens },
                { "HiddenDimension", _hiddenDimension },
                { "NumLayers", _numLayers },
                { "NumHeads", _numHeads },
                { "IntermediateSize", _intermediateSize },
                { "NumSamples", _numSamples },
                { "Temperature", _temperature },
                { "ModelSize", _modelSize },
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
    /// <b>For Beginners:</b> Creates a fresh copy of the Chronos architecture.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var options = new ChronosFinanceOptions<T>
        {
            ContextLength = _contextLength,
            ForecastHorizon = _forecastHorizon,
            NumTokens = _numTokens,
            HiddenDimension = _hiddenDimension,
            NumLayers = _numLayers,
            NumHeads = _numHeads,
            IntermediateSize = _intermediateSize,
            NumSamples = _numSamples,
            DropoutRate = _dropout,
            Temperature = _temperature,
            ModelSize = _modelSize
        };

        return new Chronos<T>(Architecture, options);
    }

    /// <summary>
    /// Writes Chronos-specific configuration during serialization.
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
        writer.Write(_numTokens);
        writer.Write(_hiddenDimension);
        writer.Write(_numLayers);
        writer.Write(_numHeads);
        writer.Write(_intermediateSize);
        writer.Write(_numSamples);
        writer.Write(_dropout);
        writer.Write(_temperature);
        writer.Write(_modelSize);

        // Serialize tokenization scaling state
        writer.Write(_hasTokenScale);
        writer.Write(NumOps.ToDouble(_lastTokenMin));
        writer.Write(NumOps.ToDouble(_lastTokenRange));
    }

    /// <summary>
    /// Reads Chronos-specific configuration during deserialization.
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
        _numTokens = reader.ReadInt32();
        _hiddenDimension = reader.ReadInt32();
        _numLayers = reader.ReadInt32();
        _numHeads = reader.ReadInt32();
        _intermediateSize = reader.ReadInt32();
        _numSamples = reader.ReadInt32();
        _dropout = reader.ReadDouble();
        _temperature = reader.ReadDouble();
        _modelSize = reader.ReadString();

        // Deserialize tokenization scaling state
        _hasTokenScale = reader.ReadBoolean();
        _lastTokenMin = NumOps.FromDouble(reader.ReadDouble());
        _lastTokenRange = NumOps.FromDouble(reader.ReadDouble());
    }

    #endregion

    #region IForecastingModel Implementation

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Chronos model, Forecast produces predictions from input data. This is the main inference step of the Chronos architecture.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
    {
        var output = _useNativeMode ? ForecastNative(historicalData) : ForecastOnnx(historicalData);

        // Detokenize to get continuous values
        var pointPredictions = Detokenize(output);

        // If quantiles requested, generate multiple samples
        if (quantiles is not null && quantiles.Length > 0)
        {
            return GenerateQuantileSamples(historicalData, quantiles);
        }

        return pointPredictions;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> For extended forecasts, Chronos generates tokens
    /// autoregressively, just like GPT generates text word by word.
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
    /// <b>For Beginners:</b> In the Chronos model, Evaluate performs a supporting step in the workflow. It keeps the Chronos architecture pipeline consistent.
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
    /// <b>For Beginners:</b> In the Chronos model, ApplyInstanceNormalization performs a supporting step in the workflow. It keeps the Chronos architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
    {
        return input;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Chronos model, GetFinancialMetrics calculates evaluation metrics. This summarizes how the Chronos architecture is performing.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> GetFinancialMetrics()
    {
        T lastLoss = LastLoss is not null ? LastLoss : NumOps.Zero;

        return new Dictionary<string, T>
        {
            ["ContextLength"] = NumOps.FromDouble(_contextLength),
            ["ForecastHorizon"] = NumOps.FromDouble(_forecastHorizon),
            ["NumTokens"] = NumOps.FromDouble(_numTokens),
            ["HiddenDimension"] = NumOps.FromDouble(_hiddenDimension),
            ["NumLayers"] = NumOps.FromDouble(_numLayers),
            ["NumSamples"] = NumOps.FromDouble(_numSamples),
            ["Temperature"] = NumOps.FromDouble(_temperature),
            ["LastLoss"] = lastLoss
        };
    }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// Performs the forward pass through Chronos.
    /// </summary>
    /// <param name="input">Input tensor (tokenized or raw).</param>
    /// <returns>Output tensor with token logits.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Chronos forward pass:
    ///
    /// <b>Step 1: Token Embedding</b>
    /// - Input tokens (from quantization) embedded to hidden dimension
    ///
    /// <b>Step 2: Transformer Processing</b>
    /// - Multiple layers of attention and feed-forward
    /// - Learns patterns in the "language" of time series
    ///
    /// <b>Step 3: Language Model Head</b>
    /// - Predicts probability distribution over all tokens
    /// - Each position predicts the next token
    /// </para>
    /// </remarks>
    private Tensor<T> Forward(Tensor<T> input)
    {
        if (!_useNativeMode)
            return ForecastOnnx(input);

        var current = input;

        // Add batch dimension if input is 1D (layers expect at least 2D [batch, features])
        bool addedBatchDim = false;
        if (current.Rank == 1)
        {
            current = current.Reshape(new[] { 1, current.Length });
            addedBatchDim = true;
        }

        // Token embedding
        if (_tokenEmbedding is not null)
            current = _tokenEmbedding.Forward(current);

        // Process through transformer layers
        foreach (var layer in _transformerLayers)
        {
            current = layer.Forward(current);
        }

        // Final normalization
        if (_finalNorm is not null)
            current = _finalNorm.Forward(current);

        // Language model head
        if (_lmHead is not null)
            current = _lmHead.Forward(current);

        // Remove batch dimension if we added it
        if (addedBatchDim && current.Rank == 2 && current.Shape[0] == 1)
        {
            current = current.Reshape(new[] { current.Shape[1] });
        }

        return current;
    }

    /// <summary>
    /// Performs the backward pass through Chronos.
    /// </summary>
    /// <param name="gradOutput">Gradient from the loss function.</param>
    /// <returns>Gradient with respect to the input.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Backpropagation updates all model weights
    /// to better predict the correct tokens.
    /// </para>
    /// </remarks>
    private Tensor<T> Backward(Tensor<T> gradOutput)
    {
        var current = gradOutput;

        // Add batch dimension if gradient is 1D (layers expect at least 2D [batch, features])
        bool addedBatchDim = false;
        if (current.Rank == 1)
        {
            current = current.Reshape(new[] { 1, current.Length });
            addedBatchDim = true;
        }

        // LM head backward
        if (_lmHead is not null)
            current = _lmHead.Backward(current);

        // Final norm backward
        if (_finalNorm is not null)
            current = _finalNorm.Backward(current);

        // Transformer layers backward (reverse order)
        for (int i = _transformerLayers.Count - 1; i >= 0; i--)
        {
            current = _transformerLayers[i].Backward(current);
        }

        // Token embedding backward
        if (_tokenEmbedding is not null)
            current = _tokenEmbedding.Backward(current);

        // Remove batch dimension if we added it
        if (addedBatchDim && current.Rank == 2 && current.Shape[0] == 1)
        {
            current = current.Reshape(new[] { current.Shape[1] });
        }

        return current;
    }

    #endregion

    #region Inference Methods

    /// <summary>
    /// Performs native mode forecasting.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Tokenizes input, runs through the model, returns logits.
    /// </para>
    /// </remarks>
    private Tensor<T> ForecastNative(Tensor<T> input)
    {
        SetTrainingMode(false);
        var tokenized = Tokenize(input);
        return Forward(tokenized);
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

    #region Tokenization

    /// <summary>
    /// Tokenizes continuous time series values into discrete tokens.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Converts continuous values to discrete "words":
    /// 1. Scale to [-1, 1] range
    /// 2. Map to bin indices (0 to NumTokens-1)
    /// 3. Create one-hot representation for embedding
    /// </para>
    /// </remarks>
    private Tensor<T> Tokenize(Tensor<T> values)
    {
        // Find min and max for scaling
        T min = NumOps.MaxValue;
        T max = NumOps.MinValue;
        for (int i = 0; i < values.Length; i++)
        {
            if (NumOps.LessThan(values.Data.Span[i], min)) min = values.Data.Span[i];
            if (NumOps.GreaterThan(values.Data.Span[i], max)) max = values.Data.Span[i];
        }

        var range = NumOps.Subtract(max, min);
        if (NumOps.Equals(range, NumOps.Zero))
            range = NumOps.One;

        _lastTokenMin = min;
        _lastTokenRange = range;
        _hasTokenScale = true;

        // Create one-hot encoded tokens
        var tokenized = new Tensor<T>(new[] { values.Length * _numTokens });

        for (int i = 0; i < values.Length; i++)
        {
            // Scale to [0, 1]
            var scaled = NumOps.Divide(NumOps.Subtract(values.Data.Span[i], min), range);
            // Map to bin index
            double scaledDouble = NumOps.ToDouble(scaled);
            int binIndex = Math.Min((int)(scaledDouble * (_numTokens - 1)), _numTokens - 1);
            binIndex = Math.Max(0, binIndex);

            // Set one-hot
            tokenized.Data.Span[i * _numTokens + binIndex] = NumOps.One;
        }

        return tokenized;
    }

    /// <summary>
    /// Detokenizes model output (logits) back to continuous values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Converts token probabilities back to values:
    /// 1. For each position, find the argmax token (or sample)
    /// 2. Convert bin index to scaled value
    /// 3. Apply inverse scaling
    /// </para>
    /// </remarks>
    private Tensor<T> Detokenize(Tensor<T> logits)
    {
        // Output has shape [forecastHorizon * numTokens]
        // For each step, find argmax token and convert to value
        var result = new Tensor<T>(new[] { _forecastHorizon });

        for (int step = 0; step < _forecastHorizon; step++)
        {
            // Find argmax for this step
            int maxIdx = 0;
            T maxVal = NumOps.MinValue;

            for (int token = 0; token < _numTokens; token++)
            {
                int idx = step * _numTokens + token;
                if (idx < logits.Length && NumOps.GreaterThan(logits.Data.Span[idx], maxVal))
                {
                    maxVal = logits.Data.Span[idx];
                    maxIdx = token;
                }
            }

            // Guard against division by zero when _numTokens == 1
            double scaledValue = _numTokens > 1 ? maxIdx / (double)(_numTokens - 1) : 0.5;
            if (_hasTokenScale)
            {
                double minVal = NumOps.ToDouble(_lastTokenMin);
                double rangeVal = NumOps.ToDouble(_lastTokenRange);
                result.Data.Span[step] = NumOps.FromDouble((scaledValue * rangeVal) + minVal);
            }
            else
            {
                // Fallback to normalized range if scale is unknown
                double normalizedValue = (scaledValue * 2) - 1;
                result.Data.Span[step] = NumOps.FromDouble(normalizedValue);
            }
        }

        return result;
    }

    /// <summary>
    /// Generates quantile samples by sampling from token distributions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Like GPT generating different text samples:
    /// 1. Run model multiple times with different random samples
    /// 2. Collect all predictions
    /// 3. Compute quantiles from the samples
    /// </para>
    /// </remarks>
    private Tensor<T> GenerateQuantileSamples(Tensor<T> input, double[] quantiles)
    {
        var allSamples = new List<double[]>();
        var rand = RandomHelper.CreateSecureRandom();

        // Generate multiple samples
        for (int s = 0; s < _numSamples; s++)
        {
            var logits = _useNativeMode ? ForecastNative(input) : ForecastOnnx(input);
            var sample = new double[_forecastHorizon];

            for (int step = 0; step < _forecastHorizon; step++)
            {
                // Sample from softmax distribution with temperature
                int tokenIdx = SampleFromLogits(logits, step, rand);
                // Guard against division by zero when _numTokens == 1
                double scaledValue = _numTokens > 1 ? tokenIdx / (double)(_numTokens - 1) : 0.5;

                if (_hasTokenScale)
                {
                    double minVal = NumOps.ToDouble(_lastTokenMin);
                    double rangeVal = NumOps.ToDouble(_lastTokenRange);
                    sample[step] = (scaledValue * rangeVal) + minVal;
                }
                else
                {
                    sample[step] = (scaledValue * 2) - 1;
                }
            }

            allSamples.Add(sample);
        }

        // Compute quantiles
        var result = new Tensor<T>(new[] { _forecastHorizon * quantiles.Length });

        for (int q = 0; q < quantiles.Length; q++)
        {
            for (int step = 0; step < _forecastHorizon; step++)
            {
                var stepValues = allSamples.Select(s => s[step]).OrderBy(v => v).ToList();
                int quantileIdx = Math.Min((int)(quantiles[q] * stepValues.Count), stepValues.Count - 1);
                result.Data.Span[q * _forecastHorizon + step] = NumOps.FromDouble(stepValues[quantileIdx]);
            }
        }

        return result;
    }

    /// <summary>
    /// Samples a token index from logits using temperature-scaled softmax.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Like GPT's sampling:
    /// 1. Apply temperature (higher = more random)
    /// 2. Convert logits to probabilities (softmax)
    /// 3. Sample according to probabilities
    /// </para>
    /// </remarks>
    private int SampleFromLogits(Tensor<T> logits, int step, System.Random rand)
    {
        // Extract logits for this step
        var stepLogits = new double[_numTokens];
        double maxLogit = double.NegativeInfinity;

        for (int token = 0; token < _numTokens; token++)
        {
            int idx = step * _numTokens + token;
            if (idx < logits.Length)
            {
                stepLogits[token] = NumOps.ToDouble(logits.Data.Span[idx]) / _temperature;
                maxLogit = Math.Max(maxLogit, stepLogits[token]);
            }
        }

        // Softmax
        double sumExp = 0;
        for (int i = 0; i < _numTokens; i++)
        {
            stepLogits[i] = Math.Exp(stepLogits[i] - maxLogit);
            sumExp += stepLogits[i];
        }

        // Sample
        double r = rand.NextDouble() * sumExp;
        double cumSum = 0;
        for (int i = 0; i < _numTokens; i++)
        {
            cumSum += stepLogits[i];
            if (cumSum >= r) return i;
        }

        return _numTokens - 1;
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Shifts input tensor by incorporating predictions for autoregressive forecasting.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Slides the context window and adds predictions as new history.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ShiftInputWithPredictions(Tensor<T> input, Tensor<T> predictions, int stepsUsed)
    {
        var newInput = new Tensor<T>(input.Shape);

        // Clamp stepsUsed to prevent negative indices
        int clampedSteps = Math.Min(stepsUsed, _contextLength);

        // Shift existing data
        for (int i = 0; i < _contextLength - clampedSteps; i++)
        {
            if (i + clampedSteps < input.Length)
                newInput.Data.Span[i] = input.Data.Span[i + clampedSteps];
        }

        // Add predictions at the end
        for (int i = 0; i < clampedSteps && i < predictions.Length; i++)
        {
            int targetIdx = _contextLength - clampedSteps + i;
            if (targetIdx >= 0 && targetIdx < _contextLength)
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
    /// <b>For Beginners:</b> Combines predictions from multiple iterations.
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
    /// Disposes resources used by the Chronos model.
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

