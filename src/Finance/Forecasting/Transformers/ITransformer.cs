using System.IO;
using AiDotNet.Finance.Interfaces;
using AiDotNet.Finance.Options;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.Finance.Forecasting.Transformers;

/// <summary>
/// iTransformer (Inverted Transformer) neural network for time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// iTransformer inverts the traditional transformer approach by treating each variable (channel)
/// as a token instead of each time step. This allows the model to learn cross-variable dependencies
/// more effectively through the attention mechanism.
/// </para>
/// <para>
/// <b>For Beginners:</b> Traditional transformers for time series treat each time step as a "word".
/// iTransformer flips this - it treats each variable (like price, volume) as a "word". This way,
/// the attention mechanism learns how different variables relate to each other, which is often
/// more useful for forecasting than just looking at temporal patterns.
/// </para>
/// <para>
/// <b>Reference:</b> Liu et al., "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting",
/// ICLR 2024. https://arxiv.org/abs/2310.06625
/// </para>
/// </remarks>
public class ITransformer<T> : NeuralNetworkBase<T>, IForecastingModel<T>
{
    #region Execution Mode

    /// <summary>
    /// Indicates whether this network uses native layers (true) or ONNX model (false).
    /// </summary>
    private readonly bool _useNativeMode;

    #endregion

    #region ONNX Mode Fields

    /// <summary>
    /// The ONNX inference session for the model.
    /// </summary>
    private readonly InferenceSession? _onnxSession;

    /// <summary>
    /// Path to the ONNX model file.
    /// </summary>
    private readonly string? _onnxModelPath;

    #endregion

    #region Native Mode Fields

    /// <summary>
    /// Variate embedding layer that embeds each variable's time series.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This layer takes the entire time series of each variable and
    /// compresses it into a dense vector (embedding). Think of it like summarizing each
    /// variable's history into a fixed-size representation.
    /// </para>
    /// </remarks>
    private ILayer<T>? _variateEmbedding;

    /// <summary>
    /// Transformer encoder layers for cross-variable attention.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> These layers use attention to learn how different variables
    /// relate to each other. Each variable "attends" to all other variables to understand
    /// their relationships (e.g., how volume affects price).
    /// </para>
    /// </remarks>
    private readonly List<ILayer<T>> _encoderLayers = [];

    /// <summary>
    /// Final layer normalization.
    /// </summary>
    private ILayer<T>? _finalNorm;

    /// <summary>
    /// Output projection layer.
    /// </summary>
    private ILayer<T>? _outputProjection;

    /// <summary>
    /// Instance normalization mean (for RevIN).
    /// </summary>
    private Tensor<T>? _instanceMean;

    /// <summary>
    /// Instance normalization standard deviation (for RevIN).
    /// </summary>
    private Tensor<T>? _instanceStd;

    #endregion

    #region Shared Fields

    /// <summary>
    /// The optimizer for training.
    /// </summary>
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <summary>
    /// The loss function for training.
    /// </summary>
    private readonly ILossFunction<T> _lossFunction;

    /// <summary>
    /// The input sequence length.
    /// </summary>
    private readonly int _sequenceLength;

    /// <summary>
    /// The prediction horizon.
    /// </summary>
    private readonly int _predictionHorizon;

    /// <summary>
    /// The number of input features.
    /// </summary>
    private readonly int _numFeatures;

    /// <summary>
    /// The number of transformer layers.
    /// </summary>
    private readonly int _numLayers;

    /// <summary>
    /// The number of attention heads.
    /// </summary>
    private readonly int _numHeads;

    /// <summary>
    /// The model dimension.
    /// </summary>
    private readonly int _modelDimension;

    /// <summary>
    /// The feedforward dimension.
    /// </summary>
    private readonly int _feedForwardDimension;

    /// <summary>
    /// Whether to use instance normalization (RevIN).
    /// </summary>
    private readonly bool _useInstanceNormalization;

    /// <summary>
    /// Dropout rate.
    /// </summary>
    private readonly double _dropout;

    #endregion

    #region IForecastingModel Properties

    /// <inheritdoc/>
    public int SequenceLength => _sequenceLength;

    /// <inheritdoc/>
    public int PredictionHorizon => _predictionHorizon;

    /// <inheritdoc/>
    public int NumFeatures => _numFeatures;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> iTransformer doesn't use patches - it embeds entire variable
    /// time series as tokens. This property returns 0 to indicate patching is not used.
    /// </para>
    /// </remarks>
    public int PatchSize => 0;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> iTransformer doesn't use patches, so stride is not applicable.
    /// This property returns 0.
    /// </para>
    /// </remarks>
    public int Stride => 0;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In iTransformer, "channel independent" means something different.
    /// The model processes all channels together through cross-variable attention, but each
    /// variable produces its own independent forecast. This returns false because channels
    /// interact through attention.
    /// </para>
    /// </remarks>
    public bool IsChannelIndependent => false;

    /// <inheritdoc/>
    public bool UseNativeMode => _useNativeMode;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates an iTransformer network using a pretrained ONNX model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="sequenceLength">Input sequence length (default: 96).</param>
    /// <param name="predictionHorizon">Prediction horizon (default: 96).</param>
    /// <param name="numFeatures">Number of input features (default: 7).</param>
    /// <param name="optimizer">Optional optimizer for fine-tuning.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor when you have a pretrained ONNX model.
    /// ONNX allows you to use models trained in other frameworks (like PyTorch) directly
    /// in C# without retraining.
    ///
    /// Example:
    /// <code>
    /// var arch = new NeuralNetworkArchitecture&lt;double&gt;(
    ///     inputType: InputType.ThreeDimensional,
    ///     inputHeight: 96,   // sequence_length
    ///     inputWidth: 7);    // num_features
    ///
    /// var model = new ITransformer&lt;double&gt;(arch, "itransformer_etth1.onnx");
    /// var forecast = model.Predict(historicalData);
    /// </code>
    /// </para>
    /// </remarks>
    public ITransformer(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int sequenceLength = 96,
        int predictionHorizon = 96,
        int numFeatures = 7,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture,
               lossFunction ?? new MeanSquaredErrorLoss<T>(),
               1.0)
    {
        // Validate ONNX model path
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        _useNativeMode = false;
        _onnxModelPath = onnxModelPath;
        _sequenceLength = sequenceLength;
        _predictionHorizon = predictionHorizon;
        _numFeatures = numFeatures;
        _numLayers = 2;
        _numHeads = 8;
        _modelDimension = 512;
        _feedForwardDimension = 512;
        _useInstanceNormalization = true;
        _dropout = 0.1;

        InferenceSession? session = null;
        try
        {
            session = new InferenceSession(onnxModelPath);
            _onnxSession = session;
            _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
            _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
            InitializeLayers();
        }
        catch
        {
            session?.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Creates an iTransformer network using native library layers.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="sequenceLength">Input sequence length (default: 96).</param>
    /// <param name="predictionHorizon">Prediction horizon (default: 96).</param>
    /// <param name="numFeatures">Number of input features (default: 7).</param>
    /// <param name="numLayers">Number of transformer encoder layers (default: 2).</param>
    /// <param name="numHeads">Number of attention heads (default: 8).</param>
    /// <param name="modelDimension">Model dimension (default: 512).</param>
    /// <param name="feedForwardDimension">Feedforward dimension (default: 512).</param>
    /// <param name="useInstanceNormalization">Whether to use RevIN (default: true).</param>
    /// <param name="dropout">Dropout rate (default: 0.1).</param>
    /// <param name="optimizer">Optional optimizer for training.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor to train an iTransformer model from scratch.
    /// iTransformer works by treating each variable as a token and learning cross-variable
    /// relationships through attention.
    ///
    /// Key parameters:
    /// - <b>sequenceLength</b>: How far back to look (e.g., 96 hours of history)
    /// - <b>predictionHorizon</b>: How far ahead to forecast (e.g., 96 hours ahead)
    /// - <b>numFeatures</b>: How many variables you have (e.g., 7 for OHLCV + indicators)
    /// - <b>numHeads</b>: Each head learns different cross-variable relationships
    ///
    /// Example:
    /// <code>
    /// var arch = new NeuralNetworkArchitecture&lt;double&gt;(
    ///     inputType: InputType.ThreeDimensional,
    ///     inputHeight: 96,
    ///     inputWidth: 7);
    ///
    /// var model = new ITransformer&lt;double&gt;(arch,
    ///     sequenceLength: 96,
    ///     predictionHorizon: 96,
    ///     numFeatures: 7);
    ///
    /// model.Train(inputs, targets);
    /// var forecast = model.Predict(newData);
    /// </code>
    /// </para>
    /// </remarks>
    public ITransformer(
        NeuralNetworkArchitecture<T> architecture,
        int sequenceLength = 96,
        int predictionHorizon = 96,
        int numFeatures = 7,
        int numLayers = 2,
        int numHeads = 8,
        int modelDimension = 512,
        int feedForwardDimension = 512,
        bool useInstanceNormalization = true,
        double dropout = 0.1,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture,
               lossFunction ?? new MeanSquaredErrorLoss<T>(),
               1.0)
    {
        ValidateParameters(sequenceLength, predictionHorizon, numFeatures, numLayers, numHeads, modelDimension);

        _useNativeMode = true;
        _sequenceLength = sequenceLength;
        _predictionHorizon = predictionHorizon;
        _numFeatures = numFeatures;
        _numLayers = numLayers;
        _numHeads = numHeads;
        _modelDimension = modelDimension;
        _feedForwardDimension = feedForwardDimension;
        _useInstanceNormalization = useInstanceNormalization;
        _dropout = dropout;

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        InitializeLayers();
    }

    /// <summary>
    /// Validates constructor parameters.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method checks that all parameters are valid before
    /// creating the model. It catches common mistakes like setting dimensions to zero
    /// or using incompatible values (e.g., model dimension not divisible by number of heads).
    /// </para>
    /// </remarks>
    private static void ValidateParameters(int sequenceLength, int predictionHorizon, int numFeatures,
        int numLayers, int numHeads, int modelDimension)
    {
        if (sequenceLength < 1)
            throw new ArgumentOutOfRangeException(nameof(sequenceLength), "Sequence length must be at least 1.");
        if (predictionHorizon < 1)
            throw new ArgumentOutOfRangeException(nameof(predictionHorizon), "Prediction horizon must be at least 1.");
        if (numFeatures < 1)
            throw new ArgumentOutOfRangeException(nameof(numFeatures), "Number of features must be at least 1.");
        if (numLayers < 1)
            throw new ArgumentOutOfRangeException(nameof(numLayers), "Number of layers must be at least 1.");
        if (numHeads < 1)
            throw new ArgumentOutOfRangeException(nameof(numHeads), "Number of heads must be at least 1.");
        if (modelDimension % numHeads != 0)
            throw new ArgumentException("Model dimension must be divisible by number of heads.", nameof(modelDimension));
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes the neural network layers for iTransformer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method sets up the building blocks of the iTransformer model.
    /// The iTransformer architecture has three main components:
    /// </para>
    /// <para>
    /// <list type="number">
    /// <item><b>Variate Embedding:</b> Converts each variable's entire time series into a
    /// dense vector (token). Unlike PatchTST which creates multiple tokens per variable,
    /// iTransformer creates exactly one token per variable.</item>
    /// <item><b>Transformer Encoders:</b> Use attention to learn how variables relate to
    /// each other. For example, learning that high volume often precedes price movements.</item>
    /// <item><b>Output Projection:</b> Converts each variable's learned representation
    /// into a forecast for that variable.</item>
    /// </list>
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultITransformerLayers(
                Architecture, _sequenceLength, _predictionHorizon, _numFeatures,
                _numLayers, _numHeads, _modelDimension, _feedForwardDimension));

            ExtractLayerReferences();
        }
    }

    /// <summary>
    /// Extracts references to specific layers from the layer collection for direct access.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> iTransformer has several distinct components that need to be
    /// accessed individually during the forward pass:
    /// </para>
    /// <para>
    /// <list type="bullet">
    /// <item><b>Variate Embedding:</b> The first layer that converts each variable's
    /// time series into an embedding token.</item>
    /// <item><b>Encoder Layers:</b> Multiple transformer encoder layers that process
    /// the variable tokens using cross-variable attention.</item>
    /// <item><b>Final Normalization:</b> Layer normalization for stable outputs.</item>
    /// <item><b>Output Projection:</b> Maps learned representations to forecasts.</item>
    /// </list>
    /// </para>
    /// </remarks>
    private void ExtractLayerReferences()
    {
        int idx = 0;
        if (Layers.Count > idx)
            _variateEmbedding = Layers[idx++];

        for (int i = 0; i < _numLayers && idx < Layers.Count; i++)
        {
            _encoderLayers.Add(Layers[idx++]);
        }

        if (Layers.Count > idx)
            _finalNorm = Layers[idx++];

        if (Layers.Count > idx)
            _outputProjection = Layers[idx];
    }

    /// <summary>
    /// Validates that custom layers meet iTransformer's architectural requirements.
    /// </summary>
    /// <param name="layers">The list of custom layers to validate.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> iTransformer requires a minimum of 3 layers to function:
    /// </para>
    /// <para>
    /// <list type="number">
    /// <item><b>Variate Embedding:</b> Converts variable time series to embeddings</item>
    /// <item><b>At least one Encoder:</b> Processes embeddings with cross-variable attention</item>
    /// <item><b>Output Projection:</b> Produces the final forecast</item>
    /// </list>
    /// </para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);

        if (layers.Count < 3)
            throw new ArgumentException("iTransformer requires at least 3 layers: variate embedding, encoder(s), and output projection.");
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Training is only supported when using native mode (C# layers),
    /// not when using a pretrained ONNX model. Native mode allows you to adjust the model's
    /// parameters to fit your specific data.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => _useNativeMode;

    /// <summary>
    /// Makes a prediction using the input tensor.
    /// </summary>
    /// <param name="input">Input tensor of shape [batch_size, sequence_length, num_features].</param>
    /// <returns>Output tensor of shape [batch_size, prediction_horizon, num_features].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes historical time series data and returns
    /// predictions for future time steps. The input should contain your historical data
    /// organized as [samples, time_steps, features].
    /// </para>
    /// <para>
    /// Example:
    /// <code>
    /// // Input: 96 time steps, 7 features
    /// var historicalData = new Tensor&lt;double&gt;(new[] { 1, 96, 7 }, data);
    /// var forecast = model.Predict(historicalData);
    /// // Output: 96 predicted time steps, 7 features
    /// </code>
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return Forecast(input);
    }

    /// <summary>
    /// Trains the model on a single batch of input-output pairs.
    /// </summary>
    /// <param name="input">Input tensor of shape [batch_size, sequence_length, num_features].</param>
    /// <param name="expectedOutput">Target tensor of shape [batch_size, prediction_horizon, num_features].</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Training teaches the model to make better predictions.
    /// The process involves:
    /// </para>
    /// <para>
    /// <list type="number">
    /// <item><b>Forward pass:</b> Make a prediction using the current parameters</item>
    /// <item><b>Loss calculation:</b> Measure how wrong the prediction was</item>
    /// <item><b>Backward pass:</b> Calculate gradients (how to improve)</item>
    /// <item><b>Parameter update:</b> Adjust parameters to reduce error</item>
    /// </list>
    /// </para>
    /// <para>
    /// In iTransformer, the model learns which cross-variable relationships are most
    /// useful for forecasting by adjusting attention weights during training.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown if called in ONNX mode.</exception>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is not supported in ONNX mode.");

        SetTrainingMode(true);

        var prediction = Forward(input);
        LastLoss = _lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());

        var outputGradient = _lossFunction.CalculateDerivative(prediction.ToVector(), expectedOutput.ToVector());
        Backward(Tensor<T>.FromVector(outputGradient));

        _optimizer.UpdateParameters(Layers);

        SetTrainingMode(false);
    }

    /// <summary>
    /// Updates the model's parameters from a flat parameter vector.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters for all layers.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Neural networks have many parameters (weights and biases).
    /// This method allows setting all parameters at once from a flattened vector,
    /// useful for loading saved models or applying external optimization results.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (parameters is null)
            throw new ArgumentNullException(nameof(parameters));

        int offset = 0;
        foreach (var layer in Layers)
        {
            var layerParams = layer.GetParameters();
            var newParams = parameters.Slice(offset, layerParams.Length);
            layer.SetParameters(newParams);
            offset += layerParams.Length;
        }
    }

    /// <summary>
    /// Gets metadata about the model for serialization and inspection.
    /// </summary>
    /// <returns>A ModelMetadata object containing model information.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Metadata describes the model's configuration and structure.
    /// This is useful for saving/loading models, logging experiments, and understanding
    /// what a saved model does without loading it fully.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "iTransformer" },
                { "SequenceLength", _sequenceLength },
                { "PredictionHorizon", _predictionHorizon },
                { "NumFeatures", _numFeatures },
                { "NumLayers", _numLayers },
                { "NumHeads", _numHeads },
                { "ModelDimension", _modelDimension },
                { "UseNativeMode", _useNativeMode },
                { "ParameterCount", GetParameterCount() }
            },
            ModelData = _useNativeMode ? this.Serialize() : Array.Empty<byte>()
        };
    }

    /// <summary>
    /// Creates a new instance of this model with the same configuration.
    /// </summary>
    /// <returns>A new iTransformer instance with identical settings.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a fresh copy of the model with the same architecture
    /// but newly initialized parameters. Useful for:
    /// <list type="bullet">
    /// <item>Creating ensemble models</item>
    /// <item>Cross-validation (fresh model for each fold)</item>
    /// <item>Resetting training</item>
    /// </list>
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (_useNativeMode)
        {
            return new ITransformer<T>(
                Architecture, _sequenceLength, _predictionHorizon, _numFeatures,
                _numLayers, _numHeads, _modelDimension, _feedForwardDimension,
                _useInstanceNormalization, _dropout, _optimizer, _lossFunction);
        }
        else
        {
            return new ITransformer<T>(
                Architecture, _onnxModelPath ?? string.Empty,
                _sequenceLength, _predictionHorizon, _numFeatures,
                _optimizer, _lossFunction);
        }
    }

    /// <summary>
    /// Writes network-specific configuration data during serialization.
    /// </summary>
    /// <param name="writer">The binary writer to write data to.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Serialization converts the model to bytes for saving to disk.
    /// This method saves all iTransformer-specific settings like sequence length,
    /// number of layers, and attention heads.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_sequenceLength);
        writer.Write(_predictionHorizon);
        writer.Write(_numFeatures);
        writer.Write(_numLayers);
        writer.Write(_numHeads);
        writer.Write(_modelDimension);
        writer.Write(_feedForwardDimension);
        writer.Write(_useInstanceNormalization);
        writer.Write(_dropout);
    }

    /// <summary>
    /// Reads network-specific configuration data during deserialization.
    /// </summary>
    /// <param name="reader">The binary reader to read data from.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Deserialization loads a model from bytes (e.g., from a saved file).
    /// This method reads back the iTransformer-specific settings that were saved.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32(); // sequenceLength
        _ = reader.ReadInt32(); // predictionHorizon
        _ = reader.ReadInt32(); // numFeatures
        _ = reader.ReadInt32(); // numLayers
        _ = reader.ReadInt32(); // numHeads
        _ = reader.ReadInt32(); // modelDimension
        _ = reader.ReadInt32(); // feedForwardDimension
        _ = reader.ReadBoolean(); // useInstanceNormalization
        _ = reader.ReadDouble(); // dropout
    }

    #endregion

    #region IForecastingModel Implementation

    /// <summary>
    /// Generates forecasts with optional uncertainty quantification.
    /// </summary>
    /// <param name="input">Input tensor of shape [batch_size, sequence_length, num_features].</param>
    /// <param name="quantiles">Optional quantiles for prediction intervals (e.g., [0.1, 0.5, 0.9]).</param>
    /// <returns>Forecast tensor of shape [batch_size, prediction_horizon, num_features].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method predicts future values for all your variables.
    /// The input is your historical data, and the output is predictions for each variable.
    /// </para>
    /// <para>
    /// Unlike PatchTST which processes patches, iTransformer processes each variable's
    /// entire time series as a single token, then uses attention to learn cross-variable
    /// relationships before generating forecasts.
    /// </para>
    /// </remarks>
    public Tensor<T> Forecast(Tensor<T> input, double[]? quantiles = null)
    {
        if (input is null)
            throw new ArgumentNullException(nameof(input));

        if (_useNativeMode)
        {
            return ForecastNative(input, quantiles);
        }
        else
        {
            return ForecastOnnx(input);
        }
    }

    /// <summary>
    /// Generates multi-step forecasts iteratively (autoregressive forecasting).
    /// </summary>
    /// <param name="input">Initial input tensor.</param>
    /// <param name="steps">Number of future steps to predict.</param>
    /// <returns>Tensor containing predictions for all requested steps.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When you need to predict further than the model's native
    /// prediction horizon, autoregressive forecasting helps by:
    /// </para>
    /// <para>
    /// <list type="number">
    /// <item>Predict the next N steps (where N is the prediction horizon)</item>
    /// <item>Add those predictions to the input</item>
    /// <item>Predict the next N steps using the updated input</item>
    /// <item>Repeat until reaching the desired number of steps</item>
    /// </list>
    /// </para>
    /// <para>
    /// Note: Accuracy decreases for distant predictions because errors compound.
    /// </para>
    /// </remarks>
    public Tensor<T> AutoregressiveForecast(Tensor<T> input, int steps)
    {
        if (input is null)
            throw new ArgumentNullException(nameof(input));
        if (steps < 1)
            throw new ArgumentOutOfRangeException(nameof(steps), "Steps must be at least 1.");

        var currentInput = input;
        var allPredictions = new List<Tensor<T>>();

        int stepsRemaining = steps;
        while (stepsRemaining > 0)
        {
            var forecast = Forecast(currentInput);
            int stepsToUse = Math.Min(stepsRemaining, _predictionHorizon);
            allPredictions.Add(forecast);
            stepsRemaining -= stepsToUse;

            if (stepsRemaining > 0)
            {
                currentInput = ShiftInputWithPredictions(currentInput, forecast, stepsToUse);
            }
        }

        return ConcatenatePredictions(allPredictions, steps);
    }

    /// <summary>
    /// Evaluates the model's forecasting performance on test data.
    /// </summary>
    /// <param name="inputs">Test input sequences.</param>
    /// <param name="targets">Actual target values.</param>
    /// <returns>Dictionary containing forecasting metrics (MAE, MSE, RMSE).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Evaluation measures how well your trained model performs
    /// on data it hasn't seen before. The metrics returned are:
    /// </para>
    /// <para>
    /// <list type="bullet">
    /// <item><b>MAE:</b> Average absolute error - easy to interpret</item>
    /// <item><b>MSE:</b> Average squared error - penalizes large errors more</item>
    /// <item><b>RMSE:</b> Square root of MSE - same units as your data</item>
    /// </list>
    /// </para>
    /// <para>
    /// Lower values are better for all metrics.
    /// </para>
    /// </remarks>
    public Dictionary<string, T> Evaluate(Tensor<T> inputs, Tensor<T> targets)
    {
        if (inputs is null)
            throw new ArgumentNullException(nameof(inputs));
        if (targets is null)
            throw new ArgumentNullException(nameof(targets));

        var predictions = Forecast(inputs);
        var metrics = new Dictionary<string, T>();

        T maeSum = NumOps.Zero;
        int count = predictions.Length;
        for (int i = 0; i < count; i++)
        {
            T diff = NumOps.Subtract(predictions.Data.Span[i], targets.Data.Span[i]);
            maeSum = NumOps.Add(maeSum, NumOps.Abs(diff));
        }
        metrics["MAE"] = NumOps.Divide(maeSum, NumOps.FromDouble(count));

        T mseSum = NumOps.Zero;
        for (int i = 0; i < count; i++)
        {
            T diff = NumOps.Subtract(predictions.Data.Span[i], targets.Data.Span[i]);
            mseSum = NumOps.Add(mseSum, NumOps.Multiply(diff, diff));
        }
        metrics["MSE"] = NumOps.Divide(mseSum, NumOps.FromDouble(count));
        metrics["RMSE"] = NumOps.Sqrt(metrics["MSE"]);

        return metrics;
    }

    /// <summary>
    /// Applies instance normalization (RevIN) during inference for distribution shift handling.
    /// </summary>
    /// <param name="input">Input tensor to normalize.</param>
    /// <returns>Normalized tensor with zero mean and unit variance per instance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> RevIN helps the model handle data where statistics change over time.
    /// For example, if stock prices go from $100 to $1000, RevIN normalizes each sequence
    /// so the model sees consistent, standardized data regardless of the absolute scale.
    /// </para>
    /// </remarks>
    public Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
    {
        if (!_useInstanceNormalization)
            return input;

        return ApplyRevIN(input, normalize: true);
    }

    /// <summary>
    /// Gets financial-specific metrics from the model.
    /// </summary>
    /// <returns>Dictionary containing model configuration metrics.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns information about the model's configuration:
    /// </para>
    /// <para>
    /// <list type="bullet">
    /// <item><b>SequenceLength:</b> Input history length</item>
    /// <item><b>PredictionHorizon:</b> Forecast length</item>
    /// <item><b>NumFeatures:</b> Number of variables (tokens in iTransformer)</item>
    /// <item><b>ParameterCount:</b> Total trainable parameters</item>
    /// </list>
    /// </para>
    /// </remarks>
    public Dictionary<string, T> GetFinancialMetrics()
    {
        return new Dictionary<string, T>
        {
            ["SequenceLength"] = NumOps.FromDouble(_sequenceLength),
            ["PredictionHorizon"] = NumOps.FromDouble(_predictionHorizon),
            ["NumFeatures"] = NumOps.FromDouble(_numFeatures),
            ["NumTokens"] = NumOps.FromDouble(_numFeatures), // In iTransformer, each feature is a token
            ["ParameterCount"] = NumOps.FromDouble(GetParameterCount())
        };
    }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// Performs the forward pass through the iTransformer network.
    /// </summary>
    /// <param name="input">Input tensor containing time series data.</param>
    /// <returns>Output tensor containing the forecast.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The forward pass is how data flows through the network.
    /// In iTransformer, the key innovation is the "inverted" approach:
    /// </para>
    /// <para>
    /// <list type="number">
    /// <item><b>RevIN (optional):</b> Normalize input to handle distribution shift</item>
    /// <item><b>Inversion:</b> Transpose data so each variable becomes a token
    /// (shape changes from [batch, time, features] to [batch, features, time])</item>
    /// <item><b>Variate Embedding:</b> Embed each variable's time series into a dense vector</item>
    /// <item><b>Cross-Variable Attention:</b> Learn relationships between variables</item>
    /// <item><b>Output Projection:</b> Generate forecasts for each variable</item>
    /// <item><b>Un-inversion:</b> Transpose back to [batch, time, features]</item>
    /// <item><b>Reverse RevIN:</b> Scale back to original data range</item>
    /// </list>
    /// </para>
    /// </remarks>
    private Tensor<T> Forward(Tensor<T> input)
    {
        var processed = _useInstanceNormalization ? ApplyRevIN(input, normalize: true) : input;

        // Invert: transpose from [batch, seq, features] to [batch, features, seq]
        var inverted = InvertInput(processed);

        // Process through layers
        var output = inverted;
        foreach (var layer in Layers)
        {
            output = layer.Forward(output);
        }

        // Un-invert: transpose back from [batch, features, pred] to [batch, pred, features]
        output = UninvertOutput(output);

        if (_useInstanceNormalization)
        {
            output = ApplyRevIN(output, normalize: false);
        }

        return output;
    }

    /// <summary>
    /// Performs the backward pass to compute gradients for training.
    /// </summary>
    /// <param name="outputGradient">Gradient of the loss with respect to the output.</param>
    /// <returns>Gradient of the loss with respect to the input.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The backward pass calculates how each parameter contributed
    /// to the prediction error. Gradients flow backward through:
    /// </para>
    /// <para>
    /// <list type="number">
    /// <item>Output projection</item>
    /// <item>Layer normalization</item>
    /// <item>Each transformer encoder (in reverse order)</item>
    /// <item>Variate embedding</item>
    /// </list>
    /// </para>
    /// <para>
    /// This is called "backpropagation" and is the foundation of neural network training.
    /// The gradients tell the optimizer how to adjust parameters to reduce error.
    /// </para>
    /// </remarks>
    private Tensor<T> Backward(Tensor<T> outputGradient)
    {
        var gradient = outputGradient;
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            gradient = Layers[i].Backward(gradient);
        }
        return gradient;
    }

    /// <summary>
    /// Generates forecasts using native C# layers.
    /// </summary>
    /// <param name="input">Input tensor containing time series data.</param>
    /// <param name="quantiles">Optional quantiles for uncertainty estimation.</param>
    /// <returns>Forecast tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method runs the iTransformer using pure C# neural network
    /// layers. It ensures the model is in inference mode (no dropout) before generating
    /// predictions.
    /// </para>
    /// </remarks>
    private Tensor<T> ForecastNative(Tensor<T> input, double[]? quantiles)
    {
        SetTrainingMode(false);
        return Forward(input);
    }

    /// <summary>
    /// Generates forecasts using a pretrained ONNX model.
    /// </summary>
    /// <param name="input">Input tensor containing time series data.</param>
    /// <returns>Forecast tensor from the ONNX model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> ONNX mode uses a pretrained model for inference.
    /// The process converts data to ONNX format, runs the model, and converts back.
    /// </para>
    /// </remarks>
    private Tensor<T> ForecastOnnx(Tensor<T> input)
    {
        if (_onnxSession is null)
            throw new InvalidOperationException("ONNX session is not initialized.");

        var inputData = new float[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            inputData[i] = Convert.ToSingle(input.Data.Span[i]);
        }

        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input.Shape);
        var inputMeta = _onnxSession.InputMetadata;
        string inputName = inputMeta.Keys.First();

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, onnxInput)
        };

        using var results = _onnxSession.Run(inputs);
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

    #region iTransformer-Specific Processing

    /// <summary>
    /// Inverts the input tensor by transposing time and feature dimensions.
    /// </summary>
    /// <param name="input">Input tensor of shape [batch, seq_len, features].</param>
    /// <returns>Inverted tensor of shape [batch, features, seq_len].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the key operation that makes iTransformer different.
    /// Traditional transformers treat time steps as tokens. By inverting (transposing),
    /// we make each variable/feature into a token instead.
    /// </para>
    /// <para>
    /// After inversion, when attention is computed, each variable "looks at" all other
    /// variables to understand their relationships. This is particularly useful for
    /// multivariate forecasting where variables are correlated (like OHLCV data).
    /// </para>
    /// </remarks>
    private Tensor<T> InvertInput(Tensor<T> input)
    {
        int batchSize = input.Rank == 3 ? input.Shape[0] : 1;
        int seqLen = input.Rank == 3 ? input.Shape[1] : input.Shape[0];
        int features = input.Rank == 3 ? input.Shape[2] : input.Shape[1];

        var invertedData = new T[input.Length];

        for (int b = 0; b < batchSize; b++)
        {
            for (int f = 0; f < features; f++)
            {
                for (int t = 0; t < seqLen; t++)
                {
                    int srcIdx = (b * seqLen * features) + (t * features) + f;
                    int dstIdx = (b * features * seqLen) + (f * seqLen) + t;
                    invertedData[dstIdx] = input.Data.Span[srcIdx];
                }
            }
        }

        return new Tensor<T>(new[] { batchSize, features, seqLen }, new Vector<T>(invertedData));
    }

    /// <summary>
    /// Un-inverts the output tensor by transposing back to standard layout.
    /// </summary>
    /// <param name="output">Output tensor of shape [batch, features, pred_horizon].</param>
    /// <returns>Un-inverted tensor of shape [batch, pred_horizon, features].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> After processing through the transformer, we need to convert
    /// the output back to the standard time series format [batch, time, features] so it
    /// can be used for evaluation and downstream tasks.
    /// </para>
    /// </remarks>
    private Tensor<T> UninvertOutput(Tensor<T> output)
    {
        int batchSize = output.Shape[0];
        int features = output.Shape[1];
        int predHorizon = output.Shape[2];

        var uninvertedData = new T[output.Length];

        for (int b = 0; b < batchSize; b++)
        {
            for (int t = 0; t < predHorizon; t++)
            {
                for (int f = 0; f < features; f++)
                {
                    int srcIdx = (b * features * predHorizon) + (f * predHorizon) + t;
                    int dstIdx = (b * predHorizon * features) + (t * features) + f;
                    uninvertedData[dstIdx] = output.Data.Span[srcIdx];
                }
            }
        }

        return new Tensor<T>(new[] { batchSize, predHorizon, features }, new Vector<T>(uninvertedData));
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Applies Reversible Instance Normalization (RevIN) to handle distribution shift.
    /// </summary>
    /// <param name="input">Input tensor to normalize or denormalize.</param>
    /// <param name="normalize">True to normalize (before model), false to denormalize (after model).</param>
    /// <returns>Normalized or denormalized tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> RevIN handles the common problem in time series where data
    /// statistics change over time (non-stationarity). For example, stock prices might
    /// average $100 one year and $500 the next.
    /// </para>
    /// <para>
    /// RevIN normalizes each input sequence to zero mean and unit variance, processes it
    /// through the model, then reverses the normalization so predictions are in the
    /// original scale.
    /// </para>
    /// </remarks>
    private Tensor<T> ApplyRevIN(Tensor<T> input, bool normalize)
    {
        var result = new Tensor<T>(input.Shape);
        T epsilon = NumOps.FromDouble(1e-5);

        if (normalize)
        {
            _instanceMean = CalculateInstanceMean(input);
            _instanceStd = CalculateInstanceStd(input, _instanceMean);

            for (int i = 0; i < input.Length; i++)
            {
                int statIdx = i % _numFeatures;
                T centered = NumOps.Subtract(input.Data.Span[i], _instanceMean.Data.Span[statIdx]);
                T stdVal = NumOps.Add(_instanceStd.Data.Span[statIdx], epsilon);
                result.Data.Span[i] = NumOps.Divide(centered, stdVal);
            }
        }
        else
        {
            if (_instanceMean is null || _instanceStd is null)
                return input;

            for (int i = 0; i < input.Length; i++)
            {
                int statIdx = i % _numFeatures;
                T scaled = NumOps.Multiply(input.Data.Span[i], _instanceStd.Data.Span[statIdx]);
                result.Data.Span[i] = NumOps.Add(scaled, _instanceMean.Data.Span[statIdx]);
            }
        }

        return result;
    }

    /// <summary>
    /// Calculates the mean value for each feature across the time dimension.
    /// </summary>
    /// <param name="input">Input tensor to calculate mean from.</param>
    /// <returns>Tensor of shape [num_features] containing the mean of each feature.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Calculates the average value of each variable over time.
    /// This is used by RevIN to center the data (subtract mean so values are around 0).
    /// </para>
    /// </remarks>
    private Tensor<T> CalculateInstanceMean(Tensor<T> input)
    {
        var mean = new T[_numFeatures];
        int samplesPerFeature = input.Length / _numFeatures;

        for (int f = 0; f < _numFeatures; f++)
        {
            T sum = NumOps.Zero;
            for (int i = f; i < input.Length; i += _numFeatures)
            {
                sum = NumOps.Add(sum, input.Data.Span[i]);
            }
            mean[f] = NumOps.Divide(sum, NumOps.FromDouble(samplesPerFeature));
        }

        return new Tensor<T>(new[] { _numFeatures }, new Vector<T>(mean));
    }

    /// <summary>
    /// Calculates the standard deviation for each feature across the time dimension.
    /// </summary>
    /// <param name="input">Input tensor to calculate standard deviation from.</param>
    /// <param name="mean">Pre-calculated mean values for each feature.</param>
    /// <returns>Tensor of shape [num_features] containing the standard deviation of each feature.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Standard deviation measures how spread out values are.
    /// RevIN divides by std to scale data to a consistent range (roughly -2 to +2).
    /// </para>
    /// </remarks>
    private Tensor<T> CalculateInstanceStd(Tensor<T> input, Tensor<T> mean)
    {
        var std = new T[_numFeatures];
        int samplesPerFeature = input.Length / _numFeatures;

        for (int f = 0; f < _numFeatures; f++)
        {
            T sumSq = NumOps.Zero;
            for (int i = f; i < input.Length; i += _numFeatures)
            {
                T diff = NumOps.Subtract(input.Data.Span[i], mean.Data.Span[f]);
                sumSq = NumOps.Add(sumSq, NumOps.Multiply(diff, diff));
            }
            T variance = NumOps.Divide(sumSq, NumOps.FromDouble(samplesPerFeature));
            std[f] = NumOps.Sqrt(variance);
        }

        return new Tensor<T>(new[] { _numFeatures }, new Vector<T>(std));
    }

    /// <summary>
    /// Shifts the input window forward in time by incorporating new predictions.
    /// </summary>
    /// <param name="input">Original input tensor.</param>
    /// <param name="predictions">New predictions to append.</param>
    /// <param name="stepsToShift">Number of time steps to shift.</param>
    /// <returns>New input tensor with predictions incorporated.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Used for autoregressive forecasting. Shifts the input window
    /// forward by dropping old data and adding predictions as new data.
    /// </para>
    /// </remarks>
    private Tensor<T> ShiftInputWithPredictions(Tensor<T> input, Tensor<T> predictions, int stepsToShift)
    {
        var newData = new T[input.Length];
        int shiftAmount = stepsToShift * _numFeatures;

        Array.Copy(input.Data.ToArray(), shiftAmount, newData, 0, input.Length - shiftAmount);
        Array.Copy(predictions.Data.ToArray(), 0, newData, input.Length - shiftAmount, shiftAmount);

        return new Tensor<T>(input.Shape, new Vector<T>(newData));
    }

    /// <summary>
    /// Concatenates multiple prediction tensors into a single long forecast.
    /// </summary>
    /// <param name="predictions">List of prediction tensors from autoregressive steps.</param>
    /// <param name="totalSteps">Total number of steps requested in the forecast.</param>
    /// <returns>Combined forecast tensor with all requested steps.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Combines multiple shorter predictions into one long forecast.
    /// Used when autoregressive forecasting produces multiple chunks of predictions.
    /// </para>
    /// </remarks>
    private Tensor<T> ConcatenatePredictions(List<Tensor<T>> predictions, int totalSteps)
    {
        var outputData = new T[totalSteps * _numFeatures];
        int currentIdx = 0;

        foreach (var pred in predictions)
        {
            int stepsToCopy = Math.Min(_predictionHorizon, totalSteps - currentIdx / _numFeatures);
            int elementsToCopy = stepsToCopy * _numFeatures;

            Array.Copy(pred.Data.ToArray(), 0, outputData, currentIdx, elementsToCopy);
            currentIdx += elementsToCopy;

            if (currentIdx >= totalSteps * _numFeatures)
                break;
        }

        return new Tensor<T>(new[] { totalSteps, _numFeatures }, new Vector<T>(outputData));
    }

    #endregion

    #region IDisposable

    /// <summary>
    /// Releases resources used by the iTransformer model.
    /// </summary>
    /// <param name="disposing">True if disposing managed resources, false if called from finalizer.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Cleanup method that releases resources when the model is no longer needed.
    /// In ONNX mode, this releases the inference session. Use the model in a using statement
    /// for automatic cleanup:
    /// <code>
    /// using var model = new ITransformer&lt;double&gt;(architecture);
    /// // Use the model...
    /// // Automatically cleaned up here
    /// </code>
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
