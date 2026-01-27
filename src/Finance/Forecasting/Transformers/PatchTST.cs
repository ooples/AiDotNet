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
/// PatchTST (Patch Time Series Transformer) neural network for long-term time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// PatchTST is a state-of-the-art transformer model for long-term time series forecasting.
/// It introduces patching (dividing time series into segments) and channel independence
/// to achieve efficient and accurate forecasting.
/// </para>
/// <para>
/// Reference: Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting
/// with Transformers", ICLR 2023. https://arxiv.org/abs/2211.14730
/// </para>
/// </remarks>
public class PatchTST<T> : NeuralNetworkBase<T>, IForecastingModel<T>
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
    /// Patch embedding layer.
    /// </summary>
    private ILayer<T>? _patchEmbedding;

    /// <summary>
    /// Transformer encoder layers.
    /// </summary>
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
    /// Positional encoding for patches.
    /// </summary>
    private Tensor<T>? _positionalEncoding;

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
    /// The patch size.
    /// </summary>
    private readonly int _patchSize;

    /// <summary>
    /// The stride between patches.
    /// </summary>
    private readonly int _stride;

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
    /// Whether to use channel-independent mode.
    /// </summary>
    private readonly bool _channelIndependent;

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
    public int PatchSize => _patchSize;

    /// <inheritdoc/>
    public int Stride => _stride;

    /// <inheritdoc/>
    public bool IsChannelIndependent => _channelIndependent;

    /// <inheritdoc/>
    public bool UseNativeMode => _useNativeMode;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a PatchTST network using pretrained ONNX model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="sequenceLength">Input sequence length (default: 96).</param>
    /// <param name="predictionHorizon">Prediction horizon (default: 24).</param>
    /// <param name="numFeatures">Number of input features (default: 7).</param>
    /// <param name="patchSize">Patch size (default: 16).</param>
    /// <param name="stride">Stride between patches (default: 8).</param>
    /// <param name="optimizer">Optional optimizer for fine-tuning.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor when you have a pretrained ONNX model.
    ///
    /// ONNX (Open Neural Network Exchange) is a format for sharing trained models.
    /// This constructor loads a model that has already learned patterns from data,
    /// so you can use it immediately for predictions without training.
    ///
    /// Example:
    /// <code>
    /// var arch = new NeuralNetworkArchitecture&lt;double&gt;(
    ///     inputType: InputType.ThreeDimensional,
    ///     inputHeight: 96,   // sequence_length
    ///     inputWidth: 7);    // num_features
    ///
    /// var model = new PatchTST&lt;double&gt;(arch, "patchtst_etth1.onnx");
    /// var forecast = model.Predict(historicalData);
    /// </code>
    /// </para>
    /// </remarks>
    public PatchTST(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int sequenceLength = 96,
        int predictionHorizon = 24,
        int numFeatures = 7,
        int patchSize = 16,
        int stride = 8,
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
        _patchSize = patchSize;
        _stride = stride;
        _numLayers = 3;
        _numHeads = 4;
        _modelDimension = 128;
        _feedForwardDimension = 256;
        _channelIndependent = true;
        _useInstanceNormalization = true;
        _dropout = 0.05;

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
    /// Creates a PatchTST network using native library layers.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="sequenceLength">Input sequence length (default: 96).</param>
    /// <param name="predictionHorizon">Prediction horizon (default: 24).</param>
    /// <param name="numFeatures">Number of input features (default: 7).</param>
    /// <param name="patchSize">Patch size (default: 16).</param>
    /// <param name="stride">Stride between patches (default: 8).</param>
    /// <param name="numLayers">Number of transformer encoder layers (default: 3).</param>
    /// <param name="numHeads">Number of attention heads (default: 4).</param>
    /// <param name="modelDimension">Model dimension (default: 128).</param>
    /// <param name="feedForwardDimension">Feedforward dimension (default: 256).</param>
    /// <param name="channelIndependent">Whether to use channel-independent mode (default: true).</param>
    /// <param name="useInstanceNormalization">Whether to use RevIN (default: true).</param>
    /// <param name="dropout">Dropout rate (default: 0.05).</param>
    /// <param name="optimizer">Optional optimizer for training.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to train a PatchTST model from scratch.
    ///
    /// PatchTST is like reading a book by looking at groups of words (patches) instead of
    /// individual letters. This makes it faster and often more accurate for time series forecasting.
    ///
    /// Key parameters explained:
    /// - <b>sequenceLength</b>: How far back in time the model looks (e.g., 96 hours of history)
    /// - <b>predictionHorizon</b>: How far ahead to forecast (e.g., next 24 hours)
    /// - <b>patchSize</b>: How many time steps are grouped together (e.g., 16 hours per patch)
    /// - <b>channelIndependent</b>: If true, each variable is processed separately (usually better)
    ///
    /// Example:
    /// <code>
    /// var arch = new NeuralNetworkArchitecture&lt;double&gt;(
    ///     inputType: InputType.ThreeDimensional,
    ///     inputHeight: 96,
    ///     inputWidth: 7);
    ///
    /// var model = new PatchTST&lt;double&gt;(arch,
    ///     sequenceLength: 96,
    ///     predictionHorizon: 24);
    ///
    /// model.Train(inputs, targets);
    /// var forecast = model.Predict(newData);
    /// </code>
    /// </para>
    /// </remarks>
    public PatchTST(
        NeuralNetworkArchitecture<T> architecture,
        int sequenceLength = 96,
        int predictionHorizon = 24,
        int numFeatures = 7,
        int patchSize = 16,
        int stride = 8,
        int numLayers = 3,
        int numHeads = 4,
        int modelDimension = 128,
        int feedForwardDimension = 256,
        bool channelIndependent = true,
        bool useInstanceNormalization = true,
        double dropout = 0.05,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture,
               lossFunction ?? new MeanSquaredErrorLoss<T>(),
               1.0)
    {
        ValidateParameters(sequenceLength, predictionHorizon, numFeatures, patchSize, stride, numLayers, numHeads, modelDimension);

        _useNativeMode = true;
        _sequenceLength = sequenceLength;
        _predictionHorizon = predictionHorizon;
        _numFeatures = numFeatures;
        _patchSize = patchSize;
        _stride = stride;
        _numLayers = numLayers;
        _numHeads = numHeads;
        _modelDimension = modelDimension;
        _feedForwardDimension = feedForwardDimension;
        _channelIndependent = channelIndependent;
        _useInstanceNormalization = useInstanceNormalization;
        _dropout = dropout;

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        InitializeLayers();
    }

    private static void ValidateParameters(int sequenceLength, int predictionHorizon, int numFeatures,
        int patchSize, int stride, int numLayers, int numHeads, int modelDimension)
    {
        if (sequenceLength < 1)
            throw new ArgumentOutOfRangeException(nameof(sequenceLength), "Sequence length must be at least 1.");
        if (predictionHorizon < 1)
            throw new ArgumentOutOfRangeException(nameof(predictionHorizon), "Prediction horizon must be at least 1.");
        if (numFeatures < 1)
            throw new ArgumentOutOfRangeException(nameof(numFeatures), "Number of features must be at least 1.");
        if (patchSize < 1)
            throw new ArgumentOutOfRangeException(nameof(patchSize), "Patch size must be at least 1.");
        if (stride < 1 || stride > patchSize)
            throw new ArgumentOutOfRangeException(nameof(stride), "Stride must be between 1 and patch size.");
        if (sequenceLength < patchSize)
            throw new ArgumentException("Sequence length must be at least patch size.", nameof(sequenceLength));
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
    /// Initializes the neural network layers for PatchTST.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method sets up the building blocks of the PatchTST model.
    /// PatchTST has a specific architecture designed for time series forecasting:
    /// </para>
    /// <para>
    /// <list type="number">
    /// <item><b>Patch Embedding:</b> Converts each patch (segment) of time series into a
    /// dense vector representation that the transformer can process.</item>
    /// <item><b>Transformer Encoders:</b> Multiple layers that use self-attention to learn
    /// which parts of the historical data are most important for prediction.</item>
    /// <item><b>Output Projection:</b> Converts the transformer output into the final
    /// forecast for the prediction horizon.</item>
    /// </list>
    /// </para>
    /// <para>
    /// If you provide custom layers in the architecture, those are used instead.
    /// Otherwise, the default PatchTST layers are created automatically.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            // Use the layers provided by the user
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else if (_useNativeMode)
        {
            // Use default layer configuration
            Layers.AddRange(LayerHelper<T>.CreateDefaultPatchTSTLayers(
                Architecture, _sequenceLength, _predictionHorizon, _numFeatures,
                _patchSize, _stride, _numLayers, _numHeads, _modelDimension, _feedForwardDimension));

            // Store references to specific layers for direct access
            ExtractLayerReferences();

            // Initialize positional encoding
            int numPatches = CalculateNumPatches();
            _positionalEncoding = CreatePositionalEncoding(numPatches, _modelDimension);
        }
    }

    /// <summary>
    /// Extracts references to specific layers from the layer collection for direct access.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> PatchTST has several distinct components that need to be
    /// accessed individually during the forward pass:
    /// </para>
    /// <para>
    /// <list type="bullet">
    /// <item><b>Patch Embedding:</b> The first layer that converts raw time series patches
    /// into embeddings the transformer can understand.</item>
    /// <item><b>Encoder Layers:</b> Multiple transformer encoder layers that process the
    /// embedded patches using self-attention mechanisms.</item>
    /// <item><b>Final Normalization:</b> Layer normalization applied after all encoder layers
    /// to stabilize the outputs.</item>
    /// <item><b>Output Projection:</b> The final layer that maps transformer outputs to
    /// the actual forecast values.</item>
    /// </list>
    /// </para>
    /// <para>
    /// This method organizes these layers so they can be accessed efficiently during inference.
    /// </para>
    /// </remarks>
    private void ExtractLayerReferences()
    {
        // Extract layer references from the Layers collection
        int idx = 0;
        if (Layers.Count > idx)
            _patchEmbedding = Layers[idx++];

        // Skip encoder layers - they're in the Layers collection
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
    /// Validates that custom layers meet PatchTST's architectural requirements.
    /// </summary>
    /// <param name="layers">The list of custom layers to validate.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> PatchTST requires a minimum of 3 layers to function:
    /// </para>
    /// <para>
    /// <list type="number">
    /// <item><b>Patch Embedding:</b> Converts patches to embeddings</item>
    /// <item><b>At least one Encoder:</b> Processes embeddings with attention</item>
    /// <item><b>Output Projection:</b> Produces the final forecast</item>
    /// </list>
    /// </para>
    /// <para>
    /// If you provide custom layers, this method ensures your configuration has
    /// at least these required components. Without them, the model cannot properly
    /// process time series data and generate forecasts.
    /// </para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);

        if (layers.Count < 3)
            throw new ArgumentException("PatchTST requires at least 3 layers: patch embedding, encoder(s), and output projection.");
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This property indicates whether this model can be trained.
    /// Training is only supported when using native mode (C# layers), not when using
    /// a pretrained ONNX model.
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
    /// (e.g., the last 96 hours of stock prices), and the output will contain predictions
    /// for the future (e.g., the next 24 hours).
    /// </para>
    /// <para>
    /// Example:
    /// <code>
    /// // Input: 96 time steps, 7 features
    /// var historicalData = new Tensor&lt;double&gt;(new[] { 1, 96, 7 }, data);
    /// var forecast = model.Predict(historicalData);
    /// // Output: 24 predicted time steps, 7 features
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
    /// <b>For Beginners:</b> Training teaches the model to make better predictions by
    /// showing it examples of historical data and what actually happened next.
    /// </para>
    /// <para>
    /// The training process:
    /// <list type="number">
    /// <item>Forward pass: The model makes a prediction based on the input</item>
    /// <item>Loss calculation: The model measures how wrong its prediction was</item>
    /// <item>Backward pass: The model calculates how to adjust its parameters to reduce error</item>
    /// <item>Parameter update: The optimizer applies the adjustments</item>
    /// </list>
    /// </para>
    /// <para>
    /// Note: Training is only available in native mode, not when using ONNX models.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown if called in ONNX mode.</exception>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is not supported in ONNX mode.");

        SetTrainingMode(true);

        // Forward pass
        var prediction = Forward(input);

        // Calculate loss
        LastLoss = _lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());

        // Backward pass
        var outputGradient = _lossFunction.CalculateDerivative(prediction.ToVector(), expectedOutput.ToVector());
        Backward(Tensor<T>.FromVector(outputGradient));

        // Update parameters
        _optimizer.UpdateParameters(Layers);

        SetTrainingMode(false);
    }

    /// <summary>
    /// Updates the model's parameters from a flat parameter vector.
    /// </summary>
    /// <param name="parameters">A vector containing all parameters for all layers.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Neural networks have many parameters (weights and biases)
    /// that determine how they process data. This method allows you to set all these
    /// parameters at once from a single flattened vector.
    /// </para>
    /// <para>
    /// This is useful for:
    /// <list type="bullet">
    /// <item>Loading saved model parameters</item>
    /// <item>Applying parameters from external optimization algorithms</item>
    /// <item>Ensemble methods that combine parameters from multiple models</item>
    /// </list>
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
    /// <b>For Beginners:</b> Metadata is information about the model itself, such as
    /// its architecture settings, parameter count, and configuration options.
    /// This is useful for:
    /// <list type="bullet">
    /// <item>Saving the model to disk</item>
    /// <item>Displaying model information to users</item>
    /// <item>Logging for experiment tracking</item>
    /// <item>Reproducing the same model configuration later</item>
    /// </list>
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "PatchTST" },
                { "SequenceLength", _sequenceLength },
                { "PredictionHorizon", _predictionHorizon },
                { "NumFeatures", _numFeatures },
                { "PatchSize", _patchSize },
                { "Stride", _stride },
                { "NumLayers", _numLayers },
                { "NumHeads", _numHeads },
                { "ModelDimension", _modelDimension },
                { "ChannelIndependent", _channelIndependent },
                { "UseNativeMode", _useNativeMode },
                { "ParameterCount", GetParameterCount() }
            },
            ModelData = _useNativeMode ? this.Serialize() : Array.Empty<byte>()
        };
    }

    /// <summary>
    /// Creates a new instance of this model with the same configuration.
    /// </summary>
    /// <returns>A new PatchTST instance with identical settings.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method creates a fresh copy of the model with the same
    /// architecture and configuration, but with newly initialized parameters.
    /// </para>
    /// <para>
    /// This is useful for:
    /// <list type="bullet">
    /// <item>Creating multiple models for ensemble methods</item>
    /// <item>Implementing cross-validation where you need a fresh model for each fold</item>
    /// <item>Resetting a model to start training from scratch</item>
    /// </list>
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (_useNativeMode)
        {
            return new PatchTST<T>(
                Architecture, _sequenceLength, _predictionHorizon, _numFeatures,
                _patchSize, _stride, _numLayers, _numHeads, _modelDimension, _feedForwardDimension,
                _channelIndependent, _useInstanceNormalization, _dropout,
                _optimizer, _lossFunction);
        }
        else
        {
            return new PatchTST<T>(
                Architecture, _onnxModelPath ?? string.Empty,
                _sequenceLength, _predictionHorizon, _numFeatures, _patchSize, _stride,
                _optimizer, _lossFunction);
        }
    }

    /// <summary>
    /// Writes network-specific configuration data during serialization.
    /// </summary>
    /// <param name="writer">The binary writer to write data to.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Serialization is the process of converting the model to bytes
    /// that can be saved to a file. This method saves all the PatchTST-specific settings
    /// like sequence length, patch size, and number of transformer layers.
    /// </para>
    /// <para>
    /// When you save a model, this data is written automatically. When you load the model
    /// later, the corresponding DeserializeNetworkSpecificData method reads it back.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_sequenceLength);
        writer.Write(_predictionHorizon);
        writer.Write(_numFeatures);
        writer.Write(_patchSize);
        writer.Write(_stride);
        writer.Write(_numLayers);
        writer.Write(_numHeads);
        writer.Write(_modelDimension);
        writer.Write(_feedForwardDimension);
        writer.Write(_channelIndependent);
        writer.Write(_useInstanceNormalization);
        writer.Write(_dropout);
    }

    /// <summary>
    /// Reads network-specific configuration data during deserialization.
    /// </summary>
    /// <param name="reader">The binary reader to read data from.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Deserialization is the process of loading a model from bytes
    /// (e.g., from a saved file). This method reads back all the PatchTST-specific settings
    /// that were saved during serialization.
    /// </para>
    /// <para>
    /// Note: The values are read to advance the reader position but are not used because
    /// the constructor already sets these values based on the model configuration.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32(); // sequenceLength
        _ = reader.ReadInt32(); // predictionHorizon
        _ = reader.ReadInt32(); // numFeatures
        _ = reader.ReadInt32(); // patchSize
        _ = reader.ReadInt32(); // stride
        _ = reader.ReadInt32(); // numLayers
        _ = reader.ReadInt32(); // numHeads
        _ = reader.ReadInt32(); // modelDimension
        _ = reader.ReadInt32(); // feedForwardDimension
        _ = reader.ReadBoolean(); // channelIndependent
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
    /// <returns>
    /// Forecast tensor. If quantiles are provided, shape is [batch_size, prediction_horizon, num_quantiles].
    /// Otherwise, shape is [batch_size, prediction_horizon, num_features] for point forecasts.
    /// </returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method makes predictions about the future based on historical data.
    /// </para>
    /// <para>
    /// You can also get uncertainty estimates by providing quantiles. For example, quantiles [0.1, 0.5, 0.9] give you:
    /// <list type="bullet">
    /// <item>10th percentile: The value the prediction is 90% likely to exceed</item>
    /// <item>50th percentile: The median prediction (middle value)</item>
    /// <item>90th percentile: The value the prediction has only 10% chance of exceeding</item>
    /// </list>
    /// Together, these form a prediction interval that shows how confident the model is.
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
    /// <b>For Beginners:</b> Sometimes you need to predict further into the future than the model's
    /// native prediction horizon. Autoregressive forecasting solves this by:
    /// </para>
    /// <para>
    /// <list type="number">
    /// <item>Making a prediction for the next N steps (where N is the prediction horizon)</item>
    /// <item>Adding those predictions to the input data</item>
    /// <item>Making another prediction for the following N steps</item>
    /// <item>Repeating until you have predictions for all requested steps</item>
    /// </list>
    /// </para>
    /// <para>
    /// Note: Accuracy typically decreases for more distant predictions because the model
    /// is building on its own (potentially incorrect) earlier predictions.
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
    /// <b>For Beginners:</b> After training a model, you need to know how well it performs
    /// on data it hasn't seen before. This method calculates common error metrics:
    /// </para>
    /// <para>
    /// <list type="bullet">
    /// <item><b>MAE (Mean Absolute Error):</b> Average of the absolute differences between
    /// predictions and actual values. Easy to interpret - it's in the same units as your data.</item>
    /// <item><b>MSE (Mean Squared Error):</b> Average of squared differences. Penalizes
    /// large errors more heavily than small ones.</item>
    /// <item><b>RMSE (Root Mean Squared Error):</b> Square root of MSE. Back in the same
    /// units as your data, but still penalizes large errors more than MAE.</item>
    /// </list>
    /// </para>
    /// <para>
    /// Lower values are better for all these metrics.
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

        // Calculate MAE
        T maeSum = NumOps.Zero;
        int count = predictions.Length;
        for (int i = 0; i < count; i++)
        {
            T diff = NumOps.Subtract(predictions.Data.Span[i], targets.Data.Span[i]);
            maeSum = NumOps.Add(maeSum, NumOps.Abs(diff));
        }
        metrics["MAE"] = NumOps.Divide(maeSum, NumOps.FromDouble(count));

        // Calculate MSE
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
    /// <b>For Beginners:</b> Financial and time series data often experience "distribution shift"
    /// where the statistical properties change over time. For example, stock prices might average
    /// $100 one year and $500 the next.
    /// </para>
    /// <para>
    /// RevIN (Reversible Instance Normalization) helps the model adapt to these changes by:
    /// <list type="number">
    /// <item>Calculating the mean and standard deviation of each input sequence</item>
    /// <item>Normalizing the data to have zero mean and unit variance</item>
    /// <item>Processing the normalized data through the model</item>
    /// <item>Reversing the normalization on the output (so predictions are in original units)</item>
    /// </list>
    /// </para>
    /// <para>
    /// This method performs step 2 (the normalization step).
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
    /// <b>For Beginners:</b> This method returns information about the model's configuration
    /// and capacity. These metrics help you understand the model's complexity and resource requirements.
    /// </para>
    /// <para>
    /// Metrics include:
    /// <list type="bullet">
    /// <item><b>SequenceLength:</b> How many historical time steps the model uses as input</item>
    /// <item><b>PredictionHorizon:</b> How many future time steps the model predicts</item>
    /// <item><b>NumFeatures:</b> How many variables/features the model processes</item>
    /// <item><b>NumPatches:</b> How many patches the sequence is divided into</item>
    /// <item><b>ParameterCount:</b> Total number of trainable parameters in the model</item>
    /// </list>
    /// </para>
    /// </remarks>
    public Dictionary<string, T> GetFinancialMetrics()
    {
        var metrics = new Dictionary<string, T>
        {
            ["SequenceLength"] = NumOps.FromDouble(_sequenceLength),
            ["PredictionHorizon"] = NumOps.FromDouble(_predictionHorizon),
            ["NumFeatures"] = NumOps.FromDouble(_numFeatures),
            ["NumPatches"] = NumOps.FromDouble(CalculateNumPatches()),
            ["ParameterCount"] = NumOps.FromDouble(GetParameterCount())
        };

        return metrics;
    }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// Performs the forward pass through the PatchTST network.
    /// </summary>
    /// <param name="input">Input tensor containing time series data.</param>
    /// <returns>Output tensor containing the forecast.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The forward pass is how data flows through the neural network
    /// to produce a prediction. In PatchTST, the forward pass works as follows:
    /// </para>
    /// <para>
    /// <list type="number">
    /// <item><b>Instance Normalization (RevIN):</b> If enabled, normalizes each input sequence
    /// to have zero mean and unit variance. This helps the model handle data where the
    /// scale changes over time (like stock prices going from $10 to $1000).</item>
    /// <item><b>Channel Processing:</b> Either processes each feature independently
    /// (channel-independent mode, usually better) or all features together.</item>
    /// <item><b>Reverse Normalization:</b> If RevIN was applied, the output is scaled back
    /// to the original data scale so predictions are in meaningful units.</item>
    /// </list>
    /// </para>
    /// </remarks>
    private Tensor<T> Forward(Tensor<T> input)
    {
        // Apply instance normalization if enabled
        var processed = _useInstanceNormalization ? ApplyRevIN(input, normalize: true) : input;

        Tensor<T> output;
        if (_channelIndependent)
        {
            output = ProcessChannelIndependent(processed);
        }
        else
        {
            output = ProcessAllChannels(processed);
        }

        // Reverse instance normalization
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
    /// <b>For Beginners:</b> The backward pass is how neural networks learn. After making
    /// a prediction (forward pass), we measure how wrong it was (the loss). The backward
    /// pass then calculates how much each parameter in the network contributed to that error.
    /// </para>
    /// <para>
    /// In PatchTST, gradients flow backward through:
    /// <list type="number">
    /// <item>Output projection layer</item>
    /// <item>Layer normalization</item>
    /// <item>Each transformer encoder layer (in reverse order)</item>
    /// <item>Patch embedding layer</item>
    /// </list>
    /// </para>
    /// <para>
    /// Each layer receives the gradient from the layer above it, computes how its parameters
    /// affected the error, and passes the gradient to the layer below. This is called
    /// "backpropagation" and is the foundation of how neural networks train.
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
    /// <b>For Beginners:</b> This method runs the PatchTST model using pure C# neural network
    /// layers. It first ensures the model is in inference mode (not training mode, which would
    /// use dropout and other training-specific behaviors), then runs the forward pass.
    /// </para>
    /// <para>
    /// Native mode is used when you want to:
    /// <list type="bullet">
    /// <item>Train the model from scratch on your own data</item>
    /// <item>Fine-tune a model for your specific use case</item>
    /// <item>Have full control over the model's internals</item>
    /// </list>
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
    /// <b>For Beginners:</b> ONNX (Open Neural Network Exchange) is a format for sharing
    /// trained neural networks. This method uses a pretrained PatchTST model saved in ONNX
    /// format to make predictions.
    /// </para>
    /// <para>
    /// The process:
    /// <list type="number">
    /// <item>Convert the input data to ONNX tensor format (float32)</item>
    /// <item>Run the ONNX inference session</item>
    /// <item>Convert the output back to our tensor format</item>
    /// </list>
    /// </para>
    /// <para>
    /// ONNX mode is ideal for production deployment because:
    /// <list type="bullet">
    /// <item>Pretrained models can be used immediately without training</item>
    /// <item>ONNX runtime is highly optimized for inference speed</item>
    /// <item>Models trained in Python/PyTorch can be used in C#</item>
    /// </list>
    /// </para>
    /// </remarks>
    private Tensor<T> ForecastOnnx(Tensor<T> input)
    {
        if (_onnxSession is null)
            throw new InvalidOperationException("ONNX session is not initialized.");

        // Convert to ONNX format
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

    #region Channel Processing

    /// <summary>
    /// Processes each feature channel independently through the PatchTST network.
    /// </summary>
    /// <param name="input">Input tensor with shape [batch_size, sequence_length, num_features].</param>
    /// <returns>Output tensor with shape [batch_size, prediction_horizon, num_features].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Channel-independent processing is a key innovation in PatchTST that
    /// often improves forecasting accuracy. Instead of processing all features together
    /// (which can cause the model to memorize spurious correlations), each feature is processed
    /// separately using the same model weights.
    /// </para>
    /// <para>
    /// For example, if you have 7 features (price, volume, etc.):
    /// <list type="number">
    /// <item>Extract the price time series → process through PatchTST → get price forecast</item>
    /// <item>Extract the volume time series → process through PatchTST → get volume forecast</item>
    /// <item>... repeat for all 7 features</item>
    /// <item>Combine all forecasts into the final output</item>
    /// </list>
    /// </para>
    /// <para>
    /// This approach helps the model learn patterns that generalize across all features,
    /// rather than overfitting to feature-specific quirks.
    /// </para>
    /// </remarks>
    private Tensor<T> ProcessChannelIndependent(Tensor<T> input)
    {
        int batchSize = input.Rank == 3 ? input.Shape[0] : 1;
        int numChannels = _numFeatures;

        var outputShape = new[] { batchSize, _predictionHorizon, numChannels };
        var outputData = new T[batchSize * _predictionHorizon * numChannels];

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < numChannels; c++)
            {
                var channelSeq = ExtractChannel(input, b, c);
                var channelOutput = ProcessSingleChannel(channelSeq);

                for (int h = 0; h < _predictionHorizon; h++)
                {
                    int outIdx = (b * _predictionHorizon * numChannels) + (h * numChannels) + c;
                    outputData[outIdx] = channelOutput.Data.Span[h];
                }
            }
        }

        return new Tensor<T>(outputShape, new Vector<T>(outputData));
    }

    /// <summary>
    /// Processes a single feature channel through the PatchTST layers.
    /// </summary>
    /// <param name="channelSeq">A single univariate time series of shape [sequence_length].</param>
    /// <returns>Forecast for this channel of shape [prediction_horizon].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is where the core PatchTST magic happens for a single variable.
    /// The time series goes through several transformations:
    /// </para>
    /// <para>
    /// <list type="number">
    /// <item><b>Patching:</b> The sequence is divided into overlapping segments (patches).
    /// For example, 96 time steps with patch_size=16 and stride=8 creates 11 patches.</item>
    /// <item><b>Embedding:</b> Each patch is converted to a dense vector.</item>
    /// <item><b>Transformer Encoding:</b> Self-attention learns which patches are most
    /// relevant for predicting the future.</item>
    /// <item><b>Projection:</b> The encoded representation is converted to the forecast.</item>
    /// </list>
    /// </para>
    /// </remarks>
    private Tensor<T> ProcessSingleChannel(Tensor<T> channelSeq)
    {
        // Create patches
        var patches = CreatePatches(channelSeq);

        // Pass through layers
        var output = patches;
        foreach (var layer in Layers)
        {
            output = layer.Forward(output);
        }

        return output;
    }

    /// <summary>
    /// Processes all feature channels together through the network (channel-dependent mode).
    /// </summary>
    /// <param name="input">Input tensor with all features.</param>
    /// <returns>Output tensor with forecasts for all features.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In channel-dependent mode, all features are processed together
    /// as a single multivariate time series. This allows the model to learn relationships
    /// between features (e.g., how volume affects price), but can lead to overfitting
    /// on datasets where such relationships are spurious.
    /// </para>
    /// <para>
    /// Research has shown that for most forecasting tasks, channel-independent mode
    /// (the default) performs better. Use channel-dependent mode only if you have strong
    /// evidence that feature interactions are important for your specific use case.
    /// </para>
    /// </remarks>
    private Tensor<T> ProcessAllChannels(Tensor<T> input)
    {
        var output = input;
        foreach (var layer in Layers)
        {
            output = layer.Forward(output);
        }
        return output;
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Calculates the number of patches that will be created from the input sequence.
    /// </summary>
    /// <returns>The number of patches.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Patching is the key idea behind PatchTST. Instead of looking at
    /// each time step individually (which is slow and doesn't capture local patterns well),
    /// PatchTST groups consecutive time steps into "patches."
    /// </para>
    /// <para>
    /// The formula is: numPatches = (sequenceLength - patchSize) / stride + 1
    /// </para>
    /// <para>
    /// For example, with sequence_length=96, patch_size=16, stride=8:
    /// numPatches = (96 - 16) / 8 + 1 = 11 patches
    /// </para>
    /// <para>
    /// Each patch becomes a "token" that the transformer processes, similar to how
    /// words become tokens in language models. This is why the original paper is called
    /// "A Time Series is Worth 64 Words."
    /// </para>
    /// </remarks>
    private int CalculateNumPatches()
    {
        return (_sequenceLength - _patchSize) / _stride + 1;
    }

    /// <summary>
    /// Divides a time series sequence into overlapping patches.
    /// </summary>
    /// <param name="sequence">Input time series of shape [sequence_length].</param>
    /// <returns>Tensor of patches with shape [num_patches, patch_size].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method implements the "patching" operation that gives
    /// PatchTST its name. Think of it like a sliding window moving across the time series.
    /// </para>
    /// <para>
    /// Example with sequence [1,2,3,4,5,6,7,8], patch_size=4, stride=2:
    /// <list type="bullet">
    /// <item>Patch 0: [1,2,3,4] (positions 0-3)</item>
    /// <item>Patch 1: [3,4,5,6] (positions 2-5, overlaps with patch 0)</item>
    /// <item>Patch 2: [5,6,7,8] (positions 4-7, overlaps with patch 1)</item>
    /// </list>
    /// </para>
    /// <para>
    /// The overlap (when stride &lt; patch_size) helps the model capture patterns that
    /// might span patch boundaries. Without overlap, important patterns could be "cut in half."
    /// </para>
    /// </remarks>
    private Tensor<T> CreatePatches(Tensor<T> sequence)
    {
        int numPatches = CalculateNumPatches();
        var patchData = new T[numPatches * _patchSize];

        for (int p = 0; p < numPatches; p++)
        {
            int startIdx = p * _stride;
            for (int i = 0; i < _patchSize; i++)
            {
                patchData[p * _patchSize + i] = sequence.Data.Span[startIdx + i];
            }
        }

        return new Tensor<T>(new[] { numPatches, _patchSize }, new Vector<T>(patchData));
    }

    /// <summary>
    /// Creates sinusoidal positional encodings for the patches.
    /// </summary>
    /// <param name="numPatches">Number of patches in the sequence.</param>
    /// <param name="modelDim">Dimension of the model embeddings.</param>
    /// <returns>Positional encoding tensor of shape [num_patches, model_dim].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Transformers don't inherently understand the order of their inputs.
    /// Positional encoding solves this by adding unique patterns to each position that tell
    /// the model "this is the 1st patch," "this is the 2nd patch," etc.
    /// </para>
    /// <para>
    /// PatchTST uses sinusoidal positional encoding (from the original "Attention Is All You Need"
    /// paper). Each position gets a unique pattern of sine and cosine waves at different frequencies:
    /// <list type="bullet">
    /// <item>Even dimensions use sin(position / 10000^(2i/d))</item>
    /// <item>Odd dimensions use cos(position / 10000^(2i/d))</item>
    /// </list>
    /// </para>
    /// <para>
    /// These patterns allow the model to:
    /// <list type="bullet">
    /// <item>Distinguish between patches at different positions</item>
    /// <item>Learn relative positions (patch 5 vs patch 3 has the same pattern as patch 7 vs patch 5)</item>
    /// <item>Generalize to longer sequences than seen during training</item>
    /// </list>
    /// </para>
    /// </remarks>
    private Tensor<T> CreatePositionalEncoding(int numPatches, int modelDim)
    {
        var pe = new T[numPatches * modelDim];

        for (int pos = 0; pos < numPatches; pos++)
        {
            for (int i = 0; i < modelDim; i++)
            {
                double angle = pos / Math.Pow(10000, (2.0 * (i / 2)) / modelDim);
                double value = (i % 2 == 0) ? Math.Sin(angle) : Math.Cos(angle);
                pe[pos * modelDim + i] = NumOps.FromDouble(value);
            }
        }

        return new Tensor<T>(new[] { numPatches, modelDim }, new Vector<T>(pe));
    }

    /// <summary>
    /// Extracts a single feature channel from the input tensor.
    /// </summary>
    /// <param name="input">Input tensor with all features.</param>
    /// <param name="batchIdx">Index of the batch sample to extract from.</param>
    /// <param name="channelIdx">Index of the feature channel to extract.</param>
    /// <returns>Univariate time series for the specified channel.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In multivariate time series, you have multiple variables tracked
    /// over time (e.g., price, volume, moving averages). This method extracts just one of those
    /// variables as a simple 1D time series.
    /// </para>
    /// <para>
    /// For example, if your input has shape [batch=32, time=96, features=7]:
    /// <list type="bullet">
    /// <item>ExtractChannel(input, 0, 0) → price history for sample 0</item>
    /// <item>ExtractChannel(input, 0, 1) → volume history for sample 0</item>
    /// <item>ExtractChannel(input, 5, 3) → feature 3 history for sample 5</item>
    /// </list>
    /// </para>
    /// <para>
    /// This is used in channel-independent mode to process each feature separately.
    /// </para>
    /// </remarks>
    private Tensor<T> ExtractChannel(Tensor<T> input, int batchIdx, int channelIdx)
    {
        var channelData = new T[_sequenceLength];

        if (input.Rank == 3)
        {
            for (int t = 0; t < _sequenceLength; t++)
            {
                int idx = (batchIdx * _sequenceLength * _numFeatures) + (t * _numFeatures) + channelIdx;
                channelData[t] = input.Data.Span[idx];
            }
        }
        else
        {
            for (int t = 0; t < _sequenceLength; t++)
            {
                channelData[t] = input.Data.Span[(t * _numFeatures) + channelIdx];
            }
        }

        return new Tensor<T>(new[] { _sequenceLength }, new Vector<T>(channelData));
    }

    /// <summary>
    /// Applies Reversible Instance Normalization (RevIN) to handle distribution shift.
    /// </summary>
    /// <param name="input">Input tensor to normalize or denormalize.</param>
    /// <param name="normalize">True to normalize (before model), false to denormalize (after model).</param>
    /// <returns>Normalized or denormalized tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> RevIN is a technique that helps models handle "distribution shift" -
    /// when the statistical properties of data change over time. This is very common in financial data
    /// where prices, volatility, and other metrics can vary dramatically.
    /// </para>
    /// <para>
    /// RevIN works in two steps:
    /// <list type="number">
    /// <item><b>Normalize (before model):</b> For each input sequence, calculate its mean and
    /// standard deviation, then transform the data to have mean=0 and std=1. This makes all
    /// inputs "look similar" to the model regardless of their original scale.</item>
    /// <item><b>Denormalize (after model):</b> Apply the reverse transformation to the model's
    /// output, scaling it back to the original data's scale. This ensures predictions are in
    /// meaningful units (e.g., actual dollar amounts, not normalized values).</item>
    /// </list>
    /// </para>
    /// <para>
    /// Without RevIN, a model trained on data with prices around $100 might fail completely
    /// when prices are around $1000, even though the patterns are the same.
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
    /// <b>For Beginners:</b> This calculates the average value of each feature in your time series.
    /// For example, if you have 7 features tracked over 96 time steps, this returns 7 values -
    /// one average for each feature.
    /// </para>
    /// <para>
    /// This is used by RevIN to center the data (subtract the mean so values are centered around 0).
    /// Centering helps neural networks learn more effectively because the data is in a consistent range.
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
    /// <b>For Beginners:</b> Standard deviation measures how "spread out" values are from the mean.
    /// A high standard deviation means values vary a lot; a low standard deviation means values
    /// are clustered near the average.
    /// </para>
    /// <para>
    /// This is used by RevIN to scale the data (divide by std so values have a spread of roughly 1).
    /// Combined with mean subtraction, this makes all input sequences "look similar" to the model:
    /// centered at 0 with values mostly between -2 and +2.
    /// </para>
    /// <para>
    /// The formula is: std = sqrt(sum((x - mean)²) / n)
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
    /// <b>For Beginners:</b> This method is used for autoregressive forecasting - predicting
    /// further into the future than the model's native horizon by using its own predictions
    /// as new input.
    /// </para>
    /// <para>
    /// For example, if you want to predict 100 steps but the model only predicts 24:
    /// <list type="number">
    /// <item>Start with historical data [t-96 to t-1], predict [t to t+23]</item>
    /// <item>Shift: new input is [t-72 to t+23] (dropped oldest 24, added predictions)</item>
    /// <item>Predict [t+24 to t+47] using the shifted input</item>
    /// <item>Repeat until you have 100 predictions</item>
    /// </list>
    /// </para>
    /// <para>
    /// This is called "autoregressive" because the model's outputs become its future inputs.
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
    /// <b>For Beginners:</b> When doing autoregressive forecasting, you make multiple predictions
    /// (each of length prediction_horizon) and need to combine them into one long forecast.
    /// </para>
    /// <para>
    /// For example, to predict 100 steps with a 24-step horizon:
    /// <list type="bullet">
    /// <item>Prediction 1: steps 0-23 (24 steps)</item>
    /// <item>Prediction 2: steps 24-47 (24 steps)</item>
    /// <item>Prediction 3: steps 48-71 (24 steps)</item>
    /// <item>Prediction 4: steps 72-95 (24 steps)</item>
    /// <item>Prediction 5: steps 96-99 (only 4 steps needed)</item>
    /// </list>
    /// </para>
    /// <para>
    /// This method takes all those predictions and stitches them together into one tensor
    /// containing exactly 100 steps.
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
    /// Releases resources used by the PatchTST model.
    /// </summary>
    /// <param name="disposing">True if disposing managed resources, false if called from finalizer.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When you're done using a model, it's important to release
    /// the resources it's using (like memory and file handles). This method handles that cleanup.
    /// </para>
    /// <para>
    /// In ONNX mode, this releases the ONNX inference session which may hold
    /// significant memory. In native mode, it releases the native layer resources.
    /// </para>
    /// <para>
    /// You typically don't call this method directly. Instead, use the model in a
    /// using statement or call Dispose() when you're done:
    /// <code>
    /// using var model = new PatchTST&lt;double&gt;(architecture);
    /// // Use the model...
    /// // Automatically disposed when leaving the using block
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
