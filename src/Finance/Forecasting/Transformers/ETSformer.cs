using System.IO;
using AiDotNet.Finance.Interfaces;
using AiDotNet.Models.Options;
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

namespace AiDotNet.Finance.Forecasting.Transformers;

/// <summary>
/// ETSformer: Exponential Smoothing Transformer for time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// ETSformer combines classical exponential smoothing (ETS) methods with transformer
/// attention mechanisms to create an interpretable time series forecasting model.
/// It explicitly models level, trend, and seasonal components.
/// </para>
/// <para>
/// <b>For Beginners:</b> ETSformer is like having a smart forecaster that can:
///
/// 1. **See the Level**: The overall value of the time series (like average price level)
/// 2. **See the Trend**: Whether values are going up or down over time
/// 3. **See Seasonality**: Repeating patterns (like daily or weekly cycles)
///
/// Unlike black-box models, ETSformer lets you inspect each component to understand
/// WHY it makes certain predictions. This is valuable in finance where explainability matters.
/// </para>
/// <para>
/// <b>Key Innovation:</b> The exponential smoothing attention mechanism applies learnable
/// decay factors, giving more weight to recent observations while still considering
/// historical patterns through the transformer architecture.
/// </para>
/// <para>
/// <b>Reference:</b> Woo et al., "ETSformer: Exponential Smoothing Transformers for
/// Time-series Forecasting", 2022. https://arxiv.org/abs/2202.01381
/// </para>
/// </remarks>
public class ETSformer<T> : NeuralNetworkBase<T>, IForecastingModel<T>
{
    #region Execution Mode

    /// <summary>
    /// Indicates whether the model operates in native C# mode (true) or ONNX inference mode (false).
    /// </summary>
    private readonly bool _useNativeMode;

    #endregion

    #region ONNX Mode Fields

    /// <summary>
    /// The ONNX inference session for running the model when in ONNX mode.
    /// </summary>
    private readonly InferenceSession? _onnxSession;

    /// <summary>
    /// The path to the ONNX model file.
    /// </summary>
    private readonly string? _onnxModelPath;

    #endregion

    #region Native Mode Fields

    /// <summary>
    /// Reference to the input embedding layer.
    /// </summary>
    private DenseLayer<T>? _inputEmbedding;

    /// <summary>
    /// Reference to encoder attention layers.
    /// </summary>
    private List<MultiHeadAttentionLayer<T>>? _encoderAttentionLayers;

    /// <summary>
    /// Reference to decoder attention layers.
    /// </summary>
    private List<MultiHeadAttentionLayer<T>>? _decoderAttentionLayers;

    /// <summary>
    /// Reference to the output projection layer.
    /// </summary>
    private DenseLayer<T>? _outputProjection;

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
    /// The optimizer used for training (gradient-based parameter updates).
    /// </summary>
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <summary>
    /// The loss function used for training.
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
    /// The model dimension (embedding size).
    /// </summary>
    private readonly int _modelDimension;

    /// <summary>
    /// The number of encoder layers.
    /// </summary>
    private readonly int _numEncoderLayers;

    /// <summary>
    /// The number of decoder layers.
    /// </summary>
    private readonly int _numDecoderLayers;

    /// <summary>
    /// The number of attention heads.
    /// </summary>
    private readonly int _numHeads;

    /// <summary>
    /// The dropout rate.
    /// </summary>
    private readonly double _dropout;

    /// <summary>
    /// Top-K frequencies for seasonal decomposition.
    /// </summary>
    private readonly int _topK;

    /// <summary>
    /// Whether to use instance normalization (RevIN).
    /// </summary>
    private readonly bool _useInstanceNormalization;

    #endregion

    #region IForecastingModel Properties

    /// <inheritdoc/>
    public int SequenceLength => _sequenceLength;

    /// <inheritdoc/>
    public int PredictionHorizon => _predictionHorizon;

    /// <inheritdoc/>
    public int NumFeatures => _numFeatures;

    /// <inheritdoc/>
    public int PatchSize => 1; // ETSformer doesn't use patching

    /// <inheritdoc/>
    public int Stride => 1;

    /// <inheritdoc/>
    public bool IsChannelIndependent => false; // ETSformer processes all channels together

    /// <inheritdoc/>
    public bool UseNativeMode => _useNativeMode;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates an ETSformer model in ONNX inference mode using a pre-trained model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the pre-trained ONNX model file.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer (not used in ONNX mode but stored for interface compliance).</param>
    /// <param name="lossFunction">Optional loss function (not used in ONNX mode but stored for interface compliance).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor when you have a pre-trained ETSformer model
    /// saved as an ONNX file. This is the fastest way to run inference since ONNX Runtime
    /// is highly optimized.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when onnxModelPath is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown when the ONNX file doesn't exist.</exception>
    public ETSformer(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        ETSformerOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        options ??= new ETSformerOptions<T>();

        _useNativeMode = false;
        _onnxSession = new InferenceSession(onnxModelPath);
        _onnxModelPath = onnxModelPath;

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        _sequenceLength = options.SequenceLength;
        _predictionHorizon = options.PredictionHorizon;
        _numFeatures = architecture.InputSize;
        _modelDimension = options.ModelDimension;
        _numEncoderLayers = options.NumEncoderLayers;
        _numDecoderLayers = options.NumDecoderLayers;
        _numHeads = options.NumHeads;
        _dropout = options.Dropout;
        _topK = options.K;
        _useInstanceNormalization = options.UseInstanceNormalization;
    }

    /// <summary>
    /// Creates an ETSformer model in native C# mode for training and inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer for training. Defaults to Adam.</param>
    /// <param name="lossFunction">Optional loss function. Defaults to MSE.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor when you want to train a new ETSformer model
    /// from scratch or fine-tune an existing one. Native mode supports full training
    /// with backpropagation.
    /// </para>
    /// </remarks>
    public ETSformer(
        NeuralNetworkArchitecture<T> architecture,
        ETSformerOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new ETSformerOptions<T>();

        _useNativeMode = true;
        _onnxSession = null;
        _onnxModelPath = null;

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        _sequenceLength = options.SequenceLength;
        _predictionHorizon = options.PredictionHorizon;
        _numFeatures = architecture.InputSize;
        _modelDimension = options.ModelDimension;
        _numEncoderLayers = options.NumEncoderLayers;
        _numDecoderLayers = options.NumDecoderLayers;
        _numHeads = options.NumHeads;
        _dropout = options.Dropout;
        _topK = options.K;
        _useInstanceNormalization = options.UseInstanceNormalization;

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes the neural network layers for the ETSformer architecture.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method builds the actual network structure.
    /// It either uses custom layers provided in the architecture or creates
    /// default ETSformer layers using the LayerHelper.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultETSformerLayers(
                Architecture,
                sequenceLength: _sequenceLength,
                predictionHorizon: _predictionHorizon,
                numFeatures: _numFeatures,
                modelDimension: _modelDimension,
                feedForwardDim: _modelDimension * 2,
                numEncoderLayers: _numEncoderLayers,
                numDecoderLayers: _numDecoderLayers,
                numHeads: _numHeads,
                dropout: _dropout));

            ExtractLayerReferences();
        }
    }

    /// <summary>
    /// Extracts references to specific layers for direct access during forward/backward passes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> While all layers are stored in a list, having direct
    /// references to key layers makes it easier to access them during the
    /// forward pass and for debugging purposes.
    /// </para>
    /// </remarks>
    private void ExtractLayerReferences()
    {
        _inputEmbedding = Layers.OfType<DenseLayer<T>>().FirstOrDefault();
        _encoderAttentionLayers = Layers.OfType<MultiHeadAttentionLayer<T>>()
            .Take(_numEncoderLayers).ToList();
        _decoderAttentionLayers = Layers.OfType<MultiHeadAttentionLayer<T>>()
            .Skip(_numEncoderLayers).Take(_numDecoderLayers).ToList();
        _outputProjection = Layers.OfType<DenseLayer<T>>().LastOrDefault();
    }

    /// <summary>
    /// Validates that custom layers meet ETSformer requirements.
    /// </summary>
    /// <param name="layers">The layers to validate.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When users provide their own layer configuration,
    /// this ensures they've included the essential layers that ETSformer needs
    /// to function correctly.
    /// </para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);

        if (!layers.Any(l => l is MultiHeadAttentionLayer<T>))
        {
            throw new ArgumentException("ETSformer requires at least one MultiHeadAttentionLayer.");
        }
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <summary>
    /// Gets whether this model supports training (native mode only).
    /// </summary>
    public override bool SupportsTraining => _useNativeMode;

    /// <summary>
    /// Makes predictions for the given input tensor.
    /// </summary>
    /// <param name="input">Input tensor of shape [batch, sequence_length, features].</param>
    /// <returns>Predicted output tensor of shape [batch, prediction_horizon, features].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the ETSformer model, Predict produces predictions from input data. This is the main inference step of the ETSformer architecture.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return _useNativeMode ? ForecastNative(input) : ForecastOnnx(input);
    }

    /// <summary>
    /// Trains the model on a batch of data.
    /// </summary>
    /// <param name="input">Input tensor of shape [batch, sequence_length, features].</param>
    /// <param name="target">Target tensor of shape [batch, prediction_horizon, features].</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Training updates the model's parameters to make better predictions.
    /// You provide input data and the correct outputs (targets), and the model learns
    /// to minimize the difference between its predictions and the targets.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is only supported in native mode.");

        SetTrainingMode(true);

        // Forward pass
        var predictions = Forward(input);

        // Compute loss - convert to vectors for loss function
        LastLoss = _lossFunction.CalculateLoss(predictions.ToVector(), target.ToVector());

        // Backward pass - convert gradient back to tensor
        var gradient = _lossFunction.CalculateDerivative(predictions.ToVector(), target.ToVector());
        Backward(Tensor<T>.FromVector(gradient));

        // Update weights via optimizer
        _optimizer.UpdateParameters(Layers);

        SetTrainingMode(false);
    }

    /// <summary>
    /// Updates the model parameters using the provided gradient vector.
    /// </summary>
    /// <param name="gradients">The gradient vector for parameter updates.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Parameters are updated through the optimizer in the Train method,
    /// so this method is intentionally empty for this model.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> gradients)
    {
        // Parameters are updated through the optimizer in Train method
    }

    /// <summary>
    /// Gets metadata describing the model.
    /// </summary>
    /// <returns>ModelMetadata object containing model information.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Metadata provides useful information about the model
    /// configuration, which is helpful for logging, debugging, or documentation.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "ETSformer" },
                { "SequenceLength", _sequenceLength },
                { "PredictionHorizon", _predictionHorizon },
                { "NumFeatures", _numFeatures },
                { "ModelDimension", _modelDimension },
                { "NumEncoderLayers", _numEncoderLayers },
                { "NumDecoderLayers", _numDecoderLayers },
                { "NumHeads", _numHeads },
                { "TopK", _topK },
                { "UseNativeMode", _useNativeMode },
                { "LayerCount", Layers.Count }
            }
        };
    }

    /// <summary>
    /// Creates a new instance of this model with the same configuration.
    /// </summary>
    /// <returns>A new ETSformer instance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the ETSformer model, CreateNewInstance builds and wires up model components. This sets up the ETSformer architecture before use.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var options = new ETSformerOptions<T>
        {
            SequenceLength = _sequenceLength,
            PredictionHorizon = _predictionHorizon,
            NumFeatures = _numFeatures,
            ModelDimension = _modelDimension,
            NumEncoderLayers = _numEncoderLayers,
            NumDecoderLayers = _numDecoderLayers,
            NumHeads = _numHeads,
            Dropout = _dropout,
            K = _topK,
            UseInstanceNormalization = _useInstanceNormalization
        };

        return new ETSformer<T>(Architecture, options);
    }

    /// <summary>
    /// Serializes model-specific data for saving.
    /// </summary>
    /// <param name="writer">Binary writer for output.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the ETSformer model, SerializeNetworkSpecificData saves or restores model-specific settings. This lets the ETSformer architecture be reused later.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_sequenceLength);
        writer.Write(_predictionHorizon);
        writer.Write(_numFeatures);
        writer.Write(_modelDimension);
        writer.Write(_numEncoderLayers);
        writer.Write(_numDecoderLayers);
        writer.Write(_numHeads);
        writer.Write(_dropout);
        writer.Write(_topK);
        writer.Write(_useInstanceNormalization);
    }

    /// <summary>
    /// Deserializes model-specific data when loading.
    /// </summary>
    /// <param name="reader">Binary reader for input.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the ETSformer model, DeserializeNetworkSpecificData saves or restores model-specific settings. This lets the ETSformer architecture be reused later.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32(); // sequenceLength
        _ = reader.ReadInt32(); // predictionHorizon
        _ = reader.ReadInt32(); // numFeatures
        _ = reader.ReadInt32(); // modelDimension
        _ = reader.ReadInt32(); // numEncoderLayers
        _ = reader.ReadInt32(); // numDecoderLayers
        _ = reader.ReadInt32(); // numHeads
        _ = reader.ReadDouble(); // dropout
        _ = reader.ReadInt32(); // topK
        _ = reader.ReadBoolean(); // useInstanceNormalization
    }

    #endregion

    #region IForecastingModel Implementation

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the ETSformer model, Forecast produces predictions from input data. This is the main inference step of the ETSformer architecture.
    /// </para>
    /// </remarks>
    public Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
    {
        if (_useInstanceNormalization)
            historicalData = ApplyInstanceNormalization(historicalData);

        var forecast = _useNativeMode ? ForecastNative(historicalData) : ForecastOnnx(historicalData);

        if (_useInstanceNormalization)
            forecast = ReverseInstanceNormalization(forecast);

        return forecast;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the ETSformer model, AutoregressiveForecast produces predictions from input data. This is the main inference step of the ETSformer architecture.
    /// </para>
    /// </remarks>
    public Tensor<T> AutoregressiveForecast(Tensor<T> input, int steps)
    {
        var predictions = new List<Tensor<T>>();
        var currentInput = input;

        int stepsRemaining = steps;
        while (stepsRemaining > 0)
        {
            var prediction = Forecast(currentInput, null);
            predictions.Add(prediction);

            int stepsUsed = Math.Min(_predictionHorizon, stepsRemaining);
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
    /// <b>For Beginners:</b> In the ETSformer model, Evaluate performs a supporting step in the workflow. It keeps the ETSformer architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public Dictionary<string, T> Evaluate(Tensor<T> predictions, Tensor<T> actuals)
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
    /// <b>For Beginners:</b> In the ETSformer model, ApplyInstanceNormalization performs a supporting step in the workflow. It keeps the ETSformer architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
    {
        int batchSize = input.Shape[0];
        int seqLen = input.Shape[1];
        int features = input.Shape[2];

        _instanceMean = new Tensor<T>(new[] { batchSize, 1, features });
        _instanceStd = new Tensor<T>(new[] { batchSize, 1, features });
        var normalized = new Tensor<T>(input.Shape);

        var epsilon = NumOps.FromDouble(1e-5);

        for (int b = 0; b < batchSize; b++)
        {
            for (int f = 0; f < features; f++)
            {
                // Calculate mean
                T sum = NumOps.Zero;
                for (int t = 0; t < seqLen; t++)
                {
                    sum = NumOps.Add(sum, input[b, t, f]);
                }
                T mean = NumOps.Divide(sum, NumOps.FromDouble(seqLen));
                _instanceMean[b, 0, f] = mean;

                // Calculate std
                T sumSq = NumOps.Zero;
                for (int t = 0; t < seqLen; t++)
                {
                    T diff = NumOps.Subtract(input[b, t, f], mean);
                    sumSq = NumOps.Add(sumSq, NumOps.Multiply(diff, diff));
                }
                T variance = NumOps.Divide(sumSq, NumOps.FromDouble(seqLen));
                T std = NumOps.Sqrt(NumOps.Add(variance, epsilon));
                _instanceStd[b, 0, f] = std;

                // Normalize
                for (int t = 0; t < seqLen; t++)
                {
                    T centered = NumOps.Subtract(input[b, t, f], mean);
                    normalized[b, t, f] = NumOps.Divide(centered, std);
                }
            }
        }

        return normalized;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the ETSformer model, GetFinancialMetrics calculates evaluation metrics. This summarizes how the ETSformer architecture is performing.
    /// </para>
    /// </remarks>
    public Dictionary<string, T> GetFinancialMetrics()
    {
        return new Dictionary<string, T>
        {
            ["SequenceLength"] = NumOps.FromDouble(_sequenceLength),
            ["PredictionHorizon"] = NumOps.FromDouble(_predictionHorizon),
            ["ModelDimension"] = NumOps.FromDouble(_modelDimension),
            ["NumHeads"] = NumOps.FromDouble(_numHeads),
            ["NumEncoderLayers"] = NumOps.FromDouble(_numEncoderLayers),
            ["NumDecoderLayers"] = NumOps.FromDouble(_numDecoderLayers),
            ["TopK"] = NumOps.FromDouble(_topK)
        };
    }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// Performs the forward pass through all layers.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Output tensor after passing through all layers.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the ETSformer model, Forward runs the forward pass through the layers. This moves data through the ETSformer architecture to compute outputs.
    /// </para>
    /// </remarks>
    private Tensor<T> Forward(Tensor<T> input)
    {
        var current = input;
        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }
        return ReshapeOutput(current);
    }

    /// <summary>
    /// Performs the backward pass for gradient computation.
    /// </summary>
    /// <param name="outputGradient">Gradient from the loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The backward pass computes gradients for all learnable
    /// parameters by propagating error signals backwards through each layer.
    /// </para>
    /// </remarks>
    private void Backward(Tensor<T> outputGradient)
    {
        var current = outputGradient;
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            current = Layers[i].Backward(current);
        }
    }

    /// <summary>
    /// Runs inference using the native C# implementation.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Forecasted values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the ETSformer model, ForecastNative produces predictions from input data. This is the main inference step of the ETSformer architecture.
    /// </para>
    /// </remarks>
    private Tensor<T> ForecastNative(Tensor<T> input)
    {
        return Forward(input);
    }

    /// <summary>
    /// Runs inference using the ONNX Runtime.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Forecasted values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the ETSformer model, ForecastOnnx produces predictions from input data. This is the main inference step of the ETSformer architecture.
    /// </para>
    /// </remarks>
    private Tensor<T> ForecastOnnx(Tensor<T> input)
    {
        if (_onnxSession is null)
            throw new InvalidOperationException("ONNX session not initialized.");

        var inputData = ConvertToFloatArray(input);
        var inputTensor = new OnnxTensors.DenseTensor<float>(inputData, input.Shape);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input", inputTensor)
        };

        using var results = _onnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        return ConvertFromOnnxTensor(outputTensor);
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Reverses instance normalization on the output.
    /// </summary>
    /// <param name="output">Normalized output tensor.</param>
    /// <returns>Denormalized tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the ETSformer model, ReverseInstanceNormalization performs a supporting step in the workflow. It keeps the ETSformer architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private Tensor<T> ReverseInstanceNormalization(Tensor<T> output)
    {
        if (_instanceMean is null || _instanceStd is null)
            return output;

        int batchSize = output.Shape[0];
        int seqLen = output.Shape[1];
        int features = output.Shape[2];

        var denormalized = new Tensor<T>(output.Shape);

        for (int b = 0; b < batchSize; b++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                for (int f = 0; f < features; f++)
                {
                    T scaled = NumOps.Multiply(output[b, t, f], _instanceStd[b, 0, f]);
                    denormalized[b, t, f] = NumOps.Add(scaled, _instanceMean[b, 0, f]);
                }
            }
        }

        return denormalized;
    }

    /// <summary>
    /// Reshapes the output to the expected dimensions.
    /// </summary>
    /// <param name="output">Output tensor to reshape.</param>
    /// <returns>Reshaped tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the ETSformer model, ReshapeOutput performs a supporting step in the workflow. It keeps the ETSformer architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private Tensor<T> ReshapeOutput(Tensor<T> output)
    {
        int batchSize = output.Shape[0];
        int horizon = _predictionHorizon;
        int features = _numFeatures;

        if (output.Shape.Length == 3 && output.Shape[1] == horizon && output.Shape[2] == features)
            return output;

        var reshaped = new Tensor<T>(new[] { batchSize, horizon, features });
        int totalElements = horizon * features;

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < totalElements && i < output.Length / batchSize; i++)
            {
                int h = i / features;
                int f = i % features;
                if (h < horizon && f < features)
                {
                    if (output.Shape.Length == 2)
                        reshaped[b, h, f] = output[b, i];
                    else
                        reshaped[b, h, f] = output[b, h, f];
                }
            }
        }

        return reshaped;
    }

    /// <summary>
    /// Shifts input tensor and appends predictions for autoregressive forecasting.
    /// </summary>
    /// <param name="input">Current input tensor.</param>
    /// <param name="prediction">Prediction tensor to append.</param>
    /// <param name="stepsUsed">Number of prediction steps to use.</param>
    /// <returns>New input tensor with shifted data.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the ETSformer model, ShiftInputWithPredictions produces predictions from input data. This is the main inference step of the ETSformer architecture.
    /// </para>
    /// </remarks>
    private Tensor<T> ShiftInputWithPredictions(Tensor<T> input, Tensor<T> prediction, int stepsUsed)
    {
        int batchSize = input.Shape[0];
        int seqLen = input.Shape[1];
        int features = input.Shape[2];

        var newInput = new Tensor<T>(input.Shape);

        for (int b = 0; b < batchSize; b++)
        {
            // Shift old data
            for (int t = 0; t < seqLen - stepsUsed; t++)
            {
                for (int f = 0; f < features; f++)
                {
                    newInput[b, t, f] = input[b, t + stepsUsed, f];
                }
            }

            // Append predictions
            for (int t = 0; t < stepsUsed; t++)
            {
                for (int f = 0; f < features; f++)
                {
                    newInput[b, seqLen - stepsUsed + t, f] = prediction[b, t, f];
                }
            }
        }

        return newInput;
    }

    /// <summary>
    /// Concatenates prediction tensors and trims to requested steps.
    /// </summary>
    /// <param name="predictions">List of prediction tensors.</param>
    /// <param name="steps">Total steps requested.</param>
    /// <returns>Concatenated prediction tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the ETSformer model, ConcatenatePredictions produces predictions from input data. This is the main inference step of the ETSformer architecture.
    /// </para>
    /// </remarks>
    private Tensor<T> ConcatenatePredictions(List<Tensor<T>> predictions, int steps)
    {
        if (predictions.Count == 0)
            return new Tensor<T>(new[] { 1, steps, _numFeatures });

        int batchSize = predictions[0].Shape[0];
        var result = new Tensor<T>(new[] { batchSize, steps, _numFeatures });

        int currentStep = 0;
        foreach (var pred in predictions)
        {
            int predSteps = Math.Min(pred.Shape[1], steps - currentStep);
            for (int b = 0; b < batchSize; b++)
            {
                for (int t = 0; t < predSteps; t++)
                {
                    for (int f = 0; f < _numFeatures; f++)
                    {
                        result[b, currentStep + t, f] = pred[b, t, f];
                    }
                }
            }
            currentStep += predSteps;
            if (currentStep >= steps)
                break;
        }

        return result;
    }

    /// <summary>
    /// Converts our tensor to a float array for ONNX.
    /// </summary>
    /// <param name="tensor">Input tensor.</param>
    /// <returns>Float array.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the ETSformer model, ConvertToFloatArray performs a supporting step in the workflow. It keeps the ETSformer architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private float[] ConvertToFloatArray(Tensor<T> tensor)
    {
        var result = new float[tensor.Length];
        for (int i = 0; i < tensor.Length; i++)
        {
            result[i] = (float)NumOps.ToDouble(tensor.Data.Span[i]);
        }
        return result;
    }

    /// <summary>
    /// Converts an ONNX tensor back to our tensor type.
    /// </summary>
    /// <param name="onnxTensor">ONNX tensor.</param>
    /// <returns>Converted tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the ETSformer model, ConvertFromOnnxTensor performs a supporting step in the workflow. It keeps the ETSformer architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private Tensor<T> ConvertFromOnnxTensor(OnnxTensors.Tensor<float> onnxTensor)
    {
        var shape = onnxTensor.Dimensions.ToArray();
        var result = new Tensor<T>(shape);
        var span = onnxTensor.ToArray();

        for (int i = 0; i < span.Length; i++)
        {
            result.Data.Span[i] = NumOps.FromDouble(span[i]);
        }

        return result;
    }

    #endregion

    #region IDisposable

    /// <summary>
    /// Disposes of managed resources.
    /// </summary>
    /// <param name="disposing">Whether this is a managed dispose.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the ETSformer model, Dispose performs a supporting step in the workflow. It keeps the ETSformer architecture pipeline consistent.
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
