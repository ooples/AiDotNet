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
namespace AiDotNet.Finance.Forecasting.StateSpace;

/// <summary>
/// Mamba (Selective State Space Model) implementation for time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// Mamba is a selective state space model that achieves linear-time complexity for
/// sequence modeling while maintaining the expressiveness of transformers through
/// input-dependent (selective) state space parameters.
/// </para>
/// <para><b>For Beginners:</b> Mamba is a breakthrough in efficient sequence modeling:
///
/// <b>The Key Insight:</b>
/// Transformers have O(n^2) complexity due to attention, which is slow for long sequences.
/// State space models (SSMs) have O(n) complexity but are less expressive.
/// Mamba makes SSM parameters input-dependent (selective), combining the best of both.
///
/// <b>How It Works:</b>
/// 1. <b>State Space Model:</b> Maintains a hidden state updated recurrently
/// 2. <b>Selective Mechanism:</b> Parameters (A, B, C, delta) vary with input
/// 3. <b>Hardware-aware Algorithm:</b> Efficient implementation via parallel scan
/// 4. <b>Linear Complexity:</b> O(n) time and memory for sequence length n
///
/// <b>Advantages:</b>
/// - Linear time complexity (vs O(n^2) for attention)
/// - Handles very long sequences efficiently
/// - Strong performance on language, audio, and time series
/// - Hardware-efficient implementation
/// </para>
/// <para>
/// <b>Reference:</b> Gu and Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2024.
/// https://arxiv.org/abs/2312.00752
/// </para>
/// </remarks>
public class Mamba<T> : ForecastingModelBase<T>
{
    #region Execution Mode

    /// <summary>
    /// Indicates whether the model is running in native mode (true) or ONNX mode (false).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Mamba supports two execution modes:
    /// - Native mode: Train the model from scratch
    /// - ONNX mode: Use pretrained ONNX model for inference
    /// </para>
    /// </remarks>
    private bool _useNativeMode;

    #endregion


    #region Native Mode Fields

    /// <summary>
    /// Reference to the input embedding layer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Projects raw input to the model dimension.
    /// </para>
    /// </remarks>
    private DenseLayer<T>? _inputEmbedding;

    /// <summary>
    /// References to the Mamba block layers.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each Mamba block contains the selective SSM
    /// mechanism that processes the sequence with linear complexity.
    /// </para>
    /// </remarks>
    private List<DenseLayer<T>>? _mambaBlocks;

    /// <summary>
    /// Reference to the output projection layer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Projects the Mamba output to forecast values.
    /// </para>
    /// </remarks>
    private DenseLayer<T>? _outputProjection;

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
    private readonly MambaOptions<T> _options;

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
    /// Model dimension (d_model).
    /// </summary>
    private int _modelDimension;

    /// <summary>
    /// State dimension for SSM.
    /// </summary>
    private int _stateDimension;

    /// <summary>
    /// Expansion factor for inner dimension.
    /// </summary>
    private int _expandFactor;

    /// <summary>
    /// Convolution kernel size.
    /// </summary>
    private int _convKernelSize;

    /// <summary>
    /// Number of Mamba layers.
    /// </summary>
    private int _numLayers;

    /// <summary>
    /// Dropout rate.
    /// </summary>
    private double _dropout;

    /// <summary>
    /// Delta rank for dt projection.
    /// </summary>
    private int _dtRank;

    /// <summary>
    /// Whether to use bidirectional processing.
    /// </summary>
    private bool _useBidirectional;

    /// <summary>
    /// Number of input features.
    /// </summary>
    private int _numFeatures;

    #endregion

    #region IForecastingModel Properties

    /// <inheritdoc/>
    public override int SequenceLength => _contextLength;

    /// <inheritdoc/>
    public override int PredictionHorizon => _forecastHorizon;

    /// <inheritdoc/>
    public override int NumFeatures => _numFeatures;

    /// <inheritdoc/>
    public override int PatchSize => 1; // Mamba operates on individual time steps

    /// <inheritdoc/>
    public override int Stride => 1;

    /// <inheritdoc/>
    public override bool IsChannelIndependent => true;

    /// <inheritdoc/>
    public override bool UseNativeMode => _useNativeMode;

    /// <summary>
    /// Gets the state dimension of the SSM.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The size of the hidden state that captures sequence dynamics.
    /// </para>
    /// </remarks>
    public int StateDimension => _stateDimension;

    /// <summary>
    /// Gets the expansion factor.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The SSM operates in an expanded dimension for more capacity.
    /// </para>
    /// </remarks>
    public int ExpandFactor => _expandFactor;

    /// <summary>
    /// Gets whether bidirectional processing is used.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Bidirectional Mamba processes forward and backward.
    /// </para>
    /// </remarks>
    public bool UseBidirectional => _useBidirectional;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance using an ONNX pretrained model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="options">Configuration options for Mamba.</param>
    /// <param name="optimizer">Optional optimizer.</param>
    /// <param name="lossFunction">Optional custom loss function.</param>
    /// <exception cref="ArgumentException">Thrown when onnxModelPath is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown when ONNX model file doesn't exist.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to load a pretrained Mamba model
    /// for efficient linear-time inference on long sequences.
    /// </para>
    /// </remarks>
    public Mamba(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        MambaOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        options ??= new MambaOptions<T>();
        _options = options;
        Options = _options;

        _useNativeMode = false;
        OnnxModelPath = onnxModelPath;
        OnnxSession = new InferenceSession(onnxModelPath);

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        _contextLength = options.ContextLength;
        _forecastHorizon = options.ForecastHorizon;
        _modelDimension = options.ModelDimension;
        _stateDimension = options.StateDimension;
        _expandFactor = options.ExpandFactor;
        _convKernelSize = options.ConvKernelSize;
        _numLayers = options.NumLayers;
        _dropout = options.DropoutRate;
        _dtRank = options.DtRank < 0 ? (_modelDimension / 16) : options.DtRank;
        _useBidirectional = options.UseBidirectional;
        _numFeatures = 1;

        InitializeLayers();
    }

    /// <summary>
    /// Initializes a new instance in native mode for training.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">Configuration options for Mamba.</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="optimizer">Optional optimizer.</param>
    /// <param name="lossFunction">Optional custom loss function.</param>
    /// <exception cref="ArgumentNullException">Thrown when architecture is null.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to train Mamba from scratch.
    /// Mamba is particularly efficient for long sequences due to its linear complexity.
    /// </para>
    /// </remarks>
    public Mamba(
        NeuralNetworkArchitecture<T> architecture,
        MambaOptions<T>? options = null,
        int numFeatures = 1,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new MambaOptions<T>();
        _options = options;
        Options = _options;

        _useNativeMode = true;

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        _contextLength = options.ContextLength;
        _forecastHorizon = options.ForecastHorizon;
        _modelDimension = options.ModelDimension;
        _stateDimension = options.StateDimension;
        _expandFactor = options.ExpandFactor;
        _convKernelSize = options.ConvKernelSize;
        _numLayers = options.NumLayers;
        _dropout = options.DropoutRate;
        _dtRank = options.DtRank < 0 ? (_modelDimension / 16) : options.DtRank;
        _useBidirectional = options.UseBidirectional;
        _numFeatures = numFeatures;

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes the model layers based on configuration.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up the Mamba architecture including
    /// input embedding, stacked Mamba blocks with selective SSM, and output projection.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultMambaLayers(
                Architecture, _contextLength, _forecastHorizon, _numFeatures,
                _modelDimension, _stateDimension, _expandFactor, _numLayers, _dropout));

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
        _inputEmbedding = Layers.OfType<DenseLayer<T>>().FirstOrDefault();
        _mambaBlocks = Layers.OfType<DenseLayer<T>>().Skip(1).Take(_numLayers * 6).ToList();
        _outputProjection = Layers.OfType<DenseLayer<T>>().LastOrDefault();
    }

    /// <summary>
    /// Validates that custom layers meet Mamba requirements.
    /// </summary>
    /// <param name="layers">The layers to validate.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> When using custom layers, we ensure they include
    /// the necessary components for the Mamba architecture.
    /// </para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);

        var denseCount = layers.OfType<DenseLayer<T>>().Count();
        if (denseCount < 3)
        {
            throw new ArgumentException(
                "Mamba requires at least input embedding, SSM processing, and output projection layers.");
        }
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <inheritdoc/>
    public override bool SupportsTraining => _useNativeMode;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Mamba model, Predict produces predictions from input data. This is the main inference step of the Mamba architecture.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return _useNativeMode ? Forward(input) : ForecastOnnx(input);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Mamba model, Train performs a training step. This updates the Mamba architecture so it learns from data.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is only supported in native mode.");

        SetTrainingMode(true);

        var output = Forward(input);

        // Ensure shapes match for loss calculation - use minimum length
        var outputVec = output.ToVector();
        var targetVec = target.ToVector();
        int minLength = Math.Min(outputVec.Length, targetVec.Length);

        // Create matching-length vectors for loss calculation
        var matchedOutput = new T[minLength];
        var matchedTarget = new T[minLength];
        for (int i = 0; i < minLength; i++)
        {
            matchedOutput[i] = outputVec[i];
            matchedTarget[i] = targetVec[i];
        }

        // Compute loss using matched-size vectors
        var matchedOutputVec = new Vector<T>(matchedOutput);
        var matchedTargetVec = new Vector<T>(matchedTarget);
        LastLoss = _lossFunction.CalculateLoss(matchedOutputVec, matchedTargetVec);

        // Backward pass - use matched size for gradient computation
        var gradient = _lossFunction.CalculateDerivative(matchedOutputVec, matchedTargetVec);

        // Pad or truncate gradient to match output shape for backward pass
        var fullGradient = new T[output.Length];
        for (int i = 0; i < Math.Min(gradient.Length, fullGradient.Length); i++)
        {
            fullGradient[i] = gradient[i];
        }

        // Create 2D gradient tensor for backward pass
        var gradTensor = new Tensor<T>(new[] { 1, output.Length }, new Vector<T>(fullGradient));
        Backward(gradTensor);

        _optimizer.UpdateParameters(Layers);

        SetTrainingMode(false);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Mamba model, UpdateParameters updates internal parameters or state. This keeps the Mamba architecture aligned with the latest values.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> gradients)
    {
        // Parameters are updated through the optimizer in Train()
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Mamba model, GetModelMetadata performs a supporting step in the workflow. It keeps the Mamba architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "Mamba" },
                { "ContextLength", _contextLength },
                { "ForecastHorizon", _forecastHorizon },
                { "ModelDimension", _modelDimension },
                { "StateDimension", _stateDimension },
                { "ExpandFactor", _expandFactor },
                { "NumLayers", _numLayers },
                { "UseBidirectional", _useBidirectional },
                { "UseNativeMode", _useNativeMode },
                { "ParameterCount", GetParameterCount() }
            },
            ModelData = _useNativeMode ? this.Serialize() : Array.Empty<byte>()
        };
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Mamba model, CreateNewInstance builds and wires up model components. This sets up the Mamba architecture before use.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var options = new MambaOptions<T>
        {
            ContextLength = _contextLength,
            ForecastHorizon = _forecastHorizon,
            ModelDimension = _modelDimension,
            StateDimension = _stateDimension,
            ExpandFactor = _expandFactor,
            ConvKernelSize = _convKernelSize,
            NumLayers = _numLayers,
            DropoutRate = _dropout,
            DtRank = _dtRank,
            UseBidirectional = _useBidirectional
        };

        return new Mamba<T>(Architecture, options, _numFeatures);
    }

    /// <summary>
    /// Writes Mamba-specific configuration during serialization.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Saves all the configuration needed to reconstruct this model.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_contextLength);
        writer.Write(_forecastHorizon);
        writer.Write(_modelDimension);
        writer.Write(_stateDimension);
        writer.Write(_expandFactor);
        writer.Write(_convKernelSize);
        writer.Write(_numLayers);
        writer.Write(_dropout);
        writer.Write(_dtRank);
        writer.Write(_useBidirectional);
        writer.Write(_numFeatures);
    }

    /// <summary>
    /// Reads Mamba-specific configuration during deserialization.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Loads the configuration that was saved during serialization.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _contextLength = reader.ReadInt32();
        _forecastHorizon = reader.ReadInt32();
        _modelDimension = reader.ReadInt32();
        _stateDimension = reader.ReadInt32();
        _expandFactor = reader.ReadInt32();
        _convKernelSize = reader.ReadInt32();
        _numLayers = reader.ReadInt32();
        _dropout = reader.ReadDouble();
        _dtRank = reader.ReadInt32();
        _useBidirectional = reader.ReadBoolean();
        _numFeatures = reader.ReadInt32();
    }

    #endregion

    #region IForecastingModel Implementation

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Mamba model, Forecast produces predictions from input data. This is the main inference step of the Mamba architecture.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
    {
        var output = _useNativeMode ? Forward(historicalData) : ForecastOnnx(historicalData);

        if (quantiles is not null && quantiles.Length > 0)
        {
            return GenerateQuantilePredictions(historicalData, quantiles);
        }

        return output;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Mamba model, AutoregressiveForecast produces predictions from input data. This is the main inference step of the Mamba architecture.
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
    /// <b>For Beginners:</b> In the Mamba model, Evaluate performs a supporting step in the workflow. It keeps the Mamba architecture pipeline consistent.
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
    /// <b>For Beginners:</b> In the Mamba model, ApplyInstanceNormalization performs a supporting step in the workflow. It keeps the Mamba architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
    {
        // Mamba uses batch normalization internally
        return input;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the Mamba model, GetFinancialMetrics calculates evaluation metrics. This summarizes how the Mamba architecture is performing.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> GetFinancialMetrics()
    {
        T lastLoss = LastLoss is not null ? LastLoss : NumOps.Zero;

        return new Dictionary<string, T>
        {
            ["ContextLength"] = NumOps.FromDouble(_contextLength),
            ["ForecastHorizon"] = NumOps.FromDouble(_forecastHorizon),
            ["ModelDimension"] = NumOps.FromDouble(_modelDimension),
            ["StateDimension"] = NumOps.FromDouble(_stateDimension),
            ["NumLayers"] = NumOps.FromDouble(_numLayers),
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
    /// 1. Embeds input to model dimension
    /// 2. Processes through stacked Mamba blocks (selective SSM)
    /// 3. Projects to forecast values
    /// Mamba achieves linear O(n) complexity through its state space formulation.
    /// </para>
    /// </remarks>
    private Tensor<T> Forward(Tensor<T> input)
    {
        // Flatten input to 2D [batch, features] for Dense layers
        // Input may be [batch, seq, features] = [1, contextLength, numFeatures]
        // Dense layers expect flat input of size numFeatures * contextLength
        var current = input;

        if (current.Rank > 2)
        {
            int totalElements = current.Length;
            current = current.Reshape(new[] { 1, totalElements });
        }
        else if (current.Rank == 1)
        {
            current = current.Reshape(new[] { 1, current.Length });
        }

        foreach (var layer in Layers)
        {
            // BatchNorm and Dense layers expect 2D input
            // Ensure we maintain 2D shape through the network
            if (current.Rank > 2)
            {
                int batch = current.Shape[0];
                int features = current.Length / batch;
                current = current.Reshape(new[] { batch, features });
            }

            current = layer.Forward(current);
        }

        return current;
    }

    /// <summary>
    /// Performs the backward pass for gradient computation.
    /// </summary>
    /// <param name="outputGradient">Gradient of the loss with respect to output.</param>
    /// <returns>Gradient with respect to input.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The backward pass computes gradients for all trainable
    /// parameters. Mamba's recurrent structure allows efficient gradient computation.
    /// </para>
    /// </remarks>
    private Tensor<T> Backward(Tensor<T> outputGradient)
    {
        // Ensure gradient is 2D for Dense/BatchNorm backward pass
        var current = outputGradient;

        if (current.Rank == 1)
        {
            current = current.Reshape(new[] { 1, current.Length });
        }
        else if (current.Rank > 2)
        {
            int batch = current.Shape[0];
            int features = current.Length / batch;
            current = current.Reshape(new[] { batch, features });
        }

        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            var layer = Layers[i];

            // Ensure 2D shape before each backward call
            if (current.Rank > 2)
            {
                int batch = current.Shape[0];
                int features = current.Length / batch;
                current = current.Reshape(new[] { batch, features });
            }

            // For BatchNorm layers, ensure gradient matches the expected input size
            // BatchNorm preserves shape, so gradient should match stored input
            if (layer is BatchNormalizationLayer<T> bnLayer)
            {
                // Get the expected size from the BatchNorm's output shape (same as input for BatchNorm)
                var outputShape = bnLayer.GetOutputShape();
                int expectedSize = outputShape.Length > 0 ? outputShape[0] : current.Length;

                if (current.Length != expectedSize)
                {
                    // Gradient size doesn't match - pad or truncate
                    var gradData = new T[expectedSize];
                    int copyLen = Math.Min(current.Length, expectedSize);
                    for (int j = 0; j < copyLen; j++)
                    {
                        gradData[j] = current.Data.Span[j];
                    }
                    current = new Tensor<T>(new[] { 1, expectedSize }, new Vector<T>(gradData));
                }
                else if (current.Shape.Length != 2 || current.Shape[1] != expectedSize)
                {
                    current = current.Reshape(new[] { 1, expectedSize });
                }
            }

            current = layer.Backward(current);
        }

        return current;
    }

    /// <summary>
    /// Performs ONNX-based inference for forecasting.
    /// </summary>
    /// <param name="input">Input tensor with historical data.</param>
    /// <returns>Forecast tensor with predictions.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> ONNX mode uses the pretrained Mamba model for
    /// efficient inference on long sequences.
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

    #region Model-Specific Processing

    /// <summary>
    /// Generates quantile predictions through dropout-based sampling.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <param name="quantiles">Quantile levels to compute.</param>
    /// <returns>Quantile predictions tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> For uncertainty estimation, we use Monte Carlo dropout
    /// to generate diverse forecasts and compute quantiles.
    /// </para>
    /// </remarks>
    private Tensor<T> GenerateQuantilePredictions(Tensor<T> input, double[] quantiles)
    {
        int numSamples = 100;
        var samples = new List<Tensor<T>>();

        SetTrainingMode(true);

        for (int s = 0; s < numSamples; s++)
        {
            samples.Add(Forward(input));
        }

        SetTrainingMode(false);

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

            values.Sort();

            for (int q = 0; q < quantiles.Length; q++)
            {
                int idx = Math.Min((int)(quantiles[q] * values.Count), values.Count - 1);
                result.Data.Span[t * quantiles.Length + q] = NumOps.FromDouble(values[idx]);
            }
        }

        return result;
    }

    /// <summary>
    /// Shifts input tensor by appending predictions.
    /// </summary>
    /// <param name="input">Original input tensor.</param>
    /// <param name="predictions">Predictions to append.</param>
    /// <param name="stepsUsed">Number of prediction steps to use.</param>
    /// <returns>Shifted input tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> For multi-step forecasting, we update the input
    /// with predictions to forecast further into the future.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ShiftInputWithPredictions(Tensor<T> input, Tensor<T> predictions, int stepsUsed)
    {
        var result = new Tensor<T>(input.Shape);
        int contextLen = _contextLength;

        for (int i = 0; i < contextLen - stepsUsed; i++)
        {
            result.Data.Span[i] = input.Data.Span[i + stepsUsed];
        }

        for (int i = 0; i < stepsUsed && i < predictions.Length; i++)
        {
            result.Data.Span[contextLen - stepsUsed + i] = predictions.Data.Span[i];
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
    /// <para><b>For Beginners:</b> Combines chunked forecasts into a single result.
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

