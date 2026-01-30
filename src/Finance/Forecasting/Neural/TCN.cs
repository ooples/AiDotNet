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
namespace AiDotNet.Finance.Forecasting.Neural;

/// <summary>
/// TCN (Temporal Convolutional Network) model for time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// TCN uses dilated causal convolutions to model temporal sequences. It offers several advantages
/// over recurrent networks:
/// - Parallel computation (faster training)
/// - Flexible receptive field through dilation
/// - No vanishing gradient problem
/// - Better handling of long sequences
/// </para>
/// <para>
/// <b>For Beginners:</b> TCN is a modern alternative to LSTM/GRU for sequence modeling.
///
/// <b>Key Innovation - Dilated Convolutions:</b>
/// Imagine reading a book. Regular convolution reads 3 consecutive words at a time.
/// Dilated convolution can skip words to cover more context:
/// - Dilation 1: Reads words 1, 2, 3
/// - Dilation 2: Reads words 1, 3, 5 (skipping every other word)
/// - Dilation 4: Reads words 1, 5, 9 (skipping 3 words)
///
/// By stacking layers with increasing dilation, TCN can "see" very far into the past
/// efficiently. With 8 layers and kernel size 3, it can consider over 1000 past time steps!
///
/// <b>Causal Convolutions:</b>
/// TCN only looks at past and present values, never future ones. This makes it suitable
/// for real-time prediction where you can't peek ahead.
///
/// <b>Residual Connections:</b>
/// Each block adds its input to its output: output = block(input) + input
/// This helps gradients flow during training and allows the network to learn
/// when to ignore certain blocks.
/// </para>
/// <para>
/// <b>Reference:</b> Bai et al., "An Empirical Evaluation of Generic Convolutional and
/// Recurrent Networks for Sequence Modeling", 2018. https://arxiv.org/abs/1803.01271
/// </para>
/// </remarks>
public class TCN<T> : ForecastingModelBase<T>
{
    #region Execution Mode

    /// <summary>
    /// Indicates whether this network uses native layers (true) or ONNX model (false).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Native mode means the model is built with layers that can
    /// be trained. ONNX mode means we're using a pre-trained model file.
    /// </para>
    /// </remarks>
    private bool _useNativeMode;

    #endregion


    #region Native Mode Fields

    /// <summary>
    /// Input projection layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This layer transforms the raw input into a format
    /// suitable for the TCN blocks (projects to the channel dimension).
    /// </para>
    /// </remarks>
    private ILayer<T>? _inputProjection;

    /// <summary>
    /// TCN blocks organized by layer. Each block contains two convolutions and dropouts.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Each block processes the data with a specific dilation factor.
    /// Block 0 has dilation 1, block 1 has dilation 2, block 2 has dilation 4, and so on.
    /// </para>
    /// </remarks>
    private readonly List<List<ILayer<T>>> _tcnBlocks = [];

    /// <summary>
    /// Output projection layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This layer transforms the TCN output into the forecast.
    /// </para>
    /// </remarks>
    private ILayer<T>? _outputProjection;

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
    /// The lookback window size.
    /// </summary>
    private int _lookbackWindow;

    /// <summary>
    /// The forecast horizon.
    /// </summary>
    private int _forecastHorizon;

    /// <summary>
    /// Number of input features.
    /// </summary>
    private int _numFeatures;

    /// <summary>
    /// Number of channels in each layer.
    /// </summary>
    private int _numChannels;

    /// <summary>
    /// Kernel size for convolutions.
    /// </summary>
    private int _kernelSize;

    /// <summary>
    /// Number of TCN layers.
    /// </summary>
    private int _numLayers;

    /// <summary>
    /// Dropout rate for regularization.
    /// </summary>
    private double _dropout;

    /// <summary>
    /// Whether to use residual connections.
    /// </summary>
    private bool _useResidualConnections;

    #endregion

    #region IForecastingModel Properties

    /// <inheritdoc/>
    public override int SequenceLength => _lookbackWindow;

    /// <inheritdoc/>
    public override int PredictionHorizon => _forecastHorizon;

    /// <inheritdoc/>
    public override int NumFeatures => _numFeatures;

    /// <inheritdoc/>
    public override int PatchSize => _kernelSize;

    /// <inheritdoc/>
    public override int Stride => 1;

    /// <inheritdoc/>
    public override bool IsChannelIndependent => false;

    /// <inheritdoc/>
    public override bool UseNativeMode => _useNativeMode;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a TCN using pretrained ONNX model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer for fine-tuning.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor when you have a pre-trained TCN model
    /// in ONNX format from frameworks like PyTorch or TensorFlow.
    /// </para>
    /// </remarks>
    public TCN(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        TCNOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        options ??= new TCNOptions<T>();
        ValidateOptions(options);

        _useNativeMode = false;
        OnnxSession = new InferenceSession(onnxModelPath);
        OnnxModelPath = onnxModelPath;

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        _lookbackWindow = options.LookbackWindow;
        _forecastHorizon = options.ForecastHorizon;
        _numFeatures = architecture.InputSize > 0 ? architecture.InputSize : 1;
        _numChannels = options.NumChannels;
        _kernelSize = options.KernelSize;
        _numLayers = options.NumLayers;
        _dropout = options.DropoutRate;
        _useResidualConnections = options.UseResidualConnections;
    }

    /// <summary>
    /// Creates a TCN in native mode for training from scratch.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor to train a new TCN model from scratch.
    /// TCN is particularly good for:
    /// - Long sequences where RNNs struggle
    /// - When you need fast training (parallelization)
    /// - Audio processing, financial time series, weather forecasting
    /// </para>
    /// </remarks>
    public TCN(
        NeuralNetworkArchitecture<T> architecture,
        TCNOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new TCNOptions<T>();
        ValidateOptions(options);

        _useNativeMode = true;
        OnnxSession = null;
        OnnxModelPath = null;

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        _lookbackWindow = options.LookbackWindow;
        _forecastHorizon = options.ForecastHorizon;
        _numFeatures = architecture.InputSize > 0 ? architecture.InputSize : 1;
        _numChannels = options.NumChannels;
        _kernelSize = options.KernelSize;
        _numLayers = options.NumLayers;
        _dropout = options.DropoutRate;
        _useResidualConnections = options.UseResidualConnections;

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes the neural network layers for TCN.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method sets up the building blocks of TCN:
    /// 1. Input projection to channel dimension
    /// 2. Multiple TCN blocks with increasing dilation
    /// 3. Output projection to forecast horizon
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultTCNLayers(
                Architecture, _lookbackWindow, _forecastHorizon, _numFeatures,
                _numChannels, _kernelSize, _numLayers, _dropout, _useResidualConnections));

            ExtractLayerReferences();
        }
    }

    /// <summary>
    /// Extracts references to specific layers from the layer collection.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> After creating all the layers, we organize them into blocks
    /// so we can apply residual connections properly during the forward pass.
    /// </para>
    /// </remarks>
    private void ExtractLayerReferences()
    {
        int idx = 0;

        // Input projection
        if (idx < Layers.Count)
            _inputProjection = Layers[idx++];

        // TCN blocks
        _tcnBlocks.Clear();
        int layersPerBlock = _dropout > 0 ? 4 : 2; // 2 convolutions + optionally 2 dropouts

        for (int layer = 0; layer < _numLayers; layer++)
        {
            var block = new List<ILayer<T>>();
            for (int i = 0; i < layersPerBlock && idx < Layers.Count - 1; i++)
            {
                block.Add(Layers[idx++]);
            }
            if (block.Count > 0)
            {
                _tcnBlocks.Add(block);
            }
        }

        // Output projection
        if (idx < Layers.Count)
            _outputProjection = Layers[idx];
    }

    /// <summary>
    /// Validates that custom layers meet TCN architectural requirements.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> If you provide your own layers, this checks they can work together.
    /// </para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);
        if (layers.Count < 3)
        {
            throw new ArgumentException(
                "TCN requires at least 3 layers: input projection, TCN block, and output projection.",
                nameof(layers));
        }
    }

    /// <summary>
    /// Validates the TCN options.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This ensures all configuration values make sense before building.
    /// </para>
    /// </remarks>
    private static void ValidateOptions(TCNOptions<T> options)
    {
        var errors = new List<string>();

        if (options.LookbackWindow < 1)
            errors.Add("LookbackWindow must be at least 1.");
        if (options.ForecastHorizon < 1)
            errors.Add("ForecastHorizon must be at least 1.");
        if (options.NumChannels < 1)
            errors.Add("NumChannels must be at least 1.");
        if (options.KernelSize < 2)
            errors.Add("KernelSize must be at least 2.");
        if (options.NumLayers < 1)
            errors.Add("NumLayers must be at least 1.");
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
    /// <b>For Beginners:</b> In the TCN model, Predict produces predictions from input data. This is the main inference step of the TCN architecture.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return _useNativeMode ? ForecastNative(input) : ForecastOnnx(input);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TCN model, Train performs a training step. This updates the TCN architecture so it learns from data.
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
    /// <b>For Beginners:</b> In the TCN model, UpdateParameters updates internal parameters or state. This keeps the TCN architecture aligned with the latest values.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> gradients)
    {
        // Parameters are updated by the optimizer in Train method
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TCN model, GetModelMetadata performs a supporting step in the workflow. It keeps the TCN architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        // Calculate receptive field: 1 + 2*(k-1)*(2^n - 1)
        int receptiveField = 1 + 2 * (_kernelSize - 1) * ((1 << _numLayers) - 1);

        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "TCN" },
                { "LookbackWindow", _lookbackWindow },
                { "ForecastHorizon", _forecastHorizon },
                { "NumChannels", _numChannels },
                { "KernelSize", _kernelSize },
                { "NumLayers", _numLayers },
                { "ReceptiveField", receptiveField },
                { "UseResidualConnections", _useResidualConnections },
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
    /// <b>For Beginners:</b> Creates a fresh copy with randomly initialized weights.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var options = new TCNOptions<T>
        {
            LookbackWindow = _lookbackWindow,
            ForecastHorizon = _forecastHorizon,
            NumChannels = _numChannels,
            KernelSize = _kernelSize,
            NumLayers = _numLayers,
            DropoutRate = _dropout,
            UseResidualConnections = _useResidualConnections
        };

        return new TCN<T>(Architecture, options);
    }

    /// <summary>
    /// Writes TCN-specific configuration during serialization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TCN model, SerializeNetworkSpecificData saves or restores model-specific settings. This lets the TCN architecture be reused later.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_lookbackWindow);
        writer.Write(_forecastHorizon);
        writer.Write(_numFeatures);
        writer.Write(_numChannels);
        writer.Write(_kernelSize);
        writer.Write(_numLayers);
        writer.Write(_dropout);
        writer.Write(_useResidualConnections);
    }

    /// <summary>
    /// Reads TCN-specific configuration during deserialization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TCN model, DeserializeNetworkSpecificData saves or restores model-specific settings. This lets the TCN architecture be reused later.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _lookbackWindow = reader.ReadInt32();
        _forecastHorizon = reader.ReadInt32();
        _numFeatures = reader.ReadInt32();
        _numChannels = reader.ReadInt32();
        _kernelSize = reader.ReadInt32();
        _numLayers = reader.ReadInt32();
        _dropout = reader.ReadDouble();
        _useResidualConnections = reader.ReadBoolean();
    }

    #endregion

    #region IForecastingModel Implementation

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TCN model, Forecast produces predictions from input data. This is the main inference step of the TCN architecture.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
    {
        return _useNativeMode ? ForecastNative(historicalData) : ForecastOnnx(historicalData);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TCN model, AutoregressiveForecast produces predictions from input data. This is the main inference step of the TCN architecture.
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
    /// <b>For Beginners:</b> In the TCN model, Evaluate performs a supporting step in the workflow. It keeps the TCN architecture pipeline consistent.
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
    /// <b>For Beginners:</b> In the TCN model, ApplyInstanceNormalization performs a supporting step in the workflow. It keeps the TCN architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
    {
        return input;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TCN model, GetFinancialMetrics calculates evaluation metrics. This summarizes how the TCN architecture is performing.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> GetFinancialMetrics()
    {
        T lastLoss = LastLoss is not null ? LastLoss : NumOps.Zero;
        int receptiveField = 1 + 2 * (_kernelSize - 1) * ((1 << _numLayers) - 1);

        return new Dictionary<string, T>
        {
            ["LookbackWindow"] = NumOps.FromDouble(_lookbackWindow),
            ["ForecastHorizon"] = NumOps.FromDouble(_forecastHorizon),
            ["NumChannels"] = NumOps.FromDouble(_numChannels),
            ["NumLayers"] = NumOps.FromDouble(_numLayers),
            ["ReceptiveField"] = NumOps.FromDouble(receptiveField),
            ["LastLoss"] = lastLoss
        };
    }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// Performs the forward pass through the TCN.
    /// </summary>
    /// <param name="input">Input tensor of shape [batch, lookback_window * features].</param>
    /// <returns>Output tensor of shape [batch, forecast_horizon].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The forward pass processes data through:
    ///
    /// 1. <b>Input Projection</b>: Transforms raw input to channel dimension
    /// 2. <b>TCN Blocks</b>: Each block applies dilated convolutions
    ///    - If using residual connections: output = block(input) + input
    ///    - Dilation doubles with each block: 1, 2, 4, 8, ...
    /// 3. <b>Output Projection</b>: Maps to forecast horizon
    /// </para>
    /// </remarks>
    private Tensor<T> Forward(Tensor<T> input)
    {
        // Input projection
        Tensor<T> current = input;

        if (_inputProjection is not null)
        {
            current = _inputProjection.Forward(input);
        }

        // Process through TCN blocks with residual connections
        foreach (var block in _tcnBlocks)
        {
            var blockInput = current;

            // Forward through all layers in the block
            foreach (var layer in block)
            {
                current = layer.Forward(current);
            }

            // Add residual connection if enabled
            if (_useResidualConnections)
            {
                current = AddTensors(current, blockInput);
            }
        }

        // Output projection
        if (_outputProjection is not null)
        {
            current = _outputProjection.Forward(current);
        }

        return current;
    }

    /// <summary>
    /// Performs the backward pass through the TCN.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Backpropagation for TCN with residual connections:
    /// - Gradients flow both through the block AND around it (via residual)
    /// - This helps train very deep networks
    /// </para>
    /// </remarks>
    private void Backward(Tensor<T> gradOutput)
    {
        var grad = gradOutput;

        // Backward through output projection
        if (_outputProjection is not null)
        {
            grad = _outputProjection.Backward(grad);
        }

        // Backward through TCN blocks in reverse order
        for (int i = _tcnBlocks.Count - 1; i >= 0; i--)
        {
            var block = _tcnBlocks[i];

            // For residual connections, gradient flows both through block and around it
            var blockGrad = grad;

            // Backward through block layers in reverse order
            for (int j = block.Count - 1; j >= 0; j--)
            {
                blockGrad = block[j].Backward(blockGrad);
            }

            // If using residual connections, add the direct gradient path
            if (_useResidualConnections)
            {
                grad = AddTensors(blockGrad, grad);
            }
            else
            {
                grad = blockGrad;
            }
        }

        // Backward through input projection
        if (_inputProjection is not null)
        {
            _ = _inputProjection.Backward(grad);
        }
    }

    /// <summary>
    /// Performs native mode forecasting.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TCN model, ForecastNative produces predictions from input data. This is the main inference step of the TCN architecture.
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
    /// <b>For Beginners:</b> In the TCN model, ForecastOnnx produces predictions from input data. This is the main inference step of the TCN architecture.
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
    /// Element-wise addition of two tensors.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Adds corresponding elements from two tensors.
    /// Used for residual connections: output = layer_output + input.
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

        // If tensors have different lengths, copy remaining elements from the longer one
        if (a.Length > length)
        {
            for (int i = length; i < a.Length; i++)
            {
                result.Data.Span[i] = a.Data.Span[i];
            }
        }

        return result;
    }

    /// <summary>
    /// Shifts the input tensor by incorporating new predictions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TCN model, ShiftInputWithPredictions produces predictions from input data. This is the main inference step of the TCN architecture.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ShiftInputWithPredictions(Tensor<T> input, Tensor<T> predictions, int stepsUsed)
    {
        int batchSize = input.Shape[0];
        int inputLen = input.Length / batchSize;

        var shifted = new Tensor<T>(input.Shape);

        for (int b = 0; b < batchSize; b++)
        {
            // Shift old values left
            for (int t = 0; t < inputLen - stepsUsed; t++)
            {
                int srcIdx = b * inputLen + t + stepsUsed;
                int dstIdx = b * inputLen + t;

                if (srcIdx < input.Length && dstIdx < shifted.Length)
                {
                    shifted.Data.Span[dstIdx] = input.Data.Span[srcIdx];
                }
            }

            // Add predictions at the end
            for (int t = 0; t < stepsUsed; t++)
            {
                int srcIdx = b * _forecastHorizon + t;
                int dstIdx = b * inputLen + (inputLen - stepsUsed + t);

                if (srcIdx < predictions.Length && dstIdx < shifted.Length)
                {
                    shifted.Data.Span[dstIdx] = predictions.Data.Span[srcIdx];
                }
            }
        }

        return shifted;
    }

    /// <summary>
    /// Concatenates multiple prediction tensors into a single result.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TCN model, ConcatenatePredictions produces predictions from input data. This is the main inference step of the TCN architecture.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ConcatenatePredictions(List<Tensor<T>> predictions, int totalSteps)
    {
        if (predictions.Count == 0)
            return new Tensor<T>(new[] { 1, totalSteps });

        int batchSize = predictions[0].Shape[0];
        var result = new Tensor<T>(new[] { batchSize, totalSteps });

        int currentStep = 0;
        foreach (var pred in predictions)
        {
            int predLen = pred.Length / batchSize;
            int stepsToCopy = Math.Min(predLen, totalSteps - currentStep);

            for (int b = 0; b < batchSize; b++)
            {
                for (int t = 0; t < stepsToCopy; t++)
                {
                    int srcIdx = b * predLen + t;
                    int dstIdx = b * totalSteps + currentStep + t;

                    if (srcIdx < pred.Length && dstIdx < result.Length)
                    {
                        result.Data.Span[dstIdx] = pred.Data.Span[srcIdx];
                    }
                }
            }

            currentStep += stepsToCopy;
            if (currentStep >= totalSteps) break;
        }

        return result;
    }

    #endregion

    #region IDisposable

    /// <summary>
    /// Releases managed resources used by this model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TCN model, Dispose performs a supporting step in the workflow. It keeps the TCN architecture pipeline consistent.
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

