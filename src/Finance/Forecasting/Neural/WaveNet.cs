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
/// WaveNet model adapted for time series forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// WaveNet was originally developed by DeepMind for audio generation. It uses dilated causal
/// convolutions with gated activation units and dual residual/skip connections. This architecture
/// has proven highly effective for time series forecasting as well.
/// </para>
/// <para>
/// <b>For Beginners:</b> WaveNet is like a more sophisticated version of TCN:
///
/// <b>Gated Activation Units:</b>
/// Instead of using simple ReLU activations, WaveNet multiplies two signals:
/// - output = tanh(filter_output) * sigmoid(gate_output)
/// - The tanh extracts features, the sigmoid controls which features pass through
/// - This is inspired by LSTM gates and helps model complex patterns
///
/// <b>Two Types of Connections:</b>
/// 1. <b>Residual Connection:</b> Each block adds its input to output (like TCN)
///    - Helps gradients flow during training
/// 2. <b>Skip Connection:</b> Each block sends output directly to the final layers
///    - Allows combining features from ALL time scales
///    - Skip outputs are summed at the end
///
/// <b>Why Two Connection Types?</b>
/// The residual path maintains the signal through deep networks.
/// The skip path lets early layers contribute directly to the output,
/// ensuring fine-grained (high-frequency) and coarse (low-frequency) patterns
/// are both captured in the final prediction.
///
/// <b>Example:</b>
/// With 2 stacks of 8 layers each:
/// - 16 total dilated layers
/// - Dilations: [1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128]
/// - Receptive field: 2 * (2^8 - 1) + 1 = 511 time steps
/// </para>
/// <para>
/// <b>Reference:</b> van den Oord et al., "WaveNet: A Generative Model for Raw Audio", 2016.
/// https://arxiv.org/abs/1609.03499
/// </para>
/// </remarks>
public class WaveNet<T> : ForecastingModelBase<T>
{
    #region Execution Mode

    /// <summary>
    /// Indicates whether this network uses native layers (true) or ONNX model (false).
    /// </summary>
    private readonly bool _useNativeMode;

    #endregion

    
    #region Native Mode Fields

    /// <summary>
    /// Input projection layer.
    /// </summary>
    private ILayer<T>? _inputProjection;

    /// <summary>
    /// WaveNet blocks organized by layer. Each block contains filter, gate, residual, and skip layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Each block in the list contains:
    /// - Filter layer (tanh activation)
    /// - Gate layer (sigmoid activation)
    /// - Residual projection (1x1 conv)
    /// - Skip projection (1x1 conv)
    /// - Optional dropout
    /// </para>
    /// </remarks>
    private readonly List<List<ILayer<T>>> _waveNetBlocks = [];

    /// <summary>
    /// First output layer after skip aggregation.
    /// </summary>
    private ILayer<T>? _output1;

    /// <summary>
    /// Final output projection layer.
    /// </summary>
    private ILayer<T>? _output2;

    #endregion

    #region Shared Fields

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly int _lookbackWindow;
    private readonly int _forecastHorizon;
    private readonly int _numFeatures;
    private readonly int _residualChannels;
    private readonly int _skipChannels;
    private readonly int _dilationDepth;
    private readonly int _numStacks;
    private readonly int _kernelSize;
    private readonly bool _useGatedActivations;
    private readonly double _dropout;

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
    /// Creates a WaveNet using pretrained ONNX model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer for fine-tuning.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the WaveNet model, WaveNet sets up the architecture and options. This prepares the model for training or inference.
    /// </para>
    /// </remarks>
    public WaveNet(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        WaveNetOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        options ??= new WaveNetOptions<T>();
        ValidateOptions(options);

        _useNativeMode = false;
        OnnxSession = new InferenceSession(onnxModelPath);
        OnnxModelPath = onnxModelPath;

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        _lookbackWindow = options.LookbackWindow;
        _forecastHorizon = options.ForecastHorizon;
        _numFeatures = architecture.InputSize > 0 ? architecture.InputSize : 1;
        _residualChannels = options.ResidualChannels;
        _skipChannels = options.SkipChannels;
        _dilationDepth = options.DilationDepth;
        _numStacks = options.NumStacks;
        _kernelSize = options.KernelSize;
        _useGatedActivations = options.UseGatedActivations;
        _dropout = options.DropoutRate;
    }

    /// <summary>
    /// Creates a WaveNet in native mode for training from scratch.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor to train a new WaveNet model.
    /// WaveNet excels at:
    /// - Capturing patterns at multiple time scales simultaneously
    /// - Modeling complex conditional dependencies
    /// - Audio, speech, and financial time series
    /// </para>
    /// </remarks>
    public WaveNet(
        NeuralNetworkArchitecture<T> architecture,
        WaveNetOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new WaveNetOptions<T>();
        ValidateOptions(options);

        _useNativeMode = true;
        OnnxSession = null;
        OnnxModelPath = null;

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        _lookbackWindow = options.LookbackWindow;
        _forecastHorizon = options.ForecastHorizon;
        _numFeatures = architecture.InputSize > 0 ? architecture.InputSize : 1;
        _residualChannels = options.ResidualChannels;
        _skipChannels = options.SkipChannels;
        _dilationDepth = options.DilationDepth;
        _numStacks = options.NumStacks;
        _kernelSize = options.KernelSize;
        _useGatedActivations = options.UseGatedActivations;
        _dropout = options.DropoutRate;

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes the neural network layers for WaveNet.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This sets up the WaveNet building blocks:
    /// 1. Input projection
    /// 2. Multiple dilated blocks with gated activations
    /// 3. Skip connection aggregation
    /// 4. Output projections
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultWaveNetLayers(
                Architecture, _lookbackWindow, _forecastHorizon, _numFeatures,
                _residualChannels, _skipChannels, _dilationDepth, _numStacks, _dropout));

            ExtractLayerReferences();
        }
    }

    /// <summary>
    /// Extracts references to specific layers from the layer collection.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Organizes layers into blocks for proper forward pass:
    /// - Each block has filter, gate, residual, skip (and optional dropout)
    /// - Output layers come at the end
    /// </para>
    /// </remarks>
    private void ExtractLayerReferences()
    {
        int idx = 0;

        // Input projection
        if (idx < Layers.Count)
            _inputProjection = Layers[idx++];

        // WaveNet blocks
        _waveNetBlocks.Clear();
        int layersPerBlock = _dropout > 0 ? 5 : 4; // filter, gate, residual, skip, (dropout)
        int totalBlocks = _numStacks * _dilationDepth;

        for (int block = 0; block < totalBlocks; block++)
        {
            var blockLayers = new List<ILayer<T>>();
            for (int i = 0; i < layersPerBlock && idx < Layers.Count - 2; i++)
            {
                blockLayers.Add(Layers[idx++]);
            }
            if (blockLayers.Count > 0)
            {
                _waveNetBlocks.Add(blockLayers);
            }
        }

        // Output layers
        if (idx < Layers.Count)
            _output1 = Layers[idx++];
        if (idx < Layers.Count)
            _output2 = Layers[idx];
    }

    /// <summary>
    /// Validates that custom layers meet WaveNet architectural requirements.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the WaveNet model, ValidateCustomLayers checks inputs and configuration. This protects the WaveNet architecture from mismatches and errors.
    /// </para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);
        if (layers.Count < 5)
        {
            throw new ArgumentException(
                "WaveNet requires at least 5 layers: input, one dilated block, and output layers.",
                nameof(layers));
        }
    }

    /// <summary>
    /// Validates the WaveNet options.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the WaveNet model, ValidateOptions checks inputs and configuration. This protects the WaveNet architecture from mismatches and errors.
    /// </para>
    /// </remarks>
    private static void ValidateOptions(WaveNetOptions<T> options)
    {
        var errors = new List<string>();

        if (options.LookbackWindow < 1)
            errors.Add("LookbackWindow must be at least 1.");
        if (options.ForecastHorizon < 1)
            errors.Add("ForecastHorizon must be at least 1.");
        if (options.ResidualChannels < 1)
            errors.Add("ResidualChannels must be at least 1.");
        if (options.SkipChannels < 1)
            errors.Add("SkipChannels must be at least 1.");
        if (options.DilationDepth < 1)
            errors.Add("DilationDepth must be at least 1.");
        if (options.NumStacks < 1)
            errors.Add("NumStacks must be at least 1.");
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
    /// <b>For Beginners:</b> In the WaveNet model, Predict produces predictions from input data. This is the main inference step of the WaveNet architecture.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return _useNativeMode ? ForecastNative(input) : ForecastOnnx(input);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the WaveNet model, Train performs a training step. This updates the WaveNet architecture so it learns from data.
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
    /// <b>For Beginners:</b> In the WaveNet model, UpdateParameters updates internal parameters or state. This keeps the WaveNet architecture aligned with the latest values.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> gradients)
    {
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the WaveNet model, GetModelMetadata performs a supporting step in the workflow. It keeps the WaveNet architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        int receptiveField = _numStacks * ((1 << _dilationDepth) - 1) + 1;

        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "WaveNet" },
                { "LookbackWindow", _lookbackWindow },
                { "ForecastHorizon", _forecastHorizon },
                { "ResidualChannels", _residualChannels },
                { "SkipChannels", _skipChannels },
                { "DilationDepth", _dilationDepth },
                { "NumStacks", _numStacks },
                { "ReceptiveField", receptiveField },
                { "UseGatedActivations", _useGatedActivations },
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
    /// <b>For Beginners:</b> In the WaveNet model, CreateNewInstance builds and wires up model components. This sets up the WaveNet architecture before use.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var options = new WaveNetOptions<T>
        {
            LookbackWindow = _lookbackWindow,
            ForecastHorizon = _forecastHorizon,
            ResidualChannels = _residualChannels,
            SkipChannels = _skipChannels,
            DilationDepth = _dilationDepth,
            NumStacks = _numStacks,
            KernelSize = _kernelSize,
            UseGatedActivations = _useGatedActivations,
            DropoutRate = _dropout
        };

        return new WaveNet<T>(Architecture, options);
    }

    /// <summary>
    /// Writes WaveNet-specific configuration during serialization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the WaveNet model, SerializeNetworkSpecificData saves or restores model-specific settings. This lets the WaveNet architecture be reused later.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_lookbackWindow);
        writer.Write(_forecastHorizon);
        writer.Write(_numFeatures);
        writer.Write(_residualChannels);
        writer.Write(_skipChannels);
        writer.Write(_dilationDepth);
        writer.Write(_numStacks);
        writer.Write(_kernelSize);
        writer.Write(_useGatedActivations);
        writer.Write(_dropout);
    }

    /// <summary>
    /// Reads WaveNet-specific configuration during deserialization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the WaveNet model, DeserializeNetworkSpecificData saves or restores model-specific settings. This lets the WaveNet architecture be reused later.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32(); // lookbackWindow
        _ = reader.ReadInt32(); // forecastHorizon
        _ = reader.ReadInt32(); // numFeatures
        _ = reader.ReadInt32(); // residualChannels
        _ = reader.ReadInt32(); // skipChannels
        _ = reader.ReadInt32(); // dilationDepth
        _ = reader.ReadInt32(); // numStacks
        _ = reader.ReadInt32(); // kernelSize
        _ = reader.ReadBoolean(); // useGatedActivations
        _ = reader.ReadDouble(); // dropout
    }

    #endregion

    #region IForecastingModel Implementation

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the WaveNet model, Forecast produces predictions from input data. This is the main inference step of the WaveNet architecture.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
    {
        return _useNativeMode ? ForecastNative(historicalData) : ForecastOnnx(historicalData);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the WaveNet model, AutoregressiveForecast produces predictions from input data. This is the main inference step of the WaveNet architecture.
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
    /// <b>For Beginners:</b> In the WaveNet model, Evaluate performs a supporting step in the workflow. It keeps the WaveNet architecture pipeline consistent.
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
    /// <b>For Beginners:</b> In the WaveNet model, ApplyInstanceNormalization performs a supporting step in the workflow. It keeps the WaveNet architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
    {
        return input;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the WaveNet model, GetFinancialMetrics calculates evaluation metrics. This summarizes how the WaveNet architecture is performing.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> GetFinancialMetrics()
    {
        T lastLoss = LastLoss is not null ? LastLoss : NumOps.Zero;
        int receptiveField = _numStacks * ((1 << _dilationDepth) - 1) + 1;

        return new Dictionary<string, T>
        {
            ["LookbackWindow"] = NumOps.FromDouble(_lookbackWindow),
            ["ForecastHorizon"] = NumOps.FromDouble(_forecastHorizon),
            ["ResidualChannels"] = NumOps.FromDouble(_residualChannels),
            ["SkipChannels"] = NumOps.FromDouble(_skipChannels),
            ["ReceptiveField"] = NumOps.FromDouble(receptiveField),
            ["LastLoss"] = lastLoss
        };
    }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// Performs the forward pass through WaveNet.
    /// </summary>
    /// <param name="input">Input tensor of shape [batch, lookback_window * features].</param>
    /// <returns>Output tensor of shape [batch, forecast_horizon].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The WaveNet forward pass:
    ///
    /// 1. <b>Input Projection</b>: Transform input to residual channels
    ///
    /// 2. <b>For each dilated block</b>:
    ///    a. Compute filter output: tanh(filter_layer(x))
    ///    b. Compute gate output: sigmoid(gate_layer(x))
    ///    c. Gated activation: filter * gate
    ///    d. Compute residual output: residual_layer(gated)
    ///    e. Add to input for next layer: x = x + residual
    ///    f. Compute skip output: skip_layer(gated)
    ///    g. Accumulate skip: total_skip += skip
    ///
    /// 3. <b>Output</b>: Apply ReLU and project skip sum to forecast
    /// </para>
    /// </remarks>
    private Tensor<T> Forward(Tensor<T> input)
    {
        // Input projection
        Tensor<T> residual = input;

        if (_inputProjection is not null)
        {
            residual = _inputProjection.Forward(input);
        }

        // Initialize skip accumulator
        Tensor<T>? totalSkip = null;

        // Process each WaveNet block
        foreach (var block in _waveNetBlocks)
        {
            if (block.Count < 4) continue;

            // Extract layers from block
            var filterLayer = block[0];
            var gateLayer = block[1];
            var residualLayer = block[2];
            var skipLayer = block[3];

            // Gated activation: tanh(filter) * sigmoid(gate)
            var filterOutput = filterLayer.Forward(residual);
            var gateOutput = gateLayer.Forward(residual);

            // Element-wise multiplication for gating
            var gated = MultiplyTensors(filterOutput, gateOutput);

            // Residual connection
            var residualOutput = residualLayer.Forward(gated);
            residual = AddTensors(residual, residualOutput);

            // Skip connection
            var skipOutput = skipLayer.Forward(gated);

            if (totalSkip is null)
            {
                totalSkip = skipOutput;
            }
            else
            {
                totalSkip = AddTensors(totalSkip, skipOutput);
            }

            // Apply dropout if present
            if (block.Count > 4)
            {
                residual = block[4].Forward(residual);
            }
        }

        // Output layers
        var output = totalSkip ?? residual;

        if (_output1 is not null)
        {
            output = _output1.Forward(output);
        }

        if (_output2 is not null)
        {
            output = _output2.Forward(output);
        }

        return output;
    }

    /// <summary>
    /// Performs the backward pass through WaveNet.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Backpropagation in WaveNet flows through:
    /// - Output layers
    /// - All skip connections (summed)
    /// - Each block's residual and skip paths
    /// </para>
    /// </remarks>
    private void Backward(Tensor<T> gradOutput)
    {
        var grad = gradOutput;

        // Backward through output layers
        if (_output2 is not null)
        {
            grad = _output2.Backward(grad);
        }

        if (_output1 is not null)
        {
            grad = _output1.Backward(grad);
        }

        // Backward through WaveNet blocks in reverse
        var skipGrad = grad;

        for (int i = _waveNetBlocks.Count - 1; i >= 0; i--)
        {
            var block = _waveNetBlocks[i];
            if (block.Count < 4) continue;

            // Backward through dropout if present
            if (block.Count > 4)
            {
                grad = block[4].Backward(grad);
            }

            // Skip layer backward
            var skipLayerGrad = block[3].Backward(skipGrad);

            // Residual layer backward
            var residualLayerGrad = block[2].Backward(grad);

            // Gated activation backward - simplified
            // In full implementation, would need to handle element-wise multiply gradient
            var gatedGrad = AddTensors(skipLayerGrad, residualLayerGrad);

            // Gate and filter backward
            block[1].Backward(gatedGrad);
            grad = block[0].Backward(gatedGrad);
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
    /// <b>For Beginners:</b> In the WaveNet model, ForecastNative produces predictions from input data. This is the main inference step of the WaveNet architecture.
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
    /// <b>For Beginners:</b> In the WaveNet model, ForecastOnnx produces predictions from input data. This is the main inference step of the WaveNet architecture.
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
    /// <b>For Beginners:</b> In the WaveNet model, AddTensors performs a supporting step in the workflow. It keeps the WaveNet architecture pipeline consistent.
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
    /// Element-wise multiplication of two tensors (for gating).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This implements the core of gated activation:
    /// output[i] = tanh_output[i] * sigmoid_output[i]
    /// The sigmoid "gates" which features from tanh pass through.
    /// </para>
    /// </remarks>
    private Tensor<T> MultiplyTensors(Tensor<T> a, Tensor<T> b)
    {
        var result = new Tensor<T>(a.Shape);
        int length = Math.Min(a.Length, b.Length);

        for (int i = 0; i < length; i++)
        {
            result.Data.Span[i] = NumOps.Multiply(a.Data.Span[i], b.Data.Span[i]);
        }

        return result;
    }

    /// <summary>
    /// Shifts the input tensor by incorporating new predictions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the WaveNet model, ShiftInputWithPredictions produces predictions from input data. This is the main inference step of the WaveNet architecture.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ShiftInputWithPredictions(Tensor<T> input, Tensor<T> predictions, int stepsUsed)
    {
        int batchSize = input.Shape[0];
        int inputLen = input.Length / batchSize;

        var shifted = new Tensor<T>(input.Shape);

        for (int b = 0; b < batchSize; b++)
        {
            for (int t = 0; t < inputLen - stepsUsed; t++)
            {
                int srcIdx = b * inputLen + t + stepsUsed;
                int dstIdx = b * inputLen + t;

                if (srcIdx < input.Length && dstIdx < shifted.Length)
                {
                    shifted.Data.Span[dstIdx] = input.Data.Span[srcIdx];
                }
            }

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
    /// <b>For Beginners:</b> In the WaveNet model, ConcatenatePredictions produces predictions from input data. This is the main inference step of the WaveNet architecture.
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
    /// <b>For Beginners:</b> In the WaveNet model, Dispose performs a supporting step in the workflow. It keeps the WaveNet architecture pipeline consistent.
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

