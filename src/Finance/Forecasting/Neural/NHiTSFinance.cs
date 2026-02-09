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
/// N-HiTS (Neural Hierarchical Interpolation for Time Series) model for financial forecasting.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// N-HiTS improves upon N-BEATS by incorporating hierarchical interpolation and multi-rate
/// signal sampling. It achieves better accuracy on long-horizon forecasting while being
/// more parameter-efficient through its stack-specific pooling approach.
/// </para>
/// <para>
/// <b>For Beginners:</b> N-HiTS works like having multiple "zoom levels" on your time series:
/// - One level looks at fine details (hourly patterns)
/// - Another looks at medium patterns (daily patterns)
/// - Another looks at the big picture (weekly/monthly trends)
///
/// Each level (stack) samples the data at different rates:
/// - High-frequency stack: Processes data at full resolution
/// - Medium-frequency stack: Downsamples by 4x (averages 4 points into 1)
/// - Low-frequency stack: Downsamples by 8x
///
/// This multi-scale approach helps N-HiTS:
/// - Be more efficient (fewer parameters than N-BEATS)
/// - Handle long horizons better
/// - Capture patterns at different time scales
/// </para>
/// <para>
/// <b>Reference:</b> Challu et al., "N-HiTS: Neural Hierarchical Interpolation for Time Series
/// Forecasting", AAAI 2023. https://arxiv.org/abs/2201.12886
/// </para>
/// </remarks>
public class NHiTSFinance<T> : ForecastingModelBase<T>
{
    #region Execution Mode

    /// <summary>
    /// Indicates whether this network uses native layers (true) or ONNX model (false).
    /// </summary>
    private bool _useNativeMode;

    #endregion


    #region Native Mode Fields

    /// <summary>
    /// Groups of layers for each stack, organized by resolution level.
    /// </summary>
    private readonly List<List<List<ILayer<T>>>> _stackBlocks = [];

    /// <summary>
    /// Final output projection layer.
    /// </summary>
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
    private readonly NHiTSOptions<T> _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// The lookback window size.
    /// </summary>
    private int _lookbackWindow;

    /// <summary>
    /// The forecast horizon.
    /// </summary>
    private int _forecastHorizon;

    /// <summary>
    /// Number of stacks.
    /// </summary>
    private int _numStacks;

    /// <summary>
    /// Number of blocks per stack.
    /// </summary>
    private int _numBlocksPerStack;

    /// <summary>
    /// Size of hidden layers.
    /// </summary>
    private int _hiddenSize;

    /// <summary>
    /// Number of hidden layers per block.
    /// </summary>
    private int _numHiddenLayers;

    /// <summary>
    /// Pooling kernel sizes for each stack.
    /// </summary>
    private int[] _poolingKernelSizes;

    /// <summary>
    /// Pooling modes for each stack.
    /// </summary>
    private string[] _poolingModes;

    /// <summary>
    /// Interpolation modes for each stack.
    /// </summary>
    private string[] _interpolationModes;

    /// <summary>
    /// Dropout rate for regularization.
    /// </summary>
    private double _dropout;

    #endregion

    #region IForecastingModel Properties

    /// <inheritdoc/>
    public override int SequenceLength => _lookbackWindow;

    /// <inheritdoc/>
    public override int PredictionHorizon => _forecastHorizon;

    /// <inheritdoc/>
    public override int NumFeatures => 1; // N-HiTS works on univariate series

    /// <inheritdoc/>
    public override int PatchSize => 1;

    /// <inheritdoc/>
    public override int Stride => 1;

    /// <inheritdoc/>
    public override bool IsChannelIndependent => true;

    /// <inheritdoc/>
    public override bool UseNativeMode => _useNativeMode;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates an N-HiTS network using pretrained ONNX model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer for fine-tuning.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the NHiTSFinance model, NHiTSFinance sets up the architecture and options. This prepares the model for training or inference.
    /// </para>
    /// </remarks>
    public NHiTSFinance(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        NHiTSOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        options ??= new NHiTSOptions<T>();
        _options = options;
        Options = _options;
        ValidateOptions(options);

        _useNativeMode = false;
        OnnxSession = new InferenceSession(onnxModelPath);
        OnnxModelPath = onnxModelPath;

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        _lookbackWindow = options.LookbackWindow;
        _forecastHorizon = options.ForecastHorizon;
        _numStacks = options.NumStacks;
        _numBlocksPerStack = options.NumBlocksPerStack;
        _hiddenSize = options.HiddenLayerSize;
        _numHiddenLayers = options.NumHiddenLayers;
        _poolingKernelSizes = options.PoolingKernelSizes;
        _poolingModes = options.PoolingModes;
        _interpolationModes = options.InterpolationModes;
        _dropout = options.DropoutRate;
    }

    /// <summary>
    /// Creates an N-HiTS network in native mode for training from scratch.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">Configuration options for the model.</param>
    /// <param name="optimizer">Optional optimizer.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor to train a new N-HiTS model.
    /// N-HiTS excels at:
    /// - Long-horizon forecasting (predicting many steps ahead)
    /// - Multi-scale pattern recognition
    /// - Parameter efficiency compared to N-BEATS
    /// </para>
    /// </remarks>
    public NHiTSFinance(
        NeuralNetworkArchitecture<T> architecture,
        NHiTSOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new NHiTSOptions<T>();
        _options = options;
        Options = _options;
        ValidateOptions(options);

        _useNativeMode = true;
        OnnxSession = null;
        OnnxModelPath = null;

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        _lookbackWindow = options.LookbackWindow;
        _forecastHorizon = options.ForecastHorizon;
        _numStacks = options.NumStacks;
        _numBlocksPerStack = options.NumBlocksPerStack;
        _hiddenSize = options.HiddenLayerSize;
        _numHiddenLayers = options.NumHiddenLayers;
        _poolingKernelSizes = options.PoolingKernelSizes;
        _poolingModes = options.PoolingModes;
        _interpolationModes = options.InterpolationModes;
        _dropout = options.DropoutRate;

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes the neural network layers for N-HiTS.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> N-HiTS builds on N-BEATS with key differences:
    /// - Each stack operates at a different time resolution
    /// - Pooling reduces input resolution before processing
    /// - Interpolation expands output back to target resolution
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
            ExtractLayerReferences();
        }
        else if (_useNativeMode)
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultNHiTSLayers(
                Architecture, _lookbackWindow, _forecastHorizon, _numStacks,
                _numBlocksPerStack, _hiddenSize, _numHiddenLayers, _dropout));

            ExtractLayerReferences();
        }
    }

    /// <summary>
    /// Extracts references to specific layers from the layer collection.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the NHiTSFinance model, ExtractLayerReferences performs a supporting step in the workflow. It keeps the NHiTSFinance architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private void ExtractLayerReferences()
    {
        int idx = 0;
        _stackBlocks.Clear();

        for (int stack = 0; stack < _numStacks; stack++)
        {
            var stackBlockList = new List<List<ILayer<T>>>();
            int layersPerBlock = _numHiddenLayers + 2; // hidden + dropout + backcast + forecast
            if (_dropout > 0)
                layersPerBlock += _numHiddenLayers - 1; // Additional dropout layers

            for (int block = 0; block < _numBlocksPerStack; block++)
            {
                var blockLayers = new List<ILayer<T>>();
                for (int i = 0; i < layersPerBlock && idx < Layers.Count - 1; i++)
                {
                    blockLayers.Add(Layers[idx++]);
                }
                stackBlockList.Add(blockLayers);
            }
            _stackBlocks.Add(stackBlockList);
        }

        if (idx < Layers.Count)
            _outputProjection = Layers[idx];
    }

    /// <summary>
    /// Validates that custom layers meet N-HiTS architectural requirements.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the NHiTSFinance model, ValidateCustomLayers checks inputs and configuration. This protects the NHiTSFinance architecture from mismatches and errors.
    /// </para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);
        if (layers.Count < 3)
        {
            throw new ArgumentException(
                "N-HiTS requires at least 3 layers: hidden, backcast, and forecast.",
                nameof(layers));
        }
    }

    /// <summary>
    /// Validates the N-HiTS options.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the NHiTSFinance model, ValidateOptions checks inputs and configuration. This protects the NHiTSFinance architecture from mismatches and errors.
    /// </para>
    /// </remarks>
    private static void ValidateOptions(NHiTSOptions<T> options)
    {
        var errors = new List<string>();

        if (options.LookbackWindow < 1)
            errors.Add("LookbackWindow must be at least 1.");
        if (options.ForecastHorizon < 1)
            errors.Add("ForecastHorizon must be at least 1.");
        if (options.NumStacks < 1)
            errors.Add("NumStacks must be at least 1.");
        if (options.NumBlocksPerStack < 1)
            errors.Add("NumBlocksPerStack must be at least 1.");
        if (options.HiddenLayerSize < 1)
            errors.Add("HiddenLayerSize must be at least 1.");

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
    /// <b>For Beginners:</b> In the NHiTSFinance model, Predict produces predictions from input data. This is the main inference step of the NHiTSFinance architecture.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return _useNativeMode ? ForecastNative(input) : ForecastOnnx(input);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the NHiTSFinance model, Train performs a training step. This updates the NHiTSFinance architecture so it learns from data.
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
    /// <b>For Beginners:</b> In the NHiTSFinance model, UpdateParameters updates internal parameters or state. This keeps the NHiTSFinance architecture aligned with the latest values.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> gradients)
    {
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the NHiTSFinance model, GetModelMetadata performs a supporting step in the workflow. It keeps the NHiTSFinance architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "N-HiTS" },
                { "LookbackWindow", _lookbackWindow },
                { "ForecastHorizon", _forecastHorizon },
                { "NumStacks", _numStacks },
                { "NumBlocksPerStack", _numBlocksPerStack },
                { "HiddenSize", _hiddenSize },
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
    /// <b>For Beginners:</b> In the NHiTSFinance model, CreateNewInstance builds and wires up model components. This sets up the NHiTSFinance architecture before use.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var options = new NHiTSOptions<T>
        {
            LookbackWindow = _lookbackWindow,
            ForecastHorizon = _forecastHorizon,
            NumStacks = _numStacks,
            NumBlocksPerStack = _numBlocksPerStack,
            HiddenLayerSize = _hiddenSize,
            NumHiddenLayers = _numHiddenLayers,
            PoolingKernelSizes = _poolingKernelSizes,
            PoolingModes = _poolingModes,
            InterpolationModes = _interpolationModes,
            DropoutRate = _dropout
        };

        return new NHiTSFinance<T>(Architecture, options);
    }

    /// <summary>
    /// Writes N-HiTS-specific configuration during serialization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the NHiTSFinance model, SerializeNetworkSpecificData saves or restores model-specific settings. This lets the NHiTSFinance architecture be reused later.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_lookbackWindow);
        writer.Write(_forecastHorizon);
        writer.Write(_numStacks);
        writer.Write(_numBlocksPerStack);
        writer.Write(_hiddenSize);
        writer.Write(_numHiddenLayers);
        writer.Write(_poolingKernelSizes.Length);
        foreach (var k in _poolingKernelSizes)
            writer.Write(k);
        writer.Write(_dropout);
    }

    /// <summary>
    /// Reads N-HiTS-specific configuration during deserialization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the NHiTSFinance model, DeserializeNetworkSpecificData saves or restores model-specific settings. This lets the NHiTSFinance architecture be reused later.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _lookbackWindow = reader.ReadInt32();
        _forecastHorizon = reader.ReadInt32();
        _numStacks = reader.ReadInt32();
        _numBlocksPerStack = reader.ReadInt32();
        _hiddenSize = reader.ReadInt32();
        _numHiddenLayers = reader.ReadInt32();
        int kernelCount = reader.ReadInt32();
        _poolingKernelSizes = new int[kernelCount];
        for (int i = 0; i < kernelCount; i++)
            _poolingKernelSizes[i] = reader.ReadInt32();
        _dropout = reader.ReadDouble();
    }

    #endregion

    #region IForecastingModel Implementation

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the NHiTSFinance model, Forecast produces predictions from input data. This is the main inference step of the NHiTSFinance architecture.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
    {
        return _useNativeMode ? ForecastNative(historicalData) : ForecastOnnx(historicalData);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the NHiTSFinance model, AutoregressiveForecast produces predictions from input data. This is the main inference step of the NHiTSFinance architecture.
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
    /// <b>For Beginners:</b> In the NHiTSFinance model, Evaluate performs a supporting step in the workflow. It keeps the NHiTSFinance architecture pipeline consistent.
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
    /// <b>For Beginners:</b> In the NHiTSFinance model, ApplyInstanceNormalization performs a supporting step in the workflow. It keeps the NHiTSFinance architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
    {
        return input;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the NHiTSFinance model, GetFinancialMetrics calculates evaluation metrics. This summarizes how the NHiTSFinance architecture is performing.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> GetFinancialMetrics()
    {
        T lastLoss = LastLoss is not null ? LastLoss : NumOps.Zero;
        return new Dictionary<string, T>
        {
            ["LookbackWindow"] = NumOps.FromDouble(_lookbackWindow),
            ["ForecastHorizon"] = NumOps.FromDouble(_forecastHorizon),
            ["NumStacks"] = NumOps.FromDouble(_numStacks),
            ["HiddenSize"] = NumOps.FromDouble(_hiddenSize),
            ["LastLoss"] = lastLoss
        };
    }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// Performs the forward pass through the N-HiTS network.
    /// </summary>
    /// <param name="input">Input tensor of shape [batch, lookback_window].</param>
    /// <returns>Output tensor of shape [batch, forecast_horizon].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> N-HiTS forward pass is similar to N-BEATS but with multi-scale processing:
    /// 1. Each stack first pools the input to its operating resolution
    /// 2. Blocks process the pooled input
    /// 3. Outputs are interpolated back to full resolution
    /// 4. All forecasts are summed
    /// </para>
    /// </remarks>
    private Tensor<T> Forward(Tensor<T> input)
    {
        var residual = input;
        var totalForecast = new Tensor<T>(new[] { input.Shape[0], _forecastHorizon });

        for (int stackIdx = 0; stackIdx < _stackBlocks.Count; stackIdx++)
        {
            int kernelSize = stackIdx < _poolingKernelSizes.Length ?
                _poolingKernelSizes[stackIdx] : 1;

            foreach (var blockLayers in _stackBlocks[stackIdx])
            {
                if (blockLayers.Count == 0) continue;

                // Pool input to stack's resolution (updated per block to reflect residuals)
                var pooledInput = ApplyPooling(residual, kernelSize);

                // Forward through block layers
                var current = pooledInput;
                int numHidden = blockLayers.Count - 2;

                for (int i = 0; i < numHidden && i < blockLayers.Count; i++)
                {
                    current = blockLayers[i].Forward(current);
                }

                // Get backcast coefficients
                Tensor<T>? backcastCoeffs = null;
                if (numHidden < blockLayers.Count)
                {
                    backcastCoeffs = blockLayers[numHidden].Forward(current);
                }

                // Get forecast coefficients
                Tensor<T>? forecastCoeffs = null;
                if (numHidden + 1 < blockLayers.Count)
                {
                    forecastCoeffs = blockLayers[numHidden + 1].Forward(current);
                }

                // Interpolate backcast and forecast to full resolution
                if (backcastCoeffs is not null)
                {
                    var backcast = InterpolateToLength(backcastCoeffs, _lookbackWindow);
                    residual = SubtractTensors(residual, backcast);
                }

                if (forecastCoeffs is not null)
                {
                    var forecast = InterpolateToLength(forecastCoeffs, _forecastHorizon);
                    totalForecast = AddTensors(totalForecast, forecast);
                }
            }
        }

        if (_outputProjection is not null)
        {
            totalForecast = _outputProjection.Forward(totalForecast);
        }

        return totalForecast;
    }

    /// <summary>
    /// Performs the backward pass through the N-HiTS network.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the NHiTSFinance model, Backward propagates gradients backward. This teaches the NHiTSFinance architecture how to adjust its weights.
    /// </para>
    /// </remarks>
    private void Backward(Tensor<T> gradOutput)
    {
        var grad = gradOutput;

        if (_outputProjection is not null)
        {
            grad = _outputProjection.Backward(grad);
        }

        for (int i = _stackBlocks.Count - 1; i >= 0; i--)
        {
            foreach (var blockLayers in _stackBlocks[i])
            {
                for (int j = blockLayers.Count - 1; j >= 0; j--)
                {
                    grad = blockLayers[j].Backward(grad);
                }
            }
        }
    }

    /// <summary>
    /// Performs native mode forecasting.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the NHiTSFinance model, ForecastNative produces predictions from input data. This is the main inference step of the NHiTSFinance architecture.
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
    /// <b>For Beginners:</b> In the NHiTSFinance model, ForecastOnnx produces predictions from input data. This is the main inference step of the NHiTSFinance architecture.
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
    /// Applies pooling to reduce input resolution.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <param name="kernelSize">Pooling kernel size.</param>
    /// <returns>Pooled tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Pooling reduces the length of the input by combining
    /// multiple values into one. With kernel size 4, every 4 values become 1.
    /// </para>
    /// </remarks>
    private Tensor<T> ApplyPooling(Tensor<T> input, int kernelSize)
    {
        if (kernelSize <= 1)
            return input;

        int batchSize = input.Shape[0];
        int seqLen = input.Shape.Length > 1 ? input.Shape[1] : input.Length / batchSize;
        int pooledLen = Math.Max(1, seqLen / kernelSize);

        var pooled = new Tensor<T>(new[] { batchSize, pooledLen });

        for (int b = 0; b < batchSize; b++)
        {
            for (int p = 0; p < pooledLen; p++)
            {
                // Average pooling
                T sum = NumOps.Zero;
                int count = 0;
                for (int k = 0; k < kernelSize && p * kernelSize + k < seqLen; k++)
                {
                    int srcIdx = b * seqLen + p * kernelSize + k;
                    if (srcIdx < input.Length)
                    {
                        sum = NumOps.Add(sum, input.Data.Span[srcIdx]);
                        count++;
                    }
                }
                int dstIdx = b * pooledLen + p;
                if (dstIdx < pooled.Length && count > 0)
                {
                    pooled.Data.Span[dstIdx] = NumOps.Divide(sum, NumOps.FromDouble(count));
                }
            }
        }

        return pooled;
    }

    /// <summary>
    /// Interpolates coefficients to target length.
    /// </summary>
    /// <param name="coeffs">Coefficient tensor.</param>
    /// <param name="targetLength">Target output length.</param>
    /// <returns>Interpolated tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Interpolation expands the coefficients back to the
    /// original resolution by estimating values between known points.
    /// </para>
    /// </remarks>
    private Tensor<T> InterpolateToLength(Tensor<T> coeffs, int targetLength)
    {
        int batchSize = coeffs.Shape[0];
        int coeffLen = coeffs.Shape.Length > 1 ? coeffs.Shape[1] : coeffs.Length / batchSize;

        if (coeffLen == targetLength)
            return coeffs;

        var result = new Tensor<T>(new[] { batchSize, targetLength });

        for (int b = 0; b < batchSize; b++)
        {
            for (int t = 0; t < targetLength; t++)
            {
                // Linear interpolation
                double pos = (double)t / targetLength * coeffLen;
                int idx0 = (int)pos;
                int idx1 = Math.Min(idx0 + 1, coeffLen - 1);
                double frac = pos - idx0;

                int src0 = b * coeffLen + idx0;
                int src1 = b * coeffLen + idx1;
                int dst = b * targetLength + t;

                if (src0 < coeffs.Length && src1 < coeffs.Length && dst < result.Length)
                {
                    T val0 = coeffs.Data.Span[src0];
                    T val1 = coeffs.Data.Span[src1];
                    T interpolated = NumOps.Add(
                        NumOps.Multiply(val0, NumOps.FromDouble(1 - frac)),
                        NumOps.Multiply(val1, NumOps.FromDouble(frac)));
                    result.Data.Span[dst] = interpolated;
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Subtracts two tensors element-wise.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the NHiTSFinance model, SubtractTensors performs a supporting step in the workflow. It keeps the NHiTSFinance architecture pipeline consistent.
    /// </para>
    /// </remarks>
    private Tensor<T> SubtractTensors(Tensor<T> a, Tensor<T> b)
    {
        var result = new Tensor<T>(a.Shape);
        int length = Math.Min(a.Length, b.Length);
        for (int i = 0; i < length; i++)
        {
            result.Data.Span[i] = NumOps.Subtract(a.Data.Span[i], b.Data.Span[i]);
        }
        return result;
    }

    /// <summary>
    /// Adds two tensors element-wise.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the NHiTSFinance model, AddTensors performs a supporting step in the workflow. It keeps the NHiTSFinance architecture pipeline consistent.
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
        return result;
    }

    /// <summary>
    /// Shifts input by incorporating recent predictions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the NHiTSFinance model, ShiftInputWithPredictions produces predictions from input data. This is the main inference step of the NHiTSFinance architecture.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ShiftInputWithPredictions(Tensor<T> input, Tensor<T> prediction, int stepsUsed)
    {
        int batchSize = input.Shape[0];
        int seqLen = input.Shape.Length > 1 ? input.Shape[1] : input.Length / batchSize;
        int steps = Math.Min(stepsUsed, seqLen);

        var shifted = new Tensor<T>(input.Shape);

        for (int b = 0; b < batchSize; b++)
        {
            for (int t = 0; t < seqLen - steps; t++)
            {
                int srcIdx = b * seqLen + t + steps;
                int dstIdx = b * seqLen + t;
                if (srcIdx < input.Length && dstIdx < shifted.Length)
                    shifted.Data.Span[dstIdx] = input.Data.Span[srcIdx];
            }

            for (int t = seqLen - steps; t < seqLen; t++)
            {
                int predIdx = b * steps + (t - (seqLen - steps));
                int dstIdx = b * seqLen + t;
                if (predIdx < prediction.Length && dstIdx < shifted.Length)
                    shifted.Data.Span[dstIdx] = prediction.Data.Span[predIdx];
            }
        }

        return shifted;
    }

    /// <summary>
    /// Concatenates multiple predictions into a single tensor.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the NHiTSFinance model, ConcatenatePredictions produces predictions from input data. This is the main inference step of the NHiTSFinance architecture.
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
            int predSteps = pred.Shape.Length > 1 ? pred.Shape[1] : pred.Length / batchSize;
            int stepsToCopy = Math.Min(predSteps, totalSteps - currentStep);

            for (int b = 0; b < batchSize; b++)
            {
                for (int t = 0; t < stepsToCopy; t++)
                {
                    int srcIdx = b * predSteps + t;
                    int dstIdx = b * totalSteps + currentStep + t;
                    if (srcIdx < pred.Length && dstIdx < result.Length)
                        result.Data.Span[dstIdx] = pred.Data.Span[srcIdx];
                }
            }

            currentStep += stepsToCopy;
            if (currentStep >= totalSteps)
                break;
        }

        return result;
    }

    #endregion

    #region IDisposable

    /// <summary>
    /// Releases resources used by the N-HiTS model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the NHiTSFinance model, Dispose performs a supporting step in the workflow. It keeps the NHiTSFinance architecture pipeline consistent.
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

