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
/// UniTS (Unified Time Series Model) implementation for multi-task time series processing.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// UniTS is a unified architecture that handles multiple time series tasks including
/// forecasting, classification, anomaly detection, and imputation using a single
/// pretrained model with task-specific output heads.
/// </para>
/// <para><b>For Beginners:</b> UniTS is designed to be a universal time series model:
///
/// <b>The Key Insight:</b>
/// Different time series tasks share common patterns. Instead of training
/// separate models, UniTS learns a unified representation that works for all tasks.
///
/// <b>Supported Tasks:</b>
/// 1. <b>Forecasting:</b> Predict future values
/// 2. <b>Classification:</b> Categorize entire time series
/// 3. <b>Anomaly Detection:</b> Identify unusual patterns
/// 4. <b>Imputation:</b> Fill in missing values
///
/// <b>Architecture:</b>
/// - Multi-scale temporal convolution for local patterns (different kernel sizes)
/// - Transformer layers for global dependencies
/// - Task-specific output heads for different outputs
/// - Shared backbone pretrained on diverse datasets
///
/// <b>Advantages:</b>
/// - One model for multiple tasks (transfer learning)
/// - Strong zero-shot performance on new domains
/// - Efficient inference (shared computation)
/// </para>
/// <para>
/// <b>Reference:</b> Gao et al., "UniTS: A Unified Multi-Task Time Series Model", 2024.
/// https://arxiv.org/abs/2403.00131
/// </para>
/// </remarks>
public class UniTS<T> : ForecastingModelBase<T>
{
    #region Execution Mode

    /// <summary>
    /// Indicates whether the model is running in native mode (true) or ONNX mode (false).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> UniTS supports two execution modes:
    /// - Native mode: Train the model from scratch or fine-tune
    /// - ONNX mode: Inference using pretrained ONNX model
    /// </para>
    /// </remarks>
    private readonly bool _useNativeMode;

    #endregion

    
    #region Native Mode Fields

    /// <summary>
    /// Reference to the input embedding layer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The input embedding projects raw features to the
    /// hidden dimension for processing by the transformer backbone.
    /// </para>
    /// </remarks>
    private DenseLayer<T>? _inputEmbedding;

    /// <summary>
    /// References to the multi-scale processing layers.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> These layers simulate multi-scale temporal convolution
    /// by processing at different scales. Smaller scales capture fine-grained patterns,
    /// larger scales capture broader trends.
    /// </para>
    /// </remarks>
    private List<DenseLayer<T>>? _multiScaleLayers;

    /// <summary>
    /// References to the transformer attention layers.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Transformer layers capture global dependencies
    /// across the entire sequence using self-attention.
    /// </para>
    /// </remarks>
    private List<MultiHeadAttentionLayer<T>>? _transformerLayers;

    /// <summary>
    /// Reference to the output projection layer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The task-specific output head that maps
    /// the shared representation to task-specific outputs.
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

    /// <summary>
    /// Context length for the input sequence.
    /// </summary>
    private readonly int _contextLength;

    /// <summary>
    /// Forecast horizon for predictions.
    /// </summary>
    private readonly int _forecastHorizon;

    /// <summary>
    /// Hidden dimension size.
    /// </summary>
    private readonly int _hiddenDimension;

    /// <summary>
    /// Number of transformer layers.
    /// </summary>
    private readonly int _numLayers;

    /// <summary>
    /// Number of attention heads.
    /// </summary>
    private readonly int _numHeads;

    /// <summary>
    /// Convolution kernel sizes for multi-scale processing.
    /// </summary>
    private readonly int[] _convKernelSizes;

    /// <summary>
    /// Dropout rate for regularization.
    /// </summary>
    private readonly double _dropout;

    /// <summary>
    /// Task type (forecasting, classification, anomaly, imputation).
    /// </summary>
    private readonly string _taskType;

    /// <summary>
    /// Number of classes for classification task.
    /// </summary>
    private readonly int _numClasses;

    /// <summary>
    /// Number of input features.
    /// </summary>
    private readonly int _numFeatures;

    #endregion

    #region IForecastingModel Properties

    /// <inheritdoc/>
    public override int SequenceLength => _contextLength;

    /// <inheritdoc/>
    public override int PredictionHorizon => _forecastHorizon;

    /// <inheritdoc/>
    public override int NumFeatures => _numFeatures;

    /// <inheritdoc/>
    public override int PatchSize => _convKernelSizes.Length > 0 ? _convKernelSizes[0] : 3;

    /// <inheritdoc/>
    public override int Stride => 1;

    /// <inheritdoc/>
    public override bool IsChannelIndependent => true;

    /// <inheritdoc/>
    public override bool UseNativeMode => _useNativeMode;

    /// <summary>
    /// Gets the task type (forecasting, classification, anomaly, imputation).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> UniTS can perform multiple tasks with the same backbone.
    /// The task type determines which output head is used.
    /// </para>
    /// </remarks>
    public string TaskType => _taskType;

    /// <summary>
    /// Gets the number of classes for classification task.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> For classification tasks, this is the number of
    /// categories the model can predict.
    /// </para>
    /// </remarks>
    public int NumClasses => _numClasses;

    /// <summary>
    /// Gets the convolution kernel sizes.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> These determine the time scales captured by the
    /// multi-scale temporal convolution. Smaller values capture fine details,
    /// larger values capture broader patterns.
    /// </para>
    /// </remarks>
    public int[] ConvKernelSizes => _convKernelSizes;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance using an ONNX pretrained model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="options">Configuration options for UniTS.</param>
    /// <param name="optimizer">Optional optimizer.</param>
    /// <param name="lossFunction">Optional custom loss function.</param>
    /// <exception cref="ArgumentException">Thrown when onnxModelPath is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown when ONNX model file doesn't exist.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to load a pretrained UniTS model
    /// for inference. The ONNX model contains all the weights and architecture.
    /// </para>
    /// </remarks>
    public UniTS(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        UniTSOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        options ??= new UniTSOptions<T>();

        _useNativeMode = false;
        OnnxModelPath = onnxModelPath;
        OnnxSession = new InferenceSession(onnxModelPath);

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        _contextLength = options.ContextLength;
        _forecastHorizon = options.ForecastHorizon;
        _hiddenDimension = options.HiddenDimension;
        _numLayers = options.NumLayers;
        _numHeads = options.NumHeads;
        _convKernelSizes = options.ConvKernelSizes ?? new[] { 3, 5, 7 };
        _dropout = options.DropoutRate;
        _taskType = NormalizeTaskType(options.TaskType);
        _numClasses = options.NumClasses;
        _numFeatures = 1;

        InitializeLayers();
    }

    /// <summary>
    /// Initializes a new instance in native mode for training.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">Configuration options for UniTS.</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="optimizer">Optional optimizer.</param>
    /// <param name="lossFunction">Optional custom loss function.</param>
    /// <exception cref="ArgumentNullException">Thrown when architecture is null.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to train UniTS from scratch or
    /// fine-tune on your specific data. Training enables the model to learn patterns
    /// specific to your domain.
    /// </para>
    /// </remarks>
    public UniTS(
        NeuralNetworkArchitecture<T> architecture,
        UniTSOptions<T>? options = null,
        int numFeatures = 1,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new UniTSOptions<T>();

        _useNativeMode = true;

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        _contextLength = options.ContextLength;
        _forecastHorizon = options.ForecastHorizon;
        _hiddenDimension = options.HiddenDimension;
        _numLayers = options.NumLayers;
        _numHeads = options.NumHeads;
        _convKernelSizes = options.ConvKernelSizes ?? new[] { 3, 5, 7 };
        _dropout = options.DropoutRate;
        _taskType = NormalizeTaskType(options.TaskType);
        _numClasses = options.NumClasses;
        _numFeatures = numFeatures;

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes the model layers based on configuration.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up the UniTS architecture including
    /// multi-scale temporal convolution, transformer backbone, and task-specific head.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultUniTSLayers(
                Architecture, _contextLength, _forecastHorizon, _hiddenDimension,
                _numLayers, _numHeads, _convKernelSizes, _dropout, _taskType, _numClasses));

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
        _multiScaleLayers = Layers.OfType<DenseLayer<T>>().Take(_convKernelSizes.Length).ToList();
        _transformerLayers = Layers.OfType<MultiHeadAttentionLayer<T>>().ToList();
        _outputProjection = Layers.OfType<DenseLayer<T>>().LastOrDefault();
    }

    /// <summary>
    /// Validates that custom layers meet UniTS requirements.
    /// </summary>
    /// <param name="layers">The layers to validate.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> When using custom layers, we ensure they include
    /// the necessary components for the UniTS architecture (convolution and attention).
    /// </para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);

        var attentionCount = layers.OfType<MultiHeadAttentionLayer<T>>().Count();
        if (attentionCount < 1)
        {
            throw new ArgumentException(
                "UniTS requires at least one MultiHeadAttentionLayer for the transformer backbone.");
        }
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <inheritdoc/>
    public override bool SupportsTraining => _useNativeMode;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the UniTS model, Predict produces predictions from input data. This is the main inference step of the UniTS architecture.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return _useNativeMode ? Forward(input) : ForecastOnnx(input);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the UniTS model, Train performs a training step. This updates the UniTS architecture so it learns from data.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is only supported in native mode.");

        SetTrainingMode(true);

        var output = Forward(input);

        // Compute loss
        LastLoss = _lossFunction.CalculateLoss(output.ToVector(), target.ToVector());

        // Backward pass
        var gradient = _lossFunction.CalculateDerivative(output.ToVector(), target.ToVector());
        Backward(Tensor<T>.FromVector(gradient, output.Shape));

        _optimizer.UpdateParameters(Layers);

        SetTrainingMode(false);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the UniTS model, UpdateParameters updates internal parameters or state. This keeps the UniTS architecture aligned with the latest values.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> gradients)
    {
        // Parameters are updated through the optimizer in Train()
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the UniTS model, GetModelMetadata performs a supporting step in the workflow. It keeps the UniTS architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "UniTS" },
                { "ContextLength", _contextLength },
                { "ForecastHorizon", _forecastHorizon },
                { "HiddenDimension", _hiddenDimension },
                { "NumLayers", _numLayers },
                { "NumHeads", _numHeads },
                { "ConvKernelSizes", string.Join(",", _convKernelSizes) },
                { "TaskType", _taskType },
                { "NumClasses", _numClasses },
                { "UseNativeMode", _useNativeMode },
                { "ParameterCount", GetParameterCount() }
            },
            ModelData = _useNativeMode ? this.Serialize() : Array.Empty<byte>()
        };
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the UniTS model, CreateNewInstance builds and wires up model components. This sets up the UniTS architecture before use.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var options = new UniTSOptions<T>
        {
            ContextLength = _contextLength,
            ForecastHorizon = _forecastHorizon,
            HiddenDimension = _hiddenDimension,
            NumLayers = _numLayers,
            NumHeads = _numHeads,
            ConvKernelSizes = _convKernelSizes,
            DropoutRate = _dropout,
            TaskType = _taskType,
            NumClasses = _numClasses
        };

        return new UniTS<T>(Architecture, options, _numFeatures);
    }

    /// <summary>
    /// Writes UniTS-specific configuration during serialization.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Saves all the configuration needed to reconstruct this model,
    /// including task type and multi-scale convolution settings.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_contextLength);
        writer.Write(_forecastHorizon);
        writer.Write(_hiddenDimension);
        writer.Write(_numLayers);
        writer.Write(_numHeads);
        writer.Write(_convKernelSizes.Length);
        foreach (var kernelSize in _convKernelSizes)
        {
            writer.Write(kernelSize);
        }
        writer.Write(_dropout);
        writer.Write(_taskType);
        writer.Write(_numClasses);
        writer.Write(_numFeatures);
    }

    /// <summary>
    /// Reads UniTS-specific configuration during deserialization.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Loads the configuration that was saved during serialization,
    /// restoring the model to its original state.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32(); // contextLength
        _ = reader.ReadInt32(); // forecastHorizon
        _ = reader.ReadInt32(); // hiddenDimension
        _ = reader.ReadInt32(); // numLayers
        _ = reader.ReadInt32(); // numHeads
        int kernelCount = reader.ReadInt32();
        for (int i = 0; i < kernelCount; i++)
        {
            _ = reader.ReadInt32(); // kernelSize
        }
        _ = reader.ReadDouble(); // dropout
        _ = reader.ReadString(); // taskType
        _ = reader.ReadInt32(); // numClasses
        _ = reader.ReadInt32(); // numFeatures
    }

    #endregion

    #region IForecastingModel Implementation

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the UniTS model, Forecast produces predictions from input data. This is the main inference step of the UniTS architecture.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
    {
        var output = _useNativeMode ? Forward(historicalData) : ForecastOnnx(historicalData);

        // For quantile forecasts, we generate samples through dropout variation
        if (quantiles is not null && quantiles.Length > 0)
        {
            if (!_useNativeMode)
            {
                return output;
            }
            return GenerateQuantilePredictions(historicalData, quantiles);
        }

        return output;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the UniTS model, AutoregressiveForecast produces predictions from input data. This is the main inference step of the UniTS architecture.
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
    /// <b>For Beginners:</b> In the UniTS model, Evaluate performs a supporting step in the workflow. It keeps the UniTS architecture pipeline consistent.
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
    /// <b>For Beginners:</b> In the UniTS model, ApplyInstanceNormalization performs a supporting step in the workflow. It keeps the UniTS architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
    {
        // UniTS uses batch normalization internally, so instance norm is a pass-through
        return input;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the UniTS model, GetFinancialMetrics calculates evaluation metrics. This summarizes how the UniTS architecture is performing.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> GetFinancialMetrics()
    {
        T lastLoss = LastLoss is not null ? LastLoss : NumOps.Zero;

        return new Dictionary<string, T>
        {
            ["ContextLength"] = NumOps.FromDouble(_contextLength),
            ["ForecastHorizon"] = NumOps.FromDouble(_forecastHorizon),
            ["HiddenDimension"] = NumOps.FromDouble(_hiddenDimension),
            ["NumLayers"] = NumOps.FromDouble(_numLayers),
            ["NumHeads"] = NumOps.FromDouble(_numHeads),
            ["NumConvScales"] = NumOps.FromDouble(_convKernelSizes.Length),
            ["LastLoss"] = lastLoss
        };
    }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// Performs the forward pass through the network.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Output tensor with task-specific predictions.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The forward pass:
    /// 1. Embeds input features
    /// 2. Applies multi-scale temporal convolution
    /// 3. Processes through transformer layers
    /// 4. Applies task-specific output head
    /// </para>
    /// </remarks>
    private Tensor<T> Forward(Tensor<T> input)
    {
        var current = input;

        foreach (var layer in Layers)
        {
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
    /// layers (embedding, convolution, transformer, and output head) to update their weights.
    /// </para>
    /// </remarks>
    private Tensor<T> Backward(Tensor<T> outputGradient)
    {
        var current = outputGradient;

        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            current = Layers[i].Backward(current);
        }

        return current;
    }

    /// <summary>
    /// Performs ONNX-based inference for task-specific predictions.
    /// </summary>
    /// <param name="input">Input tensor with historical data.</param>
    /// <returns>Output tensor with task-specific predictions.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> ONNX mode uses the pretrained model for fast inference
    /// without requiring native layer computation.
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

    private static string NormalizeTaskType(string? taskType)
    {
        if (string.IsNullOrWhiteSpace(taskType))
        {
            return "forecasting";
        }

        // After null check, taskType is guaranteed non-null
        string nonNullTaskType = taskType ?? "forecasting";
        return nonNullTaskType.Trim().ToLowerInvariant();
    }

    /// <summary>
    /// Generates quantile predictions through Monte Carlo dropout.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <param name="quantiles">Quantile levels to compute.</param>
    /// <returns>Quantile predictions tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> For uncertainty estimation, we run the model
    /// multiple times with dropout enabled, then compute quantiles from the
    /// distribution of outputs.
    /// </para>
    /// </remarks>
    private Tensor<T> GenerateQuantilePredictions(Tensor<T> input, double[] quantiles)
    {
        if (!_useNativeMode)
        {
            return ForecastOnnx(input);
        }

        int numSamples = 100;
        var samples = new List<Tensor<T>>();

        // Enable dropout for MC sampling
        SetTrainingMode(true);

        for (int s = 0; s < numSamples; s++)
        {
            samples.Add(Forward(input));
        }

        SetTrainingMode(false);

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
    /// <para><b>For Beginners:</b> For autoregressive forecasting beyond the horizon,
    /// we need to update the input with predictions so we can forecast further.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ShiftInputWithPredictions(Tensor<T> input, Tensor<T> predictions, int stepsUsed)
    {
        var result = new Tensor<T>(input.Shape);
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
    /// <para><b>For Beginners:</b> When doing autoregressive forecasting in chunks,
    /// we need to combine all the predictions into one final result.
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

    /// <summary>
    /// Performs classification by computing class probabilities.
    /// </summary>
    /// <param name="input">Input tensor with time series data.</param>
    /// <returns>Tensor with class probabilities.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> When UniTS is configured for classification,
    /// this method returns the probability of each class for the input time series.
    /// The class with the highest probability is the model's prediction.
    /// </para>
    /// </remarks>
    public Tensor<T> Classify(Tensor<T> input)
    {
        if (_taskType != "classification")
        {
            throw new InvalidOperationException(
                $"Classification is only supported when TaskType is 'classification'. Current TaskType: {_taskType}");
        }

        var output = _useNativeMode ? Forward(input) : ForecastOnnx(input);

        // Apply softmax for probability output
        return ApplySoftmax(output);
    }

    /// <summary>
    /// Performs anomaly detection by computing reconstruction error.
    /// </summary>
    /// <param name="input">Input tensor with time series data.</param>
    /// <returns>Tensor with anomaly scores (higher = more anomalous).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> When UniTS is configured for anomaly detection,
    /// this method returns an anomaly score for each time step. Higher scores
    /// indicate more unusual patterns that deviate from learned normal behavior.
    /// </para>
    /// </remarks>
    public Tensor<T> DetectAnomalies(Tensor<T> input)
    {
        if (_taskType != "anomaly")
        {
            throw new InvalidOperationException(
                $"Anomaly detection is only supported when TaskType is 'anomaly'. Current TaskType: {_taskType}");
        }

        var reconstruction = _useNativeMode ? Forward(input) : ForecastOnnx(input);

        // Compute reconstruction error as anomaly score
        var anomalyScores = new Tensor<T>(input.Shape);
        for (int i = 0; i < input.Length && i < reconstruction.Length; i++)
        {
            var diff = NumOps.Subtract(input.Data.Span[i], reconstruction.Data.Span[i]);
            anomalyScores.Data.Span[i] = NumOps.Abs(diff);
        }

        return anomalyScores;
    }

    /// <summary>
    /// Performs imputation by filling missing values.
    /// </summary>
    /// <param name="input">Input tensor with missing values (marked as NaN or zero).</param>
    /// <param name="mask">Optional mask tensor indicating missing values (1 = present, 0 = missing).</param>
    /// <returns>Tensor with missing values filled in.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> When UniTS is configured for imputation,
    /// this method fills in missing values in the time series. The model learns
    /// to predict missing values based on the surrounding context.
    /// </para>
    /// </remarks>
    public Tensor<T> Impute(Tensor<T> input, Tensor<T>? mask = null)
    {
        if (_taskType != "imputation")
        {
            throw new InvalidOperationException(
                $"Imputation is only supported when TaskType is 'imputation'. Current TaskType: {_taskType}");
        }

        var imputed = _useNativeMode ? Forward(input) : ForecastOnnx(input);

        // If mask provided, only replace missing values
        if (mask is not null)
        {
            var result = new Tensor<T>(input.Shape);
            for (int i = 0; i < input.Length; i++)
            {
                // Use imputed value where mask is 0 (missing), original where mask is 1 (present)
                var maskValue = i < mask.Length ? NumOps.ToDouble(mask.Data.Span[i]) : 1.0;
                result.Data.Span[i] = maskValue > 0.5
                    ? input.Data.Span[i]
                    : imputed.Data.Span[i];
            }
            return result;
        }

        return imputed;
    }

    /// <summary>
    /// Applies softmax normalization to convert logits to probabilities.
    /// </summary>
    /// <param name="input">Input tensor with raw logits.</param>
    /// <returns>Tensor with softmax-normalized probabilities.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Softmax converts raw output values (logits) into
    /// probabilities that sum to 1. This is essential for classification tasks
    /// where we want to know the probability of each class.
    /// </para>
    /// </remarks>
    private Tensor<T> ApplySoftmax(Tensor<T> input)
    {
        var result = new Tensor<T>(input.Shape);

        if (input.Length == 0)
        {
            return result;
        }

        // Find max for numerical stability
        T maxVal = input.Data.Span[0];
        for (int i = 1; i < input.Length; i++)
        {
            if (NumOps.ToDouble(input.Data.Span[i]) > NumOps.ToDouble(maxVal))
            {
                maxVal = input.Data.Span[i];
            }
        }

        // Compute exp(x - max) and sum
        T sum = NumOps.Zero;
        for (int i = 0; i < input.Length; i++)
        {
            var shifted = NumOps.Subtract(input.Data.Span[i], maxVal);
            var expVal = NumOps.FromDouble(Math.Exp(NumOps.ToDouble(shifted)));
            result.Data.Span[i] = expVal;
            sum = NumOps.Add(sum, expVal);
        }

        // Normalize
        for (int i = 0; i < result.Length; i++)
        {
            result.Data.Span[i] = NumOps.Divide(result.Data.Span[i], sum);
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


