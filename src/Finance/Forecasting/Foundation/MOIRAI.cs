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
/// MOIRAI (Masked EncOder-based UnIveRsAl TIme Series Foundation Model) implementation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// MOIRAI is Salesforce's universal time series foundation model that uses multi-scale
/// patching and masked encoder training for any-to-any forecasting. It can handle
/// different time series frequencies and domains without fine-tuning.
/// </para>
/// <para><b>For Beginners:</b> MOIRAI is designed to be truly universal:
///
/// <b>Multi-Scale Patching:</b>
/// Unlike single-patch models, MOIRAI uses multiple patch sizes simultaneously:
/// - Small patches (8 steps): Capture fine-grained, high-frequency patterns
/// - Medium patches (16, 32): Balance detail and context
/// - Large patches (64+): Capture long-term trends and seasonality
///
/// <b>Masked Encoder Architecture:</b>
/// During training, random patches are masked and the model learns to predict them.
/// This is similar to BERT's masked language modeling but for time series.
///
/// <b>Mixture of Distributions:</b>
/// For probabilistic forecasting, MOIRAI outputs a mixture of Gaussian distributions,
/// allowing it to model complex, multi-modal forecast uncertainties.
///
/// <b>Any-to-Any Forecasting:</b>
/// The same model can predict any horizon from any context length, making it
/// flexible for different forecasting scenarios.
/// </para>
/// <para>
/// <b>Reference:</b> Woo et al., "Unified Training of Universal Time Series Forecasting Transformers", 2024.
/// https://arxiv.org/abs/2402.02592
/// </para>
/// </remarks>
public class MOIRAI<T> : ForecastingModelBase<T>
{
    #region Execution Mode

    /// <summary>
    /// Indicates whether the model is running in native mode (true) or ONNX mode (false).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> MOIRAI supports two execution modes:
    /// - Native mode: Full training and inference with gradients
    /// - ONNX mode: Inference-only using pretrained ONNX models
    /// </para>
    /// </remarks>
    private readonly bool _useNativeMode;

    #endregion

    
    #region Native Mode Fields

    /// <summary>
    /// Reference to the multi-scale embedding layer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The embedding layer converts multi-scale patches
    /// into a unified hidden representation that the transformer can process.
    /// </para>
    /// </remarks>
    private DenseLayer<T>? _embeddingLayer;

    /// <summary>
    /// References to the transformer attention layers.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each attention layer allows patches at different
    /// scales to exchange information, learning cross-scale dependencies.
    /// </para>
    /// </remarks>
    private List<MultiHeadAttentionLayer<T>>? _attentionLayers;

    /// <summary>
    /// Reference to the distribution output head.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The output head produces mixture distribution
    /// parameters (weights, means, variances) for probabilistic forecasting.
    /// </para>
    /// </remarks>
    private DenseLayer<T>? _outputHead;

    #endregion

    #region Shared Fields

    /// <summary>
    /// The optimizer used for training the model.
    /// </summary>
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

    /// <summary>
    /// The loss function used for training.
    /// </summary>
    private readonly ILossFunction<T> _lossFunction;

    /// <summary>
    /// Context length for the input sequence.
    /// </summary>
    private int _contextLength;

    /// <summary>
    /// Forecast horizon for predictions.
    /// </summary>
    private int _forecastHorizon;

    /// <summary>
    /// Patch sizes for multi-scale patching.
    /// </summary>
    private int[] _patchSizes;

    /// <summary>
    /// Hidden dimension of the transformer.
    /// </summary>
    private int _hiddenDimension;

    /// <summary>
    /// Number of transformer layers.
    /// </summary>
    private int _numLayers;

    /// <summary>
    /// Number of attention heads.
    /// </summary>
    private int _numHeads;

    /// <summary>
    /// Intermediate size for FFN.
    /// </summary>
    private int _intermediateSize;

    /// <summary>
    /// Number of mixture components.
    /// </summary>
    private int _numMixtures;

    /// <summary>
    /// Dropout rate.
    /// </summary>
    private double _dropout;

    /// <summary>
    /// Mask ratio for training.
    /// </summary>
    private double _maskRatio;

    /// <summary>
    /// Number of input features.
    /// </summary>
    private int _numFeatures;

    /// <summary>
    /// Total number of patches across all scales.
    /// </summary>
    private int _totalPatches;

    /// <summary>
    /// Model size variant.
    /// </summary>
    private string _modelSize;

    #endregion

    #region IForecastingModel Properties

    /// <inheritdoc/>
    public override int SequenceLength => _contextLength;

    /// <inheritdoc/>
    public override int PredictionHorizon => _forecastHorizon;

    /// <inheritdoc/>
    public override int NumFeatures => _numFeatures;

    /// <inheritdoc/>
    public override int PatchSize => _patchSizes[0];

    /// <inheritdoc/>
    public override int Stride => _patchSizes[0];

    /// <inheritdoc/>
    public override bool IsChannelIndependent => true;

    /// <inheritdoc/>
    public override bool UseNativeMode => _useNativeMode;

    /// <summary>
    /// Gets the number of mixture components for distribution output.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> More mixture components can model more complex
    /// distributions but require more computation.
    /// </para>
    /// </remarks>
    public int NumMixtures => _numMixtures;

    /// <summary>
    /// Gets the array of patch sizes used for multi-scale patching.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> MOIRAI uses multiple patch sizes to capture
    /// patterns at different time scales.
    /// </para>
    /// </remarks>
    public int[] PatchSizes => _patchSizes;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance using an ONNX pretrained model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="options">Configuration options for MOIRAI.</param>
    /// <param name="optimizer">Optional optimizer for fine-tuning.</param>
    /// <param name="lossFunction">Optional custom loss function.</param>
    /// <exception cref="ArgumentException">Thrown when onnxModelPath is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown when ONNX model file doesn't exist.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to load a pretrained MOIRAI model
    /// for fast inference. The model uses multi-scale patching for universal forecasting.
    /// </para>
    /// </remarks>
    public MOIRAI(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        MOIRAIOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        options ??= new MOIRAIOptions<T>();

        _useNativeMode = false;
        OnnxModelPath = onnxModelPath;
        OnnxSession = new InferenceSession(onnxModelPath);

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        _contextLength = options.ContextLength;
        _forecastHorizon = options.ForecastHorizon;
        _patchSizes = options.PatchSizes;
        _hiddenDimension = options.HiddenDimension;
        _numLayers = options.NumLayers;
        _numHeads = options.NumHeads;
        _intermediateSize = options.IntermediateSize;
        _numMixtures = options.NumMixtures;
        _dropout = options.DropoutRate;
        _maskRatio = options.MaskRatio;
        _numFeatures = 1;
        _modelSize = options.ModelSize;

        // Calculate total patches
        _totalPatches = 0;
        foreach (var patchSize in _patchSizes)
        {
            _totalPatches += _contextLength / patchSize;
        }

        InitializeLayers();
    }

    /// <summary>
    /// Initializes a new instance in native mode for training.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">Configuration options for MOIRAI.</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="optimizer">Optional optimizer.</param>
    /// <param name="lossFunction">Optional custom loss function.</param>
    /// <exception cref="ArgumentNullException">Thrown when architecture is null.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to train MOIRAI from scratch
    /// or fine-tune on your specific time series data.
    /// </para>
    /// </remarks>
    public MOIRAI(
        NeuralNetworkArchitecture<T> architecture,
        MOIRAIOptions<T>? options = null,
        int numFeatures = 1,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new MOIRAIOptions<T>();

        _useNativeMode = true;

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        _contextLength = options.ContextLength;
        _forecastHorizon = options.ForecastHorizon;
        _patchSizes = options.PatchSizes;
        _hiddenDimension = options.HiddenDimension;
        _numLayers = options.NumLayers;
        _numHeads = options.NumHeads;
        _intermediateSize = options.IntermediateSize;
        _numMixtures = options.NumMixtures;
        _dropout = options.DropoutRate;
        _maskRatio = options.MaskRatio;
        _numFeatures = numFeatures;
        _modelSize = options.ModelSize;

        // Calculate total patches
        _totalPatches = 0;
        foreach (var patchSize in _patchSizes)
        {
            _totalPatches += _contextLength / patchSize;
        }

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes the model layers based on configuration.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method sets up the multi-scale transformer
    /// architecture with all its components including embedding, attention,
    /// and distribution output layers.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultMOIRAILayers(
                Architecture, _contextLength, _forecastHorizon, _numFeatures,
                _patchSizes, _hiddenDimension, _numLayers, _numHeads,
                _intermediateSize, _numMixtures, _dropout));

            ExtractLayerReferences();
        }
    }

    /// <summary>
    /// Extracts references to key layers for efficient access during forward/backward pass.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> After creating all layers, we keep direct references
    /// to important ones so we can access them quickly during computation.
    /// </para>
    /// </remarks>
    private void ExtractLayerReferences()
    {
        _embeddingLayer = Layers.OfType<DenseLayer<T>>().FirstOrDefault();
        _attentionLayers = Layers.OfType<MultiHeadAttentionLayer<T>>().ToList();
        _outputHead = Layers.OfType<DenseLayer<T>>().LastOrDefault();
    }

    /// <summary>
    /// Validates that custom layers meet MOIRAI requirements.
    /// </summary>
    /// <param name="layers">The layers to validate.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> When using custom layers, we ensure they
    /// include the necessary components for multi-scale processing.
    /// </para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);

        var attentionCount = layers.OfType<MultiHeadAttentionLayer<T>>().Count();
        if (attentionCount < 1)
        {
            throw new ArgumentException(
                "MOIRAI requires at least one MultiHeadAttentionLayer for cross-scale attention.");
        }
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <inheritdoc/>
    public override bool SupportsTraining => _useNativeMode;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MOIRAI model, Predict produces predictions from input data. This is the main inference step of the MOIRAI architecture.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return _useNativeMode ? Forward(input) : ForecastOnnx(input);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MOIRAI model, Train performs a training step. This updates the MOIRAI architecture so it learns from data.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is only supported in native mode.");

        SetTrainingMode(true);

        // Forward pass with masking
        var maskedInput = ApplyRandomMasking(input);
        var output = Forward(maskedInput);

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
    /// <b>For Beginners:</b> In the MOIRAI model, UpdateParameters updates internal parameters or state. This keeps the MOIRAI architecture aligned with the latest values.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> gradients)
    {
        // Parameters are updated through the optimizer in Train()
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MOIRAI model, GetModelMetadata performs a supporting step in the workflow. It keeps the MOIRAI architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "MOIRAI" },
                { "ContextLength", _contextLength },
                { "ForecastHorizon", _forecastHorizon },
                { "PatchSizes", string.Join(",", _patchSizes) },
                { "TotalPatches", _totalPatches },
                { "HiddenDimension", _hiddenDimension },
                { "NumLayers", _numLayers },
                { "NumHeads", _numHeads },
                { "NumMixtures", _numMixtures },
                { "ModelSize", _modelSize },
                { "UseNativeMode", _useNativeMode },
                { "ParameterCount", GetParameterCount() }
            },
            ModelData = _useNativeMode ? this.Serialize() : Array.Empty<byte>()
        };
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MOIRAI model, CreateNewInstance builds and wires up model components. This sets up the MOIRAI architecture before use.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var options = new MOIRAIOptions<T>
        {
            ContextLength = _contextLength,
            ForecastHorizon = _forecastHorizon,
            PatchSizes = _patchSizes,
            HiddenDimension = _hiddenDimension,
            NumLayers = _numLayers,
            NumHeads = _numHeads,
            IntermediateSize = _intermediateSize,
            NumMixtures = _numMixtures,
            DropoutRate = _dropout,
            MaskRatio = _maskRatio,
            ModelSize = _modelSize
        };

        return new MOIRAI<T>(Architecture, options, _numFeatures);
    }

    /// <summary>
    /// Writes MOIRAI-specific configuration during serialization.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Saves all the configuration needed to reconstruct this model.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_contextLength);
        writer.Write(_forecastHorizon);
        writer.Write(_patchSizes.Length);
        foreach (var ps in _patchSizes)
        {
            writer.Write(ps);
        }
        writer.Write(_hiddenDimension);
        writer.Write(_numLayers);
        writer.Write(_numHeads);
        writer.Write(_intermediateSize);
        writer.Write(_numMixtures);
        writer.Write(_dropout);
        writer.Write(_maskRatio);
        writer.Write(_numFeatures);
        writer.Write(_modelSize);
    }

    /// <summary>
    /// Reads MOIRAI-specific configuration during deserialization.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Loads the configuration that was saved during serialization.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _contextLength = reader.ReadInt32();
        _forecastHorizon = reader.ReadInt32();
        int patchCount = reader.ReadInt32();
        _patchSizes = new int[patchCount];
        for (int i = 0; i < patchCount; i++)
        {
            _patchSizes[i] = reader.ReadInt32();
        }
        _hiddenDimension = reader.ReadInt32();
        _numLayers = reader.ReadInt32();
        _numHeads = reader.ReadInt32();
        _intermediateSize = reader.ReadInt32();
        _numMixtures = reader.ReadInt32();
        _dropout = reader.ReadDouble();
        _maskRatio = reader.ReadDouble();
        _numFeatures = reader.ReadInt32();
        _modelSize = reader.ReadString();

        _totalPatches = 0;
        foreach (var patchSize in _patchSizes)
        {
            _totalPatches += _contextLength / Math.Max(1, patchSize);
        }
    }

    #endregion

    #region IForecastingModel Implementation

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MOIRAI model, Forecast produces predictions from input data. This is the main inference step of the MOIRAI architecture.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
    {
        var output = _useNativeMode ? Forward(historicalData) : ForecastOnnx(historicalData);

        // Extract point forecasts from mixture distribution parameters
        var pointPredictions = ExtractPointPredictions(output, _forecastHorizon);

        // If quantiles requested, generate samples
        if (quantiles is not null && quantiles.Length > 0)
        {
            return GenerateMixtureQuantiles(output, _forecastHorizon, quantiles);
        }

        return pointPredictions;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MOIRAI model, AutoregressiveForecast produces predictions from input data. This is the main inference step of the MOIRAI architecture.
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
    /// <b>For Beginners:</b> In the MOIRAI model, Evaluate performs a supporting step in the workflow. It keeps the MOIRAI architecture pipeline consistent.
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
    /// <b>For Beginners:</b> In the MOIRAI model, ApplyInstanceNormalization performs a supporting step in the workflow. It keeps the MOIRAI architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
    {
        // MOIRAI handles normalization internally via multi-scale patching
        return input;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the MOIRAI model, GetFinancialMetrics calculates evaluation metrics. This summarizes how the MOIRAI architecture is performing.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> GetFinancialMetrics()
    {
        T lastLoss = LastLoss is not null ? LastLoss : NumOps.Zero;

        return new Dictionary<string, T>
        {
            ["ContextLength"] = NumOps.FromDouble(_contextLength),
            ["ForecastHorizon"] = NumOps.FromDouble(_forecastHorizon),
            ["TotalPatches"] = NumOps.FromDouble(_totalPatches),
            ["HiddenDimension"] = NumOps.FromDouble(_hiddenDimension),
            ["NumLayers"] = NumOps.FromDouble(_numLayers),
            ["NumMixtures"] = NumOps.FromDouble(_numMixtures),
            ["LastLoss"] = lastLoss
        };
    }

    #endregion

    #region Forward/Backward Pass

    /// <summary>
    /// Performs the forward pass through the network.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Output tensor with mixture distribution parameters.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The forward pass processes the input through all layers
    /// sequentially. MOIRAI processes multi-scale patches through the unified encoder,
    /// producing mixture distribution parameters for probabilistic forecasting.
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
    /// <para><b>For Beginners:</b> The backward pass computes gradients by propagating
    /// error signals backwards through the network. This enables learning.
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
    /// Performs ONNX-based inference for forecasting.
    /// </summary>
    /// <param name="input">Input tensor with historical data.</param>
    /// <returns>Forecast tensor with predictions.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> ONNX Runtime provides optimized inference
    /// for pretrained models. This method converts tensors to ONNX format,
    /// runs inference, and converts results back.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ForecastOnnx(Tensor<T> input)
    {
        if (OnnxSession is null)
        {
            throw new InvalidOperationException("ONNX session not initialized.");
        }

        // Convert input to ONNX tensor format
        var inputData = ConvertToFloatArray(input);
        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input.Shape);

        // Get input name from model
        var inputMeta = OnnxSession.InputMetadata;
        var inputName = inputMeta.Keys.First();

        // Run inference
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, onnxInput)
        };

        using var results = OnnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        // Convert back to our tensor type
        return ConvertFromOnnxTensor(outputTensor);
    }

    #endregion

    #region Model-Specific Processing

    /// <summary>
    /// Applies random masking to input for masked encoder training.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Masked input tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> During training, we randomly mask some patches
    /// and train the model to predict them. This self-supervised approach
    /// helps the model learn robust representations of time series patterns.
    /// </para>
    /// </remarks>
    private Tensor<T> ApplyRandomMasking(Tensor<T> input)
    {
        var masked = new Tensor<T>(input.Shape);
        var rand = RandomHelper.CreateSecureRandom();

        // Copy input data
        for (int i = 0; i < input.Length; i++)
        {
            masked.Data.Span[i] = input.Data.Span[i];
        }

        // Apply masking based on mask ratio
        int numPatches = _totalPatches;
        int numToMask = (int)(numPatches * _maskRatio);

        var patchIndices = Enumerable.Range(0, numPatches).ToList();
        for (int i = 0; i < numToMask && patchIndices.Count > 0; i++)
        {
            int idx = rand.Next(patchIndices.Count);
            int patchIdx = patchIndices[idx];
            patchIndices.RemoveAt(idx);

            // Mask this patch (set to zero or learned mask token)
            int patchStart = patchIdx * _hiddenDimension;
            for (int j = 0; j < _hiddenDimension && patchStart + j < masked.Length; j++)
            {
                masked.Data.Span[patchStart + j] = NumOps.Zero;
            }
        }

        return masked;
    }

    /// <summary>
    /// Extracts point predictions from mixture distribution parameters.
    /// </summary>
    /// <param name="mixtureOutput">Output tensor with mixture parameters.</param>
    /// <param name="horizon">Forecast horizon.</param>
    /// <returns>Point predictions tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The mixture output contains weights, means, and variances
    /// for each mixture component. The point prediction is the weighted average of means.
    /// </para>
    /// </remarks>
    private Tensor<T> ExtractPointPredictions(Tensor<T> mixtureOutput, int horizon)
    {
        var result = new Tensor<T>(new[] { 1, horizon, 1 });
        int paramsPerStep = _numMixtures * 3; // weight, mean, variance per mixture

        for (int t = 0; t < horizon; t++)
        {
            T weightedMean = NumOps.Zero;
            T totalWeight = NumOps.Zero;

            // Apply softmax to weights and compute weighted mean
            for (int m = 0; m < _numMixtures; m++)
            {
                int baseIdx = t * paramsPerStep + m * 3;
                if (baseIdx + 1 < mixtureOutput.Length)
                {
                    T weight = mixtureOutput.Data.Span[baseIdx];
                    T mean = mixtureOutput.Data.Span[baseIdx + 1];

                    // Softmax approximation: exp(weight)
                    T expWeight = NumOps.Exp(weight);
                    weightedMean = NumOps.Add(weightedMean, NumOps.Multiply(expWeight, mean));
                    totalWeight = NumOps.Add(totalWeight, expWeight);
                }
            }

            // Normalize
            if (NumOps.GreaterThan(totalWeight, NumOps.Zero))
            {
                weightedMean = NumOps.Divide(weightedMean, totalWeight);
            }

            result.Data.Span[t] = weightedMean;
        }

        return result;
    }

    /// <summary>
    /// Generates quantile predictions from mixture distribution.
    /// </summary>
    /// <param name="mixtureOutput">Output tensor with mixture parameters.</param>
    /// <param name="horizon">Forecast horizon.</param>
    /// <param name="quantiles">Quantile levels to compute.</param>
    /// <returns>Quantile predictions tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Quantiles from a Gaussian mixture are computed by
    /// sampling from the mixture and finding the empirical quantiles.
    /// </para>
    /// </remarks>
    private Tensor<T> GenerateMixtureQuantiles(Tensor<T> mixtureOutput, int horizon, double[] quantiles)
    {
        var result = new Tensor<T>(new[] { 1, horizon, quantiles.Length });
        var rand = RandomHelper.CreateSecureRandom();
        int numSamples = 1000; // Number of samples for quantile estimation
        int paramsPerStep = _numMixtures * 3;

        for (int t = 0; t < horizon; t++)
        {
            // Collect samples from mixture
            var samples = new List<double>();

            // Parse mixture parameters for this timestep
            var weights = new double[_numMixtures];
            var means = new double[_numMixtures];
            var variances = new double[_numMixtures];

            for (int m = 0; m < _numMixtures; m++)
            {
                int baseIdx = t * paramsPerStep + m * 3;
                if (baseIdx + 2 < mixtureOutput.Length)
                {
                    weights[m] = NumOps.ToDouble(mixtureOutput.Data.Span[baseIdx]);
                    means[m] = NumOps.ToDouble(mixtureOutput.Data.Span[baseIdx + 1]);
                    variances[m] = Math.Max(0.01, Math.Exp(NumOps.ToDouble(mixtureOutput.Data.Span[baseIdx + 2])));
                }
            }

            // Softmax weights
            double maxWeight = weights.Max();
            double sumExp = 0;
            for (int m = 0; m < _numMixtures; m++)
            {
                weights[m] = Math.Exp(weights[m] - maxWeight);
                sumExp += weights[m];
            }
            for (int m = 0; m < _numMixtures; m++)
            {
                weights[m] /= sumExp;
            }

            // Sample from mixture
            for (int s = 0; s < numSamples; s++)
            {
                // Select component
                double u = rand.NextDouble();
                double cumWeight = 0;
                int component = _numMixtures - 1;
                for (int m = 0; m < _numMixtures; m++)
                {
                    cumWeight += weights[m];
                    if (u < cumWeight)
                    {
                        component = m;
                        break;
                    }
                }

                // Sample from Gaussian
                double z = SampleStandardNormal(rand);
                double sample = means[component] + Math.Sqrt(variances[component]) * z;
                samples.Add(sample);
            }

            // Sort and extract quantiles
            samples.Sort();
            for (int q = 0; q < quantiles.Length; q++)
            {
                int idx = Math.Min((int)(quantiles[q] * numSamples), numSamples - 1);
                result.Data.Span[t * quantiles.Length + q] = NumOps.FromDouble(samples[idx]);
            }
        }

        return result;
    }

    /// <summary>
    /// Samples from a standard normal distribution using Box-Muller transform.
    /// </summary>
    /// <param name="rand">Random number generator.</param>
    /// <returns>A sample from N(0,1).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The Box-Muller transform converts uniform random
    /// numbers into Gaussian random numbers, which we need for sampling from
    /// the mixture components.
    /// </para>
    /// </remarks>
    private double SampleStandardNormal(Random rand)
    {
        double u1 = 1.0 - rand.NextDouble();
        double u2 = 1.0 - rand.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    /// <summary>
    /// Shifts input tensor by appending predictions and removing oldest values.
    /// </summary>
    /// <param name="input">Original input tensor.</param>
    /// <param name="predictions">Predictions to append.</param>
    /// <param name="stepsUsed">Number of prediction steps to use.</param>
    /// <returns>Shifted input tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> For autoregressive forecasting, we need to update
    /// the input with predictions so we can forecast further into the future.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ShiftInputWithPredictions(Tensor<T> input, Tensor<T> predictions, int stepsUsed)
    {
        var result = new Tensor<T>(input.Shape);
        int contextLen = _contextLength;

        // Shift old values left
        for (int i = 0; i < contextLen - stepsUsed; i++)
        {
            result.Data.Span[i] = input.Data.Span[i + stepsUsed];
        }

        // Append predictions
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

    #endregion

    #region Helper Methods

    /// <summary>
    /// Converts tensor to float array for ONNX compatibility.
    /// </summary>
    /// <param name="tensor">Input tensor.</param>
    /// <returns>Float array.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> ONNX Runtime works with float arrays,
    /// so we need to convert our generic tensor data to floats.
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
    /// Converts ONNX tensor back to our tensor type.
    /// </summary>
    /// <param name="onnxTensor">ONNX tensor.</param>
    /// <returns>Converted tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> After ONNX inference, we convert the results
    /// back to our tensor format for consistent API usage.
    /// </para>
    /// </remarks>
    private Tensor<T> ConvertFromOnnxTensor(OnnxTensors.Tensor<float> onnxTensor)
    {
        var shape = onnxTensor.Dimensions.ToArray();
        var result = new Tensor<T>(shape);

        int i = 0;
        foreach (var val in onnxTensor)
        {
            result.Data.Span[i++] = NumOps.FromDouble(val);
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

