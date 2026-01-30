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
/// Time-LLM (Large Language Model Reprogramming for Time Series) implementation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// Time-LLM repurposes frozen large language models for time series forecasting by
/// learning a reprogramming layer that translates time series into text-like representations
/// that the LLM can understand.
/// </para>
/// <para><b>For Beginners:</b> Time-LLM is a clever way to use powerful language models for time series:
///
/// <b>The Key Insight:</b>
/// LLMs like GPT/LLaMA are amazing at pattern recognition in sequences.
/// Time-LLM asks: "Can we make time series 'speak' the language of LLMs?"
///
/// <b>How It Works:</b>
/// 1. <b>Patch Reprogramming:</b> Convert time series patches into "prompt-like" tokens
/// 2. <b>Text Prototypes:</b> Learn embeddings that bridge numeric and text domains
/// 3. <b>Frozen LLM:</b> The LLM weights stay fixed (no fine-tuning needed)
/// 4. <b>Output Projection:</b> Map LLM output back to forecast values
///
/// <b>Advantages:</b>
/// - Leverages powerful pretrained LLMs without expensive fine-tuning
/// - Works with any LLM backbone (GPT-2, LLaMA, etc.)
/// - Only trains small reprogramming layers
/// - Zero-shot transfer to new domains
/// </para>
/// <para>
/// <b>Reference:</b> Jin et al., "Time-LLM: Time Series Forecasting by Reprogramming Large Language Models", 2024.
/// https://arxiv.org/abs/2310.01728
/// </para>
/// </remarks>
public class TimeLLM<T> : ForecastingModelBase<T>
{
    #region Execution Mode

    /// <summary>
    /// Indicates whether the model is running in native mode (true) or ONNX mode (false).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Time-LLM supports two execution modes:
    /// - Native mode: Training the reprogramming layers (LLM is simulated)
    /// - ONNX mode: Inference using pretrained ONNX model with real frozen LLM
    /// </para>
    /// </remarks>
    private bool _useNativeMode;

    #endregion


    #region Native Mode Fields

    /// <summary>
    /// Reference to the patch embedding layer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The patch embedding layer converts time series
    /// patches into vectors that can be processed by the reprogramming module.
    /// </para>
    /// </remarks>
    private DenseLayer<T>? _patchEmbedding;

    /// <summary>
    /// References to the reprogramming attention layers.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> These layers learn to translate time series
    /// patterns into representations the LLM can understand.
    /// </para>
    /// </remarks>
    private List<MultiHeadAttentionLayer<T>>? _reprogrammingLayers;

    /// <summary>
    /// Reference to the output projection layer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The output projection maps the LLM's output
    /// back to forecast values.
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
    private int _contextLength;

    /// <summary>
    /// Forecast horizon for predictions.
    /// </summary>
    private int _forecastHorizon;

    /// <summary>
    /// Patch length for input segmentation.
    /// </summary>
    private int _patchLength;

    /// <summary>
    /// Patch stride.
    /// </summary>
    private int _patchStride;

    /// <summary>
    /// LLM hidden dimension.
    /// </summary>
    private int _llmDimension;

    /// <summary>
    /// Number of text prototypes.
    /// </summary>
    private int _numPrototypes;

    /// <summary>
    /// Number of reprogramming layers.
    /// </summary>
    private int _numLayers;

    /// <summary>
    /// Number of attention heads.
    /// </summary>
    private int _numHeads;

    /// <summary>
    /// Dropout rate.
    /// </summary>
    private double _dropout;

    /// <summary>
    /// LLM backbone type.
    /// </summary>
    private string _llmBackbone;

    /// <summary>
    /// Number of input features.
    /// </summary>
    private int _numFeatures;

    /// <summary>
    /// Number of patches.
    /// </summary>
    private int _numPatches;

    #endregion

    #region IForecastingModel Properties

    /// <inheritdoc/>
    public override int SequenceLength => _contextLength;

    /// <inheritdoc/>
    public override int PredictionHorizon => _forecastHorizon;

    /// <inheritdoc/>
    public override int NumFeatures => _numFeatures;

    /// <inheritdoc/>
    public override int PatchSize => _patchLength;

    /// <inheritdoc/>
    public override int Stride => _patchStride;

    /// <inheritdoc/>
    public override bool IsChannelIndependent => true;

    /// <inheritdoc/>
    public override bool UseNativeMode => _useNativeMode;

    /// <summary>
    /// Gets the LLM backbone type.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Which pretrained LLM is used as the backbone.
    /// </para>
    /// </remarks>
    public string LLMBackbone => _llmBackbone;

    /// <summary>
    /// Gets the number of text prototypes.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Text prototypes help bridge time series and text domains.
    /// </para>
    /// </remarks>
    public int NumPrototypes => _numPrototypes;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance using an ONNX pretrained model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="options">Configuration options for Time-LLM.</param>
    /// <param name="optimizer">Optional optimizer.</param>
    /// <param name="lossFunction">Optional custom loss function.</param>
    /// <exception cref="ArgumentException">Thrown when onnxModelPath is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown when ONNX model file doesn't exist.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to load a pretrained Time-LLM model
    /// that includes both the reprogramming layers and frozen LLM for inference.
    /// </para>
    /// </remarks>
    public TimeLLM(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        TimeLLMOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        options ??= new TimeLLMOptions<T>();

        _useNativeMode = false;
        OnnxModelPath = onnxModelPath;
        OnnxSession = new InferenceSession(onnxModelPath);

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        _contextLength = options.ContextLength;
        _forecastHorizon = options.ForecastHorizon;
        _patchLength = options.PatchLength;
        _patchStride = options.PatchStride;
        _llmDimension = options.LLMDimension;
        _numPrototypes = options.NumPrototypes;
        _numLayers = options.NumLayers;
        _numHeads = options.NumHeads;
        _dropout = options.DropoutRate;
        _llmBackbone = options.LLMBackbone;
        _numFeatures = 1;

        // Calculate number of patches
        _numPatches = (_contextLength - _patchLength) / _patchStride + 1;

        InitializeLayers();
    }

    /// <summary>
    /// Initializes a new instance in native mode for training.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">Configuration options for Time-LLM.</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="optimizer">Optional optimizer.</param>
    /// <param name="lossFunction">Optional custom loss function.</param>
    /// <exception cref="ArgumentNullException">Thrown when architecture is null.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to train Time-LLM's reprogramming
    /// layers. In native mode, the LLM is simulated with transformer layers.
    /// </para>
    /// </remarks>
    public TimeLLM(
        NeuralNetworkArchitecture<T> architecture,
        TimeLLMOptions<T>? options = null,
        int numFeatures = 1,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new TimeLLMOptions<T>();

        _useNativeMode = true;

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        _contextLength = options.ContextLength;
        _forecastHorizon = options.ForecastHorizon;
        _patchLength = options.PatchLength;
        _patchStride = options.PatchStride;
        _llmDimension = options.LLMDimension;
        _numPrototypes = options.NumPrototypes;
        _numLayers = options.NumLayers;
        _numHeads = options.NumHeads;
        _dropout = options.DropoutRate;
        _llmBackbone = options.LLMBackbone;
        _numFeatures = numFeatures;

        // Calculate number of patches
        _numPatches = (_contextLength - _patchLength) / _patchStride + 1;

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes the model layers based on configuration.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up the reprogramming architecture including
    /// patch embedding, text prototypes, reprogramming transformer, and output projection.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultTimeLLMLayers(
                Architecture, _contextLength, _forecastHorizon, _numFeatures,
                _patchLength, _patchStride, _llmDimension, _numPrototypes,
                _numLayers, _numHeads, _dropout));

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
        _patchEmbedding = Layers.OfType<DenseLayer<T>>().FirstOrDefault();
        _reprogrammingLayers = Layers.OfType<MultiHeadAttentionLayer<T>>().ToList();
        _outputProjection = Layers.OfType<DenseLayer<T>>().LastOrDefault();
    }

    /// <summary>
    /// Validates that custom layers meet Time-LLM requirements.
    /// </summary>
    /// <param name="layers">The layers to validate.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> When using custom layers, we ensure they include
    /// the necessary components for the reprogramming architecture.
    /// </para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);

        var attentionCount = layers.OfType<MultiHeadAttentionLayer<T>>().Count();
        if (attentionCount < 1)
        {
            throw new ArgumentException(
                "Time-LLM requires at least one MultiHeadAttentionLayer for reprogramming.");
        }
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <inheritdoc/>
    public override bool SupportsTraining => _useNativeMode;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TimeLLM model, Predict produces predictions from input data. This is the main inference step of the TimeLLM architecture.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return _useNativeMode ? Forward(input) : ForecastOnnx(input);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TimeLLM model, Train performs a training step. This updates the TimeLLM architecture so it learns from data.
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
    /// <b>For Beginners:</b> In the TimeLLM model, UpdateParameters updates internal parameters or state. This keeps the TimeLLM architecture aligned with the latest values.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> gradients)
    {
        // Parameters are updated through the optimizer in Train()
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TimeLLM model, GetModelMetadata performs a supporting step in the workflow. It keeps the TimeLLM architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "TimeLLM" },
                { "ContextLength", _contextLength },
                { "ForecastHorizon", _forecastHorizon },
                { "PatchLength", _patchLength },
                { "PatchStride", _patchStride },
                { "LLMDimension", _llmDimension },
                { "NumPrototypes", _numPrototypes },
                { "NumLayers", _numLayers },
                { "NumHeads", _numHeads },
                { "LLMBackbone", _llmBackbone },
                { "UseNativeMode", _useNativeMode },
                { "ParameterCount", GetParameterCount() }
            },
            ModelData = _useNativeMode ? this.Serialize() : Array.Empty<byte>()
        };
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TimeLLM model, CreateNewInstance builds and wires up model components. This sets up the TimeLLM architecture before use.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var options = new TimeLLMOptions<T>
        {
            ContextLength = _contextLength,
            ForecastHorizon = _forecastHorizon,
            PatchLength = _patchLength,
            PatchStride = _patchStride,
            LLMDimension = _llmDimension,
            NumPrototypes = _numPrototypes,
            NumLayers = _numLayers,
            NumHeads = _numHeads,
            DropoutRate = _dropout,
            LLMBackbone = _llmBackbone
        };

        return new TimeLLM<T>(Architecture, options, _numFeatures);
    }

    /// <summary>
    /// Writes Time-LLM-specific configuration during serialization.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Saves all the configuration needed to reconstruct this model.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_contextLength);
        writer.Write(_forecastHorizon);
        writer.Write(_patchLength);
        writer.Write(_patchStride);
        writer.Write(_llmDimension);
        writer.Write(_numPrototypes);
        writer.Write(_numLayers);
        writer.Write(_numHeads);
        writer.Write(_dropout);
        writer.Write(_numFeatures);
        writer.Write(_llmBackbone);
    }

    /// <summary>
    /// Reads Time-LLM-specific configuration during deserialization.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Loads the configuration that was saved during serialization.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _contextLength = reader.ReadInt32();
        _forecastHorizon = reader.ReadInt32();
        _patchLength = reader.ReadInt32();
        _patchStride = reader.ReadInt32();
        _llmDimension = reader.ReadInt32();
        _numPrototypes = reader.ReadInt32();
        _numLayers = reader.ReadInt32();
        _numHeads = reader.ReadInt32();
        _dropout = reader.ReadDouble();
        _numFeatures = reader.ReadInt32();
        _llmBackbone = reader.ReadString();
    }

    #endregion

    #region IForecastingModel Implementation

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TimeLLM model, Forecast produces predictions from input data. This is the main inference step of the TimeLLM architecture.
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
    /// <b>For Beginners:</b> In the TimeLLM model, AutoregressiveForecast produces predictions from input data. This is the main inference step of the TimeLLM architecture.
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
    /// <b>For Beginners:</b> In the TimeLLM model, Evaluate performs a supporting step in the workflow. It keeps the TimeLLM architecture pipeline consistent.
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
    /// <b>For Beginners:</b> In the TimeLLM model, ApplyInstanceNormalization performs a supporting step in the workflow. It keeps the TimeLLM architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
    {
        // Time-LLM handles normalization through the patch embedding
        return input;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TimeLLM model, GetFinancialMetrics calculates evaluation metrics. This summarizes how the TimeLLM architecture is performing.
    /// </para>
    /// </remarks>
    public override Dictionary<string, T> GetFinancialMetrics()
    {
        T lastLoss = LastLoss is not null ? LastLoss : NumOps.Zero;

        return new Dictionary<string, T>
        {
            ["ContextLength"] = NumOps.FromDouble(_contextLength),
            ["ForecastHorizon"] = NumOps.FromDouble(_forecastHorizon),
            ["PatchLength"] = NumOps.FromDouble(_patchLength),
            ["NumPatches"] = NumOps.FromDouble(_numPatches),
            ["LLMDimension"] = NumOps.FromDouble(_llmDimension),
            ["NumPrototypes"] = NumOps.FromDouble(_numPrototypes),
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
    /// 1. Embeds input patches
    /// 2. Applies reprogramming transformation
    /// 3. Processes through simulated LLM layers
    /// 4. Projects to forecast values
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
    /// <para><b>For Beginners:</b> The backward pass computes gradients only for
    /// the reprogramming and projection layers. The simulated LLM layers are
    /// also trainable in native mode (unlike the real frozen LLM in ONNX mode).
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
    /// <para><b>For Beginners:</b> ONNX mode uses the pretrained model which includes
    /// both the trained reprogramming layers and the frozen LLM backbone.
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
        int batchSize = input.Shape.Length > 1 ? input.Shape[0] : 1;
        int contextLen = input.Shape.Length > 1 ? input.Shape[1] : _contextLength;
        int features = input.Shape.Length > 2 ? input.Shape[2] : 1;
        int steps = Math.Min(stepsUsed, contextLen);

        var result = new Tensor<T>(input.Shape);

        int predSteps = predictions.Shape.Length > 1 ? predictions.Shape[1] : predictions.Length / Math.Max(1, batchSize);
        int predFeatures = predictions.Shape.Length > 2 ? predictions.Shape[2] : 1;
        int featureCopy = Math.Min(features, predFeatures);

        for (int b = 0; b < batchSize; b++)
        {
            for (int t = 0; t < contextLen - steps; t++)
            {
                for (int f = 0; f < features; f++)
                {
                    int srcIdx = b * contextLen * features + (t + steps) * features + f;
                    int dstIdx = b * contextLen * features + t * features + f;
                    if (srcIdx < input.Length && dstIdx < result.Length)
                    {
                        result.Data.Span[dstIdx] = input.Data.Span[srcIdx];
                    }
                }
            }

            for (int t = 0; t < steps && t < predSteps; t++)
            {
                for (int f = 0; f < featureCopy; f++)
                {
                    int predIdx = predFeatures > 1
                        ? b * predSteps * predFeatures + t * predFeatures + f
                        : b * predSteps + t;
                    int dstIdx = b * contextLen * features + (contextLen - steps + t) * features + f;
                    if (predIdx < predictions.Length && dstIdx < result.Length)
                    {
                        result.Data.Span[dstIdx] = predictions.Data.Span[predIdx];
                    }
                }
            }
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
        if (predictions.Count == 0)
        {
            return new Tensor<T>(new[] { 1, totalSteps });
        }

        int batchSize = predictions[0].Shape.Length > 1 ? predictions[0].Shape[0] : 1;
        int predFeatures = predictions[0].Shape.Length > 2 ? predictions[0].Shape[2] : 1;
        bool hasFeatureDim = predictions[0].Shape.Length > 2;
        var resultShape = hasFeatureDim
            ? new[] { batchSize, totalSteps, predFeatures }
            : new[] { batchSize, totalSteps };

        var result = new Tensor<T>(resultShape);
        int position = 0;

        foreach (var pred in predictions)
        {
            int predSteps = pred.Shape.Length > 1 ? pred.Shape[1] : pred.Length / Math.Max(1, batchSize);
            int toCopy = Math.Min(predSteps, totalSteps - position);

            for (int b = 0; b < batchSize; b++)
            {
                for (int t = 0; t < toCopy; t++)
                {
                    if (hasFeatureDim)
                    {
                        for (int f = 0; f < predFeatures; f++)
                        {
                            int srcIdx = b * predSteps * predFeatures + t * predFeatures + f;
                            int dstIdx = b * totalSteps * predFeatures + (position + t) * predFeatures + f;
                            if (srcIdx < pred.Length && dstIdx < result.Length)
                            {
                                result.Data.Span[dstIdx] = pred.Data.Span[srcIdx];
                            }
                        }
                    }
                    else
                    {
                        int srcIdx = b * predSteps + t;
                        int dstIdx = b * totalSteps + position + t;
                        if (srcIdx < pred.Length && dstIdx < result.Length)
                        {
                            result.Data.Span[dstIdx] = pred.Data.Span[srcIdx];
                        }
                    }
                }
            }

            position += toCopy;
            if (position >= totalSteps)
                break;
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

