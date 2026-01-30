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
/// TimeGPT-style time series foundation model implementation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// TimeGPT represents a GPT-style architecture adapted for time series forecasting,
/// featuring large-scale pre-training on diverse time series data with zero-shot
/// and few-shot forecasting capabilities.
/// </para>
/// <para><b>For Beginners:</b> TimeGPT brings GPT-like capabilities to time series:
///
/// <b>The Key Insight:</b>
/// Just as GPT was trained on internet-scale text data to become a general-purpose
/// language model, TimeGPT is trained on millions of diverse time series to become
/// a general-purpose forecasting model.
///
/// <b>Core Features:</b>
/// 1. <b>Large-scale Pre-training:</b> Trained on millions of time series
/// 2. <b>Zero-shot Forecasting:</b> No training needed for new data
/// 3. <b>Uncertainty Quantification:</b> Provides prediction intervals
/// 4. <b>Multi-horizon:</b> Forecasts at any horizon
///
/// <b>Advantages:</b>
/// - Works out-of-the-box on new time series
/// - No hyperparameter tuning required
/// - Handles various frequencies and domains
/// - Production-ready forecasting
/// </para>
/// <para>
/// <b>Reference:</b> Garza et al., "TimeGPT-1", 2023.
/// https://arxiv.org/abs/2310.03589
/// </para>
/// </remarks>
public class TimeGPT<T> : ForecastingModelBase<T>
{
    #region Execution Mode

    /// <summary>
    /// Indicates whether the model is running in native mode (true) or ONNX mode (false).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> TimeGPT supports two execution modes:
    /// - Native mode: Fine-tune or train from scratch
    /// - ONNX mode: Use pretrained foundation model
    /// </para>
    /// </remarks>
    private bool _useNativeMode;

    #endregion


    #region Native Mode Fields

    /// <summary>
    /// Reference to the input embedding layer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Projects raw time series values to the hidden dimension.
    /// </para>
    /// </remarks>
    private DenseLayer<T>? _inputEmbedding;

    /// <summary>
    /// References to the transformer layers.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The large transformer backbone that captures
    /// temporal patterns across the sequence.
    /// </para>
    /// </remarks>
    private List<MultiHeadAttentionLayer<T>>? _transformerLayers;

    /// <summary>
    /// Reference to the output projection layer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Projects the transformer output to forecast values.
    /// </para>
    /// </remarks>
    private DenseLayer<T>? _outputProjection;

    #endregion

    #region Shared Fields

    /// <summary>
    /// The optimizer used for training/fine-tuning.
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
    /// Hidden dimension size.
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
    /// Dropout rate.
    /// </summary>
    private double _dropout;

    /// <summary>
    /// Whether to use conformal prediction for uncertainty.
    /// </summary>
    private bool _useConformalPrediction;

    /// <summary>
    /// Confidence level for prediction intervals.
    /// </summary>
    private double _confidenceLevel;

    /// <summary>
    /// Number of fine-tuning steps.
    /// </summary>
    private int _fineTuningSteps;

    /// <summary>
    /// Learning rate for fine-tuning.
    /// </summary>
    private double _fineTuningLearningRate;

    /// <summary>
    /// Number of input features.
    /// </summary>
    private int _numFeatures;

    /// <summary>
    /// Calibration residuals for conformal prediction.
    /// </summary>
    private List<double>? _calibrationResiduals;

    #endregion

    #region IForecastingModel Properties

    /// <inheritdoc/>
    public override int SequenceLength => _contextLength;

    /// <inheritdoc/>
    public override int PredictionHorizon => _forecastHorizon;

    /// <inheritdoc/>
    public override int NumFeatures => _numFeatures;

    /// <inheritdoc/>
    public override int PatchSize => 1; // TimeGPT operates on raw values, not patches

    /// <inheritdoc/>
    public override int Stride => 1;

    /// <inheritdoc/>
    public override bool IsChannelIndependent => true;

    /// <inheritdoc/>
    public override bool UseNativeMode => _useNativeMode;

    /// <summary>
    /// Gets whether conformal prediction is used for uncertainty.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Conformal prediction provides calibrated uncertainty intervals.
    /// </para>
    /// </remarks>
    public bool UseConformalPrediction => _useConformalPrediction;

    /// <summary>
    /// Gets the confidence level for prediction intervals.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The probability that the true value falls within the interval.
    /// </para>
    /// </remarks>
    public double ConfidenceLevel => _confidenceLevel;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance using an ONNX pretrained model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the ONNX model file.</param>
    /// <param name="options">Configuration options for TimeGPT.</param>
    /// <param name="optimizer">Optional optimizer.</param>
    /// <param name="lossFunction">Optional custom loss function.</param>
    /// <exception cref="ArgumentException">Thrown when onnxModelPath is null or empty.</exception>
    /// <exception cref="FileNotFoundException">Thrown when ONNX model file doesn't exist.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to load a pretrained TimeGPT model.
    /// This is the recommended way to use TimeGPT for zero-shot forecasting.
    /// </para>
    /// </remarks>
    public TimeGPT(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        TimeGPTOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        options ??= new TimeGPTOptions<T>();

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
        _dropout = options.DropoutRate;
        _useConformalPrediction = options.UseConformalPrediction;
        _confidenceLevel = options.ConfidenceLevel;
        _fineTuningSteps = options.FineTuningSteps;
        _fineTuningLearningRate = options.FineTuningLearningRate;
        _numFeatures = 1;

        InitializeLayers();
    }

    /// <summary>
    /// Initializes a new instance in native mode for training or fine-tuning.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">Configuration options for TimeGPT.</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="optimizer">Optional optimizer.</param>
    /// <param name="lossFunction">Optional custom loss function.</param>
    /// <exception cref="ArgumentNullException">Thrown when architecture is null.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to train TimeGPT from scratch
    /// or fine-tune on your specific domain data.
    /// </para>
    /// </remarks>
    public TimeGPT(
        NeuralNetworkArchitecture<T> architecture,
        TimeGPTOptions<T>? options = null,
        int numFeatures = 1,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), 1.0)
    {
        options ??= new TimeGPTOptions<T>();

        _useNativeMode = true;

        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();

        _contextLength = options.ContextLength;
        _forecastHorizon = options.ForecastHorizon;
        _hiddenDimension = options.HiddenDimension;
        _numLayers = options.NumLayers;
        _numHeads = options.NumHeads;
        _dropout = options.DropoutRate;
        _useConformalPrediction = options.UseConformalPrediction;
        _confidenceLevel = options.ConfidenceLevel;
        _fineTuningSteps = options.FineTuningSteps;
        _fineTuningLearningRate = options.FineTuningLearningRate;
        _numFeatures = numFeatures;

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes the model layers based on configuration.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This sets up the TimeGPT architecture including
    /// input embedding, large transformer backbone, and output projection.
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultTimeGPTLayers(
                Architecture, _contextLength, _forecastHorizon, _numFeatures,
                _hiddenDimension, _numLayers, _numHeads, _dropout));

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
        _transformerLayers = Layers.OfType<MultiHeadAttentionLayer<T>>().ToList();
        _outputProjection = Layers.OfType<DenseLayer<T>>().LastOrDefault();
    }

    /// <summary>
    /// Validates that custom layers meet TimeGPT requirements.
    /// </summary>
    /// <param name="layers">The layers to validate.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> When using custom layers, we ensure they include
    /// the necessary components for the TimeGPT architecture.
    /// </para>
    /// </remarks>
    protected override void ValidateCustomLayers(List<ILayer<T>> layers)
    {
        base.ValidateCustomLayers(layers);

        var attentionCount = layers.OfType<MultiHeadAttentionLayer<T>>().Count();
        if (attentionCount < 1)
        {
            throw new ArgumentException(
                "TimeGPT requires at least one MultiHeadAttentionLayer for the transformer backbone.");
        }
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <inheritdoc/>
    public override bool SupportsTraining => _useNativeMode;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TimeGPT model, Predict produces predictions from input data. This is the main inference step of the TimeGPT architecture.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return _useNativeMode ? Forward(input) : ForecastOnnx(input);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TimeGPT model, Train performs a training step. This updates the TimeGPT architecture so it learns from data.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is only supported in native mode.");

        SetTrainingMode(true);
        try
        {
            var output = Forward(input);

            // Compute loss
            LastLoss = _lossFunction.CalculateLoss(output.ToVector(), target.ToVector());

            // Backward pass
            var gradient = _lossFunction.CalculateDerivative(output.ToVector(), target.ToVector());
            Backward(Tensor<T>.FromVector(gradient, output.Shape));

            _optimizer.UpdateParameters(Layers);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TimeGPT model, UpdateParameters updates internal parameters or state. This keeps the TimeGPT architecture aligned with the latest values.
    /// </para>
    /// </remarks>
    public override void UpdateParameters(Vector<T> gradients)
    {
        // Parameters are updated through the optimizer in Train()
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TimeGPT model, GetModelMetadata performs a supporting step in the workflow. It keeps the TimeGPT architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NetworkType", "TimeGPT" },
                { "ContextLength", _contextLength },
                { "ForecastHorizon", _forecastHorizon },
                { "HiddenDimension", _hiddenDimension },
                { "NumLayers", _numLayers },
                { "NumHeads", _numHeads },
                { "UseConformalPrediction", _useConformalPrediction },
                { "ConfidenceLevel", _confidenceLevel },
                { "UseNativeMode", _useNativeMode },
                { "ParameterCount", GetParameterCount() }
            },
            ModelData = _useNativeMode ? this.Serialize() : Array.Empty<byte>()
        };
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TimeGPT model, CreateNewInstance builds and wires up model components. This sets up the TimeGPT architecture before use.
    /// </para>
    /// </remarks>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        var options = new TimeGPTOptions<T>
        {
            ContextLength = _contextLength,
            ForecastHorizon = _forecastHorizon,
            HiddenDimension = _hiddenDimension,
            NumLayers = _numLayers,
            NumHeads = _numHeads,
            DropoutRate = _dropout,
            UseConformalPrediction = _useConformalPrediction,
            ConfidenceLevel = _confidenceLevel,
            FineTuningSteps = _fineTuningSteps,
            FineTuningLearningRate = _fineTuningLearningRate
        };

        return new TimeGPT<T>(Architecture, options, _numFeatures);
    }

    /// <summary>
    /// Writes TimeGPT-specific configuration during serialization.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Saves all the configuration needed to reconstruct this model.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_useNativeMode);
        writer.Write(_contextLength);
        writer.Write(_forecastHorizon);
        writer.Write(_hiddenDimension);
        writer.Write(_numLayers);
        writer.Write(_numHeads);
        writer.Write(_dropout);
        writer.Write(_useConformalPrediction);
        writer.Write(_confidenceLevel);
        writer.Write(_fineTuningSteps);
        writer.Write(_fineTuningLearningRate);
        writer.Write(_numFeatures);

        // Save calibration residuals if available
        int residualCount = _calibrationResiduals?.Count ?? 0;
        writer.Write(residualCount);
        if (_calibrationResiduals is not null)
        {
            foreach (var residual in _calibrationResiduals)
            {
                writer.Write(residual);
            }
        }
    }

    /// <summary>
    /// Reads TimeGPT-specific configuration during deserialization.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Loads the configuration that was saved during serialization.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _useNativeMode = reader.ReadBoolean();
        _contextLength = reader.ReadInt32();
        _forecastHorizon = reader.ReadInt32();
        _hiddenDimension = reader.ReadInt32();
        _numLayers = reader.ReadInt32();
        _numHeads = reader.ReadInt32();
        _dropout = reader.ReadDouble();
        _useConformalPrediction = reader.ReadBoolean();
        _confidenceLevel = reader.ReadDouble();
        _fineTuningSteps = reader.ReadInt32();
        _fineTuningLearningRate = reader.ReadDouble();
        _numFeatures = reader.ReadInt32();

        int residualCount = reader.ReadInt32();
        if (residualCount > 0)
        {
            _calibrationResiduals = new List<double>(residualCount);
            for (int i = 0; i < residualCount; i++)
            {
                _calibrationResiduals.Add(reader.ReadDouble());
            }
        }
    }

    #endregion

    #region IForecastingModel Implementation

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TimeGPT model, Forecast produces predictions from input data. This is the main inference step of the TimeGPT architecture.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forecast(Tensor<T> historicalData, double[]? quantiles = null)
    {
        var output = _useNativeMode ? Forward(historicalData) : ForecastOnnx(historicalData);

        // For quantile forecasts with conformal prediction
        if (quantiles is not null && quantiles.Length > 0)
        {
            if (_useConformalPrediction && _calibrationResiduals is not null)
            {
                return GenerateConformalIntervals(output, quantiles);
            }
            else if (_useNativeMode)
            {
                // Quantile generation via MC dropout only works in native mode
                return GenerateQuantilePredictions(historicalData, quantiles);
            }
            // In ONNX mode without conformal prediction, return point forecast
            // (quantile estimation requires native mode dropout sampling)
        }

        return output;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TimeGPT model, AutoregressiveForecast produces predictions from input data. This is the main inference step of the TimeGPT architecture.
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
    /// <b>For Beginners:</b> In the TimeGPT model, Evaluate performs a supporting step in the workflow. It keeps the TimeGPT architecture pipeline consistent.
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
    /// <b>For Beginners:</b> In the TimeGPT model, ApplyInstanceNormalization performs a supporting step in the workflow. It keeps the TimeGPT architecture pipeline consistent.
    /// </para>
    /// </remarks>
    public override Tensor<T> ApplyInstanceNormalization(Tensor<T> input)
    {
        // TimeGPT typically normalizes through the embedding layer
        return input;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> In the TimeGPT model, GetFinancialMetrics calculates evaluation metrics. This summarizes how the TimeGPT architecture is performing.
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
            ["ConfidenceLevel"] = NumOps.FromDouble(_confidenceLevel),
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
    /// 1. Embeds input time series
    /// 2. Processes through large transformer backbone
    /// 3. Projects to forecast values
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
    /// parameters to update them during fine-tuning.
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
    /// <para><b>For Beginners:</b> ONNX mode uses the pretrained TimeGPT model for
    /// zero-shot forecasting without requiring any training.
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
    /// Calibrates the model using historical data for conformal prediction.
    /// </summary>
    /// <param name="historicalData">Historical time series data.</param>
    /// <param name="actuals">Actual values to compare against.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Conformal prediction needs to be calibrated on
    /// historical data to provide statistically valid prediction intervals.
    /// The calibration computes residuals that are used to determine interval widths.
    /// </para>
    /// </remarks>
    public void CalibrateConformalPrediction(List<Tensor<T>> historicalData, List<Tensor<T>> actuals)
    {
        if (!_useConformalPrediction)
            return;

        _calibrationResiduals = new List<double>();

        for (int i = 0; i < historicalData.Count && i < actuals.Count; i++)
        {
            var prediction = Predict(historicalData[i]);
            var actual = actuals[i];

            for (int j = 0; j < prediction.Length && j < actual.Length; j++)
            {
                double pred = NumOps.ToDouble(prediction.Data.Span[j]);
                double act = NumOps.ToDouble(actual.Data.Span[j]);
                _calibrationResiduals.Add(Math.Abs(pred - act));
            }
        }

        // Sort residuals for quantile computation
        _calibrationResiduals.Sort();
    }

    /// <summary>
    /// Generates prediction intervals using conformal prediction.
    /// </summary>
    /// <param name="pointForecast">Point forecast tensor.</param>
    /// <param name="quantiles">Quantile levels to compute.</param>
    /// <returns>Tensor with prediction intervals.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Conformal prediction provides calibrated intervals
    /// that are guaranteed to contain the true value with the specified probability.
    /// </para>
    /// </remarks>
    private Tensor<T> GenerateConformalIntervals(Tensor<T> pointForecast, double[] quantiles)
    {
        if (_calibrationResiduals is null || _calibrationResiduals.Count == 0)
        {
            if (!_useNativeMode)
            {
                // Cannot generate quantile predictions in ONNX mode without calibration data
                // Return point forecast replicated across quantiles
                var fallbackResult = new Tensor<T>(new[] { 1, _forecastHorizon, quantiles.Length });
                for (int t = 0; t < _forecastHorizon && t < pointForecast.Length; t++)
                {
                    for (int q = 0; q < quantiles.Length; q++)
                    {
                        fallbackResult.Data.Span[t * quantiles.Length + q] = pointForecast.Data.Span[t];
                    }
                }
                return fallbackResult;
            }
            // Fall back to MC dropout quantile generation in native mode
            return GenerateQuantilePredictions(pointForecast, quantiles);
        }

        var result = new Tensor<T>(new[] { 1, _forecastHorizon, quantiles.Length });

        for (int t = 0; t < _forecastHorizon && t < pointForecast.Length; t++)
        {
            double center = NumOps.ToDouble(pointForecast.Data.Span[t]);

            for (int q = 0; q < quantiles.Length; q++)
            {
                // Get conformal width for this quantile
                int idx = (int)(quantiles[q] * _calibrationResiduals.Count);
                idx = Math.Min(idx, _calibrationResiduals.Count - 1);
                double width = _calibrationResiduals[idx];

                // For symmetric intervals, adjust based on quantile
                double offset = (quantiles[q] - 0.5) * 2 * width;
                result.Data.Span[t * quantiles.Length + q] = NumOps.FromDouble(center + offset);
            }
        }

        return result;
    }

    /// <summary>
    /// Generates quantile predictions through dropout-based sampling.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <param name="quantiles">Quantile levels to compute.</param>
    /// <returns>Quantile predictions tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> When conformal prediction is not available,
    /// we use Monte Carlo dropout to generate diverse forecasts and compute quantiles.
    /// </para>
    /// </remarks>
    private Tensor<T> GenerateQuantilePredictions(Tensor<T> input, double[] quantiles)
    {
        int numSamples = 100;
        var samples = new List<Tensor<T>>();

        // Enable dropout for MC sampling
        SetTrainingMode(true);
        try
        {
            for (int s = 0; s < numSamples; s++)
            {
                samples.Add(Forward(input));
            }
        }
        finally
        {
            SetTrainingMode(false);
        }

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

            // Guard against empty values list - use default value if no samples available
            if (values.Count == 0)
            {
                for (int q = 0; q < quantiles.Length; q++)
                {
                    result.Data.Span[t * quantiles.Length + q] = NumOps.Zero;
                }
                continue;
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
    /// Fine-tunes the model on domain-specific data.
    /// </summary>
    /// <param name="trainingData">Training input data.</param>
    /// <param name="targets">Target values.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> While TimeGPT works zero-shot, fine-tuning on your
    /// specific domain data can improve accuracy. Only a few hundred steps are typically needed.
    /// </para>
    /// </remarks>
    public void FineTune(List<Tensor<T>> trainingData, List<Tensor<T>> targets)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Fine-tuning is only supported in native mode.");

        if (trainingData is null)
            throw new ArgumentNullException(nameof(trainingData));
        if (targets is null)
            throw new ArgumentNullException(nameof(targets));
        if (trainingData.Count == 0 || targets.Count == 0)
            throw new ArgumentException("Fine-tuning requires non-empty training data and targets.");
        if (trainingData.Count != targets.Count)
            throw new ArgumentException("Training data and targets must have the same number of samples.");

        if (_fineTuningSteps <= 0)
            return;

        var rand = RandomHelper.CreateSecureRandom();

        for (int step = 0; step < _fineTuningSteps; step++)
        {
            // Random sample from training data
            int idx = rand.Next(trainingData.Count);
            var input = trainingData[idx];
            var target = targets[idx];

            // Training step with reduced learning rate
            Train(input, target);
        }
    }

    /// <summary>
    /// Shifts input tensor by appending predictions.
    /// </summary>
    /// <param name="input">Original input tensor.</param>
    /// <param name="predictions">Predictions to append.</param>
    /// <param name="stepsUsed">Number of prediction steps to use.</param>
    /// <returns>Shifted input tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> For multi-step forecasting, we need to update
    /// the input with predictions so we can forecast further.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ShiftInputWithPredictions(Tensor<T> input, Tensor<T> predictions, int stepsUsed)
    {
        var result = new Tensor<T>(input.Shape);
        // Use effective context length based on actual input size
        int effectiveContext = Math.Min(_contextLength, input.Length);
        int steps = Math.Min(stepsUsed, effectiveContext);

        // Shift old values left - guard against input.Length < _contextLength
        for (int i = 0; i < effectiveContext - steps && i + steps < input.Length; i++)
        {
            result.Data.Span[i] = input.Data.Span[i + steps];
        }

        // Append predictions - guard against result and predictions bounds
        for (int i = 0; i < steps && i < predictions.Length; i++)
        {
            int targetIdx = effectiveContext - steps + i;
            if (targetIdx >= 0 && targetIdx < result.Length)
            {
                result.Data.Span[targetIdx] = predictions.Data.Span[i];
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
    /// <para><b>For Beginners:</b> When doing forecasting in chunks, we combine
    /// all predictions into one final result.
    /// </para>
    /// </remarks>
    protected override Tensor<T> ConcatenatePredictions(List<Tensor<T>> predictions, int totalSteps)
    {
        if (predictions.Count == 0)
        {
            return new Tensor<T>(new[] { 1, totalSteps, _numFeatures });
        }

        // Determine dimensions from first prediction
        var first = predictions[0];
        int batchSize = first.Shape.Length > 0 ? first.Shape[0] : 1;
        int numFeatures = first.Shape.Length > 2 ? first.Shape[2] : _numFeatures;

        var result = new Tensor<T>(new[] { batchSize, totalSteps, numFeatures });
        int currentStep = 0;

        foreach (var pred in predictions)
        {
            int predSteps = pred.Shape.Length > 1 ? pred.Shape[1] : pred.Length / (batchSize * numFeatures);
            int stepsToCopy = Math.Min(predSteps, totalSteps - currentStep);

            for (int b = 0; b < batchSize; b++)
            {
                for (int t = 0; t < stepsToCopy; t++)
                {
                    for (int f = 0; f < numFeatures; f++)
                    {
                        int srcIdx = b * predSteps * numFeatures + t * numFeatures + f;
                        int dstIdx = b * totalSteps * numFeatures + (currentStep + t) * numFeatures + f;
                        if (srcIdx < pred.Length && dstIdx < result.Length)
                        {
                            result.Data.Span[dstIdx] = pred.Data.Span[srcIdx];
                        }
                    }
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

