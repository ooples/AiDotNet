using System.IO;
using AiDotNet.Autodiff;
using AiDotNet.Finance.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.Finance.Base;

/// <summary>
/// Base class for all financial AI models, providing dual ONNX/native mode support.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// This abstract class provides the core infrastructure for financial models, following the
/// BLIP-2/RealESRGAN dual-mode pattern from the broader AiDotNet library. It supports both:
/// </para>
/// <para>
/// <b>Native Mode:</b> Full training capabilities using pure C# neural network layers.
/// Use the native constructor when you need to train models from scratch or fine-tune.
/// </para>
/// <para>
/// <b>ONNX Mode:</b> Fast inference using pretrained ONNX models.
/// Use the ONNX constructor for production deployment with pretrained models.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of this as a "foundation" class that all financial models build upon.
///
/// It provides common functionality that every financial model needs:
/// - Making predictions (inference)
/// - Training on data (learning)
/// - Saving/loading models (persistence)
/// - Computing gradients (for optimization)
/// - Integration with the AiDotNet ecosystem
///
/// The dual-mode design means you can choose:
/// - Native mode: More control, can train, uses more memory
/// - ONNX mode: Faster inference, pretrained, read-only
/// </para>
/// </remarks>
public abstract class FinancialModelBase<T> : NeuralNetworkBase<T>, IFinancialModel<T>
{
    #region Execution Mode

    /// <summary>
    /// Indicates whether this model uses native layers (true) or ONNX model (false).
    /// </summary>
    private readonly bool _useNativeMode;

    #endregion

    #region ONNX Mode Fields

    /// <summary>
    /// The ONNX inference session for the model.
    /// </summary>
    protected readonly InferenceSession? OnnxSession;

    /// <summary>
    /// Path to the ONNX model file.
    /// </summary>
    protected readonly string? OnnxModelPath;

    #endregion

    #region Model Configuration

    /// <summary>
    /// The model's expected input sequence length.
    /// </summary>
    protected readonly int _sequenceLength;

    /// <summary>
    /// The model's prediction horizon.
    /// </summary>
    protected readonly int _predictionHorizon;

    /// <summary>
    /// The number of input features.
    /// </summary>
    protected readonly int _numFeatures;

    /// <summary>
    /// Loss history for training monitoring.
    /// </summary>
    protected readonly List<T> _lossHistory = new();

    /// <summary>
    /// Stores the last training loss for diagnostics.
    /// </summary>
    protected T _lastTrainingLoss;

    #endregion

    #region Properties

    /// <inheritdoc/>
    public bool UseNativeMode => _useNativeMode;

    /// <inheritdoc/>
    public override bool SupportsTraining => _useNativeMode;

    /// <inheritdoc/>
    public int SequenceLength => _sequenceLength;

    /// <inheritdoc/>
    public int PredictionHorizon => _predictionHorizon;

    /// <inheritdoc/>
    public int NumFeatures => _numFeatures;

    /// <inheritdoc/>
    public abstract string ModelName { get; }

    /// <summary>
    /// Gets the last recorded training loss.
    /// </summary>
    public T LastTrainingLoss => _lastTrainingLoss;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance using native mode for training and inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="sequenceLength">The input sequence length.</param>
    /// <param name="predictionHorizon">The prediction horizon (future steps to forecast).</param>
    /// <param name="numFeatures">The number of input features.</param>
    /// <param name="lossFunction">Optional loss function. Defaults to MSE.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor when you want to train a model from scratch:
    /// <code>
    /// var arch = new NeuralNetworkArchitecture&lt;double&gt;(
    ///     inputType: InputType.ThreeDimensional,
    ///     inputHeight: 96,    // sequence_length
    ///     inputWidth: 7,      // num_features
    ///     inputDepth: 1);
    ///
    /// var model = new PatchTST&lt;double&gt;(arch,
    ///     sequenceLength: 96,
    ///     predictionHorizon: 24,
    ///     numFeatures: 7);
    ///
    /// // Train the model
    /// model.Train(inputs, targets);
    /// </code>
    /// </para>
    /// </remarks>
    protected FinancialModelBase(
        NeuralNetworkArchitecture<T> architecture,
        int sequenceLength,
        int predictionHorizon,
        int numFeatures,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>())
    {
        ValidateConstructorArguments(sequenceLength, predictionHorizon, numFeatures);

        _useNativeMode = true;
        _sequenceLength = sequenceLength;
        _predictionHorizon = predictionHorizon;
        _numFeatures = numFeatures;
        _lastTrainingLoss = NumOps.Zero;

        InitializeLayers();
    }

    /// <summary>
    /// Initializes a new instance using a pretrained ONNX model for inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the pretrained ONNX model file.</param>
    /// <param name="sequenceLength">The input sequence length expected by the ONNX model.</param>
    /// <param name="predictionHorizon">The prediction horizon of the ONNX model.</param>
    /// <param name="numFeatures">The number of input features expected by the ONNX model.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor when you have a pretrained ONNX model:
    /// <code>
    /// var model = new PatchTST&lt;float&gt;(arch, "patchtst_etth1.onnx",
    ///     sequenceLength: 96,
    ///     predictionHorizon: 24,
    ///     numFeatures: 7);
    ///
    /// // Make predictions (training not supported)
    /// var forecast = model.Forecast(historicalData);
    /// </code>
    ///
    /// Note: Training is not supported in ONNX mode. Use the native constructor for training.
    /// </para>
    /// </remarks>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    protected FinancialModelBase(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int sequenceLength,
        int predictionHorizon,
        int numFeatures)
        : base(architecture, new MeanSquaredErrorLoss<T>())
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}");

        ValidateConstructorArguments(sequenceLength, predictionHorizon, numFeatures);

        _useNativeMode = false;
        OnnxModelPath = onnxModelPath;
        _sequenceLength = sequenceLength;
        _predictionHorizon = predictionHorizon;
        _numFeatures = numFeatures;
        _lastTrainingLoss = NumOps.Zero;

        // Load ONNX model
        try
        {
            OnnxSession = new InferenceSession(onnxModelPath);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to load ONNX model: {ex.Message}", ex);
        }

        InitializeLayers();
    }

    /// <summary>
    /// Validates constructor arguments.
    /// </summary>
    private static void ValidateConstructorArguments(int sequenceLength, int predictionHorizon, int numFeatures)
    {
        if (sequenceLength < 1)
            throw new ArgumentOutOfRangeException(nameof(sequenceLength), sequenceLength,
                "Sequence length must be at least 1.");
        if (predictionHorizon < 1)
            throw new ArgumentOutOfRangeException(nameof(predictionHorizon), predictionHorizon,
                "Prediction horizon must be at least 1.");
        if (numFeatures < 1)
            throw new ArgumentOutOfRangeException(nameof(numFeatures), numFeatures,
                "Number of features must be at least 1.");
    }

    #endregion

    #region IFinancialModel Implementation

    /// <inheritdoc/>
    public virtual Tensor<T> Forecast(Tensor<T> input, double[]? quantiles = null)
    {
        if (input is null)
            throw new ArgumentNullException(nameof(input));

        // Validate input shape
        ValidateInputShape(input);

        if (_useNativeMode)
        {
            return ForecastNative(input, quantiles);
        }
        else
        {
            return ForecastOnnx(input);
        }
    }

    /// <inheritdoc/>
    public virtual Dictionary<string, T> GetFinancialMetrics()
    {
        var metrics = new Dictionary<string, T>();

        // Add basic training metrics
        if (_lossHistory.Count > 0)
        {
            // Calculate average loss from recent history
            int lookback = Math.Min(100, _lossHistory.Count);
            T sum = NumOps.Zero;
            for (int i = _lossHistory.Count - lookback; i < _lossHistory.Count; i++)
            {
                sum = NumOps.Add(sum, _lossHistory[i]);
            }
            metrics["AverageLoss"] = NumOps.Divide(sum, NumOps.FromDouble(lookback));
            metrics["LastLoss"] = _lastTrainingLoss;
        }

        // Add model configuration
        metrics["SequenceLength"] = NumOps.FromDouble(_sequenceLength);
        metrics["PredictionHorizon"] = NumOps.FromDouble(_predictionHorizon);
        metrics["NumFeatures"] = NumOps.FromDouble(_numFeatures);

        return metrics;
    }

    #endregion

    #region Abstract Methods

    /// <summary>
    /// Performs forecasting using native layers.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <param name="quantiles">Optional quantiles for uncertainty estimation.</param>
    /// <returns>Forecast tensor.</returns>
    protected abstract Tensor<T> ForecastNative(Tensor<T> input, double[]? quantiles);

    /// <summary>
    /// Validates the input tensor shape.
    /// </summary>
    /// <param name="input">Input tensor to validate.</param>
    protected abstract void ValidateInputShape(Tensor<T> input);

    #endregion

    #region ONNX Inference

    /// <summary>
    /// Performs inference using the ONNX model.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Output tensor from ONNX inference.</returns>
    protected virtual Tensor<T> ForecastOnnx(Tensor<T> input)
    {
        if (OnnxSession is null)
            throw new InvalidOperationException("ONNX session is not initialized.");

        // Convert input tensor to ONNX tensor format
        var inputData = new float[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            inputData[i] = Convert.ToSingle(input.Data.Span[i]);
        }

        // Create ONNX input tensor
        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input.Shape);

        // Get input name from model
        var inputMeta = OnnxSession.InputMetadata;
        string inputName = inputMeta.Keys.First();

        // Run inference
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, onnxInput)
        };

        using var results = OnnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        // Convert back to our tensor format
        var outputShape = outputTensor.Dimensions.ToArray();
        var outputData = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++)
        {
            outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));
        }

        return new Tensor<T>(outputShape, new Vector<T>(outputData));
    }

    #endregion

    #region Training

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is not supported in ONNX mode. Use native mode for training.");

        if (input is null)
            throw new ArgumentNullException(nameof(input));
        if (expectedOutput is null)
            throw new ArgumentNullException(nameof(expectedOutput));

        // Perform forward pass
        var output = Predict(input);

        // Calculate loss
        var loss = LossFunction.CalculateLoss(output.ToVector(), expectedOutput.ToVector());
        _lastTrainingLoss = loss;

        // Track loss history
        _lossHistory.Add(loss);
        if (_lossHistory.Count > 1000)
            _lossHistory.RemoveAt(0);

        // Perform backward pass and update parameters
        TrainCore(input, expectedOutput, output);
    }

    /// <summary>
    /// Core training implementation for derived classes.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <param name="target">Target tensor.</param>
    /// <param name="output">Model output from forward pass.</param>
    protected abstract void TrainCore(Tensor<T> input, Tensor<T> target, Tensor<T> output);

    #endregion

    #region Prediction

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return Forecast(input);
    }

    #endregion

    #region Serialization

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var additionalInfo = new Dictionary<string, object>
        {
            { "ModelName", ModelName },
            { "SequenceLength", _sequenceLength },
            { "PredictionHorizon", _predictionHorizon },
            { "NumFeatures", _numFeatures },
            { "UseNativeMode", _useNativeMode }
        };

        if (!_useNativeMode && OnnxModelPath is not null)
        {
            additionalInfo["OnnxModelPath"] = OnnxModelPath;
        }

        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = additionalInfo,
            ModelData = _useNativeMode ? this.Serialize() : Array.Empty<byte>()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Serialization is not supported in ONNX mode.");

        writer.Write(_sequenceLength);
        writer.Write(_predictionHorizon);
        writer.Write(_numFeatures);

        // Derived classes add their own data via override
        SerializeModelSpecificData(writer);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Deserialization is not supported in ONNX mode.");

        // Read configuration (values are set in constructor, just advance reader)
        _ = reader.ReadInt32(); // sequenceLength
        _ = reader.ReadInt32(); // predictionHorizon
        _ = reader.ReadInt32(); // numFeatures

        // Derived classes read their own data via override
        DeserializeModelSpecificData(reader);
    }

    /// <summary>
    /// Serializes model-specific data. Override in derived classes.
    /// </summary>
    /// <param name="writer">Binary writer.</param>
    protected virtual void SerializeModelSpecificData(BinaryWriter writer)
    {
        // Default implementation does nothing
        // Derived classes override to add their specific data
    }

    /// <summary>
    /// Deserializes model-specific data. Override in derived classes.
    /// </summary>
    /// <param name="reader">Binary reader.</param>
    protected virtual void DeserializeModelSpecificData(BinaryReader reader)
    {
        // Default implementation does nothing
        // Derived classes override to read their specific data
    }

    #endregion

    #region IDisposable

    /// <summary>
    /// Disposes resources.
    /// </summary>
    /// <param name="disposing">Whether to dispose managed resources.</param>
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
