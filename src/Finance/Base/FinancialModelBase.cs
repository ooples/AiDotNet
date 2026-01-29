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
    protected readonly bool _baseUseNativeMode;

    #endregion

    #region ONNX Mode Fields

    /// <summary>
    /// The ONNX inference session for the model.
    /// </summary>
    protected InferenceSession? OnnxSession;

    /// <summary>
    /// Path to the ONNX model file.
    /// </summary>
    protected string? OnnxModelPath;

    #endregion

    #region Model Configuration

    /// <summary>
    /// The model's expected input sequence length.
    /// </summary>
    protected readonly int _baseSequenceLength;

    /// <summary>
    /// The model's prediction horizon.
    /// </summary>
    protected readonly int _basePredictionHorizon;

    /// <summary>
    /// The number of input features.
    /// </summary>
    protected readonly int _baseNumFeatures;

    /// <summary>
    /// Loss history for training monitoring.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This list tracks recent training errors so you can
    /// see whether the model is improving over time.
    /// </para>
    /// </remarks>
    protected readonly List<T> _lossHistory = new();

    /// <summary>
    /// Stores the last training loss for diagnostics.
    /// </summary>
    protected T _lastTrainingLoss;

    #endregion

    #region Properties

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> True means the model is using trainable C# layers.
    /// False means it's using a pretrained ONNX model for fast inference only.
    /// </para>
    /// </remarks>
    public virtual bool UseNativeMode => _baseUseNativeMode;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Training only works in native mode. ONNX mode is
    /// inference-only, so this returns false there.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => UseNativeMode;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is how many past time steps the model expects
    /// as input for each prediction.
    /// </para>
    /// </remarks>
    public virtual int SequenceLength => _baseSequenceLength;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is how far into the future the model predicts
    /// for each forecast.
    /// </para>
    /// </remarks>
    public virtual int PredictionHorizon => _basePredictionHorizon;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is how many variables (features) the model
    /// expects at each time step (e.g., price, volume, indicators).
    /// </para>
    /// </remarks>
    public virtual int NumFeatures => _baseNumFeatures;

    /// <summary>
    /// Gets the last recorded training loss.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the most recent "how wrong was the model?" number
    /// from training. Smaller values mean the model's predictions are closer to the
    /// expected output.
    /// </para>
    /// </remarks>
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

        _baseUseNativeMode = true;
        _baseSequenceLength = sequenceLength;
        _basePredictionHorizon = predictionHorizon;
        _baseNumFeatures = numFeatures;
        _lastTrainingLoss = NumOps.Zero;

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

        _baseUseNativeMode = false;
        OnnxModelPath = onnxModelPath;
        _baseSequenceLength = sequenceLength;
        _basePredictionHorizon = predictionHorizon;
        _baseNumFeatures = numFeatures;
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

    }

    /// <summary>
    /// Initializes a new instance with deferred configuration.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="lossFunction">Optional loss function. Defaults to MSE.</param>
    /// <param name="maxGradNorm">Maximum gradient norm for clipping.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor lets derived models set up their
    /// sequence length and other settings after the base class is created.
    /// It mirrors the legacy pattern used by earlier financial models.
    /// </para>
    /// </remarks>
    protected FinancialModelBase(
        NeuralNetworkArchitecture<T> architecture,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), maxGradNorm)
    {
        _baseUseNativeMode = true;
        _baseSequenceLength = 0;
        _basePredictionHorizon = 0;
        _baseNumFeatures = architecture.InputSize;
        _lastTrainingLoss = NumOps.Zero;
    }

    /// <summary>
    /// Validates constructor arguments.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is a safety check to make sure you do not accidentally
    /// build a model with impossible settings (like a sequence length of zero).
    /// Catching these early prevents confusing errors later.
    /// </para>
    /// </remarks>
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
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main prediction method.
    /// It decides whether to use the native C# model (trainable) or the ONNX model
    /// (fast inference) based on how the class was constructed.
    /// </para>
    /// </remarks>
    public virtual Tensor<T> Forecast(Tensor<T> input, double[]? quantiles = null)
    {
        if (input is null)
            throw new ArgumentNullException(nameof(input));

        // Validate input shape
        ValidateInputShape(input);

        if (UseNativeMode)
        {
            return ForecastNative(input, quantiles);
        }
        else
        {
            return ForecastOnnx(input);
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Returns a simple report card for the model,
    /// including configuration (sequence length, features) and training loss history.
    /// </para>
    /// </remarks>
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
        metrics["SequenceLength"] = NumOps.FromDouble(SequenceLength);
        metrics["PredictionHorizon"] = NumOps.FromDouble(PredictionHorizon);
        metrics["NumFeatures"] = NumOps.FromDouble(NumFeatures);

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
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the "native mode" prediction path.
    /// Derived models implement their own forward pass here using C# layers.
    /// </para>
    /// </remarks>
    protected virtual Tensor<T> ForecastNative(Tensor<T> input, double[]? quantiles)
    {
        throw new NotSupportedException("ForecastNative is not implemented for this model.");
    }

    /// <summary>
    /// Validates the input tensor shape.
    /// </summary>
    /// <param name="input">Input tensor to validate.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Ensures the input data has the right dimensions
    /// before the model tries to process it. This avoids shape mismatch errors.
    /// </para>
    /// </remarks>
    protected virtual void ValidateInputShape(Tensor<T> input)
    {
    }

    #endregion

    #region ONNX Inference

    /// <summary>
    /// Performs inference using the ONNX model.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <returns>Output tensor from ONNX inference.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> ONNX mode is like running a pre-trained black box.
    /// This method converts your input to ONNX format, runs the model, and converts
    /// the output back to AiDotNet tensors.
    /// </para>
    /// </remarks>
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
        if (inputMeta.Count == 0)
            throw new InvalidOperationException("ONNX model has no input metadata.");

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
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Training only works in native mode.
    /// This method runs a forward pass, calculates the error (loss),
    /// and then lets derived classes update the model weights.
    /// </para>
    /// </remarks>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!UseNativeMode)
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
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Derived models implement their specific training logic here,
    /// including backward passes and optimizer updates.
    /// </para>
    /// </remarks>
    protected virtual void TrainCore(Tensor<T> input, Tensor<T> target, Tensor<T> output)
    {
        throw new NotSupportedException("TrainCore is not implemented for this model.");
    }

    #endregion

    #region Prediction

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Predict is an alias for Forecast so the model works with
    /// the broader AiDotNet interfaces. It forwards to the same logic.
    /// </para>
    /// </remarks>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return Forecast(input);
    }

    #endregion

    #region Serialization

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Metadata is how the library keeps track of what a model is,
    /// including its configuration and (for native mode) the serialized weights.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var additionalInfo = new Dictionary<string, object>
        {
            { "SequenceLength", SequenceLength },
            { "PredictionHorizon", PredictionHorizon },
            { "NumFeatures", NumFeatures },
            { "UseNativeMode", UseNativeMode }
        };

        if (!UseNativeMode && OnnxModelPath is not null)
        {
            additionalInfo["OnnxModelPath"] = OnnxModelPath;
        }

        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = additionalInfo,
            ModelData = UseNativeMode ? this.Serialize() : Array.Empty<byte>()
        };
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Saves the core configuration (sequence length, horizon, features)
    /// before allowing derived classes to store their own settings.
    /// </para>
    /// </remarks>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        if (!UseNativeMode)
            throw new InvalidOperationException("Serialization is not supported in ONNX mode.");

        writer.Write(SequenceLength);
        writer.Write(PredictionHorizon);
        writer.Write(NumFeatures);

        // Derived classes add their own data via override
        SerializeModelSpecificData(writer);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Loads the core configuration saved by SerializeNetworkSpecificData,
    /// then lets derived classes restore their extra settings.
    /// </para>
    /// </remarks>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        if (!UseNativeMode)
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
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Derived models can save extra settings here
    /// (like the number of layers or hidden size).
    /// </para>
    /// </remarks>
    protected virtual void SerializeModelSpecificData(BinaryWriter writer)
    {
        // Default implementation does nothing
        // Derived classes override to add their specific data
    }

    /// <summary>
    /// Deserializes model-specific data. Override in derived classes.
    /// </summary>
    /// <param name="reader">Binary reader.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Derived models load their extra settings here
    /// to restore a saved model correctly.
    /// </para>
    /// </remarks>
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
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Frees memory and file handles (like ONNX sessions)
    /// when you're done with the model.
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
