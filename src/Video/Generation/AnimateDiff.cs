using System.IO;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.Video.Generation;

/// <summary>
/// AnimateDiff: Motion module for animating text-to-image diffusion models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// AnimateDiff is a motion module that:
/// - Adds temporal coherence to image diffusion models
/// - Converts image generators into video generators
/// - Learns motion patterns from video data
/// </para>
/// <para>
/// <b>For Beginners:</b> AnimateDiff makes still image generators create videos.
/// It plugs into existing models like Stable Diffusion to add movement.
/// Instead of generating one image, it generates multiple frames that flow smoothly.
///
/// Example usage (native mode for training):
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     inputHeight: 64, inputWidth: 64, inputDepth: 320);
/// var model = new AnimateDiff&lt;double&gt;(arch);
/// model.Train(inputFeatures, motionFeatures);
/// var animatedFeatures = model.AddMotion(staticFeatures);
/// </code>
///
/// Example usage (ONNX mode for inference only):
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     inputHeight: 64, inputWidth: 64, inputDepth: 320);
/// var model = new AnimateDiff&lt;double&gt;(arch, "animatediff.onnx");
/// var animatedFeatures = model.AddMotion(staticFeatures);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models"
/// https://arxiv.org/abs/2307.04725
/// </para>
/// </remarks>
public class AnimateDiff<T> : NeuralNetworkBase<T>
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
    private readonly InferenceSession? _onnxSession;

    /// <summary>
    /// Path to the ONNX model file.
    /// </summary>
    private readonly string? _onnxModelPath;

    #endregion

    #region Native Mode Fields

    /// <summary>
    /// The optimizer used for training.
    /// </summary>
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;

    /// <summary>
    /// The loss function for training.
    /// </summary>
    private readonly ILossFunction<T> _lossFunction;

    /// <summary>
    /// Number of input feature channels.
    /// </summary>
    private readonly int _inputChannels;

    /// <summary>
    /// Number of motion transformer layers.
    /// </summary>
    private readonly int _numLayers;

    /// <summary>
    /// Number of video frames.
    /// </summary>
    private readonly int _numFrames;

    /// <summary>
    /// Feature height.
    /// </summary>
    private readonly int _featureHeight;

    /// <summary>
    /// Feature width.
    /// </summary>
    private readonly int _featureWidth;

    #endregion

    #region Properties

    /// <summary>
    /// Gets whether this model uses native mode (true) or ONNX mode (false).
    /// </summary>
    internal bool UseNativeMode => _useNativeMode;

    /// <summary>
    /// Gets whether training is supported (only in native mode).
    /// </summary>
    public override bool SupportsTraining => _useNativeMode;

    /// <summary>
    /// Gets the number of input channels.
    /// </summary>
    internal int InputChannels => _inputChannels;

    /// <summary>
    /// Gets the number of frames processed.
    /// </summary>
    internal int NumFrames => _numFrames;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates an AnimateDiff model using native layers for training and inference.
    /// </summary>
    /// <param name="architecture">Architecture for the motion module.</param>
    /// <param name="optimizer">Optional optimizer for training. Default: Adam.</param>
    /// <param name="lossFunction">Optional loss function. Default: MSE.</param>
    /// <param name="inputChannels">Number of input feature channels (default: 320).</param>
    /// <param name="numLayers">Number of motion transformer layers (default: 8).</param>
    /// <param name="numFrames">Number of video frames (default: 16).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Create a trainable AnimateDiff model:
    /// <code>
    /// var arch = new NeuralNetworkArchitecture&lt;double&gt;(
    ///     inputType: InputType.ThreeDimensional,
    ///     inputHeight: 64, inputWidth: 64, inputDepth: 320);
    /// var model = new AnimateDiff&lt;double&gt;(arch);
    /// </code>
    /// </para>
    /// </remarks>
    public AnimateDiff(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int inputChannels = 320,
        int numLayers = 8,
        int numFrames = 16)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>())
    {
        if (inputChannels < 1)
            throw new ArgumentOutOfRangeException(nameof(inputChannels), inputChannels, "Input channels must be at least 1.");
        if (numLayers < 1)
            throw new ArgumentOutOfRangeException(nameof(numLayers), numLayers, "Number of layers must be at least 1.");
        if (numFrames < 1)
            throw new ArgumentOutOfRangeException(nameof(numFrames), numFrames, "Number of frames must be at least 1.");

        _useNativeMode = true;
        _inputChannels = inputChannels;
        _numLayers = numLayers;
        _numFrames = numFrames;
        _featureHeight = architecture.InputHeight > 0 ? architecture.InputHeight : 64;
        _featureWidth = architecture.InputWidth > 0 ? architecture.InputWidth : 64;

        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    /// <summary>
    /// Creates an AnimateDiff model using a pretrained ONNX model for inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the pretrained ONNX model.</param>
    /// <param name="numFrames">Number of frames the model processes (default: 16).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor when you have a pretrained model
    /// in ONNX format. Training is not supported in ONNX mode.
    /// <code>
    /// var arch = new NeuralNetworkArchitecture&lt;double&gt;(
    ///     inputType: InputType.ThreeDimensional,
    ///     inputHeight: 64, inputWidth: 64, inputDepth: 320);
    /// var model = new AnimateDiff&lt;double&gt;(arch, "animatediff.onnx");
    /// var animated = model.AddMotion(features);
    /// </code>
    /// </para>
    /// </remarks>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    public AnimateDiff(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int numFrames = 16)
        : base(architecture, new MeanSquaredErrorLoss<T>())
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"AnimateDiff ONNX model not found: {onnxModelPath}");

        _useNativeMode = false;
        _onnxModelPath = onnxModelPath;
        _inputChannels = 320;
        _numLayers = 8;
        _numFrames = numFrames;
        _featureHeight = architecture.InputHeight > 0 ? architecture.InputHeight : 64;
        _featureWidth = architecture.InputWidth > 0 ? architecture.InputWidth : 64;
        _lossFunction = new MeanSquaredErrorLoss<T>();

        try
        {
            _onnxSession = new InferenceSession(onnxModelPath);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to load ONNX model: {ex.Message}", ex);
        }

        InitializeLayers();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Adds motion to static features from an image diffusion model.
    /// </summary>
    /// <param name="staticFeatures">Static features tensor [B, C, H, W] or [C, H, W].</param>
    /// <returns>Motion-enhanced features tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes features from a static image generator
    /// and adds temporal consistency to create animated output. The input comes from
    /// an image diffusion model's intermediate layers.
    /// </para>
    /// </remarks>
    public Tensor<T> AddMotion(Tensor<T> staticFeatures)
    {
        if (staticFeatures is null)
            throw new ArgumentNullException(nameof(staticFeatures));

        if (_useNativeMode)
        {
            return Forward(staticFeatures);
        }
        else
        {
            return PredictOnnx(staticFeatures);
        }
    }

    /// <summary>
    /// Processes temporal features for motion modeling.
    /// </summary>
    /// <param name="temporalFeatures">Temporal features spanning multiple frames.</param>
    /// <returns>Processed motion features.</returns>
    public Tensor<T> ProcessMotion(Tensor<T> temporalFeatures)
    {
        return AddMotion(temporalFeatures);
    }

    /// <summary>
    /// Blends motion module output with original features.
    /// </summary>
    /// <param name="originalFeatures">Original static features.</param>
    /// <param name="motionFeatures">Motion module output.</param>
    /// <param name="blendFactor">Blend factor (0-1, default: 1.0).</param>
    /// <returns>Blended features.</returns>
    public Tensor<T> BlendFeatures(Tensor<T> originalFeatures, Tensor<T> motionFeatures, double blendFactor = 1.0)
    {
        if (originalFeatures is null)
            throw new ArgumentNullException(nameof(originalFeatures));
        if (motionFeatures is null)
            throw new ArgumentNullException(nameof(motionFeatures));

        blendFactor = MathHelper.Clamp(blendFactor, 0.0, 1.0);
        double inverseFactor = 1.0 - blendFactor;

        int minLen = Math.Min(originalFeatures.Length, motionFeatures.Length);
        var resultData = new T[minLen];

        for (int i = 0; i < minLen; i++)
        {
            double orig = NumOps.ToDouble(originalFeatures.Data.Span[i]);
            double motion = NumOps.ToDouble(motionFeatures.Data.Span[i]);
            double blended = inverseFactor * orig + blendFactor * motion;
            resultData[i] = NumOps.FromDouble(blended);
        }

        return new Tensor<T>(originalFeatures.Shape, new Vector<T>(resultData));
    }

    #endregion

    #region Inference

    /// <summary>
    /// Performs a forward pass through the network.
    /// </summary>
    private Tensor<T> Forward(Tensor<T> input)
    {
        var result = input;
        foreach (var layer in Layers)
        {
            result = layer.Forward(result);
        }
        return result;
    }

    /// <summary>
    /// Performs inference using the ONNX model.
    /// </summary>
    private Tensor<T> PredictOnnx(Tensor<T> input)
    {
        if (_onnxSession is null)
            throw new InvalidOperationException("ONNX session is not initialized.");

        var inputData = new float[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            inputData[i] = Convert.ToSingle(input.Data.Span[i]);
        }

        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input.Shape);
        var inputMeta = _onnxSession.InputMetadata;
        string inputName = inputMeta.Keys.First();

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, onnxInput)
        };

        using var results = _onnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        var outputShape = outputTensor.Dimensions.ToArray();
        var outputData = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++)
        {
            outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));
        }

        return new Tensor<T>(outputShape, new Vector<T>(outputData));
    }

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return AddMotion(input);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is not supported in ONNX mode. Use native mode for training.");

        var prediction = Predict(input);
        var loss = _lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());
        LastLoss = loss;

        var outputGradient = _lossFunction.CalculateDerivative(prediction.ToVector(), expectedOutput.ToVector());
        var outputGradientTensor = new Tensor<T>(prediction.Shape, outputGradient);

        var currentGradient = outputGradientTensor;
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            currentGradient = Layers[i].Backward(currentGradient);
        }

        if (_optimizer != null)
        {
            _optimizer.UpdateParameters(Layers);
        }
    }

    #endregion

    #region Layer Initialization

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
        {
            ClearLayers();
            return;
        }

        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultAnimateDiffLayers(
                inputChannels: _inputChannels,
                inputHeight: _featureHeight,
                inputWidth: _featureWidth,
                numLayers: _numLayers,
                numFrames: _numFrames));
        }
    }

    #endregion

    #region Serialization

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Parameter updates are not supported in ONNX mode.");

        int index = 0;
        foreach (var layer in Layers)
        {
            int layerParameterCount = layer.ParameterCount;
            var layerParameters = parameters.Slice(index, layerParameterCount);
            layer.UpdateParameters(layerParameters);
            index += layerParameterCount;
        }
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var additionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "AnimateDiff" },
            { "InputChannels", _inputChannels },
            { "NumLayers", _numLayers },
            { "NumFrames", _numFrames },
            { "FeatureHeight", _featureHeight },
            { "FeatureWidth", _featureWidth },
            { "UseNativeMode", _useNativeMode }
        };

        if (!_useNativeMode && _onnxModelPath != null)
        {
            additionalInfo["OnnxModelPath"] = _onnxModelPath;
        }

        return new ModelMetadata<T>
        {
            ModelType = ModelType.TextToVideo,
            AdditionalInfo = additionalInfo,
            ModelData = _useNativeMode ? this.Serialize() : Array.Empty<byte>()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Serialization is not supported in ONNX mode.");

        writer.Write(_inputChannels);
        writer.Write(_numLayers);
        writer.Write(_numFrames);
        writer.Write(_featureHeight);
        writer.Write(_featureWidth);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Deserialization is not supported in ONNX mode.");

        _ = reader.ReadInt32(); // inputChannels
        _ = reader.ReadInt32(); // numLayers
        _ = reader.ReadInt32(); // numFrames
        _ = reader.ReadInt32(); // featureHeight
        _ = reader.ReadInt32(); // featureWidth
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new AnimateDiff<T>(
            Architecture,
            _optimizer,
            _lossFunction,
            _inputChannels,
            _numLayers,
            _numFrames);
    }

    #endregion
}
