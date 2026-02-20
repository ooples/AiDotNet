using System.IO;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Video.Options;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.Video.Restoration;

/// <summary>
/// VRT: A Video Restoration Transformer for video super-resolution, deblurring, and denoising.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// VRT (Video Restoration Transformer) is a powerful architecture for video restoration tasks:
/// - Video super-resolution (increasing video resolution)
/// - Video deblurring (removing motion blur)
/// - Video denoising (removing noise from videos)
/// </para>
/// <para>
/// <b>For Beginners:</b> VRT improves video quality by analyzing multiple frames
/// together. Unlike image restoration that processes one frame at a time, VRT
/// uses temporal information to produce better, more consistent results.
///
/// Example usage (native mode for training):
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     inputHeight: 64, inputWidth: 64, inputDepth: 3);
/// var model = new VRT&lt;double&gt;(arch, scaleFactor: 4);
/// model.Train(lowResFrames, highResFrames);
/// var restoredFrame = model.Restore(lowResFrame);
/// </code>
///
/// Example usage (ONNX mode for inference only):
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     inputHeight: 64, inputWidth: 64, inputDepth: 3);
/// var model = new VRT&lt;double&gt;(arch, "vrt.onnx");
/// var restoredFrame = model.Restore(lowResFrame);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "VRT: A Video Restoration Transformer"
/// https://arxiv.org/abs/2201.12288
/// </para>
/// </remarks>
public class VRT<T> : VideoSuperResolutionBase<T>
{
    private readonly VRTOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

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
    /// Embedding dimension for the model.
    /// </summary>
    private readonly int _embedDim;

    /// <summary>
    /// Number of temporal frames processed together.
    /// </summary>
    private readonly int _numFrames;

    /// <summary>
    /// Number of transformer blocks.
    /// </summary>
    private readonly int _numBlocks;

    /// <summary>
    /// Upscaling factor for super-resolution.
    /// </summary>
    private readonly int _scaleFactor;

    /// <summary>
    /// Input frame height.
    /// </summary>
    private readonly int _inputHeight;

    /// <summary>
    /// Input frame width.
    /// </summary>
    private readonly int _inputWidth;

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


    #endregion

    #region Constructors

    /// <summary>
    /// Creates a VRT model using native layers for training and inference.
    /// </summary>
    /// <param name="architecture">Architecture for the video restoration network.</param>
    /// <param name="optimizer">Optional optimizer for training. Default: Adam.</param>
    /// <param name="lossFunction">Optional loss function. Default: MSE.</param>
    /// <param name="embedDim">Embedding dimension (default: 120).</param>
    /// <param name="numFrames">Number of frames to process together (default: 6).</param>
    /// <param name="numBlocks">Number of transformer blocks (default: 8).</param>
    /// <param name="scaleFactor">Upscaling factor (default: 4).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Create a trainable VRT model:
    /// <code>
    /// var arch = new NeuralNetworkArchitecture&lt;double&gt;(
    ///     inputType: InputType.ThreeDimensional,
    ///     inputHeight: 64, inputWidth: 64, inputDepth: 3);
    /// var model = new VRT&lt;double&gt;(arch, scaleFactor: 4);
    /// </code>
    /// </para>
    /// </remarks>
    public VRT(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int embedDim = 120,
        int numFrames = 6,
        int numBlocks = 8,
        int scaleFactor = 4,
        VRTOptions? options = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>())
    {
        _options = options ?? new VRTOptions();
        Options = _options;
        if (embedDim < 1)
            throw new ArgumentOutOfRangeException(nameof(embedDim), embedDim, "Embedding dimension must be at least 1.");
        if (numFrames < 1)
            throw new ArgumentOutOfRangeException(nameof(numFrames), numFrames, "Number of frames must be at least 1.");
        if (scaleFactor < 1)
            throw new ArgumentOutOfRangeException(nameof(scaleFactor), scaleFactor, "Scale factor must be at least 1.");

        _useNativeMode = true;
        _embedDim = embedDim;
        _numFrames = numFrames;
        NumFrames = numFrames;
        _numBlocks = numBlocks;
        _scaleFactor = scaleFactor;
        ScaleFactor = scaleFactor;
        _inputHeight = architecture.InputHeight > 0 ? architecture.InputHeight : 64;
        _inputWidth = architecture.InputWidth > 0 ? architecture.InputWidth : 64;

        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    /// <summary>
    /// Creates a VRT model using a pretrained ONNX model for inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the pretrained ONNX model.</param>
    /// <param name="scaleFactor">Scale factor of the model (default: 4).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor when you have a pretrained model
    /// in ONNX format. Training is not supported in ONNX mode.
    /// <code>
    /// var arch = new NeuralNetworkArchitecture&lt;double&gt;(
    ///     inputType: InputType.ThreeDimensional,
    ///     inputHeight: 64, inputWidth: 64, inputDepth: 3);
    /// var model = new VRT&lt;double&gt;(arch, "vrt_sr_x4.onnx");
    /// var restoredFrame = model.Restore(lowResFrame);
    /// </code>
    /// </para>
    /// </remarks>
    /// <exception cref="FileNotFoundException">Thrown if the ONNX model file is not found.</exception>
    public VRT(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int scaleFactor = 4,
        VRTOptions? options = null)
        : base(architecture, new MeanSquaredErrorLoss<T>())
    {
        _options = options ?? new VRTOptions();
        Options = _options;
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"VRT ONNX model not found: {onnxModelPath}");

        _useNativeMode = false;
        _onnxModelPath = onnxModelPath;
        _embedDim = 120;
        _numFrames = 6;
        NumFrames = 6;
        _numBlocks = 8;
        _scaleFactor = scaleFactor;
        ScaleFactor = scaleFactor;
        _inputHeight = architecture.InputHeight > 0 ? architecture.InputHeight : 64;
        _inputWidth = architecture.InputWidth > 0 ? architecture.InputWidth : 64;
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
    /// Restores a video frame using the VRT model.
    /// </summary>
    /// <param name="input">Input video frame(s) tensor [B, C, H, W] or [C, H, W].</param>
    /// <returns>Restored video frame tensor.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method enhances a video frame by:
    /// - Super-resolution: Making it higher resolution
    /// - Deblurring: Removing motion blur
    /// - Denoising: Removing visual noise
    /// </para>
    /// </remarks>
    public Tensor<T> Restore(Tensor<T> input)
    {
        if (input is null)
            throw new ArgumentNullException(nameof(input));

        if (_useNativeMode)
        {
            return Forward(input);
        }
        else
        {
            return PredictOnnx(input);
        }
    }

    /// <summary>
    /// Performs video super-resolution.
    /// </summary>
    /// <param name="lowResFrames">Low-resolution video frames.</param>
    /// <returns>High-resolution video frames.</returns>
    public Tensor<T> SuperResolve(Tensor<T> lowResFrames)
    {
        return Restore(lowResFrames);
    }

    /// <summary>
    /// Performs video deblurring.
    /// </summary>
    /// <param name="blurryFrames">Blurry video frames.</param>
    /// <returns>Deblurred video frames.</returns>
    public Tensor<T> Deblur(Tensor<T> blurryFrames)
    {
        return Restore(blurryFrames);
    }

    /// <summary>
    /// Performs video denoising.
    /// </summary>
    /// <param name="noisyFrames">Noisy video frames.</param>
    /// <returns>Denoised video frames.</returns>
    public Tensor<T> Denoise(Tensor<T> noisyFrames)
    {
        return Restore(noisyFrames);
    }

    #endregion

    #region Inference

    /// <summary>
    /// Performs a forward pass through the network.
    /// </summary>
    protected override Tensor<T> Forward(Tensor<T> input)
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
        return Restore(input);
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
            int inputChannels = Architecture.InputDepth > 0 ? Architecture.InputDepth : 3;
            int inputHeight = Architecture.InputHeight > 0 ? Architecture.InputHeight : 64;
            int inputWidth = Architecture.InputWidth > 0 ? Architecture.InputWidth : 64;

            Layers.AddRange(LayerHelper<T>.CreateDefaultVRTLayers(
                inputChannels: inputChannels,
                inputHeight: inputHeight,
                inputWidth: inputWidth,
                embedDim: _embedDim,
                numFrames: _numFrames,
                numBlocks: _numBlocks,
                scaleFactor: _scaleFactor));
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
            { "ModelName", "VRT" },
            { "EmbedDim", _embedDim },
            { "NumFrames", _numFrames },
            { "NumBlocks", _numBlocks },
            { "ScaleFactor", _scaleFactor },
            { "InputHeight", _inputHeight },
            { "InputWidth", _inputWidth },
            { "UseNativeMode", _useNativeMode }
        };

        if (!_useNativeMode && _onnxModelPath != null)
        {
            additionalInfo["OnnxModelPath"] = _onnxModelPath;
        }

        return new ModelMetadata<T>
        {
            ModelType = ModelType.VideoSuperResolution,
            AdditionalInfo = additionalInfo,
            ModelData = _useNativeMode ? this.Serialize() : Array.Empty<byte>()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Serialization is not supported in ONNX mode.");

        writer.Write(_embedDim);
        writer.Write(_numFrames);
        writer.Write(_numBlocks);
        writer.Write(_scaleFactor);
        writer.Write(_inputHeight);
        writer.Write(_inputWidth);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Deserialization is not supported in ONNX mode.");

        _ = reader.ReadInt32(); // embedDim
        _ = reader.ReadInt32(); // numFrames
        _ = reader.ReadInt32(); // numBlocks
        _ = reader.ReadInt32(); // scaleFactor
        _ = reader.ReadInt32(); // inputHeight
        _ = reader.ReadInt32(); // inputWidth
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new VRT<T>(
            Architecture,
            _optimizer,
            _lossFunction,
            _embedDim,
            _numFrames,
            _numBlocks,
            _scaleFactor);
    }

    #endregion

    #region Base Class Abstract Methods

    /// <inheritdoc/>
    public override Tensor<T> Upscale(Tensor<T> lowResFrames)
    {
        return Forward(lowResFrames);
    }

    /// <inheritdoc/>
    protected override Tensor<T> PreprocessFrames(Tensor<T> rawFrames)
    {
        return NormalizeFrames(rawFrames);
    }

    /// <inheritdoc/>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput)
    {
        return DenormalizeFrames(modelOutput);
    }

    #endregion

}
