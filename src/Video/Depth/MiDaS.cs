using System.IO;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Video.Options;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.Video.Depth;

/// <summary>
/// MiDaS: Towards Robust Monocular Depth Estimation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> MiDaS estimates depth from a single image (monocular depth estimation).
/// Unlike stereo vision which uses two cameras, MiDaS uses deep learning to predict depth
/// from visual cues like texture, perspective, and object sizes.
///
/// Key capabilities:
/// - Single image to depth map conversion
/// - Works on arbitrary images without camera parameters
/// - Outputs relative depth (closer objects have higher values)
/// - Robust across different scenes and domains
///
/// Example usage:
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     inputHeight: 384, inputWidth: 384, inputDepth: 3);
/// var model = new MiDaS&lt;double&gt;(arch);
/// var depthMap = model.EstimateDepth(image);
/// </code>
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - ViT-based encoder for feature extraction
/// - Multi-scale fusion decoder
/// - Scale and shift invariant loss for training
/// - Trained on diverse mixed datasets for robustness
/// </para>
/// <para>
/// <b>Reference:</b> "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer"
/// https://arxiv.org/abs/1907.01341
/// </para>
/// </remarks>
public class MiDaS<T> : NeuralNetworkBase<T>
{
    private readonly MiDaSOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #region Execution Mode

    private readonly bool _useNativeMode;

    #endregion

    #region ONNX Mode Fields

    private readonly InferenceSession? _onnxSession;
    private readonly string? _onnxModelPath;

    #endregion

    #region Native Mode Fields

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ILossFunction<T> _lossFunction;
    private readonly int _embedDim;
    private readonly int _numLayers;
    private readonly int _imageSize;
    private readonly MiDaSVariant _variant;

    #endregion

    #region Properties

    internal bool UseNativeMode => _useNativeMode;
    public override bool SupportsTraining => _useNativeMode;
    internal int EmbedDim => _embedDim;
    internal int ImageSize => _imageSize;
    internal MiDaSVariant Variant => _variant;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a MiDaS model using native layers for training and inference.
    /// </summary>
    public MiDaS(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int embedDim = 768,
        int numLayers = 12,
        MiDaSVariant variant = MiDaSVariant.DPTLarge,
        MiDaSOptions? options = null)
        : base(architecture, lossFunction ?? new ScaleInvariantDepthLoss<T>())
    {
        _options = options ?? new MiDaSOptions();
        Options = _options;

        _useNativeMode = true;
        _embedDim = embedDim;
        _numLayers = numLayers;
        _imageSize = architecture.InputHeight > 0 ? architecture.InputHeight : 384;
        _variant = variant;

        _lossFunction = lossFunction ?? new ScaleInvariantDepthLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    /// <summary>
    /// Creates a MiDaS model using a pretrained ONNX model for inference.
    /// </summary>
    public MiDaS(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        MiDaSVariant variant = MiDaSVariant.DPTLarge,
        MiDaSOptions? options = null)
        : base(architecture, new ScaleInvariantDepthLoss<T>())
    {
        _options = options ?? new MiDaSOptions();
        Options = _options;

        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"MiDaS ONNX model not found: {onnxModelPath}");

        _useNativeMode = false;
        _onnxModelPath = onnxModelPath;
        _embedDim = 768;
        _numLayers = 12;
        _imageSize = architecture.InputHeight > 0 ? architecture.InputHeight : 384;
        _variant = variant;
        _lossFunction = new ScaleInvariantDepthLoss<T>();

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
    /// Estimates depth from an input image.
    /// </summary>
    /// <param name="image">Input image tensor [B, C, H, W] or [C, H, W].</param>
    /// <returns>Depth map tensor with same spatial dimensions.</returns>
    public Tensor<T> EstimateDepth(Tensor<T> image)
    {
        if (image is null)
            throw new ArgumentNullException(nameof(image));

        return _useNativeMode ? Forward(image) : PredictOnnx(image);
    }

    /// <summary>
    /// Estimates depth for multiple video frames.
    /// </summary>
    public List<Tensor<T>> EstimateDepthForVideo(List<Tensor<T>> frames)
    {
        var depthMaps = new List<Tensor<T>>();
        foreach (var frame in frames)
        {
            depthMaps.Add(EstimateDepth(frame));
        }
        return depthMaps;
    }

    /// <summary>
    /// Normalizes depth map to 0-1 range for visualization.
    /// </summary>
    public Tensor<T> NormalizeDepthMap(Tensor<T> depthMap)
    {
        double minVal = double.MaxValue;
        double maxVal = double.MinValue;

        for (int i = 0; i < depthMap.Length; i++)
        {
            double val = Convert.ToDouble(depthMap.Data.Span[i]);
            if (val < minVal) minVal = val;
            if (val > maxVal) maxVal = val;
        }

        double range = maxVal - minVal + 1e-8;
        return depthMap.Transform((v, _) =>
            NumOps.FromDouble((Convert.ToDouble(v) - minVal) / range));
    }

    #endregion

    #region Inference

    private Tensor<T> Forward(Tensor<T> input)
    {
        var result = input;
        foreach (var layer in Layers)
        {
            result = layer.Forward(result);
        }
        return result;
    }

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

        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, onnxInput) };

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

    public override Tensor<T> Predict(Tensor<T> input) => EstimateDepth(input);

    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is not supported in ONNX mode.");

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

        _optimizer?.UpdateParameters(Layers);
    }

    #endregion

    #region Layer Initialization

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) { ClearLayers(); return; }

        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            int inputChannels = Architecture.InputDepth > 0 ? Architecture.InputDepth : 3;
            int inputHeight = Architecture.InputHeight > 0 ? Architecture.InputHeight : 384;
            int inputWidth = Architecture.InputWidth > 0 ? Architecture.InputWidth : 384;

            Layers.AddRange(LayerHelper<T>.CreateDefaultMiDaSLayers(
                inputChannels, inputHeight, inputWidth, _embedDim, _numLayers));
        }
    }

    #endregion

    #region Serialization

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Parameter updates are not supported in ONNX mode.");

        int offset = 0;
        foreach (var layer in Layers)
        {
            var layerParams = layer.GetParameters();
            int paramCount = layerParams.Length;
            if (paramCount > 0 && offset + paramCount <= parameters.Length)
            {
                var slice = new Vector<T>(paramCount);
                for (int i = 0; i < paramCount; i++) slice[i] = parameters[offset + i];
                layer.SetParameters(slice);
                offset += paramCount;
            }
        }
    }

    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        ModelType = ModelType.DepthEstimation,
        AdditionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "MiDaS" },
            { "EmbedDim", _embedDim },
            { "NumLayers", _numLayers },
            { "ImageSize", _imageSize },
            { "Variant", _variant.ToString() },
            { "UseNativeMode", _useNativeMode }
        },
        ModelData = _useNativeMode ? this.Serialize() : []
    };

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        if (!_useNativeMode) throw new InvalidOperationException("Serialization is not supported in ONNX mode.");
        writer.Write(_embedDim);
        writer.Write(_numLayers);
        writer.Write(_imageSize);
        writer.Write((int)_variant);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        if (!_useNativeMode) throw new InvalidOperationException("Deserialization is not supported in ONNX mode.");
        for (int i = 0; i < 4; i++) _ = reader.ReadInt32();
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() =>
        new MiDaS<T>(Architecture, _optimizer, _lossFunction, _embedDim, _numLayers, _variant);

    #endregion
}

/// <summary>
/// MiDaS model variant.
/// </summary>
public enum MiDaSVariant
{
    /// <summary>DPT-Large: Highest quality, slowest.</summary>
    DPTLarge,
    /// <summary>DPT-Hybrid: Good balance of speed and quality.</summary>
    DPTHybrid,
    /// <summary>MiDaS v2.1 Small: Fastest, lower quality.</summary>
    Small
}
