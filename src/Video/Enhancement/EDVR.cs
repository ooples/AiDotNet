using System.IO;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.Video.Enhancement;

/// <summary>
/// EDVR: Video Restoration with Enhanced Deformable Convolutional Networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> EDVR is a state-of-the-art video restoration model that can:
/// - Upscale low-resolution video (super-resolution)
/// - Remove blur and noise (deblurring/denoising)
/// - Handle complex motions and occlusions
///
/// The model uses alignment to compensate for motion between frames,
/// then fuses information from multiple frames to restore high-quality output.
///
/// Example usage:
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     inputHeight: 256, inputWidth: 256, inputDepth: 3);
/// var model = new EDVR&lt;double&gt;(arch);
/// var enhancedFrames = model.Enhance(frames);
/// </code>
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - PCD (Pyramid, Cascading and Deformable) alignment module
/// - TSA (Temporal and Spatial Attention) fusion module
/// - Deformable convolutions for motion-adaptive feature alignment
/// </para>
/// <para>
/// <b>Reference:</b> "EDVR: Video Restoration with Enhanced Deformable Convolutional Networks" CVPR 2019
/// https://arxiv.org/abs/1905.02716
/// </para>
/// </remarks>
public class EDVR<T> : NeuralNetworkBase<T>
{
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
    private readonly int _numFeatures;
    private readonly int _numFrames;
    private readonly int _numBlocks;
    private readonly int _scaleFactor;
    private readonly int _imageSize;

    #endregion

    #region Properties

    internal bool UseNativeMode => _useNativeMode;
    public override bool SupportsTraining => _useNativeMode;
    internal int NumFeatures => _numFeatures;
    internal int NumFrames => _numFrames;
    internal int ScaleFactor => _scaleFactor;

    #endregion

    #region Constructors

    public EDVR(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int numFeatures = 64,
        int numFrames = 5,
        int numBlocks = 5,
        int scaleFactor = 4)
        : base(architecture, lossFunction ?? new CharbonnierLoss<T>())
    {
        _useNativeMode = true;
        _numFeatures = numFeatures;
        _numFrames = numFrames;
        _numBlocks = numBlocks;
        _scaleFactor = scaleFactor;
        _imageSize = architecture.InputHeight > 0 ? architecture.InputHeight : 256;

        _lossFunction = lossFunction ?? new CharbonnierLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    public EDVR(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int scaleFactor = 4)
        : base(architecture, new CharbonnierLoss<T>())
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"EDVR ONNX model not found: {onnxModelPath}");

        _useNativeMode = false;
        _onnxModelPath = onnxModelPath;
        _numFeatures = 64;
        _numFrames = 5;
        _numBlocks = 5;
        _scaleFactor = scaleFactor;
        _imageSize = architecture.InputHeight > 0 ? architecture.InputHeight : 256;
        _lossFunction = new CharbonnierLoss<T>();

        try { _onnxSession = new InferenceSession(onnxModelPath); }
        catch (Exception ex) { throw new InvalidOperationException($"Failed to load ONNX model: {ex.Message}", ex); }

        InitializeLayers();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Enhances video frames with super-resolution and restoration.
    /// </summary>
    public List<Tensor<T>> Enhance(List<Tensor<T>> frames)
    {
        var results = new List<Tensor<T>>();
        int halfWindow = _numFrames / 2;

        for (int i = 0; i < frames.Count; i++)
        {
            var inputFrames = new List<Tensor<T>>();
            for (int j = -halfWindow; j <= halfWindow; j++)
            {
                int idx = MathHelper.Clamp(i + j, 0, frames.Count - 1);
                inputFrames.Add(frames[idx]);
            }
            results.Add(EnhanceFrame(inputFrames));
        }
        return results;
    }

    /// <summary>
    /// Enhances a single frame using neighboring frames.
    /// </summary>
    public Tensor<T> EnhanceFrame(List<Tensor<T>> neighborFrames)
    {
        var stacked = StackFrames(neighborFrames);
        return _useNativeMode ? Forward(stacked) : PredictOnnx(stacked);
    }

    #endregion

    #region Inference

    private Tensor<T> Forward(Tensor<T> input)
    {
        var result = input;
        foreach (var layer in Layers) result = layer.Forward(result);
        return result;
    }

    private Tensor<T> PredictOnnx(Tensor<T> input)
    {
        if (_onnxSession is null) throw new InvalidOperationException("ONNX session is not initialized.");

        var inputData = new float[input.Length];
        for (int i = 0; i < input.Length; i++) inputData[i] = Convert.ToSingle(input.Data.Span[i]);

        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input.Shape);
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_onnxSession.InputMetadata.Keys.First(), onnxInput) };

        using var results = _onnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        var outputData = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++) outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));

        return new Tensor<T>(outputTensor.Dimensions.ToArray(), new Vector<T>(outputData));
    }

    private Tensor<T> StackFrames(List<Tensor<T>> frames)
    {
        var first = frames[0];
        int b = first.Rank == 4 ? first.Shape[0] : 1;
        int c = first.Rank == 4 ? first.Shape[1] : first.Shape[0];
        int h = first.Rank == 4 ? first.Shape[2] : first.Shape[1];
        int w = first.Rank == 4 ? first.Shape[3] : first.Shape[2];

        var stacked = new Tensor<T>([b, c * frames.Count, h, w]);
        for (int f = 0; f < frames.Count; f++)
        {
            Array.Copy(frames[f].Data.ToArray(), 0, stacked.Data.ToArray(), f * c * h * w, c * h * w);
        }
        return stacked;
    }

    public override Tensor<T> Predict(Tensor<T> input) => Forward(input);

    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode) throw new InvalidOperationException("Training is not supported in ONNX mode.");

        var prediction = Predict(input);
        LastLoss = _lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());

        var gradient = _lossFunction.CalculateDerivative(prediction.ToVector(), expectedOutput.ToVector());
        var gradTensor = new Tensor<T>(prediction.Shape, gradient);

        for (int i = Layers.Count - 1; i >= 0; i--) gradTensor = Layers[i].Backward(gradTensor);
        _optimizer?.UpdateParameters(Layers);
    }

    #endregion

    #region Layer Initialization

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) { ClearLayers(); return; }

        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
            Layers.AddRange(Architecture.Layers);
        else
        {
            int ch = Architecture.InputDepth > 0 ? Architecture.InputDepth : 3;
            int h = Architecture.InputHeight > 0 ? Architecture.InputHeight : 256;
            int w = Architecture.InputWidth > 0 ? Architecture.InputWidth : 256;
            Layers.AddRange(LayerHelper<T>.CreateDefaultEDVRLayers(ch, h, w, _numFeatures, _numFrames, 8, _numBlocks));
        }
    }

    #endregion

    #region Serialization

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode) throw new InvalidOperationException("Parameter updates are not supported in ONNX mode.");
        int offset = 0;
        foreach (var layer in Layers)
        {
            var p = layer.GetParameters();
            if (p.Length > 0 && offset + p.Length <= parameters.Length)
            {
                var slice = new Vector<T>(p.Length);
                for (int i = 0; i < p.Length; i++) slice[i] = parameters[offset + i];
                layer.SetParameters(slice);
                offset += p.Length;
            }
        }
    }

    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        ModelType = ModelType.VideoSuperResolution,
        AdditionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "EDVR" }, { "NumFeatures", _numFeatures }, { "NumFrames", _numFrames },
            { "NumBlocks", _numBlocks }, { "ScaleFactor", _scaleFactor }, { "UseNativeMode", _useNativeMode }
        },
        ModelData = _useNativeMode ? this.Serialize() : []
    };

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_numFeatures); writer.Write(_numFrames); writer.Write(_numBlocks); writer.Write(_scaleFactor);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        for (int i = 0; i < 4; i++) _ = reader.ReadInt32();
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() =>
        new EDVR<T>(Architecture, _optimizer, _lossFunction, _numFeatures, _numFrames, _numBlocks, _scaleFactor);

    #endregion
}
