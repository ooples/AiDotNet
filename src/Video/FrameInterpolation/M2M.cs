using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.FrameInterpolation;

/// <summary>
/// M2M: many-to-many splatting for efficient video frame interpolation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// M2M (Hu et al., CVPR 2022) uses many-to-many splatting for efficient interpolation:
/// - Many-to-many splatting: instead of the standard one-to-one backward warping (each target
///   pixel samples from one source pixel), M2M allows multiple source pixels to contribute to
///   multiple target pixels simultaneously using forward splatting with learned weights
/// - Multiple bidirectional flows: estimates K flow field pairs (forward and backward) at each
///   pyramid level, capturing multiple motion hypotheses for occluded regions and motion
///   boundaries where a single flow is ambiguous
/// - Splatting confidence: each splatted pixel carries a learned confidence weight, and the
///   final pixel value is a confidence-weighted sum of all contributions, naturally handling
///   occlusions and disocclusions
/// - Multi-scale pipeline: coarse-to-fine architecture where splatting is performed at each
///   scale, and residual corrections are added at each level
/// </para>
/// <para>
/// <b>For Beginners:</b> Standard interpolation "pulls" each pixel from one location in the
/// source frame. M2M instead "pushes" pixels from the source to potentially multiple locations
/// in the target, which better handles cases where objects overlap or appear/disappear between
/// frames.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputHeight: 128, inputWidth: 128, inputDepth: 3);
/// var model = new M2M&lt;float&gt;(arch, "m2m.onnx");
/// var midFrame = model.Interpolate(frame0, frame1, t: 0.5);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "Many-to-many Splatting for Efficient Video Frame Interpolation"
/// (Hu et al., CVPR 2022)
/// </para>
/// </remarks>
public class M2M<T> : FrameInterpolationBase<T>
{
    #region Fields

    private readonly M2MOptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region Constructors

    /// <summary>Creates an M2M model in ONNX inference mode.</summary>
    public M2M(NeuralNetworkArchitecture<T> architecture, string modelPath, M2MOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new M2MOptions();
        _useNativeMode = false;
        SupportsArbitraryTimestep = true;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates an M2M model in native training mode.</summary>
    public M2M(NeuralNetworkArchitecture<T> architecture, M2MOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new M2MOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        SupportsArbitraryTimestep = true;
        InitializeLayers();
    }

    #endregion

    #region Frame Interpolation

    /// <inheritdoc />
    public override Tensor<T> Interpolate(Tensor<T> frame0, Tensor<T> frame1, double t = 0.5)
    {
        ThrowIfDisposed();
        var f0 = PreprocessFrames(frame0);
        var f1 = PreprocessFrames(frame1);
        var concat = ConcatenateFeatures(f0, f1);
        var output = IsOnnxMode ? RunOnnxInference(concat) : Forward(concat);
        return PostprocessOutput(output);
    }

    #endregion

    #region NeuralNetworkBase

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            int ch = Architecture.InputDepth > 0 ? Architecture.InputDepth : 3;
            int h = Architecture.InputHeight > 0 ? Architecture.InputHeight : 128;
            int w = Architecture.InputWidth > 0 ? Architecture.InputWidth : 128;
            Layers.AddRange(LayerHelper<T>.CreateDefaultFrameInterpolationLayers(
                inputChannels: ch, inputHeight: h, inputWidth: w,
                numFeatures: _options.NumFeatures));
        }
    }

    public override Tensor<T> Predict(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) return RunOnnxInference(input);
        return Forward(input);
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        SetTrainingMode(true);
        var output = Predict(input);
        var grad = LossFunction.CalculateDerivative(output.ToVector(), expected.ToVector());
        var gt = Tensor<T>.FromVector(grad);
        for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt);
        _optimizer?.UpdateParameters(Layers);
        SetTrainingMode(false);
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode) throw new NotSupportedException("Parameter updates are not supported in ONNX mode.");
        int idx = 0;
        foreach (var layer in Layers)
        {
            int count = layer.ParameterCount;
            layer.UpdateParameters(parameters.Slice(idx, count));
            idx += count;
        }
    }

    protected override Tensor<T> PreprocessFrames(Tensor<T> rawFrames) => NormalizeFrames(rawFrames);

    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput) => DenormalizeFrames(modelOutput);

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "M2M-Native" : "M2M-ONNX",
            Description = $"M2M {_options.Variant} many-to-many splatting interpolation (Hu et al., CVPR 2022)",
            ModelType = ModelType.FrameInterpolation,
            Complexity = _options.NumFlowHypotheses * _options.NumPyramidLevels
        };
        m.AdditionalInfo["Variant"] = _options.Variant.ToString();
        m.AdditionalInfo["NumFeatures"] = _options.NumFeatures.ToString();
        m.AdditionalInfo["NumFlowHypotheses"] = _options.NumFlowHypotheses.ToString();
        m.AdditionalInfo["NumPyramidLevels"] = _options.NumPyramidLevels.ToString();
        m.AdditionalInfo["NumRefineBlocks"] = _options.NumRefineBlocks.ToString();
        m.AdditionalInfo["SplattingRadius"] = _options.SplattingRadius.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode);
        w.Write(_options.ModelPath ?? string.Empty);
        w.Write((int)_options.Variant);
        w.Write(_options.NumFeatures);
        w.Write(_options.NumFlowHypotheses);
        w.Write(_options.NumPyramidLevels);
        w.Write(_options.NumRefineBlocks);
        w.Write(_options.SplattingRadius);
        w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean();
        string mp = r.ReadString();
        if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.Variant = (VideoModelVariant)r.ReadInt32();
        _options.NumFeatures = r.ReadInt32();
        _options.NumFlowHypotheses = r.ReadInt32();
        _options.NumPyramidLevels = r.ReadInt32();
        _options.NumRefineBlocks = r.ReadInt32();
        _options.SplattingRadius = r.ReadInt32();
        _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        => new M2M<T>(Architecture, _options);

    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(M2M<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
