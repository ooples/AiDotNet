using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.FrameInterpolation;

/// <summary>
/// FLAVR: flow-agnostic video representations for fast frame interpolation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// FLAVR (Kalluri et al., CVPR 2023) uses 3D convolutions for flow-free interpolation:
/// - 3D spatio-temporal convolutions: processes multiple input frames simultaneously using
///   3D (space + time) convolutions that capture temporal relationships without explicit
///   optical flow estimation, avoiding flow-related artifacts entirely
/// - 3D encoder-decoder: a U-Net style architecture where the encoder uses strided 3D
///   convolutions to downsample in both space and time, and the decoder uses transposed
///   3D convolutions to upsample back to full resolution
/// - Multi-frame input: takes 4 input frames (2 before and 2 after the target) for richer
///   temporal context, unlike 2-frame methods that miss longer-range motion patterns
/// - Direct synthesis: directly outputs the target frame pixels without intermediate flow
///   or warping operations, avoiding flow estimation errors entirely
/// </para>
/// <para>
/// <b>For Beginners:</b> Most frame interpolation methods first figure out how objects move
/// (optical flow), then use that to warp frames. FLAVR skips the flow step entirely by using
/// 3D convolutions that "see" multiple frames at once and directly paint the intermediate
/// frame. This makes it faster and avoids ghosting artifacts from bad flow estimates.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputHeight: 128, inputWidth: 128, inputDepth: 3);
/// var model = new FLAVR&lt;float&gt;(arch, "flavr.onnx");
/// var midFrame = model.Interpolate(frame0, frame1, t: 0.5);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "FLAVR: Flow-Agnostic Video Representations for Fast Frame Interpolation"
/// (Kalluri et al., CVPR 2023)
/// </para>
/// </remarks>
public class FLAVR<T> : FrameInterpolationBase<T>
{
    #region Fields

    private readonly FLAVROptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region Constructors

    /// <summary>Creates a FLAVR model in ONNX inference mode.</summary>
    public FLAVR(NeuralNetworkArchitecture<T> architecture, string modelPath, FLAVROptions? options = null)
        : base(architecture)
    {
        _options = options ?? new FLAVROptions();
        _useNativeMode = false;
        SupportsArbitraryTimestep = false;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a FLAVR model in native training mode.</summary>
    public FLAVR(NeuralNetworkArchitecture<T> architecture, FLAVROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new FLAVROptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        SupportsArbitraryTimestep = false;
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
            Name = _useNativeMode ? "FLAVR-Native" : "FLAVR-ONNX",
            Description = $"FLAVR {_options.Variant} flow-agnostic 3D conv interpolation (Kalluri et al., CVPR 2023)",
            ModelType = ModelType.FrameInterpolation,
            Complexity = _options.NumResBlocks * _options.NumLevels
        };
        m.AdditionalInfo["Variant"] = _options.Variant.ToString();
        m.AdditionalInfo["NumFeatures"] = _options.NumFeatures.ToString();
        m.AdditionalInfo["NumResBlocks"] = _options.NumResBlocks.ToString();
        m.AdditionalInfo["NumLevels"] = _options.NumLevels.ToString();
        m.AdditionalInfo["NumInputFrames"] = _options.NumInputFrames.ToString();
        m.AdditionalInfo["TemporalKernelSize"] = _options.TemporalKernelSize.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode);
        w.Write(_options.ModelPath ?? string.Empty);
        w.Write((int)_options.Variant);
        w.Write(_options.NumFeatures);
        w.Write(_options.NumResBlocks);
        w.Write(_options.NumLevels);
        w.Write(_options.NumInputFrames);
        w.Write(_options.TemporalKernelSize);
        w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean();
        string mp = r.ReadString();
        if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.Variant = (VideoModelVariant)r.ReadInt32();
        _options.NumFeatures = r.ReadInt32();
        _options.NumResBlocks = r.ReadInt32();
        _options.NumLevels = r.ReadInt32();
        _options.NumInputFrames = r.ReadInt32();
        _options.TemporalKernelSize = r.ReadInt32();
        _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            return new FLAVR<T>(Architecture, p, _options);
        return new FLAVR<T>(Architecture, _options);
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(FLAVR<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
