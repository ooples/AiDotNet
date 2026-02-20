using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.FrameInterpolation;

/// <summary>
/// IFRNet: intermediate feature refine network for efficient frame interpolation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// IFRNet (Kong et al., CVPR 2022) uses coarse-to-fine intermediate feature refinement:
/// - Encoder-decoder with skip connections: shared encoder extracts multi-scale features
///   from both input frames, decoder progressively refines the interpolation result from
///   coarsest to finest scale
/// - Intermediate feature refinement (IFR): at each decoder level, instead of refining the
///   optical flow, IFRNet directly refines the intermediate features of the target frame,
///   avoiding error accumulation from iterated flow estimation
/// - Coarse-to-fine pyramid: 3-level pyramid where each level operates at half the resolution
///   of the next, with learned upsampling and skip connections between levels
/// - Task-oriented flow: optical flow is used only as an initial guide for feature warping,
///   then discarded in favor of direct feature refinement which is more robust to flow errors
/// </para>
/// <para>
/// <b>For Beginners:</b> Most frame interpolation methods first estimate motion (optical flow),
/// then use it to warp frames. If the flow is wrong, the result is wrong. IFRNet takes a
/// different approach: it starts with a rough flow estimate, uses it to get initial features,
/// then directly refines those features to produce the final frame. This is more forgiving
/// of flow errors and produces sharper results with fewer artifacts.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputHeight: 128, inputWidth: 128, inputDepth: 3);
/// var model = new IFRNet&lt;float&gt;(arch, "ifrnet.onnx");
/// var midFrame = model.Interpolate(frame0, frame1, t: 0.5);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "IFRNet: Intermediate Feature Refine Network for Efficient Frame
/// Interpolation" (Kong et al., CVPR 2022)
/// </para>
/// </remarks>
public class IFRNet<T> : FrameInterpolationBase<T>
{
    #region Fields

    private readonly IFRNetOptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region Constructors

    /// <summary>Creates an IFRNet model in ONNX inference mode.</summary>
    public IFRNet(NeuralNetworkArchitecture<T> architecture, string modelPath, IFRNetOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new IFRNetOptions();
        _useNativeMode = false;
        SupportsArbitraryTimestep = true;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates an IFRNet model in native training mode.</summary>
    public IFRNet(NeuralNetworkArchitecture<T> architecture, IFRNetOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new IFRNetOptions();
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
            Name = _useNativeMode ? "IFRNet-Native" : "IFRNet-ONNX",
            Description = $"IFRNet {_options.Variant} intermediate feature refinement (Kong et al., CVPR 2022)",
            ModelType = ModelType.FrameInterpolation,
            Complexity = _options.NumRefineBlocks * _options.NumPyramidLevels
        };
        m.AdditionalInfo["Variant"] = _options.Variant.ToString();
        m.AdditionalInfo["NumFeatures"] = _options.NumFeatures.ToString();
        m.AdditionalInfo["NumRefineBlocks"] = _options.NumRefineBlocks.ToString();
        m.AdditionalInfo["NumPyramidLevels"] = _options.NumPyramidLevels.ToString();
        m.AdditionalInfo["UseTaskOrientedFlow"] = _options.UseTaskOrientedFlow.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode);
        w.Write(_options.ModelPath ?? string.Empty);
        w.Write((int)_options.Variant);
        w.Write(_options.NumFeatures);
        w.Write(_options.NumRefineBlocks);
        w.Write(_options.NumPyramidLevels);
        w.Write(_options.UseTaskOrientedFlow);
        w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean();
        string mp = r.ReadString();
        if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.Variant = (VideoModelVariant)r.ReadInt32();
        _options.NumFeatures = r.ReadInt32();
        _options.NumRefineBlocks = r.ReadInt32();
        _options.NumPyramidLevels = r.ReadInt32();
        _options.UseTaskOrientedFlow = r.ReadBoolean();
        _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
        else if (_useNativeMode)
        {
            Layers.Clear();
            InitializeLayers();
        }
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            return new IFRNet<T>(Architecture, p, _options);
        return new IFRNet<T>(Architecture, _options);
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(IFRNet<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
