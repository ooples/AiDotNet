using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.Enhancement;

/// <summary>
/// DualX-VSR: dual axial spatial-temporal transformer without motion compensation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DualX-VSR (2025) eliminates explicit motion compensation through dual axial attention:
/// - Dual axial attention: decomposes 3D attention into two orthogonal axis pairs
///   (height-temporal and width-temporal), capturing full spatial-temporal context with
///   linear complexity instead of cubic
/// - Motion-free alignment: the crossed axial attention patterns implicitly capture
///   inter-frame correspondence without computing optical flow or deformable offsets
/// - Symmetric bidirectional propagation: features propagate both forward and backward
///   in time with shared axial attention weights
/// - Each dual axial block performs height-temporal attention followed by width-temporal
///   attention, ensuring every position can attend to any other position in the 3D volume
///   through the composition of two axis-aligned operations
/// </para>
/// <para>
/// <b>For Beginners:</b> Most video SR models need to figure out how objects moved between
/// frames (optical flow). DualX-VSR skips this step by using a clever attention pattern
/// that looks along two crossing axes simultaneously. Like reading a crossword puzzle by
/// checking both across and down -- you understand the full picture without tracing each
/// letter's path.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputHeight: 64, inputWidth: 64, inputDepth: 3);
/// var model = new DualXVSR&lt;float&gt;(arch, "dualxvsr.onnx");
/// var hrFrames = model.Upscale(lrFrames);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "DualX-VSR: Dual Axial Spatial-Temporal Transformer for Video
/// Super-Resolution" (2025)
/// </para>
/// </remarks>
public class DualXVSR<T> : VideoSuperResolutionBase<T>
{
    #region Fields

    private readonly DualXVSROptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region Constructors

    /// <summary>Creates a DualX-VSR model in ONNX inference mode.</summary>
    public DualXVSR(NeuralNetworkArchitecture<T> architecture, string modelPath, DualXVSROptions? options = null)
        : base(architecture)
    {
        _options = options ?? new DualXVSROptions();
        _useNativeMode = false;
        ScaleFactor = _options.ScaleFactor;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a DualX-VSR model in native training mode.</summary>
    public DualXVSR(NeuralNetworkArchitecture<T> architecture, DualXVSROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new DualXVSROptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        ScaleFactor = _options.ScaleFactor;
        InitializeLayers();
    }

    #endregion

    #region Video Super-Resolution

    /// <inheritdoc />
    public override Tensor<T> Upscale(Tensor<T> lowResFrames)
    {
        ThrowIfDisposed();
        var preprocessed = PreprocessFrames(lowResFrames);
        var output = IsOnnxMode ? RunOnnxInference(preprocessed) : Forward(preprocessed);
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
            int h = Architecture.InputHeight > 0 ? Architecture.InputHeight : 64;
            int w = Architecture.InputWidth > 0 ? Architecture.InputWidth : 64;
            Layers.AddRange(LayerHelper<T>.CreateDefaultVideoSuperResolutionLayers(
                inputChannels: ch, inputHeight: h, inputWidth: w,
                numFeatures: _options.NumFeatures,
                numResBlocks: _options.NumAxialBlocks,
                scaleFactor: _options.ScaleFactor));
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
            Name = _useNativeMode ? "DualXVSR-Native" : "DualXVSR-ONNX",
            Description = $"DualX-VSR {_options.Variant} dual axial spatial-temporal transformer VSR (2025)",
            ModelType = ModelType.VideoSuperResolution,
            Complexity = _options.NumAxialBlocks
        };
        m.AdditionalInfo["Variant"] = _options.Variant.ToString();
        m.AdditionalInfo["NumFeatures"] = _options.NumFeatures.ToString();
        m.AdditionalInfo["NumAxialBlocks"] = _options.NumAxialBlocks.ToString();
        m.AdditionalInfo["NumHeads"] = _options.NumHeads.ToString();
        m.AdditionalInfo["TemporalWindow"] = _options.TemporalWindow.ToString();
        m.AdditionalInfo["ScaleFactor"] = _options.ScaleFactor.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode);
        w.Write(_options.ModelPath ?? string.Empty);
        w.Write((int)_options.Variant);
        w.Write(_options.NumFeatures);
        w.Write(_options.NumAxialBlocks);
        w.Write(_options.ScaleFactor);
        w.Write(_options.NumHeads);
        w.Write(_options.TemporalWindow);
        w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean();
        string mp = r.ReadString();
        if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.Variant = (VideoModelVariant)r.ReadInt32();
        _options.NumFeatures = r.ReadInt32();
        _options.NumAxialBlocks = r.ReadInt32();
        _options.ScaleFactor = r.ReadInt32();
        _options.NumHeads = r.ReadInt32();
        _options.TemporalWindow = r.ReadInt32();
        _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        => new DualXVSR<T>(Architecture, _options);

    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(DualXVSR<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
