using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.Enhancement;

/// <summary>
/// PSRT: progressive spatio-temporal alignment with window-based attention for video SR.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// PSRT (Shi et al., 2022) uses progressive window-based spatio-temporal attention:
/// - Spatio-temporal attention blocks (STABs): joint spatial and temporal attention within
///   3D windows (height x width x time) for aligned multi-frame feature fusion
/// - Progressive alignment: a coarse-to-fine encoder-decoder structure where early layers
///   capture large motions with downsampled features, and later layers refine sub-pixel
///   alignment at full resolution
/// - Window-based attention: limits attention to local spatio-temporal windows with shifted
///   window partitioning for cross-window information flow (Swin-style)
/// - Temporal mutual attention: cross-attention between the reference frame and each
///   supporting frame to explicitly align temporal features
/// </para>
/// <para>
/// <b>For Beginners:</b> PSRT aligns video frames step by step, starting with big motion
/// corrections (like camera shake) and progressively refining small details (sub-pixel
/// alignment). It uses "windows" in both space and time to efficiently find matching regions
/// across frames, similar to how Swin Transformer works but extended to 3D video volumes.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputHeight: 64, inputWidth: 64, inputDepth: 3);
/// var model = new PSRT&lt;float&gt;(arch, "psrt.onnx");
/// var hrFrames = model.Upscale(lrFrames);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "PSRT: Progressive Spatio-temporal Alignment for Video
/// Super-Resolution" (Shi et al., 2022)
/// </para>
/// </remarks>
public class PSRT<T> : VideoSuperResolutionBase<T>
{
    #region Fields

    private readonly PSRTOptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region Constructors

    /// <summary>Creates a PSRT model in ONNX inference mode.</summary>
    public PSRT(NeuralNetworkArchitecture<T> architecture, string modelPath, PSRTOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new PSRTOptions();
        _useNativeMode = false;
        ScaleFactor = _options.ScaleFactor;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a PSRT model in native training mode.</summary>
    public PSRT(NeuralNetworkArchitecture<T> architecture, PSRTOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new PSRTOptions();
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
                numResBlocks: _options.NumSTABs,
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
            Name = _useNativeMode ? "PSRT-Native" : "PSRT-ONNX",
            Description = $"PSRT {_options.Variant} progressive spatio-temporal alignment VSR (Shi et al., 2022)",
            ModelType = ModelType.VideoSuperResolution,
            Complexity = _options.NumSTABs
        };
        m.AdditionalInfo["Variant"] = _options.Variant.ToString();
        m.AdditionalInfo["NumFeatures"] = _options.NumFeatures.ToString();
        m.AdditionalInfo["NumSTABs"] = _options.NumSTABs.ToString();
        m.AdditionalInfo["WindowSize"] = _options.WindowSize.ToString();
        m.AdditionalInfo["TemporalRadius"] = _options.TemporalRadius.ToString();
        m.AdditionalInfo["NumHeads"] = _options.NumHeads.ToString();
        m.AdditionalInfo["ScaleFactor"] = _options.ScaleFactor.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode);
        w.Write(_options.ModelPath ?? string.Empty);
        w.Write((int)_options.Variant);
        w.Write(_options.NumFeatures);
        w.Write(_options.NumSTABs);
        w.Write(_options.ScaleFactor);
        w.Write(_options.WindowSize);
        w.Write(_options.TemporalRadius);
        w.Write(_options.NumHeads);
        w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean();
        string mp = r.ReadString();
        if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.Variant = (VideoModelVariant)r.ReadInt32();
        _options.NumFeatures = r.ReadInt32();
        _options.NumSTABs = r.ReadInt32();
        _options.ScaleFactor = r.ReadInt32();
        _options.WindowSize = r.ReadInt32();
        _options.TemporalRadius = r.ReadInt32();
        _options.NumHeads = r.ReadInt32();
        _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        => new PSRT<T>(Architecture, _options);

    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(PSRT<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
