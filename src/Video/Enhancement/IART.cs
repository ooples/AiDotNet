using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.Enhancement;

/// <summary>
/// IART: implicit resampling-based alignment transformer for video super-resolution.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// IART (Kai et al., CVPR 2024 Highlight) uses implicit neural representations for alignment:
/// - Implicit resampling: instead of warping features to discrete grid positions (which
///   causes interpolation artifacts), IART uses a continuous implicit function to sample
///   features at arbitrary sub-pixel positions with learned kernels
/// - Alignment transformer: cross-attention between reference and supporting frames where
///   sampling positions are offset by flow-guided implicit coordinates, achieving sub-pixel
///   accurate alignment without grid discretization
/// - Multi-scale implicit alignment: alignment at multiple feature resolutions, from
///   coarse structural alignment to fine texture-level resampling
/// - High-frequency preservation: the implicit function preserves sharp edges, thin lines,
///   and fine textures that grid-based bilinear/bicubic warping typically blurs
/// </para>
/// <para>
/// <b>For Beginners:</b> When aligning video frames, most models "warp" one frame to match
/// another using a grid. This can blur fine details because pixel positions don't perfectly
/// line up with grid points. IART solves this by learning a continuous function that can
/// read features at ANY position (not just grid points), like being able to zoom into
/// any spot on a map with perfect clarity rather than being limited to the printed grid.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputHeight: 64, inputWidth: 64, inputDepth: 3);
/// var model = new IART&lt;float&gt;(arch, "iart.onnx");
/// var hrFrames = model.Upscale(lrFrames);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "IART: Implicit Resampling-based Alignment Transformer for Video
/// Super-Resolution" (Kai et al., CVPR 2024 Highlight)
/// </para>
/// </remarks>
public class IART<T> : VideoSuperResolutionBase<T>
{
    #region Fields

    private readonly IARTOptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region Constructors

    /// <summary>Creates an IART model in ONNX inference mode.</summary>
    public IART(NeuralNetworkArchitecture<T> architecture, string modelPath, IARTOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new IARTOptions();
        _useNativeMode = false;
        ScaleFactor = _options.ScaleFactor;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates an IART model in native training mode.</summary>
    public IART(NeuralNetworkArchitecture<T> architecture, IARTOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new IARTOptions();
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
                numResBlocks: _options.NumResBlocks,
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
            Name = _useNativeMode ? "IART-Native" : "IART-ONNX",
            Description = $"IART {_options.Variant} implicit resampling alignment transformer VSR (Kai et al., CVPR 2024)",
            ModelType = ModelType.VideoSuperResolution,
            Complexity = _options.NumTransformerBlocks
        };
        m.AdditionalInfo["Variant"] = _options.Variant.ToString();
        m.AdditionalInfo["NumFeatures"] = _options.NumFeatures.ToString();
        m.AdditionalInfo["NumTransformerBlocks"] = _options.NumTransformerBlocks.ToString();
        m.AdditionalInfo["NumHeads"] = _options.NumHeads.ToString();
        m.AdditionalInfo["NumScales"] = _options.NumScales.ToString();
        m.AdditionalInfo["ImplicitDim"] = _options.ImplicitDim.ToString();
        m.AdditionalInfo["ScaleFactor"] = _options.ScaleFactor.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode);
        w.Write(_options.ModelPath ?? string.Empty);
        w.Write((int)_options.Variant);
        w.Write(_options.NumFeatures);
        w.Write(_options.NumTransformerBlocks);
        w.Write(_options.NumResBlocks);
        w.Write(_options.ScaleFactor);
        w.Write(_options.NumHeads);
        w.Write(_options.NumScales);
        w.Write(_options.ImplicitDim);
        w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean();
        string mp = r.ReadString();
        if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.Variant = (VideoModelVariant)r.ReadInt32();
        _options.NumFeatures = r.ReadInt32();
        _options.NumTransformerBlocks = r.ReadInt32();
        _options.NumResBlocks = r.ReadInt32();
        _options.ScaleFactor = r.ReadInt32();
        _options.NumHeads = r.ReadInt32();
        _options.NumScales = r.ReadInt32();
        _options.ImplicitDim = r.ReadInt32();
        _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        => new IART<T>(Architecture, _options);

    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(IART<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
