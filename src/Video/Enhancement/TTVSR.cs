using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.Enhancement;

/// <summary>
/// TTVSR: learning trajectory-aware transformer for long-range video super-resolution.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// TTVSR (Liu et al., ECCV 2022) learns trajectory-aware features for temporal modeling:
/// - Trajectory-aware attention: instead of attending to fixed spatial locations across
///   frames, attention follows estimated motion trajectories so features are gathered along
///   the actual path each visual element traveled
/// - Cross-scale feature tokenization: visual tokens are extracted at multiple spatial
///   scales and fused, capturing both fine textures and coarse structures simultaneously
/// - Location map: a learned spatial routing map that helps the transformer locate the
///   correct trajectory positions across the full video sequence
/// - Long-range modeling: trajectories span the full video, not just adjacent frames,
///   enabling information flow from temporally distant frames along motion paths
/// </para>
/// <para>
/// <b>For Beginners:</b> Imagine following a ball as it moves across frames. Instead of
/// looking at the same fixed spot in every frame (which would miss the ball after it
/// moves), TTVSR tracks where objects actually go and gathers information along their
/// travel path. This is much more effective because real video has complex motion, and
/// the best information for upscaling a pixel often comes from a completely different
/// position in neighboring frames.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputHeight: 64, inputWidth: 64, inputDepth: 3);
/// var model = new TTVSR&lt;float&gt;(arch, "ttvsr.onnx");
/// var hrFrames = model.Upscale(lrFrames);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "TTVSR: Learning Trajectory-Aware Transformer for Video
/// Super-Resolution" (Liu et al., ECCV 2022)
/// </para>
/// </remarks>
public class TTVSR<T> : VideoSuperResolutionBase<T>
{
    #region Fields

    private readonly TTVSROptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region Constructors

    /// <summary>Creates a TTVSR model in ONNX inference mode.</summary>
    public TTVSR(NeuralNetworkArchitecture<T> architecture, string modelPath, TTVSROptions? options = null)
        : base(architecture)
    {
        _options = options ?? new TTVSROptions();
        _useNativeMode = false;
        ScaleFactor = _options.ScaleFactor;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a TTVSR model in native training mode.</summary>
    public TTVSR(NeuralNetworkArchitecture<T> architecture, TTVSROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new TTVSROptions();
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
            Name = _useNativeMode ? "TTVSR-Native" : "TTVSR-ONNX",
            Description = $"TTVSR {_options.Variant} trajectory-aware transformer VSR (Liu et al., ECCV 2022)",
            ModelType = ModelType.VideoSuperResolution,
            Complexity = _options.NumTransformerBlocks
        };
        m.AdditionalInfo["Variant"] = _options.Variant.ToString();
        m.AdditionalInfo["NumFeatures"] = _options.NumFeatures.ToString();
        m.AdditionalInfo["NumTransformerBlocks"] = _options.NumTransformerBlocks.ToString();
        m.AdditionalInfo["TrajectoryLength"] = _options.TrajectoryLength.ToString();
        m.AdditionalInfo["NumHeads"] = _options.NumHeads.ToString();
        m.AdditionalInfo["NumScales"] = _options.NumScales.ToString();
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
        w.Write(_options.TrajectoryLength);
        w.Write(_options.NumHeads);
        w.Write(_options.NumScales);
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
        _options.TrajectoryLength = r.ReadInt32();
        _options.NumHeads = r.ReadInt32();
        _options.NumScales = r.ReadInt32();
        _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
        => new TTVSR<T>(Architecture, _options);

    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(TTVSR<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
