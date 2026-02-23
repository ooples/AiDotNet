using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.Enhancement;

/// <summary>
/// Real-ESRGAN Video: practical real-world video super-resolution with temporal consistency.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Real-ESRGAN Video (Wang et al., 2022) extends the image-based Real-ESRGAN to video:
/// - RRDB backbone: Residual-in-Residual Dense Blocks provide per-frame feature extraction
///   with strong representational capacity from densely connected layers
/// - Second-order degradation model: training simulates realistic degradations by applying
///   blur-resize-noise-JPEG twice in sequence, covering a much wider range of real-world
///   artifacts than first-order models
/// - Temporal consistency module: flow-guided feature alignment between adjacent frames
///   followed by temporal aggregation that fuses aligned features with learned attention
/// - U-Net discriminator: provides both global structure feedback and local detail feedback
///   through its multi-scale architecture
/// </para>
/// <para>
/// <b>For Beginners:</b> Real-ESRGAN is one of the most widely-used practical upscaling
/// tools. The video version adds temporal awareness so each upscaled frame looks consistent
/// with its neighbors (no flickering). The key innovation is training with a "double
/// degradation" model -- it learns to handle all the messy artifacts (compression, noise,
/// blur) that real videos have, not just the simple downscaling used in lab benchmarks.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputHeight: 64, inputWidth: 64, inputDepth: 3);
/// var model = new RealESRGANVideo&lt;float&gt;(arch, "realesrgan_video.onnx");
/// var hrFrames = model.Upscale(lrFrames);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure
/// Synthetic Data" (Wang et al., 2022)
/// </para>
/// </remarks>
public class RealESRGANVideo<T> : VideoSuperResolutionBase<T>
{
    #region Fields

    private readonly RealESRGANVideoOptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region Constructors

    /// <summary>Creates a Real-ESRGAN Video model in ONNX inference mode.</summary>
    public RealESRGANVideo(NeuralNetworkArchitecture<T> architecture, string modelPath, RealESRGANVideoOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new RealESRGANVideoOptions();
        _useNativeMode = false;
        ScaleFactor = _options.ScaleFactor;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a Real-ESRGAN Video model in native training mode.</summary>
    public RealESRGANVideo(NeuralNetworkArchitecture<T> architecture, RealESRGANVideoOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new RealESRGANVideoOptions();
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
                numResBlocks: _options.NumRRDBBlocks,
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
            Name = _useNativeMode ? "RealESRGANVideo-Native" : "RealESRGANVideo-ONNX",
            Description = $"Real-ESRGAN Video {_options.Variant} practical blind VSR (Wang et al., 2022)",
            ModelType = ModelType.VideoSuperResolution,
            Complexity = _options.NumRRDBBlocks
        };
        m.AdditionalInfo["Variant"] = _options.Variant.ToString();
        m.AdditionalInfo["NumFeatures"] = _options.NumFeatures.ToString();
        m.AdditionalInfo["NumRRDBBlocks"] = _options.NumRRDBBlocks.ToString();
        m.AdditionalInfo["DenseLayersPerBlock"] = _options.DenseLayersPerBlock.ToString();
        m.AdditionalInfo["ResidualScale"] = _options.ResidualScale.ToString();
        m.AdditionalInfo["ScaleFactor"] = _options.ScaleFactor.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode);
        w.Write(_options.ModelPath ?? string.Empty);
        w.Write((int)_options.Variant);
        w.Write(_options.NumFeatures);
        w.Write(_options.NumRRDBBlocks);
        w.Write(_options.ScaleFactor);
        w.Write(_options.DenseLayersPerBlock);
        w.Write(_options.ResidualScale);
        w.Write(_options.PerceptualWeight);
        w.Write(_options.GANWeight);
        w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean();
        string mp = r.ReadString();
        if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.Variant = (VideoModelVariant)r.ReadInt32();
        _options.NumFeatures = r.ReadInt32();
        _options.NumRRDBBlocks = r.ReadInt32();
        _options.ScaleFactor = r.ReadInt32();
        _options.DenseLayersPerBlock = r.ReadInt32();
        _options.ResidualScale = r.ReadDouble();
        _options.PerceptualWeight = r.ReadDouble();
        _options.GANWeight = r.ReadDouble();
        _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            return new RealESRGANVideo<T>(Architecture, p, _options);
        return new RealESRGANVideo<T>(Architecture, _options);
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(RealESRGANVideo<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
