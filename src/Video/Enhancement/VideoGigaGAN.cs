using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.Enhancement;

/// <summary>
/// VideoGigaGAN: towards detail-rich video super-resolution with large-scale GAN.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// VideoGigaGAN (Xu et al., CVPR 2025) is the first large-scale GAN for video SR:
/// - GigaGAN backbone: upscaled StyleGAN architecture with 1B+ parameters, generating
///   exceptional spatial detail in a single forward pass (faster than diffusion models)
/// - Feature propagation with anti-aliasing: temporal feature propagation uses anti-aliased
///   flow warping to prevent temporal aliasing artifacts that cause flickering
/// - High-frequency shuttle: a dedicated parallel pathway that extracts and preserves
///   genuine high-frequency details (edges, textures, text) from the input, preventing the
///   generator from hallucinating false details while maintaining real ones
/// - Temporal discriminator: a 3D discriminator evaluates both per-frame quality and
///   temporal consistency, penalizing flickering and motion artifacts
/// - Supports up to 8x upscaling with rich perceptual details
///
/// <b>Note:</b> The full VideoGigaGAN architecture (GigaGAN backbone, high-frequency shuttle,
/// temporal discriminator, anti-aliased flow warping) is available through ONNX inference mode.
/// Native training mode uses a simplified baseline encoder-decoder for research and fine-tuning.
/// </para>
/// <para>
/// <b>For Beginners:</b> VideoGigaGAN is like a very talented speed-painter. While
/// diffusion models gradually "develop" a high-res image from noise (slow but versatile),
/// VideoGigaGAN directly "paints" detail in one stroke -- much faster. A special "detail
/// shuttle" ensures real details are preserved and fake ones aren't invented, while
/// anti-aliasing prevents the annoying flickering between frames.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputHeight: 64, inputWidth: 64, inputDepth: 3);
/// var model = new VideoGigaGAN&lt;float&gt;(arch, "videogigagan.onnx");
/// var hrFrames = model.Upscale(lrFrames);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "VideoGigaGAN: Towards Detail-rich Video Super-Resolution"
/// (Xu et al., CVPR 2025)
/// </para>
/// </remarks>
public class VideoGigaGAN<T> : VideoSuperResolutionBase<T>
{
    #region Fields

    private readonly VideoGigaGANOptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region Constructors

    /// <summary>Creates a VideoGigaGAN model in ONNX inference mode.</summary>
    public VideoGigaGAN(NeuralNetworkArchitecture<T> architecture, string modelPath, VideoGigaGANOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new VideoGigaGANOptions();
        _useNativeMode = false;
        ScaleFactor = _options.ScaleFactor;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a VideoGigaGAN model in native training mode.</summary>
    public VideoGigaGAN(NeuralNetworkArchitecture<T> architecture, VideoGigaGANOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new VideoGigaGANOptions();
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
            Name = _useNativeMode ? "VideoGigaGAN-Native" : "VideoGigaGAN-ONNX",
            Description = $"VideoGigaGAN {_options.Variant} large-scale GAN VSR (Xu et al., CVPR 2025)",
            ModelType = ModelType.VideoSuperResolution,
            Complexity = _options.NumResBlocks
        };
        m.AdditionalInfo["Variant"] = _options.Variant.ToString();
        m.AdditionalInfo["NumFeatures"] = _options.NumFeatures.ToString();
        m.AdditionalInfo["NumResBlocks"] = _options.NumResBlocks.ToString();
        m.AdditionalInfo["NumStyleLayers"] = _options.NumStyleLayers.ToString();
        m.AdditionalInfo["PerceptualWeight"] = _options.PerceptualWeight.ToString();
        m.AdditionalInfo["GANWeight"] = _options.GANWeight.ToString();
        m.AdditionalInfo["ScaleFactor"] = _options.ScaleFactor.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode);
        w.Write(_options.ModelPath ?? string.Empty);
        w.Write((int)_options.Variant);
        w.Write(_options.NumFeatures);
        w.Write(_options.NumResBlocks);
        w.Write(_options.ScaleFactor);
        w.Write(_options.NumStyleLayers);
        w.Write(_options.PerceptualWeight);
        w.Write(_options.GANWeight);
        w.Write(_options.HFShuttleWeight);
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
        _options.ScaleFactor = r.ReadInt32();
        _options.NumStyleLayers = r.ReadInt32();
        _options.PerceptualWeight = r.ReadDouble();
        _options.GANWeight = r.ReadDouble();
        _options.HFShuttleWeight = r.ReadDouble();
        _options.DropoutRate = r.ReadDouble();
        ScaleFactor = _options.ScaleFactor;
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
            return new VideoGigaGAN<T>(Architecture, p, _options);
        return new VideoGigaGAN<T>(Architecture, _options);
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(VideoGigaGAN<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        if (disposing) OnnxModel?.Dispose();
        base.Dispose(disposing);
    }

    #endregion
}
