using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.FrameInterpolation;

/// <summary>
/// SwinVFI: Swin Transformer-based video frame interpolation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SwinVFI (2022) applies Swin Transformer architecture to frame interpolation:
/// - Swin Transformer encoder: uses shifted-window self-attention with linear complexity for
///   encoding input frame pairs at high resolution
/// - Cross-frame window attention: extends shifted-window attention to cross-attend between
///   features from both input frames, capturing inter-frame correspondences
/// - Hierarchical feature pyramid: multi-scale feature extraction with Swin blocks at each level
/// - Flow-free synthesis: directly synthesizes the intermediate frame from cross-attended
///   features without explicit optical flow estimation
/// </para>
/// <para>
/// <b>For Beginners:</b> SwinVFI uses the Swin Transformer to look at both input frames
/// simultaneously and figure out what goes between them, without needing to estimate motion.
/// The "shifted window" approach makes it efficient for full-resolution video frames.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputHeight: 128, inputWidth: 128, inputDepth: 3);
/// var model = new SwinVFI&lt;float&gt;(arch, "swinvfi.onnx");
/// var midFrame = model.Interpolate(frame0, frame1, t: 0.5);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "SwinVFI: Swin Transformer-based Video Frame Interpolation" (2022)
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.FrameInterpolation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Video Frame Interpolation with Transformer",
    "https://arxiv.org/abs/2205.07230",
    Year = 2022,
    Authors = "Liying Lu, Ruizheng Wu, Huaijia Lin, Jiangbo Lu, Jiaya Jia")]
public partial class SwinVFI<T> : FrameInterpolationBase<T>
{
    #region Fields

    private readonly SwinVFIOptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region Constructors

    /// <summary>Creates a SwinVFI model in ONNX inference mode.</summary>
    public SwinVFI(NeuralNetworkArchitecture<T> architecture, string modelPath, SwinVFIOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new SwinVFIOptions();
        _useNativeMode = false;
        SupportsArbitraryTimestep = false;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a SwinVFI model in native training mode.</summary>
    public SwinVFI(NeuralNetworkArchitecture<T> architecture, SwinVFIOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new SwinVFIOptions();
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
        if (t < 0.0 || t > 1.0)
            throw new ArgumentOutOfRangeException(nameof(t), t, "Timestep must be in [0, 1].");
        if (!SupportsArbitraryTimestep && Math.Abs(t - 0.5) > 1e-6)
            throw new NotSupportedException("SwinVFI only supports midpoint interpolation (t=0.5).");
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
            Layers.AddRange(Architecture.Layers);
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

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) return RunOnnxInference(input);
        return Forward(input);
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        SetTrainingMode(true);
        try
        {
            TrainWithTape(input, expected, _optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    // Identity: tape training runs the raw layer stack (no NormalizeFrames) and the sigmoid head
    // emits [0,1] frames, so /255+*255 only on inference was a train/eval mismatch (MoreData).
    protected override Tensor<T> PreprocessFrames(Tensor<T> rawFrames) => rawFrames;
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput) => modelOutput;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "SwinVFI-Native" : "SwinVFI-ONNX",
            Description = $"SwinVFI {_options.Variant} Swin Transformer interpolation (2022)",
            Complexity = _options.NumSwinBlocks * _options.NumStages
        };
        m.AdditionalInfo["Variant"] = _options.Variant.ToString();
        m.AdditionalInfo["NumFeatures"] = _options.NumFeatures.ToString();
        m.AdditionalInfo["NumSwinBlocks"] = _options.NumSwinBlocks.ToString();
        m.AdditionalInfo["NumHeads"] = _options.NumHeads.ToString();
        m.AdditionalInfo["WindowSize"] = _options.WindowSize.ToString();
        m.AdditionalInfo["NumStages"] = _options.NumStages.ToString();
        return m;
    }





    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(SwinVFI<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
