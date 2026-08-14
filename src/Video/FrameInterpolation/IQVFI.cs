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
/// IQ-VFI: image quality-aware video frame interpolation with degradation adaptation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// IQ-VFI (2024) adapts interpolation based on input image quality:
/// - Quality assessment module: estimates per-pixel quality scores for input frames using a
///   learned no-reference image quality assessment (NR-IQA) branch, identifying regions with
///   noise, blur, compression artifacts, or other degradations
/// - Degradation-adaptive flow: the optical flow estimation network receives quality maps as
///   additional conditioning, so it can be more conservative in degraded regions (where flow
///   estimation is unreliable) and more aggressive in clean regions
/// - Quality-guided fusion: the blending weights between warped frames incorporate quality
///   scores, favoring the higher-quality frame contribution in each spatial region
/// - Quality-aware training: training uses a quality-stratified sampling strategy that ensures
///   the model sees diverse degradation levels and learns robust interpolation for each
/// </para>
/// <para>
/// <b>For Beginners:</b> Most frame interpolation methods assume input frames are clean and
/// high-quality. IQ-VFI first checks how good each part of the input frames is, then adjusts
/// its interpolation strategy accordingly. This means it works better on real-world videos
/// that may have noise, blur, or compression artifacts.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputHeight: 128, inputWidth: 128, inputDepth: 3);
/// var model = new IQVFI&lt;float&gt;(arch, "iqvfi.onnx");
/// var midFrame = model.Interpolate(frame0, frame1, t: 0.5);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "IQ-VFI: Image Quality-Aware Video Frame Interpolation" (2024)
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.FrameInterpolation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
// CITATION CORRECTED. This model previously cited "IQ-VFI: Image Quality-Aware Video Frame
// Interpolation" at arXiv:2408.11108, attributed to Xinyi Zhang, Fanghua Yu, Jie Huang and Feng Zhao.
// Every part of that was wrong: arXiv:2408.11108 is "The densities in diffuse and translucent
// molecular clouds", an ASTROPHYSICS paper on interstellar cloud densities by Neufeld et al. The "IQ"
// in IQ-VFI is Implicit Quadratic, not Image Quality — and the model had been built to the misread
// title, which is why its options carried NumQualityBlocks and QualityThreshold while the paper's
// actual mechanism (acceleration-modulated quadratic motion) was absent entirely.
[ResearchPaper("IQ-VFI: Implicit Quadratic Motion Estimation for Video Frame Interpolation",
    "https://openaccess.thecvf.com/content/CVPR2024/papers/Hu_IQ-VFI_Implicit_Quadratic_Motion_Estimation_for_Video_Frame_Interpolation_CVPR_2024_paper.pdf",
    Year = 2024,
    Authors = "Mengshun Hu, Kui Jiang, Zhihang Zhong, Zheng Wang, Yinqiang Zheng")]
public partial class IQVFI<T> : FrameInterpolationBase<T>
{
    /// <summary>
    /// Gets the implicit quadratic motion model, which modulates linear intermediate flows into
    /// quadratic ones using a latent acceleration prior.
    /// </summary>
    /// <remarks>
    /// This is the mechanism the paper is named for and it was entirely missing before the citation was
    /// corrected. See <see cref="ImplicitQuadraticMotion{T}"/>.
    /// </remarks>
    public ImplicitQuadraticMotion<T> QuadraticMotion { get; } = new();

    /// <summary>
    /// Gets the knowledge-distillation objective: pyramid reconstruction plus acceleration and motion
    /// distillation from a privileged teacher, gated by the selective mask.
    /// </summary>
    public IQVFIDistillation<T> Distillation { get; } = new();

    #region Fields

    private readonly IQVFIOptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region Constructors

    /// <summary>Creates an IQ-VFI model in ONNX inference mode.</summary>
    public IQVFI(NeuralNetworkArchitecture<T> architecture, string modelPath, IQVFIOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new IQVFIOptions();
        _useNativeMode = false;
        SupportsArbitraryTimestep = true;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates an IQ-VFI model in native training mode.</summary>
    public IQVFI(NeuralNetworkArchitecture<T> architecture, IQVFIOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new IQVFIOptions();
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
        if (t < 0.0 || t > 1.0)
            throw new ArgumentOutOfRangeException(nameof(t), t, "Timestep t must be in [0, 1].");
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
            Name = _useNativeMode ? "IQVFI-Native" : "IQVFI-ONNX",
            Description = $"IQ-VFI {_options.Variant} quality-aware degradation-adaptive interpolation (2024)",
            Complexity = _options.NumQualityBlocks * _options.NumPyramidLevels
        };
        m.AdditionalInfo["Variant"] = _options.Variant.ToString();
        m.AdditionalInfo["NumFeatures"] = _options.NumFeatures.ToString();
        m.AdditionalInfo["NumQualityBlocks"] = _options.NumQualityBlocks.ToString();
        m.AdditionalInfo["NumFlowRefinementIters"] = _options.NumFlowRefinementIters.ToString();
        m.AdditionalInfo["NumPyramidLevels"] = _options.NumPyramidLevels.ToString();
        m.AdditionalInfo["QualityThreshold"] = _options.QualityThreshold.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode);
        w.Write(_options.ModelPath ?? string.Empty);
        w.Write((int)_options.Variant);
        w.Write(_options.NumFeatures);
        w.Write(_options.NumQualityBlocks);
        w.Write(_options.NumFlowRefinementIters);
        w.Write(_options.NumPyramidLevels);
        w.Write(_options.QualityThreshold);
        w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean();
        string mp = r.ReadString();
        if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.Variant = (VideoModelVariant)r.ReadInt32();
        _options.NumFeatures = r.ReadInt32();
        _options.NumQualityBlocks = r.ReadInt32();
        _options.NumFlowRefinementIters = r.ReadInt32();
        _options.NumPyramidLevels = r.ReadInt32();
        _options.QualityThreshold = r.ReadDouble();
        _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(IQVFI<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
