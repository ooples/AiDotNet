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
/// BiMVFI: bidirectional motion field-based video frame interpolation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// BiMVFI (Seo et al., CVPR 2025) handles non-uniform motion with bidirectional fields:
/// - Bidirectional motion fields: estimates forward (0 to t) and backward (1 to t) motion
///   fields independently, each with its own confidence map, instead of a single symmetric flow
/// - Adaptive blending: per-pixel confidence weights learned from both motion fields determine
///   how to blend warped frames, gracefully handling occlusion regions where only one direction
///   provides valid information
/// - Non-uniform motion modeling: dedicated occlusion reasoning module detects regions with
///   non-uniform motion (e.g., independently moving objects) and applies motion-compensated
///   attention to those areas specifically
/// - Multi-scale architecture: 3-level feature pyramid with cross-scale feature propagation
///   for handling both small sub-pixel motions and large inter-frame displacements
/// </para>
/// <para>
/// <b>For Beginners:</b> When objects move at different speeds or occlude each other, a single
/// motion estimate fails. BiMVFI solves this by estimating motion from both directions (past
/// and future) and letting each pixel choose which direction gives a better result. Where an
/// object appears in one direction but not the other, it trusts the visible direction.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputHeight: 128, inputWidth: 128, inputDepth: 3);
/// var model = new BiMVFI&lt;float&gt;(arch, "bimvfi.onnx");
/// var midFrame = model.Interpolate(frame0, frame1, t: 0.5);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "BiMVFI: Bidirectional Motion Field-Based Video Frame Interpolation"
/// (Seo et al., CVPR 2025)
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.FrameInterpolation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("BiM-VFI: Bidirectional Motion Field-Guided Frame Interpolation for Video with Non-uniform Motions",
    "https://arxiv.org/abs/2412.11365",
    Year = 2024,
    Authors = "Wonyong Seo, Jihyong Oh, Munchurl Kim")]
public partial class BiMVFI<T> : FrameInterpolationBase<T>
{
    #region Fields

    private readonly BiMVFIOptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region Constructors

    /// <summary>Creates a BiMVFI model in ONNX inference mode.</summary>
    public BiMVFI(NeuralNetworkArchitecture<T> architecture, string modelPath, BiMVFIOptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        _options = options ?? new BiMVFIOptions();
        _useNativeMode = false;
        SupportsArbitraryTimestep = true;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a BiMVFI model in native training mode.</summary>
    public BiMVFI(NeuralNetworkArchitecture<T> architecture, BiMVFIOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new BiMVFIOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        SupportsArbitraryTimestep = true;
        InitializeLayers();
    }

    #endregion

    #region Frame Interpolation

    /// <inheritdoc />
    /// <remarks>
    /// In ONNX mode, the timestep <paramref name="t"/> is passed to the model which natively
    /// supports arbitrary timestep interpolation. In native mode, the baseline encoder-decoder
    /// does not yet incorporate <paramref name="t"/> and always produces mid-frame output.
    /// </remarks>
    public override Tensor<T> Interpolate(Tensor<T> frame0, Tensor<T> frame1, double t = 0.5)
    {
        ThrowIfDisposed();
        if (t < 0.0 || t > 1.0)
            throw new ArgumentOutOfRangeException(nameof(t), t, "Timestep must be in [0, 1].");
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
            Name = _useNativeMode ? "BiMVFI-Native" : "BiMVFI-ONNX",
            Description = $"BiMVFI {_options.Variant} bidirectional motion field interpolation (Seo et al., CVPR 2025)",
            Complexity = _options.NumResBlocks * _options.NumScales
        };
        m.AdditionalInfo["Variant"] = _options.Variant.ToString();
        m.AdditionalInfo["NumFeatures"] = _options.NumFeatures.ToString();
        m.AdditionalInfo["NumResBlocks"] = _options.NumResBlocks.ToString();
        m.AdditionalInfo["NumScales"] = _options.NumScales.ToString();
        m.AdditionalInfo["OcclusionAwareBlending"] = _options.OcclusionAwareBlending.ToString();
        m.AdditionalInfo["ConfidenceThreshold"] = _options.ConfidenceThreshold.ToString();
        return m;
    }





    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(BiMVFI<T>));
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
