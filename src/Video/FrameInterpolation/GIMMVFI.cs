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
/// GIMM-VFI: generalizable implicit motion modeling for video frame interpolation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// GIMM-VFI (NeurIPS 2024) uses implicit neural representations for continuous-time motion:
/// - Implicit motion function: learns a continuous function M(x, y, t) that maps any spatial
///   position (x, y) and any timestep t in [0, 1] to a motion vector, enabling interpolation
///   at arbitrary time intervals without retraining or additional forward passes
/// - Motion encoding network: encodes two input frames into a shared motion latent space
///   using cross-correlation features, producing a motion representation that can be queried
///   at any desired timestep
/// - Generalizable across timesteps: a single forward pass through the motion encoder produces
///   a representation that the implicit function can query at any t, unlike methods that need
///   separate inference per timestep
/// - Adaptive sampling: the implicit function can be queried at higher density in regions with
///   complex motion and lower density in static regions for efficient computation
/// </para>
/// <para>
/// <b>For Beginners:</b> GIMM-VFI learns a smooth "motion field" that describes how everything
/// in the scene moves over time. Once it processes two frames, it can generate a new frame at
/// ANY point between them. This is great for variable slow-motion effects or non-uniform
/// frame rate conversion where you need different time spacings between frames.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputHeight: 128, inputWidth: 128, inputDepth: 3);
/// var model = new GIMMVFI&lt;float&gt;(arch, "gimmvfi.onnx");
/// var midFrame = model.Interpolate(frame0, frame1, t: 0.5);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "GIMM-VFI: Generalizable Implicit Motion Modeling for Video Frame
/// Interpolation" (NeurIPS 2024)
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.FrameInterpolation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Generalizable Implicit Motion Modeling for Video Frame Interpolation",
    "https://arxiv.org/abs/2407.08680",
    Year = 2024,
    Authors = "Zujin Guo, Wei Li, Chen Change Loy")]
public partial class GIMMVFI<T> : FrameInterpolationBase<T>
{
    #region Fields

    private readonly GIMMVFIOptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region Constructors

    /// <summary>Creates a GIMM-VFI model in ONNX inference mode.</summary>
    public GIMMVFI(NeuralNetworkArchitecture<T> architecture, string modelPath, GIMMVFIOptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        _options = options ?? new GIMMVFIOptions();
        _useNativeMode = false;
        SupportsArbitraryTimestep = true;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a GIMM-VFI model in native training mode.</summary>
    public GIMMVFI(NeuralNetworkArchitecture<T> architecture, GIMMVFIOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new GIMMVFIOptions();
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
        ThrowIfDisposed();
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
            Name = _useNativeMode ? "GIMMVFI-Native" : "GIMMVFI-ONNX",
            Description = $"GIMM-VFI {_options.Variant} implicit motion modeling interpolation (NeurIPS 2024)",
            Complexity = _options.NumEncoderBlocks * _options.NumImplicitLayers
        };
        m.AdditionalInfo["Variant"] = _options.Variant.ToString();
        m.AdditionalInfo["NumFeatures"] = _options.NumFeatures.ToString();
        m.AdditionalInfo["NumEncoderBlocks"] = _options.NumEncoderBlocks.ToString();
        m.AdditionalInfo["ImplicitDim"] = _options.ImplicitDim.ToString();
        m.AdditionalInfo["NumImplicitLayers"] = _options.NumImplicitLayers.ToString();
        m.AdditionalInfo["NumFrequencies"] = _options.NumFrequencies.ToString();
        return m;
    }





    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(GIMMVFI<T>));
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
