using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.Enhancement;

/// <summary>
/// FlashVSR: one-step diffusion streaming video super-resolution at real-time speed.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// FlashVSR (Zhuang et al., 2025) achieves ~17 FPS streaming 4x video super-resolution
/// through a one-step diffusion framework with three key innovations:
/// - Locality-Constrained Sparse Attention (LCSA) for efficient spatial feature extraction
/// - A tiny conditional decoder that produces HR output in a single denoising step
/// - Flow-guided deformable alignment for temporal fusion across frames
///
/// The model is trained via knowledge distillation from a multi-step diffusion teacher,
/// compressing 20+ denoising steps into a single forward pass.
/// </para>
/// <para>
/// <b>For Beginners:</b> FlashVSR makes low-resolution video sharper in real time.
/// Most AI upscalers are too slow for live video because they need many processing steps.
/// FlashVSR solves this by learning to do it in just one step, making it fast enough
/// for streaming at ~17 frames per second while maintaining high quality.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputHeight: 128, inputWidth: 128, inputDepth: 3);
/// var model = new FlashVSR&lt;float&gt;(arch, "flashvsr_base.onnx");
/// var hrFrames = model.Upscale(lrFrames);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "FlashVSR: Efficient Real-Time Video Super-Resolution via One-Step Diffusion"
/// (Zhuang et al., 2025)
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("FlashVSR: Towards Real-Time Diffusion-Based Streaming Video Super-Resolution",
    "https://arxiv.org/abs/2510.12747",
    Year = 2025,
    Authors = "Junhao Zhuang, Shi Guo, Xin Cai, Xiaohui Li, Yihao Liu, Chun Yuan, Tianfan Xue")]
public partial class FlashVSR<T> : VideoSuperResolutionBase<T>
{
    #region Fields

    private readonly FlashVSROptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a FlashVSR model with default architecture (paper-default
    /// 128x128x3 input, scale x4) in native training mode. This overload
    /// exists so generated test scaffolds and discovery callers that only
    /// have access to a parameterless constructor produce a model with a
    /// fully-populated Layers chain (ParameterCount &gt; 0) instead of an
    /// ONNX-only stub. Matches the parameterless-ctor pattern used by
    /// BasicVSRPlusPlus and EDVR.
    /// </summary>
    public FlashVSR()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.ThreeDimensional,
            taskType: Enums.NeuralNetworkTaskType.Regression,
            inputHeight: 128, inputWidth: 128, inputDepth: 3,
            outputSize: 2))
    {
    }

    /// <summary>Creates a FlashVSR model in ONNX inference mode.</summary>
    public FlashVSR(NeuralNetworkArchitecture<T> architecture, string modelPath, FlashVSROptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options = options ?? new FlashVSROptions();
        _useNativeMode = false;
        ScaleFactor = _options.ScaleFactor;
        NumFrames = _options.NumInputFrames;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a FlashVSR model in native training mode.</summary>
    public FlashVSR(NeuralNetworkArchitecture<T> architecture, FlashVSROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new FlashVSROptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        ScaleFactor = _options.ScaleFactor;
        NumFrames = _options.NumInputFrames;
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
            int h = Architecture.InputHeight > 0 ? Architecture.InputHeight : 128;
            int w = Architecture.InputWidth > 0 ? Architecture.InputWidth : 128;
            Layers.AddRange(LayerHelper<T>.CreateDefaultVideoSuperResolutionLayers(
                inputChannels: ch, inputHeight: h, inputWidth: w,
                numFeatures: _options.NumFeatures,
                numResBlocks: _options.NumLCSABlocks,
                scaleFactor: _options.ScaleFactor));
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
    protected override Tensor<T> PreprocessFrames(Tensor<T> rawFrames) => NormalizeFrames(rawFrames);

    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput) => DenormalizeFrames(modelOutput);

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "FlashVSR-Native" : "FlashVSR-ONNX",
            Description = $"FlashVSR {_options.Variant} one-step diffusion VSR (Zhuang et al., 2025)",
            Complexity = _options.NumLCSABlocks
        };
        m.AdditionalInfo["Variant"] = _options.Variant.ToString();
        m.AdditionalInfo["NumFeatures"] = _options.NumFeatures.ToString();
        m.AdditionalInfo["ScaleFactor"] = _options.ScaleFactor.ToString();
        m.AdditionalInfo["NumLCSABlocks"] = _options.NumLCSABlocks.ToString();
        return m;
    }





    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(FlashVSR<T>));
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
