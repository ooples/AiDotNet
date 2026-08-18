using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.Enhancement;

/// <summary>
/// RealViformer: investigating attention for real-world video super-resolution.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// RealViformer (Zhang and Yao, ECCV 2024) investigates attention for real-world VSR:
/// - Channel attention dominance: empirically shows that SE-style channel attention
///   outperforms spatial attention for real-world degradations with complex noise
/// - Bidirectional recurrent propagation: temporal feature propagation with channel
///   attention at each recurrent step for adaptive per-frame fusion weights
/// - ProbSparse attention: Informer-style sparse self-attention that selects only the
///   top-k most informative queries based on KL-divergence scoring, enabling efficient
///   processing of longer video sequences
/// - Real-world degradation training: second-order degradation model (blur, downsample,
///   noise, JPEG applied twice) for handling practical video artifacts
/// </para>
/// <para>
/// <b>For Beginners:</b> RealViformer is designed for real phone recordings and compressed
/// video streams, not just lab-quality test clips. It discovered that "which color channels
/// matter" (channel attention) works better than "which spatial locations matter" for the
/// messy, complex degradations in real footage. It also borrows an efficiency trick from
/// time-series forecasting to handle longer videos without running out of memory.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputHeight: 64, inputWidth: 64, inputDepth: 3);
/// var model = new RealViformer&lt;float&gt;(arch, "realviformer.onnx");
/// var hrFrames = model.Upscale(lrFrames);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "RealViformer: Investigating Attention for Real-World Video
/// Super-Resolution" (Zhang and Yao, ECCV 2024)
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("RealViformer: Investigating Attention for Real-World Video Super-Resolution",
    "https://arxiv.org/abs/2407.13987",
    Year = 2024,
    Authors = "Yuehan Zhang, Angela Yao")]
public partial class RealViformer<T> : VideoSuperResolutionBase<T>
{
    #region Fields

    private readonly RealViformerOptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region Constructors

    /// <summary>Creates a RealViformer model in ONNX inference mode.</summary>
    public RealViformer(NeuralNetworkArchitecture<T> architecture, string modelPath, RealViformerOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new RealViformerOptions();
        _useNativeMode = false;
        ScaleFactor = _options.ScaleFactor;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a RealViformer model in native training mode.</summary>
    public RealViformer(NeuralNetworkArchitecture<T> architecture, RealViformerOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new RealViformerOptions();
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
            Name = _useNativeMode ? "RealViformer-Native" : "RealViformer-ONNX",
            Description = $"RealViformer {_options.Variant} channel attention real-world VSR (Zhang and Yao, ECCV 2024)",
            Complexity = _options.NumResBlocks
        };
        m.AdditionalInfo["Variant"] = _options.Variant.ToString();
        m.AdditionalInfo["NumFeatures"] = _options.NumFeatures.ToString();
        m.AdditionalInfo["NumResBlocks"] = _options.NumResBlocks.ToString();
        m.AdditionalInfo["ChannelReductionRatio"] = _options.ChannelReductionRatio.ToString();
        m.AdditionalInfo["SparseTopKFactor"] = _options.SparseTopKFactor.ToString();
        m.AdditionalInfo["ScaleFactor"] = _options.ScaleFactor.ToString();
        return m;
    }





    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(RealViformer<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
