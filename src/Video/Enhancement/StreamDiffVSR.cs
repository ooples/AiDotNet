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
/// Stream-DiffVSR: causally-conditioned diffusion for online video super-resolution.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Stream-DiffVSR (Li et al., 2025) achieves low-latency online video SR through:
/// - Auto-regressive temporal guidance using previously generated HR frames
/// - A 4-step distilled denoiser (compressed from ~50 diffusion steps)
/// - Causal temporal conditioning (past frames only, no future lookahead)
///
/// This enables streaming 4x video super-resolution with temporal consistency.
/// </para>
/// <para>
/// <b>For Beginners:</b> Most video upscalers need to see future frames, which adds delay.
/// Stream-DiffVSR only uses past frames, making it suitable for live streaming or
/// video calls where latency matters.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputHeight: 128, inputWidth: 128, inputDepth: 3);
/// var model = new StreamDiffVSR&lt;float&gt;(arch, "streamdiffvsr.onnx");
/// var hrFrames = model.Upscale(lrFrames);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "Stream-DiffVSR: Low-Latency Streamable Diffusion-based Video Super-Resolution"
/// (Li et al., 2025)
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
// Citation corrected: the title, arXiv id and author list were all wrong. arXiv 2501.14727 is
// "Estimation-theoretic analysis of lensless imaging" (Kabuli, Singh & Waller) — an unrelated
// computational-imaging paper — so anyone following this reference found nothing about video
// super-resolution, and the recorded authors belonged to a different work entirely.
[ResearchPaper("Stream-DiffVSR: Low-Latency Streamable Video Super-Resolution via Auto-Regressive Diffusion",
    "https://arxiv.org/abs/2512.23709",
    Year = 2025,
    Authors = "Hau-Shiang Shiu, Chin-Yang Lin, Zhixiang Wang, Chi-Wei Hsiao, Po-Fan Yu, Yu-Chih Chen, Yu-Lun Liu")]
public class StreamDiffVSR<T> : VideoSuperResolutionBase<T>
{
    #region Fields

    private readonly StreamDiffVSROptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region Constructors

    /// <summary>Creates a StreamDiffVSR model in ONNX inference mode.</summary>
    public StreamDiffVSR(NeuralNetworkArchitecture<T> architecture, string modelPath, StreamDiffVSROptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrEmpty(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        _options = options ?? new StreamDiffVSROptions();
        _useNativeMode = false;
        ScaleFactor = _options.ScaleFactor;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a StreamDiffVSR model in native training mode.</summary>
    public StreamDiffVSR(NeuralNetworkArchitecture<T> architecture, StreamDiffVSROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new StreamDiffVSROptions();
        _useNativeMode = true;
        // Appendix C: "AdamW optimizer (beta1 = 0.9, beta2 = 0.999, weight decay 0.01)" at a CONSTANT
        // learning rate of 5e-5, for every training stage. These now come from StreamDiffVSROptions,
        // which carries the paper's values as its defaults.
        //
        // Two defects were fixed here. The optimizer was constructed as AdamWOptimizer(this) with
        // DEFAULT options, discarding the paper's learning rate entirely and running 20x higher at
        // AdamW's own 1e-3. Worse, the field was never published to the tape trainer at all — no
        // SetBaseTrainOptimizer call and no GetOrCreateBaseOptimizer override — so training silently
        // used the base class's lazily-created Adam and this field was dead. The measured symptom was
        // the memorization probe rising monotonically (0.256 -> 0.471) instead of descending.
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AiDotNet.Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
                Beta1 = _options.AdamBeta1,
                Beta2 = _options.AdamBeta2,
                WeightDecay = _options.WeightDecay,
                EnableGradientClipping = true,
                MaxGradientNorm = 1.0
            });
        SetBaseTrainOptimizer(_optimizer);
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
            int h = Architecture.InputHeight > 0 ? Architecture.InputHeight : 128;
            int w = Architecture.InputWidth > 0 ? Architecture.InputWidth : 128;
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
            Name = _useNativeMode ? "StreamDiffVSR-Native" : "StreamDiffVSR-ONNX",
            Description = $"Stream-DiffVSR {_options.Variant} causal diffusion VSR (Li et al., 2025)",
            Complexity = _options.NumResBlocks
        };
        m.AdditionalInfo["Variant"] = _options.Variant.ToString();
        m.AdditionalInfo["NumFeatures"] = _options.NumFeatures.ToString();
        m.AdditionalInfo["NumDenoisingSteps"] = _options.NumDenoisingSteps.ToString();
        m.AdditionalInfo["ScaleFactor"] = _options.ScaleFactor.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode);
        w.Write(_options.ModelPath ?? string.Empty);
        w.Write((int)_options.Variant);
        w.Write(_options.NumFeatures);
        w.Write(_options.NumDenoisingSteps);
        w.Write(_options.NumResBlocks);
        w.Write(_options.TemporalRadius);
        w.Write(_options.ScaleFactor);
        w.Write(_options.LatentDim);
        w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean();
        string mp = r.ReadString();
        if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.Variant = (VideoModelVariant)r.ReadInt32();
        _options.NumFeatures = r.ReadInt32();
        _options.NumDenoisingSteps = r.ReadInt32();
        _options.NumResBlocks = r.ReadInt32();
        _options.TemporalRadius = r.ReadInt32();
        _options.ScaleFactor = r.ReadInt32();
        _options.LatentDim = r.ReadInt32();
        _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(StreamDiffVSR<T>));
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
