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
/// DOVE: harnessing large-scale video diffusion priors for general video restoration.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DOVE (Chen et al., 2025) uses pretrained video diffusion models as powerful restoration priors:
/// - Degradation estimation: analyzes input video to determine the type and severity of degradation
/// - Guided generation: conditions a pretrained Stable Video Diffusion backbone on the
///   degradation estimate, generating clean video through controlled denoising
/// - Video diffusion prior: the SVD backbone's knowledge of natural video dynamics provides
///   inherent temporal consistency without explicit flow or alignment modules
///
/// Unlike discriminative approaches (BasicVSR, EDVR), DOVE is generative: it synthesizes
/// plausible clean video conditioned on the degraded input.
/// </para>
/// <para>
/// <b>For Beginners:</b> DOVE repurposes a video generation AI to restore degraded video.
/// Think of it like asking an artist who knows what clean video looks like to paint a
/// restored version of your noisy/blurry footage, rather than trying to mathematically
/// reverse the damage.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputHeight: 128, inputWidth: 128, inputDepth: 3);
/// var model = new DOVE&lt;float&gt;(arch, "dove_base.onnx");
/// var restored = model.Upscale(degradedFrames);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "DOVE: Harnessing Large-Scale Video Diffusion Priors for General Video
/// Restoration" (Chen et al., 2025)
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("DOVE: Efficient One-Step Diffusion Model for Real-World Video Super-Resolution",
    "https://arxiv.org/abs/2505.16239",
    Year = 2025,
    Authors = "Zheng Chen, Zichen Zou, Kewei Zhang, Xiongfei Su, Xin Yuan, Yong Guo, Yulun Zhang")]
public partial class DOVE<T> : VideoSuperResolutionBase<T>
{
    #region Fields

    private readonly DOVEOptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region Constructors

    /// <summary>Creates a DOVE model in ONNX inference mode.</summary>
    public DOVE(NeuralNetworkArchitecture<T> architecture, string modelPath, DOVEOptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options = options ?? new DOVEOptions();
        _useNativeMode = false;
        ScaleFactor = _options.ScaleFactor;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a DOVE model in native training mode.</summary>
    public DOVE(NeuralNetworkArchitecture<T> architecture, DOVEOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new DOVEOptions();
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
            Name = _useNativeMode ? "DOVE-Native" : "DOVE-ONNX",
            Description = $"DOVE {_options.Variant} video diffusion prior restoration (Chen et al., 2025)",
            Complexity = _options.NumResBlocks
        };
        m.AdditionalInfo["Variant"] = _options.Variant.ToString();
        m.AdditionalInfo["NumFeatures"] = _options.NumFeatures.ToString();
        m.AdditionalInfo["NumDenoisingSteps"] = _options.NumDenoisingSteps.ToString();
        m.AdditionalInfo["GuidanceScale"] = _options.GuidanceScale.ToString();
        m.AdditionalInfo["ScaleFactor"] = _options.ScaleFactor.ToString();
        return m;
    }





    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(DOVE<T>));
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
