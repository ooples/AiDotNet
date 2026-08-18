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
/// RealBasicVSR-Sharp: perceptually-optimized real-world video super-resolution.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// RealBasicVSR-Sharp (Chan et al., CVPR 2022) is the perceptual variant of RealBasicVSR:
/// - Same pre-cleaning module + BasicVSR backbone as RealBasicVSR
/// - Trained with perceptual loss (VGG feature matching) for better texture recovery
/// - Adds GAN discriminator loss for sharper, more photo-realistic outputs
/// - Produces visually sharper results at the cost of slightly lower PSNR
///
/// This variant is preferred when visual quality matters more than pixel accuracy,
/// such as for display on screens or social media.
/// </para>
/// <para>
/// <b>For Beginners:</b> There are two ways to measure video quality: mathematical accuracy
/// (PSNR) and visual quality (how it looks to your eyes). The base RealBasicVSR maximizes
/// PSNR, while this "Sharp" variant maximizes visual quality. The Sharp version produces
/// images with crisper textures and more natural-looking details.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputHeight: 64, inputWidth: 64, inputDepth: 3);
/// var model = new RealBasicVSRSharp&lt;float&gt;(arch, "realbasicvsr_sharp.onnx");
/// var hrFrames = model.Upscale(lrFrames);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "Investigating Tradeoffs in Real-World Video Super-Resolution"
/// (Chan et al., CVPR 2022)
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Investigating Tradeoffs in Real-World Video Super-Resolution",
    "https://arxiv.org/abs/2111.12704",
    Year = 2022,
    Authors = "Kelvin C.K. Chan, Shangchen Zhou, Xiangyu Xu, Chen Change Loy")]
public partial class RealBasicVSRSharp<T> : VideoSuperResolutionBase<T>
{
    #region Fields

    private readonly RealBasicVSRSharpOptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region Constructors

    /// <summary>Creates a RealBasicVSR-Sharp model in ONNX inference mode.</summary>
    public RealBasicVSRSharp(NeuralNetworkArchitecture<T> architecture, string modelPath, RealBasicVSRSharpOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new RealBasicVSRSharpOptions();
        _useNativeMode = false;
        ScaleFactor = _options.ScaleFactor;
        NumFrames = _options.NumFrames;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a RealBasicVSR-Sharp model in native training mode.</summary>
    public RealBasicVSRSharp(NeuralNetworkArchitecture<T> architecture, RealBasicVSRSharpOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new RealBasicVSRSharpOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        ScaleFactor = _options.ScaleFactor;
        NumFrames = _options.NumFrames;
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
            Name = _useNativeMode ? "RealBasicVSRSharp-Native" : "RealBasicVSRSharp-ONNX",
            Description = "RealBasicVSR-Sharp perceptual VSR (Chan et al., CVPR 2022)",
            Complexity = _options.NumResBlocks + _options.CleaningModuleBlocks
        };
        m.AdditionalInfo["NumFeatures"] = _options.NumFeatures.ToString();
        m.AdditionalInfo["NumResBlocks"] = _options.NumResBlocks.ToString();
        m.AdditionalInfo["CleaningModuleBlocks"] = _options.CleaningModuleBlocks.ToString();
        m.AdditionalInfo["PerceptualWeight"] = _options.PerceptualWeight.ToString();
        m.AdditionalInfo["GANWeight"] = _options.GANWeight.ToString();
        m.AdditionalInfo["ScaleFactor"] = _options.ScaleFactor.ToString();
        return m;
    }





    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(RealBasicVSRSharp<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
