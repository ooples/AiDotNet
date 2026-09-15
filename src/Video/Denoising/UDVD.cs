using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.Denoising;

/// <summary>
/// UDVD unsupervised deep video denoising with a multi-frame blind-spot network.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "Unsupervised Deep Video Denoising" (Sheth et al., ICCV 2021)</item>
/// </list></para>
/// <para><b>For Beginners:</b> UDVD learns from noisy video alone. It uses nearby frames and a
/// blind spot around each predicted pixel so that independent noise cannot simply be copied to
/// the output.</para>
/// <para>
/// The paper maps five contiguous noisy frames to an estimate of the middle frame. Four rotated,
/// vertically-causal branches exclude the center pixel, then a three-layer 1x1 head combines the
/// directional features. Native <see cref="Train(Tensor{T}, Tensor{T})"/> supports the framework's
/// paired-target training contract; callers can reproduce the paper's self-supervised objective by
/// supplying the noisy middle frame as the target. ONNX mode is inference-only.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a UDVD model for blind self-supervised video denoising
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.Generative,
///     inputHeight: 256, inputWidth: 256, inputDepth: 3);
/// var options = new UDVDOptions();
/// var udvd = new UDVD&lt;double&gt;(architecture, options);
///
/// // Or load a pre-trained ONNX model for streaming inference
/// var udvdOnnx = new UDVD&lt;double&gt;(architecture, "udvd_model.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Unsupervised Deep Video Denoising",
    "https://arxiv.org/abs/2011.15045",
    Year = 2021,
    Authors = "Dev Yashpal Sheth, Sreyas Mohan, Joshua L. Vincent, Ramon Manzorro, Peter A. Crozier, Mitesh M. Khapra, Eero P. Simoncelli, Carlos Fernandez-Granda")]
public partial class UDVD<T> : VideoDenoisingBase<T>
{
    private readonly UDVDOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    /// <summary>
    /// Creates a UDVD model for ONNX inference.
    /// </summary>
    public UDVD(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        UDVDOptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrEmpty(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        _options = options ?? new UDVDOptions();
        _useNativeMode = false;
        IsBlindDenoising = true;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a UDVD model for native training and inference.
    /// </summary>
    public UDVD(
        NeuralNetworkArchitecture<T> architecture,
        UDVDOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new UDVDOptions();
        _useNativeMode = true;
        // The paper/released training recipe uses Adam at 1e-4. Merely storing an optimizer is not
        // sufficient: Train must also pass this instance into TrainWithTape (see below), otherwise
        // the base trainer silently falls back to its generic optimizer and UDVD diverges.
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
            });
        IsBlindDenoising = true;
        InitializeLayers();
    }

    /// <inheritdoc/>
    public override Tensor<T> Denoise(Tensor<T> noisyFrames)
    {
        ThrowIfDisposed();
        var preprocessed = PreprocessFrames(noisyFrames);
        var output = IsOnnxMode ? RunOnnxInference(preprocessed) : Forward(preprocessed);
        return PostprocessOutput(output);
    }

    /// <inheritdoc/>
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultVideoDenoisingLayers(
                inputChannels: ch, inputHeight: h, inputWidth: w,
                numFeatures: _options.NumFeatures,
                temporalFrames: _options.TemporalBufferSize));
        }
    }

    /// <inheritdoc/>
    protected override Tensor<T> PreprocessFrames(Tensor<T> rawFrames) => NormalizeFrames(rawFrames);

    /// <inheritdoc/>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput) => DenormalizeFrames(modelOutput);

    /// <inheritdoc/>
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GetOrCreateBaseOptimizer()
        => _optimizer ?? base.GetOrCreateBaseOptimizer();

    // UpdateParameters restated the base verbatim; ModelBase routes it to SetParameters.
    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ModelName", "UDVD" },
                { "Variant", _options.Variant.ToString() },
                { "NumFeatures", _options.NumFeatures },
                { "NumLevels", _options.NumLevels },
                { "NumResBlocks", _options.NumResBlocks },
                { "TemporalBufferSize", _options.TemporalBufferSize }
            },
            ModelData = SerializeForMetadata()
        };
    }

    /// <inheritdoc/>


    /// <inheritdoc/>


    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(UDVD<T>));
    }

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        if (disposing) OnnxModel?.Dispose();
        base.Dispose(disposing);
    }
}
