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
/// BSVD bidirectional streaming video denoising with real-time buffers.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "BSVD: Bidirectional Streaming Video Denoising" (Qi et al., ACM MM 2022)</item>
/// </list></para>
/// <para><b>For Beginners:</b> BSVD (Blind Spot Video Denoising) removes noise from video without needing clean reference frames. It uses a blind-spot network that learns denoising patterns directly from noisy video data.</para>
/// <para>
/// BSVD enables real-time video denoising through bidirectional streaming with efficient
/// buffer management. It processes video in forward and backward passes, maintaining compact
/// latent buffers for constant-memory operation regardless of video length.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a BSVD model for real-time bidirectional video denoising
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     inputHeight: 256, inputWidth: 256, inputDepth: 3);
/// var options = new BSVDOptions();
/// var bsvd = new BSVD&lt;double&gt;(architecture, options);
///
/// // Or load a pre-trained ONNX model for inference
/// var bsvdOnnx = new BSVD&lt;double&gt;(architecture, "bsvd_model.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Real-time Streaming Video Denoising with Bidirectional Buffers",
    "https://arxiv.org/abs/2207.06937",
    Year = 2022,
    Authors = "Chenyang Qi, Junming Chen, Xin Yang, Qifeng Chen")]
public class BSVD<T> : VideoDenoisingBase<T>
{
    private readonly BSVDOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    /// <summary>
    /// Creates a BSVD model for ONNX inference.
    /// </summary>
    public BSVD(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        BSVDOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new BSVDOptions();
        _useNativeMode = false;
        IsBlindDenoising = true;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a BSVD model for native training and inference.
    /// </summary>
    public BSVD(
        NeuralNetworkArchitecture<T> architecture,
        BSVDOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new BSVDOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
                // Optimizer settings taken from the reference implementation's training config
                // (ChenyangQiQi/BSVD, options/train/bsvd_c64_unblind.yml), not guessed:
                //   optim_g: Adam, lr 1e-3, betas [0.9, 0.99], weight_decay 0
                //   use_grad_clip: 5
                // beta2 = 0.99 rather than the 0.999 default adapts the second moment about ten
                // times faster, which materially shortens Adam's early-step overshoot; combined
                // with the grad-norm bound it is what keeps the first iterations well-behaved.
                Beta1 = _options.AdamBeta1,
                Beta2 = _options.AdamBeta2,
                Epsilon = 1e-8,
                UseAdaptiveLearningRate = false,
                UseAdaptiveBetas = false,
                UseAMSGrad = false,
                EnableGradientClipping = true,
                MaxGradientNorm = _options.GradientClipNorm
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
            // Honour the architecture's temporal frame count when the caller
            // built it via InputType.FourDimensional. The default helper
            // multiplies inputChannels × temporalFrames (frames stacked as
            // additional channels in the first conv); without this the
            // generated test feeds a [frames, channels, h, w] input whose
            // depth dimension is just `channels`, but the conv was built for
            // `channels * 5` (the helper's default temporalFrames=5).
            int temporalFrames = Architecture.InputFrames > 0 ? Architecture.InputFrames : 5;
            Layers.AddRange(LayerHelper<T>.CreateDefaultBSVDLayers(
                inputChannels: ch, inputHeight: h, inputWidth: w,
                numFeatures: _options.NumFeatures, temporalFrames: temporalFrames,
                numUNetStages: _options.NumUNetStages,
                shiftedChannelRatio: _options.ShiftedChannelRatio));
        }
    }

    /// <summary>
    /// Route the generic inspection path (used by
    /// <see cref="AiDotNet.NeuralNetworks.NeuralNetworkBase{T}.GetNamedLayerActivations"/>
    /// and test harnesses) through the same preprocessing that
    /// <see cref="Denoise"/> applies. Without this override the base walks
    /// each <see cref="Layers"/> entry directly and the first conv — built
    /// for `inputChannels * temporalFrames` folded channels — rejects a raw
    /// [frames, channels, h, w] input.
    /// </summary>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        var preprocessed = PreprocessFrames(input);
        return base.GetNamedLayerActivations(preprocessed);
    }

    /// <summary>
    /// Same rationale as <see cref="GetNamedLayerActivations"/>: the
    /// tape-based <see cref="AiDotNet.NeuralNetworks.NeuralNetworkBase{T}.TrainWithTape"/>
    /// path runs <c>ForwardForTraining</c> on the raw input. Without this
    /// override the first conv sees [frames, channels, h, w] directly and
    /// rejects the unfolded channel depth.
    /// </summary>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        var preprocessed = PreprocessFrames(input);
        var modelOutput = ForwardPreprocessedForTraining(preprocessed);

        // Keep the public output contract during tape-based training as well
        // as inference. Engine.Interpolate participates in the active
        // computation tape, unlike a manually populated output tensor, so
        // gradients still flow through the decoder when its native output is
        // a few pixels short of the requested frame size.
        return RestoreSpatialResolution(modelOutput);
    }

    /// <inheritdoc/>
    protected override Tensor<T> PreprocessFrames(Tensor<T> rawFrames)
    {
        // CreateDefaultVideoDenoisingLayers builds the first conv with
        // (inputChannels * temporalFrames) input channels — i.e. the temporal
        // axis is folded into the channel axis before the conv. A raw video
        // input of shape [frames, channels, h, w] therefore needs reshaping
        // to [1, frames * channels, h, w] before normalisation. Anything
        // already in the [1, C, h, w] layout (or higher rank with batch=1)
        // passes straight through.
        var shape = rawFrames.Shape;
        if (shape.Length == 4
            && Architecture.InputType == AiDotNet.Enums.InputType.FourDimensional
            && shape[0] == Architecture.InputFrames
            && shape[1] == Architecture.InputDepth
            && shape[2] == Architecture.InputHeight
            && shape[3] == Architecture.InputWidth)
        {
            int folded = shape[0] * shape[1];
            rawFrames = rawFrames.Reshape(new[] { 1, folded, shape[2], shape[3] });
        }
        return NormalizeFrames(rawFrames);
    }

    /// <inheritdoc/>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput)
    {
        var denormalized = DenormalizeFrames(modelOutput);
        return RestoreSpatialResolution(denormalized);
    }

    /// <inheritdoc/>
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GetOrCreateBaseOptimizer()
        => _optimizer ?? base.GetOrCreateBaseOptimizer();

    private Tensor<T> RestoreSpatialResolution(Tensor<T> output)
    {
        int targetHeight = Architecture.InputHeight;
        int targetWidth = Architecture.InputWidth;

        // The shared native video-denoising decoder can be a few pixels short
        // after its stride-2 convolution/deconvolution path (for example,
        // 32x32 becomes 29x29). BSVD's public contract returns a denoised
        // frame at the source resolution, so restore that resolution here.
        if (output.Shape.Length != 4
            || targetHeight <= 0
            || targetWidth <= 0
            || (output.Shape[2] == targetHeight && output.Shape[3] == targetWidth))
        {
            return output;
        }

        return Engine.Interpolate(
            output,
            [targetHeight, targetWidth],
            InterpolateMode.Bilinear,
            alignCorners: false);
    }

    // UpdateParameters re-sliced the flat vector across Layers by hand -- the base walks
    // exactly the same enumeration, so this said nothing the base does not already say.
    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ModelName", "BSVD" },
                { "Variant", _options.Variant.ToString() },
                { "NumFeatures", _options.NumFeatures },
                { "NumRecurrentBlocks", _options.NumRecurrentBlocks },
                { "BufferDim", _options.BufferDim },
                { "NumLevels", _options.NumLevels }
            },
            ModelData = SerializeForMetadata()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write((int)_options.Variant);
        writer.Write(_options.NumFeatures);
        writer.Write(_options.NumRecurrentBlocks);
        writer.Write(_options.BufferDim);
        writer.Write(_options.NumLevels);
        writer.Write(_options.LearningRate);
        writer.Write(_options.DropoutRate);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _options.Variant = (VideoModelVariant)reader.ReadInt32();
        _options.NumFeatures = reader.ReadInt32();
        _options.NumRecurrentBlocks = reader.ReadInt32();
        _options.BufferDim = reader.ReadInt32();
        _options.NumLevels = reader.ReadInt32();
        _options.LearningRate = reader.ReadDouble();
        _options.DropoutRate = reader.ReadDouble();
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new BSVD<T>(Architecture, _options);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(BSVD<T>));
    }

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        if (disposing)
        {
            OnnxModel?.Dispose();
        }
        base.Dispose(disposing);
    }
}
