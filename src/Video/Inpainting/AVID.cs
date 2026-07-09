using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.Inpainting;

/// <summary>
/// AVID diffusion-based video inpainting supporting arbitrary-length videos.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "AVID: Any-Length Video Inpainting with Diffusion Model" (Zhang et al., CVPR 2024)</item>
/// </list></para>
/// <para><b>For Beginners:</b> AVID (Adaptive Video Inpainting via Diffusion) fills in missing or damaged regions of video using diffusion models. It adaptively propagates content from neighboring frames and regions.</para>
/// <para>
/// AVID uses a diffusion U-Net with temporal attention to iteratively denoise masked video regions,
/// processing long videos through an autoregressive temporal pipeline with overlapping windows
/// that maintains temporal consistency across the full sequence.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create an AVID model for diffusion-based video inpainting
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     inputHeight: 256, inputWidth: 256, inputDepth: 3);
/// var options = new AVIDOptions();
/// var avid = new AVID&lt;double&gt;(architecture, options);
///
/// // Or load a pre-trained ONNX model for inference
/// var avidOnnx = new AVID&lt;double&gt;(architecture, "avid_model.onnx");
/// </code>
/// </example>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("AVID: Any-Length Video Inpainting with Diffusion Model",
    "https://arxiv.org/abs/2312.03816",
    Year = 2024,
    Authors = "Zhixing Zhang, Bichen Wu, Xiaoyan Wang, Yaqiao Luo, Zijian He, Peter Vajda, Dimitris Metaxas, Licheng Yu")]
public class AVID<T> : VideoInpaintingBase<T>
{
    private readonly AVIDOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    /// <summary>
    /// Creates an AVID model for ONNX inference.
    /// </summary>
    public AVID(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        AVIDOptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrEmpty(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        _options = options ?? new AVIDOptions();
        _useNativeMode = false;
        SupportsTemporalPropagation = true;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>
    /// Creates an AVID model for native training and inference.
    /// </summary>
    public AVID(
        NeuralNetworkArchitecture<T> architecture,
        AVIDOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new AVIDOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        SupportsTemporalPropagation = true;
        InitializeLayers();
    }

    /// <inheritdoc/>
    public override Tensor<T> Inpaint(Tensor<T> frames, Tensor<T> masks)
    {
        ThrowIfDisposed();
        var preprocessed = PreprocessFrames(frames);
        var combined = ConcatFramesAndMasks(preprocessed, masks);
        var output = IsOnnxMode ? RunOnnxInference(combined) : Forward(combined);
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultVideoInpaintingLayers(
                inputChannels: ch, inputHeight: h, inputWidth: w,
                numFeatures: _options.NumFeatures));
        }
    }

    /// <inheritdoc/>
    protected override Tensor<T> PreprocessFrames(Tensor<T> rawFrames) => NormalizeInpaintFrames(rawFrames);

    /// <inheritdoc/>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput) => DenormalizeInpaintFrames(modelOutput);

    private bool _shapesProbed;

    /// <inheritdoc/>
    protected override void ResolveLazyLayerShapes()
    {
        // AVID's inference path (PredictCore -> Inpaint) concatenates a 1-channel mask before the
        // encoder, so the lazy first conv must resolve to InputDepth+1 — not the InputDepth the base
        // linear walk infers from the architecture input shape. Probe the real inference forward once
        // on a tiny dummy frame so callers that run before any real forward (GetParameters,
        // serialization, Clone) resolve the encoder to the same depth training and inference feed it.
        if (_shapesProbed || Layers.Count == 0) return;
        _shapesProbed = true;
        int c = Architecture.InputDepth > 0 ? Architecture.InputDepth : 3;
        _ = PredictCore(new Tensor<T>([1, c, 32, 32]));
    }

    /// <inheritdoc/>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        // Training must apply the SAME transform inference does (Inpaint): normalize the frames,
        // concatenate a 1-channel mask (InputDepth -> InputDepth+1 so the encoder conv matches),
        // run the layer stack, then denormalize. Feeding the raw InputDepth frames straight through
        // the base would resolve/expect a different first-conv depth than inference AND train in a
        // different value space, so the two paths would diverge. Delegate the actual layer walk
        // (autodiff tape, gradient checkpointing, seed-wiring) to the base by handing it the
        // mask-concatenated tensor; normalize/denormalize are Engine ops so gradients still flow.
        // Use a fresh RANDOM per-step hole mask (PyTorch video-inpainting recipe). A mask that varies
        // every step exercises the encoder's mask-channel weights without becoming a constant the model
        // can exploit as a shortcut — so training keeps using the frame content and stays input-sensitive.
        // Inference's PredictCore uses the deterministic CreateDefaultInpaintingMask.
        var mask = CreateTrainingMask(input.Shape[0], input.Shape[2], input.Shape[3]);
        var combined = ConcatFramesAndMasks(PreprocessFrames(input), mask);
        return PostprocessOutput(base.ForwardForTraining(combined));
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        SetTrainingMode(true);
        try
        {
        TrainWithTape(input, expected);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode) throw new NotSupportedException("Parameter updates are not supported in ONNX mode.");
        int required = 0;
        foreach (var layer in Layers) required += layer.GetParameters().Length;
        if (parameters.Length < required)
            throw new ArgumentException($"Parameter vector length {parameters.Length} is less than required {required}.", nameof(parameters));
        int offset = 0;
        foreach (var layer in Layers)
        {
            var p = layer.GetParameters();
            var sub = new Vector<T>(p.Length);
            for (int i = 0; i < p.Length; i++) sub[i] = parameters[offset + i];
            layer.SetParameters(sub);
            offset += p.Length;
        }
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ModelName", "AVID" },
                { "Variant", _options.Variant.ToString() },
                { "NumFeatures", _options.NumFeatures },
                { "NumDiffusionSteps", _options.NumDiffusionSteps },
                { "NumResBlocks", _options.NumResBlocks },
                { "NumHeads", _options.NumHeads }
            },
            ModelData = SerializeForMetadata()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write((int)_options.Variant);
        writer.Write(_options.NumFeatures);
        writer.Write(_options.NumDiffusionSteps);
        writer.Write(_options.NumResBlocks);
        writer.Write(_options.NumHeads);
        writer.Write(_options.TemporalOverlap);
        writer.Write(_options.LearningRate);
        writer.Write(_options.DropoutRate);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _options.Variant = (VideoModelVariant)reader.ReadInt32();
        _options.NumFeatures = reader.ReadInt32();
        _options.NumDiffusionSteps = reader.ReadInt32();
        _options.NumResBlocks = reader.ReadInt32();
        _options.NumHeads = reader.ReadInt32();
        _options.TemporalOverlap = reader.ReadInt32();
        _options.LearningRate = reader.ReadDouble();
        _options.DropoutRate = reader.ReadDouble();
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            return new AVID<T>(Architecture, p, _options);
        return new AVID<T>(Architecture, _options);
    }

    private static Tensor<T> ConcatFramesAndMasks(Tensor<T> frames, Tensor<T> masks)
    {
        if (frames.Rank != 4)
            throw new ArgumentException($"Frames must be rank 4 [N, C, H, W], got rank {frames.Rank}.", nameof(frames));
        if (masks.Rank != 4)
            throw new ArgumentException($"Masks must be rank 4 [N, 1, H, W], got rank {masks.Rank}.", nameof(masks));
        int n = frames.Shape[0];
        int c = frames.Shape[1];
        int h = frames.Shape[2];
        int w = frames.Shape[3];
        if (masks.Shape[0] != n || masks.Shape[2] != h || masks.Shape[3] != w)
            throw new ArgumentException($"Masks spatial dimensions must match frames. Frames: [{n},{c},{h},{w}], Masks: [{masks.Shape[0]},{masks.Shape[1]},{masks.Shape[2]},{masks.Shape[3]}].", nameof(masks));
        var combined = new Tensor<T>([n, c + 1, h, w]);
        int frameSize = c * h * w;
        int maskSize = h * w;
        int combinedSize = (c + 1) * h * w;
        for (int f = 0; f < n; f++)
        {
            // Copy frame channels
            for (int i = 0; i < frameSize; i++)
                combined.Data.Span[f * combinedSize + i] = frames.Data.Span[f * frameSize + i];
            // Copy mask channel
            for (int i = 0; i < maskSize; i++)
                combined.Data.Span[f * combinedSize + frameSize + i] = masks.Data.Span[f * maskSize + i];
        }
        return combined;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(AVID<T>));
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
