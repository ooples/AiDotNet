using AiDotNet.Attributes;
using AiDotNet.Extensions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.VisionLanguage.VideoLanguage;

/// <summary>
/// VideoLLaMA 2: spatial-temporal convolution for video tokens.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// VideoLLaMA 2 (Alibaba, 2024) advances spatial-temporal modeling for video understanding
/// using convolution-based video token aggregation. It applies spatial-temporal convolutions
/// to compress frame-level visual tokens along both spatial and temporal dimensions, with
/// optional audio branch support for multi-modal video understanding.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "VideoLLaMA 2: Advancing Spatial-Temporal Modeling" (Alibaba, 2024)</item></list></para>
/// <para><b>For Beginners:</b> VideoLLaMA 2 is a video-language model with spatial-temporal
/// convolution for efficient video token processing. Default values follow the original
/// paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a VideoLLaMA 2 model for spatial-temporal video understanding
/// // with convolution-based video token aggregation and audio support
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.FourDimensional,
///     taskType: NeuralNetworkTaskType.Classification,
///     inputHeight: 336, inputWidth: 336, inputDepth: 3, inputFrames: 8, outputSize: 512);
///
/// // ONNX inference mode with pre-trained model
/// var model = new VideoLLaMA2&lt;double&gt;(architecture, "videollama2.onnx");
///
/// // Training mode with native layers
/// var trainModel = new VideoLLaMA2&lt;double&gt;(architecture, new VideoLLaMA2Options());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelDomain(ModelDomain.Video)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "VideoLLaMA 2: Advancing Spatial-Temporal Modeling and Audio Understanding in Video-LLMs",
    "https://arxiv.org/abs/2406.07476",
    Year = 2024,
    Authors = "Cheng et al."
)]
public class VideoLLaMA2<T> : VisionLanguageModelBase<T>, IVideoLanguageModel<T>
{
    private readonly VideoLLaMA2Options _options;

    public override ModelOptions GetOptions() => _options;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;
    private int _encoderLayerEnd;
    private int _decoderLayerStart;

    // The STC connector contains a composite reshape/permute/Conv3D graph whose backward is not
    // currently safe to capture and replay as one fused training plan. The eager tape executes the
    // same paper architecture and optimizer without poisoning the model's parameters on step one.
    protected override bool SupportsFusedCompiledTraining => false;

    public VideoLLaMA2(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        VideoLLaMA2Options? options = null
    )
        : base(architecture)
    {
        _options = options ?? new VideoLLaMA2Options();
        ValidateOptions(_options);
        _useNativeMode = false;
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.DecoderDim;
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        InitializeLayers();
    }

    public VideoLLaMA2(
        NeuralNetworkArchitecture<T> architecture,
        VideoLLaMA2Options? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new VideoLLaMA2Options();
        if (architecture.InputType == InputType.FourDimensional && architecture.InputHeight > 0)
        {
            if (architecture.InputWidth > 0 && architecture.InputWidth != architecture.InputHeight)
                throw new ArgumentException(
                    "VideoLLaMA2 native mode requires square frames (InputWidth must equal InputHeight).",
                    nameof(architecture)
                );
            if (architecture.InputDepth > 0 && architecture.InputDepth != 3)
                throw new ArgumentException(
                    "VideoLLaMA2 native mode requires 3-channel RGB frames (InputDepth must equal 3).",
                    nameof(architecture)
                );
            _options = new VideoLLaMA2Options(_options) { ImageSize = architecture.InputHeight };
        }
        ValidateOptions(_options);
        _useNativeMode = true;
        // The released first-stage connector training recipe uses AdamW at 1e-3 with zero weight decay.
        // Both values are options, and callers may still inject an entirely different optimizer.
        _optimizer = optimizer ?? CreateDefaultOptimizer();
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.DecoderDim;
        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        InitializeLayers();
    }

    public int EmbeddingDimension => _options.DecoderDim;
    int IVisualEncoder<T>.ImageSize => _options.ImageSize;
    int IVisualEncoder<T>.ImageChannels => 3;
    public int MaxGenerationLength => _options.MaxGenerationLength;
    public int DecoderEmbeddingDim => _options.DecoderDim;
    public string LanguageModelName => _options.LanguageModelName;
    public int MaxFrames => _options.MaxFrames;

    public Tensor<T> EncodeImage(Tensor<T> image)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return L2Normalize(OnnxModel.Run(p));
        return L2Normalize(EncodeFrameFeatures(p, alreadyPreprocessed: true));
    }

    /// <summary>
    /// Generates from a single image using VideoLLaMA 2's STC connector in single-frame mode.
    /// For a single image, the spatial-temporal convolution reduces to spatial processing only.
    /// Visual features are fused with text tokens via cross-attention before the LLM decoder.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);

        var encoderOut = EncodeFrameFeatures(p, alreadyPreprocessed: true);
        Tensor<T> bridgeInput = encoderOut;
        if (_options.EnableSpatialTemporalConv)
        {
            // A custom no-padding kernel may require more than one temporal sample. Repeating a
            // still frame is the standard image-as-video representation and keeps the STC path intact.
            int requiredFrames = Math.Max(1, _options.STCKernelSize - 2 * _options.STCPadding);
            var repeated = Enumerable.Repeat(encoderOut, requiredFrames).ToArray();
            bridgeInput = Engine.TensorStack(repeated, axis: 0);
        }

        return RunConnectorAndDecoder(bridgeInput, prompt);
    }

    /// <summary>
    /// Generates output from video frames using VideoLLaMA 2's Spatial-Temporal Convolution (STC)
    /// connector. Per the paper (Alibaba 2024), frame features are arranged into a 3D grid
    /// (temporal x spatial_h x spatial_w) and processed with depthwise separable 3D convolutions
    /// to capture both spatial layout and temporal dynamics. This compresses the video tokens
    /// while preserving spatiotemporal relationships, unlike simple averaging.
    /// </summary>
    public Tensor<T> GenerateFromVideo(IReadOnlyList<Tensor<T>> frames, string? prompt = null)
    {
        ThrowIfDisposed();
        int count = Math.Min(frames.Count, _options.MaxFrames);
        if (count == 0)
            throw new ArgumentException("At least one frame is required.", nameof(frames));

        if (IsOnnxMode && OnnxModel is not null)
        {
            var processedFrames = new Tensor<T>[count];
            for (int i = 0; i < count; i++)
                processedFrames[i] = PreprocessImage(frames[i]);
            return OnnxModel.Run(Engine.TensorStack(processedFrames, axis: 0));
        }

        // Encode every frame without L2-normalizing away the magnitude information consumed by STC.
        var frameFeatures = new Tensor<T>[count];
        for (int f = 0; f < count; f++)
            frameFeatures[f] = EncodeFrameFeatures(frames[f]);

        if (_options.EnableSpatialTemporalConv)
        {
            int requiredFrames = Math.Max(1, _options.STCKernelSize - 2 * _options.STCPadding);
            if (frameFeatures.Length < requiredFrames)
            {
                Array.Resize(ref frameFeatures, requiredFrames);
                for (int i = count; i < requiredFrames; i++)
                    frameFeatures[i] = frameFeatures[count - 1];
            }
        }

        var videoFeatures = Engine.TensorStack(frameFeatures, axis: 0);
        return RunConnectorAndDecoder(videoFeatures, prompt);
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
            return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            _encoderLayerEnd = Layers.Count / 2;
            _decoderLayerStart = _encoderLayerEnd;
        }
        else
        {
            // VideoLLaMA2 (Cheng et al. 2024, arXiv:2406.07476) aggregates the per-frame residual-ViT
            // features with an STC (Spatial-Temporal Convolution) connector — a 3D conv over time+space —
            // before the LLM. Residual ViT + LLM fix the old shared builder's post-training collapse.
            Layers.AddRange(
                LayerHelper<T>.CreateDefaultVideoSTCVLMLayers(
                    visionDim: _options.VisionDim,
                    decoderDim: _options.DecoderDim,
                    numVisionLayers: _options.NumVisionLayers,
                    numDecoderLayers: _options.NumDecoderLayers,
                    visionNumHeads: _options.VisionNumHeads,
                    decoderNumHeads: _options.DecoderNumHeads,
                    decoderNumKeyValueHeads: _options.DecoderNumKeyValueHeads,
                    visionFfnDim: _options.VisionFfnDim,
                    decoderFfnDim: _options.DecoderFfnDim,
                    dropoutRate: _options.DropoutRate,
                    imageHeight: _options.ImageSize,
                    imageWidth: _options.ImageSize,
                    imageChannels: 3,
                    patchSize: _options.PatchSize,
                    maxSequenceLength: _options.MaxSequenceLength,
                    ropeTheta: _options.RoPETheta,
                    enableSpatialTemporalConv: _options.EnableSpatialTemporalConv,
                    stcKernelSize: _options.STCKernelSize,
                    stcStride: _options.STCStride,
                    stcPadding: _options.STCPadding,
                    stcStageDepth: _options.STCStageDepth,
                    stcMlpDepth: _options.STCMlpDepth
                )
            );
            ComputeEncoderDecoderBoundary();
        }
    }

    private void ComputeEncoderDecoderBoundary()
    {
        // Patch embedding + pre-LN + one composite residual block per CLIP layer.
        _encoderLayerEnd = 2 + _options.NumVisionLayers;
        // The connector is one composite layer; the explicit no-STC fallback is an MLP stack.
        _decoderLayerStart = _encoderLayerEnd
            + (_options.EnableSpatialTemporalConv ? 1 : _options.STCMlpDepth);
    }

    private Tensor<T> EncodeFrameFeatures(Tensor<T> image, bool alreadyPreprocessed = false)
    {
        var output = alreadyPreprocessed ? image : PreprocessImage(image);
        for (int i = 0; i < _encoderLayerEnd; i++)
            output = Layers[i].Forward(output);
        return output;
    }

    private Tensor<T> RunConnectorAndDecoder(Tensor<T> visualFeatures, string? prompt)
    {
        var output = visualFeatures;
        for (int i = _encoderLayerEnd; i < _decoderLayerStart; i++)
            output = Layers[i].Forward(output);

        // Native mode currently owns only the visual connector/decoder weights; prompt embedding
        // weights come from the published checkpoint. Reject a silent fabricated embedding while
        // still allowing ONNX mode above to consume the complete exported graph.
        if (!string.IsNullOrEmpty(prompt))
            throw new NotSupportedException(
                "Prompt-conditioned generation in native mode requires a loaded VideoLLaMA2 language-token embedding checkpoint.");

        for (int i = _decoderLayerStart; i < Layers.Count; i++)
            output = Layers[i].Forward(output);
        return output;
    }

    private Tensor<T> TokenizeText(string text)
    {
        if (_tokenizer is null)
            throw new InvalidOperationException("Tokenizer not initialized.");
        var encoding = _tokenizer.Encode(text);
        int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength);
        var tokens = new Tensor<T>([seqLen]);
        for (int i = 0; i < seqLen; i++)
            tokens[i] = NumOps.FromDouble(encoding.TokenIds[i]);
        return tokens;
    }

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);
        var c = input;
        foreach (var l in Layers)
            c = l.Forward(c);
        return c;
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode)
            throw new NotSupportedException("Training is not supported in ONNX mode.");
        SetTrainingMode(true);
        TrainWithTape(input, expected, _optimizer);
        SetTrainingMode(false);
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode)
            throw new NotSupportedException("Cannot update parameters in ONNX mode.");
        int idx = 0;
        foreach (var l in Layers)
        {
            int c = (int)l.ParameterCount;
            l.UpdateParameters(parameters.Slice(idx, c));
            idx += c;
        }
    }

    protected override Tensor<T> PreprocessImage(Tensor<T> image) =>
        NormalizeImage(image, _options.ImageMean, _options.ImageStd);

    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "VideoLLaMA2-Native" : "VideoLLaMA2-ONNX",
            Description = "VideoLLaMA 2: spatial-temporal convolution for video tokens.",
            FeatureCount = _options.DecoderDim,
            Complexity = _options.NumVisionLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "VideoLLaMA2";
        m.AdditionalInfo["LanguageModel"] = _options.LanguageModelName;
        m.AdditionalInfo["VisionEncoder"] = _options.VisionEncoderName;
        m.AdditionalInfo["SpatialTemporalConv"] = _options.EnableSpatialTemporalConv.ToString();
        m.AdditionalInfo["PatchSize"] = _options.PatchSize.ToString(System.Globalization.CultureInfo.InvariantCulture);
        m.AdditionalInfo["STCStageDepth"] = _options.STCStageDepth.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_useNativeMode);
        writer.Write(_options.ModelPath ?? string.Empty);
        writer.Write(_options.ImageSize);
        writer.Write(_options.VisionDim);
        writer.Write(_options.DecoderDim);
        writer.Write(_options.NumVisionLayers);
        writer.Write(_options.NumDecoderLayers);
        writer.Write(_options.NumHeads);
        writer.Write(_options.MaxFrames);
        writer.Write(_options.EnableSpatialTemporalConv);
        writer.Write(_options.VisionEncoderName);
        writer.Write(_options.PatchSize);
        writer.Write(_options.VisionNumHeads);
        writer.Write(_options.DecoderNumHeads);
        writer.Write(_options.DecoderNumKeyValueHeads);
        writer.Write(_options.VisionFfnDim);
        writer.Write(_options.DecoderFfnDim);
        writer.Write(_options.RoPETheta);
        writer.Write(_options.STCKernelSize);
        writer.Write(_options.STCStride);
        writer.Write(_options.STCPadding);
        writer.Write(_options.STCStageDepth);
        writer.Write(_options.STCMlpDepth);
        writer.Write(_options.LearningRate);
        writer.Write(_options.WeightDecay);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _useNativeMode = reader.ReadBoolean();
        string mp = reader.ReadString();
        if (!string.IsNullOrEmpty(mp))
            _options.ModelPath = mp;
        _options.ImageSize = reader.ReadInt32();
        _options.VisionDim = reader.ReadInt32();
        _options.DecoderDim = reader.ReadInt32();
        _options.NumVisionLayers = reader.ReadInt32();
        _options.NumDecoderLayers = reader.ReadInt32();
        _options.NumHeads = reader.ReadInt32();
        _options.MaxFrames = reader.ReadInt32();
        _options.EnableSpatialTemporalConv = reader.ReadBoolean();
        _options.VisionEncoderName = reader.ReadString();
        _options.PatchSize = reader.ReadInt32();
        _options.VisionNumHeads = reader.ReadInt32();
        _options.DecoderNumHeads = reader.ReadInt32();
        _options.DecoderNumKeyValueHeads = reader.ReadInt32();
        _options.VisionFfnDim = reader.ReadInt32();
        _options.DecoderFfnDim = reader.ReadInt32();
        _options.RoPETheta = reader.ReadDouble();
        _options.STCKernelSize = reader.ReadInt32();
        _options.STCStride = reader.ReadInt32();
        _options.STCPadding = reader.ReadInt32();
        _options.STCStageDepth = reader.ReadInt32();
        _options.STCMlpDepth = reader.ReadInt32();
        _options.LearningRate = reader.ReadDouble();
        _options.WeightDecay = reader.ReadDouble();
        ValidateOptions(_options);
        _optimizer = _useNativeMode ? CreateDefaultOptimizer() : null;
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new VideoLLaMA2<T>(Architecture, mp, _options);
        return new VideoLLaMA2<T>(Architecture, _options);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(VideoLLaMA2<T>));
    }

    private static void ValidateOptions(VideoLLaMA2Options options)
    {
        if (options.ImageSize <= 0) throw new ArgumentOutOfRangeException(nameof(options.ImageSize));
        if (options.PatchSize <= 0 || options.ImageSize % options.PatchSize != 0)
        {
            throw new ArgumentException(
                $"ImageSize ({options.ImageSize}) must be evenly divisible by PatchSize ({options.PatchSize}).",
                nameof(options));
        }
        if (options.MaxFrames <= 0) throw new ArgumentOutOfRangeException(nameof(options.MaxFrames));
        if (options.STCKernelSize <= 0) throw new ArgumentOutOfRangeException(nameof(options.STCKernelSize));
        if (options.STCStride <= 0) throw new ArgumentOutOfRangeException(nameof(options.STCStride));
        if (options.STCPadding < 0) throw new ArgumentOutOfRangeException(nameof(options.STCPadding));
        if (options.STCStageDepth < 0) throw new ArgumentOutOfRangeException(nameof(options.STCStageDepth));
        if (options.STCMlpDepth <= 0) throw new ArgumentOutOfRangeException(nameof(options.STCMlpDepth));
        if (options.LearningRate <= 0 || double.IsNaN(options.LearningRate) || double.IsInfinity(options.LearningRate))
            throw new ArgumentOutOfRangeException(nameof(options.LearningRate));
        if (options.WeightDecay < 0 || double.IsNaN(options.WeightDecay) || double.IsInfinity(options.WeightDecay))
            throw new ArgumentOutOfRangeException(nameof(options.WeightDecay));
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreateDefaultOptimizer() =>
        new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
                WeightDecay = _options.WeightDecay
            });

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
