using AiDotNet.Attributes;
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
/// SlowFast-LLaVA: token-efficient slow/fast pathways for long video.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// SlowFast-LLaVA (Meta, 2025) is a training-free video understanding baseline that uses
/// dual slow/fast pathways for token-efficient long video processing. The slow pathway processes
/// fewer frames at high spatial resolution, while the fast pathway samples many frames at low
/// spatial resolution, combining detailed spatial and broad temporal information efficiently.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "SlowFast-LLaVA: A Strong Training-Free Baseline for Video Large Language Models" (Meta, 2025)</item></list></para>
/// <para><b>For Beginners:</b> SlowFast-LLaVA is a training-free video-language model using
/// dual pathways for efficient long video understanding. Default values follow the original
/// paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a SlowFast-LLaVA model for long video understanding
/// // with token-efficient dual slow/fast pathways for temporal modeling
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.ImageClassification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
///
/// // ONNX inference mode with pre-trained model
/// var model = new SlowFastLLaVA&lt;double&gt;(architecture, "slowfastllava.onnx");
///
/// // Training mode with native layers
/// var trainModel = new SlowFastLLaVA&lt;double&gt;(architecture, new SlowFastLLaVAOptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelDomain(ModelDomain.Video)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "SlowFast-LLaVA: A Strong Training-Free Baseline for Video Large Language Models",
    "https://arxiv.org/abs/2407.15841",
    Year = 2024,
    Authors = "Xu et al."
)]
public partial class SlowFastLLaVA<T> : VisionLanguageModelBase<T>, IVideoLanguageModel<T>
{
    private readonly SlowFastLLaVAOptions _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;
    private int _encoderLayerEnd;

    public SlowFastLLaVA(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        SlowFastLLaVAOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new SlowFastLLaVAOptions();
        Options = _options;
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

    public SlowFastLLaVA(
        NeuralNetworkArchitecture<T> architecture,
        SlowFastLLaVAOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new SlowFastLLaVAOptions();
        Options = _options;
        if (architecture.InputType == InputType.FourDimensional && architecture.InputHeight > 0)
        {
            if (architecture.InputWidth > 0 && architecture.InputWidth != architecture.InputHeight)
                throw new ArgumentException(
                    "SlowFastLLaVA native mode requires square frames (InputWidth must equal InputHeight).",
                    nameof(architecture)
                );
            if (architecture.InputDepth > 0 && architecture.InputDepth != 3)
                throw new ArgumentException(
                    "SlowFastLLaVA native mode requires 3-channel RGB frames (InputDepth must equal 3).",
                    nameof(architecture)
                );
            _options = new SlowFastLLaVAOptions(_options) { ImageSize = architecture.InputHeight };
            Options = _options;
        }
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.DecoderDim;
        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        InitializeLayers();
    }

    /// <inheritdoc />
    public int EmbeddingDimension => _options.DecoderDim;
    int IVisualEncoder<T>.ImageSize => _options.ImageSize;
    int IVisualEncoder<T>.ImageChannels => 3;
    /// <inheritdoc />
    public int MaxGenerationLength => _options.MaxGenerationLength;
    /// <inheritdoc />
    public int DecoderEmbeddingDim => _options.DecoderDim;
    /// <inheritdoc />
    public string LanguageModelName => _options.LanguageModelName;
    /// <inheritdoc />
    public int MaxFrames => _options.MaxFrames;

    /// <inheritdoc />
    public Tensor<T> EncodeImage(Tensor<T> image)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return L2Normalize(OnnxModel.Run(p));
        var c = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            c = Layers[i].Forward(c);
        return L2Normalize(c);
    }

    /// <summary>
    /// Generates from a single image using SlowFast-LLaVA's slow pathway at full resolution.
    /// For a single image, only the Slow pathway applies (full spatial resolution, no temporal
    /// dynamics). The Fast pathway is skipped since there's no temporal dimension to capture.
    /// Text tokens are fused via cross-attention to condition the generation.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);

        int dim = _options.DecoderDim;

        // Step 1: Slow pathway vision encoder (full spatial resolution)
        var encoderOut = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            encoderOut = Layers[i].Forward(encoderOut);
        int visLen = encoderOut.Length;

        // Step 2: Tokenize prompt
        Tensor<T>? promptTokens = null;
        int promptLen = 0;
        if (prompt is not null)
        {
            promptTokens = TokenizeText(prompt);
            promptLen = promptTokens.Length;
        }

        // Step 3: Slow pathway visual features + text cross-attention
        // 2-layer MLP cross-modal connector (Linear -> GELU -> Linear)
        var decoderInput = GatedConnectorInput(encoderOut, promptTokens, dim, _options.VocabSize);

        // Step 4: LLM decoder
        var output = decoderInput;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);
        return output;
    }

    /// <summary>
    /// Generates output from video frames using SlowFast-LLaVA's dual-pathway architecture.
    /// Per the paper (Meta 2025), the Slow pathway processes a small number of frames (default 8)
    /// at full spatial resolution to capture fine-grained details, while the Fast pathway processes
    /// many frames (default 64) with aggressive spatial pooling to capture temporal dynamics.
    /// The two pathways are fused via lateral connections before being fed to the LLM.
    /// This is training-free: it uses an existing image VLM's encoder without new parameters.
    /// </summary>
    public Tensor<T> GenerateFromVideo(IReadOnlyList<Tensor<T>> frames, string? prompt = null)
    {
        ThrowIfDisposed();
        int totalFrames = Math.Min(frames.Count, _options.MaxFrames);
        if (totalFrames == 0)
            throw new ArgumentException("At least one frame is required.", nameof(frames));

        int slowCount = Math.Min(_options.SlowFrames, totalFrames);
        int fastCount = Math.Min(_options.FastFrames, totalFrames);

        // === Slow Pathway: fewer frames, full spatial resolution ===
        // Uniformly sample slowCount frames from the video
        var slowFeatures = new Tensor<T>[slowCount];
        for (int s = 0; s < slowCount; s++)
        {
            int frameIdx =
                totalFrames > 1 ? (int)((long)s * (totalFrames - 1) / (slowCount - 1)) : 0;
            frameIdx = Math.Min(frameIdx, totalFrames - 1);
            slowFeatures[s] = EncodeImage(frames[frameIdx]);
        }

        int dim = slowFeatures[0].Length;

        // Slow pathway: preserve full spatial detail, average across slow frames
        var engine = Engine;
        var slowOutput = engine.TensorDivideScalar(
            engine.ReduceSum(StackLeading(slowFeatures, slowCount, dim), [0], false), NumOps.FromDouble(slowCount));

        // === Fast Pathway: many frames, aggressive spatial pooling ===
        // Uniformly sample fastCount frames and apply 2x spatial pooling (halve feature dim)
        int fastDim = dim / 2; // Aggressive spatial pooling reduces dimensionality
        var fastOutput = new Tensor<T>([fastDim]);
        T half = NumOps.FromDouble(0.5);

        for (int f = 0; f < fastCount; f++)
        {
            int frameIdx =
                totalFrames > 1 ? (int)((long)f * (totalFrames - 1) / (fastCount - 1)) : 0;
            frameIdx = Math.Min(frameIdx, totalFrames - 1);
            var enc = EncodeImage(frames[frameIdx]);

            // Spatial pooling: average adjacent pairs of features (2x reduction); missing values read as 0.
            var pairs = FitLength(enc, fastDim * 2).Reshape([fastDim, 2]);
            var pooled = engine.TensorMultiplyScalar(engine.ReduceSum(pairs, [1], false), half);
            fastOutput = engine.TensorAdd(fastOutput, pooled);
        }

        // Average fast pathway across frames
        T fastScale = NumOps.FromDouble(1.0 / fastCount);
        fastOutput = engine.TensorMultiplyScalar(fastOutput, fastScale);

        // === Lateral Fusion: combine slow and fast pathways ===
        // Per the paper, fast pathway features are projected up and added to slow pathway:
        // fused[d] = 0.7 * slow[d] + 0.3 * fast[d / 2] (0 past the fast features).
        var fastReplicated = FitLength(
            engine.TensorBroadcastTo(fastOutput.Reshape([fastDim, 1]), [fastDim, 2]).Reshape([fastDim * 2]), dim);
        var fused = engine.TensorAdd(
            engine.TensorMultiplyScalar(slowOutput, NumOps.FromDouble(0.7)),
            engine.TensorMultiplyScalar(fastReplicated, NumOps.FromDouble(0.3)));

        // Project fused features through the LLM decoder
        var output = fused;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);
        return output;
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
            return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            _encoderLayerEnd = Layers.Count / 2;
        }
        else
        {
            Layers.AddRange(
                LayerHelper<T>.CreateDefaultVideoTemporalVLMLayers(
                    _options.VisionDim,
                    _options.VisionDim,
                    _options.DecoderDim,
                    _options.NumVisionLayers,
                    2,
                    _options.NumDecoderLayers,
                    _options.NumHeads,
                    _options.DropoutRate,
                    imageHeight: _options.ImageSize,
                    imageWidth: _options.ImageSize,
                    imageChannels: 3,
                    patchSize: 16
                )
            );
            ComputeEncoderDecoderBoundary();
        }
    }

    private void ComputeEncoderDecoderBoundary()
    {
        int lpb = _options.DropoutRate > 0 ? 6 : 5;
        _encoderLayerEnd = 2 + _options.NumVisionLayers * lpb + 2 * lpb + 2;
    }

    private Tensor<T> TokenizeText(string text)
    {
        if (_tokenizer is null)
            throw new InvalidOperationException("Tokenizer not initialized.");
        var encoding = _tokenizer.Encode(text);
        int seqLen = Math.Min(encoding.TokenIds.Count, _options.MaxSequenceLength);
        return TokenIdsToTensor(encoding.TokenIds, seqLen);
    }

    /// <inheritdoc />
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

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    /// <inheritdoc />
    protected override Tensor<T> PreprocessImage(Tensor<T> image) =>
        NormalizeImage(image, _options.ImageMean, _options.ImageStd);

    /// <inheritdoc />
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "SlowFast-LLaVA-Native" : "SlowFast-LLaVA-ONNX",
            Description = "SlowFast-LLaVA: token-efficient slow/fast pathways for long video.",
            FeatureCount = _options.DecoderDim,
            Complexity = _options.NumVisionLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "SlowFast-LLaVA";
        m.AdditionalInfo["LanguageModel"] = _options.LanguageModelName;
        m.AdditionalInfo["SlowFrames"] = _options.SlowFrames.ToString();
        m.AdditionalInfo["FastFrames"] = _options.FastFrames.ToString();
        return m;
    }





    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(SlowFastLLaVA<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
