using AiDotNet.LearningRateSchedulers;
using AiDotNet.Enums;
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
/// Video-LLaVA: united visual representation for video understanding.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Video-LLaVA (PKU, 2024) learns united visual representations for video understanding by
/// aligning image and video features before projection into the language model. It processes
/// both images and videos through separate encoders that are aligned into a shared feature
/// space, enabling the model to leverage image-text knowledge for video understanding.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Video-LLaVA: Learning United Visual Representation by Alignment Before Projection" (PKU, 2024)</item></list></para>
/// <para><b>For Beginners:</b> Video-LLaVA is a video-language model that aligns image and
/// video features for unified visual understanding. Default values follow the original paper
/// settings.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Video-LLaVA model for unified video understanding
/// // with aligned visual representations before projection
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.TwoDimensional,
///     taskType: NeuralNetworkTaskType.ImageClassification,
///     inputHeight: 224, inputWidth: 224, inputDepth: 3, outputSize: 512);
///
/// // ONNX inference mode with pre-trained model
/// var model = new VideoLLaVA&lt;double&gt;(architecture, "videollava.onnx");
///
/// // Training mode with native layers
/// var trainModel = new VideoLLaVA&lt;double&gt;(architecture, new VideoLLaVAOptions());
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
    "Video-LLaVA: Learning United Visual Representation by Alignment Before Projection",
    "https://arxiv.org/abs/2311.10122",
    Year = 2024,
    Authors = "Lin et al."
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 0.001, ReferenceBatchSize = 256,
                WarmupFraction = 0.03,
                Schedule = LearningRateSchedulerType.CosineAnnealing,
                Source = "Lin et al. 2024, Sec. 4: the first stage trains for one epoch at a batch size "
                        + "of 256 using the AdamW optimizer with a cosine learning rate schedule. The "
                        + "initial learning rate for both stages is 1e-3 with a warmup ratio of 0.03, so "
                        + "the rate is not blocked by the stage split.")]
public partial class VideoLLaVA<T> : VisionLanguageModelBase<T>, IVideoLanguageModel<T>
{
    private readonly VideoLLaVAOptions _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;
    private int _encoderLayerEnd;

    public VideoLLaVA(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        VideoLLaVAOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new VideoLLaVAOptions();
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

    public VideoLLaVA(
        NeuralNetworkArchitecture<T> architecture,
        VideoLLaVAOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new VideoLLaVAOptions();
        Options = _options;
        if (architecture.InputType == InputType.FourDimensional && architecture.InputHeight > 0)
        {
            _options = new VideoLLaVAOptions(_options) { ImageSize = architecture.InputHeight };
            Options = _options;
        }
        _useNativeMode = true;
        _optimizer = optimizer
    ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
    ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
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
    /// Generates from a single image using Video-LLaVA's alignment-before-projection approach.
    /// A single image is treated as a single-frame video: the LanguageBind encoder extracts visual
    /// features, alignment embedding is added, then visual tokens are fused with text tokens via
    /// cross-attention before MLP projection to LLM space.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);

        int dim = _options.DecoderDim;

        // Step 1: LanguageBind vision encoder
        var encoderOut = p;
        for (int i = 0; i < _encoderLayerEnd; i++)
            encoderOut = Layers[i].Forward(encoderOut);

        // Fuse visual features with prompt tokens via ConcatenateTensors
        Tensor<T> fusedInput;
        if (prompt is not null)
        {
            var promptTokens = TokenizeText(prompt);
            fusedInput = encoderOut.ConcatenateTensors(promptTokens);
        }
        else
        {
            fusedInput = encoderOut;
        }

        var output = fusedInput;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);
        return output;
    }

    /// <summary>
    /// Generates output from video frames using Video-LLaVA's unified alignment before projection.
    /// Per the paper (PKU 2024), frames are encoded with a shared LanguageBind vision encoder,
    /// then aligned into a unified visual representation via learned temporal embedding before
    /// being projected to the LLM embedding space. This alignment step ensures temporal coherence
    /// across frames by adding sinusoidal temporal position embeddings and applying a learned
    /// linear alignment layer before the MLP projection to LLM space.
    /// </summary>
    public Tensor<T> GenerateFromVideo(IReadOnlyList<Tensor<T>> frames, string? prompt = null)
    {
        ThrowIfDisposed();
        int count = Math.Min(frames.Count, _options.MaxFrames);
        if (count == 0)
            throw new ArgumentException("At least one frame is required.", nameof(frames));

        // Step 1: Encode each frame through the shared LanguageBind vision encoder
        var frameFeatures = new Tensor<T>[count];
        for (int f = 0; f < count; f++)
            frameFeatures[f] = EncodeImage(frames[f]);

        int dim = frameFeatures[0].Length;

        // Step 2: Add sinusoidal temporal position embeddings (alignment before projection)
        // Video-LLaVA uses temporal embeddings to distinguish frame order
        var engine = Engine;
        var stacked = StackLeading(frameFeatures, count, dim);
        Tensor<T> unified;
        if (dim == 0)
        {
            unified = new Tensor<T>([0]);
        }
        else
        {
            // embedding[f, d] = sin / cos(pos_f * freq_d * pi) for even / odd d, with
            // pos_f = f / max(1, count - 1) and freq_d = 10000^(-2 * floor(d / 2) / dim).
            var featureIndex = engine.TensorArange<T>(0, dim, 1);
            var pairIndex = engine.TensorMultiplyScalar(
                engine.TensorFloor(engine.TensorMultiplyScalar(featureIndex, NumOps.FromDouble(0.5))), NumOps.FromDouble(2.0));
            var freq = engine.TensorExp(engine.TensorMultiplyScalar(pairIndex, NumOps.FromDouble(-Math.Log(10000.0) / dim)));
            var pos = engine.TensorMultiplyScalar(
                engine.TensorArange<T>(0, count, 1), NumOps.FromDouble(Math.PI / Math.Max(1, count - 1)));
            var phase = engine.TensorOuter(pos, freq);
            var even = engine.TensorEquals(
                engine.TensorRemainder(featureIndex, Tensor<T>.CreateDefault([dim], NumOps.FromDouble(2))), NumOps.Zero);
            var embedding = engine.TensorWhere(
                engine.TensorBroadcastTo(even.Reshape([1, dim]), [count, dim]), engine.TensorSin(phase), engine.TensorCos(phase));
            stacked = engine.TensorAdd(stacked, engine.TensorMultiplyScalar(embedding, NumOps.FromDouble(0.1)));

            // Step 3: Unified alignment - weighted temporal aggregation with soft attention on
            // feature magnitude: weight = exp(0.5 |x|); unified = sum(x * w) / sum(w) when sum(w) > 1e-8.
            var weight = engine.TensorExp(engine.TensorMultiplyScalar(engine.TensorAbs(stacked), NumOps.FromDouble(0.5)));
            var sumWeighted = engine.ReduceSum(engine.TensorMultiply(stacked, weight), [0], false);
            var sumWeights = engine.ReduceSum(weight, [0], false);
            var usable = engine.TensorGreaterThan(sumWeights, NumOps.FromDouble(1e-8));
            var ones = Tensor<T>.CreateDefault([dim], NumOps.One);
            unified = engine.TensorWhere(
                usable, engine.TensorDivide(sumWeighted, engine.TensorWhere(usable, sumWeights, ones)), new Tensor<T>([dim]));
        }

        // Step 4: Project aligned features through the LLM decoder
        var output = unified;
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
            // Video-LLaVA (Lin et al. 2024, arXiv:2311.10122) = LanguageBind pre-aligned vision
            // encoder → shared LINEAR projection → LLM (alignment before projection; no separate
            // temporal transformer). Residual ViT + LLM blocks (the old shared builder's
            // non-residual stack collapsed to an input-independent output after training).
            Layers.AddRange(
                LayerHelper<T>.CreateDefaultVideoLinearProjectorVLMLayers(
                    _options.VisionDim,
                    _options.DecoderDim,
                    _options.NumVisionLayers,
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
        // Layout of CreateDefaultVideoLinearProjectorVLMLayers: PatchEmbedding(1) + LayerNorm(1)
        // + numVisionLayers·(TransformerEncoderLayer(1) + optional Dropout) then the projection +
        // LLM decoder. The vision-encoder segment (what EncodeImage runs to produce visual
        // features, before the LLM projection) ends after the vision blocks.
        int perBlock = _options.DropoutRate > 0 ? 2 : 1;
        _encoderLayerEnd = 2 + _options.NumVisionLayers * perBlock;
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
            Name = _useNativeMode ? "Video-LLaVA-Native" : "Video-LLaVA-ONNX",
            Description = "Video-LLaVA: united visual representation for video understanding.",
            FeatureCount = _options.DecoderDim,
            Complexity = _options.NumVisionLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "Video-LLaVA";
        m.AdditionalInfo["LanguageModel"] = _options.LanguageModelName;
        return m;
    }





    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(VideoLLaVA<T>));
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
