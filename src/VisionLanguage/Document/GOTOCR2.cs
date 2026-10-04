using AiDotNet.LearningRateSchedulers;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Extensions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.VisionLanguage.Document;

/// <summary>
/// GOT-OCR2: 580M unified OCR model for text, tables, charts, equations, and music scores.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// GOT-OCR2 (StepFun, 2024) is a 580M unified end-to-end OCR model that handles diverse visual
/// text including plain text, tables, charts, mathematical equations, and music scores. A SAM ViTDet-B
/// encoder reads a 1024 px page. Two stride-2 convolutions reduce it to 256 image tokens, which fill the
/// image slots of a ChatML prompt for the Qwen-0.5B decoder. The decoder then generates the text directly,
/// with no separate detection or recognition stage.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "General OCR Theory: Towards OCR-2.0 via a Unified End-to-end Model" (StepFun, 2024)</item></list></para>
/// <para><b>For Beginners:</b> GOT-OCR2 is a unified OCR model that handles text, tables,
/// charts, equations, and music scores. Default values follow the original paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a GOT-OCR2 model for unified end-to-end OCR
/// // handling text, tables, charts, equations, and music scores
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.TextGeneration,
///     inputHeight: 1024, inputWidth: 1024, inputDepth: 3, outputSize: 151860);
///
/// // ONNX inference mode with pre-trained model
/// var model = new GOTOCR2&lt;double&gt;(architecture, "gotocr2.onnx");
///
/// // Native SAM ViTDet + Qwen model; read a page
/// var trainModel = new GOTOCR2&lt;double&gt;(architecture, new GOTOCR2Options());
/// string text = trainModel.ReadText(pageImage);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "General OCR Theory: Towards OCR-2.0 via a Unified End-to-end Model",
    "https://arxiv.org/abs/2409.01704",
    Year = 2024,
    Authors = "Wei et al."
)]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-4, ReferenceBatchSize = 128,
                MinLearningRate = 0,
                Schedule = LearningRateSchedulerType.CosineAnnealing,
                Phase = TrainingPhase.PreTraining,
                Source = "Wei et al. 2024, Sec. 3.3: the AdamW optimizer with a cosine annealing "
                        + "scheduler and a start learning rate of 1e-4, at a global batch size of 128 "
                        + "over 3 epochs of pre-training.")]
public partial class GOTOCR2<T> : VisionLanguageModelBase<T>, IDocumentUnderstandingModel<T>
{
    private readonly GOTOCR2Options _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    private const string SystemTurn = "system\nYou should follow the instructions carefully and explain your answers in detail.";

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;
    private SamViTDetEncoderLayer<T>? _vision;
    private GotOcr2ProjectorLayer<T>? _projector;
    private QwenDecoderLayer<T>? _decoder;

    /// <summary>Creates GOT-OCR2 backed by an ONNX export.</summary>
    public GOTOCR2(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        GOTOCR2Options? options = null,
        ITokenizer? tokenizer = null
    )
        : base(architecture)
    {
        _options = options ?? new GOTOCR2Options();
        _options.Validate();
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
        _tokenizer = tokenizer ?? LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.Qwen);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a trainable GOT-OCR2. With no options it builds the published model: SAM ViTDet-B, the conv
    /// projector, and Qwen-0.5B. The default tokenizer is the repository's Qwen-style BPE. Pass the checkpoint's
    /// tokenizer for parity.
    /// </summary>
    public GOTOCR2(
        NeuralNetworkArchitecture<T> architecture,
        GOTOCR2Options? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ITokenizer? tokenizer = null
    )
        : base(architecture)
    {
        _options = options ?? new GOTOCR2Options();
        if (architecture.InputType == InputType.ThreeDimensional && architecture.InputHeight > 0
            && architecture.InputHeight == architecture.InputWidth)
            _options.ImageSize = architecture.InputHeight;
        _options.Validate();
        _useNativeMode = true;
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.DecoderDim;
        _tokenizer = tokenizer ?? LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.Qwen);
        InitializeLayers();
    }

    /// <inheritdoc/>
    public int EmbeddingDimension => _options.DecoderDim;
    int IVisualEncoder<T>.ImageSize => _options.ImageSize;
    int IVisualEncoder<T>.ImageChannels => 3;

    /// <summary>Most tokens one generation produces.</summary>
    public int MaxGenerationLength => _options.MaxGenerationLength;

    /// <summary>The decoder width, which the projected image tokens share.</summary>
    public int DecoderEmbeddingDim => _options.DecoderDim;

    /// <inheritdoc/>
    public bool IsOcrFree => _options.IsOcrFree;

    /// <summary>Number of image tokens: the patch grid reduced 4x per side by the projector.</summary>
    public int ImageTokenCount => (_options.ImageSize / _options.PatchSize / 4) * (_options.ImageSize / _options.PatchSize / 4);

    private SamViTDetEncoderLayer<T> Vision => _vision ?? throw NotNative();
    private GotOcr2ProjectorLayer<T> Projector => _projector ?? throw NotNative();
    private QwenDecoderLayer<T> Decoder => _decoder ?? throw NotNative();

    private NotSupportedException NotNative() => new(_useNativeMode
        ? "This GOT-OCR2 was built from custom layers, so it has no encoder/decoder graph to prompt."
        : "GOT-OCR2 is in ONNX mode; the native model is not built.");

    /// <summary>The page's projected image tokens <c>[ImageTokenCount, DecoderDim]</c>.</summary>
    public Tensor<T> EncodeImage(Tensor<T> image)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);
        SetTrainingMode(false);
        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        return ImageTokens(p);
    }

    /// <summary>
    /// Greedily reads the page and returns the generated token ids, ending with <c>&lt;|im_end|&gt;</c> when one
    /// was produced. <paramref name="prompt"/> replaces the default query ("OCR: ", or "OCR with format: ").
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, string? prompt = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);
        SetTrainingMode(false);
        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        return TokenTensor(Generate(ImageTokens(p), prompt ?? DefaultQuery));
    }

    /// <summary>Reads the page under GOT-OCR2's OCR query and returns the generated token ids.</summary>
    public Tensor<T> ExtractText(Tensor<T> documentImage)
    {
        ThrowIfDisposed();
        return GenerateFromImage(documentImage);
    }

    /// <summary>Reads the page and decodes the answer to text.</summary>
    public string ReadText(Tensor<T> documentImage, string? prompt = null)
    {
        var ids = GenerateFromImage(documentImage, prompt);
        var tokens = new List<int>();
        for (int i = 0; i < ids.Length; i++)
        {
            int id = (int)Math.Round(NumOps.ToDouble(ids[i]));
            if (id == _options.ImEndTokenId || id == _options.EndOfTextTokenId) break;
            if (id < TextLimit) tokens.Add(id);
        }
        return tokens.Count > 0 ? _tokenizer.Decode(tokens, skipSpecialTokens: true) : string.Empty;
    }

    /// <summary>
    /// GOT-OCR2 is trained for reading, not question answering. The question replaces the OCR query in the user
    /// turn, which is the closest faithful use.
    /// </summary>
    public Tensor<T> AnswerDocumentQuestion(Tensor<T> documentImage, string question)
    {
        ThrowIfDisposed();
        if (string.IsNullOrWhiteSpace(question)) throw new ArgumentException("A question is required.", nameof(question));
        return GenerateFromImage(documentImage, question);
    }

    /// <summary>
    /// Teacher-forced logits <c>[T, vocab]</c>. Row t predicts the token after <c>answer[t - 1]</c>. Row 0 comes
    /// from the prompt's last position and predicts the answer's first token.
    /// </summary>
    public Tensor<T> PredictTokens(Tensor<T> image, IReadOnlyList<int> answer, string? prompt = null)
    {
        ThrowIfDisposed();
        if (answer is null) throw new ArgumentNullException(nameof(answer));
        SetTrainingMode(false);
        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        return AnswerLogits(ImageTokens(PreprocessImage(image)), prompt ?? DefaultQuery, answer);
    }

    /// <summary>One teacher-forced step that teaches the model to read <paramref name="image"/> as <paramref name="answerIds"/>.</summary>
    public void TrainOcr(Tensor<T> image, IReadOnlyList<int> answerIds, string? prompt = null)
    {
        if (answerIds is null) throw new ArgumentNullException(nameof(answerIds));
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        var target = TokenTensor(answerIds);
        string query = prompt ?? DefaultQuery;
        TrainWithCustomObjective(PreprocessImage(image), target, (pixels, labels) => Loss(pixels, labels, query), _optimizer);
    }

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
            return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            return;
        }
        var vision = new SamViTDetEncoderLayer<T>(_options.ImageSize, _options.PatchSize, _options.VisionDim,
            _options.NumVisionLayers, _options.NumHeads, _options.VisionMlpDim, _options.WindowSize,
            _options.GlobalAttentionEvery, _options.NeckChannels);
        var projector = new GotOcr2ProjectorLayer<T>(_options.NeckChannels, _options.DecoderDim, vision.Grid);
        var decoder = new QwenDecoderLayer<T>(_options.VocabSize, _options.DecoderDim, _options.DecoderHeads,
            _options.DecoderKeyValueHeads, _options.DecoderFeedForwardDim, _options.NumDecoderLayers,
            _options.MaxSequenceLength, _options.RopeTheta, _options.RmsNormEpsilon);
        Layers.Add(LayerGraphContract.FromExternalInput(vision));
        Layers.Add(projector);
        // The image tokens enter the decoder through its <imgpad> slots, not as its sequential input.
        Layers.Add(LayerGraphContract.FromDerivedInput(decoder, "text"));
        _vision = vision;
        _projector = projector;
        _decoder = decoder;
    }

    private string DefaultQuery => _options.FormatOutput ? "OCR with format: " : "OCR: ";

    /// <summary>Ordinary text ids stay below every special id, the image tokens included.</summary>
    private int TextLimit => Math.Min(_options.ImageStartTokenId,
        Math.Min(_options.EndOfTextTokenId, Math.Min(_options.ImStartTokenId, _options.ImEndTokenId)));

    private IEnumerable<int> Text(string text) =>
        _tokenizer.Encode(text).TokenIds.Select(id => Math.Min(Math.Max(id, 0), TextLimit - 1));

    private Tensor<T> ImageTokens(Tensor<T> preprocessed)
    {
        var image = preprocessed.Rank == 4 && preprocessed.Shape[0] == 1
            ? Engine.Reshape(preprocessed, new[] { preprocessed.Shape[1], preprocessed.Shape[2], preprocessed.Shape[3] })
            : preprocessed;
        return Projector.Forward(Vision.Forward(image));
    }

    /// <summary>
    /// The ChatML prompt the GOT-OCR2 processor builds, with one <c>&lt;imgpad&gt;</c> slot per image token:
    /// <c>&lt;|im_start|&gt;system ...&lt;|im_end|&gt;&lt;|im_start|&gt;user\n&lt;img&gt;[pads]&lt;/img&gt;\n{query}&lt;|im_end|&gt;&lt;|im_start|&gt;assistant\n</c>.
    /// </summary>
    internal (List<int> Ids, int ImageStart) Prompt(string query)
    {
        var ids = new List<int> { _options.ImStartTokenId };
        ids.AddRange(Text(SystemTurn));
        ids.Add(_options.ImEndTokenId);
        ids.Add(_options.ImStartTokenId);
        ids.AddRange(Text("user\n"));
        ids.Add(_options.ImageStartTokenId);
        int imageStart = ids.Count;
        for (int i = 0; i < ImageTokenCount; i++) ids.Add(_options.ImagePadTokenId);
        ids.Add(_options.ImageEndTokenId);
        ids.AddRange(Text("\n" + query));
        ids.Add(_options.ImEndTokenId);
        ids.Add(_options.ImStartTokenId);
        ids.AddRange(Text("assistant\n"));
        return (ids, imageStart);
    }

    /// <summary>Logits for the prompt followed by <paramref name="answer"/>, keeping the rows that predict the answer.</summary>
    private Tensor<T> AnswerLogits(Tensor<T> imageTokens, string query, IReadOnlyList<int> answer)
    {
        var (ids, imageStart) = Prompt(query);
        int promptLength = ids.Count;
        if (answer.Count == 0) throw new ArgumentException("The answer needs at least one token.", nameof(answer));
        // Row promptLength - 1 + t predicts answer[t], so the final answer token is never fed back in.
        for (int t = 0; t < answer.Count - 1; t++) ids.Add(answer[t]);
        if (ids.Count > Decoder.MaxPositions)
            throw new ArgumentException($"The prompt and answer ({ids.Count} tokens) exceed MaxSequenceLength ({Decoder.MaxPositions}).", nameof(answer));
        var logits = Decoder.Forward(ids, imageTokens, imageStart);
        return Engine.TensorSlice(logits, new[] { promptLength - 1, 0 }, new[] { answer.Count, _options.VocabSize });
    }

    private List<int> Generate(Tensor<T> imageTokens, string query)
    {
        var decoder = Decoder;
        var (ids, imageStart) = Prompt(query);
        int limit = Math.Min(_options.MaxGenerationLength, decoder.MaxPositions - ids.Count);
        return GreedyDecode(context => decoder.Forward(context, imageTokens, imageStart), ids, limit,
            (token, _) => token == _options.ImEndTokenId || token == _options.EndOfTextTokenId);
    }

    /// <summary>
    /// Cross-entropy against <paramref name="target"/>. A <c>[1, vocab]</c> target is a distribution over the
    /// first answer token. Anything else is answer ids, trained with teacher forcing.
    /// </summary>
    private Tensor<T> Loss(Tensor<T> preprocessed, Tensor<T> target, string query)
    {
        var decoder = Decoder;
        var imageTokens = ImageTokens(preprocessed);
        if (target.Rank == 2 && target.Shape[0] == 1 && target.Shape[1] == _options.VocabSize)
        {
            var (ids, imageStart) = Prompt(query);
            var logits = decoder.Forward(ids, imageTokens, imageStart);
            return SoftTargetCrossEntropy(Engine.TensorSlice(logits, new[] { ids.Count - 1, 0 }, new[] { 1, _options.VocabSize }), target);
        }

        var labels = TokenIds(target, decoder.ClampToken);
        if (labels.Length == 0) throw new ArgumentException("A GOT-OCR2 target needs at least one token.", nameof(target));
        return TokenCrossEntropy(AnswerLogits(imageTokens, query, labels), labels);
    }

    /// <summary>Next-token logits <c>[1, vocab]</c> at the end of the OCR prompt: the answer's first token.</summary>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);
        if (_decoder is null)
        {
            var c = input;
            foreach (var l in Layers)
                c = l.Forward(c);
            return c;
        }
        SetTrainingMode(false);
        var (ids, imageStart) = Prompt(DefaultQuery);
        var logits = _decoder.Forward(ids, ImageTokens(PreprocessImage(input)), imageStart);
        return Engine.TensorSlice(logits, new[] { ids.Count - 1, 0 }, new[] { 1, _options.VocabSize });
    }

    /// <summary>
    /// One training step under the OCR query. A <c>[1, vocab]</c> target is a distribution over the first
    /// answer token; any other target is answer ids, trained with teacher forcing.
    /// </summary>
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode)
            throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (expected is null) throw new ArgumentNullException(nameof(expected));
        if (_decoder is null)
        {
            SetTrainingMode(true);
            TrainWithTape(input, expected, _optimizer);
            SetTrainingMode(false);
            return;
        }
        string query = DefaultQuery;
        TrainWithCustomObjective(PreprocessImage(input), expected, (pixels, target) => Loss(pixels, target, query), _optimizer);
    }

    /// <inheritdoc/>
    protected override bool SupportsParameterMutation => _useNativeMode;

    /// <inheritdoc/>
    protected override Tensor<T> PreprocessImage(Tensor<T> image) =>
        NormalizeImage(image, _options.ImageMean, _options.ImageStd);

    /// <inheritdoc/>
    protected override Tensor<T> PostprocessOutput(Tensor<T> output) => output;

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "GOT-OCR2-Native" : "GOT-OCR2-ONNX",
            Description =
                "GOT-OCR2: a SAM ViTDet encoder and a Qwen-0.5B decoder reading text, tables, charts, formulas and music.",
            FeatureCount = _options.DecoderDim,
            Complexity = _options.NumVisionLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "GOT-OCR2 (SAM ViTDet-B + Qwen)";
        m.AdditionalInfo["OcrFree"] = _options.IsOcrFree.ToString();
        m.AdditionalInfo["ImageTokens"] = ImageTokenCount.ToString(System.Globalization.CultureInfo.InvariantCulture);
        m.AdditionalInfo["FormatOutput"] = _options.FormatOutput.ToString();
        return m;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(GOTOCR2<T>));
    }

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        if (disposing)
        {
            OnnxModel?.Dispose();
        }
        base.Dispose(disposing);
    }
}
