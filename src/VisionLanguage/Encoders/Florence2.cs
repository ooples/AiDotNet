using AiDotNet.LearningRateSchedulers;
using AiDotNet.Enums;
using AiDotNet.Attributes;
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

namespace AiDotNet.VisionLanguage.Encoders;

/// <summary>
/// Florence-2 unified vision foundation model for captioning, detection, grounding, and OCR.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Florence-2 (Xiao et al., Microsoft 2024) is a lightweight sequence-to-sequence vision model
/// (0.23B-0.77B) handling multiple tasks through a unified prompt-based approach. It uses DaViT
/// (Dual Attention ViT) as the vision encoder and a multi-task decoder for captioning, detection,
/// grounding, OCR, and segmentation.
/// </para>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks" (Xiao et al., 2024)</item></list></para>
/// <para><b>For Beginners:</b> Florence-2 from Microsoft is a lightweight vision model
/// (0.23B-0.77B parameters) that handles many tasks through text prompts — captioning,
/// object detection, grounding, OCR, and segmentation — all in a single unified model.
/// It uses DaViT (Dual Attention ViT) as its vision encoder and generates structured
/// text output for each task. Default values follow the original paper settings.</para>
/// </remarks>
/// <example>
/// <code>
/// // Create a Florence-2 model for unified multi-task vision understanding
/// // handling captioning, detection, grounding, OCR, and segmentation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.TextGeneration,
///     inputHeight: 768, inputWidth: 768, inputDepth: 3, outputSize: 51289);
///
/// // ONNX inference mode with pre-trained model
/// var model = new Florence2&lt;double&gt;(architecture, "florence2.onnx");
///
/// // Native DaViT + BART model; read the text on a page
/// var trainModel = new Florence2&lt;double&gt;(architecture, new Florence2Options());
/// var answer = trainModel.RunTask(pageImage, Florence2Task.Ocr);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Detection)]
[ModelTask(ModelTask.Segmentation)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks",
    "https://arxiv.org/abs/2311.06242",
    Year = 2024,
    Authors = "Xiao et al."
)]
[PaperOptimizer(OptimizerKind.AdamW, MinLearningRate = 0,
                Schedule = LearningRateSchedulerType.CosineAnnealing,
                Source = "Xiao et al. 2023, Sec. 4.1: AdamW with cosine learning rate decay. The paper "
                        + "states neither a peak rate nor a batch size in its training description, so "
                        + "neither is declared.")]
public partial class Florence2<T> : VisionLanguageModelBase<T>, IVisualEncoder<T>
{
    private readonly Florence2Options _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <summary>BART's begin-of-sequence token.</summary>
    private const int BosTokenId = 0;

    /// <summary>BART's end-of-sequence token, also the decoder start token.</summary>
    private const int EosTokenId = 2;

    /// <summary>First id available to ordinary text (ids 0, 1 and 2 are BOS, PAD and EOS).</summary>
    private const int FirstTextTokenId = 3;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;
    private DaViTLayer<T>? _vision;
    private Florence2ProjectorLayer<T>? _projector;
    private BartEncoderDecoderLayer<T>? _language;

    /// <summary>Creates Florence-2 backed by an ONNX export.</summary>
    public Florence2(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        Florence2Options? options = null,
        ITokenizer? tokenizer = null
    )
        : base(architecture)
    {
        _options = options ?? new Florence2Options();
        _options.Validate();
        _useNativeMode = false;
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.EmbeddingDim;
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        _tokenizer = tokenizer ?? LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.OPT);
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a trainable Florence-2. With no options it builds Florence-2-base. The default tokenizer is the
    /// repository's GPT-2-style BPE, the family BART's vocabulary belongs to. The location tokens occupy the last
    /// <see cref="Florence2Options.NumLocationBins"/> ids.
    /// </summary>
    public Florence2(
        NeuralNetworkArchitecture<T> architecture,
        Florence2Options? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ITokenizer? tokenizer = null
    )
        : base(architecture)
    {
        _options = options ?? new Florence2Options();
        if (architecture.InputType == InputType.ThreeDimensional && architecture.InputHeight > 0
            && architecture.InputHeight == architecture.InputWidth)
            _options.ImageSize = architecture.InputHeight;
        _options.Validate();
        _useNativeMode = true;
        // Florence-2 fine-tunes AdamW with cosine decay; the paper gives no peak rate. 1e-5 is the stable end
        // of ViT/transformer fine-tuning, and higher rates overshoot this deep seq2seq stack early on.
        _optimizer = optimizer ?? PaperOptimizerFactory.VerifyHandBuilt(this,
            new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(
                this,
                new Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
                {
                    InitialLearningRate = 1e-5,
                }));
        base.ImageSize = _options.ImageSize;
        base.ImageChannels = 3;
        base.EmbeddingDim = _options.EmbeddingDim;
        _tokenizer = tokenizer ?? LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.OPT);
        InitializeLayers();
    }

    /// <inheritdoc/>
    public int EmbeddingDimension => _options.EmbeddingDim;
    int IVisualEncoder<T>.ImageSize => _options.ImageSize;
    int IVisualEncoder<T>.ImageChannels => 3;

    /// <summary>Number of image tokens spliced into the encoder: the mean token plus one per stride-32 position.</summary>
    public int ImageTokenCount => 1 + ((_options.ImageSize / 32) * (_options.ImageSize / 32));

    private int FirstLocationTokenId => _options.VocabSize - _options.NumLocationBins;

    private DaViTLayer<T> Vision => _vision ?? throw NotNative();
    private Florence2ProjectorLayer<T> Projector => _projector ?? throw NotNative();
    private BartEncoderDecoderLayer<T> Language => _language ?? throw NotNative();

    private NotSupportedException NotNative() => new(_useNativeMode
        ? "This Florence-2 was built from custom layers, so it has no DaViT/BART graph to prompt."
        : "Florence-2 is in ONNX mode; the native model is not built.");

    /// <summary>
    /// Encodes an image into its projected image tokens <c>[1 + h*w, EmbeddingDim]</c>: the vision half of the
    /// model, before the BART encoder.
    /// </summary>
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
    /// Teacher-forced decoder logits <c>[T, vocab]</c> for <paramref name="decoderInput"/> under
    /// <paramref name="task"/>'s prompt. Row t predicts the token after <c>decoderInput[t]</c>.
    /// </summary>
    public Tensor<T> PredictTokens(Tensor<T> image, IReadOnlyList<int> decoderInput, Florence2Task? task = null, string? taskInput = null)
    {
        ThrowIfDisposed();
        if (decoderInput is null) throw new ArgumentNullException(nameof(decoderInput));
        SetTrainingMode(false);
        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var memory = EncodeWithPrompt(PreprocessImage(image), task ?? _options.DefaultTask, taskInput);
        return Language.Decode(decoderInput, memory);
    }

    /// <summary>
    /// Greedily generates the answer to <paramref name="task"/> and returns its token ids, ending with EOS when
    /// one was produced within <see cref="Florence2Options.MaxOutputTokens"/>.
    /// </summary>
    public Tensor<T> GenerateFromImage(Tensor<T> image, Florence2Task? task = null, string? taskInput = null)
    {
        ThrowIfDisposed();
        var p = PreprocessImage(image);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(p);
        SetTrainingMode(false);
        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var generated = Generate(EncodeWithPrompt(p, task ?? _options.DefaultTask, taskInput));
        var result = new Tensor<T>(new[] { generated.Count });
        for (int i = 0; i < generated.Count; i++) result[i] = NumOps.FromDouble(generated[i]);
        return result;
    }

    /// <summary>
    /// Runs <paramref name="task"/> and parses the answer. Text tokens become <see cref="Florence2TaskResult.Text"/>.
    /// Each label followed by four location tokens becomes a region in pixel coordinates.
    /// </summary>
    /// <remarks>
    /// Location bins dequantise as the reference does: <c>(bin + 0.5) / NumLocationBins * size</c>.
    /// </remarks>
    public Florence2TaskResult RunTask(Tensor<T> image, Florence2Task task, string? taskInput = null)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));
        var generated = GenerateFromImage(image, task, taskInput);
        var ids = new int[generated.Length];
        for (int i = 0; i < ids.Length; i++) ids[i] = (int)Math.Round(NumOps.ToDouble(generated[i]));
        return ParseAnswer(ids, image.Shape[image.Rank - 1], image.Shape[image.Rank - 2]);
    }

    /// <summary>
    /// Splits generated ids into decoded text and regions. Each label is followed by four location tokens
    /// (x0, y0, x1, y1), which are dequantised to pixels of a <paramref name="width"/> x <paramref name="height"/> page.
    /// </summary>
    internal Florence2TaskResult ParseAnswer(IReadOnlyList<int> ids, int width, int height)
    {
        var text = new List<int>();
        var label = new List<int>();
        var corners = new List<double>();
        var regions = new List<Florence2Region>();
        int bins = _options.NumLocationBins;
        for (int i = 0; i < ids.Count; i++)
        {
            int id = ids[i];
            if (id == EosTokenId) break;
            if (id < FirstTextTokenId) continue;
            if (id >= FirstLocationTokenId)
            {
                corners.Add((id - FirstLocationTokenId + 0.5) / bins);
                if (corners.Count < 4) continue;
                regions.Add(new Florence2Region(
                    label.Count > 0 ? _tokenizer.Decode(label, skipSpecialTokens: true).Trim() : string.Empty,
                    corners[0] * width, corners[1] * height, corners[2] * width, corners[3] * height));
                corners.Clear();
                label.Clear();
            }
            else
            {
                corners.Clear();
                text.Add(id);
                label.Add(id);
            }
        }
        return new Florence2TaskResult(text.Count > 0 ? _tokenizer.Decode(text, skipSpecialTokens: true).Trim() : string.Empty, regions);
    }

    /// <summary>
    /// One teacher-forced step that teaches the model to answer <paramref name="task"/> with
    /// <paramref name="targetIds"/>. The target should end with EOS (id 2).
    /// </summary>
    public void TrainCaption(Tensor<T> image, IReadOnlyList<int> targetIds, Florence2Task? task = null, string? taskInput = null)
    {
        if (targetIds is null) throw new ArgumentNullException(nameof(targetIds));
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        var target = new Tensor<T>(new[] { targetIds.Count });
        for (int i = 0; i < targetIds.Count; i++) target[i] = NumOps.FromDouble(targetIds[i]);
        var resolvedTask = task ?? _options.DefaultTask;
        TrainWithCustomObjective(PreprocessImage(image), target,
            (pixels, labels) => Loss(pixels, labels, resolvedTask, taskInput), _optimizer);
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
        var vision = new DaViTLayer<T>(_options.ImageSize, _options.VisionBaseDim, _options.VisionBaseHeads,
            _options.VisionThirdStageDepth, _options.WindowSize, _options.FfnMultiplier);
        var projector = new Florence2ProjectorLayer<T>(vision.OutputChannels, _options.EmbeddingDim, vision.OutputGrid);
        var language = new BartEncoderDecoderLayer<T>(_options.VocabSize, _options.TextDim, _options.NumHeads,
            _options.TextFeedForwardDim, _options.NumLayers, _options.NumDecoderLayers, _options.MaxTextPositions,
            _options.DropoutRate);
        Layers.Add(LayerGraphContract.FromExternalInput(vision));
        Layers.Add(projector);
        // The image tokens enter BART spliced ahead of the prompt's embeddings, not as its sequential input.
        Layers.Add(LayerGraphContract.FromDerivedInput(language, "text"));
        _vision = vision;
        _projector = projector;
        _language = language;
    }

    private Tensor<T> ImageTokens(Tensor<T> preprocessed)
    {
        var image = preprocessed.Rank == 4 && preprocessed.Shape[0] == 1
            ? Engine.Reshape(preprocessed, new[] { preprocessed.Shape[1], preprocessed.Shape[2], preprocessed.Shape[3] })
            : preprocessed;
        return Projector.Forward(Vision.Forward(image));
    }

    /// <summary>The prompt text the processor substitutes for <paramref name="task"/>.</summary>
    internal static string PromptFor(Florence2Task task, string? taskInput)
    {
        string Require(string template)
        {
            if (string.IsNullOrWhiteSpace(taskInput))
                throw new ArgumentException($"The {task} task needs an input.", nameof(taskInput));
            return template.Replace("{input}", taskInput);
        }

        return task switch
        {
            Florence2Task.Ocr => "What is the text in the image?",
            Florence2Task.OcrWithRegion => "What is the text in the image, with regions?",
            Florence2Task.Caption => "What does the image describe?",
            Florence2Task.DetailedCaption => "Describe in detail what is shown in the image.",
            Florence2Task.MoreDetailedCaption => "Describe with a paragraph what is shown in the image.",
            Florence2Task.ObjectDetection => "Locate the objects with category name in the image.",
            Florence2Task.DenseRegionCaption => "Locate the objects in the image, with their descriptions.",
            Florence2Task.RegionProposal => "Locate the region proposals in the image.",
            Florence2Task.CaptionToPhraseGrounding => Require("Locate the phrases in the caption: {input}"),
            Florence2Task.OpenVocabularyDetection => Require("Locate {input} in the image."),
            Florence2Task.RegionToCategory => Require("What is the region {input}?"),
            Florence2Task.RegionToDescription => Require("What does the region {input} describe?"),
            Florence2Task.RegionToOcr => Require("What text is in the region {input}?"),
            _ => throw new ArgumentOutOfRangeException(nameof(task), task, "Unknown Florence-2 task.")
        };
    }

    /// <summary>
    /// The BART encoder output for an image under a task prompt. The encoder input is the image tokens, then
    /// <c>&lt;s&gt; prompt &lt;/s&gt;</c>, in the reference's order.
    /// </summary>
    private Tensor<T> EncodeWithPrompt(Tensor<T> preprocessed, Florence2Task task, string? taskInput)
    {
        var language = Language;
        var promptIds = new List<int> { BosTokenId };
        promptIds.AddRange(_tokenizer.Encode(PromptFor(task, taskInput)).TokenIds.Select(ClampText));
        promptIds.Add(EosTokenId);
        var embeddings = Engine.TensorConcatenate(new[] { ImageTokens(preprocessed), language.Embed(promptIds) }, 0);
        if (embeddings.Shape[0] > language.MaxPositions)
        {
            int keep = language.MaxPositions;
            embeddings = Engine.TensorSlice(embeddings, new[] { 0, 0 }, new[] { keep, embeddings.Shape[1] });
        }
        return language.Encode(embeddings);
    }

    /// <summary>Keeps ordinary text out of the special ids and the location range.</summary>
    private int ClampText(int id) => Math.Min(Math.Max(id, FirstTextTokenId), FirstLocationTokenId - 1);

    private List<int> Generate(Tensor<T> memory)
    {
        var language = Language;
        var ids = new List<int> { BartEncoderDecoderLayer<T>.DecoderStartTokenId };
        var generated = new List<int>();
        int limit = Math.Min(_options.MaxOutputTokens, language.MaxPositions - 1);
        for (int step = 0; step < limit; step++)
        {
            var logits = language.Decode(ids, memory);
            int last = logits.Shape[0] - 1, best = 0;
            double bestValue = double.NegativeInfinity;
            for (int v = 0; v < _options.VocabSize; v++)
            {
                double value = NumOps.ToDouble(logits[last, v]);
                if (value > bestValue) { bestValue = value; best = v; }
            }
            generated.Add(best);
            if (best == EosTokenId && step > 0) break;
            ids.Add(best);
        }
        return generated;
    }

    /// <summary>
    /// Cross-entropy of the decoder against <paramref name="target"/>. A <c>[1, vocab]</c> target is a
    /// distribution over the first token. Anything else is target ids, trained with teacher forcing: the decoder
    /// sees the start token followed by the target shifted right.
    /// </summary>
    private Tensor<T> Loss(Tensor<T> preprocessed, Tensor<T> target, Florence2Task task, string? taskInput)
    {
        var language = Language;
        var memory = EncodeWithPrompt(preprocessed, task, taskInput);
        if (target.Rank == 2 && target.Shape[0] == 1 && target.Shape[1] == _options.VocabSize)
        {
            var first = Engine.TensorLogSoftmax(language.Decode(new[] { BartEncoderDecoderLayer<T>.DecoderStartTokenId }, memory), axis: 1);
            return Engine.TensorMultiplyScalar(Engine.ReduceSum(Engine.TensorMultiply(target, first), null), NumOps.FromDouble(-1.0));
        }

        var labels = new int[Math.Min(target.Length, language.MaxPositions)];
        for (int i = 0; i < labels.Length; i++) labels[i] = language.ClampToken((int)Math.Round(NumOps.ToDouble(target.Data.Span[i])));
        if (labels.Length == 0) throw new ArgumentException("A Florence-2 target needs at least one token.", nameof(target));
        var decoderInput = new int[labels.Length];
        decoderInput[0] = BartEncoderDecoderLayer<T>.DecoderStartTokenId;
        for (int t = 1; t < labels.Length; t++) decoderInput[t] = labels[t - 1];
        var log = Engine.TensorLogSoftmax(language.Decode(decoderInput, memory), axis: 1);
        var entries = new int[labels.Length];
        for (int t = 0; t < labels.Length; t++) entries[t] = (t * _options.VocabSize) + labels[t];
        var picked = AiDotNet.ComputerVision.CvTensorOps<T>.Select(Engine.Reshape(log, new[] { log.Length }), entries, 0);
        return Engine.TensorMultiplyScalar(Engine.ReduceSum(picked, null), NumOps.FromDouble(-1.0 / labels.Length));
    }

    /// <summary>
    /// Next-token logits <c>[1, vocab]</c> for the decoder start token, given the image and the default task's
    /// prompt.
    /// </summary>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);
        if (_language is null)
        {
            var c = input;
            foreach (var l in Layers)
                c = l.Forward(c);
            return c;
        }
        SetTrainingMode(false);
        var memory = EncodeWithPrompt(PreprocessImage(input), _options.DefaultTask, null);
        return _language.Decode(new[] { BartEncoderDecoderLayer<T>.DecoderStartTokenId }, memory);
    }

    /// <summary>
    /// One training step under the default task. A <c>[1, vocab]</c> target is a distribution over the first
    /// token; any other target is token ids, trained with teacher forcing.
    /// </summary>
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode)
            throw new NotSupportedException("Training is not supported in ONNX mode.");
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (expected is null) throw new ArgumentNullException(nameof(expected));
        if (_language is null)
        {
            SetTrainingMode(true);
            TrainWithTape(input, expected, _optimizer);
            SetTrainingMode(false);
            return;
        }
        var task = _options.DefaultTask;
        TrainWithCustomObjective(PreprocessImage(input), expected, (pixels, target) => Loss(pixels, target, task, null), _optimizer);
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
            Name = _useNativeMode ? "Florence-2-Native" : "Florence-2-ONNX",
            Description = "Florence-2: Unified Vision Foundation Model (Xiao et al., 2024)",
            FeatureCount = _options.EmbeddingDim,
            Complexity = _options.NumLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "Florence-2 (DaViT + BART)";
        m.AdditionalInfo["ModelSize"] = _options.ModelSize.ToString();
        m.AdditionalInfo["VisionBaseDim"] = _options.VisionBaseDim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        m.AdditionalInfo["TextDim"] = _options.TextDim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return m;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(Florence2<T>));
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

/// <summary>One region a Florence-2 task answer located: its label and its pixel box.</summary>
/// <param name="Label">The text generated before the region's four location tokens.</param>
/// <param name="X0">Left edge in pixels.</param>
/// <param name="Y0">Top edge in pixels.</param>
/// <param name="X1">Right edge in pixels.</param>
/// <param name="Y1">Bottom edge in pixels.</param>
public sealed record Florence2Region(string Label, double X0, double Y0, double X1, double Y1);

/// <summary>A parsed Florence-2 task answer: its text, plus any regions written as location tokens.</summary>
/// <param name="Text">All non-location text tokens, decoded.</param>
/// <param name="Regions">Each label followed by four location tokens.</param>
public sealed record Florence2TaskResult(string Text, IReadOnlyList<Florence2Region> Regions);
