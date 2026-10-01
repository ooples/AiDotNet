using AiDotNet.LearningRateSchedulers;
using AiDotNet.Attributes;
using AiDotNet.Document.Interfaces;
using AiDotNet.Document.Options;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;
using Microsoft.ML.OnnxRuntime;
using AiDotNet.Validation;

namespace AiDotNet.Document.VisionLanguage;

/// <summary>
/// UDOP (Unifying Vision, Text, and Layout for Universal Document Processing) neural network.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// UDOP is a foundation model for document AI that unifies text, image, and layout modalities
/// within a single encoder-decoder framework. It can perform multiple document tasks through
/// task-specific prompting.
/// </para>
/// <para>
/// <b>For Beginners:</b> UDOP can handle many document tasks with one model:
/// 1. Document classification
/// 2. Information extraction (NER, key-value pairs)
/// 3. Document question answering
/// 4. Document layout analysis
/// 5. Document generation
///
/// Example usage:
/// <code>
/// var model = new UDOP&lt;float&gt;(architecture);
/// var result = model.AnswerQuestion(documentImage, "What is the invoice total?");
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "Unifying Vision, Text, and Layout for Universal Document Processing" (CVPR 2023)
/// https://arxiv.org/abs/2212.02623
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelDomain(ModelDomain.Multimodal)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.FoundationModel)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Detection)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.VeryHigh)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Unifying Vision, Text, and Layout for Universal Document Processing", "https://arxiv.org/abs/2212.02623", Year = 2023, Authors = "Zineng Tang, Ziyi Yang, Guoxin Wang, Yuwei Fang, Yang Liu, Chenguang Zhu, Michael Zeng, Cha Zhang, Mohit Bansal")]
[PaperOptimizer(OptimizerKind.Adam, Beta1 = 0.9, Beta2 = 0.98, LearningRate = 5e-5,
                WeightDecay = 1e-2, WarmupSteps = 1000, ReferenceBatchSize = 16,
                Source = "Tang et al. 2023, Sec. 5: Adam with learning rate 5e-5, 1000 warmup steps, batch size 16, weight decay 1e-2, beta1 0.9 and beta2 0.98 for the DUE-Benchmark finetuning experiments. FUNSD and CORD use 3e-4 and RVL-CDIP 1e-3, which are different datasets and not what this declares. Built by the model rather than by the factory because it constructs explicit options; the declaration verifies those values instead of replacing them.")]
public partial class UDOP<T> : DocumentNeuralNetworkBase<T>, ILayoutDetector<T>, IDocumentQA<T>, IDocumentClassifier<T>
{
    private readonly UDOPOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #region Fields

    /// <summary>T5 decoder start token, which is also the pad token.</summary>
    private const int DecoderStartTokenId = 0;

    /// <summary>T5 end-of-sequence token.</summary>
    private const int EosTokenId = 1;

    private const string QuestionAnsweringPrompt = "Question answering. ";
    private const string ClassificationPrompt = "Document Classification on RVLCDIP.";
    private const string LayoutAnalysisPrompt = "Layout Analysis.";

    private readonly bool _useNativeMode;
    private readonly InferenceSession? _onnxSession;
    private readonly ITokenizer _tokenizer;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly int _hiddenDim;
    private readonly int _numEncoderLayers;
    private readonly int _numDecoderLayers;
    private readonly int _numHeads;
    private readonly int _vocabSize;

    private UdopTransformerLayer<T>? _transformer;

    #endregion

    #region Properties

    /// <inheritdoc/>
    public override DocumentType SupportedDocumentTypes => DocumentType.All;

    /// <inheritdoc/>
    public override bool RequiresOCR => true;

    /// <summary>Side of the square page image the patch embedding expects.</summary>
    public int ExpectedImageSize => ImageSize;

    /// <inheritdoc/>
    protected override LayerInputDomain ResolveDocumentInputDomain(int[]? inputShape) => inputShape switch
    {
        [_, 5] => LayerInputDomain.Indices(Math.Max(_vocabSize, UdopTransformerLayer<T>.CoordinateGrid + 1)),
        { Length: < 3 } => LayerInputDomain.Indices(_vocabSize),
        _ => LayerInputDomain.Continuous
    };

    /// <inheritdoc/>
    public IReadOnlyList<LayoutElementType> SupportedElementTypes { get; } =
    [
        LayoutElementType.Text,
        LayoutElementType.Title,
        LayoutElementType.List,
        LayoutElementType.Table,
        LayoutElementType.Figure,
        LayoutElementType.Caption,
        LayoutElementType.Header,
        LayoutElementType.Footer,
        LayoutElementType.FormField,
        LayoutElementType.Equation
    ];

    /// <inheritdoc/>
    public IReadOnlyList<string> AvailableCategories { get; } =
    [
        "letter", "form", "email", "handwritten", "advertisement",
        "scientific", "specification", "file_folder", "news_article",
        "budget", "invoice", "presentation", "questionnaire", "resume", "memo"
    ];

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a UDOP model that runs a pretrained ONNX export.
    /// </summary>
    public UDOP(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        ITokenizer tokenizer,
        UDOPOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture: architecture, lossFunction ?? new CrossEntropyWithLogitsLoss<T>(), 1.0)
    {
        _options = options ?? new UDOPOptions();
        Options = _options;

        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentNullException(nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"ONNX model not found: {onnxModelPath}", onnxModelPath);

        Guard.NotNull(tokenizer);
        _tokenizer = tokenizer;
        _useNativeMode = false;
        _options.Validate();

        _hiddenDim = _options.HiddenDim;
        _numEncoderLayers = _options.NumEncoderLayers;
        _numDecoderLayers = _options.NumDecoderLayers;
        _numHeads = _options.NumHeads;
        _vocabSize = _options.VocabSize;
        _optimizer = PaperOptimizerFactory.VerifyHandBuilt(this,
            optimizer ?? CreatePaperOptimizer());

        ImageSize = _options.ImageSize;
        MaxSequenceLength = _options.MaxSequenceLength;

        _onnxSession = new InferenceSession(onnxModelPath);

        InitializeLayers();
    }

    /// <summary>
    /// Creates a trainable UDOP model built from <paramref name="options"/>. With no options, it builds UDOP-large.
    /// </summary>
    public UDOP(
        NeuralNetworkArchitecture<T> architecture,
        UDOPOptions? options = null,
        ITokenizer? tokenizer = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture: architecture, lossFunction ?? new CrossEntropyWithLogitsLoss<T>(), 1.0)
    {
        _options = options ?? new UDOPOptions();
        _options.Validate();
        Options = _options;

        _useNativeMode = true;
        _hiddenDim = _options.HiddenDim;
        _numEncoderLayers = _options.NumEncoderLayers;
        _numDecoderLayers = _options.NumDecoderLayers;
        _numHeads = _options.NumHeads;
        _vocabSize = _options.VocabSize;
        _optimizer = optimizer ?? CreatePaperOptimizer();

        ImageSize = _options.ImageSize;
        MaxSequenceLength = _options.MaxSequenceLength;

        // UDOP's tokenizer is T5's SentencePiece vocabulary plus layout tokens.
        _tokenizer = tokenizer ?? LanguageModelTokenizerFactory.CreateForBackbone(LanguageModelBackbone.FlanT5);

        InitializeLayers();
    }

    private AdamWOptimizer<T, Tensor<T>, Tensor<T>> CreatePaperOptimizer() =>
        new(this, new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = _options.LearningRate,
            Beta1 = 0.9,
            Beta2 = 0.98,
            WeightDecay = 0.01
        });

    #endregion

    #region Initialization

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
        {
            return;
        }

        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
            return;
        }

        var transformer = new UdopTransformerLayer<T>(
            _vocabSize, _hiddenDim, _numHeads, _options.KeyValueDim, _options.FeedForwardDim,
            _numEncoderLayers, _numDecoderLayers, ImageSize, _options.PatchSize,
            _options.RelativeAttentionBuckets, _options.RelativeAttentionMaxDistance,
            _options.RelativeAttentionMaxDistance2D, _options.Max2DPositions, _options.LayerNormEpsilon);
        Layers.Add(LayerGraphContract.FromExternalInput(transformer));
        _transformer = transformer;
    }

    private UdopTransformerLayer<T> Transformer => _transformer
        ?? throw new NotSupportedException(_useNativeMode
            ? "This UDOP was built from custom layers, so it has no encoder-decoder to prompt."
            : "Prompted decoding needs the native model. The ONNX session exposes a single forward pass.");

    #endregion

    #region Encoder-decoder

    /// <summary>
    /// Builds the encoder input from a task prompt, the OCR tokens and the page, then encodes it.
    /// </summary>
    /// <remarks>
    /// As in the reference, the prompt goes first with all-zero boxes. Zero boxes mark the prompt as a target
    /// segment, so its tokens do not take an image patch.
    /// </remarks>
    private Tensor<T> EncodeInputs(IReadOnlyList<int> prompt, Tensor<T>? packedTokens, Tensor<T>? preprocessedImage)
    {
        var transformer = Transformer;
        var tokens = new List<int>(prompt.Select(transformer.ClampToken));
        var boxes = new List<double[]>(prompt.Select(_ => new double[4]));
        if (packedTokens is not null)
        {
            if (packedTokens.Rank != 2 || packedTokens.Shape[1] != 5)
                throw new ArgumentException("UDOP expects packed rows [S, 5] (token, x0, y0, x1, y1) on the 0-1000 grid.", nameof(packedTokens));
            var (ocrTokens, ocrBoxes) = transformer.Unpack(packedTokens);
            tokens.AddRange(ocrTokens);
            boxes.AddRange(ocrBoxes);
        }
        if (tokens.Count > MaxSequenceLength)
            throw new ArgumentException($"UDOP takes at most {MaxSequenceLength} prompt and OCR tokens; got {tokens.Count}.", nameof(packedTokens));
        Tensor<T>? image = null;
        if (preprocessedImage is not null)
        {
            image = EnsureBatchDimension(preprocessedImage);
            if (image.Shape[0] != 1)
                throw new ArgumentException($"UDOP encodes one page at a time; got a batch of {image.Shape[0]}.", nameof(preprocessedImage));
        }
        return transformer.Encode(tokens.ToArray(), boxes.ToArray(), image);
    }

    /// <summary>Splits a model input into its packed tokens or its page: packed <c>[S, 5]</c>, ids <c>[S]</c>, or an image.</summary>
    private (Tensor<T>? Packed, Tensor<T>? Image) RouteInput(Tensor<T> input)
    {
        if (input.Rank >= 3) return (null, input);
        if (input.Rank == 2 && input.Shape[1] == 5) return (input, null);
        if (input.Rank == 1 || (input.Rank == 2 && input.Shape[1] == 1))
        {
            int s = input.Shape[0];
            var packed = new Tensor<T>(new[] { s, 5 });
            for (int i = 0; i < s; i++) packed[i, 0] = input.Rank == 1 ? input[i] : input[i, 0];
            return (packed, null);
        }
        throw new ArgumentException(
            $"UDOP expects packed rows [S, 5] (token, x0, y0, x1, y1), token ids [S], or a page image; got shape [{string.Join(", ", input.Shape.ToArray())}].",
            nameof(input));
    }

    private Tensor<T> EncodeRouted(Tensor<T> input)
    {
        var (packed, image) = RouteInput(input);
        return EncodeInputs(Array.Empty<int>(), packed, image);
    }

    /// <summary>
    /// Returns the decoder's next-token logits <c>[1, vocab]</c> for the start token, given OCR tokens and a page.
    /// </summary>
    /// <param name="packedTokens">Packed rows <c>[S, 5]</c>: token id, then x0, y0, x1, y1 on the 0-1000 grid.</param>
    /// <param name="pageImage">The page, <c>[3, H, W]</c> or <c>[1, 3, H, W]</c>, at <see cref="ExpectedImageSize"/>.</param>
    public Tensor<T> PredictDocument(Tensor<T> packedTokens, Tensor<T> pageImage) =>
        DecoderLogits(packedTokens, pageImage, new[] { DecoderStartTokenId });

    /// <summary>
    /// Teacher-forced decoder logits <c>[T, vocab]</c>: row <c>t</c> predicts the token after
    /// <c>decoderInput[t]</c>. Either input may be null, but not both.
    /// </summary>
    public Tensor<T> DecoderLogits(Tensor<T>? packedTokens, Tensor<T>? pageImage, IReadOnlyList<int> decoderInput)
    {
        if (decoderInput is null) throw new ArgumentNullException(nameof(decoderInput));
        if (packedTokens is null && pageImage is null)
            throw new ArgumentException("Pass OCR tokens, a page image, or both.", nameof(packedTokens));
        SetTrainingMode(false);
        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var memory = EncodeInputs(Array.Empty<int>(), packedTokens, PreparePage(pageImage));
        return Transformer.Decode(decoderInput.ToArray(), memory);
    }

    /// <summary>
    /// Greedily decodes from the start token until EOS or <see cref="UDOPOptions.MaxGenerationLength"/>. The
    /// returned ids include the EOS token when one was produced.
    /// </summary>
    /// <param name="packedTokens">Optional OCR tokens, packed <c>[S, 5]</c>.</param>
    /// <param name="pageImage">Optional page image.</param>
    /// <param name="promptTokens">Optional task-prompt token ids, placed in front of the OCR tokens.</param>
    public Tensor<T> GenerateTokens(Tensor<T>? packedTokens, Tensor<T>? pageImage, IReadOnlyList<int>? promptTokens = null)
    {
        if (packedTokens is null && pageImage is null && (promptTokens is null || promptTokens.Count == 0))
            throw new ArgumentException("Pass OCR tokens, a page image, or a prompt.", nameof(packedTokens));
        SetTrainingMode(false);
        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var memory = EncodeInputs(promptTokens ?? Array.Empty<int>(), packedTokens, PreparePage(pageImage));
        var generated = Generate(memory, _options.MaxGenerationLength, 0.0);
        var result = new Tensor<T>(new[] { generated.Count });
        for (int i = 0; i < generated.Count; i++) result[i] = NumOps.FromDouble(generated[i].Token);
        return result;
    }

    /// <summary>
    /// One teacher-forced training step on target token ids, which should end with EOS (id 1). The model learns to
    /// generate <paramref name="targetTokens"/> from the OCR tokens and page.
    /// </summary>
    public void TrainDocument(Tensor<T>? packedTokens, Tensor<T>? pageImage, IReadOnlyList<int> targetTokens)
    {
        if (targetTokens is null) throw new ArgumentNullException(nameof(targetTokens));
        if (!_useNativeMode) throw new NotSupportedException("Training not supported in ONNX mode.");
        if (packedTokens is null && pageImage is null)
            throw new ArgumentException("Pass OCR tokens, a page image, or both.", nameof(packedTokens));
        var target = new Tensor<T>(new[] { targetTokens.Count });
        for (int i = 0; i < targetTokens.Count; i++) target[i] = NumOps.FromDouble(targetTokens[i]);
        var page = PreparePage(pageImage);
        if (packedTokens is not null)
        {
            TrainWithCustomObjective(packedTokens, target,
                (tokens, labels) => SequenceLoss(EncodeInputs(Array.Empty<int>(), tokens, page), labels), _optimizer);
            return;
        }
        var onlyPage = page ?? throw new ArgumentNullException(nameof(pageImage));
        TrainWithCustomObjective(onlyPage, target,
            (image, labels) => SequenceLoss(EncodeInputs(Array.Empty<int>(), null, image), labels), _optimizer);
    }

    private Tensor<T>? PreparePage(Tensor<T>? pageImage)
    {
        if (pageImage is null) return null;
        ValidateImageShape(pageImage);
        return PreprocessDocument(pageImage);
    }

    /// <summary>
    /// Cross-entropy of the decoder against <paramref name="target"/>. A <c>[1, vocab]</c> target is a
    /// distribution over the first generated token. Anything else is read as target token ids, trained with
    /// teacher forcing: the decoder sees the start token followed by the target shifted right.
    /// </summary>
    private Tensor<T> SequenceLoss(Tensor<T> memory, Tensor<T> target)
    {
        var transformer = Transformer;
        if (target.Rank == 2 && target.Shape[0] == 1 && target.Shape[1] == _vocabSize)
        {
            var logProbabilities = Engine.TensorLogSoftmax(transformer.Decode(new[] { DecoderStartTokenId }, memory), axis: 1);
            return Engine.TensorMultiplyScalar(Engine.ReduceSum(Engine.TensorMultiply(target, logProbabilities), null), NumOps.FromDouble(-1.0));
        }

        var labels = new int[target.Length];
        for (int i = 0; i < labels.Length; i++) labels[i] = transformer.ClampToken((int)Math.Round(NumOps.ToDouble(target.Data.Span[i])));
        if (labels.Length == 0) throw new ArgumentException("A UDOP target needs at least one token.", nameof(target));
        var decoderInput = new int[labels.Length];
        decoderInput[0] = DecoderStartTokenId;
        for (int t = 1; t < labels.Length; t++) decoderInput[t] = labels[t - 1];
        var log = Engine.TensorLogSoftmax(transformer.Decode(decoderInput, memory), axis: 1);
        var entries = new int[labels.Length];
        for (int t = 0; t < labels.Length; t++) entries[t] = (t * _vocabSize) + labels[t];
        var picked = AiDotNet.ComputerVision.CvTensorOps<T>.Select(Engine.Reshape(log, new[] { log.Length }), entries, 0);
        return Engine.TensorMultiplyScalar(Engine.ReduceSum(picked, null), NumOps.FromDouble(-1.0 / labels.Length));
    }

    /// <summary>
    /// Decodes from the start token: greedy when <paramref name="temperature"/> is 0, sampled otherwise. Stops
    /// after EOS, which is included in the result.
    /// </summary>
    private List<(int Token, double Probability)> Generate(Tensor<T> memory, int maxLength, double temperature)
    {
        var transformer = Transformer;
        var ids = new List<int> { DecoderStartTokenId };
        var generated = new List<(int Token, double Probability)>();
        var random = RandomHelper.Shared;
        for (int step = 0; step < maxLength; step++)
        {
            var logits = transformer.Decode(ids.ToArray(), memory);
            int last = logits.Shape[0] - 1;
            var row = new double[_vocabSize];
            double max = double.NegativeInfinity;
            for (int v = 0; v < _vocabSize; v++)
            {
                row[v] = NumOps.ToDouble(logits[last, v]) / (temperature > 0 ? temperature : 1.0);
                max = Math.Max(max, row[v]);
            }
            double total = 0;
            for (int v = 0; v < _vocabSize; v++) { row[v] = Math.Exp(row[v] - max); total += row[v]; }
            int pick = 0;
            if (temperature > 0)
            {
                double u = random.NextDouble() * total, running = 0;
                for (pick = 0; pick < _vocabSize - 1; pick++)
                {
                    running += row[pick];
                    if (running >= u) break;
                }
            }
            else
            {
                for (int v = 1; v < _vocabSize; v++) if (row[v] > row[pick]) pick = v;
            }
            generated.Add((pick, row[pick] / total));
            if (pick == EosTokenId) break;
            ids.Add(pick);
        }
        return generated;
    }

    private int[] EncodeText(string text) =>
        _tokenizer.Encode(text).TokenIds.Where(id => id != DecoderStartTokenId && id != EosTokenId).ToArray();

    /// <summary>Encodes a page under a task prompt. Native mode only.</summary>
    private Tensor<T> EncodePrompted(Tensor<T> documentImage, string prompt)
    {
        ValidateImageShape(documentImage);
        if (!_useNativeMode) throw new NotSupportedException("Prompted decoding needs the native model. The ONNX session exposes a single forward pass.");
        SetTrainingMode(false);
        return EncodeInputs(EncodeText(prompt), null, PreprocessDocument(documentImage));
    }

    #endregion

    #region ILayoutDetector Implementation

    /// <inheritdoc/>
    public DocumentLayoutResult<T> DetectLayout(Tensor<T> documentImage)
    {
        return DetectLayout(documentImage, 0.5);
    }

    /// <summary>
    /// Layout analysis by prompting: the decoder writes a category name followed by four location tokens per
    /// region, as UDOP's layout task does.
    /// </summary>
    /// <remarks>
    /// Location tokens are the last <see cref="UDOPOptions.NumLocationBins"/> ids of the vocabulary. Each one
    /// quantises a 0-1 coordinate. A region's confidence is the mean probability of its location tokens.
    /// </remarks>
    public DocumentLayoutResult<T> DetectLayout(Tensor<T> documentImage, double confidenceThreshold)
    {
        var startTime = DateTime.UtcNow;
        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var memory = EncodePrompted(documentImage, LayoutAnalysisPrompt);
        var generated = Generate(memory, _options.MaxGenerationLength, 0.0);

        int height = documentImage.Shape[documentImage.Rank - 2], width = documentImage.Shape[documentImage.Rank - 1];
        int bins = _options.NumLocationBins, firstLocation = _vocabSize - bins;
        var regions = new List<LayoutRegion<T>>();
        var label = new List<int>();
        var corners = new List<(double Value, double Probability)>();
        foreach (var (token, probability) in generated)
        {
            if (token == EosTokenId) break;
            if (token >= firstLocation)
            {
                corners.Add(((double)(token - firstLocation) / (bins - 1), probability));
                if (corners.Count < 4) continue;
                double confidence = corners.Average(c => c.Probability);
                if (confidence >= confidenceThreshold)
                {
                    regions.Add(new LayoutRegion<T>
                    {
                        ElementType = ParseElementType(label),
                        Confidence = NumOps.FromDouble(confidence),
                        ConfidenceValue = confidence,
                        Index = regions.Count,
                        BoundingBox = new Vector<T>(new[]
                        {
                            NumOps.FromDouble(corners[0].Value * width), NumOps.FromDouble(corners[1].Value * height),
                            NumOps.FromDouble(corners[2].Value * width), NumOps.FromDouble(corners[3].Value * height)
                        })
                    });
                }
                corners.Clear();
                label.Clear();
            }
            else
            {
                if (corners.Count > 0) corners.Clear();
                if (token != DecoderStartTokenId) label.Add(token);
            }
        }

        return new DocumentLayoutResult<T>
        {
            Regions = regions,
            ProcessingTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
        };
    }

    private LayoutElementType ParseElementType(List<int> label)
    {
        string text = label.Count > 0 ? _tokenizer.Decode(label, skipSpecialTokens: true).Trim() : string.Empty;
        foreach (var type in SupportedElementTypes)
            if (text.IndexOf(type.ToString(), StringComparison.OrdinalIgnoreCase) >= 0) return type;
        return LayoutElementType.Text;
    }

    #endregion

    #region IDocumentQA Implementation

    /// <inheritdoc/>
    public DocumentQAResult<T> AnswerQuestion(Tensor<T> documentImage, string question)
    {
        return AnswerQuestion(documentImage, question, 256, 0.0);
    }

    /// <summary>
    /// Answers by prompting the encoder with <c>"Question answering. {question}"</c> and decoding. Decoding is
    /// greedy at temperature 0 and sampled otherwise.
    /// </summary>
    public DocumentQAResult<T> AnswerQuestion(Tensor<T> documentImage, string question, int maxAnswerLength, double temperature = 0.0)
    {
        if (question is null) throw new ArgumentNullException(nameof(question));
        var startTime = DateTime.UtcNow;
        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var memory = EncodePrompted(documentImage, QuestionAnsweringPrompt + question);
        var generated = Generate(memory, Math.Max(0, Math.Min(maxAnswerLength, _options.MaxGenerationLength)), temperature)
            .Where(g => g.Token != EosTokenId && g.Token != DecoderStartTokenId).ToList();
        string answer = generated.Count > 0 ? _tokenizer.Decode(generated.Select(g => g.Token).ToList(), skipSpecialTokens: true).Trim() : string.Empty;
        double confidence = generated.Count > 0 && answer.Length > 0 ? generated.Average(g => g.Probability) : 0.0;

        return new DocumentQAResult<T>
        {
            Answer = answer.Length > 0 ? answer : "[No answer found]",
            Confidence = NumOps.FromDouble(confidence),
            ConfidenceValue = confidence,
            Question = question,
            ProcessingTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
        };
    }

    /// <inheritdoc/>
    public IEnumerable<DocumentQAResult<T>> AnswerQuestions(Tensor<T> documentImage, IEnumerable<string> questions)
    {
        foreach (var q in questions)
            yield return AnswerQuestion(documentImage, q);
    }

    /// <inheritdoc/>
    public Dictionary<string, DocumentQAResult<T>> ExtractFields(Tensor<T> documentImage, IEnumerable<string> fieldPrompts)
    {
        var results = new Dictionary<string, DocumentQAResult<T>>();
        foreach (var field in fieldPrompts)
            results[field] = AnswerQuestion(documentImage, $"What is the {field}?");
        return results;
    }

    #endregion

    #region IDocumentClassifier Implementation

    /// <inheritdoc/>
    public DocumentClassificationResult<T> ClassifyDocument(Tensor<T> documentImage)
    {
        return ClassifyDocument(documentImage, 5);
    }

    /// <summary>
    /// Classifies by prompting with UDOP's RVL-CDIP task prompt and scoring every category name as a decoder target.
    /// </summary>
    /// <remarks>
    /// Each category's score is the total log-probability of its tokens followed by EOS. A softmax over those
    /// scores gives the reported probabilities. UDOP has no classification head: the label is generated text.
    /// </remarks>
    public DocumentClassificationResult<T> ClassifyDocument(Tensor<T> documentImage, int topK)
    {
        var startTime = DateTime.UtcNow;
        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var memory = EncodePrompted(documentImage, ClassificationPrompt);
        var transformer = Transformer;

        var scores = new double[AvailableCategories.Count];
        for (int c = 0; c < scores.Length; c++)
        {
            var labels = EncodeText(AvailableCategories[c].Replace('_', ' ')).Select(transformer.ClampToken).Append(EosTokenId).ToArray();
            var decoderInput = new int[labels.Length];
            decoderInput[0] = DecoderStartTokenId;
            for (int t = 1; t < labels.Length; t++) decoderInput[t] = labels[t - 1];
            var log = Engine.TensorLogSoftmax(transformer.Decode(decoderInput, memory), axis: 1);
            for (int t = 0; t < labels.Length; t++) scores[c] += NumOps.ToDouble(log[t, labels[t]]);
        }
        double best = scores.Max();
        var weights = scores.Select(s => Math.Exp(s - best)).ToArray();
        double total = weights.Sum();
        var topPredictions = AvailableCategories.Select((category, i) => (Category: category, Score: weights[i] / total))
            .OrderByDescending(p => p.Score).Take(Math.Max(1, topK)).ToList();

        return new DocumentClassificationResult<T>
        {
            PredictedCategory = topPredictions[0].Category,
            Confidence = NumOps.FromDouble(topPredictions[0].Score),
            ConfidenceValue = topPredictions[0].Score,
            TopPredictions = topPredictions,
            ProcessingTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds
        };
    }

    #endregion

    #region IDocumentModel Implementation

    /// <summary>Returns the encoder's output states for a page: one row per unclaimed image patch.</summary>
    public Tensor<T> EncodeDocument(Tensor<T> documentImage)
    {
        ValidateImageShape(documentImage);
        var preprocessed = PreprocessDocument(documentImage);
        if (!_useNativeMode) return RunOnnxInference(preprocessed);
        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        SetTrainingMode(false);
        return EncodeInputs(Array.Empty<int>(), null, preprocessed);
    }

    /// <inheritdoc/>
    public void ValidateInputShape(Tensor<T> documentImage)
    {
        ValidateImageShape(documentImage);
    }

    /// <inheritdoc/>
    public string GetModelSummary()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("UDOP Model Summary");
        sb.AppendLine("==================");
        sb.AppendLine($"Mode: {(_useNativeMode ? "Native (Trainable)" : "ONNX (Inference)")}");
        sb.AppendLine("Architecture: T5 encoder-decoder with layout-induced vision-text fusion and 2-D relative biases");
        sb.AppendLine($"Hidden Dimension: {_hiddenDim}");
        sb.AppendLine($"Encoder Layers: {_numEncoderLayers}");
        sb.AppendLine($"Decoder Layers: {_numDecoderLayers}");
        sb.AppendLine($"Attention Heads: {_numHeads} x {_options.KeyValueDim}");
        sb.AppendLine($"Feed-Forward Dimension: {_options.FeedForwardDim}");
        sb.AppendLine($"Vocabulary: {_vocabSize}");
        sb.AppendLine($"Image Size: {ImageSize}x{ImageSize} (patch {_options.PatchSize})");
        sb.AppendLine($"Max Sequence Length: {MaxSequenceLength}");
        sb.AppendLine("Capabilities: Layout, QA, Classification, Generation (all by prompting)");
        sb.AppendLine($"Total Layers: {Layers.Count}");
        return sb.ToString();
    }

    #endregion

    #region Preprocessing

    /// <inheritdoc/>
    protected override Tensor<T> ApplyDefaultPreprocessing(Tensor<T> rawImage)
    {
        var image = EnsureBatchDimension(rawImage);
        int batchSize = image.Shape[0];
        int channels = image.Shape[1];
        int height = image.Shape[2];
        int width = image.Shape[3];

        var normalized = new Tensor<T>(image._shape);
        double[] means = [0.485, 0.456, 0.406];
        double[] stds = [0.229, 0.224, 0.225];

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                double mean = c < means.Length ? means[c] : 0.5;
                double std = c < stds.Length ? stds[c] : 0.5;
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        int idx = b * channels * height * width + c * height * width + h * width + w;
                        normalized.Data.Span[idx] = NumOps.FromDouble((NumOps.ToDouble(image.Data.Span[idx]) - mean) / std);
                    }
                }
            }
        }
        return normalized;
    }

    /// <inheritdoc/>
    protected override Tensor<T> ApplyDefaultPostprocessing(Tensor<T> modelOutput) => modelOutput;

    #endregion

    #region Serialization

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "UDOP",
            Description = "UDOP for unified document processing (CVPR 2023)",
            FeatureCount = _hiddenDim,
            Complexity = _numEncoderLayers + _numDecoderLayers,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "hidden_dim", _hiddenDim },
                { "num_encoder_layers", _numEncoderLayers },
                { "num_decoder_layers", _numDecoderLayers },
                { "num_heads", _numHeads },
                { "key_value_dim", _options.KeyValueDim },
                { "feed_forward_dim", _options.FeedForwardDim },
                { "image_size", ImageSize },
                { "patch_size", _options.PatchSize },
                { "vocab_size", _vocabSize },
                { "use_native_mode", _useNativeMode }
            },
            ModelDataProvider = () => SafeSerialize()
        };
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <summary>
    /// Next-token logits <c>[1, vocab]</c> for the decoder start token. The input is packed OCR rows <c>[S, 5]</c>,
    /// token ids <c>[S]</c>, or a page image.
    /// </summary>
    protected override Tensor<T> Forward(Tensor<T> input)
        => _transformer is not null
            ? _transformer.Decode(new[] { DecoderStartTokenId }, EncodeRouted(input))
            : base.Forward(input);

    /// <inheritdoc/>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        EnsureLayerRandomSeedsWired();
        return Forward(input);
    }

    /// <inheritdoc/>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        if (input is null)
            throw new ArgumentNullException(nameof(input));
        if (_transformer is null)
            return base.GetNamedLayerActivations(input);

        using var _ = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        var memory = EncodeRouted(PreprocessDocument(input));
        return new Dictionary<string, Tensor<T>>
        {
            ["encoder"] = memory,
            ["output"] = _transformer.Decode(new[] { DecoderStartTokenId }, memory)
        };
    }

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        var preprocessed = PreprocessDocument(input);
        return _useNativeMode ? Forward(preprocessed) : RunOnnxInference(preprocessed);
    }

    /// <summary>
    /// One training step. A <c>[1, vocab]</c> target is a distribution over the first generated token. Any other
    /// target is a token-id sequence, trained with teacher forcing.
    /// </summary>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
            throw new NotSupportedException("Training not supported in ONNX mode.");
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (expectedOutput is null) throw new ArgumentNullException(nameof(expectedOutput));

        if (_transformer is null)
        {
            SetTrainingMode(true);
            try { TrainWithTape(PreprocessDocument(input), expectedOutput, _optimizer); }
            finally { SetTrainingMode(false); }
            return;
        }

        // PredictCore evaluates ImageNet-normalised pages, so train on that same representation.
        TrainWithCustomObjective(PreprocessDocument(input), expectedOutput,
            (routed, target) => SequenceLoss(EncodeRouted(routed), target), _optimizer);
    }

    /// <inheritdoc/>
    protected override bool SupportsParameterMutation => _useNativeMode;

    #endregion

    #region Disposal

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
            _onnxSession?.Dispose();
        base.Dispose(disposing);
    }

    #endregion
}
