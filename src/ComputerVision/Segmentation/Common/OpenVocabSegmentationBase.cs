using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Validation;
using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.ComputerVision.Segmentation.Common;

/// <summary>
/// Abstract base class for open-vocabulary segmentation models that segment objects described
/// by arbitrary text without being limited to a fixed class set.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Traditional segmentation models only recognize objects they were trained
/// on (e.g., "car", "person"). Open-vocabulary models use language understanding (typically CLIP)
/// to segment anything you describe in text, even novel concepts.
///
/// Models extending this base class: SAN, CAT-Seg, SED, Open-Vocabulary SAM, Grounded SAM 2, Mask-Adapter.
/// </para>
/// </remarks>
public abstract partial class OpenVocabSegmentationBase<T> : SegmentationModelBase<T>, IOpenVocabSegmentation<T>
{
    // protected and mutable so a derived model can read them and restore them on deserialization -
    // see PanopticSegmentationBase._numStuffClasses for why private/readonly makes a base unadoptable.
    protected int _maxCategories;
    protected int _maxPromptLength;

    /// <inheritdoc/>
    public int MaxCategories => _maxCategories;

    /// <inheritdoc/>
    public int MaxPromptLength => _maxPromptLength;

    /// <summary>
    /// The CLIP-style text encoder that turns class names into the embeddings pixels are scored against.
    /// </summary>
    protected readonly ITextEncoder<T> _textEncoder;

    /// <summary>The inverse temperature <c>1 / tau</c> that scales every cosine similarity.</summary>
    protected readonly T _logitScale;

    /// <summary>Normalized embeddings <c>[NumClasses, D]</c> of the closed-set vocabulary Predict/Train score against.</summary>
    private readonly Tensor<T> _vocabularyEmbeddings;

    /// <summary>The query vocabulary of an in-flight <see cref="SegmentWithText"/> call, or <c>null</c>.</summary>
    private Tensor<T>? _queryEmbeddings;

    /// <summary>
    /// Gets the text encoder whose embeddings the pixel embeddings are compared with.
    /// </summary>
    public ITextEncoder<T> TextEncoder => _textEncoder;

    /// <summary>
    /// Gets the width of the pixel embeddings the decoder must produce: the text encoder's embedding dimension.
    /// </summary>
    protected int TextEmbeddingDimension => _textEncoder.TextEmbeddingDimension;

    /// <summary>
    /// Initializes the base in native (trainable) mode.
    /// </summary>
    protected OpenVocabSegmentationBase(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer,
        ILossFunction<T>? lossFunction,
        ITextEncoder<T> textEncoder,
        TextQueriedSegmentationOptions options,
        int maxCategories = 256,
        int maxPromptLength = 77)
        : base(architecture, optimizer, lossFunction, RequireOptions(options).NumClasses)
    {
        Guard.Positive(maxCategories);
        Guard.Positive(maxPromptLength);
        _maxCategories = maxCategories;
        _maxPromptLength = maxPromptLength;
        _textEncoder = textEncoder ?? throw new ArgumentNullException(nameof(textEncoder));
        (_logitScale, _vocabularyEmbeddings) = PrepareVocabulary(textEncoder, options);
    }

    /// <summary>
    /// Initializes the base in ONNX (inference-only) mode. The ONNX graph must output pixel embeddings
    /// <c>[B, D, H, W]</c> with <c>D</c> equal to the text encoder's embedding dimension.
    /// </summary>
    /// <remarks>
    /// Only the native constructor carries optional parameters (RS0026: two new overloads may not both).
    /// </remarks>
    protected OpenVocabSegmentationBase(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        ITextEncoder<T> textEncoder,
        TextQueriedSegmentationOptions options,
        int maxCategories,
        int maxPromptLength)
        : base(architecture, onnxModelPath, RequireOptions(options).NumClasses)
    {
        Guard.Positive(maxCategories);
        Guard.Positive(maxPromptLength);
        _maxCategories = maxCategories;
        _maxPromptLength = maxPromptLength;
        _textEncoder = textEncoder ?? throw new ArgumentNullException(nameof(textEncoder));
        (_logitScale, _vocabularyEmbeddings) = PrepareVocabulary(textEncoder, options);
    }

    private static TextQueriedSegmentationOptions RequireOptions(TextQueriedSegmentationOptions options)
        => options ?? throw new ArgumentNullException(nameof(options));

    private static (T LogitScale, Tensor<T> Vocabulary) PrepareVocabulary(
        ITextEncoder<T> textEncoder, TextQueriedSegmentationOptions options)
    {
        if (textEncoder is null) throw new ArgumentNullException(nameof(textEncoder));
        options.ValidateTextQuerySettings();
        var numOps = MathHelper.GetNumericOperations<T>();
        return (numOps.FromDouble(1.0 / options.Temperature),
            TextEmbeddingClassifier.EncodeNormalized(textEncoder, options.ResolveClassNames()));
    }

    /// <summary>
    /// Scores pixel embeddings <c>[D, H, W]</c> or <c>[B, D, H, W]</c> against the active vocabulary
    /// (the closed-set class names, or the names of an in-flight <see cref="SegmentWithText"/> call),
    /// returning <c>cos / tau</c> logits with one channel per name.
    /// </summary>
    protected Tensor<T> ClassifyPixels(Tensor<T> pixelEmbeddings)
        => TextEmbeddingClassifier.Classify(pixelEmbeddings, _queryEmbeddings ?? _vocabularyEmbeddings, _logitScale);

    /// <inheritdoc/>
    /// <remarks>
    /// Scores every pixel against the CLIP text embedding of each name and labels each pixel with its
    /// most likely name. The probabilities are the softmax over the names (SAN, Mask-Adapter,
    /// Open-Vocabulary SAM) or each name's own sigmoid (CAT-Seg, SED; see
    /// <c>ScoresNamesIndependently</c>). A name's mask is the pixels it wins; its score is its mean
    /// probability there.
    /// </remarks>
    public virtual OpenVocabSegmentationResult<T> SegmentWithText(Tensor<T> image, IReadOnlyList<string> classNames)
    {
        if (classNames is null) throw new ArgumentNullException(nameof(classNames));
        var text = TextEmbeddingClassifier.EncodeNormalized(_textEncoder, classNames);
        Tensor<T> logits;
        _queryEmbeddings = text;
        try
        {
            logits = SegmentationTensorOps.EnsureUnbatched(Predict(image));
        }
        finally
        {
            _queryEmbeddings = null;
        }

        return BuildResult(Engine, NumOps, logits, classNames, ScoresNamesIndependently);
    }

    /// <summary>
    /// Whether the model scores every name with its own sigmoid rather than a softmax over the names.
    /// </summary>
    /// <remarks>
    /// CAT-Seg and SED train their per-name cost maps with binary cross-entropy and read them through a
    /// sigmoid at inference (<c>outputs.sigmoid()</c> in both official repositories); SAN, Mask-Adapter and
    /// Open-Vocabulary SAM classify with a softmax over the names. The labels are the argmax either way.
    /// </remarks>
    internal virtual bool ScoresNamesIndependently => false;

    /// <summary>
    /// Turns per-name logits <c>[K, H, W]</c> into the argmax segmentation result.
    /// </summary>
    /// <remarks>
    /// Each pixel is labelled with its highest-scoring name, and a name's score is its mean probability over
    /// the pixels it wins: the softmax over the names, or each name's own sigmoid when
    /// <paramref name="independentSigmoid"/> is set. A single name cannot be classified by a softmax (it is
    /// 1 at every pixel), so it is decided per pixel by its sigmoid instead -- in the mask where
    /// <c>cos / tau &gt; 0</c> -- and pixels outside the mask get semantic label -1. Areas are counted in
    /// <c>int</c> and each score is the mean over its own pixels only.
    /// </remarks>
    internal static OpenVocabSegmentationResult<T> BuildResult(
        IEngine engine, INumericOperations<T> numOps, Tensor<T> logits, IReadOnlyList<string> classNames,
        bool independentSigmoid = false)
    {
        int numText = classNames.Count;
        if (numText == 1)
        {
            return BuildSingleNameResult(engine, numOps, logits, classNames[0]);
        }

        int pixels = logits.Shape[1] * logits.Shape[2];
        var probs = independentSigmoid ? engine.TensorSigmoid(logits) : engine.TensorSoftmax(logits, 0);
        var labels = engine.TensorArgMax(logits, 0);
        var masks = engine.TensorPermute(engine.TensorOneHot<T>(labels, numText), new[] { 2, 0, 1 });

        // Every name's pixel count (exact, in int) and each pixel's probability for the name it won.
        var labelIndex = engine.Reshape(labels, new[] { pixels });
        int[] areas = engine.TensorBinCount(labelIndex, numText).ToArray();
        var winningProbability = engine.TensorGatherClassValues(
            engine.TensorPermute(engine.Reshape(probs, new[] { numText, pixels }), new[] { 1, 0 }),
            labelIndex.Cast<T>());
        var scores = new Vector<T>(Enumerable.Range(0, numText)
            .Select(k => MaskStatistics.LabelMean(winningProbability, labelIndex, k, areas[k])));
        return new OpenVocabSegmentationResult<T>
        {
            Masks = masks,
            ClassNames = classNames.ToArray(),
            Scores = scores,
            SemanticMap = labels.Cast<T>()
        };
    }

    /// <summary>
    /// The result for one name from its logits <c>[1, H, W]</c>: the pixels whose sigmoid exceeds 0.5
    /// (decided on the logit, <c>x &gt; 0</c>, so a rounded sigmoid cannot drop them), scored by their mean
    /// probability.
    /// </summary>
    private static OpenVocabSegmentationResult<T> BuildSingleNameResult(
        IEngine engine, INumericOperations<T> numOps, Tensor<T> logits, string name)
    {
        var probs = engine.TensorSigmoid(logits);
        var mask = MaskStatistics.SigmoidGreaterThan(logits, 0.5);
        int area = MaskStatistics.CountNonZero(mask);
        T score = MaskStatistics.MaskedMean(probs, mask, area);
        var semantic = engine.TensorAddScalar(mask, numOps.Negate(numOps.One));
        return new OpenVocabSegmentationResult<T>
        {
            Masks = mask,
            ClassNames = new[] { name },
            Scores = new Vector<T>(new[] { score }),
            SemanticMap = engine.Reshape(semantic, new[] { logits.Shape[1], logits.Shape[2] })
        };
    }

    /// <inheritdoc/>
    public virtual OpenVocabSegmentationResult<T> SegmentWithPrompt(Tensor<T> image, string prompt)
        => SegmentWithText(image, new[] { prompt });
}
