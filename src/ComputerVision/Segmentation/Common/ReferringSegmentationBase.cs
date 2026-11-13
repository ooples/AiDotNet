using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.VisionLanguage.Interfaces;

namespace AiDotNet.ComputerVision.Segmentation.Common;

/// <summary>
/// Abstract base class for referring segmentation models that segment objects from natural language
/// descriptions with complex reasoning about spatial relationships and attributes.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Referring segmentation goes beyond open-vocabulary by understanding
/// complex descriptions like "the person standing behind the counter" or "the animal that could
/// be dangerous". These models typically combine a large language model (LLM) with a segmentation
/// backbone.
///
/// Models extending this base class: LISA, VideoLISA, OMG-LLaVA, PixelLM.
/// </para>
/// </remarks>
public abstract partial class ReferringSegmentationBase<T> : SegmentationModelBase<T>, IReferringSegmentation<T>
{
    // protected and mutable so a derived model can read it and restore it on deserialization - see
    // PanopticSegmentationBase._numStuffClasses for why private/readonly makes a base unadoptable.
    protected int _maxTextLength;

    /// <inheritdoc/>
    public int MaxTextLength => _maxTextLength;

    /// <inheritdoc/>
    public virtual bool SupportsConversation => false;

    /// <inheritdoc/>
    public virtual bool SupportsVideoInput => false;

    /// <summary>
    /// The CLIP-style text encoder that turns a referring expression into the query embedding.
    /// </summary>
    protected readonly ITextEncoder<T> _textEncoder;

    /// <summary>The inverse temperature <c>1 / tau</c> that scales every cosine similarity.</summary>
    protected readonly T _logitScale;

    /// <summary>The sigmoid probability a pixel must exceed to be in the mask, kept in double so the decision is exact.</summary>
    protected readonly double _maskThreshold;

    /// <summary>Normalized embeddings <c>[NumClasses, D]</c> of the closed-set vocabulary Predict/Train score against.</summary>
    private readonly Tensor<T> _vocabularyEmbeddings;

    /// <summary>The expression embedding of an in-flight <see cref="SegmentFromExpression"/> call, or <c>null</c>.</summary>
    private Tensor<T>? _queryEmbeddings;

    /// <summary>
    /// Gets the text encoder whose embedding of the expression the pixel embeddings are compared with.
    /// </summary>
    public ITextEncoder<T> TextEncoder => _textEncoder;

    /// <summary>
    /// Gets the width of the pixel embeddings the decoder must produce: the text encoder's embedding dimension.
    /// </summary>
    protected int TextEmbeddingDimension => _textEncoder.TextEmbeddingDimension;

    /// <summary>
    /// Initializes the base in native (trainable) mode.
    /// </summary>
    protected ReferringSegmentationBase(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer,
        ILossFunction<T>? lossFunction,
        ITextEncoder<T> textEncoder,
        ReferringSegmentationOptions options,
        int maxTextLength = 512)
        : base(architecture, optimizer, lossFunction, RequireOptions(options).NumClasses)
    {
        _maxTextLength = maxTextLength;
        _textEncoder = textEncoder ?? throw new ArgumentNullException(nameof(textEncoder));
        (_logitScale, _maskThreshold, _vocabularyEmbeddings) = PrepareVocabulary(textEncoder, options);
    }

    /// <summary>
    /// Initializes the base in ONNX (inference-only) mode. The ONNX graph must output pixel embeddings
    /// <c>[B, D, H, W]</c> with <c>D</c> equal to the text encoder's embedding dimension.
    /// </summary>
    /// <remarks>
    /// Only the native constructor carries optional parameters (RS0026: two new overloads may not both).
    /// </remarks>
    protected ReferringSegmentationBase(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        ITextEncoder<T> textEncoder,
        ReferringSegmentationOptions options,
        int maxTextLength)
        : base(architecture, onnxModelPath, RequireOptions(options).NumClasses)
    {
        _maxTextLength = maxTextLength;
        _textEncoder = textEncoder ?? throw new ArgumentNullException(nameof(textEncoder));
        (_logitScale, _maskThreshold, _vocabularyEmbeddings) = PrepareVocabulary(textEncoder, options);
    }

    private static ReferringSegmentationOptions RequireOptions(ReferringSegmentationOptions options)
        => options ?? throw new ArgumentNullException(nameof(options));

    private static (T LogitScale, double MaskThreshold, Tensor<T> Vocabulary) PrepareVocabulary(
        ITextEncoder<T> textEncoder, ReferringSegmentationOptions options)
    {
        if (textEncoder is null) throw new ArgumentNullException(nameof(textEncoder));
        options.ValidateTextQuerySettings();
        options.ValidateMaskThreshold();
        var numOps = MathHelper.GetNumericOperations<T>();
        return (numOps.FromDouble(1.0 / options.Temperature),
            options.MaskThreshold,
            TextEmbeddingClassifier.EncodeNormalized(textEncoder, options.ResolveClassNames()));
    }

    /// <summary>
    /// Scores pixel embeddings <c>[D, H, W]</c> or <c>[B, D, H, W]</c> against the active text
    /// embeddings (the closed-set class names, or the expression of an in-flight
    /// <see cref="SegmentFromExpression"/> call), returning <c>cos / tau</c> logits.
    /// </summary>
    protected Tensor<T> ClassifyPixels(Tensor<T> pixelEmbeddings)
        => TextEmbeddingClassifier.Classify(pixelEmbeddings, _queryEmbeddings ?? _vocabularyEmbeddings, _logitScale);

    /// <inheritdoc/>
    /// <remarks>
    /// The expression is encoded once by the text encoder (LISA's <c>[SEG]</c> embedding, PixelLM's and
    /// OMG-LLaVA's segmentation token), every pixel is scored by its cosine with that embedding over the
    /// temperature, and the mask keeps the pixels whose sigmoid probability exceeds the mask threshold.
    /// The confidence is the mean probability over the mask, and the box is the mask's tight XYXY box.
    /// </remarks>
    public virtual ReferringSegmentationResult<T> SegmentFromExpression(Tensor<T> image, string expression)
    {
        if (expression is null) throw new ArgumentNullException(nameof(expression));
        var text = TextEmbeddingClassifier.EncodeNormalized(_textEncoder, new[] { expression });
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

        return BuildResult(Engine, NumOps, logits, _maskThreshold, expression);
    }

    /// <summary>
    /// Turns expression logits <c>[1, H, W]</c> into the mask, confidence, box and response.
    /// </summary>
    internal static ReferringSegmentationResult<T> BuildResult(
        IEngine engine, INumericOperations<T> numOps, Tensor<T> logits, double maskThreshold, string expression)
    {
        var probs = engine.TensorSigmoid(logits);
        // Decided on the logit (LISA's logit > 0 at the default 0.5): a sigmoid rounded to T sits exactly on the
        // threshold for logits near the boundary and would drop pixels that are above it.
        var mask = MaskStatistics.SigmoidGreaterThan(logits, maskThreshold);
        // The area is counted in int (a Half sum of the mask rounds above 2048 pixels and overflows above 65504)
        // and the confidence is the mean over the mask's own pixels, so a NaN logit outside it cannot leak in.
        int area = MaskStatistics.CountNonZero(mask);
        T confidence = MaskStatistics.MaskedMean(probs, mask, area);
        var boxes = engine.MasksToBoxes(mask).Cast<T>();
        string response = area > 0
            ? string.Format(CultureInfo.InvariantCulture, "Segmented region matching '{0}' with {1} pixels ({2:F2} confidence)",
                expression, area, numOps.ToDouble(confidence))
            : $"No region found matching '{expression}'";
        return new ReferringSegmentationResult<T> { Masks = mask, TextResponse = response, Confidence = confidence, BoundingBoxes = boxes };
    }

    /// <summary>
    /// Segments the expression in each frame of <c>[F, C, H, W]</c> (or one <c>[C, H, W]</c> frame),
    /// stamping each result with its frame index.
    /// </summary>
    protected List<ReferringSegmentationResult<T>> SegmentEachFrame(Tensor<T> frames, string expression)
    {
        if (frames is null) throw new ArgumentNullException(nameof(frames));
        var results = new List<ReferringSegmentationResult<T>>();
        if (frames.Rank == 3)
        {
            var single = SegmentFromExpression(frames, expression);
            single.FrameIndex = 0;
            results.Add(single);
            return results;
        }

        int numFrames = frames.Shape[0];
        for (int f = 0; f < numFrames; f++)
        {
            var result = SegmentFromExpression(frames.GetSlice(f), expression);
            result.FrameIndex = f;
            results.Add(result);
        }

        return results;
    }

    /// <inheritdoc/>
    public virtual ReferringSegmentationResult<T> SegmentFromConversation(
        Tensor<T> image,
        IReadOnlyList<(string Role, string Message)> conversationHistory,
        string currentQuery)
    {
        if (!SupportsConversation)
        {
            throw new NotSupportedException(
                $"{GetType().Name} does not support conversational segmentation. " +
                "Use SegmentFromExpression() for single-turn queries, or use a model like LISA that supports conversation.");
        }

        return SegmentFromConversationInternal(image, conversationHistory, currentQuery);
    }

    /// <summary>
    /// Model-specific conversational segmentation. Only called when SupportsConversation is true.
    /// </summary>
    protected virtual ReferringSegmentationResult<T> SegmentFromConversationInternal(
        Tensor<T> image,
        IReadOnlyList<(string Role, string Message)> conversationHistory,
        string currentQuery)
    {
        throw new NotSupportedException("Conversational segmentation not implemented by this model.");
    }

    /// <inheritdoc/>
    public virtual List<ReferringSegmentationResult<T>> SegmentVideoFromExpression(
        Tensor<T> frames, string expression)
    {
        if (!SupportsVideoInput)
        {
            throw new NotSupportedException(
                $"{GetType().Name} does not support video input. " +
                "Use a model like VideoLISA that supports the SupportsVideoInput property.");
        }

        return SegmentVideoFromExpressionInternal(frames, expression);
    }

    /// <summary>
    /// Model-specific video referring segmentation. Only called when SupportsVideoInput is true.
    /// </summary>
    protected virtual List<ReferringSegmentationResult<T>> SegmentVideoFromExpressionInternal(
        Tensor<T> frames, string expression)
    {
        throw new NotSupportedException("Video referring segmentation not implemented by this model.");
    }
}
