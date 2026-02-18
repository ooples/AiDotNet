using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;

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
/// Models extending this base class: LISA, VideoLISA, GLaMM, OMG-LLaVA, PixelLM.
/// </para>
/// </remarks>
public abstract class ReferringSegmentationBase<T> : SegmentationModelBase<T>, IReferringSegmentation<T>
{
    private readonly int _maxTextLength;

    /// <inheritdoc/>
    public int MaxTextLength => _maxTextLength;

    /// <inheritdoc/>
    public virtual bool SupportsConversation => false;

    /// <inheritdoc/>
    public virtual bool SupportsVideoInput => false;

    /// <summary>
    /// Initializes the base in native (trainable) mode.
    /// </summary>
    protected ReferringSegmentationBase(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer,
        ILossFunction<T>? lossFunction,
        int numClasses,
        int maxTextLength = 512)
        : base(architecture, optimizer, lossFunction, numClasses)
    {
        _maxTextLength = maxTextLength;
    }

    /// <summary>
    /// Initializes the base in ONNX (inference-only) mode.
    /// </summary>
    protected ReferringSegmentationBase(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int numClasses,
        int maxTextLength = 512)
        : base(architecture, onnxModelPath, numClasses)
    {
        _maxTextLength = maxTextLength;
    }

    /// <inheritdoc/>
    public abstract ReferringSegmentationResult<T> SegmentFromExpression(Tensor<T> image, string expression);

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
