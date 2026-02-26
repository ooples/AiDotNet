using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Validation;

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
public abstract class OpenVocabSegmentationBase<T> : SegmentationModelBase<T>, IOpenVocabSegmentation<T>
{
    private readonly int _maxCategories;
    private readonly int _maxPromptLength;

    /// <inheritdoc/>
    public int MaxCategories => _maxCategories;

    /// <inheritdoc/>
    public int MaxPromptLength => _maxPromptLength;

    /// <summary>
    /// Initializes the base in native (trainable) mode.
    /// </summary>
    protected OpenVocabSegmentationBase(
        NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer,
        ILossFunction<T>? lossFunction,
        int numClasses,
        int maxCategories = 256,
        int maxPromptLength = 77)
        : base(architecture, optimizer, lossFunction, numClasses)
    {
        Guard.Positive(maxCategories);
        Guard.Positive(maxPromptLength);
        _maxCategories = maxCategories;
        _maxPromptLength = maxPromptLength;
    }

    /// <summary>
    /// Initializes the base in ONNX (inference-only) mode.
    /// </summary>
    protected OpenVocabSegmentationBase(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int numClasses,
        int maxCategories = 256,
        int maxPromptLength = 77)
        : base(architecture, onnxModelPath, numClasses)
    {
        Guard.Positive(maxCategories);
        Guard.Positive(maxPromptLength);
        _maxCategories = maxCategories;
        _maxPromptLength = maxPromptLength;
    }

    /// <inheritdoc/>
    public abstract OpenVocabSegmentationResult<T> SegmentWithText(Tensor<T> image, IReadOnlyList<string> classNames);

    /// <inheritdoc/>
    public abstract OpenVocabSegmentationResult<T> SegmentWithPrompt(Tensor<T> image, string prompt);
}
