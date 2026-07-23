using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.Augmentation;

/// <summary>
/// Specifies the mixing strategy used for label mixing.
/// </summary>
public enum MixingStrategy
{
    /// <summary>
    /// Mixup: Linear interpolation of entire images.
    /// </summary>
    Mixup,

    /// <summary>
    /// CutMix: Cut and paste rectangular regions.
    /// </summary>
    CutMix,

    /// <summary>
    /// MixupCutMix: Dynamically choose between Mixup and CutMix.
    /// </summary>
    MixupCutMix,

    /// <summary>
    /// Custom mixing strategy.
    /// </summary>
    Custom
}

/// <summary>
/// Event arguments for label mixing operations in augmentations like Mixup and CutMix.
/// </summary>
/// <remarks>
/// <para>
/// When augmentations like Mixup or CutMix are applied, they blend data from two samples
/// together. The labels must also be blended proportionally. This event allows the training
/// loop to handle the soft label generation appropriately.
/// </para>
/// <para><b>For Beginners:</b> In normal classification, an image is 100% one class (like "cat").
/// With Mixup/CutMix, we blend two images together, so the label becomes something like
/// "70% cat, 30% dog". This helps the model learn smoother decision boundaries.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class LabelMixingEventArgs<T> : EventArgs
{
    /// <summary>
    /// Gets the original hard labels for the first sample.
    /// </summary>
    public Vector<T> OriginalLabels1 { get; }

    /// <summary>
    /// Gets the original hard labels for the second (mixed) sample.
    /// </summary>
    public Vector<T> OriginalLabels2 { get; }

    /// <summary>
    /// Gets or sets the resulting mixed soft labels.
    /// </summary>
    /// <remarks>
    /// The training loop should set this to: lambda * OriginalLabels1 + (1 - lambda) * OriginalLabels2
    /// </remarks>
    public Vector<T>? MixedLabels { get; set; }

    /// <summary>
    /// Gets the mixing coefficient (lambda).
    /// </summary>
    /// <remarks>
    /// For Mixup: lambda determines the blend ratio.
    /// For CutMix: lambda represents the area ratio of the original image kept.
    /// </remarks>
    public T MixingLambda { get; }

    /// <summary>
    /// Gets the index of the first sample in the batch.
    /// </summary>
    public int SampleIndex1 { get; }

    /// <summary>
    /// Gets the index of the second sample in the batch.
    /// </summary>
    public int SampleIndex2 { get; }

    /// <summary>
    /// Gets the mixing strategy used.
    /// </summary>
    public MixingStrategy Strategy { get; }

    /// <summary>
    /// Gets additional metadata about the mixing operation.
    /// </summary>
    /// <remarks>
    /// For CutMix, this may contain the bounding box coordinates of the cut region.
    /// </remarks>
    public IDictionary<string, object> Metadata { get; }

    /// <summary>
    /// Creates a new label mixing event.
    /// </summary>
    /// <param name="originalLabels1">The labels for the first sample.</param>
    /// <param name="originalLabels2">The labels for the second sample.</param>
    /// <param name="mixingLambda">The mixing coefficient.</param>
    /// <param name="sampleIndex1">The index of the first sample.</param>
    /// <param name="sampleIndex2">The index of the second sample.</param>
    /// <param name="strategy">The mixing strategy used.</param>
    public LabelMixingEventArgs(
        Vector<T> originalLabels1,
        Vector<T> originalLabels2,
        T mixingLambda,
        int sampleIndex1,
        int sampleIndex2,
        MixingStrategy strategy)
    {
        Guard.NotNull(originalLabels1);
        OriginalLabels1 = originalLabels1;
        Guard.NotNull(originalLabels2);
        OriginalLabels2 = originalLabels2;
        MixingLambda = mixingLambda;
        SampleIndex1 = sampleIndex1;
        SampleIndex2 = sampleIndex2;
        Strategy = strategy;
        Metadata = new Dictionary<string, object>();
    }
}

/// <summary>
/// Event arguments raised when an augmentation is applied.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class AugmentationAppliedEventArgs<T> : EventArgs
{
    /// <summary>
    /// Gets the name of the augmentation that was applied.
    /// </summary>
    public string AugmentationName { get; }

    /// <summary>
    /// Gets the parameters used for this application.
    /// </summary>
    public IDictionary<string, object> Parameters { get; }

    /// <summary>
    /// Gets the sample index within the batch.
    /// </summary>
    public int SampleIndex { get; }

    /// <summary>
    /// Gets whether the augmentation was actually applied (vs. skipped due to probability).
    /// </summary>
    public bool WasApplied { get; }

    /// <summary>
    /// Creates a new augmentation applied event.
    /// </summary>
    /// <param name="augmentationName">The name of the augmentation.</param>
    /// <param name="parameters">The parameters used.</param>
    /// <param name="sampleIndex">The sample index.</param>
    /// <param name="wasApplied">Whether the augmentation was applied.</param>
    public AugmentationAppliedEventArgs(
        string augmentationName,
        IDictionary<string, object> parameters,
        int sampleIndex,
        bool wasApplied)
    {
        AugmentationName = augmentationName;
        Parameters = parameters;
        SampleIndex = sampleIndex;
        WasApplied = wasApplied;
    }
}
