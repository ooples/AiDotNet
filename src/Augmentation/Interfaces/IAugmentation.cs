using AiDotNet.Augmentation.Data;
using AiDotNet.Augmentation.Events;

namespace AiDotNet.Augmentation.Interfaces;

/// <summary>
/// Base interface for all data augmentations across domains (image, audio, tabular).
/// </summary>
/// <remarks>
/// <para>
/// Augmentations are stochastic transformations applied during training to improve model
/// generalization. Unlike preprocessing transforms, augmentations:
/// - Produce different outputs for the same input (stochastic)
/// - Are typically disabled during inference (training-only)
/// - Have a probability of being applied
/// - Can be composed in pipelines
/// </para>
/// <para><b>For Beginners:</b> Augmentation is like creating variations of your training data.
/// If you're training an image classifier, flipping images horizontally gives the model
/// more examples to learn from without collecting new data. This helps the model
/// generalize better to new, unseen data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TData">The data type being augmented (e.g., ImageTensor, AudioTensor).</typeparam>
public interface IAugmentation<T, TData>
{
    /// <summary>
    /// Gets the name of this augmentation for logging and debugging.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Gets the probability of this augmentation being applied (0.0 to 1.0).
    /// </summary>
    double Probability { get; }

    /// <summary>
    /// Gets whether this augmentation should only be applied during training.
    /// </summary>
    /// <remarks>
    /// Most augmentations are training-only. Test-Time Augmentation (TTA) uses
    /// specific augmentations during inference to improve predictions.
    /// </remarks>
    bool IsTrainingOnly { get; }

    /// <summary>
    /// Gets whether this augmentation is currently enabled.
    /// </summary>
    bool IsEnabled { get; set; }

    /// <summary>
    /// Applies the augmentation to the input data.
    /// </summary>
    /// <param name="data">The input data to augment.</param>
    /// <param name="context">The augmentation context containing random state and targets.</param>
    /// <returns>The augmented data.</returns>
    TData Apply(TData data, AugmentationContext<T>? context = null);

    /// <summary>
    /// Gets the parameters of this augmentation for serialization/logging.
    /// </summary>
    /// <returns>A dictionary of parameter names to values.</returns>
    IDictionary<string, object> GetParameters();
}

/// <summary>
/// Interface for augmentations that modify labels (e.g., Mixup, CutMix).
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <typeparam name="TData">The data type being augmented.</typeparam>
public interface ILabelMixingAugmentation<T, TData> : IAugmentation<T, TData>
{
    /// <summary>
    /// Event raised when labels need to be mixed due to data mixing augmentation.
    /// </summary>
    event EventHandler<LabelMixingEventArgs<T>>? OnLabelMixing;

    /// <summary>
    /// Gets the mixing lambda value from the last application.
    /// </summary>
    T LastMixingLambda { get; }
}

/// <summary>
/// Interface for augmentations that can transform spatial targets
/// (bounding boxes, keypoints, segmentation masks).
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <typeparam name="TData">The data type being augmented.</typeparam>
public interface ISpatialAugmentation<T, TData> : IAugmentation<T, TData>
{
    /// <summary>
    /// Gets whether this augmentation supports bounding box transformation.
    /// </summary>
    bool SupportsBoundingBoxes { get; }

    /// <summary>
    /// Gets whether this augmentation supports keypoint transformation.
    /// </summary>
    bool SupportsKeypoints { get; }

    /// <summary>
    /// Gets whether this augmentation supports segmentation mask transformation.
    /// </summary>
    bool SupportsMasks { get; }

    /// <summary>
    /// Applies the augmentation and transforms all spatial targets.
    /// </summary>
    /// <param name="sample">The augmented sample containing data and targets.</param>
    /// <param name="context">The augmentation context.</param>
    /// <returns>The augmented sample with transformed targets.</returns>
    AugmentedSample<T, TData> ApplyWithTargets(AugmentedSample<T, TData> sample, AugmentationContext<T>? context = null);
}
