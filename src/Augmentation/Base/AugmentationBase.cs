using AiDotNet.Augmentation.Data;
using AiDotNet.Augmentation.Events;
using AiDotNet.Augmentation.Interfaces;

namespace AiDotNet.Augmentation.Base;

/// <summary>
/// Abstract base class for all augmentations providing common functionality.
/// </summary>
/// <remarks>
/// <para>
/// AugmentationBase provides:
/// - Probability-based application control
/// - Training/inference mode awareness
/// - Parameter serialization
/// - Event hooks for monitoring
/// </para>
/// <para><b>For Beginners:</b> This is the foundation that all augmentations build upon.
/// It handles common tasks like deciding whether to apply the augmentation based on
/// probability, tracking parameters, and ensuring augmentations behave correctly
/// during training vs. inference.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <typeparam name="TData">The data type being augmented.</typeparam>
public abstract class AugmentationBase<T, TData> : IAugmentation<T, TData>
{
    /// <summary>
    /// Event raised when this augmentation is applied.
    /// </summary>
    public event EventHandler<AugmentationAppliedEventArgs<T>>? OnAugmentationApplied;

    /// <summary>
    /// Gets the name of this augmentation.
    /// </summary>
    public virtual string Name => GetType().Name;

    /// <summary>
    /// Gets the probability of this augmentation being applied.
    /// </summary>
    public double Probability { get; protected set; }

    /// <summary>
    /// Gets whether this augmentation should only be applied during training.
    /// </summary>
    public virtual bool IsTrainingOnly => true;

    /// <summary>
    /// Gets or sets whether this augmentation is currently enabled.
    /// </summary>
    public bool IsEnabled { get; set; } = true;

    /// <summary>
    /// Initializes a new instance of the augmentation.
    /// </summary>
    /// <param name="probability">The probability of applying this augmentation (0.0 to 1.0).</param>
    protected AugmentationBase(double probability = 1.0)
    {
        if (probability < 0.0 || probability > 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(probability), "Probability must be between 0.0 and 1.0");
        }
        Probability = probability;
    }

    /// <summary>
    /// Applies the augmentation to the input data.
    /// </summary>
    /// <param name="data">The input data.</param>
    /// <param name="context">The augmentation context.</param>
    /// <returns>The augmented data (or original if not applied).</returns>
    public TData Apply(TData data, AugmentationContext<T>? context = null)
    {
        context = context ?? new AugmentationContext<T>();

        // Check if enabled
        if (!IsEnabled)
        {
            RaiseAugmentationApplied(context, false);
            return data;
        }

        // Check training mode
        if (IsTrainingOnly && !context.IsTraining)
        {
            RaiseAugmentationApplied(context, false);
            return data;
        }

        // Check probability
        if (!context.ShouldApply(Probability))
        {
            RaiseAugmentationApplied(context, false);
            return data;
        }

        // Apply the augmentation
        var result = ApplyAugmentation(data, context);
        RaiseAugmentationApplied(context, true);
        return result;
    }

    /// <summary>
    /// Implement this method to perform the actual augmentation.
    /// </summary>
    /// <param name="data">The input data.</param>
    /// <param name="context">The augmentation context.</param>
    /// <returns>The augmented data.</returns>
    protected abstract TData ApplyAugmentation(TData data, AugmentationContext<T> context);

    /// <summary>
    /// Gets the parameters of this augmentation.
    /// </summary>
    /// <returns>A dictionary of parameter names to values.</returns>
    public virtual IDictionary<string, object> GetParameters()
    {
        return new Dictionary<string, object>
        {
            ["name"] = Name,
            ["probability"] = Probability,
            ["isEnabled"] = IsEnabled,
            ["isTrainingOnly"] = IsTrainingOnly
        };
    }

    /// <summary>
    /// Raises the augmentation applied event.
    /// </summary>
    protected void RaiseAugmentationApplied(AugmentationContext<T> context, bool wasApplied)
    {
        OnAugmentationApplied?.Invoke(this, new AugmentationAppliedEventArgs<T>(
            Name,
            GetParameters(),
            context.SampleIndex,
            wasApplied
        ));
    }
}

/// <summary>
/// Base class for augmentations that transform spatial targets (bounding boxes, keypoints, masks).
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <typeparam name="TData">The data type being augmented.</typeparam>
public abstract class SpatialAugmentationBase<T, TData> : AugmentationBase<T, TData>, ISpatialAugmentation<T, TData>
{
    /// <summary>
    /// Gets whether this augmentation supports bounding box transformation.
    /// </summary>
    public virtual bool SupportsBoundingBoxes => true;

    /// <summary>
    /// Gets whether this augmentation supports keypoint transformation.
    /// </summary>
    public virtual bool SupportsKeypoints => true;

    /// <summary>
    /// Gets whether this augmentation supports mask transformation.
    /// </summary>
    public virtual bool SupportsMasks => true;

    /// <summary>
    /// Initializes a new spatial augmentation.
    /// </summary>
    /// <param name="probability">The probability of applying this augmentation.</param>
    protected SpatialAugmentationBase(double probability = 1.0) : base(probability)
    {
    }

    /// <summary>
    /// Applies the augmentation to data and all spatial targets.
    /// </summary>
    /// <param name="sample">The augmented sample containing data and targets.</param>
    /// <param name="context">The augmentation context.</param>
    /// <returns>The augmented sample with transformed targets.</returns>
    public AugmentedSample<T, TData> ApplyWithTargets(AugmentedSample<T, TData> sample, AugmentationContext<T>? context = null)
    {
        context = context ?? new AugmentationContext<T>();

        // Check if should apply
        if (!IsEnabled || (IsTrainingOnly && !context.IsTraining) || !context.ShouldApply(Probability))
        {
            return sample;
        }

        // Clone the sample to avoid modifying the original
        var result = sample.Clone();

        // Apply the data transformation and get the transform matrix/parameters
        var transformParams = ApplyWithTransformParams(result.Data, context);
        result.Data = transformParams.data;

        // Transform bounding boxes
        if (SupportsBoundingBoxes && result.HasBoundingBoxes && result.BoundingBoxes is not null)
        {
            for (int i = 0; i < result.BoundingBoxes.Count; i++)
            {
                result.BoundingBoxes[i] = TransformBoundingBox(result.BoundingBoxes[i], transformParams.parameters, context);
            }

            // Remove invalid boxes
            result.BoundingBoxes = result.BoundingBoxes.Where(b => b.IsValid()).ToList();
        }

        // Transform keypoints
        if (SupportsKeypoints && result.HasKeypoints && result.Keypoints is not null)
        {
            for (int i = 0; i < result.Keypoints.Count; i++)
            {
                result.Keypoints[i] = TransformKeypoint(result.Keypoints[i], transformParams.parameters, context);
            }
        }

        // Transform masks
        if (SupportsMasks && result.HasMasks && result.Masks is not null)
        {
            for (int i = 0; i < result.Masks.Count; i++)
            {
                result.Masks[i] = TransformMask(result.Masks[i], transformParams.parameters, context);
            }
        }

        RaiseAugmentationApplied(context, true);
        return result;
    }

    /// <summary>
    /// Applies the augmentation and returns both the result and transform parameters.
    /// </summary>
    /// <param name="data">The input data.</param>
    /// <param name="context">The augmentation context.</param>
    /// <returns>The transformed data and transform parameters.</returns>
    protected abstract (TData data, IDictionary<string, object> parameters) ApplyWithTransformParams(TData data, AugmentationContext<T> context);

    /// <summary>
    /// Transforms a bounding box according to the spatial transformation.
    /// </summary>
    /// <param name="box">The bounding box to transform.</param>
    /// <param name="transformParams">The transformation parameters.</param>
    /// <param name="context">The augmentation context.</param>
    /// <returns>The transformed bounding box.</returns>
    protected abstract BoundingBox<T> TransformBoundingBox(BoundingBox<T> box, IDictionary<string, object> transformParams, AugmentationContext<T> context);

    /// <summary>
    /// Transforms a keypoint according to the spatial transformation.
    /// </summary>
    /// <param name="keypoint">The keypoint to transform.</param>
    /// <param name="transformParams">The transformation parameters.</param>
    /// <param name="context">The augmentation context.</param>
    /// <returns>The transformed keypoint.</returns>
    protected abstract Keypoint<T> TransformKeypoint(Keypoint<T> keypoint, IDictionary<string, object> transformParams, AugmentationContext<T> context);

    /// <summary>
    /// Transforms a segmentation mask according to the spatial transformation.
    /// </summary>
    /// <param name="mask">The mask to transform.</param>
    /// <param name="transformParams">The transformation parameters.</param>
    /// <param name="context">The augmentation context.</param>
    /// <returns>The transformed mask.</returns>
    protected abstract SegmentationMask<T> TransformMask(SegmentationMask<T> mask, IDictionary<string, object> transformParams, AugmentationContext<T> context);

    /// <summary>
    /// Default implementation that calls ApplyWithTransformParams.
    /// </summary>
    protected override TData ApplyAugmentation(TData data, AugmentationContext<T> context)
    {
        return ApplyWithTransformParams(data, context).data;
    }
}

/// <summary>
/// Base class for label-mixing augmentations like Mixup and CutMix.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <typeparam name="TData">The data type being augmented.</typeparam>
public abstract class LabelMixingAugmentationBase<T, TData> : AugmentationBase<T, TData>, ILabelMixingAugmentation<T, TData>
{
    /// <summary>
    /// Event raised when labels need to be mixed.
    /// </summary>
    public event EventHandler<LabelMixingEventArgs<T>>? OnLabelMixing;

    /// <summary>
    /// Gets the mixing lambda from the last application.
    /// </summary>
    public T LastMixingLambda { get; protected set; } = default!;

    /// <summary>
    /// Gets or sets the alpha parameter for Beta distribution sampling.
    /// </summary>
    public double Alpha { get; set; } = 1.0;

    /// <summary>
    /// Initializes a new label mixing augmentation.
    /// </summary>
    /// <param name="probability">The probability of applying this augmentation.</param>
    /// <param name="alpha">The alpha parameter for Beta distribution.</param>
    protected LabelMixingAugmentationBase(double probability = 1.0, double alpha = 1.0) : base(probability)
    {
        Alpha = alpha;
    }

    /// <summary>
    /// Samples a mixing lambda value from Beta(alpha, alpha) distribution.
    /// </summary>
    /// <param name="context">The augmentation context.</param>
    /// <returns>The sampled lambda value.</returns>
    protected double SampleLambda(AugmentationContext<T> context)
    {
        if (Alpha <= 0)
        {
            return 1.0;
        }

        return context.SampleBeta(Alpha, Alpha);
    }

    /// <summary>
    /// Raises the label mixing event.
    /// </summary>
    protected void RaiseLabelMixing(LabelMixingEventArgs<T> args)
    {
        OnLabelMixing?.Invoke(this, args);
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["alpha"] = Alpha;
        return parameters;
    }
}
