using AiDotNet.Augmentation.Image;
using AiDotNet.Validation;



namespace AiDotNet.Augmentation;

/// <summary>
/// Randomly selects and applies exactly one augmentation from a set.
/// </summary>
/// <remarks>
/// <para>
/// OneOf randomly chooses one augmentation from the provided set and applies it.
/// This is useful when you want variety but don't want to apply multiple similar
/// augmentations that might compound their effects too strongly.
/// </para>
/// <para><b>For Beginners:</b> Think of this like flipping a coin to decide
/// which transformation to apply. You might flip the image OR rotate it,
/// but not both at the same time.
/// </para>
/// <para><b>Example usage:</b>
/// <code>
/// var randomTransform = new OneOf&lt;float, ImageTensor&lt;float&gt;&gt;(
///     new HorizontalFlip&lt;float&gt;(),
///     new VerticalFlip&lt;float&gt;(),
///     new Rotation&lt;float&gt;()
/// );
/// var augmented = randomTransform.Apply(image, context);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <typeparam name="TData">The data type being augmented.</typeparam>
public class OneOf<T, TData> : IAugmentation<T, TData>, ISpatialAugmentation<T, TData>
{
    private readonly List<IAugmentation<T, TData>> _augmentations;
    private readonly List<double> _weights;

    /// <inheritdoc />
    public string Name => "OneOf";

    /// <inheritdoc />
    public double Probability { get; }

    /// <inheritdoc />
    public bool IsTrainingOnly { get; set; } = true;

    /// <inheritdoc />
    public bool IsEnabled { get; set; } = true;

    /// <inheritdoc />
    public bool SupportsBoundingBoxes => _augmentations.OfType<ISpatialAugmentation<T, TData>>()
        .Any(a => a.SupportsBoundingBoxes);

    /// <inheritdoc />
    public bool SupportsKeypoints => _augmentations.OfType<ISpatialAugmentation<T, TData>>()
        .Any(a => a.SupportsKeypoints);

    /// <inheritdoc />
    public bool SupportsMasks => _augmentations.OfType<ISpatialAugmentation<T, TData>>()
        .Any(a => a.SupportsMasks);

    /// <summary>
    /// Gets the list of augmentations to choose from.
    /// </summary>
    public IReadOnlyList<IAugmentation<T, TData>> Augmentations => _augmentations;

    /// <summary>
    /// Creates a new OneOf with uniform weights.
    /// </summary>
    /// <param name="augmentations">The augmentations to choose from.</param>
    /// <param name="probability">The probability of applying any augmentation (default 1.0).</param>
    public OneOf(IEnumerable<IAugmentation<T, TData>> augmentations, double probability = 1.0)
    {
        if (probability < 0 || probability > 1)
            throw new ArgumentOutOfRangeException(nameof(probability), "Probability must be between 0 and 1");

        Guard.NotNull(augmentations);
        _augmentations = augmentations.ToList();
        if (_augmentations.Count == 0)
            throw new ArgumentException("At least one augmentation is required", nameof(augmentations));

        // Uniform weights
        _weights = Enumerable.Repeat(1.0 / _augmentations.Count, _augmentations.Count).ToList();
        Probability = probability;
    }

    /// <summary>
    /// Creates a new OneOf with specified weights.
    /// </summary>
    /// <param name="augmentationsWithWeights">The augmentations with their selection weights.</param>
    /// <param name="probability">The probability of applying any augmentation (default 1.0).</param>
    public OneOf(
        IEnumerable<(IAugmentation<T, TData> augmentation, double weight)> augmentationsWithWeights,
        double probability = 1.0)
    {
        if (probability < 0 || probability > 1)
            throw new ArgumentOutOfRangeException(nameof(probability), "Probability must be between 0 and 1");

        var list = augmentationsWithWeights?.ToList() ?? throw new ArgumentNullException(nameof(augmentationsWithWeights));
        if (list.Count == 0)
            throw new ArgumentException("At least one augmentation is required", nameof(augmentationsWithWeights));

        _augmentations = list.Select(x => x.augmentation).ToList();
        _weights = list.Select(x => x.weight).ToList();

        // Normalize weights
        double sum = _weights.Sum();
        if (sum <= 0)
            throw new ArgumentException("Weights must sum to a positive value");

        for (int i = 0; i < _weights.Count; i++)
        {
            _weights[i] /= sum;
        }

        Probability = probability;
    }

    /// <summary>
    /// Creates a new OneOf with uniform weights.
    /// </summary>
    /// <param name="augmentations">The augmentations to choose from.</param>
    public OneOf(params IAugmentation<T, TData>[] augmentations)
        : this(augmentations.AsEnumerable(), 1.0)
    {
    }

    /// <inheritdoc />
    public TData Apply(TData data, AugmentationContext<T>? context = null)
    {
        context ??= new AugmentationContext<T>();

        if (!IsEnabled)
            return data;

        // Check if any augmentation should be applied
        if (Probability < 1.0 && context.GetRandomDouble(0, 1) > Probability)
            return data;

        // Select one augmentation based on weights
        var selected = SelectAugmentation(context);
        return selected.Apply(data, context);
    }

    /// <inheritdoc />
    public AugmentedSample<T, TData> ApplyWithTargets(AugmentedSample<T, TData> sample, AugmentationContext<T>? context = null)
    {
        context ??= new AugmentationContext<T>();

        if (!IsEnabled)
            return sample;

        // Check if any augmentation should be applied
        if (Probability < 1.0 && context.GetRandomDouble(0, 1) > Probability)
            return sample;

        // Select one augmentation based on weights
        var selected = SelectAugmentation(context);

        if (selected is ISpatialAugmentation<T, TData> spatial)
        {
            return spatial.ApplyWithTargets(sample, context);
        }

        return new AugmentedSample<T, TData>(selected.Apply(sample.Data, context))
        {
            BoundingBoxes = sample.BoundingBoxes,
            Keypoints = sample.Keypoints,
            Masks = sample.Masks,
            Labels = sample.Labels,
            Metadata = sample.Metadata
        };
    }

    private IAugmentation<T, TData> SelectAugmentation(AugmentationContext<T> context)
    {
        double r = context.GetRandomDouble(0, 1);
        double cumulative = 0;

        for (int i = 0; i < _augmentations.Count; i++)
        {
            cumulative += _weights[i];
            if (r <= cumulative)
            {
                return _augmentations[i];
            }
        }

        // Fallback to last (shouldn't happen with proper weights)
        return _augmentations[^1];
    }

    /// <inheritdoc />
    public IDictionary<string, object> GetParameters()
    {
        return new Dictionary<string, object>
        {
            ["probability"] = Probability,
            ["num_augmentations"] = _augmentations.Count,
            ["augmentations"] = _augmentations.Select(a => a.Name).ToList(),
            ["weights"] = _weights.ToList()
        };
    }
}
