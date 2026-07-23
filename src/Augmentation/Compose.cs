using AiDotNet.Augmentation.Image;
using AiDotNet.Validation;



namespace AiDotNet.Augmentation;

/// <summary>
/// Applies multiple augmentations sequentially.
/// </summary>
/// <remarks>
/// <para>
/// Compose chains multiple augmentations together, applying them one after another.
/// Each augmentation sees the output of the previous one. This is the most common
/// way to build augmentation pipelines.
/// </para>
/// <para><b>For Beginners:</b> Think of this like a recipe with multiple steps.
/// First flip the image, then adjust brightness, then add noise. Each step
/// transforms the result of the previous step.
/// </para>
/// <para><b>Example usage:</b>
/// <code>
/// var pipeline = new Compose&lt;float, ImageTensor&lt;float&gt;&gt;(
///     new HorizontalFlip&lt;float&gt;(),
///     new Brightness&lt;float&gt;(),
///     new GaussianNoise&lt;float&gt;()
/// );
/// var augmented = pipeline.Apply(image, context);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <typeparam name="TData">The data type being augmented.</typeparam>
public class Compose<T, TData> : IAugmentation<T, TData>, ISpatialAugmentation<T, TData>
{
    private readonly List<IAugmentation<T, TData>> _augmentations;

    /// <inheritdoc />
    public string Name => "Compose";

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
    /// Gets the list of augmentations in this composition.
    /// </summary>
    public IReadOnlyList<IAugmentation<T, TData>> Augmentations => _augmentations;

    /// <summary>
    /// Creates a new composition of augmentations.
    /// </summary>
    /// <param name="augmentations">The augmentations to apply in sequence.</param>
    /// <param name="probability">The probability of applying this entire pipeline (default 1.0).</param>
    public Compose(IEnumerable<IAugmentation<T, TData>> augmentations, double probability = 1.0)
    {
        if (probability < 0 || probability > 1)
            throw new ArgumentOutOfRangeException(nameof(probability), "Probability must be between 0 and 1");

        Guard.NotNull(augmentations);
        _augmentations = augmentations.ToList();
        Probability = probability;
    }

    /// <summary>
    /// Creates a new composition of augmentations.
    /// </summary>
    /// <param name="augmentations">The augmentations to apply in sequence.</param>
    public Compose(params IAugmentation<T, TData>[] augmentations)
        : this(augmentations.AsEnumerable(), 1.0)
    {
    }

    /// <inheritdoc />
    public TData Apply(TData data, AugmentationContext<T>? context = null)
    {
        context ??= new AugmentationContext<T>();

        if (!IsEnabled)
            return data;

        // Check if this composition should be applied
        if (Probability < 1.0 && context.GetRandomDouble(0, 1) > Probability)
            return data;

        var current = data;
        foreach (var augmentation in _augmentations.Where(a => a.IsEnabled))
        {
            current = augmentation.Apply(current, context);
        }

        return current;
    }

    /// <inheritdoc />
    public AugmentedSample<T, TData> ApplyWithTargets(AugmentedSample<T, TData> sample, AugmentationContext<T>? context = null)
    {
        context ??= new AugmentationContext<T>();

        if (!IsEnabled)
            return sample;

        // Check if this composition should be applied
        if (Probability < 1.0 && context.GetRandomDouble(0, 1) > Probability)
            return sample;

        var current = sample;
        foreach (var augmentation in _augmentations.Where(a => a.IsEnabled))
        {
            current = augmentation is ISpatialAugmentation<T, TData> spatial
                ? spatial.ApplyWithTargets(current, context)
                : new AugmentedSample<T, TData>(augmentation.Apply(current.Data, context))
                {
                    BoundingBoxes = current.BoundingBoxes,
                    Keypoints = current.Keypoints,
                    Masks = current.Masks,
                    Labels = current.Labels,
                    Metadata = current.Metadata
                };
        }

        return current;
    }

    /// <inheritdoc />
    public IDictionary<string, object> GetParameters()
    {
        return new Dictionary<string, object>
        {
            ["probability"] = Probability,
            ["num_augmentations"] = _augmentations.Count,
            ["augmentations"] = _augmentations.Select(a => a.Name).ToList()
        };
    }

    /// <summary>
    /// Creates a new composition with an additional augmentation appended.
    /// </summary>
    /// <param name="augmentation">The augmentation to append.</param>
    /// <returns>A new Compose instance with the augmentation added.</returns>
    /// <remarks>
    /// This method returns a new immutable instance rather than modifying the current one,
    /// making it safe for use in multi-threaded scenarios.
    /// </remarks>
    public Compose<T, TData> With(IAugmentation<T, TData> augmentation)
    {
        if (augmentation is null)
            throw new ArgumentNullException(nameof(augmentation));

        var newList = new List<IAugmentation<T, TData>>(_augmentations) { augmentation };
        return new Compose<T, TData>(newList, Probability)
        {
            IsTrainingOnly = IsTrainingOnly,
            IsEnabled = IsEnabled
        };
    }
}
