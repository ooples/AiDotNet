using AiDotNet.Augmentation.Data;
using AiDotNet.Augmentation.Interfaces;

namespace AiDotNet.Augmentation.Composition;

/// <summary>
/// Randomly selects and applies N augmentations from a set.
/// </summary>
/// <remarks>
/// <para>
/// SomeOf randomly chooses N augmentations from the provided set and applies them
/// in sequence. The number N can be fixed or randomly sampled from a range.
/// This provides more variety than OneOf while still limiting the total augmentation.
/// </para>
/// <para><b>For Beginners:</b> Think of this like drawing cards from a deck.
/// You pick 2 or 3 random transformations and apply them one after another.
/// This creates more variety in your training data.
/// </para>
/// <para><b>Example usage:</b>
/// <code>
/// var randomTransforms = new SomeOf&lt;float, ImageTensor&lt;float&gt;&gt;(
///     n: 2,
///     new HorizontalFlip&lt;float&gt;(),
///     new VerticalFlip&lt;float&gt;(),
///     new Rotation&lt;float&gt;(),
///     new Scale&lt;float&gt;()
/// );
/// var augmented = randomTransforms.Apply(image, context);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <typeparam name="TData">The data type being augmented.</typeparam>
public class SomeOf<T, TData> : IAugmentation<T, TData>, ISpatialAugmentation<T, TData>
{
    private readonly List<IAugmentation<T, TData>> _augmentations;

    /// <inheritdoc />
    public string Name => "SomeOf";

    /// <inheritdoc />
    public double Probability { get; }

    /// <inheritdoc />
    public bool IsTrainingOnly { get; set; } = true;

    /// <inheritdoc />
    public bool IsEnabled { get; set; } = true;

    /// <summary>
    /// Gets the minimum number of augmentations to apply.
    /// </summary>
    public int MinN { get; }

    /// <summary>
    /// Gets the maximum number of augmentations to apply.
    /// </summary>
    public int MaxN { get; }

    /// <summary>
    /// Gets whether the selected augmentations should be applied in random order.
    /// </summary>
    public bool RandomizeOrder { get; }

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
    /// Creates a new SomeOf with a fixed number of augmentations.
    /// </summary>
    /// <param name="n">The number of augmentations to apply.</param>
    /// <param name="augmentations">The augmentations to choose from.</param>
    /// <param name="probability">The probability of applying any augmentation (default 1.0).</param>
    /// <param name="randomizeOrder">Whether to randomize the order of application (default true).</param>
    public SomeOf(
        int n,
        IEnumerable<IAugmentation<T, TData>> augmentations,
        double probability = 1.0,
        bool randomizeOrder = true)
        : this(n, n, augmentations, probability, randomizeOrder)
    {
    }

    /// <summary>
    /// Creates a new SomeOf with a range of augmentations to apply.
    /// </summary>
    /// <param name="minN">The minimum number of augmentations to apply.</param>
    /// <param name="maxN">The maximum number of augmentations to apply.</param>
    /// <param name="augmentations">The augmentations to choose from.</param>
    /// <param name="probability">The probability of applying any augmentation (default 1.0).</param>
    /// <param name="randomizeOrder">Whether to randomize the order of application (default true).</param>
    public SomeOf(
        int minN,
        int maxN,
        IEnumerable<IAugmentation<T, TData>> augmentations,
        double probability = 1.0,
        bool randomizeOrder = true)
    {
        if (probability < 0 || probability > 1)
            throw new ArgumentOutOfRangeException(nameof(probability), "Probability must be between 0 and 1");
        if (minN < 0)
            throw new ArgumentOutOfRangeException(nameof(minN), "MinN must be non-negative");
        if (maxN < minN)
            throw new ArgumentException("MaxN must be >= MinN", nameof(maxN));

        _augmentations = augmentations?.ToList() ?? throw new ArgumentNullException(nameof(augmentations));
        if (_augmentations.Count == 0)
            throw new ArgumentException("At least one augmentation is required", nameof(augmentations));

        MinN = Math.Min(minN, _augmentations.Count);
        MaxN = Math.Min(maxN, _augmentations.Count);
        Probability = probability;
        RandomizeOrder = randomizeOrder;
    }

    /// <summary>
    /// Creates a new SomeOf with a fixed number of augmentations.
    /// </summary>
    /// <param name="n">The number of augmentations to apply.</param>
    /// <param name="augmentations">The augmentations to choose from.</param>
    public SomeOf(int n, params IAugmentation<T, TData>[] augmentations)
        : this(n, n, augmentations.AsEnumerable(), 1.0, true)
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

        // Select N augmentations
        var selected = SelectAugmentations(context);

        // Apply in sequence
        var current = data;
        foreach (var augmentation in selected)
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

        // Check if any augmentation should be applied
        if (Probability < 1.0 && context.GetRandomDouble(0, 1) > Probability)
            return sample;

        // Select N augmentations
        var selected = SelectAugmentations(context);

        // Apply in sequence
        var current = sample;
        foreach (var augmentation in selected)
        {
            if (augmentation is ISpatialAugmentation<T, TData> spatial)
            {
                current = spatial.ApplyWithTargets(current, context);
            }
            else
            {
                current = new AugmentedSample<T, TData>(augmentation.Apply(current.Data, context))
                {
                    BoundingBoxes = current.BoundingBoxes,
                    Keypoints = current.Keypoints,
                    Masks = current.Masks,
                    Labels = current.Labels,
                    Metadata = current.Metadata
                };
            }
        }

        return current;
    }

    private List<IAugmentation<T, TData>> SelectAugmentations(AugmentationContext<T> context)
    {
        // Determine N
        int n = MinN == MaxN ? MinN : context.GetRandomInt(MinN, MaxN + 1);

        // Create list of indices and shuffle
        var indices = Enumerable.Range(0, _augmentations.Count).ToList();

        // Fisher-Yates shuffle
        for (int i = indices.Count - 1; i > 0; i--)
        {
            int j = context.GetRandomInt(0, i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        // Take first N
        var selected = indices.Take(n).Select(i => _augmentations[i]).ToList();

        // Optionally shuffle the order of application
        if (RandomizeOrder && selected.Count > 1)
        {
            for (int i = selected.Count - 1; i > 0; i--)
            {
                int j = context.GetRandomInt(0, i + 1);
                (selected[i], selected[j]) = (selected[j], selected[i]);
            }
        }

        return selected;
    }

    /// <inheritdoc />
    public IDictionary<string, object> GetParameters()
    {
        return new Dictionary<string, object>
        {
            ["probability"] = Probability,
            ["min_n"] = MinN,
            ["max_n"] = MaxN,
            ["randomize_order"] = RandomizeOrder,
            ["num_augmentations"] = _augmentations.Count,
            ["augmentations"] = _augmentations.Select(a => a.Name).ToList()
        };
    }
}
