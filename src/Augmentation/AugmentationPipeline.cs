

namespace AiDotNet.Augmentation;

/// <summary>
/// Specifies how multiple augmentations are applied in a pipeline.
/// </summary>
public enum AugmentationOrder
{
    /// <summary>
    /// Apply augmentations in the order they were added.
    /// </summary>
    Sequential,

    /// <summary>
    /// Randomly shuffle the order of augmentations each time.
    /// </summary>
    Random,

    /// <summary>
    /// Apply only one randomly selected augmentation.
    /// </summary>
    OneOf
}

/// <summary>
/// Represents a pipeline of augmentations that are applied in sequence or composition.
/// </summary>
/// <remarks>
/// <para>
/// AugmentationPipeline allows you to compose multiple augmentations together.
/// Each augmentation in the pipeline is applied with its own probability,
/// and the order can be sequential, random, or one-of.
/// </para>
/// <para><b>For Beginners:</b> Think of this as a recipe of transformations.
/// You might want to first flip an image, then rotate it, then adjust the colors.
/// This pipeline handles all of that automatically.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <typeparam name="TData">The data type being augmented.</typeparam>
public class AugmentationPipeline<T, TData> : IAugmentationPolicy<T, TData>
{
    private readonly List<IAugmentation<T, TData>> _augmentations = new();

    /// <inheritdoc />
    public string Name { get; set; }

    /// <inheritdoc />
    public IList<IAugmentation<T, TData>> Augmentations => _augmentations;

    /// <summary>
    /// Gets or sets the application order for augmentations.
    /// </summary>
    public AugmentationOrder Order { get; set; } = AugmentationOrder.Sequential;

    /// <inheritdoc />
    public int AugmentationCount => _augmentations.Count;

    /// <inheritdoc />
    public IList<string> AugmentationNames => _augmentations.Select(a => a.Name).ToList();

    /// <summary>
    /// Creates a new augmentation pipeline with an optional name.
    /// </summary>
    /// <param name="name">The name of this pipeline.</param>
    public AugmentationPipeline(string? name = null)
    {
        Name = name ?? "AugmentationPipeline";
    }

    /// <summary>
    /// Adds an augmentation to the pipeline.
    /// </summary>
    /// <param name="augmentation">The augmentation to add.</param>
    /// <returns>This pipeline for method chaining.</returns>
    /// <exception cref="ArgumentNullException">Thrown if augmentation is null.</exception>
    public AugmentationPipeline<T, TData> Add(IAugmentation<T, TData> augmentation)
    {
        if (augmentation is null)
        {
            throw new ArgumentNullException(nameof(augmentation));
        }

        _augmentations.Add(augmentation);
        return this;
    }

    /// <summary>
    /// Adds multiple augmentations to the pipeline.
    /// </summary>
    /// <param name="augmentations">The augmentations to add.</param>
    /// <returns>This pipeline for method chaining.</returns>
    /// <exception cref="ArgumentNullException">Thrown if augmentations is null.</exception>
    public AugmentationPipeline<T, TData> AddRange(IEnumerable<IAugmentation<T, TData>> augmentations)
    {
        if (augmentations is null)
        {
            throw new ArgumentNullException(nameof(augmentations));
        }

        _augmentations.AddRange(augmentations);
        return this;
    }

    /// <summary>
    /// Creates a sub-pipeline that applies one-of the specified augmentations.
    /// </summary>
    /// <param name="augmentations">The augmentations to choose from.</param>
    /// <returns>This pipeline for method chaining.</returns>
    public AugmentationPipeline<T, TData> OneOf(params IAugmentation<T, TData>[] augmentations)
    {
        var subPipeline = new AugmentationPipeline<T, TData>("OneOf")
        {
            Order = AugmentationOrder.OneOf
        };
        subPipeline.AddRange(augmentations);
        _augmentations.Add(new PipelineWrapper<T, TData>(subPipeline));
        return this;
    }

    /// <summary>
    /// Creates a sub-pipeline with random ordering.
    /// </summary>
    /// <param name="augmentations">The augmentations to shuffle.</param>
    /// <returns>This pipeline for method chaining.</returns>
    public AugmentationPipeline<T, TData> Shuffle(params IAugmentation<T, TData>[] augmentations)
    {
        var subPipeline = new AugmentationPipeline<T, TData>("Shuffle")
        {
            Order = AugmentationOrder.Random
        };
        subPipeline.AddRange(augmentations);
        _augmentations.Add(new PipelineWrapper<T, TData>(subPipeline));
        return this;
    }

    /// <inheritdoc />
    public TData Apply(TData data, AugmentationContext<T>? context = null)
    {
        context = context ?? new AugmentationContext<T>();

        if (_augmentations.Count == 0)
        {
            return data;
        }

        var result = data;

        switch (Order)
        {
            case AugmentationOrder.Sequential:
                foreach (var augmentation in _augmentations)
                {
                    result = augmentation.Apply(result, context);
                }
                break;

            case AugmentationOrder.Random:
                var shuffled = _augmentations.OrderBy(_ => context.Random.Next()).ToList();
                foreach (var augmentation in shuffled)
                {
                    result = augmentation.Apply(result, context);
                }
                break;

            case AugmentationOrder.OneOf:
                if (_augmentations.Count > 0)
                {
                    int index = context.GetRandomInt(0, _augmentations.Count);
                    result = _augmentations[index].Apply(result, context);
                }
                break;
        }

        return result;
    }

    /// <inheritdoc />
    public IDictionary<string, object> GetConfiguration()
    {
        return new Dictionary<string, object>
        {
            ["name"] = Name,
            ["order"] = Order.ToString(),
            ["augmentations"] = _augmentations.Select(a => a.GetParameters()).ToList()
        };
    }
}

/// <summary>
/// Wraps an augmentation pipeline as a single augmentation for nesting.
/// </summary>
internal class PipelineWrapper<T, TData> : IAugmentation<T, TData>
{
    private readonly AugmentationPipeline<T, TData> _pipeline;

    public string Name => _pipeline.Name;
    public double Probability => 1.0;
    public bool IsTrainingOnly => true;
    public bool IsEnabled { get; set; } = true;

    public PipelineWrapper(AugmentationPipeline<T, TData> pipeline)
    {
        _pipeline = pipeline;
    }

    public TData Apply(TData data, AugmentationContext<T>? context = null)
    {
        if (!IsEnabled) return data;
        return _pipeline.Apply(data, context);
    }

    public IDictionary<string, object> GetParameters()
    {
        return _pipeline.GetConfiguration();
    }
}
