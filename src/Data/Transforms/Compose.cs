namespace AiDotNet.Data.Transforms;

/// <summary>
/// Chains multiple transforms of the same type into a single transform.
/// </summary>
/// <typeparam name="T">The data type that flows through all transforms.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Compose lets you combine multiple transforms into one.
/// Instead of applying each transform separately, you create a pipeline:
/// <code>
/// var pipeline = new Compose&lt;double[]&gt;(
///     new NormalizeTransform&lt;double&gt;(mean, std),
///     new MinMaxScaleTransform&lt;double&gt;(0, 1)
/// );
/// var result = pipeline.Apply(rawData);
/// </code>
/// </para>
/// </remarks>
public class Compose<T> : ITransform<T, T>
{
    private readonly ITransform<T, T>[] _transforms;

    /// <summary>
    /// Creates a composed transform from an ordered list of transforms.
    /// </summary>
    /// <param name="transforms">The transforms to apply in order.</param>
    public Compose(params ITransform<T, T>[] transforms)
    {
        if (transforms is null)
        {
            throw new ArgumentNullException(nameof(transforms));
        }

        _transforms = transforms;
    }

    /// <summary>
    /// Creates a composed transform from an enumerable of transforms.
    /// </summary>
    /// <param name="transforms">The transforms to apply in order.</param>
    public Compose(IEnumerable<ITransform<T, T>> transforms)
    {
        if (transforms is null)
        {
            throw new ArgumentNullException(nameof(transforms));
        }

        _transforms = transforms.ToArray();
    }

    /// <summary>
    /// Gets the number of transforms in this composition.
    /// </summary>
    public int Count => _transforms.Length;

    /// <inheritdoc/>
    public T Apply(T input)
    {
        T current = input;
        for (int i = 0; i < _transforms.Length; i++)
        {
            current = _transforms[i].Apply(current);
        }

        return current;
    }
}
