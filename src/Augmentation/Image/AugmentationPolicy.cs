namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Defines a serializable augmentation policy with named transforms and parameters.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class AugmentationPolicy<T>
{
    public string Name { get; set; } = "custom";
    public List<AugmentationPolicyEntry<T>> Entries { get; } = new();

    /// <summary>
    /// Adds a transform to the policy.
    /// </summary>
    public AugmentationPolicy<T> Add(IAugmentation<T, ImageTensor<T>> augmenter, double probability = 1.0)
    {
        Entries.Add(new AugmentationPolicyEntry<T>(augmenter, probability));
        return this;
    }

    /// <summary>
    /// Applies the policy to an image.
    /// </summary>
    public ImageTensor<T> Apply(ImageTensor<T> image, AugmentationContext<T> context)
    {
        var result = image;
        foreach (var entry in Entries)
        {
            if (context.GetRandomDouble(0, 1) < entry.Probability)
                result = entry.Augmenter.Apply(result, context);
        }
        return result;
    }

    /// <summary>
    /// Gets all parameters as a serializable dictionary.
    /// </summary>
    public IDictionary<string, object> GetParameters()
    {
        var p = new Dictionary<string, object>();
        p["name"] = Name;
        p["num_entries"] = Entries.Count;
        for (int i = 0; i < Entries.Count; i++)
        {
            p[$"entry_{i}_type"] = Entries[i].Augmenter.GetType().Name;
            p[$"entry_{i}_probability"] = Entries[i].Probability;
            foreach (var kv in Entries[i].Augmenter.GetParameters())
                p[$"entry_{i}_{kv.Key}"] = kv.Value;
        }
        return p;
    }
}

/// <summary>
/// An entry in an augmentation policy.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class AugmentationPolicyEntry<T>
{
    public IAugmentation<T, ImageTensor<T>> Augmenter { get; }
    public double Probability { get; }

    public AugmentationPolicyEntry(IAugmentation<T, ImageTensor<T>> augmenter, double probability)
    {
        Augmenter = augmenter; Probability = probability;
    }
}
