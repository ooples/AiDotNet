namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Registry for built-in and custom augmentation policies.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public static class PolicyRegistry<T>
{
    private static readonly Dictionary<string, Func<AugmentationPolicy<T>>> _policies = new();

    static PolicyRegistry()
    {
        Register("light_augmentation", CreateLightPolicy);
        Register("medium_augmentation", CreateMediumPolicy);
        Register("heavy_augmentation", CreateHeavyPolicy);
    }

    /// <summary>
    /// Registers a named policy factory.
    /// </summary>
    public static void Register(string name, Func<AugmentationPolicy<T>> factory)
    {
        _policies[name] = factory;
    }

    /// <summary>
    /// Gets a policy by name.
    /// </summary>
    public static AugmentationPolicy<T> Get(string name)
    {
        if (!_policies.TryGetValue(name, out var factory))
            throw new KeyNotFoundException($"Policy '{name}' not found in registry");
        return factory();
    }

    /// <summary>
    /// Gets all registered policy names.
    /// </summary>
    public static IEnumerable<string> GetNames() => _policies.Keys;

    private static AugmentationPolicy<T> CreateLightPolicy()
    {
        var policy = new AugmentationPolicy<T> { Name = "light_augmentation" };
        policy.Add(new HorizontalFlip<T>(), 0.5);
        policy.Add(new Brightness<T>(0.8, 1.2), 0.3);
        policy.Add(new Contrast<T>(0.8, 1.2), 0.3);
        return policy;
    }

    private static AugmentationPolicy<T> CreateMediumPolicy()
    {
        var policy = new AugmentationPolicy<T> { Name = "medium_augmentation" };
        policy.Add(new HorizontalFlip<T>(), 0.5);
        policy.Add(new Rotation<T>(-15, 15), 0.3);
        policy.Add(new Brightness<T>(0.7, 1.3), 0.4);
        policy.Add(new Contrast<T>(0.7, 1.3), 0.4);
        policy.Add(new GaussianBlur<T>(), 0.2);
        policy.Add(new CoarseDropout<T>(), 0.2);
        return policy;
    }

    private static AugmentationPolicy<T> CreateHeavyPolicy()
    {
        var policy = new AugmentationPolicy<T> { Name = "heavy_augmentation" };
        policy.Add(new HorizontalFlip<T>(), 0.5);
        policy.Add(new Rotation<T>(-30, 30), 0.5);
        policy.Add(new Scale<T>(0.8, 1.2), 0.3);
        policy.Add(new Brightness<T>(0.5, 1.5), 0.5);
        policy.Add(new Contrast<T>(0.5, 1.5), 0.5);
        policy.Add(new Saturation<T>(0.5, 1.5), 0.4);
        policy.Add(new GaussianBlur<T>(), 0.3);
        policy.Add(new GaussianNoise<T>(), 0.3);
        policy.Add(new CoarseDropout<T>(), 0.3);
        policy.Add(new ElasticTransform<T>(), 0.2);
        return policy;
    }
}
