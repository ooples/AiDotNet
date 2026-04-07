namespace AiDotNet.Helpers;

/// <summary>
/// Provides safe runtime capability checks for interface segregation.
/// After removing IParameterizable, IFeatureAware, IGradientComputable, and IJitCompilable
/// from IFullModel, callers must validate capabilities before use. These methods provide
/// clear error messages when a model doesn't support the requested capability.
/// </summary>
public static class InterfaceGuard
{
    /// <summary>
    /// Returns the model as IParameterizable or throws with a clear message.
    /// </summary>
    public static IParameterizable<T, TInput, TOutput> Parameterizable<T, TInput, TOutput>(
        IFullModel<T, TInput, TOutput> model)
    {
        if (model is IParameterizable<T, TInput, TOutput> p)
            return p;
        throw new InvalidOperationException(
            $"{model.GetType().Name} does not implement IParameterizable<{typeof(T).Name}, {typeof(TInput).Name}, {typeof(TOutput).Name}>. " +
            "This operation requires a model with trainable parameters.");
    }

    /// <summary>
    /// Returns the model as IGradientComputable or throws with a clear message.
    /// </summary>
    public static IGradientComputable<T, TInput, TOutput> GradientComputable<T, TInput, TOutput>(
        IFullModel<T, TInput, TOutput> model)
    {
        if (model is IGradientComputable<T, TInput, TOutput> g)
            return g;
        throw new InvalidOperationException(
            $"{model.GetType().Name} does not implement IGradientComputable. " +
            "This operation requires a model that supports gradient computation.");
    }

    /// <summary>
    /// Returns the model as IFeatureAware or throws with a clear message.
    /// </summary>
    public static IFeatureAware FeatureAware<T, TInput, TOutput>(
        IFullModel<T, TInput, TOutput> model)
    {
        if (model is IFeatureAware f)
            return f;
        throw new InvalidOperationException(
            $"{model.GetType().Name} does not implement IFeatureAware.");
    }

    /// <summary>
    /// Returns the model as IParameterizable if supported, null otherwise.
    /// </summary>
    public static IParameterizable<T, TInput, TOutput>? TryParameterizable<T, TInput, TOutput>(
        IFullModel<T, TInput, TOutput>? model)
        => model as IParameterizable<T, TInput, TOutput>;

    /// <summary>
    /// Returns the model as IGradientComputable if supported, null otherwise.
    /// </summary>
    public static IGradientComputable<T, TInput, TOutput>? TryGradientComputable<T, TInput, TOutput>(
        IFullModel<T, TInput, TOutput> model)
        => model as IGradientComputable<T, TInput, TOutput>;

    /// <summary>
    /// Returns the model as IFeatureAware if supported, null otherwise.
    /// </summary>
    public static IFeatureAware? TryFeatureAware(object model)
        => model as IFeatureAware;
}
