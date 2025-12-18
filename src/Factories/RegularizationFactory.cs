namespace AiDotNet.Factories;

/// <summary>
/// A factory class that creates regularization components for machine learning models.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Regularization is a technique used to prevent overfitting in machine learning models. 
/// Overfitting happens when a model learns the training data too well, including its noise and outliers, 
/// making it perform poorly on new, unseen data.
/// </para>
/// <para>
/// Think of regularization like adding training wheels to a bicycle - it constrains the model to keep it 
/// from becoming too complex and "memorizing" the training data instead of learning general patterns.
/// </para>
/// </remarks>
public static class RegularizationFactory
{
    /// <summary>
    /// Creates a regularization component based on the specified options.
    /// </summary>
    /// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
    /// <param name="options">Configuration options for the regularization.</param>
    /// <returns>An implementation of IRegularization<T> for the specified regularization type.</returns>
    /// <exception cref="ArgumentException">Thrown when an unknown regularization type is specified.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Different regularization types apply different kinds of constraints to your model. 
    /// The right choice depends on your specific problem and data characteristics.
    /// </para>
    /// <para>
    /// Available regularization types include:
    /// <list type="bullet">
    /// <item><description>None: No regularization is applied. The model is free to fit the training data as closely as possible.</description></item>
    /// <item><description>L1 (Lasso): Adds a penalty proportional to the absolute value of the model's coefficients. This can reduce some coefficients to exactly zero, effectively removing certain features from the model.</description></item>
    /// <item><description>L2 (Ridge): Adds a penalty proportional to the square of the model's coefficients. This shrinks all coefficients toward zero but rarely makes them exactly zero.</description></item>
    /// <item><description>ElasticNet: Combines L1 and L2 regularization, giving you the benefits of both approaches.</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    public static IRegularization<T, TInput, TOutput> CreateRegularization<T, TInput, TOutput>(RegularizationOptions options)
    {
        return options.Type switch
        {
            RegularizationType.None => new NoRegularization<T, TInput, TOutput>(),
            RegularizationType.L1 => new L1Regularization<T, TInput, TOutput>(options),
            RegularizationType.L2 => new L2Regularization<T, TInput, TOutput>(options),
            RegularizationType.ElasticNet => new ElasticNetRegularization<T, TInput, TOutput>(options),
            _ => throw new ArgumentException($"Unknown regularization type: {options.Type}", nameof(options.Type))
        };
    }

    /// <summary>
    /// Determines the regularization type from an existing regularization component.
    /// </summary>
    /// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
    /// <param name="regularization">The regularization component to identify.</param>
    /// <returns>The type of the provided regularization component.</returns>
    /// <exception cref="ArgumentException">Thrown when an unsupported regularization type is provided.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method examines a regularization object and tells you what type it is. 
    /// It's like looking at a tool and identifying whether it's a hammer, screwdriver, or wrench.
    /// </para>
    /// <para>
    /// This is useful when you have a regularization object but don't know its specific type, such as 
    /// when saving or loading models, or when working with models created elsewhere in your code.
    /// </para>
    /// </remarks>
    public static RegularizationType GetRegularizationType<T, TInput, TOutput>(IRegularization<T, TInput, TOutput> regularization)
    {
        return regularization switch
        {
            NoRegularization<T, TInput, TOutput> => RegularizationType.None,
            L1Regularization<T, TInput, TOutput> => RegularizationType.L1,
            L2Regularization<T, TInput, TOutput> => RegularizationType.L2,
            ElasticNetRegularization<T, TInput, TOutput> => RegularizationType.ElasticNet,
            _ => throw new ArgumentException($"Unsupported regularization type: {regularization.GetType().Name}")
        };
    }
}
