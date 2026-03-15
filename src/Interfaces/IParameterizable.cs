namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for models that have optimizable parameters.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[AiDotNet.Configuration.YamlConfigurable("Parameterizable")]
public interface IParameterizable<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the parameters that can be optimized.
    /// </summary>
    Vector<T> GetParameters();

    /// <summary>
    /// Sets the model parameters.
    /// </summary>
    /// <param name="parameters">The parameter vector to set.</param>
    /// <remarks>
    /// This method allows direct modification of the model's internal parameters.
    /// This is useful for optimization algorithms that need to update parameters iteratively.
    /// If the length of <paramref name="parameters"/> does not match <see cref="ParameterCount"/>,
    /// an <see cref="ArgumentException"/> should be thrown.
    /// </remarks>
    /// <exception cref="ArgumentException">
    /// Thrown when the length of <paramref name="parameters"/> does not match <see cref="ParameterCount"/>.
    /// </exception>
    void SetParameters(Vector<T> parameters);

    /// <summary>
    /// Gets the number of parameters in the model.
    /// </summary>
    /// <remarks>
    /// This property returns the total count of trainable parameters in the model.
    /// It's useful for understanding model complexity and memory requirements.
    /// </remarks>
    int ParameterCount { get; }

    /// <summary>
    /// Gets whether this model supports direct parameter-based initialization.
    /// </summary>
    /// <remarks>
    /// Models that learn their structure during training (decision trees, ensemble methods, clustering)
    /// may not support having random parameters injected before training. The optimizer uses this
    /// property to decide whether to call <see cref="SetParameters"/> during random initialization.
    /// The default implementation returns <c>true</c> when <see cref="ParameterCount"/> is greater than zero.
    /// </remarks>
    bool SupportsParameterInitialization => ParameterCount > 0;

    /// <summary>
    /// Creates a new instance with the specified parameters.
    /// </summary>
    IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters);
}
