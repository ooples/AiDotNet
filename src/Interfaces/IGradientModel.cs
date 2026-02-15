namespace AiDotNet.Interfaces;

/// <summary>
/// Represents a gradient for optimization algorithms.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
[AiDotNet.Configuration.YamlConfigurable("GradientModel")]
public interface IGradientModel<T>
{
    /// <summary>
    /// Gets the gradient vector.
    /// </summary>
    Vector<T> Parameters { get; }

    /// <summary>
    /// Evaluates the gradient at a specific point.
    /// </summary>
    /// <param name="input">The input vector.</param>
    /// <returns>The gradient value at the specified point.</returns>
    T Evaluate(Vector<T> input);
}
