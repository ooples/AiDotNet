namespace AiDotNet.Models;

/// <summary>
/// Default implementation of a gradient model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class GradientModel<T> : IGradientModel<T>
{
    /// <summary>
    /// Initializes a new instance of the GradientModel class.
    /// </summary>
    /// <param name="parameters">The gradient vector.</param>
    public GradientModel(Vector<T> parameters)
    {
        Parameters = parameters;
    }

    /// <summary>
    /// Gets the gradient vector.
    /// </summary>
    public Vector<T> Parameters { get; }

    /// <summary>
    /// Evaluates the gradient at a specific point.
    /// </summary>
    /// <param name="input">The input vector.</param>
    /// <returns>The gradient value at the specified point.</returns>
    public T Evaluate(Vector<T> input)
    {
        // Simple dot product for basic gradients
        return input.DotProduct(Parameters);
    }
}
