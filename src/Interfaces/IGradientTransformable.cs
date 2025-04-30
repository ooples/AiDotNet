namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for models that can transform prediction gradients to parameter gradients.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
public interface IGradientTransformable<T, TInput, TOutput>
{
    /// <summary>
    /// Transforms prediction-level gradients to parameter-level gradients.
    /// </summary>
    /// <param name="input">The input data.</param>
    /// <param name="predictionGradient">The gradient with respect to predictions.</param>
    /// <returns>The gradient with respect to model parameters.</returns>
    Vector<T> TransformGradient(TInput input, Vector<T> predictionGradient);
}