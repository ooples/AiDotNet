namespace AiDotNet.Interfaces;

/// <summary>
/// Defines a radial basis function (RBF) that measures similarity based on distance.
/// </summary>
/// <remarks>
/// Radial basis functions are mathematical functions whose value depends only on the distance
/// from a central point. They are commonly used in machine learning for creating complex models
/// from simpler building blocks.
/// 
/// <b>For Beginners:</b> Think of a radial basis function as a "similarity detector" that works like this:
/// - It measures how similar or close two points are to each other
/// - The closer two points are, the higher the output value
/// - The function creates a smooth "hill" or "bump" shape centered at a specific point
/// - As you move away from the center, the function's value decreases
/// 
/// Common examples include the Gaussian (bell curve) function and the multiquadric function.
/// These are used in many AI applications like:
/// - Function approximation (finding patterns in data)
/// - Classification (sorting data into categories)
/// - Time series prediction (forecasting future values)
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
public interface IRadialBasisFunction<T>
{
    /// <summary>
    /// Calculates the value of the radial basis function at a given distance.
    /// </summary>
    /// <remarks>
    /// This is the core function that determines the "shape" of the radial basis function.
    /// 
    /// <b>For Beginners:</b> This method takes a distance value and returns how strong the similarity is.
    /// Think of it like asking: "If I'm this far away from the center, how strong is the connection?"
    /// 
    /// For example, with a Gaussian RBF:
    /// - When r = 0 (directly at the center), the function returns its maximum value (typically 1)
    /// - As r increases (moving away from center), the value smoothly decreases toward 0
    /// - The width parameter (often called sigma or epsilon) controls how quickly the value falls off
    /// </remarks>
    /// <param name="r">The distance from the center point.</param>
    /// <returns>The function value at the given distance.</returns>
    T Compute(T r);

    /// <summary>
    /// Calculates the derivative (rate of change) of the radial basis function with respect to distance.
    /// </summary>
    /// <remarks>
    /// The derivative tells us how quickly the function value changes as the distance changes.
    /// This is important for optimization algorithms that need to adjust the function parameters.
    /// 
    /// <b>For Beginners:</b> This method calculates how quickly the similarity changes as you move
    /// farther from or closer to the center. It's like measuring the steepness of the "hill"
    /// at a particular distance.
    /// 
    /// This is mainly used during the training process when the model needs to adjust itself
    /// to better fit the data. You typically won't need to call this directly unless you're
    /// implementing a custom learning algorithm.
    /// </remarks>
    /// <param name="r">The distance from the center point.</param>
    /// <returns>The derivative value at the given distance.</returns>
    T ComputeDerivative(T r);

    /// <summary>
    /// Calculates the derivative of the radial basis function with respect to its width parameter.
    /// </summary>
    /// <remarks>
    /// This derivative is used when optimizing the width parameter of the radial basis function.
    /// The width parameter controls how quickly the function value decreases as distance increases.
    /// 
    /// <b>For Beginners:</b> The width of an RBF determines how far its influence reaches. This method
    /// helps the learning algorithm figure out whether to make the "hill" wider or narrower.
    /// 
    /// - A wider RBF (larger width) affects points farther away from its center
    /// - A narrower RBF (smaller width) has a more localized effect
    /// 
    /// Like the regular derivative, this is mainly used during the training process and you
    /// typically won't need to call it directly unless implementing custom algorithms.
    /// </remarks>
    /// <param name="r">The distance from the center point.</param>
    /// <returns>The derivative with respect to the width parameter at the given distance.</returns>
    T ComputeWidthDerivative(T r);
}
