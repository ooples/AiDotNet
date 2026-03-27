using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for optimizing a scalar-valued function over a vector of parameters.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> A function optimizer finds the parameter values that minimize
/// a given function. For example, finding the weights that minimize a loss function.
/// Different optimizers use different strategies: gradient descent follows the slope
/// downhill, L-BFGS approximates the curvature for faster convergence, etc.</para>
///
/// <para>This interface abstracts the optimization strategy so algorithms can swap
/// optimizers without changing their structure. For example, NOTEARS can use L-BFGS
/// (default) or gradient descent for experimentation.</para>
/// </remarks>
public interface IFunctionOptimizer<T>
{
    /// <summary>
    /// Minimizes a function starting from the given initial parameters.
    /// </summary>
    /// <param name="initialParameters">Starting point for optimization.</param>
    /// <param name="objectiveAndGradient">
    /// Function that computes both the objective value and gradient at a given point.
    /// Takes a parameter vector, returns (objective value, gradient vector).
    /// </param>
    /// <param name="maxIterations">Maximum number of optimization iterations.</param>
    /// <param name="tolerance">Convergence tolerance on gradient norm.</param>
    /// <returns>The optimized parameter vector.</returns>
    Vector<T> Minimize(
        Vector<T> initialParameters,
        Func<Vector<T>, (T objective, Vector<T> gradient)> objectiveAndGradient,
        int maxIterations,
        T tolerance);
}
