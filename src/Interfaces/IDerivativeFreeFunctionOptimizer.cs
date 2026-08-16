using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for optimizing a scalar-valued function over a vector of parameters when no
/// gradient is available.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This is the gradient-free counterpart of <see cref="IFunctionOptimizer{T}"/>. Where
/// <see cref="IFunctionOptimizer{T}"/> requires the caller to supply both the objective value
/// and its gradient, this interface requires only the objective value, so it can be used with
/// objectives that are non-differentiable, discontinuous, noisy, or only available as a
/// black-box evaluation (a simulation, a backtest, an external process).
/// </para>
/// <para>
/// Implemented by search-based optimizers such as Nelder-Mead, simulated annealing, particle
/// swarm, genetic algorithms, and CMA-ES.
/// </para>
/// <para><b>For Beginners:</b> Some functions do not have a usable slope. Maybe the function has
/// sharp corners, maybe it is the output of a simulation you cannot differentiate, or maybe it
/// is just a formula somebody handed you as a black box. Gradient-free optimizers work by
/// sampling the function at many points and moving toward whichever points came back best,
/// rather than by following a slope downhill. They typically need many more function
/// evaluations than a gradient method, but they work where gradient methods cannot run at all.
/// </para>
/// </remarks>
public interface IDerivativeFreeFunctionOptimizer<T>
{
    /// <summary>
    /// Minimizes a function starting from the given initial parameters, using only objective
    /// values (no gradients).
    /// </summary>
    /// <param name="initialParameters">Starting point for optimization.</param>
    /// <param name="objective">
    /// Function that computes the objective value at a given point. Takes a parameter vector,
    /// returns the scalar value to minimize.
    /// </param>
    /// <param name="maxIterations">Maximum number of optimization iterations.</param>
    /// <param name="tolerance">
    /// Convergence tolerance. Interpreted by each implementation as the smallest meaningful
    /// spread in objective values (or search-region size) before the search is considered
    /// converged.
    /// </param>
    /// <returns>The optimized parameter vector.</returns>
    Vector<T> Minimize(
        Vector<T> initialParameters,
        Func<Vector<T>, T> objective,
        int maxIterations,
        T tolerance);
}
