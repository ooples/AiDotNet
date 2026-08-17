using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Control;

/// <summary>
/// Solves the discrete-time algebraic Riccati equation by the structure-preserving doubling
/// algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Solves <c>P = AᵀPA − AᵀPB(R + BᵀPB)⁻¹BᵀPA + Q</c> for the stabilizing <c>P</c>, implementing the
/// doubling algorithm of B. D. O. Anderson, "Second-order convergent algorithms for the steady-state
/// Riccati equation", <i>International Journal of Control</i> 28(2), 1978, pp. 295-306, in the
/// structure-preserving form of E. K.-W. Chu, H.-Y. Fan, W.-W. Lin and C.-S. Wang, "Structure-
/// preserving algorithms for periodic discrete-time algebraic Riccati equations",
/// <i>International Journal of Control</i> 77(8), 2004, pp. 767-788.
/// </para>
/// <para>
/// <b>Why doubling rather than the obvious iteration.</b> The direct approach is to iterate the
/// Riccati difference equation, substituting the current <c>P</c> into the right-hand side to get
/// the next one. It converges, but only linearly: each step buys a fixed fraction of the remaining
/// error, so reaching machine precision can take hundreds of steps. The doubling algorithm advances
/// the equivalent of <c>2ᵏ</c> of those steps on iteration <c>k</c>, which makes the convergence
/// quadratic — the number of correct digits roughly doubles each time — at the cost of one extra
/// matrix inverse per step. Twenty iterations is a great many.
/// </para>
/// <para>
/// It also needs no eigenvalue or Schur decomposition, which matters here: the alternative textbook
/// method builds a symplectic matrix and extracts its stable invariant subspace, requiring an
/// <i>ordered</i> Schur factorization that is considerably more machinery to get right.
/// </para>
/// <para><b>For Beginners:</b> The Riccati equation asks for a cost surface that is consistent with
/// itself — the cost of being somewhere must equal the cost of the best move plus the cost of
/// wherever that move lands you. This solves that self-consistency condition. The answer is what
/// tells a controller how expensive each situation really is, counting all future consequences.
/// </para>
/// </remarks>
public sealed class DiscreteAlgebraicRiccatiSolver<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly AlgebraicRiccatiSolverOptions _options;

    /// <summary>
    /// Creates a discrete-time Riccati solver with the default options.
    /// </summary>
    public DiscreteAlgebraicRiccatiSolver()
        : this(new AlgebraicRiccatiSolverOptions())
    {
    }

    /// <summary>
    /// Creates a discrete-time Riccati solver.
    /// </summary>
    /// <param name="options">Solver configuration.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="options"/> is null.</exception>
    /// <exception cref="ArgumentException">
    /// Thrown when the iteration limit or tolerance is not positive.
    /// </exception>
    public DiscreteAlgebraicRiccatiSolver(AlgebraicRiccatiSolverOptions options)
    {
        if (options is null) throw new ArgumentNullException(nameof(options));
        _options = new AlgebraicRiccatiSolverOptions(options);

        if (options.MaxIterations <= 0)
        {
            throw new ArgumentException("MaxIterations must be positive.", nameof(options));
        }

        if (options.Tolerance <= 0.0)
        {
            throw new ArgumentException("Tolerance must be positive.", nameof(options));
        }
    }

    /// <summary>
    /// Solves the discrete-time algebraic Riccati equation.
    /// </summary>
    /// <param name="stateMatrix">The state matrix <c>A</c>, <c>n</c>-by-<c>n</c>.</param>
    /// <param name="inputMatrix">The input matrix <c>B</c>, <c>n</c>-by-<c>m</c>.</param>
    /// <param name="stateCost">The state cost <c>Q</c>, symmetric positive-semidefinite.</param>
    /// <param name="inputCost">The input cost <c>R</c>, symmetric positive-definite.</param>
    /// <returns>The stabilizing solution, with its residual.</returns>
    /// <exception cref="ArgumentNullException">Thrown when any argument is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the dimensions are inconsistent.</exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when <c>R</c> is singular, or when the iteration encounters a singular matrix — which
    /// indicates the problem does not have a stabilizing solution rather than a defect in the
    /// arithmetic.
    /// </exception>
    public AlgebraicRiccatiSolution<T> Solve(
        Matrix<T> stateMatrix, Matrix<T> inputMatrix, Matrix<T> stateCost, Matrix<T> inputCost)
    {
        RiccatiValidation<T>.Validate(stateMatrix, inputMatrix, stateCost, inputCost);

        int n = stateMatrix.Rows;

        var inputCostInverse = ControlMath<T>.TryInvert(inputCost)
            ?? throw new InvalidOperationException(
                "The input cost matrix R is singular. R must be positive definite: a direction of " +
                "zero input cost would let the controller apply unbounded effort for free, and the " +
                "problem has no finite optimum.");

        // G = B R⁻¹ Bᵀ, the map from costate to state increment.
        var g = ControlMath<T>.Multiply(
            ControlMath<T>.Multiply(inputMatrix, inputCostInverse),
            ControlMath<T>.Transpose(inputMatrix));

        var a = stateMatrix;
        var h = stateCost;
        var identity = Matrix<T>.CreateIdentity(n);

        bool converged = false;
        int iteration = 0;

        for (iteration = 1; iteration <= _options.MaxIterations; iteration++)
        {
            // W = I + G H, whose inverse is the only factorization the step needs.
            var w = ControlMath<T>.Add(identity, ControlMath<T>.Multiply(g, h));

            var wInverse = ControlMath<T>.TryInvert(w)
                ?? throw new InvalidOperationException(
                    "The doubling iteration met a singular matrix. This means the pair (A, B) is " +
                    "not stabilizable, so no feedback can hold the state bounded and the Riccati " +
                    "equation has no stabilizing solution.");

            var aTimesWInverse = ControlMath<T>.Multiply(a, wInverse);

            var nextA = ControlMath<T>.Multiply(aTimesWInverse, a);

            var nextG = ControlMath<T>.Add(
                g,
                ControlMath<T>.Multiply(
                    ControlMath<T>.Multiply(aTimesWInverse, g), ControlMath<T>.Transpose(a)));

            var nextH = ControlMath<T>.Add(
                h,
                ControlMath<T>.Multiply(
                    ControlMath<T>.Multiply(ControlMath<T>.Transpose(a), h),
                    ControlMath<T>.Multiply(wInverse, a)));

            nextG = ControlMath<T>.Symmetrize(nextG);
            nextH = ControlMath<T>.Symmetrize(nextH);

            double change = ControlMath<T>.Distance(nextH, h);
            double scale = 1.0 + ControlMath<T>.FrobeniusNorm(nextH);

            a = nextA;
            g = nextG;
            h = nextH;

            if (change / scale <= _options.Tolerance)
            {
                converged = true;
                break;
            }
        }

        T residual = Residual(h, stateMatrix, inputMatrix, stateCost, inputCost);

        return new AlgebraicRiccatiSolution<T>(
            h, residual, converged, Math.Min(iteration, _options.MaxIterations));
    }

    /// <summary>
    /// Computes the optimal feedback gain <c>K = (R + BᵀPB)⁻¹BᵀPA</c>, so the optimal input is
    /// <c>u = −Kx</c>.
    /// </summary>
    /// <param name="solution">The Riccati solution <c>P</c>.</param>
    /// <param name="stateMatrix">The state matrix <c>A</c>.</param>
    /// <param name="inputMatrix">The input matrix <c>B</c>.</param>
    /// <param name="inputCost">The input cost <c>R</c>.</param>
    /// <returns>The gain matrix <c>K</c>.</returns>
    /// <exception cref="InvalidOperationException">
    /// Thrown when <c>R + BᵀPB</c> is singular.
    /// </exception>
    public static Matrix<T> ComputeGain(
        Matrix<T> solution, Matrix<T> stateMatrix, Matrix<T> inputMatrix, Matrix<T> inputCost)
    {
        var inputTransposed = ControlMath<T>.Transpose(inputMatrix);
        var solutionTimesInput = ControlMath<T>.Multiply(solution, inputMatrix);

        var middle = ControlMath<T>.Add(
            inputCost, ControlMath<T>.Multiply(inputTransposed, solutionTimesInput));

        var middleInverse = ControlMath<T>.TryInvert(middle)
            ?? throw new InvalidOperationException(
                "R + BᵀPB is singular, so no finite optimal gain exists.");

        return ControlMath<T>.Multiply(
            ControlMath<T>.Multiply(middleInverse, inputTransposed),
            ControlMath<T>.Multiply(solution, stateMatrix));
    }

    /// <summary>
    /// Substitutes a candidate back into the Riccati equation and returns the Frobenius norm of what
    /// is left over.
    /// </summary>
    private static T Residual(
        Matrix<T> candidate,
        Matrix<T> stateMatrix,
        Matrix<T> inputMatrix,
        Matrix<T> stateCost,
        Matrix<T> inputCost)
    {
        var stateTransposed = ControlMath<T>.Transpose(stateMatrix);
        var inputTransposed = ControlMath<T>.Transpose(inputMatrix);

        var candidateTimesState = ControlMath<T>.Multiply(candidate, stateMatrix);
        var candidateTimesInput = ControlMath<T>.Multiply(candidate, inputMatrix);

        // AᵀPA + Q − P
        var accumulated = ControlMath<T>.Subtract(
            ControlMath<T>.Add(
                ControlMath<T>.Multiply(stateTransposed, candidateTimesState), stateCost),
            candidate);

        // − AᵀPB (R + BᵀPB)⁻¹ BᵀPA
        var middle = ControlMath<T>.Add(
            inputCost, ControlMath<T>.Multiply(inputTransposed, candidateTimesInput));

        var middleInverse = ControlMath<T>.TryInvert(middle);
        if (middleInverse is null) return NumOps.FromDouble(double.PositiveInfinity);

        var correction = ControlMath<T>.Multiply(
            ControlMath<T>.Multiply(stateTransposed, candidateTimesInput),
            ControlMath<T>.Multiply(
                middleInverse, ControlMath<T>.Multiply(inputTransposed, candidateTimesState)));

        return NumOps.FromDouble(
            ControlMath<T>.FrobeniusNorm(ControlMath<T>.Subtract(accumulated, correction)));
    }
}
