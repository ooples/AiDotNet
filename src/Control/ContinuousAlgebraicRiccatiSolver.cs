using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Control;

/// <summary>
/// Solves the continuous-time algebraic Riccati equation by the matrix sign-function iteration.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Solves <c>AᵀP + PA − PBR⁻¹BᵀP + Q = 0</c> for the stabilizing <c>P</c>, implementing the method of
/// J. D. Roberts, "Linear model reduction and solution of the algebraic Riccati equation by use of
/// the sign function", <i>International Journal of Control</i> 32(4), 1980, pp. 677-687 (circulated
/// as a Cambridge report in 1971), with the scaling of R. Byers, "Solving the algebraic Riccati
/// equation with the matrix sign function", <i>Linear Algebra and its Applications</i> 85, 1987,
/// pp. 267-279.
/// </para>
/// <para>
/// <b>How it works.</b> The solution is encoded in the stable invariant subspace of the Hamiltonian
/// matrix <c>H = [[A, −G], [−Q, −Aᵀ]]</c> where <c>G = BR⁻¹Bᵀ</c>. The matrix sign function maps every
/// eigenvalue with negative real part to <c>−1</c> and every one with positive real part to
/// <c>+1</c>, leaving the eigenvectors untouched — so <c>sign(H) + I</c> annihilates exactly the
/// stable subspace. Reading <c>P</c> off is then a least-squares solve. The sign function itself
/// comes from Newton's iteration on <c>Z² = I</c>, namely <c>Z ← ½(Z + Z⁻¹)</c>, which converges
/// quadratically and needs nothing but matrix inversion.
/// </para>
/// <para>
/// <b>Why not eigenvectors directly.</b> The classical route extracts the stable subspace from an
/// eigendecomposition of <c>H</c>. But <c>H</c> is not symmetric, and its eigenvalues are generally
/// complex — so that route needs complex arithmetic, or an ordered real Schur factorization, to be
/// correct. The sign-function iteration stays in real arithmetic throughout and uses only the LU
/// factorization already in this library.
/// </para>
/// <para>
/// The iteration converges precisely when <c>H</c> has no eigenvalues on the imaginary axis, which is
/// the same condition under which a stabilizing solution exists at all — so the method fails exactly
/// when, and only when, there is nothing to find.
/// </para>
/// <para><b>For Beginners:</b> The Hamiltonian matrix packs the dynamics and the costs into one
/// object whose "stable half" is the answer. The sign function is a way of separating that stable
/// half from the unstable one using only repeated matrix inversion — no eigenvalues are ever
/// computed.
/// </para>
/// </remarks>
public sealed class ContinuousAlgebraicRiccatiSolver<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly AlgebraicRiccatiSolverOptions _options;

    /// <summary>
    /// Creates a continuous-time Riccati solver with the default options.
    /// </summary>
    public ContinuousAlgebraicRiccatiSolver()
        : this(new AlgebraicRiccatiSolverOptions())
    {
    }

    /// <summary>
    /// Creates a continuous-time Riccati solver.
    /// </summary>
    /// <param name="options">Solver configuration.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="options"/> is null.</exception>
    /// <exception cref="ArgumentException">
    /// Thrown when the iteration limit or tolerance is not positive.
    /// </exception>
    public ContinuousAlgebraicRiccatiSolver(AlgebraicRiccatiSolverOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));

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
    /// Solves the continuous-time algebraic Riccati equation.
    /// </summary>
    /// <param name="stateMatrix">The state matrix <c>A</c>, <c>n</c>-by-<c>n</c>.</param>
    /// <param name="inputMatrix">The input matrix <c>B</c>, <c>n</c>-by-<c>m</c>.</param>
    /// <param name="stateCost">The state cost <c>Q</c>, symmetric positive-semidefinite.</param>
    /// <param name="inputCost">The input cost <c>R</c>, symmetric positive-definite.</param>
    /// <returns>The stabilizing solution, with its residual.</returns>
    /// <exception cref="ArgumentNullException">Thrown when any argument is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the dimensions are inconsistent.</exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when <c>R</c> is singular, or when the Hamiltonian has an eigenvalue on the imaginary
    /// axis — in which case no stabilizing solution exists.
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

        var g = ControlMath<T>.Multiply(
            ControlMath<T>.Multiply(inputMatrix, inputCostInverse),
            ControlMath<T>.Transpose(inputMatrix));

        var hamiltonian = BuildHamiltonian(stateMatrix, g, stateCost, n);

        var (sign, converged, iterations) = ComputeSign(hamiltonian);

        var solution = ExtractSolution(sign, n);
        T residual = Residual(solution, stateMatrix, g, stateCost);

        return new AlgebraicRiccatiSolution<T>(solution, residual, converged, iterations);
    }

    /// <summary>
    /// Computes the optimal feedback gain <c>K = R⁻¹BᵀP</c>, so the optimal input is <c>u = −Kx</c>.
    /// </summary>
    /// <param name="solution">The Riccati solution <c>P</c>.</param>
    /// <param name="inputMatrix">The input matrix <c>B</c>.</param>
    /// <param name="inputCost">The input cost <c>R</c>.</param>
    /// <returns>The gain matrix <c>K</c>.</returns>
    /// <exception cref="InvalidOperationException">Thrown when <c>R</c> is singular.</exception>
    public static Matrix<T> ComputeGain(
        Matrix<T> solution, Matrix<T> inputMatrix, Matrix<T> inputCost)
    {
        var inputCostInverse = ControlMath<T>.TryInvert(inputCost)
            ?? throw new InvalidOperationException("The input cost matrix R is singular.");

        return ControlMath<T>.Multiply(
            ControlMath<T>.Multiply(inputCostInverse, ControlMath<T>.Transpose(inputMatrix)),
            solution);
    }

    /// <summary>
    /// Assembles <c>H = [[A, −G], [−Q, −Aᵀ]]</c>.
    /// </summary>
    private static Matrix<T> BuildHamiltonian(
        Matrix<T> stateMatrix, Matrix<T> g, Matrix<T> stateCost, int n)
    {
        var hamiltonian = new Matrix<T>(2 * n, 2 * n);

        for (int r = 0; r < n; r++)
        {
            for (int c = 0; c < n; c++)
            {
                hamiltonian[r, c] = stateMatrix[r, c];
                hamiltonian[r, n + c] = NumOps.Negate(g[r, c]);
                hamiltonian[n + r, c] = NumOps.Negate(stateCost[r, c]);
                hamiltonian[n + r, n + c] = NumOps.Negate(stateMatrix[c, r]);
            }
        }

        return hamiltonian;
    }

    /// <summary>
    /// Runs Newton's iteration <c>Z ← ½(Z + Z⁻¹)</c> for the matrix sign function.
    /// </summary>
    private (Matrix<T> Sign, bool Converged, int Iterations) ComputeSign(Matrix<T> hamiltonian)
    {
        var current = hamiltonian;
        bool converged = false;
        int iteration;

        for (iteration = 1; iteration <= _options.MaxIterations; iteration++)
        {
            var inverse = ControlMath<T>.TryInvert(current)
                ?? throw new InvalidOperationException(
                    "The Hamiltonian matrix is singular, which places an eigenvalue on the " +
                    "imaginary axis. The continuous-time Riccati equation then has no stabilizing " +
                    "solution — check that (A, B) is stabilizable and that Q is positive " +
                    "semidefinite.");

            double scale = 1.0;
            if (_options.UseSignFunctionScaling)
            {
                double currentNorm = ControlMath<T>.FrobeniusNorm(current);
                double inverseNorm = ControlMath<T>.FrobeniusNorm(inverse);

                if (currentNorm > 0.0 && inverseNorm > 0.0)
                {
                    scale = Math.Sqrt(inverseNorm / currentNorm);
                }
            }

            var next = ControlMath<T>.Scale(
                ControlMath<T>.Add(
                    ControlMath<T>.Scale(current, scale),
                    ControlMath<T>.Scale(inverse, 1.0 / scale)),
                0.5);

            double change = ControlMath<T>.Distance(next, current);
            double magnitude = 1.0 + ControlMath<T>.FrobeniusNorm(next);

            current = next;

            if (change / magnitude <= _options.Tolerance)
            {
                converged = true;
                break;
            }
        }

        return (current, converged, Math.Min(iteration, _options.MaxIterations));
    }

    /// <summary>
    /// Reads <c>P</c> out of the sign matrix.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The stable invariant subspace is the null space of <c>M = sign(H) + I</c>. Writing a basis for
    /// it as <c>[X₁; X₂]</c> with <c>P = X₂X₁⁻¹</c> and splitting <c>M</c> into <c>n</c>-by-<c>n</c>
    /// blocks turns <c>M[X₁; X₂] = 0</c> into the overdetermined system
    /// <c>[M₁₂; M₂₂]P = −[M₁₁; M₂₁]</c>, which is solved in the least-squares sense through its
    /// normal equations. Using both block rows rather than either alone is what makes this stable:
    /// one of them can be rank-deficient on its own.
    /// </para>
    /// </remarks>
    private static Matrix<T> ExtractSolution(Matrix<T> sign, int n)
    {
        var shifted = ControlMath<T>.Add(sign, Matrix<T>.CreateIdentity(2 * n));

        var upperLeft = ControlMath<T>.Block(shifted, 0, 0, n);
        var upperRight = ControlMath<T>.Block(shifted, 0, n, n);
        var lowerLeft = ControlMath<T>.Block(shifted, n, 0, n);
        var lowerRight = ControlMath<T>.Block(shifted, n, n, n);

        var upperRightTransposed = ControlMath<T>.Transpose(upperRight);
        var lowerRightTransposed = ControlMath<T>.Transpose(lowerRight);

        // (M₁₂ᵀM₁₂ + M₂₂ᵀM₂₂) P = −(M₁₂ᵀM₁₁ + M₂₂ᵀM₂₁)
        var normalMatrix = ControlMath<T>.Add(
            ControlMath<T>.Multiply(upperRightTransposed, upperRight),
            ControlMath<T>.Multiply(lowerRightTransposed, lowerRight));

        var rightHandSide = ControlMath<T>.Scale(
            ControlMath<T>.Add(
                ControlMath<T>.Multiply(upperRightTransposed, upperLeft),
                ControlMath<T>.Multiply(lowerRightTransposed, lowerLeft)),
            -1.0);

        var factored = new LuDecomposition<T>(normalMatrix);

        var solution = new Matrix<T>(n, n);
        for (int c = 0; c < n; c++)
        {
            var column = new Vector<T>(n);
            for (int r = 0; r < n; r++) column[r] = rightHandSide[r, c];

            var solved = factored.Solve(column);
            for (int r = 0; r < n; r++) solution[r, c] = solved[r];
        }

        return ControlMath<T>.Symmetrize(solution);
    }

    /// <summary>
    /// Substitutes a candidate back into <c>AᵀP + PA − PGP + Q</c> and measures what is left over.
    /// </summary>
    private static T Residual(
        Matrix<T> candidate, Matrix<T> stateMatrix, Matrix<T> g, Matrix<T> stateCost)
    {
        var stateTransposed = ControlMath<T>.Transpose(stateMatrix);

        var accumulated = ControlMath<T>.Add(
            ControlMath<T>.Multiply(stateTransposed, candidate),
            ControlMath<T>.Multiply(candidate, stateMatrix));

        accumulated = ControlMath<T>.Add(accumulated, stateCost);

        accumulated = ControlMath<T>.Subtract(
            accumulated,
            ControlMath<T>.Multiply(ControlMath<T>.Multiply(candidate, g), candidate));

        return NumOps.FromDouble(ControlMath<T>.FrobeniusNorm(accumulated));
    }
}
