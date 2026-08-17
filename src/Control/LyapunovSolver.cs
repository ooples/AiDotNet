using AiDotNet.DecompositionMethods.MatrixDecomposition;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Control;

/// <summary>
/// Solves the continuous and discrete Lyapunov equations exactly.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Solves <c>AᵀP + PA + Q = 0</c> (continuous) and <c>AᵀPA − P + Q = 0</c> (discrete) for the
/// symmetric <c>P</c>. Both are linear in <c>P</c>, so unlike the Riccati equations these need no
/// iteration at all: rewriting the unknown matrix as a vector turns each into an ordinary linear
/// system that a single factorization solves.
/// </para>
/// <para>
/// <b>What the answer means.</b> Lyapunov's theorem: the system is stable exactly when, for every
/// positive-definite <c>Q</c>, the solution <c>P</c> comes back positive definite. That gives a
/// stability test requiring no eigenvalues — and more usefully, <c>xᵀPx</c> is then an explicit
/// energy-like quantity that decreases along every trajectory, which is a certificate of stability
/// rather than merely a verdict.
/// </para>
/// <para>
/// <b>Where else it turns up.</b> With <c>Q = BBᵀ</c> the solution is the controllability Gramian,
/// measuring how easily each direction of the state space can be driven; with <c>Q = CᵀC</c> it is
/// the observability Gramian, measuring how visibly each direction shows up in the outputs.
/// Comparing the two is the basis of balanced model reduction — the states that are both hard to
/// reach and hard to see are the ones safe to discard.
/// </para>
/// <para>
/// <b>On the method.</b> Writing the unknown as a vector produces an <c>n²</c>-by-<c>n²</c> system,
/// which is direct and exact but grows quickly. For the sizes control problems actually reach — a
/// handful to a few dozen states — this is immediate, and it has the decisive advantage of needing
/// no ordered Schur factorization, which the asymptotically better Bartels-Stewart algorithm does.
/// The coefficients are built by expanding the equation entry by entry rather than through Kronecker
/// products, which keeps the indexing conventions from being a source of silent error.
/// </para>
/// <para><b>For Beginners:</b> The question is whether a system settles down or blows up. Rather
/// than computing eigenvalues, this looks for a bowl-shaped energy function that the system always
/// slides down. If one exists, the system is stable — and you have the energy function, which is
/// worth more than the yes/no answer.
/// </para>
/// </remarks>
public sealed class LyapunovSolver<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Solves the continuous-time Lyapunov equation <c>AᵀP + PA + Q = 0</c>.
    /// </summary>
    /// <param name="stateMatrix">The state matrix <c>A</c>.</param>
    /// <param name="constantTerm">The matrix <c>Q</c>, normally symmetric positive-semidefinite.</param>
    /// <returns>The symmetric solution <c>P</c>.</returns>
    /// <exception cref="ArgumentNullException">Thrown when a matrix is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the matrices are not square or disagree.</exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when no unique solution exists, which happens exactly when two eigenvalues of <c>A</c>
    /// sum to zero — including the case of an eigenvalue at the origin.
    /// </exception>
    public Matrix<T> SolveContinuous(Matrix<T> stateMatrix, Matrix<T> constantTerm)
    {
        Validate(stateMatrix, constantTerm);

        int n = stateMatrix.Rows;

        // Expanding (AᵀP + PA)[r,c] gives sum_k A[k,r]P[k,c] + sum_k P[r,k]A[k,c], so the
        // coefficient of P[i,j] is A[i,r] when j == c, plus A[j,c] when i == r.
        var system = new Matrix<T>(n * n, n * n);

        for (int r = 0; r < n; r++)
        {
            for (int c = 0; c < n; c++)
            {
                int row = r * n + c;

                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        T coefficient = NumOps.Zero;

                        if (j == c) coefficient = NumOps.Add(coefficient, stateMatrix[i, r]);
                        if (i == r) coefficient = NumOps.Add(coefficient, stateMatrix[j, c]);

                        system[row, i * n + j] = coefficient;
                    }
                }
            }
        }

        return SolveSystem(system, constantTerm, n, continuous: true);
    }

    /// <summary>
    /// Solves the discrete-time Lyapunov (Stein) equation <c>AᵀPA − P + Q = 0</c>.
    /// </summary>
    /// <param name="stateMatrix">The state matrix <c>A</c>.</param>
    /// <param name="constantTerm">The matrix <c>Q</c>, normally symmetric positive-semidefinite.</param>
    /// <returns>The symmetric solution <c>P</c>.</returns>
    /// <exception cref="ArgumentNullException">Thrown when a matrix is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the matrices are not square or disagree.</exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when no unique solution exists, which happens exactly when the product of two
    /// eigenvalues of <c>A</c> equals one.
    /// </exception>
    public Matrix<T> SolveDiscrete(Matrix<T> stateMatrix, Matrix<T> constantTerm)
    {
        Validate(stateMatrix, constantTerm);

        int n = stateMatrix.Rows;

        // (AᵀPA)[r,c] = sum_k sum_l A[k,r] P[k,l] A[l,c], so the coefficient of P[i,j] is
        // A[i,r]*A[j,c]; the -P term subtracts one on the diagonal entry.
        var system = new Matrix<T>(n * n, n * n);

        for (int r = 0; r < n; r++)
        {
            for (int c = 0; c < n; c++)
            {
                int row = r * n + c;

                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        T coefficient = NumOps.Multiply(stateMatrix[i, r], stateMatrix[j, c]);
                        if (i == r && j == c) coefficient = NumOps.Subtract(coefficient, NumOps.One);

                        system[row, i * n + j] = coefficient;
                    }
                }
            }
        }

        return SolveSystem(system, constantTerm, n, continuous: false);
    }

    /// <summary>
    /// Returns the controllability Gramian, which measures how easily each direction of the state
    /// space can be driven by the inputs.
    /// </summary>
    /// <param name="stateMatrix">The state matrix <c>A</c>, which must be stable.</param>
    /// <param name="inputMatrix">The input matrix <c>B</c>.</param>
    /// <param name="timeDomain">Whether the system is discrete or continuous.</param>
    /// <remarks>
    /// <para>
    /// A near-singular Gramian means some combination of states is nearly unreachable — no input can
    /// move it much — which is worth knowing before designing a controller that assumes otherwise.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when a matrix is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the dimensions disagree.</exception>
    public Matrix<T> ControllabilityGramian(
        Matrix<T> stateMatrix,
        Matrix<T> inputMatrix,
        ControlTimeDomain timeDomain = ControlTimeDomain.Discrete)
    {
        if (stateMatrix is null) throw new ArgumentNullException(nameof(stateMatrix));
        if (inputMatrix is null) throw new ArgumentNullException(nameof(inputMatrix));

        if (inputMatrix.Rows != stateMatrix.Rows)
        {
            throw new ArgumentException(
                $"B must have one row per state: expected {stateMatrix.Rows}, got " +
                $"{inputMatrix.Rows}.", nameof(inputMatrix));
        }

        // The Gramian satisfies the Lyapunov equation of the transposed system with Q = BBᵀ.
        var product = ControlMath<T>.Multiply(inputMatrix, ControlMath<T>.Transpose(inputMatrix));
        var transposed = ControlMath<T>.Transpose(stateMatrix);

        return timeDomain == ControlTimeDomain.Discrete
            ? SolveDiscrete(transposed, product)
            : SolveContinuous(transposed, product);
    }

    /// <summary>
    /// Returns the observability Gramian, which measures how visibly each direction of the state
    /// space shows up in the outputs.
    /// </summary>
    /// <param name="stateMatrix">The state matrix <c>A</c>, which must be stable.</param>
    /// <param name="observationMatrix">The observation matrix <c>C</c>.</param>
    /// <param name="timeDomain">Whether the system is discrete or continuous.</param>
    /// <exception cref="ArgumentNullException">Thrown when a matrix is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the dimensions disagree.</exception>
    public Matrix<T> ObservabilityGramian(
        Matrix<T> stateMatrix,
        Matrix<T> observationMatrix,
        ControlTimeDomain timeDomain = ControlTimeDomain.Discrete)
    {
        if (stateMatrix is null) throw new ArgumentNullException(nameof(stateMatrix));
        if (observationMatrix is null) throw new ArgumentNullException(nameof(observationMatrix));

        if (observationMatrix.Columns != stateMatrix.Rows)
        {
            throw new ArgumentException(
                $"C must have one column per state: expected {stateMatrix.Rows}, got " +
                $"{observationMatrix.Columns}.", nameof(observationMatrix));
        }

        var product = ControlMath<T>.Multiply(
            ControlMath<T>.Transpose(observationMatrix), observationMatrix);

        return timeDomain == ControlTimeDomain.Discrete
            ? SolveDiscrete(stateMatrix, product)
            : SolveContinuous(stateMatrix, product);
    }

    /// <summary>
    /// Solves the vectorized system and folds the answer back into a symmetric matrix.
    /// </summary>
    private static Matrix<T> SolveSystem(
        Matrix<T> system, Matrix<T> constantTerm, int n, bool continuous)
    {
        var rightHandSide = new Vector<T>(n * n);
        for (int r = 0; r < n; r++)
        {
            for (int c = 0; c < n; c++)
            {
                rightHandSide[r * n + c] = NumOps.Negate(constantTerm[r, c]);
            }
        }

        Vector<T> solved;
        try
        {
            solved = new LuDecomposition<T>(system).Solve(rightHandSide);
        }
        catch (Exception)
        {
            throw BuildSingularException(continuous);
        }

        var solution = new Matrix<T>(n, n);
        for (int r = 0; r < n; r++)
        {
            for (int c = 0; c < n; c++)
            {
                double value = NumOps.ToDouble(solved[r * n + c]);
                if (double.IsNaN(value) || double.IsInfinity(value))
                {
                    throw BuildSingularException(continuous);
                }

                solution[r, c] = solved[r * n + c];
            }
        }

        // The solution is symmetric as a matter of theory; projecting removes the asymmetry rounding
        // introduces rather than letting it propagate into whatever consumes the result.
        return ControlMath<T>.Symmetrize(solution);
    }

    private static InvalidOperationException BuildSingularException(bool continuous) =>
        new(continuous
            ? "The continuous Lyapunov equation has no unique solution. This happens exactly when " +
              "two eigenvalues of A sum to zero — most commonly because A has an eigenvalue at the " +
              "origin, or a pair symmetric about the imaginary axis."
            : "The discrete Lyapunov equation has no unique solution. This happens exactly when the " +
              "product of two eigenvalues of A equals one — most commonly because A has an " +
              "eigenvalue on the unit circle.");

    private static void Validate(Matrix<T> stateMatrix, Matrix<T> constantTerm)
    {
        if (stateMatrix is null) throw new ArgumentNullException(nameof(stateMatrix));
        if (constantTerm is null) throw new ArgumentNullException(nameof(constantTerm));

        if (stateMatrix.Rows != stateMatrix.Columns)
        {
            throw new ArgumentException(
                $"A must be square; it is {stateMatrix.Rows}-by-{stateMatrix.Columns}.",
                nameof(stateMatrix));
        }

        if (stateMatrix.Rows == 0)
        {
            throw new ArgumentException("A must have at least one row.", nameof(stateMatrix));
        }

        if (constantTerm.Rows != stateMatrix.Rows || constantTerm.Columns != stateMatrix.Rows)
        {
            throw new ArgumentException(
                $"Q must be {stateMatrix.Rows}-by-{stateMatrix.Rows} to match A; it is " +
                $"{constantTerm.Rows}-by-{constantTerm.Columns}.", nameof(constantTerm));
        }
    }
}
