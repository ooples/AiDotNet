using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Control;

/// <summary>
/// Finds a point satisfying a linear matrix inequality
/// <c>F(x) = F₀ + x₁F₁ + … + x_mF_m ⪰ 0</c>.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Minimizes the largest eigenvalue of <c>−F(x)</c> by the subgradient method (N. Z. Shor,
/// <i>Minimization Methods for Non-Differentiable Functions</i>, Springer 1985), which is feasible
/// exactly when that minimum reaches zero. The formulation of control problems as linear matrix
/// inequalities follows S. Boyd, L. El Ghaoui, E. Feron and V. Balakrishnan, <i>Linear Matrix
/// Inequalities in System and Control Theory</i> (SIAM 1994).
/// </para>
/// <para>
/// <b>Why this is a convex problem despite the matrix.</b> The eigenvalues of a matrix are not
/// linear in its entries, so requiring them all to be non-negative sounds hopeless. But the largest
/// eigenvalue of a symmetric matrix is the maximum of <c>vᵀMv</c> over unit vectors <c>v</c> — a
/// maximum of linear functions of <c>M</c>, and therefore convex. Since <c>F(x)</c> is affine in
/// <c>x</c>, the composition is convex in <c>x</c>, and the set where it is non-positive is convex.
/// That single observation is why an enormous range of control problems — stability, robustness,
/// performance bounds, gain synthesis — can be solved rather than merely searched.
/// </para>
/// <para>
/// <b>The subgradient.</b> The largest eigenvalue is not differentiable where the top eigenvalue is
/// repeated, so there is no gradient to follow. There is always a subgradient: if <c>v</c> is a unit
/// eigenvector of the largest eigenvalue, then <c>−vᵀF_i v</c> is a valid subgradient component. The
/// step schedule shrinks as one over the square root of the iteration count, which is what makes the
/// method converge for a nonsmooth convex objective.
/// </para>
/// <para>
/// <b>What the result does and does not claim.</b> A reported <c>Feasible</c> is verified — the
/// returned matrix is factorized to confirm it really is positive semidefinite, so the status is
/// never an inference from convergence behaviour. A failure is reported as
/// <c>IterationLimit</c> and never as "infeasible", because a search that did not find a point has
/// not shown that none exists.
/// </para>
/// <para><b>For Beginners:</b> Some questions in control take the form "is there a matrix making all
/// of these conditions hold at once?" — for instance, is there an energy function proving this
/// system is stable? Written the right way, that question is convex, meaning it has no false summits
/// to get stuck on and a systematic search will find an answer if one exists.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Find a diagonal P = diag(x0, x1) that is positive definite:
/// //   F(x) = -epsilon*I + x0*E00 + x1*E11 &gt;= 0
/// var result = new LinearMatrixInequalitySolver&lt;double&gt;().Solve(constantTerm, basis);
/// if (result.Status == LinearMatrixInequalityStatus.Feasible)
/// {
///     // result.Matrix is positive semidefinite, verified.
/// }
/// </code>
/// </example>
public sealed class LinearMatrixInequalitySolver<T>
{
    private const double MatrixValidationTolerance = 1e-10;
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly LinearMatrixInequalityOptions _options;

    /// <summary>
    /// Creates a solver with the default options.
    /// </summary>
    public LinearMatrixInequalitySolver()
        : this(new LinearMatrixInequalityOptions())
    {
    }

    /// <summary>
    /// Creates a solver.
    /// </summary>
    /// <param name="options">Solver configuration.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="options"/> is null.</exception>
    /// <exception cref="ArgumentException">Thrown when a setting is out of range.</exception>
    public LinearMatrixInequalitySolver(LinearMatrixInequalityOptions options)
    {
        if (options is null) throw new ArgumentNullException(nameof(options));
        _options = new LinearMatrixInequalityOptions(options);
    }

    /// <summary>
    /// Searches for coefficients satisfying <c>F₀ + Σ xᵢFᵢ ⪰ 0</c>.
    /// </summary>
    /// <param name="constantTerm">The constant matrix <c>F₀</c>, which must be symmetric.</param>
    /// <param name="basis">
    /// The matrices <c>F₁ … F_m</c> multiplying each coefficient, all the same size as
    /// <paramref name="constantTerm"/> and all symmetric.
    /// </param>
    /// <param name="initialGuess">Where to start, or <c>null</c> to start from the origin.</param>
    /// <returns>The result, including a verified feasibility status.</returns>
    /// <exception cref="ArgumentNullException">Thrown when an argument is null.</exception>
    /// <exception cref="ArgumentException">
    /// Thrown when the matrices are not square, disagree in size, or the basis is empty.
    /// </exception>
    public LinearMatrixInequalityResult<T> Solve(
        Matrix<T> constantTerm, IReadOnlyList<Matrix<T>> basis, Vector<T>? initialGuess = null)
    {
        if (constantTerm is null) throw new ArgumentNullException(nameof(constantTerm));
        if (basis is null) throw new ArgumentNullException(nameof(basis));

        if (constantTerm.Rows != constantTerm.Columns)
        {
            throw new ArgumentException(
                $"F0 must be square; it is {constantTerm.Rows}-by-{constantTerm.Columns}.",
                nameof(constantTerm));
        }

        int size = constantTerm.Rows;
        if (size == 0)
        {
            throw new ArgumentException("F0 must have at least one row.", nameof(constantTerm));
        }

        ValidateSymmetric(constantTerm, nameof(constantTerm), "F0");

        if (basis.Count == 0)
        {
            throw new ArgumentException(
                "At least one basis matrix is required; with none there is no variable to search " +
                "over and the question is simply whether F0 is already positive semidefinite.",
                nameof(basis));
        }

        for (int i = 0; i < basis.Count; i++)
        {
            if (basis[i] is null)
            {
                throw new ArgumentException($"Basis matrix {i} is null.", nameof(basis));
            }

            if (basis[i].Rows != size || basis[i].Columns != size)
            {
                throw new ArgumentException(
                    $"Basis matrix {i} must be {size}-by-{size} to match F0; it is " +
                    $"{basis[i].Rows}-by-{basis[i].Columns}.", nameof(basis));
            }

            ValidateSymmetric(basis[i], nameof(basis), $"Basis matrix {i}");
        }

        int variableCount = basis.Count;

        if (initialGuess is not null && initialGuess.Length != variableCount)
        {
            throw new ArgumentException(
                $"The initial guess must have {variableCount} entries to match the basis; it has " +
                $"{initialGuess.Length}.", nameof(initialGuess));
        }

        var current = new Vector<T>(variableCount);
        if (initialGuess is not null)
        {
            for (int i = 0; i < variableCount; i++) current[i] = initialGuess[i];
        }

        var best = current;
        double bestSmallest = double.NegativeInfinity;
        Matrix<T>? bestMatrix = null;

        int iteration;
        for (iteration = 1; iteration <= _options.MaxIterations; iteration++)
        {
            var matrix = Assemble(constantTerm, basis, current);

            // The largest eigenvalue of -F(x) is the negative of the smallest of F(x); its
            // eigenvector is the direction in which F(x) is least positive definite.
            var negated = ControlMath<T>.Scale(matrix, -1.0);
            var (largest, direction) = LargestEigenpair(negated);

            double smallest = -largest;

            if (bestMatrix is null || smallest > bestSmallest)
            {
                bestSmallest = smallest;
                best = Copy(current);
                bestMatrix = matrix;
            }

            if (smallest >= _options.Margin) break;

            // Subgradient of lambda_max(-F(x)) with respect to x_i is -v' F_i v.
            var subgradient = new double[variableCount];
            for (int i = 0; i < variableCount; i++)
            {
                subgradient[i] = -Quadratic(basis[i], direction);
            }

            double norm = 0.0;
            for (int i = 0; i < variableCount; i++) norm += subgradient[i] * subgradient[i];
            norm = Math.Sqrt(norm);

            if (norm <= 0.0)
            {
                // No direction improves the worst eigenvalue: every basis matrix is flat along it, so
                // this point is a minimum of a convex function and no further progress is possible.
                break;
            }

            double step = _options.InitialStepSize / Math.Sqrt(iteration);

            var next = new Vector<T>(variableCount);
            for (int i = 0; i < variableCount; i++)
            {
                next[i] = NumOps.Subtract(
                    current[i], NumOps.FromDouble(step * subgradient[i] / norm));
            }

            current = next;
        }

        if (bestMatrix is null)
        {
            throw new InvalidOperationException("The LMI search did not execute an iteration.");
        }

        // Feasibility is confirmed by factorizing the matrix rather than trusting the eigenvalue
        // estimate that drove the search.
        bool feasible = IsPositiveSemidefinite(bestMatrix);

        return new LinearMatrixInequalityResult<T>(
            feasible
                ? LinearMatrixInequalityStatus.Feasible
                : LinearMatrixInequalityStatus.IterationLimit,
            best,
            bestMatrix,
            NumOps.FromDouble(bestSmallest),
            Math.Min(iteration, _options.MaxIterations));
    }

    private static void ValidateSymmetric(Matrix<T> matrix, string parameterName, string displayName)
    {
        for (int row = 0; row < matrix.Rows; row++)
        {
            for (int column = row; column < matrix.Columns; column++)
            {
                double upper = NumOps.ToDouble(matrix[row, column]);
                double lower = NumOps.ToDouble(matrix[column, row]);
                if (double.IsNaN(upper) || double.IsInfinity(upper) ||
                    double.IsNaN(lower) || double.IsInfinity(lower))
                {
                    throw new ArgumentException(
                        $"{displayName} must contain only finite values; entries ({row}, {column}) " +
                        $"and ({column}, {row}) were {upper} and {lower}.", parameterName);
                }

                double scale = Math.Max(1.0, Math.Max(Math.Abs(upper), Math.Abs(lower)));
                if (Math.Abs(upper - lower) > MatrixValidationTolerance * scale)
                {
                    throw new ArgumentException(
                        $"{displayName} must be symmetric within tolerance " +
                        $"{MatrixValidationTolerance}; entries ({row}, {column}) and " +
                        $"({column}, {row}) were {upper} and {lower}.", parameterName);
                }
            }
        }
    }

    private static Vector<T> Copy(Vector<T> source)
    {
        var result = new Vector<T>(source.Length);
        for (int i = 0; i < source.Length; i++) result[i] = source[i];
        return result;
    }

    /// <summary>
    /// Builds <c>F₀ + Σ xᵢFᵢ</c>.
    /// </summary>
    private static Matrix<T> Assemble(
        Matrix<T> constantTerm, IReadOnlyList<Matrix<T>> basis, Vector<T> coefficients)
    {
        int size = constantTerm.Rows;
        var result = new Matrix<T>(size, size);

        for (int r = 0; r < size; r++)
        {
            for (int c = 0; c < size; c++) result[r, c] = constantTerm[r, c];
        }

        for (int i = 0; i < basis.Count; i++)
        {
            var term = basis[i];
            T weight = coefficients[i];

            for (int r = 0; r < size; r++)
            {
                for (int c = 0; c < size; c++)
                {
                    result[r, c] = NumOps.Add(result[r, c], NumOps.Multiply(weight, term[r, c]));
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Returns the largest eigenvalue of a symmetric matrix and a unit eigenvector for it.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Power iteration on <c>M + σI</c>, shifted far enough that every eigenvalue is positive so the
    /// largest is also the largest in magnitude — which is the one power iteration converges to. The
    /// shift is the Gershgorin bound, which is cheap and always sufficient.
    /// </para>
    /// <para>
    /// This is used rather than a full eigendecomposition because only the extreme eigenpair is
    /// needed, and because the iteration's behaviour is predictable on the symmetric matrices this
    /// solver always produces.
    /// </para>
    /// </remarks>
    private (double Eigenvalue, Vector<T> Eigenvector) LargestEigenpair(Matrix<T> matrix)
    {
        int size = matrix.Rows;

        // Gershgorin: every eigenvalue lies within sum of |row| of the diagonal, so shifting by that
        // bound makes the matrix positive definite.
        double shift = 0.0;
        for (int r = 0; r < size; r++)
        {
            double rowTotal = 0.0;
            for (int c = 0; c < size; c++) rowTotal += Math.Abs(NumOps.ToDouble(matrix[r, c]));
            shift = Math.Max(shift, rowTotal);
        }

        shift += 1.0;

        var vector = new double[size];
        for (int i = 0; i < size; i++) vector[i] = 1.0 / Math.Sqrt(size);

        var next = new double[size];

        for (int step = 0; step < _options.PowerIterations; step++)
        {
            for (int r = 0; r < size; r++)
            {
                double total = shift * vector[r];
                for (int c = 0; c < size; c++)
                {
                    total += NumOps.ToDouble(matrix[r, c]) * vector[c];
                }

                next[r] = total;
            }

            double norm = 0.0;
            for (int i = 0; i < size; i++) norm += next[i] * next[i];
            norm = Math.Sqrt(norm);

            if (norm <= 0.0) break;

            for (int i = 0; i < size; i++) vector[i] = next[i] / norm;
        }

        // Rayleigh quotient against the original matrix, undoing the shift implicitly.
        double eigenvalue = 0.0;
        for (int r = 0; r < size; r++)
        {
            double total = 0.0;
            for (int c = 0; c < size; c++)
            {
                total += NumOps.ToDouble(matrix[r, c]) * vector[c];
            }

            eigenvalue += vector[r] * total;
        }

        var eigenvector = new Vector<T>(size);
        for (int i = 0; i < size; i++) eigenvector[i] = NumOps.FromDouble(vector[i]);

        return (eigenvalue, eigenvector);
    }

    /// <summary>
    /// Returns <c>vᵀMv</c>.
    /// </summary>
    private static double Quadratic(Matrix<T> matrix, Vector<T> vector)
    {
        double total = 0.0;
        for (int r = 0; r < matrix.Rows; r++)
        {
            double inner = 0.0;
            for (int c = 0; c < matrix.Columns; c++)
            {
                inner += NumOps.ToDouble(matrix[r, c]) * NumOps.ToDouble(vector[c]);
            }

            total += NumOps.ToDouble(vector[r]) * inner;
        }

        return total;
    }

    /// <summary>
    /// Tests positive semidefiniteness by attempting a Cholesky factorization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A symmetric matrix is positive semidefinite exactly when this factorization runs to
    /// completion without meeting a negative pivot. The test is used instead of comparing computed
    /// eigenvalues against zero because it is decisive: it depends only on the arithmetic actually
    /// performed, not on how accurately an iterative eigenvalue estimate has converged.
    /// </para>
    /// </remarks>
    internal static bool IsPositiveSemidefinite(
        Matrix<T> matrix, double tolerance = MatrixValidationTolerance)
    {
        int size = matrix.Rows;
        var factor = new double[size, size];

        for (int r = 0; r < size; r++)
        {
            for (int c = 0; c <= r; c++)
            {
                double total = NumOps.ToDouble(matrix[r, c]);
                for (int k = 0; k < c; k++) total -= factor[r, k] * factor[c, k];

                if (r == c)
                {
                    if (total < -tolerance) return false;

                    factor[r, c] = Math.Sqrt(Math.Max(total, 0.0));
                }
                else
                {
                    // A zero pivot with a non-zero off-diagonal above it means the matrix is
                    // indefinite rather than merely singular.
                    if (factor[c, c] <= tolerance)
                    {
                        if (Math.Abs(total) > tolerance) return false;
                        factor[r, c] = 0.0;
                    }
                    else
                    {
                        factor[r, c] = total / factor[c, c];
                    }
                }
            }
        }

        return true;
    }
}
