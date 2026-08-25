using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Control;

/// <summary>
/// How a linear matrix inequality search ended.
/// </summary>
public enum LinearMatrixInequalityStatus
{
    /// <summary>
    /// A point satisfying the inequality was found, and verified by factorizing the resulting matrix.
    /// </summary>
    Feasible,

    /// <summary>
    /// No satisfying point was found within the iteration limit.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is deliberately not called "infeasible". A subgradient search that fails to find a point
    /// has not shown that none exists — it may simply have run out of iterations, or the feasible set
    /// may be so thin that no strictly interior point was reachable. Proving infeasibility needs a
    /// dual certificate this method does not produce, so claiming it would be an unsupported
    /// assertion.
    /// </para>
    /// </remarks>
    IterationLimit,
}

/// <summary>
/// The result of searching for a point satisfying a linear matrix inequality.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public sealed class LinearMatrixInequalityResult<T>
{
    /// <summary>Gets how the search ended.</summary>
    public LinearMatrixInequalityStatus Status { get; }

    /// <summary>
    /// Gets the coefficients found, or the best reached when the search hit its limit.
    /// </summary>
    public Vector<T> Variables { get; }

    /// <summary>
    /// Gets the matrix <c>F(x)</c> at <see cref="Variables"/>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Returned so a caller can check the answer independently rather than take the status on trust.
    /// When the status is <see cref="LinearMatrixInequalityStatus.Feasible"/> this matrix is positive
    /// semidefinite, and that has been verified rather than inferred.
    /// </para>
    /// </remarks>
    public Matrix<T> Matrix { get; }

    /// <summary>
    /// Gets the smallest eigenvalue of <see cref="Matrix"/>, which is non-negative exactly when the
    /// inequality is satisfied.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is the margin: a comfortably positive value means the point is well inside the feasible
    /// set and will survive rounding and modelling error, while a value barely above zero means it is
    /// sitting on the boundary and should not be relied on.
    /// </para>
    /// </remarks>
    public T SmallestEigenvalue { get; }

    /// <summary>Gets the number of iterations performed.</summary>
    public int Iterations { get; }

    /// <summary>
    /// Creates a linear matrix inequality result.
    /// </summary>
    /// <param name="status">How the search ended.</param>
    /// <param name="variables">The coefficients found.</param>
    /// <param name="matrix">The matrix at those coefficients.</param>
    /// <param name="smallestEigenvalue">The smallest eigenvalue of that matrix.</param>
    /// <param name="iterations">The number of iterations performed.</param>
    /// <exception cref="ArgumentNullException">Thrown when an argument is null.</exception>
    public LinearMatrixInequalityResult(
        LinearMatrixInequalityStatus status,
        Vector<T> variables,
        Matrix<T> matrix,
        T smallestEigenvalue,
        int iterations)
    {
        Status = status;
        Variables = variables ?? throw new ArgumentNullException(nameof(variables));
        Matrix = matrix ?? throw new ArgumentNullException(nameof(matrix));
        SmallestEigenvalue = smallestEigenvalue;
        Iterations = iterations;
    }
}
