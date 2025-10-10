namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

/// <summary>
/// Implements the Gram-Schmidt orthogonalization process to decompose a matrix into an orthogonal matrix Q and an upper triangular matrix R.
/// </summary>
/// <remarks>
/// The Gram-Schmidt process transforms a set of vectors into a set of orthogonal vectors (vectors that are perpendicular to each other).
/// This decomposition is useful for solving linear systems, computing least squares solutions, and other numerical applications.
/// 
/// In simple terms, this class takes a matrix and breaks it down into two special matrices:
/// - Q: A matrix with perpendicular columns (orthogonal matrix)
/// - R: An upper triangular matrix (only has values on and above the diagonal)
/// 
/// Together, these matrices can be multiplied to get back the original matrix: A = Q * R
/// </remarks>
/// <typeparam name="T">The numeric data type used in calculations (e.g., float, double).</typeparam>
public class GramSchmidtDecomposition<T> : IMatrixDecomposition<T>
{
    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps = default!;

    /// <summary>
    /// Gets the orthogonal matrix Q from the decomposition.
    /// </summary>
    /// <remarks>
    /// The Q matrix has columns that are orthogonal to each other (perpendicular).
    /// Each column has a length (norm) of 1, making it an "orthonormal" matrix.
    /// This property makes Q useful for transforming vectors while preserving their length.
    /// </remarks>
    public Matrix<T> Q { get; private set; }

    /// <summary>
    /// Gets the upper triangular matrix R from the decomposition.
    /// </summary>
    /// <remarks>
    /// The R matrix is upper triangular, meaning it only has non-zero values on and above the diagonal.
    /// This structure makes it easy to solve systems of equations through back-substitution.
    /// </remarks>
    public Matrix<T> R { get; private set; }

    /// <summary>
    /// Gets the original matrix that was decomposed.
    /// </summary>
    public Matrix<T> A { get; private set; }

    /// <summary>
    /// Creates a new Gram-Schmidt decomposition for the specified matrix.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <param name="algorithm">The type of Gram-Schmidt algorithm to use (Classical or Modified).</param>
    /// <remarks>
    /// The constructor initializes the decomposition by computing the Q and R matrices.
    /// Two algorithm variants are available:
    /// - Classical: The original algorithm, which can suffer from numerical instability
    /// - Modified: An improved version with better numerical stability for most applications
    /// </remarks>
    public GramSchmidtDecomposition(Matrix<T> matrix, GramSchmidtAlgorithmType algorithm = GramSchmidtAlgorithmType.Classical)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        A = matrix;
        (Q, R) = Decompose(matrix, algorithm);
    }

    /// <summary>
    /// Selects and applies the appropriate Gram-Schmidt algorithm based on the specified type.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <param name="algorithm">The type of Gram-Schmidt algorithm to use.</param>
    /// <returns>A tuple containing the Q and R matrices.</returns>
    /// <exception cref="ArgumentException">Thrown when an unsupported algorithm type is specified.</exception>
    private (Matrix<T> Q, Matrix<T> R) Decompose(Matrix<T> matrix, GramSchmidtAlgorithmType algorithm)
    {
        return algorithm switch
        {
            GramSchmidtAlgorithmType.Classical => ComputeClassicalGramSchmidt(matrix),
            GramSchmidtAlgorithmType.Modified => ComputeModifiedGramSchmidt(matrix),
            _ => throw new ArgumentException("Unsupported Gram-Schmidt algorithm.")
        };
    }

    /// <summary>
    /// Computes the QR decomposition using the Classical Gram-Schmidt algorithm.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>A tuple containing the Q and R matrices.</returns>
    /// <remarks>
    /// The Classical Gram-Schmidt algorithm works as follows:
    /// 1. For each column in the input matrix:
    ///    a. Take the current column vector
    ///    b. Subtract its projections onto all previous orthogonal vectors
    ///    c. Normalize the resulting vector to have length 1
    /// 
    /// While simpler to understand, this method can suffer from numerical instability
    /// when dealing with nearly linearly dependent columns.
    /// </remarks>
    private (Matrix<T> Q, Matrix<T> R) ComputeClassicalGramSchmidt(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;

        var Q = new Matrix<T>(m, n);
        var R = new Matrix<T>(n, n);

        for (int j = 0; j < n; j++)
        {
            // Get the j-th column of the original matrix
            var v = matrix.GetColumn(j);

            // Subtract projections onto previous orthogonal vectors
            for (int i = 0; i < j; i++)
            {
                // Calculate projection coefficient (dot product)
                R[i, j] = Q.GetColumn(i).DotProduct(matrix.GetColumn(j));
                
                // Subtract the projection from v
                v = v.Subtract(Q.GetColumn(i).Multiply(R[i, j]));
            }

            // Calculate the norm (length) of the resulting vector
            R[j, j] = v.Norm();
            
            // Normalize the vector and store it as a column in Q
            Q.SetColumn(j, v.Divide(R[j, j]));
        }

        return (Q, R);
    }

    /// <summary>
    /// Computes the QR decomposition using the Modified Gram-Schmidt algorithm.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>A tuple containing the Q and R matrices.</returns>
    /// <remarks>
    /// The Modified Gram-Schmidt algorithm is a more numerically stable version that:
    /// 1. Takes each column vector one by one
    /// 2. Immediately updates the vector after each projection subtraction
    /// 3. This reduces accumulated rounding errors compared to the classical method
    /// 
    /// This method is generally preferred for practical applications as it produces
    /// more accurate results, especially with nearly linearly dependent columns.
    /// </remarks>
    private (Matrix<T> Q, Matrix<T> R) ComputeModifiedGramSchmidt(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;

        var Q = new Matrix<T>(m, n);
        var R = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            // Get the i-th column of the original matrix
            var v = matrix.GetColumn(i);

            // Subtract projections onto previous orthogonal vectors
            for (int j = 0; j < i; j++)
            {
                // Calculate projection coefficient (dot product)
                R[j, i] = Q.GetColumn(j).DotProduct(v);
                
                // Immediately update v by subtracting the projection
                v = v.Subtract(Q.GetColumn(j).Multiply(R[j, i]));
            }

            // Calculate the norm (length) of the resulting vector
            R[i, i] = v.Norm();
            
            // Normalize the vector and store it as a column in Q
            Q.SetColumn(i, v.Divide(R[i, i]));
        }

        return (Q, R);
    }

    /// <summary>
    /// Solves a system of linear equations Ax = b using the QR decomposition.
    /// </summary>
    /// <param name="b">The right-hand side vector of the equation Ax = b.</param>
    /// <returns>The solution vector x.</returns>
    /// <remarks>
    /// This method solves the system in two steps:
    /// 1. First, it solves Qy = b for y by computing y = Q^T * b
    ///    (This works because Q is orthogonal, so Q^T * Q = I)
    /// 2. Then, it solves Rx = y using back-substitution
    ///    (This is efficient because R is upper triangular)
    /// 
    /// The result is the solution vector x that satisfies Ax = b.
    /// </remarks>
    public Vector<T> Solve(Vector<T> b)
    {
        // Solve Qy = b by computing y = Q^T * b
        var y = Q.Transpose().Multiply(b);

        // Solve Rx = y using back-substitution
        return BackSubstitution(R, y);
    }

    /// <summary>
    /// Performs back-substitution to solve an upper triangular system Rx = y.
    /// </summary>
    /// <param name="R">The upper triangular matrix.</param>
    /// <param name="y">The right-hand side vector.</param>
    /// <returns>The solution vector x.</returns>
    /// <remarks>
    /// Back-substitution is an efficient method for solving triangular systems.
    /// It works by:
    /// 1. Starting with the last equation (bottom row) and solving for the last variable
    /// 2. Moving upward, substituting known values to solve for each variable
    /// 
    /// For example, in a 3x3 system:
    /// - First solve for x₃ from the last equation
    /// - Then solve for x₂ using the known value of x₃
    /// - Finally solve for x₁ using the known values of x₂ and x₃
    /// </remarks>
    private Vector<T> BackSubstitution(Matrix<T> R, Vector<T> y)
    {
        int n = R.Rows;
        var x = new Vector<T>(n);

        // Start from the bottom row and work upward
        for (int i = n - 1; i >= 0; i--)
        {
            // Calculate the sum of known terms
            var sum = _numOps.Zero;
            for (int j = i + 1; j < n; j++)
            {
                sum = _numOps.Add(sum, _numOps.Multiply(R[i, j], x[j]));
            }
            
            // Solve for the current variable
            x[i] = _numOps.Divide(_numOps.Subtract(y[i], sum), R[i, i]);
        }

        return x;
    }

    /// <summary>
    /// Calculates the inverse of the original matrix A using the QR decomposition.
    /// </summary>
    /// <returns>The inverse matrix of A.</returns>
    /// <remarks>
    /// This method finds the inverse by solving the equation AX = I, where:
    /// - A is the original matrix
    /// - I is the identity matrix
    /// - X is the inverse of A
    /// 
    /// The process works by:
    /// 1. Creating an identity matrix of the same size as A
    /// 2. Solving the system AX = I for each column of X separately
    /// 3. Combining the solution vectors to form the complete inverse matrix
    /// 
    /// The inverse exists only if A is square and has full rank (all columns are linearly independent).
    /// Using QR decomposition for matrix inversion is numerically stable compared to some other methods.
    /// </remarks>
    public Matrix<T> Invert()
    {
        int n = A.Rows;
        var identity = Matrix<T>.CreateIdentity(n);
        var inverse = new Matrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            inverse.SetColumn(i, Solve(identity.GetColumn(i)));
        }

        return inverse;
    }
}