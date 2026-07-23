global using AiDotNet.DecompositionMethods.MatrixDecomposition;

namespace AiDotNet.Helpers;

/// <summary>
/// Provides methods for solving linear systems of equations using various matrix decomposition techniques.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A linear system is a collection of linear equations with the same variables.
/// For example: 2x + 3y = 5 and 4x - y = 1 form a linear system. In matrix form, this is written as Ax = b,
/// where A is the coefficient matrix, x is the vector of variables we're solving for, and b is the vector of constants.
/// This helper class provides different ways to solve for x given A and b.
/// </para>
/// </remarks>
public static class MatrixSolutionHelper
{
    /// <summary>
    /// Solves a linear system of equations Ax = b using the specified decomposition method.
    /// </summary>
    /// <typeparam name="T">The numeric type used in the matrices and vectors.</typeparam>
    /// <param name="A">The coefficient matrix of the linear system.</param>
    /// <param name="b">The constant vector of the linear system.</param>
    /// <param name="decompositionType">The matrix decomposition method to use for solving the system.</param>
    /// <returns>The solution vector x that satisfies Ax = b.</returns>
    /// <exception cref="NotSupportedException">Thrown when an unsupported decomposition type is used.</exception>
    /// <exception cref="ArgumentException">Thrown when the decomposition type is not recognized.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method solves equations of the form Ax = b, where A is a matrix, and b is a vector.
    /// Think of it like solving multiple equations at once. The "decomposition type" is just the mathematical
    /// approach used to solve the system. Different approaches work better for different types of problems.
    /// For example:
    /// - LU works well for general square matrices
    /// - Cholesky is faster but only works for symmetric, positive-definite matrices
    /// - SVD is more robust but slower, and works even for non-square matrices
    /// </para>
    /// </remarks>
    public static Vector<T> SolveLinearSystem<T>(Matrix<T> A, Vector<T> b, MatrixDecompositionType decompositionType)
    {
        return decompositionType switch
        {
            MatrixDecompositionType.Lu => new LuDecomposition<T>(A).Solve(b),
            MatrixDecompositionType.Cholesky => new CholeskyDecomposition<T>(A).Solve(b),
            MatrixDecompositionType.Qr => new QrDecomposition<T>(A).Solve(b),
            MatrixDecompositionType.Svd => new SvdDecomposition<T>(A).Solve(b),
            MatrixDecompositionType.Cramer => SolveCramer(A, b),
            MatrixDecompositionType.GramSchmidt => SolveGramSchmidt(A, b),
            MatrixDecompositionType.Normal => SolveNormal(A, b),
            MatrixDecompositionType.Lq => new LqDecomposition<T>(A).Solve(b),
            MatrixDecompositionType.Takagi => throw new NotSupportedException("Takagi decomposition is not suitable for solving linear systems."),
            MatrixDecompositionType.Hessenberg => SolveHessenberg(A, b),
            MatrixDecompositionType.Schur => SolveSchur(A, b),
            MatrixDecompositionType.Eigen => SolveEigen(A, b),
            _ => throw new ArgumentException("Unsupported decomposition type", nameof(decompositionType))
        };
    }

    /// <summary>
    /// Solves a linear system using a pre-computed matrix decomposition.
    /// </summary>
    /// <typeparam name="T">The numeric type used in the matrices and vectors.</typeparam>
    /// <param name="b">The constant vector of the linear system.</param>
    /// <param name="decompositionMethod">The pre-computed matrix decomposition to use.</param>
    /// <returns>The solution vector x that satisfies Ax = b.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method is useful when you've already decomposed your matrix A
    /// and want to solve for multiple different b vectors without repeating the decomposition.
    /// This is more efficient because matrix decomposition is usually the most time-consuming part
    /// of solving a linear system.
    /// </para>
    /// </remarks>
    public static Vector<T> SolveLinearSystem<T>(Vector<T> b, IMatrixDecomposition<T> decompositionMethod)
    {
        return decompositionMethod.Solve(b);
    }

    /// <summary>
    /// Solves a linear system using Cramer's rule.
    /// </summary>
    /// <typeparam name="T">The numeric type used in the matrices and vectors.</typeparam>
    /// <param name="A">The coefficient matrix of the linear system.</param>
    /// <param name="b">The constant vector of the linear system.</param>
    /// <returns>The solution vector x that satisfies Ax = b.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Cramer's rule is a method that uses determinants to solve linear systems.
    /// For each variable xi, it replaces the i-th column of matrix A with vector b,
    /// then divides the determinant of this new matrix by the determinant of A.
    /// While elegant mathematically, it's generally not efficient for large systems
    /// because calculating determinants is computationally expensive.
    /// </para>
    /// </remarks>
    private static Vector<T> SolveCramer<T>(Matrix<T> A, Vector<T> b)
    {
        var det = A.Determinant();
        var x = new Vector<T>(b.Length);
        var numOps = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < b.Length; i++)
        {
            var Ai = A.Clone();
            for (int j = 0; j < b.Length; j++)
            {
                Ai[j, i] = b[j];
            }
            x[i] = numOps.Divide(Ai.Determinant(), det);
        }

        return x;
    }

    /// <summary>
    /// Solves a linear system using the Gram-Schmidt decomposition.
    /// </summary>
    /// <typeparam name="T">The numeric type used in the matrices and vectors.</typeparam>
    /// <param name="A">The coefficient matrix of the linear system.</param>
    /// <param name="b">The constant vector of the linear system.</param>
    /// <returns>The solution vector x that satisfies Ax = b.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Gram-Schmidt process converts a set of vectors into a set of
    /// orthogonal vectors (vectors at right angles to each other). This decomposition
    /// is similar to QR decomposition and is useful for solving linear systems because
    /// orthogonal vectors make calculations simpler. It's particularly useful in machine
    /// learning for feature selection and dimensionality reduction.
    /// </para>
    /// </remarks>
    private static Vector<T> SolveGramSchmidt<T>(Matrix<T> A, Vector<T> b)
    {
        var gs = new GramSchmidtDecomposition<T>(A);
        return gs.Solve(b);
    }

    /// <summary>
    /// Solves a linear system using the normal equations approach.
    /// </summary>
    /// <typeparam name="T">The numeric type used in the matrices and vectors.</typeparam>
    /// <param name="A">The coefficient matrix of the linear system.</param>
    /// <param name="b">The constant vector of the linear system.</param>
    /// <returns>The solution vector x that satisfies Ax = b.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The normal equations approach transforms the original system Ax = b
    /// into A^T·A·x = A^T·b (where A^T is the transpose of A). This creates a symmetric,
    /// positive-definite matrix that can be solved efficiently using Cholesky decomposition.
    /// This method is commonly used in linear regression and least squares problems where
    /// you're trying to find the best-fit line or curve for a set of data points.
    /// </para>
    /// <para>
    /// Note: While efficient, this method can sometimes be numerically unstable if A is
    /// ill-conditioned (has a high condition number).
    /// </para>
    /// </remarks>
    private static Vector<T> SolveNormal<T>(Matrix<T> A, Vector<T> b)
    {
        var ATA = A.Transpose().Multiply(A);
        var ATb = A.Transpose().Multiply(b);

        return new CholeskyDecomposition<T>(ATA).Solve(ATb);
    }

    /// <summary>
    /// Solves a linear system using Hessenberg decomposition.
    /// </summary>
    /// <typeparam name="T">The numeric type used in the matrices and vectors.</typeparam>
    /// <param name="A">The coefficient matrix of the linear system.</param>
    /// <param name="b">The constant vector of the linear system.</param>
    /// <returns>The solution vector x that satisfies Ax = b.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Hessenberg decomposition transforms a matrix into a form that's
    /// almost triangular (zeros below the first subdiagonal). This is often an intermediate
    /// step in eigenvalue calculations and can be used to solve linear systems more efficiently
    /// than working with the original matrix. It's particularly useful for certain types of
    /// structured matrices that appear in control theory and differential equations.
    /// </para>
    /// </remarks>
    private static Vector<T> SolveHessenberg<T>(Matrix<T> A, Vector<T> b)
    {
        var hessenberg = new HessenbergDecomposition<T>(A);
        return hessenberg.Solve(b);
    }

    /// <summary>
    /// Solves a linear system using Schur decomposition.
    /// </summary>
    /// <typeparam name="T">The numeric type used in the matrices and vectors.</typeparam>
    /// <param name="A">The coefficient matrix of the linear system.</param>
    /// <param name="b">The constant vector of the linear system.</param>
    /// <returns>The solution vector x that satisfies Ax = b.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Schur decomposition factors a matrix A into Q·T·Q^H, where Q is unitary
    /// (its inverse equals its conjugate transpose), T is upper triangular, and Q^H is the
    /// conjugate transpose of Q. This decomposition is useful for calculating eigenvalues and
    /// can be used to solve linear systems. It's particularly valuable in stability analysis
    /// of dynamical systems and in certain machine learning algorithms.
    /// </para>
    /// </remarks>
    private static Vector<T> SolveSchur<T>(Matrix<T> A, Vector<T> b)
    {
        var schur = new SchurDecomposition<T>(A);
        return schur.Solve(b);
    }

    /// <summary>
    /// Solves a linear system using Eigenvalue decomposition.
    /// </summary>
    /// <typeparam name="T">The numeric type used in the matrices and vectors.</typeparam>
    /// <param name="A">The coefficient matrix of the linear system.</param>
    /// <param name="b">The constant vector of the linear system.</param>
    /// <returns>The solution vector x that satisfies Ax = b.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Eigenvalue decomposition breaks down a matrix into special vectors (eigenvectors) 
    /// and values (eigenvalues) that represent the matrix's fundamental characteristics. 
    /// Think of it like finding the "natural directions" and "stretching factors" of the matrix.
    /// </para>
    /// <para>
    /// When we solve a system using eigenvalue decomposition, we're essentially transforming the problem 
    /// into a coordinate system where the solution becomes much simpler to calculate. This method is 
    /// particularly useful for understanding dynamic systems, principal component analysis in data science, 
    /// and many physics applications.
    /// </para>
    /// <para>
    /// While powerful, eigenvalue decomposition is typically more computationally intensive than other methods
    /// and works best for square matrices that have a complete set of eigenvectors.
    /// </para>
    /// </remarks>
    private static Vector<T> SolveEigen<T>(Matrix<T> A, Vector<T> b)
    {
        var _eigenDecomposition = new EigenDecomposition<T>(A);
        return _eigenDecomposition.Solve(b);
    }
}
