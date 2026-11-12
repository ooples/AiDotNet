using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

/// <summary>
/// Implements the Bidiagonal Decomposition of a matrix, which factors a matrix into U*B*V^T,
/// where U and V are orthogonal matrices and B is a bidiagonal matrix.
/// </summary>
/// <typeparam name="T">The numeric type used in the matrix (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// Bidiagonal decomposition transforms a matrix A into the form A = U*B*V^T, where:
/// - U is an orthogonal matrix (left singular vectors)
/// - B is a bidiagonal matrix (non-zero elements only on main diagonal and superdiagonal)
/// - V^T is the transpose of an orthogonal matrix (right singular vectors)
/// </para>
/// <para>
/// <b>For Beginners:</b> Bidiagonal decomposition is like organizing a complex matrix into a simpler form
/// that's easier to work with. Think of it like arranging books on a shelf: instead of having them scattered,
/// you organize them so most spaces are empty (zeros) and important information (non-zero values) is concentrated
/// in just two lines. This makes many calculations much faster and more efficient, especially when computing
/// singular values or solving systems of equations.
/// </para>
/// <para>
/// Common applications include:
/// <list type="bullet">
/// <item>Computing Singular Value Decomposition (SVD)</item>
/// <item>Solving least squares problems</item>
/// <item>Computing eigenvalues of symmetric matrices</item>
/// <item>Data compression and dimensionality reduction</item>
/// </list>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.DimensionalityReduction)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
    [ResearchPaper("Matrix Computations", "https://doi.org/10.56021/9781421407944")]
public class BidiagonalDecomposition<T> : MatrixDecompositionBase<T>
{
    /// <summary>
    /// Gets the left orthogonal matrix in the decomposition.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This matrix helps transform the original matrix's columns.
    /// It represents how the original data is rotated or reflected in the column space.
    /// </remarks>
    public Matrix<T> U { get; private set; }

    /// <summary>
    /// Gets the bidiagonal matrix in the decomposition.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> A bidiagonal matrix is a special matrix where non-zero values appear only
    /// on the main diagonal and the diagonal immediately above it (called the superdiagonal).
    /// All other elements are zero, making computations much more efficient.
    /// </remarks>
    public Matrix<T> B { get; private set; }

    /// <summary>
    /// Gets the right orthogonal matrix in the decomposition.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This matrix helps transform the original matrix's rows.
    /// It represents how the original data is rotated or reflected in the row space.
    /// </remarks>
    public Matrix<T> V { get; private set; }

    private BidiagonalAlgorithmType _algorithm;

    /// <summary>
    /// Creates a new bidiagonal decomposition of the specified matrix.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <param name="algorithm">The algorithm to use for decomposition (default is Householder).</param>
    /// <remarks>
    /// <para>
    /// Bidiagonal decomposition breaks down a matrix A into three simpler matrices: U, B, and V,
    /// where A = U*B*V^T. This makes many matrix operations easier to perform.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This constructor takes your original matrix and breaks it down into simpler
    /// components using the specified algorithm. The Householder algorithm (default) is generally the most
    /// stable and efficient method for most matrices.
    /// </para>
    /// </remarks>
    public BidiagonalDecomposition(Matrix<T> matrix, BidiagonalAlgorithmType algorithm = BidiagonalAlgorithmType.Householder)
        : base(matrix)
    {
        _algorithm = algorithm;
        U = new Matrix<T>(matrix.Rows, matrix.Rows);
        B = new Matrix<T>(matrix.Columns, matrix.Columns);
        V = new Matrix<T>(matrix.Columns, matrix.Columns);
        Decompose();
    }

    /// <summary>
    /// Performs the bidiagonal decomposition.
    /// </summary>
    protected override void Decompose()
    {
        ComputeDecomposition(_algorithm);
    }

    /// <summary>
    /// Performs the bidiagonal decomposition using the specified algorithm.
    /// </summary>
    /// <param name="algorithm">The algorithm to use for decomposition.</param>
    /// <exception cref="ArgumentException">Thrown when an unsupported algorithm is specified.</exception>
    /// <remarks>
    /// Different algorithms have different performance characteristics and numerical stability.
    /// Householder is generally the most stable and commonly used method.
    /// </remarks>
    public void Decompose(BidiagonalAlgorithmType algorithm = BidiagonalAlgorithmType.Householder)
    {
        _algorithm = algorithm;
        ComputeDecomposition(algorithm);
    }

    /// <summary>
    /// Performs the actual decomposition computation using the specified algorithm.
    /// </summary>
    /// <param name="algorithm">The algorithm to use for decomposition.</param>
    /// <exception cref="ArgumentException">Thrown when an unsupported algorithm is specified.</exception>
    private void ComputeDecomposition(BidiagonalAlgorithmType algorithm)
    {
        switch (algorithm)
        {
            case BidiagonalAlgorithmType.Householder:
                DecomposeHouseholder();
                break;
            case BidiagonalAlgorithmType.Givens:
                DecomposeGivens();
                break;
            case BidiagonalAlgorithmType.Lanczos:
                DecomposeLanczos();
                break;
            default:
                throw new ArgumentException("Unsupported Bidiagonal decomposition algorithm.");
        }
    }

    /// <summary>
    /// Performs bidiagonal decomposition using Householder reflections.
    /// </summary>
    /// <remarks>
    /// Householder reflections are a way to transform vectors by reflecting them across a plane.
    /// This method is numerically stable and commonly used for matrix decompositions.
    /// </remarks>
    private void DecomposeHouseholder()
    {
        int m = A.Rows;
        int n = A.Columns;
        B = A.Clone();
        U = Matrix<T>.CreateIdentity(m);
        V = Matrix<T>.CreateIdentity(n);

        int minDim = Math.Min(m, n);
        T tiny = NumOps.FromDouble(1e-14);
        T two = NumOps.FromDouble(2);

        for (int k = 0; k < minDim; k++)
        {
            // Left reflection: zero B[k+1:, k]
            if (k < m - 1)
            {
                Vector<T> x = B.GetColumnSegment(k, k, m - k);
                Vector<T> v = HouseholderVector(x);

                if (NumOps.GreaterThan(v.Norm(), tiny))
                {
                    Matrix<T> P = Matrix<T>.CreateIdentity(m - k).Subtract(v.OuterProduct(v).Multiply(two));
                    Matrix<T> subB = B.GetSubMatrix(k, k, m - k, n - k);
                    B.SetSubMatrix(k, k, P.Multiply(subB));

                    Matrix<T> subU = U.GetSubMatrix(0, k, m, m - k);
                    U.SetSubMatrix(0, k, subU.Multiply(P.Transpose()));
                }
            }

            // Right reflection: zero B[k, k+2:]
            if (k < n - 2)
            {
                Vector<T> x = B.GetRowSegment(k, k + 1, n - k - 1);
                Vector<T> v = HouseholderVector(x);

                if (NumOps.GreaterThan(v.Norm(), tiny))
                {
                    Matrix<T> P = Matrix<T>.CreateIdentity(n - k - 1).Subtract(v.OuterProduct(v).Multiply(two));
                    Matrix<T> subB = B.GetSubMatrix(k, k + 1, m - k, n - k - 1);
                    B.SetSubMatrix(k, k + 1, subB.Multiply(P));

                    Matrix<T> subV = V.GetSubMatrix(0, k + 1, n, n - k - 1);
                    V.SetSubMatrix(0, k + 1, subV.Multiply(P));
                }
            }
        }
    }

    /// <summary>
    /// Performs bidiagonal decomposition using Givens rotations.
    /// </summary>
    /// <remarks>
    /// Givens rotations are a way to zero out specific elements in a matrix by rotating
    /// two rows or columns. This method is useful for creating bidiagonal matrices.
    /// </remarks>
    private void DecomposeGivens()
    {
        int m = A.Rows;
        int n = A.Columns;
        B = A.Clone();
        U = Matrix<T>.CreateIdentity(m);
        V = Matrix<T>.CreateIdentity(n);

        for (int k = 0; k < Math.Min(m - 1, n); k++)
        {
            for (int i = m - 1; i > k; i--)
            {
                GivensRotation(B, U, i - 1, i, k, k, true);
            }

            if (k < n - 2)
            {
                for (int j = n - 1; j > k + 1; j--)
                {
                    GivensRotation(B, V, k, k, j - 1, j, false);
                }
            }
        }
    }

    /// <summary>
    /// Performs bidiagonal decomposition using the Lanczos algorithm.
    /// </summary>
    /// <remarks>
    /// The Lanczos algorithm is an iterative method that can efficiently compute
    /// partial decompositions of large matrices. It's particularly useful for
    /// sparse matrices (matrices with mostly zero values).
    /// </remarks>
    private void DecomposeLanczos()
    {
        int m = A.Rows;
        int n = A.Columns;
        B = new Matrix<T>(m, n);
        U = new Matrix<T>(m, m);
        V = new Matrix<T>(n, n);

        // Random unit starting vector from a secure source.
        var rand = RandomHelper.CreateSecureRandom();
        Vector<T> v = Matrix<T>.CreateRandom(rand, n, 1).GetColumn(0);
        v = (Vector<T>)Engine.Divide(v, v.Norm());

        Matrix<T> At = A.Transpose();
        Vector<T> previousU = new Vector<T>(m);
        T previousBeta = NumOps.Zero;
        int steps = Math.Min(m, n);
        if (steps > 0)
        {
            V.SetColumn(0, v);
        }

        for (int j = 0; j < steps; j++)
        {
            // Golub-Kahan-Lanczos step (Golub & Van Loan, Alg. 10.4.1):
            //   u_j = A v_j - beta_(j-1) u_(j-1),  alpha_j = ||u_j||
            //   v_(j+1) = A^T u_j - alpha_j v_j,   beta_j = ||v_(j+1)||
            // with full reorthogonalisation against the vectors already accepted.
            Vector<T> u = A.Multiply(v).Subtract(previousU.Multiply(previousBeta));
            if (j > 0)
            {
                Matrix<T> acceptedU = U.GetSubMatrix(0, 0, m, j);
                u = u.Subtract(acceptedU.Multiply(acceptedU.Transpose().Multiply(u)));
            }

            T alpha = u.Norm();
            u = (Vector<T>)Engine.Divide(u, alpha);
            B[j, j] = alpha;
            U.SetColumn(j, u);

            if (j < n - 1)
            {
                Vector<T> next = At.Multiply(u).Subtract(v.Multiply(alpha));
                Matrix<T> acceptedV = V.GetSubMatrix(0, 0, n, j + 1);
                next = next.Subtract(acceptedV.Multiply(acceptedV.Transpose().Multiply(next)));
                T beta = next.Norm();
                v = (Vector<T>)Engine.Divide(next, beta);
                B[j, j + 1] = beta;
                V.SetColumn(j + 1, v);
                previousBeta = beta;
            }

            previousU = u;
        }
    }

    /// <summary>
    /// Solves the linear system Ax = b using the bidiagonal decomposition.
    /// </summary>
    /// <param name="b">The right-hand side vector of the equation Ax = b.</param>
    /// <returns>The solution vector x.</returns>
    /// <exception cref="ArgumentException">Thrown when the length of vector b doesn't match the number of rows in matrix A.</exception>
    /// <remarks>
    /// <para>
    /// This method uses the decomposition to efficiently solve the system without
    /// directly inverting the matrix, which is more numerically stable.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This finds the values of x that satisfy Ax = b. Using the bidiagonal
    /// decomposition makes this much faster than directly solving the original equation, especially
    /// for large matrices. It's like solving a puzzle by first organizing the pieces into groups.
    /// </para>
    /// </remarks>
    public override Vector<T> Solve(Vector<T> b)
    {
        if (b.Length != A.Rows)
            throw new ArgumentException("Vector b must have the same length as the number of rows in matrix A.");

        // Solve Ax = b using U*B*V^T * x = b
        Vector<T> y = U.Transpose().Multiply(b);
        Vector<T> z = SolveBidiagonal(y);
        return V.Multiply(z);
    }

    /// <summary>
    /// Computes the inverse of the original matrix A using the bidiagonal decomposition.
    /// </summary>
    /// <returns>The inverse matrix of A.</returns>
    /// <remarks>
    /// <para>
    /// Matrix inversion is computationally expensive and can be numerically unstable.
    /// When possible, use the Solve method instead of explicitly computing the inverse.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The inverse of a matrix is like the reciprocal of a number.
    /// Just as 5 * (1/5) = 1, a matrix multiplied by its inverse gives the identity matrix.
    /// However, computing the inverse is expensive, so it's usually better to use Solve() instead.
    /// </para>
    /// </remarks>
    public override Matrix<T> Invert()
    {
        int n = A.Columns;
        if (A.Rows != n)
            throw new ArgumentException($"Matrix must be square to be inverted. Got {A.Rows}x{A.Columns} matrix.");

        if (n == 0)
        {
            return new Matrix<T>(0, 0);
        }

        // A^{-1} = V B^{-1} U^T: one upper-bidiagonal solve for every column of U^T.
        return V.Multiply(SolveTriangularSystem(UpperBidiagonalBand(), U.Transpose(), upper: true));
    }

    /// <summary>
    /// Computes a Householder reflection vector for the given input vector.
    /// </summary>
    /// <param name="x">The input vector.</param>
    /// <returns>A Householder vector that can be used to create a reflection matrix.</returns>
    /// <remarks>
    /// <para>
    /// A Householder reflection is a transformation that reflects a vector across a plane.
    /// It's used to zero out specific elements in a matrix during decomposition.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Think of a Householder reflection like holding up a mirror to a vector.
    /// The reflection helps us systematically create zeros in the matrix, simplifying its structure
    /// step by step during the decomposition process.
    /// </para>
    /// </remarks>
    private Vector<T> HouseholderVector(Vector<T> x)
    {
        T norm = x.Norm();

        // If the input vector has zero norm, return a zero vector
        // (no Householder transformation needed)
        if (NumOps.LessThan(norm, NumOps.FromDouble(1e-14)))
        {
            return new Vector<T>(x.Length);
        }

        Vector<T> v = x.Clone();
        // Use opposite sign to avoid cancellation
        v[0] = NumOps.LessThan(x[0], NumOps.Zero)
            ? NumOps.Subtract(v[0], norm)
            : NumOps.Add(v[0], norm);

        T vNorm = v.Norm();

        // If the Householder vector has zero norm, return a zero vector
        if (NumOps.LessThan(vNorm, NumOps.FromDouble(1e-14)))
        {
            return new Vector<T>(x.Length);
        }

        // VECTORIZED: Normalize using Engine division
        return (Vector<T>)Engine.Divide(v, vNorm);
    }

    /// <summary>
    /// Applies a Givens rotation to the specified matrices.
    /// </summary>
    /// <param name="M">The matrix to which the rotation is applied.</param>
    /// <param name="Q">The orthogonal matrix that accumulates the rotations.</param>
    /// <param name="i">The first row/column index for the rotation.</param>
    /// <param name="k">The second row/column index for the rotation.</param>
    /// <param name="j">The first column/row index for the rotation.</param>
    /// <param name="l">The second column/row index for the rotation.</param>
    /// <param name="isLeft">If true, applies a left rotation (row operation); otherwise, applies a right rotation (column operation).</param>
    /// <remarks>
    /// <para>
    /// A Givens rotation is a simple rotation in a plane spanned by two coordinate axes.
    /// It's used to selectively zero out specific elements in a matrix during decomposition.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> A Givens rotation is like turning a dial to make specific values become zero.
    /// It's precise and only affects two rows or columns at a time, making it useful for sparse matrices
    /// or when you need to zero out specific elements without disturbing others.
    /// </para>
    /// </remarks>
    private void GivensRotation(Matrix<T> M, Matrix<T> Q, int i, int k, int j, int l, bool isLeft)
    {
        T a = M[i, j];
        T b = M[k, l];
        T r = NumOps.Sqrt(NumOps.Add(NumOps.Multiply(a, a), NumOps.Multiply(b, b)));
        T c = NumOps.Divide(a, r);
        T s = NumOps.Divide(b, r);

        if (isLeft)
        {
            // VECTORIZED: Apply rotation to rows using vector operations
            Vector<T> rowI = M.GetRow(i);
            Vector<T> rowK = M.GetRow(k);
            Vector<T> newRowI = rowI.Multiply(c).Add(rowK.Multiply(s));
            Vector<T> newRowK = rowI.Multiply(NumOps.Negate(s)).Add(rowK.Multiply(c));
            M.SetRow(i, newRowI);
            M.SetRow(k, newRowK);

            // VECTORIZED: Update Q columns using vector operations
            Vector<T> colI = Q.GetColumn(i);
            Vector<T> colK = Q.GetColumn(k);
            Vector<T> newColI = colI.Multiply(c).Add(colK.Multiply(s));
            Vector<T> newColK = colI.Multiply(NumOps.Negate(s)).Add(colK.Multiply(c));
            Q.SetColumn(i, newColI);
            Q.SetColumn(k, newColK);
        }
        else
        {
            // VECTORIZED: Apply rotation to columns using vector operations
            // M := M * G (right multiply)
            Vector<T> colJ = M.GetColumn(j);
            Vector<T> colL = M.GetColumn(l);
            Vector<T> newColJ = colJ.Multiply(c).Add(colL.Multiply(s));
            Vector<T> newColL = colJ.Multiply(NumOps.Negate(s)).Add(colL.Multiply(c));
            M.SetColumn(j, newColJ);
            M.SetColumn(l, newColL);

            // VECTORIZED: Update Q (which is V) columns using vector operations
            // V := V * G (right multiply), so update columns j and l
            Vector<T> qColJ = Q.GetColumn(j);
            Vector<T> qColL = Q.GetColumn(l);
            Vector<T> newQColJ = qColJ.Multiply(c).Add(qColL.Multiply(s));
            Vector<T> newQColL = qColJ.Multiply(NumOps.Negate(s)).Add(qColL.Multiply(c));
            Q.SetColumn(j, newQColJ);
            Q.SetColumn(l, newQColL);
        }
    }

    /// <summary>
    /// Solves a bidiagonal system of linear equations.
    /// </summary>
    /// <param name="y">The right-hand side vector of the equation Bx = y.</param>
    /// <returns>The solution vector x.</returns>
    /// <remarks>
    /// This method efficiently solves a system where the coefficient matrix is bidiagonal
    /// (has non-zero elements only on the main diagonal and the superdiagonal).
    /// It uses back-substitution, which is much faster than general matrix inversion.
    /// </remarks>
    private Vector<T> SolveBidiagonal(Vector<T> y)
    {
        int n = B.Columns;
        if (n == 0)
        {
            return new Vector<T>(0);
        }

        // Back substitution on the diagonal and superdiagonal of B, as one engine solve.
        var rhs = y.Length == n ? y : y.GetSubVector(0, n);
        return SolveTriangularSystem(UpperBidiagonalBand(), rhs, upper: true);
    }

    /// <summary>The leading n x n diagonal and superdiagonal of B (everything else zeroed).</summary>
    private Matrix<T> UpperBidiagonalBand()
    {
        int n = B.Columns;
        var square = Engine.TensorSlice(Tensor<T>.FromMatrix(B), new[] { 0, 0 }, new[] { n, n });
        return Engine.TensorTriu(Engine.TensorTril(square, 1), 0).ToMatrix();
    }

    /// <summary>
    /// Returns the three factor matrices of the bidiagonal decomposition.
    /// </summary>
    /// <returns>
    /// A tuple containing:
    /// - U: The left orthogonal matrix
    /// - B: The bidiagonal matrix
    /// - V: The right orthogonal matrix
    /// </returns>
    /// <remarks>
    /// The original matrix A can be reconstructed as A = U * B * V^T.
    /// This decomposition is useful for solving linear systems and computing singular values.
    /// </remarks>
    public (Matrix<T> U, Matrix<T> B, Matrix<T> V) GetFactors()
    {
        return (U, B, V);
    }
}
