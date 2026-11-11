global using AiDotNet.Enums.AlgorithmTypes;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Exceptions;

namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

/// <summary>
/// Implements LU decomposition for matrices, which factorizes a matrix into a product of lower and upper triangular matrices.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (typically double or float).</typeparam>
/// <remarks>
/// <para>
/// LU decomposition factors a matrix A into the product of a lower triangular matrix L and an upper
/// triangular matrix U, often with row permutations represented by a permutation matrix P (P*A = L*U).
/// This decomposition is fundamental in numerical linear algebra and is used for solving linear systems,
/// computing determinants, and matrix inversion.
/// </para>
/// <para>
/// <b>For Beginners:</b> LU decomposition breaks a matrix into two simpler triangular matrices:
/// L (lower triangular, with values only on and below the diagonal) and U (upper triangular, with
/// values only on and above the diagonal). Think of it like factoring a number into simpler parts,
/// but for matrices. This makes solving equations much faster.
/// </para>
/// <para>
/// Real-world applications:
/// - Solving systems of linear equations efficiently
/// - Computing matrix determinants
/// - Finding matrix inverses in numerical algorithms
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.DimensionalityReduction)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
    [ResearchPaper("Matrix Computations", "https://doi.org/10.56021/9781421407944")]
public class LuDecomposition<T> : MatrixDecompositionBase<T>
{
    /// <summary>
    /// Gets the lower triangular matrix from the decomposition.
    /// </summary>
    public Matrix<T> L { get; private set; } = new Matrix<T>(0, 0);

    /// <summary>
    /// Gets the upper triangular matrix from the decomposition.
    /// </summary>
    public Matrix<T> U { get; private set; } = new Matrix<T>(0, 0);

    /// <summary>
    /// Gets the permutation vector that tracks row exchanges during pivoting.
    /// Each value represents the original row index for the current row position.
    /// </summary>
    public Vector<int> P { get; private set; } = new Vector<int>(0);

    /// <summary>
    /// The algorithm to use for LU decomposition.
    /// </summary>
    private readonly LuAlgorithmType _algorithm;

    /// <summary>
    /// Initializes a new instance of the LuDecomposition class and performs the decomposition.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <param name="luAlgorithm">The algorithm to use for LU decomposition. Defaults to partial pivoting.</param>
    /// <remarks>
    /// <para>
    /// LU decomposition factorizes a matrix A into the product of a lower triangular matrix L and an upper triangular matrix U,
    /// possibly with row permutations (P*A = L*U). This is useful for solving linear systems and calculating determinants.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This constructor takes your input matrix and immediately breaks it down into
    /// the L and U components using the specified algorithm. Different algorithms have different trade-offs
    /// in terms of speed and numerical stability.
    /// </para>
    /// </remarks>
    public LuDecomposition(Matrix<T> matrix, LuAlgorithmType luAlgorithm = LuAlgorithmType.PartialPivoting)
        : base(matrix)
    {
        _algorithm = luAlgorithm;

        Decompose();
    }

    /// <summary>
    /// Performs the LU decomposition.
    /// </summary>
    protected override void Decompose()
    {
        (L, U, P) = ComputeDecomposition(A, _algorithm);
    }

    /// <summary>
    /// Solves the linear system Ax = b using the LU decomposition.
    /// </summary>
    /// <param name="b">The right-hand side vector of the equation Ax = b.</param>
    /// <returns>The solution vector x.</returns>
    /// <remarks>
    /// This method solves the equation in two steps:
    /// 1. Forward substitution to solve Ly = Pb
    /// 2. Back substitution to solve Ux = y
    /// </remarks>
    public override Vector<T> Solve(Vector<T> b)
    {
        var pb = PermutateVector(b, P);
        var y = ForwardSubstitution(L, pb);

        return BackSubstitution(U, y);
    }

    /// <summary>
    /// Performs the matrix decomposition using the specified algorithm.
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <param name="algorithm">The algorithm to use for decomposition.</param>
    /// <returns>A tuple containing the L matrix, U matrix, and permutation vector P.</returns>
    private (Matrix<T> L, Matrix<T> U, Vector<int> P) ComputeDecomposition(Matrix<T> matrix, LuAlgorithmType algorithm)
    {
        return algorithm switch
        {
            LuAlgorithmType.Doolittle => ComputeLuDoolittle(matrix),
            LuAlgorithmType.Crout => ComputeLuCrout(matrix),
            LuAlgorithmType.PartialPivoting => ComputeLuPartialPivoting(matrix),
            LuAlgorithmType.CompletePivoting => ComputeLuCompletePivoting(matrix),
            LuAlgorithmType.Cholesky => ComputeCholesky(matrix),
            _ => throw new ArgumentException("Unsupported LU decomposition algorithm."),
        };
    }

    /// <summary>
    /// Computes LU decomposition with partial pivoting (Gaussian elimination with row pivoting).
    /// </summary>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <returns>A tuple containing the L matrix, U matrix, and permutation vector P.</returns>
    /// <remarks>
    /// Partial pivoting selects the largest element in the current column as the pivot,
    /// which improves numerical stability by reducing round-off errors.
    /// </remarks>
    private (Matrix<T> L, Matrix<T> U, Vector<int> P) ComputeLuPartialPivoting(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        if (n != matrix.Columns)
            throw new ArgumentException("Matrix must be square for LU decomposition.");

        if (n == 0)
        {
            return (new Matrix<T>(0, 0), new Matrix<T>(0, 0), new Vector<int>(0));
        }

        // PA = LU with partial (row) pivoting in one engine factorization.
        var (permutation, lower, upper) = Linalg.LU(Tensor<T>.FromMatrix(matrix));

        // The k-th pivot is the largest |A[i,k]| in the remaining column, i.e. |U[k,k]|; report the
        // first zero or non-finite one as the elimination did.
        var unusable = UnusablePivotMask(Engine.TensorDiagonal(upper));
        if (Engine.TensorCountNonzero(unusable) > 0)
        {
            int index = Engine.TensorArgMax(unusable, 0)[0];
            throw new MatrixFactorizationException(
                $"LU factorization encountered a zero or non-finite pivot at index {index}.");
        }

        // Row order P with (PA)[i] = A[P[i]] (pb[i] = b[P[i]]). Read it from whichever orientation
        // of the returned permutation matrix reproduces L U.
        var a = Tensor<T>.FromMatrix(matrix);
        var lu = Engine.TensorMatMul(lower, upper);
        T residual = Engine.TensorSumOfSquares(Engine.TensorSubtract(Engine.TensorMatMul(permutation, a), lu));
        T transposedResidual = Engine.TensorSumOfSquares(Engine.TensorSubtract(
            Engine.TensorMatMul(Engine.TensorTranspose(permutation), a), lu));
        var rowOrder = NumOps.LessThanOrEquals(residual, transposedResidual)
            ? Engine.TensorArgMax(permutation, 1)
            : Engine.TensorArgMax(permutation, 0);
        return (lower.ToMatrix(), upper.ToMatrix(), rowOrder.ToVector());
    }

    /// <summary>
    /// Performs LU decomposition with complete pivoting.
    /// </summary>
    /// <param name="matrix">The square matrix to decompose.</param>
    /// <returns>
    /// A tuple containing:
    /// - L: Lower triangular matrix with unit diagonal
    /// - U: Upper triangular matrix
    /// - P: Permutation vector for rows
    /// </returns>
    /// <remarks>
    /// Complete pivoting searches for the largest element in the entire remaining submatrix
    /// to use as a pivot. This improves numerical stability compared to partial pivoting.
    /// The method rearranges both rows and columns during the decomposition process.
    /// </remarks>
    private (Matrix<T> L, Matrix<T> U, Vector<int> P) ComputeLuCompletePivoting(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        if (n != matrix.Columns)
            throw new ArgumentException("Matrix must be square for LU decomposition.");

        var (L, U, rowOrder, columnOrder) = EliminateRightLooking(matrix, completePivoting: true, zeroPivotGivesZeroMultipliers: false);
        if (n == 0)
        {
            return (L, U, rowOrder);
        }

        // Undo the column pivoting: column j of U belongs at original column Q[j].
        var adjustedU = Engine.TensorIndexCopy(
            new Tensor<T>(new[] { n, n }),
            1,
            Tensor<int>.FromVector(columnOrder),
            Tensor<T>.FromMatrix(U));

        return (L, adjustedU.ToMatrix(), rowOrder);
    }

    /// <summary>
    /// Gaussian elimination with right-looking whole-block engine updates: for each pivot k the
    /// multipliers A[k+1:, k] / A[k, k] form column k of L and the trailing block receives the
    /// rank-1 update l * A[k, k+1:].
    /// </summary>
    /// <param name="matrix">The square matrix (not modified).</param>
    /// <param name="completePivoting">
    /// True to bring the largest |A[i,j]| of the remaining block to the pivot position by a row and
    /// a column swap (first maximum in row-major order) and reject zero or non-finite pivots.
    /// </param>
    /// <param name="zeroPivotGivesZeroMultipliers">True to use zero multipliers below a zero pivot.</param>
    /// <returns>Unit lower L, upper U, the row order P and the column order Q (PAQ = LU).</returns>
    private (Matrix<T> L, Matrix<T> U, Vector<int> rowOrder, Vector<int> columnOrder) EliminateRightLooking(
        Matrix<T> matrix, bool completePivoting, bool zeroPivotGivesZeroMultipliers)
    {
        int n = matrix.Rows;
        if (n == 0)
        {
            return (new Matrix<T>(0, 0), new Matrix<T>(0, 0), new Vector<int>(0), new Vector<int>(0));
        }

        var identityOrder = Engine.TensorArange<int>(0, n, 1).Reshape(new[] { n, 1 });

        Matrix<T> work = matrix.Clone();
        Matrix<T> multipliers = new(n, n);
        Matrix<int> rowOrder = identityOrder.ToMatrix();
        Matrix<int> columnOrder = identityOrder.ToMatrix();
        T zero = NumOps.Zero;

        for (int k = 0; k < n - 1; k++)
        {
            int rest = n - k - 1;
            if (completePivoting)
            {
                int size = n - k;
                var block = Engine.TensorAbs(Engine.TensorSlice(Tensor<T>.FromMatrix(work), new[] { k, k }, new[] { size, size }));
                int flat = Engine.TensorArgMax(block.Reshape(new[] { size * size }), 0)[0];
                int pivotRow = k + flat / size;
                int pivotCol = k + flat % size;
                EnsureUsablePivot(Engine.TensorSlice(block, new[] { flat / size, flat % size }, new[] { 1, 1 }), k);

                if (pivotRow != k)
                {
                    Engine.SwapRows(work, k, pivotRow);
                    // Columns k.. of the multiplier rows are still zero, so a full-row swap only
                    // exchanges the multipliers computed so far.
                    Engine.SwapRows(multipliers, k, pivotRow);
                    Engine.SwapRows(rowOrder, k, pivotRow);
                }

                if (pivotCol != k)
                {
                    Engine.SwapColumns(work, k, pivotCol);
                    Engine.SwapRows(columnOrder, k, pivotCol);
                }
            }

            var current = Tensor<T>.FromMatrix(work);
            var inversePivot = Engine.TensorReciprocal(Engine.TensorSlice(current, new[] { k, k }, new[] { 1, 1 }));
            if (zeroPivotGivesZeroMultipliers)
            {
                inversePivot = Engine.TensorWhere(Engine.TensorIsFinite(inversePivot), inversePivot, Engine.TensorFullLike(inversePivot, zero));
            }

            var column = Engine.TensorSlice(current, new[] { k + 1, k }, new[] { rest, 1 });
            var factors = Engine.TensorMultiplyScalar(column, Engine.TensorSum(inversePivot));
            var pivotRowTail = Engine.TensorSlice(current, new[] { k, k + 1 }, new[] { 1, rest });
            var trailing = Engine.TensorSlice(current, new[] { k + 1, k + 1 }, new[] { rest, rest });
            trailing = Engine.TensorSubtract(trailing, Engine.TensorMatMul(factors, pivotRowTail));

            work.SetSubMatrix(k + 1, k + 1, trailing.ToMatrix());
            multipliers.SetSubMatrix(k + 1, k, factors.ToMatrix());
        }

        var final = Tensor<T>.FromMatrix(work);
        if (completePivoting)
        {
            EnsureUsablePivot(Engine.TensorSlice(final, new[] { n - 1, n - 1 }, new[] { 1, 1 }), n - 1);
        }

        var lower = Engine.TensorAdd(Tensor<T>.FromMatrix(multipliers), Engine.TensorEye<T>(n));
        var upper = Engine.TensorTriu(final, 0);
        return (lower.ToMatrix(), upper.ToMatrix(), rowOrder.GetColumn(0), columnOrder.GetColumn(0));
    }

    /// <summary>
    /// Computes the Cholesky decomposition of a symmetric positive-definite matrix.
    /// </summary>
    /// <param name="matrix">The input matrix to decompose (must be symmetric positive-definite).</param>
    /// <returns>
    /// A tuple containing:
    /// - L: Lower triangular matrix where L * L^T = matrix
    /// - U: Upper triangular matrix (transpose of L)
    /// - P: Identity permutation vector
    /// </returns>
    /// <remarks>
    /// Cholesky decomposition is a special form of LU decomposition for symmetric positive-definite matrices.
    /// It is more efficient than standard LU decomposition for these types of matrices.
    /// A symmetric positive-definite matrix is a matrix that is symmetric (equal to its transpose)
    /// and has all positive eigenvalues (a mathematical property that ensures stability).
    /// </remarks>
    private (Matrix<T> L, Matrix<T> U, Vector<int> P) ComputeCholesky(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        if (n != matrix.Columns)
            throw new ArgumentException("Matrix must be square for Cholesky decomposition.");

        if (n == 0)
        {
            return (new Matrix<T>(0, 0), new Matrix<T>(0, 0), new Vector<int>(0));
        }

        var identityOrder = Engine.TensorArange<int>(0, n, 1).ToVector();

        // A = L L^T in one engine factorization; like the scalar recurrence, a matrix that is not
        // positive definite yields a non-finite factor rather than an exception.
        var (factor, _) = Linalg.CholeskyEx(Tensor<T>.FromMatrix(matrix), upper: false);
        Matrix<T> L = factor.ToMatrix();
        return (L, L.Transpose(), identityOrder);
    }

    /// <summary>
    /// Computes the LU decomposition using Doolittle's method.
    /// </summary>
    /// <param name="matrix">The input matrix to decompose.</param>
    /// <returns>
    /// A tuple containing:
    /// - L: Lower triangular matrix with 1's on the diagonal
    /// - U: Upper triangular matrix
    /// - P: Identity permutation vector
    /// </returns>
    /// <remarks>
    /// Doolittle's method is a form of LU decomposition where the diagonal elements of L are all 1's.
    /// This method does not use pivoting, so it may be less numerically stable for some matrices.
    /// It's best used when the matrix is known to be well-conditioned (not close to singular).
    /// </remarks>
    private (Matrix<T> L, Matrix<T> U, Vector<int> P) ComputeLuDoolittle(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        if (n != matrix.Columns)
            throw new ArgumentException("Matrix must be square for LU decomposition.");

        // Doolittle: unit lower L, no pivoting.
        var (L, U, rowOrder, _) = EliminateRightLooking(matrix, completePivoting: false, zeroPivotGivesZeroMultipliers: false);
        return (L, U, rowOrder);
    }

    /// <summary>
    /// Computes the LU decomposition using Crout's method.
    /// </summary>
    /// <param name="matrix">The input matrix to decompose.</param>
    /// <returns>
    /// A tuple containing:
    /// - L: Lower triangular matrix
    /// - U: Upper triangular matrix with 1's on the diagonal
    /// - P: Identity permutation vector
    /// </returns>
    /// <remarks>
    /// <para>
    /// Crout's method is a form of LU decomposition where the diagonal elements of U are all 1's.
    /// This method does not use pivoting, so it may be less numerically stable for some matrices.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Crout's method is a way to break down a complex matrix into simpler parts.
    /// Think of it like factoring a number (e.g., 12 = 3 * 4). Here, we're factoring a matrix into 
    /// two triangular matrices - one with values only below the diagonal (L) and one with values 
    /// only above the diagonal and 1's on the diagonal itself (U). This makes solving equations much easier.
    /// </para>
    /// </remarks>
    private (Matrix<T> L, Matrix<T> U, Vector<int> P) ComputeLuCrout(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        if (n != matrix.Columns)
            throw new ArgumentException("Matrix must be square for LU decomposition.");

        // Crout (L general lower, U unit upper) is the transposed Doolittle factorization of A^T:
        // A^T = L' U'  =>  A = U'^T L'^T. A zero diagonal in L leaves that row of U zero.
        var (lowerT, upperT, rowOrder, _) = EliminateRightLooking(matrix.Transpose(), completePivoting: false, zeroPivotGivesZeroMultipliers: true);
        return (upperT.Transpose(), lowerT.Transpose(), rowOrder);
    }

    /// <summary>
    /// Rearranges a vector according to the permutation vector P.
    /// </summary>
    /// <param name="b">The original vector to permutate.</param>
    /// <param name="P">The permutation vector that defines the new order.</param>
    /// <returns>The permutated vector.</returns>
    /// <remarks>
    /// <para>
    /// This method is used internally to apply row permutations to the right-hand side vector
    /// when solving linear systems.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> When we rearrange rows in our matrix during decomposition, we need to 
    /// make the same changes to our right-hand side vector. This method does that rearrangement.
    /// It's like if you reordered the questions in a test, you'd need to reorder the answers too!
    /// </para>
    /// </remarks>
    private Vector<T> PermutateVector(Vector<T> b, Vector<int> P)
    {
        if (b.Length == 0)
        {
            return new Vector<T>(0);
        }

        return Engine.TensorIndexSelect(Tensor<T>.FromVector(b), Tensor<int>.FromVector(P), 0).ToVector();
    }

    /// <summary>
    /// Performs forward substitution to solve the system Ly = b where L is lower triangular.
    /// </summary>
    /// <param name="L">The lower triangular matrix.</param>
    /// <param name="b">The right-hand side vector.</param>
    /// <returns>The solution vector y.</returns>
    /// <remarks>
    /// <para>
    /// Forward substitution is used to solve a system of equations where the coefficient matrix
    /// is lower triangular. It works by solving for one variable at a time, starting from the first.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Forward substitution is like solving a cascade of simple equations.
    /// Since L is lower triangular (has zeros above the diagonal), we can solve for the first 
    /// variable directly, then use that to solve for the second, and so on. It's like a domino 
    /// effect where each solution helps us find the next one.
    /// </para>
    /// </remarks>
    private Vector<T> ForwardSubstitution(Matrix<T> L, Vector<T> b)
    {
        if (L.Rows == 0)
        {
            return new Vector<T>(0);
        }

        if (Engine.TensorCountNonzero(UnusablePivotMask(Engine.TensorDiagonal(Tensor<T>.FromMatrix(L)))) > 0)
        {
            throw new MatrixFactorizationException("The lower triangular factor is singular.");
        }

        return SolveTriangularSystem(L, b, upper: false);
    }

    /// <summary>
    /// Performs back substitution to solve the system Ux = y where U is upper triangular.
    /// </summary>
    /// <param name="U">The upper triangular matrix.</param>
    /// <param name="y">The right-hand side vector.</param>
    /// <returns>The solution vector x.</returns>
    /// <remarks>
    /// <para>
    /// Back substitution is used to solve a system of equations where the coefficient matrix
    /// is upper triangular. It works by solving for one variable at a time, starting from the last.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Back substitution is like solving a waterfall of equations from bottom to top.
    /// Since U is upper triangular (has zeros below the diagonal), we can solve for the last 
    /// variable directly, then use that to solve for the second-to-last, and so on. It's the 
    /// reverse process of forward substitution, working backwards through our variables.
    /// </para>
    /// </remarks>
    private Vector<T> BackSubstitution(Matrix<T> U, Vector<T> y)
    {
        int n = U.Columns;
        if (n == 0)
        {
            return new Vector<T>(0);
        }

        // Back substitution meets the pivots bottom-up, so the last invalid one is reported.
        var unusable = UnusablePivotMask(Engine.TensorDiagonal(Tensor<T>.FromMatrix(U)));
        if (Engine.TensorCountNonzero(unusable) > 0)
        {
            int fromEnd = Engine.TensorArgMax(Engine.TensorFlip(unusable, new[] { 0 }), 0)[0];
            throw new MatrixFactorizationException(
                $"The upper triangular factor has an invalid pivot at index {n - 1 - fromEnd}.");
        }

        return SolveTriangularSystem(U, y, upper: true);
    }

    private void EnsureUsablePivot(Tensor<T> pivot, int index)
    {
        if (Engine.TensorCountNonzero(UnusablePivotMask(pivot)) > 0)
        {
            throw new MatrixFactorizationException(
                $"LU factorization encountered a zero or non-finite pivot at index {index}.");
        }
    }

    /// <summary>1 where a pivot is zero, NaN or infinite; 0 where it is usable.</summary>
    private Tensor<T> UnusablePivotMask(Tensor<T> pivots)
    {
        var usable = Engine.TensorWhere(
            Engine.TensorIsFinite(pivots),
            Engine.TensorNotEquals(pivots, NumOps.Zero),
            Engine.TensorFullLike(pivots, NumOps.Zero));
        return Engine.TensorSubtract(Engine.TensorOnesLike(usable), usable);
    }

    /// <summary>
    /// Calculates the inverse of the original matrix using the LU decomposition.
    /// </summary>
    /// <returns>The inverse of the original matrix.</returns>
    /// <remarks>
    /// Matrix inversion is computationally expensive and numerically less stable than directly
    /// solving a system. When possible, use the Solve method instead of calculating the inverse.
    /// </remarks>
    // Uses default implementation from base class (MatrixDecompositionBase.Invert)
}
