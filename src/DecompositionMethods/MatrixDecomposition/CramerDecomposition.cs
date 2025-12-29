namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

/// <summary>
/// Implements Cramer's rule for solving systems of linear equations and matrix inversion.
/// </summary>
/// <typeparam name="T">The numeric data type used in calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Cramer's rule is a method for solving systems of linear equations using determinants.
/// It works by replacing columns in the coefficient matrix with the solution vector
/// and calculating ratios of determinants. This method is primarily educational and not
/// recommended for large matrices due to its computational complexity.
/// </para>
/// <para>
/// <b>For Beginners:</b> Cramer's Rule is a formula-based method for solving systems of equations.
/// It uses determinants (a special number calculated from a matrix) to find the solution directly.
/// Think of it like using a formula to solve a math problem rather than using step-by-step algebra.
/// While elegant, it becomes slow for large matrices.
/// </para>
/// <para>
/// Real-world applications:
/// - Solving small systems of equations (2x2 or 3x3) in engineering
/// - Teaching linear algebra concepts
/// - Theoretical analysis and mathematical proofs
/// </para>
/// </remarks>
public class CramerDecomposition<T> : MatrixDecompositionBase<T>
{
    /// <summary>
    /// Creates a new Cramer's rule decomposition for the specified matrix.
    /// </summary>
    /// <param name="matrix">The square matrix to decompose.</param>
    /// <exception cref="ArgumentException">Thrown when the input matrix is not square.</exception>
    public CramerDecomposition(Matrix<T> matrix) : base(matrix)
    {
        if (A.Rows != A.Columns)
        {
            throw new ArgumentException("Cramer's rule requires a square matrix.");
        }
    }

    /// <summary>
    /// Decompose is not applicable for Cramer's rule as it directly solves without factorization.
    /// </summary>
    protected override void Decompose()
    {
        // Cramer's rule doesn't require a decomposition step
        // The solving is done directly in the Solve method using determinants
    }

    /// <summary>
    /// Solves a system of linear equations Ax = b using Cramer's rule.
    /// </summary>
    /// <remarks>
    /// Cramer's rule works by replacing each column of the coefficient matrix A
    /// with the solution vector b, one at a time, and calculating the ratio of
    /// determinants. For each variable x_i, we compute:
    /// x_i = det(A_i) / det(A)
    /// where A_i is matrix A with column i replaced by vector b.
    /// </remarks>
    /// <param name="b">The right-hand side vector of the equation Ax = b.</param>
    /// <returns>The solution vector x that satisfies Ax = b.</returns>
    /// <exception cref="ArgumentException">Thrown when the dimensions of A and b don't match.</exception>
    /// <exception cref="InvalidOperationException">Thrown when the matrix is singular (has zero determinant).</exception>
    public override Vector<T> Solve(Vector<T> b)
    {
        if (A.Rows != b.Length)
        {
            throw new ArgumentException("The number of rows in A must match the length of b.");
        }

        T detA = Determinant(A);
        if (NumOps.Equals(detA, NumOps.Zero))
        {
            throw new InvalidOperationException("The matrix is singular and cannot be solved using Cramer's rule.");
        }

        Vector<T> x = new(A.Columns);
        for (int i = 0; i < A.Columns; i++)
        {
            Matrix<T> Ai = ReplaceColumn(A, b, i);
            x[i] = NumOps.Divide(Determinant(Ai), detA);
        }

        return x;
    }

    /// <summary>
    /// Calculates the inverse of the original matrix using the adjugate method.
    /// </summary>
    /// <remarks>
    /// This method computes the inverse by calculating the adjugate (classical adjoint)
    /// of the matrix and dividing by its determinant. The adjugate is the transpose of
    /// the cofactor matrix. For each element (i,j), we compute:
    /// inverse[j,i] = cofactor(A,i,j) / det(A)
    /// </remarks>
    /// <returns>The inverse of the original matrix A.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the matrix is singular (has zero determinant).</exception>
    public override Matrix<T> Invert()
    {
        T detA = Determinant(A);
        if (NumOps.Equals(detA, NumOps.Zero))
        {
            throw new InvalidOperationException("The matrix is singular and cannot be inverted.");
        }

        Matrix<T> inverse = new(A.Rows, A.Columns);

        // VECTORIZED: Compute cofactors and construct inverse using row operations
        for (int i = 0; i < A.Rows; i++)
        {
            Vector<T> inverseRow = new Vector<T>(A.Columns);
            for (int j = 0; j < A.Columns; j++)
            {
                T cofactor = Cofactor(A, i, j);
                inverseRow[j] = NumOps.Divide(cofactor, detA);
            }
            // Transpose during construction: row i of cofactor matrix → column i of inverse
            for (int j = 0; j < A.Columns; j++)
            {
                inverse[j, i] = inverseRow[j];
            }
        }

        return inverse;
    }

    /// <summary>
    /// Calculates the determinant of a matrix recursively using cofactor expansion.
    /// </summary>
    /// <remarks>
    /// The determinant is a scalar value that can be computed from the elements of a square matrix.
    /// It has many uses, including checking if a matrix is invertible (non-zero determinant).
    /// This implementation uses the cofactor expansion along the first row of the matrix.
    /// </remarks>
    /// <param name="matrix">The matrix whose determinant to calculate.</param>
    /// <returns>The determinant value.</returns>
    private T Determinant(Matrix<T> matrix)
    {
        // Base case: 0x0 matrix (empty matrix) has determinant 1 by convention
        // This is needed for cofactor expansion of 1x1 matrices to work correctly
        if (matrix.Rows == 0)
        {
            return NumOps.One;
        }

        if (matrix.Rows == 1)
        {
            return matrix[0, 0];
        }

        T det = NumOps.Zero;
        for (int j = 0; j < matrix.Columns; j++)
        {
            det = NumOps.Add(det, NumOps.Multiply(matrix[0, j], Cofactor(matrix, 0, j)));
        }

        return det;
    }

    /// <summary>
    /// Calculates the cofactor of a matrix element at the specified position.
    /// </summary>
    /// <remarks>
    /// A cofactor is the signed minor of an element. The minor is the determinant of the
    /// submatrix formed by removing the row and column of the specified element.
    /// The sign is determined by the position: positive if row+column is even, negative if odd.
    /// </remarks>
    /// <param name="matrix">The matrix to calculate the cofactor from.</param>
    /// <param name="row">The row index of the element.</param>
    /// <param name="col">The column index of the element.</param>
    /// <returns>The cofactor value.</returns>
    private T Cofactor(Matrix<T> matrix, int row, int col)
    {
        Matrix<T> minor = new(matrix.Rows - 1, matrix.Columns - 1);
        int m = 0;

        // VECTORIZED: Use row-based operations to construct minor matrix
        for (int i = 0; i < matrix.Rows; i++)
        {
            if (i == row) continue;

            Vector<T> sourceRow = matrix.GetRow(i);
            Vector<T> minorRow = new Vector<T>(sourceRow.Where((val, idx) => idx != col));
            minor.SetRow(m, minorRow);
            m++;
        }

        T sign = (row + col) % 2 == 0 ? NumOps.One : NumOps.Negate(NumOps.One);
        return NumOps.Multiply(sign, Determinant(minor));
    }

    /// <summary>
    /// Creates a copy of a matrix with one column replaced by a vector.
    /// </summary>
    /// <remarks>
    /// This helper method is used in Cramer's rule to create the modified matrices
    /// needed for calculating each variable in the solution.
    /// </remarks>
    /// <param name="original">The original matrix to copy.</param>
    /// <param name="column">The vector to insert as a column.</param>
    /// <param name="colIndex">The index of the column to replace.</param>
    /// <returns>A new matrix with the specified column replaced.</returns>
    private static Matrix<T> ReplaceColumn(Matrix<T> original, Vector<T> column, int colIndex)
    {
        Matrix<T> result = original.Clone();

        // VECTORIZED: Use SetColumn to replace entire column at once
        result.SetColumn(colIndex, column);

        return result;
    }
}
