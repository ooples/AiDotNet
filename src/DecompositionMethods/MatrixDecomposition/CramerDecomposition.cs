using AiDotNet.Attributes;
using AiDotNet.Enums;

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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.DimensionalityReduction)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
    [ResearchPaper("Matrix Computations", "https://doi.org/10.56021/9781421407944")]
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

        int n = A.Columns;
        if (n == 0)
        {
            return new Vector<T>(0);
        }

        // Batch i is A with column i replaced by b: select b where the column index equals the
        // batch index, then take all n determinants in one batched engine call.
        var shape = new[] { n, n, n };
        var stackedA = Engine.TensorBroadcastTo(Tensor<T>.FromMatrix(A).Reshape(new[] { 1, n, n }), shape);
        var stackedB = Engine.TensorBroadcastTo(Tensor<T>.FromVector(b, new[] { 1, n, 1 }), shape);
        var columnMask = Engine.TensorBroadcastTo(Engine.TensorEye<T>(n).Reshape(new[] { n, 1, n }), shape);
        var replaced = Engine.TensorWhere(columnMask, stackedB, stackedA);

        var determinants = LinalgDispatch<T>.Det(replaced);
        return Engine.TensorDivideScalar(determinants, detA).Reshape(new[] { n }).ToVector();
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

        if (A.Rows == 0)
        {
            return new Matrix<T>(0, 0);
        }

        // adj(A) / det(A) is the inverse; the engine computes it in one whole-matrix call.
        return LinalgDispatch<T>.Inv(Tensor<T>.FromMatrix(A)).ToMatrix();
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
        if (matrix.Rows == 0)
        {
            return NumOps.One;
        }

        return LinalgDispatch<T>.Det(Tensor<T>.FromMatrix(matrix))[0];
    }
}
