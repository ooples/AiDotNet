using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

/// <summary>
/// Represents a UDU' decomposition of a matrix, which factorizes a symmetric matrix A into U*D*U',
/// where U is an upper triangular matrix with ones on the diagonal, D is a diagonal matrix,
/// and U' is the transpose of U.
/// </summary>
/// <typeparam name="T">The numeric type used in the matrix (e.g., double, float)</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> UDU' decomposition is a way to break down a complex matrix into simpler parts.
/// Think of it like factoring a number (e.g., 12 = 3 * 4). This decomposition is particularly
/// useful for solving systems of linear equations and for numerical stability in calculations.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.DimensionalityReduction)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
    [ResearchPaper("Matrix Computations", "https://doi.org/10.56021/9781421407944")]
public class UduDecomposition<T> : MatrixDecompositionBase<T>
{
    /// <summary>
    /// Gets the upper triangular matrix U from the decomposition.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> An upper triangular matrix has non-zero values only on and above the main diagonal
    /// (the diagonal from top-left to bottom-right). All values below this diagonal are zero.
    /// </para>
    /// </remarks>
    public Matrix<T> U { get; private set; } = new Matrix<T>(0, 0);

    /// <summary>
    /// Gets the diagonal matrix D represented as a vector of its diagonal elements.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A diagonal matrix has non-zero values only along its main diagonal.
    /// We store it as a vector to save memory since all other values are zero.
    /// </para>
    /// </remarks>
    public Vector<T> D { get; private set; } = new Vector<T>(0);

    private readonly UduAlgorithmType _algorithm;

    /// <summary>
    /// Initializes a new instance of the UduDecomposition class and performs the decomposition.
    /// </summary>
    /// <param name="matrix">The matrix to decompose</param>
    /// <param name="algorithm">The algorithm to use for decomposition (default is Crout)</param>
    /// <exception cref="ArgumentException">Thrown when the matrix is not square</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor takes your original matrix and immediately breaks it down
    /// into the U and D components. You can choose between two different methods (algorithms)
    /// for doing this decomposition.
    /// </para>
    /// </remarks>
    public UduDecomposition(Matrix<T> matrix, UduAlgorithmType algorithm = UduAlgorithmType.Crout)
        : base(matrix)
    {
        if (!matrix.IsSquareMatrix())
            throw new ArgumentException("Matrix must be square for UDU decomposition.");

        _algorithm = algorithm;

        Decompose();
    }

    /// <summary>
    /// Performs the UDU' decomposition.
    /// </summary>
    protected override void Decompose()
    {
        int n = A.Rows;
        U = new Matrix<T>(n, n);
        D = new Vector<T>(n);

        ComputeDecomposition(_algorithm);
    }

    /// <summary>
    /// Computes the UDU' decomposition using the specified algorithm.
    /// </summary>
    /// <param name="algorithm">The algorithm to use for decomposition</param>
    /// <exception cref="ArgumentException">Thrown when an unsupported algorithm is specified</exception>
    private void ComputeDecomposition(UduAlgorithmType algorithm)
    {
        switch (algorithm)
        {
            case UduAlgorithmType.Crout:
                DecomposeCrout();
                break;
            case UduAlgorithmType.Doolittle:
                DecomposeDoolittle();
                break;
            default:
                throw new ArgumentException("Unsupported UDU decomposition algorithm.");
        }
    }

    /// <summary>
    /// Performs the UDU' decomposition using the Crout algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Crout algorithm is one method for breaking down a matrix into simpler parts.
    /// It works by calculating the elements of U and D in a specific order, starting from the top-left
    /// and working across rows and down columns.
    /// </para>
    /// </remarks>
    private void DecomposeCrout()
    {
        (U, D) = ComputeUnpivotedUdu(A);
    }

    /// <summary>
    /// A = U D U^T (U unit upper triangular) from the unpivoted L D L^T of the index-reversed
    /// matrix: with J the exchange matrix, J A J = L' D' L'^T gives U = J L' J and D = J D'.
    /// </summary>
    private (Matrix<T> U, Vector<T> D) ComputeUnpivotedUdu(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        if (n == 0)
        {
            return (new Matrix<T>(0, 0), new Vector<T>(0));
        }

        var reversed = Engine.TensorFlip(Tensor<T>.FromMatrix(matrix), new[] { 0, 1 }).ToMatrix();
        var (lower, diagonal) = ComputeUnpivotedLdl(reversed);
        var upper = Engine.TensorFlip(Tensor<T>.FromMatrix(lower), new[] { 0, 1 }).ToMatrix();
        var d = Engine.TensorFlip(Tensor<T>.FromVector(diagonal), new[] { 0 }).ToVector();
        return (upper, d);
    }

    /// <summary>
    /// Performs the UDU' decomposition using the Doolittle algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Doolittle algorithm is another method for matrix decomposition.
    /// It's similar to Crout but processes the matrix in a slightly different order.
    /// Different algorithms may be more efficient or stable depending on the specific matrix.
    /// </para>
    /// </remarks>
    private void DecomposeDoolittle()
    {
        // Same factors as the Crout ordering (rows within a column are computed bottom-up).
        (U, D) = ComputeUnpivotedUdu(A);
    }

    /// <summary>
    /// Solves the linear system Ax = b using the UDU' decomposition.
    /// </summary>
    /// <param name="b">The right-hand side vector</param>
    /// <returns>The solution vector x</returns>
    /// <exception cref="ArgumentException">Thrown when the length of vector b doesn't match the matrix dimensions</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method finds the values of x in the equation Ax = b.
    /// Think of it like solving for x in the equation 3x = 6 (where x = 2),
    /// but with matrices instead of simple numbers. The decomposition makes
    /// this process more efficient by breaking it into three simpler steps:
    /// forward substitution, diagonal scaling, and backward substitution.
    /// </para>
    /// </remarks>
    public override Vector<T> Solve(Vector<T> b)
    {
        if (b.Length != A.Rows)
            throw new ArgumentException("Vector b must have the same length as the number of rows in matrix A.");

        if (b.Length == 0)
        {
            return new Vector<T>(0);
        }

        return SolveFactored(Tensor<T>.FromVector(b, new[] { b.Length, 1 }).ToMatrix()).GetColumn(0);
    }

    /// <summary>
    /// Solves A X = rhs with the stored factorization A = U D U^T: back substitution with U (unit
    /// upper), division by D, then forward substitution with U^T (unit lower), for every column of
    /// <paramref name="rhs"/> at once.
    /// </summary>
    private Matrix<T> SolveFactored(Matrix<T> rhs)
    {
        var u = Tensor<T>.FromMatrix(U);
        var y = Linalg.SolveTriangular(u, Tensor<T>.FromMatrix(rhs), upper: true, unitDiagonal: true);
        var d = Engine.TensorBroadcastTo(Tensor<T>.FromVector(D, new[] { D.Length, 1 }), new[] { rhs.Rows, rhs.Columns });
        y = Engine.TensorMultiply(y, Engine.TensorReciprocal(d));
        return Linalg.SolveTriangular(Engine.TensorTranspose(u), y, upper: false, unitDiagonal: true).ToMatrix();
    }

    /// <summary>
    /// Calculates the inverse of the original matrix A using the UDU' decomposition.
    /// </summary>
    /// <returns>The inverse of matrix A</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The inverse of a matrix is like the reciprocal of a number. Just as 1/3 is the
    /// reciprocal of 3 (because 3 * 1/3 = 1), the inverse of a matrix A is another matrix that,
    /// when multiplied by A, gives the identity matrix (the matrix equivalent of the number 1).
    /// This method finds the inverse by solving multiple equation systems, one for each column.
    /// </para>
    /// </remarks>
    public override Matrix<T> Invert()
    {
        int n = A.Rows;
        if (n == 0)
        {
            return new Matrix<T>(0, 0);
        }

        // Solve for every column of the identity at once.
        return SolveFactored(Matrix<T>.CreateIdentity(n));
    }

    /// <summary>
    /// Returns the U and D factors from the UDU' decomposition.
    /// </summary>
    /// <returns>A tuple containing the U matrix and D vector that represent the decomposition factors</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method gives you direct access to the two main components of the decomposition:
    /// the U matrix and the D vector. These are the "building blocks" that, when combined properly,
    /// can reconstruct your original matrix. Think of it like getting the individual ingredients
    /// that went into making a cake. You might need these separate components for further
    /// mathematical operations or to better understand the structure of your original matrix.
    /// </para>
    /// </remarks>
    public (Matrix<T> U, Vector<T> D) GetFactors()
    {
        return (U, D);
    }
}
