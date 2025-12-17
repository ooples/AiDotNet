namespace AiDotNet.Factories;

/// <summary>
/// A factory class that creates matrix decomposition objects for solving linear algebra problems.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Matrix decomposition is a way of breaking down a complex matrix into simpler
/// components that are easier to work with mathematically. It's like factoring a number (e.g., 12 = 3 × 4),
/// but for matrices.
/// </para>
/// <para>
/// This factory helps you create different types of matrix decompositions without needing to know their 
/// internal implementation details. Think of it like ordering a specific tool from a catalog - you just 
/// specify what you need, and the factory provides it.
/// </para>
/// </remarks>
public static class MatrixDecompositionFactory
{
    /// <summary>
    /// Creates a matrix decomposition of the specified type for the given matrix.
    /// </summary>
    /// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
    /// <param name="matrix">The matrix to decompose.</param>
    /// <param name="decompositionType">The type of decomposition to create.</param>
    /// <returns>An implementation of IMatrixDecomposition<T> for the specified decomposition type.</returns>
    /// <exception cref="ArgumentException">Thrown when an unsupported decomposition type is specified.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Different decomposition types are useful for different kinds of problems. 
    /// For example, some are better for solving systems of equations, while others are better for finding 
    /// eigenvalues or reducing computational complexity.
    /// </para>
    /// <para>
    /// Available decomposition types include:
    /// <list type="bullet">
    /// <item><description>Lu: Decomposes a matrix into a lower triangular matrix (L) and an upper triangular matrix (U).</description></item>
    /// <item><description>Qr: Decomposes a matrix into an orthogonal matrix (Q) and an upper triangular matrix (R).</description></item>
    /// <item><description>Cholesky: A specialized decomposition for symmetric, positive-definite matrices.</description></item>
    /// <item><description>Svd: Singular Value Decomposition, useful for dimensionality reduction and data analysis.</description></item>
    /// <item><description>Cramer: Uses Cramer's rule to solve systems of linear equations.</description></item>
    /// <item><description>Eigen: Finds the eigenvalues and eigenvectors of a matrix.</description></item>
    /// <item><description>Schur: Transforms a matrix into an upper triangular form.</description></item>
    /// <item><description>Takagi: A decomposition for complex symmetric matrices.</description></item>
    /// <item><description>Polar: Decomposes a matrix into a product of a unitary matrix and a positive semi-definite Hermitian matrix.</description></item>
    /// <item><description>Hessenberg: Transforms a matrix into Hessenberg form (almost triangular).</description></item>
    /// <item><description>Tridiagonal: Transforms a matrix into a tridiagonal form.</description></item>
    /// <item><description>Bidiagonal: Transforms a matrix into a bidiagonal form.</description></item>
    /// <item><description>Ldl: A variant of Cholesky decomposition for indefinite matrices.</description></item>
    /// <item><description>Udu: A decomposition that produces a unit upper triangular matrix, a diagonal matrix, and another unit upper triangular matrix.</description></item>
    /// <item><description>Nmf: Non-negative Matrix Factorization, decomposes a non-negative matrix into two non-negative matrices.</description></item>
    /// <item><description>Ica: Independent Component Analysis, separates mixed signals into statistically independent components.</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    public static IMatrixDecomposition<T> CreateDecomposition<T>(Matrix<T> matrix, MatrixDecompositionType decompositionType)
    {
        return decompositionType switch
        {
            MatrixDecompositionType.Lu => new LuDecomposition<T>(matrix),
            MatrixDecompositionType.Qr => new QrDecomposition<T>(matrix),
            MatrixDecompositionType.Cholesky => new CholeskyDecomposition<T>(matrix),
            MatrixDecompositionType.Svd => new SvdDecomposition<T>(matrix),
            MatrixDecompositionType.Cramer => new CramerDecomposition<T>(matrix),
            MatrixDecompositionType.Eigen => new EigenDecomposition<T>(matrix),
            MatrixDecompositionType.Schur => new SchurDecomposition<T>(matrix),
            MatrixDecompositionType.Takagi => new TakagiDecomposition<T>(matrix),
            MatrixDecompositionType.Polar => new PolarDecomposition<T>(matrix),
            MatrixDecompositionType.Hessenberg => new HessenbergDecomposition<T>(matrix),
            MatrixDecompositionType.Tridiagonal => new TridiagonalDecomposition<T>(matrix),
            MatrixDecompositionType.Bidiagonal => new BidiagonalDecomposition<T>(matrix),
            MatrixDecompositionType.Ldl => new LdlDecomposition<T>(matrix),
            MatrixDecompositionType.Udu => new UduDecomposition<T>(matrix),
            MatrixDecompositionType.Nmf => new NmfDecomposition<T>(matrix),
            MatrixDecompositionType.Ica => new IcaDecomposition<T>(matrix),
            _ => throw new ArgumentException($"Unsupported decomposition type: {decompositionType}")
        };
    }

    /// <summary>
    /// Gets the decomposition type from an existing matrix decomposition object.
    /// </summary>
    /// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
    /// <param name="decomposition">The matrix decomposition object to identify.</param>
    /// <returns>The type of the provided matrix decomposition.</returns>
    /// <exception cref="ArgumentException">Thrown when an unsupported decomposition object is provided.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method does the reverse of CreateDecomposition - it takes a decomposition 
    /// object and tells you what type it is. This is useful when you have a decomposition object but don't 
    /// know its specific type.
    /// </para>
    /// <para>
    /// This method is particularly helpful when you need to serialize/deserialize decomposition objects or 
    /// when you're working with decompositions created elsewhere in your code.
    /// </para>
    /// </remarks>
    public static MatrixDecompositionType GetDecompositionType<T>(IMatrixDecomposition<T>? decomposition)
    {
        return decomposition switch
        {
            LuDecomposition<T> => MatrixDecompositionType.Lu,
            QrDecomposition<T> => MatrixDecompositionType.Qr,
            CholeskyDecomposition<T> => MatrixDecompositionType.Cholesky,
            SvdDecomposition<T> => MatrixDecompositionType.Svd,
            CramerDecomposition<T> => MatrixDecompositionType.Cramer,
            EigenDecomposition<T> => MatrixDecompositionType.Eigen,
            SchurDecomposition<T> => MatrixDecompositionType.Schur,
            TakagiDecomposition<T> => MatrixDecompositionType.Takagi,
            PolarDecomposition<T> => MatrixDecompositionType.Polar,
            HessenbergDecomposition<T> => MatrixDecompositionType.Hessenberg,
            TridiagonalDecomposition<T> => MatrixDecompositionType.Tridiagonal,
            BidiagonalDecomposition<T> => MatrixDecompositionType.Bidiagonal,
            LdlDecomposition<T> => MatrixDecompositionType.Ldl,
            UduDecomposition<T> => MatrixDecompositionType.Udu,
            NmfDecomposition<T> => MatrixDecompositionType.Nmf,
            IcaDecomposition<T> => MatrixDecompositionType.Ica,
            _ => throw new ArgumentException($"Unsupported decomposition type: {decomposition?.GetType().Name}")
        };
    }
}
