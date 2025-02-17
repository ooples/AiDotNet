namespace AiDotNet.Extensions;

public static class MatrixExtensions
{
    public static Matrix<T> AddConstantColumn<T>(this Matrix<T> matrix, T value)
    {
        var newMatrix = new Matrix<T>(matrix.Rows, matrix.Columns + 1);
        for (int i = 0; i < matrix.Rows; i++)
        {
            newMatrix[i, 0] = value;
            for (int j = 0; j < matrix.Columns; j++)
            {
                newMatrix[i, j + 1] = matrix[i, j];
            }
        }

        return newMatrix;
    }

    public static Matrix<T> AddVectorToEachRow<T>(this Matrix<T> matrix, Vector<T> vector)
    {
        if (matrix.Columns != vector.Length)
            throw new ArgumentException("Vector length must match matrix column count");

        var ops = MathHelper.GetNumericOperations<T>();
        var result = new Matrix<T>(matrix.Rows, matrix.Columns);
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                result[i, j] = ops.Add(matrix[i, j], vector[j]);
            }
        }

        return result;
    }

    public static Vector<T> SumColumns<T>(this Matrix<T> matrix)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(matrix.Columns);
        for (int j = 0; j < matrix.Columns; j++)
        {
            T sum = ops.Zero;
            for (int i = 0; i < matrix.Rows; i++)
            {
                sum = ops.Add(sum, matrix[i, j]);
            }

            result[j] = sum;
        }

        return result;
    }

    public static Vector<T> GetColumn<T>(this Matrix<T> matrix, int columnIndex)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var column = new Vector<T>(matrix.Rows);
        for (int i = 0; i < matrix.Rows; i++)
        {
            column[i] = matrix[i, columnIndex];
        }

        return column;
    }

    public static Vector<T> BackwardSubstitution<T>(this Matrix<T> aMatrix, Vector<T> bVector)
    {
        int n = aMatrix.Rows;
        var xVector = new Vector<T>(n);
        var ops = MathHelper.GetNumericOperations<T>();
        for (int i = n - 1; i >= 0; --i)
        {
            xVector[i] = bVector[i];
            for (int j = i + 1; j < n; ++j)
            {
                xVector[i] = ops.Subtract(xVector[i], ops.Multiply(aMatrix[i, j], xVector[j]));
            }
            xVector[i] = ops.Divide(xVector[i], aMatrix[i, i]);
        }

        return xVector;
    }

    public static IEnumerable<MatrixType> GetMatrixTypes<T>(this Matrix<T> matrix, IMatrixDecomposition<T> matrixDecomposition, 
    T? tolerance = default, int subDiagonalThreshold = 1, int superDiagonalThreshold = 1, T? sparsityThreshold = default, T? denseThreshold = default,
    int blockRows = 2, int blockCols = 2)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        tolerance ??= ops.FromDouble(1e-10);
        sparsityThreshold ??= ops.FromDouble(0.5);
        denseThreshold ??= ops.FromDouble(0.5);

        var complexMatrix = matrix.ToComplexMatrix();
        foreach (var matrixType in EnumHelper.GetEnumValues<MatrixType>())
        {
            switch (matrixType)
            {
                case MatrixType.Square:
                    if (matrix.IsSquareMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Rectangular:
                    if (matrix.IsRectangularMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Diagonal:
                    if (matrix.IsDiagonalMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Identity:
                    if (matrix.IsIdentityMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Symmetric:
                    if (matrix.IsSymmetricMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.SkewSymmetric:
                    if (matrix.IsSkewSymmetricMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.UpperTriangular:
                    if (matrix.IsUpperTriangularMatrix(tolerance))
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.LowerTriangular:
                    if (matrix.IsLowerTriangularMatrix(tolerance))
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Zero:
                    if (matrix.IsZeroMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Scalar:
                    if (matrix.IsScalarMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.UpperBidiagonal:
                    if (matrix.IsUpperBidiagonalMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.LowerBidiagonal:
                    if (matrix.IsLowerBidiagonalMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Tridiagonal:
                    if (matrix.IsTridiagonalMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Band:
                    if (matrix.IsBandMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Hermitian:
                    if (complexMatrix.IsHermitianMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.SkewHermitian:
                    if (complexMatrix.IsSkewHermitianMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Orthogonal:
                    if (matrix.IsOrthogonalMatrix(matrixDecomposition))
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Unitary:
                    var complexDecomposition = matrixDecomposition.ToComplexDecomposition();
                    if (complexMatrix.IsUnitaryMatrix(complexDecomposition))
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Singular:
                    if (matrix.IsSingularMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.NonSingular:
                    if (matrix.IsNonSingularMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.PositiveDefinite:
                    if (matrix.IsPositiveDefiniteMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.PositiveSemiDefinite:
                    if (matrix.IsPositiveSemiDefiniteMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.OrthogonalProjection:
                    if (matrix.IsOrthogonalProjectionMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Idempotent:
                    if (matrix.IsIdempotentMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Involutory:
                    if (matrix.IsInvolutoryMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Stochastic:
                    if (matrix.IsStochasticMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.DoublyStochastic:
                    if (matrix.IsDoublyStochasticMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Permutation:
                    if (matrix.IsPermutationMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Adjacency:
                    if (matrix.IsAdjacencyMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Incidence:
                    if (matrix.IsIncidenceMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Laplacian:
                    if (matrix.IsLaplacianMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Toeplitz:
                    if (matrix.IsToeplitzMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Hankel:
                    if (matrix.IsHankelMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Circulant:
                    if (matrix.IsCirculantMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Block:
                    if (matrix.IsBlockMatrix(blockRows, blockCols))
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Sparse:
                    if (matrix.IsSparseMatrix(sparsityThreshold))
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Dense:
                    if (matrix.IsDenseMatrix(denseThreshold))
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Partitioned:
                    if (matrix.IsPartitionedMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Companion:
                    if (matrix.IsCompanionMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Vandermonde:
                    if (matrix.IsVandermondeMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Hilbert:
                    if (matrix.IsHilbertMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                case MatrixType.Cauchy:
                    if (matrix.IsCauchyMatrix())
                    {
                        yield return matrixType;
                    }
                    break;
                default:
                    yield return matrixType;
                    break;
            }
        }
    }

    public static bool IsBlockMatrix<T>(this Matrix<T> matrix, int blockRows, int blockCols)
    {
        var rows = matrix.Rows;
        var cols = matrix.Columns;

        // Check if the matrix dimensions are compatible with the block size
        if (rows % blockRows != 0 || cols % blockCols != 0)
        {
            return false;
        }

        int numBlockRows = rows / blockRows;
        int numBlockCols = cols / blockCols;

        // Check if each block is consistent
        for (int blockRow = 0; blockRow < numBlockRows; blockRow++)
        {
            for (int blockCol = 0; blockCol < numBlockCols; blockCol++)
            {
                var block = matrix.GetBlock(blockRow * blockRows, blockCol * blockCols, blockRows, blockCols);
                if (!block.IsConsistentBlock())
                {
                    return false;
                }
            }
        }

        return true;
    }

    public static Matrix<T> GetBlock<T>(this Matrix<T> matrix, int startRow, int startCol, int blockRows, int blockCols)
    {
        var block = new Matrix<T>(blockRows, blockCols);
        for (int i = 0; i < blockRows; i++)
        {
            for (int j = 0; j < blockCols; j++)
            {
                block[i, j] = matrix[startRow + i, startCol + j];
            }
        }

        return block;
    }

    public static bool IsConsistentBlock<T>(this Matrix<T> block)
    {
        var rows = block.Rows;
        var cols = block.Columns;
        var ops = MathHelper.GetNumericOperations<T>();
        
        // Implement your own logic to check consistency within the block
        // For simplicity, let's assume the block is consistent if all elements are the same
        T firstElement = block[0, 0];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (!ops.Equals(block[i, j], firstElement))
                {
                    return false;
                }
            }
        }

        return true;
    }

    public static bool IsUpperTriangularMatrix<T>(this Matrix<T> matrix, T? tolerance = default)
    {
        var ops = MathHelper.GetNumericOperations<T>();
    
        // If tolerance is not provided (i.e., it's default(T)), use a small value based on the type
        if (tolerance?.Equals(default(T)) ?? true)
        {
            tolerance = ops.FromDouble(1e-10); // Use a small value as default tolerance
        }

        for (int i = 1; i < matrix.Rows; i++)
        {
            for (int j = 0; j < i; j++)
            {
                // If any element below the main diagonal is non-zero (greater than tolerance), the matrix is not upper triangular.
                if (ops.GreaterThan(ops.Abs(matrix[i, j]), tolerance))
                {
                    return false;
                }
            }
        }

        return true;
    }

    public static bool IsSparseMatrix<T>(this Matrix<T> matrix, T? sparsityThreshold = default)
    {
        var NumOps = MathHelper.GetNumericOperations<T>();
        sparsityThreshold ??= NumOps.FromDouble(0.5);
        int rows = matrix.Rows;
        int cols = matrix.Columns;
        int totalElements = rows * cols;
        int zeroCount = 0;

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (NumOps.Equals(matrix[i, j], NumOps.Zero))
                {
                    zeroCount++;
                }
            }
        }

        T sparsityRatio = NumOps.Divide(NumOps.FromDouble(zeroCount), NumOps.FromDouble(totalElements));
        return NumOps.GreaterThanOrEquals(sparsityRatio, sparsityThreshold);
    }

    public static bool IsDenseMatrix<T>(this Matrix<T> matrix, T? denseThreshold = default)
    {
        var NumOps = MathHelper.GetNumericOperations<T>();
        denseThreshold ??= NumOps.FromDouble(0.5);
        int rows = matrix.Rows;
        int cols = matrix.Columns;
        int totalElements = rows * cols;
        int nonZeroCount = 0;

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (!NumOps.Equals(matrix[i, j], NumOps.Zero))
                {
                    nonZeroCount++;
                }
            }
        }

        T densityRatio = NumOps.Divide(NumOps.FromDouble(nonZeroCount), NumOps.FromDouble(totalElements));
        return NumOps.GreaterThanOrEquals(densityRatio, denseThreshold);
    }

    public static bool IsLowerTriangularMatrix<T>(this Matrix<T> matrix, T? tolerance = default)
    {
        var NumOps = MathHelper.GetNumericOperations<T>();
        tolerance ??= NumOps.FromDouble(1e-10);
        var rows = matrix.Rows;

        for (int i = 0; i < rows - 1; i++)
        {
            for (int j = i + 1; j < rows; j++)
            {
                // If any element above the main diagonal is non-zero, the matrix is not lower triangular.
                if (NumOps.GreaterThan(NumOps.Abs(matrix[i, j]), tolerance))
                {
                    return false;
                }
            }
        }

        return true;
    }

    public static bool IsSquareMatrix<T>(this Matrix<T> matrix)
    {
        return matrix.Rows == matrix.Columns;
    }

    public static bool IsRectangularMatrix<T>(this Matrix<T> matrix)
    {
        return matrix.Rows != matrix.Columns;
    }

    public static bool IsSymmetricMatrix<T>(this Matrix<T> matrix)
    {
        var rows = matrix.Rows;
        var ops = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                if (!ops.Equals(matrix[i, j], matrix[j, i]))
                {
                    return false;
                }
            }
        }

        return true;
    }

    public static bool IsDiagonalMatrix<T>(this Matrix<T> matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<T>();

        if (rows != cols)
        {
            return false; // A diagonal matrix must be square
        }

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (i != j && !ops.Equals(matrix[i, j], ops.Zero))
                {
                    return false;
                }
            }
        }

        return true;
    }

    public static bool IsIdentityMatrix<T>(this Matrix<T> matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<T>();

        if (rows != cols)
        {
            return false; // An identity matrix must be square
        }

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (i == j)
                {
                    if (!ops.Equals(matrix[i, j], ops.One))
                    {
                        return false; // Diagonal elements must be 1
                    }
                }
                else
                {
                    if (!ops.Equals(matrix[i, j], ops.Zero))
                    {
                        return false; // Non-diagonal elements must be 0
                    }
                }
            }
        }

        return true;
    }

    public static bool IsSkewSymmetricMatrix<T>(this Matrix<T> matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<T>();

        if (rows != cols)
        {
            return false; // A skew-symmetric matrix must be square
        }

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (!ops.Equals(matrix[i, j], ops.Negate(matrix[j, i])))
                {
                    return false;
                }
            }
        }

        return true;
    }

    public static bool IsScalarMatrix<T>(this Matrix<T> matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<T>();

        if (rows != cols)
        {
            return false; // A scalar matrix must be square
        }

        T diagonalValue = matrix[0, 0];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (i == j)
                {
                    if (!ops.Equals(matrix[i, j], diagonalValue))
                    {
                        return false; // All diagonal elements must be equal
                    }
                }
                else
                {
                    if (!ops.Equals(matrix[i, j], ops.Zero))
                    {
                        return false; // All off-diagonal elements must be zero
                    }
                }
            }
        }

        return true;
    }

    public static bool IsUpperBidiagonalMatrix<T>(this Matrix<T> matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<T>();

        if (rows != cols)
        {
            return false; // An upper bidiagonal matrix must be square
        }

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (i > j || j > i + 1)
                {
                    if (!ops.Equals(matrix[i, j], ops.Zero))
                    {
                        return false; // Elements below the main diagonal or above the first superdiagonal must be zero
                    }
                }
            }
        }

        return true;
    }

    public static bool IsLowerBidiagonalMatrix<T>(this Matrix<T> matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<T>();

        if (rows != cols)
        {
            return false; // A lower bidiagonal matrix must be square
        }

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (i < j || i > j + 1)
                {
                    if (!ops.Equals(matrix[i, j], ops.Zero))
                    {
                        return false; // Elements above the main diagonal or below the first subdiagonal must be zero
                    }
                }
            }
        }

        return true;
    }

    public static bool IsTridiagonalMatrix<T>(this Matrix<T> matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<T>();

        if (rows != cols)
        {
            return false; // A tridiagonal matrix must be square
        }

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (Math.Abs(i - j) > 1 && !ops.Equals(matrix[i, j], ops.Zero))
                {
                    return false; // Elements outside the three diagonals must be zero
                }
            }
        }

        return true;
    }

    public static bool IsBandMatrix<T>(this Matrix<T> matrix, int subDiagonalThreshold = 1, int superDiagonalThreshold = 1)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<T>();

        if (rows != cols)
        {
            return false; // A band matrix must be square
        }

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (Math.Abs(i - j) > subDiagonalThreshold || Math.Abs(i - j) > superDiagonalThreshold)
                {
                    if (!ops.Equals(matrix[i, j], ops.Zero))
                    {
                        return false; // Elements outside the specified bands must be zero
                    }
                }
            }
        }

        return true;
    }

    public static bool IsHermitianMatrix<T>(this Matrix<Complex<T>> matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;

        if (rows != cols)
        {
            return false; // A Hermitian matrix must be square
        }

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (matrix[i, j] != matrix[j, i].Conjugate())
                {
                    return false; // Check if element is equal to its conjugate transpose
                }
            }
        }

        return true;
    }

    public static bool IsSkewHermitianMatrix<T>(this Matrix<Complex<T>> matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;

        if (rows != cols)
        {
            return false; // A skew-Hermitian matrix must be square
        }

        var ops = MathHelper.GetNumericOperations<Complex<T>>();

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (!ops.Equals(matrix[i, j], ops.Negate(matrix[j, i].Conjugate())))
                {
                    return false; // Check if element is equal to the negation of its conjugate transpose
                }
            }
        }

        return true;
    }

    public static bool IsOrthogonalMatrix<T>(this Matrix<T> matrix, IMatrixDecomposition<T> matrixDecomposition)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;

        if (rows != cols)
        {
            return false; // An orthogonal matrix must be square
        }

        var transpose = matrix.Transpose();
        var inverse = matrixDecomposition.Invert();

        var ops = MathHelper.GetNumericOperations<T>();

        // Check if transpose is equal to inverse
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (ops.GreaterThan(ops.Abs(ops.Subtract(transpose[i, j], inverse[i, j])), ops.FromDouble(0.0001)))
                {
                    return false;
                }
            }
        }

        return true;
    }

    public static Matrix<Complex<T>> CreateComplexMatrix<T>(this Matrix<Complex<T>> matrix, int rows, int cols)
    {
        return new Matrix<Complex<T>>(rows, cols);
    }

    public static Matrix<Complex<T>> ConjugateTranspose<T>(this Matrix<Complex<T>> matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;
        var result = matrix.CreateComplexMatrix(cols, rows);

        for (int i = 0; i < cols; i++)
        {
            for (int j = 0; j < rows; j++)
            {
                result[i, j] = matrix[j, i].Conjugate();
            }
        }

        return result;
    }

    public static bool IsSingularMatrix<T>(this Matrix<T> matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;

        if (rows != cols)
        {
            return false; // A singular matrix must be square
        }

        // Calculate the determinant
        T determinant = matrix.GetDeterminant();

        // If determinant is zero, the matrix is singular
        var ops = MathHelper.GetNumericOperations<T>();
        return ops.LessThan(ops.Abs(determinant), ops.FromDouble(1e-10));
    }

    public static bool IsNonSingularMatrix<T>(this Matrix<T> matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;

        if (rows != cols)
        {
            return false; // A singular matrix must be square
        }

        // Calculate the determinant
        T determinant = matrix.GetDeterminant();

        // If determinant is not zero, the matrix is non-singular
        var ops = MathHelper.GetNumericOperations<T>();
        return ops.GreaterThanOrEquals(ops.Abs(determinant), ops.FromDouble(1e-10));
    }

    public static bool IsPositiveDefiniteMatrix<T>(this Matrix<T> matrix, T? tolerance = default)
    {
        if (!matrix.IsSymmetricMatrix())
        {
            return false; // Positive definite matrices must be symmetric
        }

        var ops = MathHelper.GetNumericOperations<T>();

        // Set default tolerance if not provided
        tolerance ??= ops.FromDouble(1e-10);

        try
        {
            // Attempt Cholesky decomposition
            var choleskyDecomposition = new CholeskyDecomposition<T>(matrix);

            // Get the lower triangular matrix L
            var L = choleskyDecomposition.L;

            // Check if all diagonal elements of L are positive
            for (int i = 0; i < L.Rows; i++)
            {
                if (ops.LessThanOrEquals(L[i, i], tolerance))
                {
                    return false;
                }
            }

            return true;
        }
        catch (Exception)
        {
            // If Cholesky decomposition fails, the matrix is not positive definite
            return false;
        }
    }

    public static bool IsIdempotentMatrix<T>(this Matrix<T> matrix)
    {
        var rows = matrix.Rows;
        var product = matrix.Multiply(matrix);
        var ops = MathHelper.GetNumericOperations<T>();

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < rows; j++)
            {
                if (ops.GreaterThan(ops.Abs(ops.Subtract(matrix[i, j], product[i, j])), ops.FromDouble(1e-10)))
                {
                    return false;
                }
            }
        }

        return true;
    }

    public static bool IsStochasticMatrix<T>(this Matrix<T> matrix)
    {
        var rows = matrix.Rows;
        var cols = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<T>();

        for (int i = 0; i < rows; i++)
        {
            T rowSum = ops.Zero;
            for (int j = 0; j < cols; j++)
            {
                if (ops.LessThan(matrix[i, j], ops.Zero))
                {
                    return false; // All elements must be non-negative
                }
                rowSum = ops.Add(rowSum, matrix[i, j]);
            }
            if (ops.GreaterThan(ops.Abs(ops.Subtract(rowSum, ops.One)), ops.FromDouble(1e-10)))
            {
                return false; // Each row must sum to 1
            }
        }

        return true;
    }

    public static bool IsDoublyStochasticMatrix<T>(this Matrix<T> matrix)
    {
        var rows = matrix.Rows;
        var cols = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<T>();

        // Check if matrix is square
        if (!matrix.IsSquareMatrix())
        {
            return false;
        }

        // Check if all elements are non-negative and rows sum to 1
        for (int i = 0; i < rows; i++)
        {
            T rowSum = ops.Zero;
            for (int j = 0; j < cols; j++)
            {
                if (ops.LessThan(matrix[i, j], ops.Zero))
                {
                    return false; // All elements must be non-negative
                }
                rowSum = ops.Add(rowSum, matrix[i, j]);
            }
            if (ops.GreaterThan(ops.Abs(ops.Subtract(rowSum, ops.One)), ops.FromDouble(1e-10)))
            {
                return false; // Each row must sum to 1
            }
        }

        // Check if columns sum to 1
        for (int j = 0; j < cols; j++)
        {
            T colSum = ops.Zero;
            for (int i = 0; i < rows; i++)
            {
                colSum = ops.Add(colSum, matrix[i, j]);
            }
            if (ops.GreaterThan(ops.Abs(ops.Subtract(colSum, ops.One)), ops.FromDouble(1e-10)))
            {
                return false; // Each column must sum to 1
            }
        }

        return true;
    }

    public static bool IsAdjacencyMatrix<T>(this Matrix<T> matrix)
    {
        var rows = matrix.Rows;
        var cols = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<T>();

        // Check if the matrix is square
        if (rows != cols)
        {
            return false;
        }

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                // Check if the element is either 0 or 1
                if (!ops.Equals(matrix[i, j], ops.Zero) && !ops.Equals(matrix[i, j], ops.One))
                {
                    return false;
                }
            }
        }

        return true;
    }

    public static bool IsCirculantMatrix<T>(this Matrix<T> matrix)
    {
        var rows = matrix.Rows;
        var cols = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<T>();

        // Check if the matrix is square
        if (!matrix.IsSquareMatrix())
        {
            return false;
        }

        // Check if each row is a cyclic shift of the previous row
        for (int i = 1; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (!ops.Equals(matrix[i, j], matrix[0, (j + i) % cols]))
                {
                    return false;
                }
            }
        }

        return true;
    }

    public static bool IsPartitionedMatrix<T>(this Matrix<T> matrix)
    {
        var rows = matrix.Rows;
        var cols = matrix.Columns;

        if (rows == 0 || cols == 0)
        {
            return false;
        }

        return rows % (int)Math.Sqrt(cols) == 0;
    }

    public static bool IsVandermondeMatrix<T>(this Matrix<T> matrix)
    {
        var rows = matrix.Rows;
        var cols = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<T>();

        // Check if the matrix has at least 2 rows and if each column forms a geometric progression
        for (int j = 1; j < cols; j++)
        {
            for (int i = 1; i < rows; i++)
            {
                // Check if the current element is equal to the previous element multiplied by x_i
                T expectedValue = ops.Multiply(matrix[i - 1, j], matrix[i, 0]);
                if (ops.GreaterThan(ops.Abs(ops.Subtract(matrix[i, j], expectedValue)), ops.FromDouble(1e-10)))
                {
                    return false;
                }
            }
        }

        return true;
    }

    public static bool IsCauchyMatrix<T>(this Matrix<T> matrix)
    {
        var rows = matrix.Rows;
        var cols = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<T>();

        // Check if the matrix is square
        if (!matrix.IsSquareMatrix())
        {
            return false;
        }

        // Check if each element satisfies the Cauchy matrix definition
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                T element = matrix[i, j];
                T x = matrix[i, 0];
                T y = matrix[0, j];

                // Avoid division by zero
                if (ops.LessThan(ops.Abs(ops.Subtract(x, y)), ops.FromDouble(1e-10)))
                {
                    return false;
                }

                T expectedValue = ops.Divide(ops.One, ops.Subtract(x, y));
                if (ops.GreaterThan(ops.Abs(ops.Subtract(element, expectedValue)), ops.FromDouble(1e-10)))
                {
                    return false;
                }
            }
        }

        return true;
    }

    public static bool IsHilbertMatrix<T>(this Matrix<T> matrix)
    {
        var rows = matrix.Rows;
        var ops = MathHelper.GetNumericOperations<T>();

        // Check if the matrix is square
        if (!matrix.IsSquareMatrix())
        {
            return false;
        }

        // Check if the elements satisfy the Hilbert matrix definition
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < rows; j++)
            {
                T expectedValue = ops.Divide(ops.One, ops.FromDouble(i + j + 1)); // Hilbert matrix definition
                if (ops.GreaterThan(ops.Abs(ops.Subtract(matrix[i, j], expectedValue)), ops.FromDouble(1e-10)))
                {
                    return false;
                }
            }
        }

        return true;
    }

    public static bool IsCompanionMatrix<T>(this Matrix<T> matrix)
    {
        var rows = matrix.Rows;
        var cols = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<T>();

        // Check if the matrix is square
        if (!matrix.IsSquareMatrix())
        {
            return false;
        }

        // Check if the first row contains the coefficients of a polynomial in reverse order
        for (int j = 0; j < cols - 1; j++)
        {
            if (!ops.Equals(matrix[0, j], ops.Zero) && !ops.Equals(matrix[0, j], ops.One))
            {
                return false;
            }
        }
        if (!ops.Equals(matrix[0, cols - 1], ops.One))
        {
            return false;
        }

        // Check if each subdiagonal contains a 1
        for (int i = 1; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (i == j + 1 && !ops.Equals(matrix[i, j], ops.One))
                {
                    return false;
                }
                if (i != j + 1 && !ops.Equals(matrix[i, j], ops.Zero))
                {
                    return false;
                }
            }
        }

        return true;
    }

    public static bool IsHankelMatrix<T>(this Matrix<T> matrix)
    {
        var rows = matrix.Rows;
        var cols = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<T>();

        // Check if each element is the same as the one diagonally above and to the right of it
        for (int i = 0; i < rows - 1; i++)
        {
            for (int j = 0; j < cols - 1; j++)
            {
                if (!ops.Equals(matrix[i, j], matrix[i + 1, j + 1]))
                {
                    return false;
                }
            }
        }

        return true;
    }

    public static bool IsToeplitzMatrix<T>(this Matrix<T> matrix)
    {
        var rows = matrix.Rows;
        var cols = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<T>();

        // Check if each element is the same as the one diagonally above it
        for (int i = 1; i < rows; i++)
        {
            for (int j = 1; j < cols; j++)
            {
                if (!ops.Equals(matrix[i, j], matrix[i - 1, j - 1]))
                {
                    return false;
                }
            }
        }

        return true;
    }

    public static bool IsLaplacianMatrix<T>(this Matrix<T> matrix)
    {
        var rows = matrix.Rows;
        var ops = MathHelper.GetNumericOperations<T>();

        // Check if the matrix is square
        if (!matrix.IsSquareMatrix())
        {
            return false;
        }

        for (int i = 0; i < rows; i++)
        {
            T rowSum = ops.Zero;
            T colSum = ops.Zero;

            for (int j = 0; j < rows; j++)
            {
                // Check for symmetry
                if (!ops.Equals(matrix[i, j], matrix[j, i]))
                {
                    return false;
                }

                // Check if off-diagonal elements are non-positive
                if (i != j && ops.GreaterThan(matrix[i, j], ops.Zero))
                {
                    return false;
                }

                // Sum the row and column elements
                rowSum = ops.Add(rowSum, matrix[i, j]);
                colSum = ops.Add(colSum, matrix[j, i]);
            }

            // Check if the sum of each row and column is zero
            if (ops.GreaterThan(ops.Abs(rowSum), ops.FromDouble(1e-10)) || 
                ops.GreaterThan(ops.Abs(colSum), ops.FromDouble(1e-10)))
            {
                return false;
            }

            // Check if diagonal elements are non-negative and equal to the sum of absolute values of the off-diagonal elements
            T offDiagonalSum = ops.Zero;
            for (int j = 0; j < rows; j++)
            {
                if (i != j)
                {
                    offDiagonalSum = ops.Add(offDiagonalSum, ops.Abs(matrix[i, j]));
                }
            }
            if (!ops.Equals(matrix[i, i], offDiagonalSum))
            {
                return false;
            }
        }

        return true;
    }

    public static bool IsIncidenceMatrix<T>(this Matrix<T> matrix)
    {
        var rows = matrix.Rows; // number of vertices
        var cols = matrix.Columns; // number of edges
        var ops = MathHelper.GetNumericOperations<T>();

        for (int j = 0; j < cols; j++)
        {
            int countOnes = 0;

            for (int i = 0; i < rows; i++)
            {
                // Check if the element is either 0 or 1
                if (!ops.Equals(matrix[i, j], ops.Zero) && !ops.Equals(matrix[i, j], ops.One))
                {
                    return false;
                }

                // Count the number of 1's in the current column
                if (ops.Equals(matrix[i, j], ops.One))
                {
                    countOnes++;
                }
            }

            // Each column must have exactly two 1's for an undirected graph
            if (countOnes != 2)
            {
                return false;
            }
        }

        return true;
    }

    public static bool IsPermutationMatrix<T>(this Matrix<T> matrix)
    {
        var rows = matrix.Rows;
        int cols = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<T>();

        // Check if the matrix is square
        if (!matrix.IsSquareMatrix())
        {
            return false;
        }

        for (int i = 0; i < rows; i++)
        {
            int rowCount = 0;
            int colCount = 0;
            for (int j = 0; j < cols; j++)
            {
                // Check if the element is either 0 or 1
                if (!ops.Equals(matrix[i, j], ops.Zero) && !ops.Equals(matrix[i, j], ops.One))
                {
                    return false;
                }

                // Count the number of 1's in the current row
                if (ops.Equals(matrix[i, j], ops.One))
                {
                    rowCount++;
                }

                // Count the number of 1's in the current column
                if (ops.Equals(matrix[j, i], ops.One))
                {
                    colCount++;
                }
            }

            // Each row and column must have exactly one 1
            if (rowCount != 1 || colCount != 1)
            {
                return false;
            }
        }

        return true;
    }

    public static bool IsInvolutoryMatrix<T>(this Matrix<T> matrix)
    {
        if (!matrix.IsSquareMatrix())
        {
            return false;
        }
        var product = matrix.Multiply(matrix);

        return product.IsIdentityMatrix();
    }

    public static bool IsOrthogonalProjectionMatrix<T>(this Matrix<T> matrix)
    {
        if (!matrix.IsSymmetricMatrix())
        {
            return false;
        }

        if (!matrix.IsIdempotentMatrix())
        {
            return false;
        }

        return true;
    }

    public static bool IsPositiveSemiDefiniteMatrix<T>(this Matrix<T> matrix)
    {
        if (!matrix.IsSymmetricMatrix())
        {
            return false; // Positive semi-definite matrices must be symmetric
        }

        var eigenvalues = matrix.Eigenvalues();
        var ops = MathHelper.GetNumericOperations<T>();
        // Check if all eigenvalues are non-negative
        foreach (var eigenvalue in eigenvalues)
        {
            if (ops.LessThan(eigenvalue, ops.Zero))
            {
                return false;
            }
        }

        return true;
    }

    public static Vector<T> Eigenvalues<T>(this Matrix<T> matrix)
    {
        // QR algorithm for finding eigenvalues of a symmetric matrix
        var rows = matrix.Rows;
        var a = new Matrix<T>(rows, rows);

        // Copy the matrix data manually
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < rows; j++)
            {
                a[i, j] = matrix[i, j];
            }
        }

        var ops = MathHelper.GetNumericOperations<T>();
        var eigenvalues = new Vector<T>(rows);

        for (int k = rows - 1; k > 0; k--)
        {
            while (ops.GreaterThan(ops.Abs(a[k, k - 1]), ops.FromDouble(1e-10))) // Replace ops.Epsilon with a small value
            {
                T mu = a[k, k];
                for (int i = 0; i <= k; i++)
                {
                    for (int j = 0; j <= k; j++)
                    {
                        a[i, j] = ops.Subtract(a[i, j], ops.Multiply(ops.Multiply(mu, a[k, i]), a[k, j]));
                    }
                }

                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < rows; j++)
                    {
                        a[i, j] = ops.Subtract(a[i, j], ops.Multiply(ops.Multiply(mu, a[i, k]), a[j, k]));
                    }
                }
            }

            eigenvalues[k] = a[k, k];
            for (int i = 0; i <= k; i++)
            {
                a[k, i] = a[i, k] = ops.Zero;
            }
        }
        eigenvalues[0] = a[0, 0];

        return eigenvalues;
    }

    public static T GetDeterminant<T>(this Matrix<T> matrix)
    {
        var rows = matrix.Rows;
        var ops = MathHelper.GetNumericOperations<T>();

        // Base case: for 1x1 matrix, determinant is the single element
        if (rows == 1)
        {
            return matrix[0, 0];
        }

        T det = ops.Zero;
        // Recursive case: compute the determinant using cofactor expansion
        for (int j = 0; j < rows; j++)
        {
            // Calculate the cofactor of matrix[0, j]
            var submatrix = Matrix<T>.CreateMatrix<T>(rows - 1, rows - 1);
            for (int i = 1; i < rows; i++)
            {
                for (int k = 0; k < rows; k++)
                {
                    if (k < j)
                    {
                        submatrix[i - 1, k] = matrix[i, k];
                    }
                    else if (k > j)
                    {
                        submatrix[i - 1, k - 1] = matrix[i, k];
                    }
                }
            }

            // Add the cofactor to the determinant
            T sign = ops.FromDouble(Math.Pow(-1, j));
            T cofactor = ops.Multiply(sign, matrix[0, j]);
            T subDet = submatrix.GetDeterminant();
            det = ops.Add(det, ops.Multiply(cofactor, subDet));
        }

        return det;
    }

    public static Matrix<T> InvertUpperTriangularMatrix<T>(this Matrix<T> matrix)
    {
        int n = matrix.Rows;
        var ops = MathHelper.GetNumericOperations<T>();

        // Create the inverse matrix
        var inverse = Matrix<T>.CreateMatrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i <= j)
                {
                    inverse[i, j] = ops.Divide(ops.One, matrix[i, j]);
                }
                else
                {
                    inverse[i, j] = ops.Zero;
                }
            }
        }

        return inverse;
    }

    public static Vector<T> ForwardSubstitution<T>(this Matrix<T> aMatrix, Vector<T> bVector)
    {
        int n = aMatrix.Rows;
        var ops = MathHelper.GetNumericOperations<T>();
        var x = new Vector<T>(n);

        for (int i = 0; i < n; i++)
        {
            x[i] = bVector[i];
            for (int j = 0; j < i; j++)
            {
                x[i] = ops.Subtract(x[i], ops.Multiply(aMatrix[i, j], x[j]));
            }

            x[i] = ops.Divide(x[i], aMatrix[i, i]);
        }

        return x;
    }

    public static Matrix<Complex<T>> ToComplexMatrix<T>(this Matrix<T> matrix)
    {
        var rows = matrix.Rows;
        var cols = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<T>();
        var complexMatrix = Matrix<Complex<T>>.CreateMatrix<Complex<T>>(rows, cols);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                complexMatrix[i, j] = new Complex<T>(matrix[i, j], ops.Zero);
            }
        }

        return complexMatrix;
    }

    public static Vector<Complex<T>> ToComplexVector<T>(this Vector<T> vector)
    {
        var count = vector.Length;
        var ops = MathHelper.GetNumericOperations<T>();
        var complexVector = new Vector<Complex<T>>(count);

        for (int i = 0; i < count; i++)
        {
            complexVector[i] = new Complex<T>(vector[i], ops.Zero);
        }

        return complexVector;
    }

    public static Matrix<T> ToRealMatrix<T>(this Matrix<Complex<T>> matrix)
    {
        var rows = matrix.Rows;
        var realMatrix = new Matrix<T>(rows, rows);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < rows; j++)
            {
                realMatrix[i, j] = matrix[i, j].Real;
            }
        }

        return realMatrix;
    }

    public static Matrix<T> InvertLowerTriangularMatrix<T>(this Matrix<T> matrix)
    {
        int n = matrix.Rows;
        var ops = MathHelper.GetNumericOperations<T>();
        var invL = Matrix<T>.CreateMatrix<T>(n, n);

        for (int i = 0; i < n; i++)
        {
            invL[i, i] = ops.Divide(ops.One, matrix[i, i]);
            for (int j = 0; j < i; j++)
            {
                T sum = ops.Zero;
                for (int k = j; k < i; k++)
                {
                    sum = ops.Add(sum, ops.Multiply(matrix[i, k], invL[k, j]));
                }
                invL[i, j] = ops.Negate(ops.Divide(sum, matrix[i, i]));
            }
        }

        return invL;
    }

    public static Matrix<T> InverseGaussianJordanElimination<T>(this Matrix<T> matrix)
    {
        var rows = matrix.Rows;
        var ops = MathHelper.GetNumericOperations<T>();
        var augmentedMatrix = new Matrix<T>(rows, 2 * rows);

        // Copy matrix into the left half of the augmented matrix
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < rows; j++)
            {
                augmentedMatrix[i, j] = matrix[i, j];
            }
        }

        // Add identity matrix to the right half of the augmented matrix
        for (int i = 0; i < rows; i++)
        {
            augmentedMatrix[i, i + rows] = ops.One;
        }

        // Perform Gaussian elimination with partial pivoting
        for (int i = 0; i < rows; i++)
        {
            // Find pivot row
            int maxRowIndex = i;
            T maxValue = ops.Abs(augmentedMatrix[i, i]);
            for (int k = i + 1; k < rows; k++)
            {
                T absValue = ops.Abs(augmentedMatrix[k, i]);
                if (ops.GreaterThan(absValue, maxValue))
                {
                    maxRowIndex = k;
                    maxValue = absValue;
                }
            }

            // Check for singularity
            if (ops.Equals(maxValue, ops.Zero))
            {
                throw new InvalidOperationException("Matrix is singular and cannot be inverted.");
            }

            // Swap current row with pivot row
            if (maxRowIndex != i)
            {
                for (int j = 0; j < 2 * rows; j++)
                {
                    (augmentedMatrix[maxRowIndex, j], augmentedMatrix[i, j]) = (augmentedMatrix[i, j], augmentedMatrix[maxRowIndex, j]);
                }
            }

            // Make diagonal element 1
            T pivot = augmentedMatrix[i, i];
            for (int j = 0; j < 2 * rows; j++)
            {
                augmentedMatrix[i, j] = ops.Divide(augmentedMatrix[i, j], pivot);
            }

            // Make other elements in the column zero
            for (int k = 0; k < rows; k++)
            {
                if (k != i)
                {
                    T factor = augmentedMatrix[k, i];
                    for (int j = 0; j < 2 * rows; j++)
                    {
                        augmentedMatrix[k, j] = ops.Subtract(augmentedMatrix[k, j], ops.Multiply(factor, augmentedMatrix[i, j]));
                    }
                }
            }
        }

        // Extract the right half of the augmented matrix (the inverse)
        var inverseMatrix = new Matrix<T>(rows, rows);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < rows; j++)
            {
                inverseMatrix[i, j] = augmentedMatrix[i, j + rows];
            }
        }

        return inverseMatrix;
    }

    public static Matrix<T> Extract<T>(this Matrix<T> matrix, int rows, int columns)
    {
        var result = Matrix<T>.CreateMatrix<T>(rows, columns);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                result[i, j] = matrix[i, j];
            }
        }

        return result;
    }

    public static int GetNullity<T>(this Matrix<T> matrix, T? threshold = default)
    {
        var rows = matrix.Rows;
        var columns = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<T>();
        var weightsVector = new Vector<T>(columns);
        var epsilon = ops.FromDouble(1e-10); // Small number instead of Epsilon
        var thresh = threshold != null && ops.GreaterThanOrEquals(threshold, ops.Zero)
            ? threshold
            : ops.Multiply(
                ops.FromDouble(0.5 * Math.Sqrt(rows + columns + 1)),
                ops.Multiply(weightsVector[0], epsilon)
            );
        int nullity = 0;

        for (int i = 0; i < columns; i++)
        {
            if (ops.LessThanOrEquals(weightsVector[i], thresh))
            {
                nullity++;
            }
        }

        return nullity;
    }

    public static Matrix<T> Nullspace<T>(this Matrix<T> matrix, T threshold)
    {
        int rows = matrix.Rows, columns = matrix.Columns, nullIndex = 0;
        var ops = MathHelper.GetNumericOperations<T>();
        var weightsVector = new Vector<T>(columns);
        var epsilon = ops.FromDouble(1e-10); // Small number instead of Epsilon
        var thresh = ops.GreaterThanOrEquals(threshold, ops.Zero)
            ? threshold
            : ops.Multiply(
                ops.FromDouble(0.5 * Math.Sqrt(rows + columns + 1)),
                ops.Multiply(weightsVector[0], epsilon)
            );
        var nullspaceMatrix = Matrix<T>.CreateMatrix<T>(columns, matrix.GetNullity(thresh));
        var vMatrix = matrix.Copy();

        for (int i = 0; i < columns; i++)
        {
            if (ops.LessThanOrEquals(weightsVector[i], thresh))
            {
                for (int j = 0; j < columns; j++)
                {
                    nullspaceMatrix[j, nullIndex] = vMatrix[j, i];
                }
                nullIndex++;
            }
        }

        return nullspaceMatrix;
    }

    public static Matrix<T> GetRange<T>(this Matrix<T> matrix, T threshold)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int rows = matrix.Rows, columns = matrix.Columns, rank = 0;
        var weightsVector = new Vector<T>(columns);
        var rangeMatrix = new Matrix<T>(rows, matrix.GetRank(threshold));
        var uMatrix = matrix.Copy();

        for (int i = 0; i < columns; i++)
        {
            if (ops.GreaterThan(weightsVector[i], threshold))
            {
                for (int j = 0; j < rows; j++)
                {
                    rangeMatrix[j, rank] = uMatrix[j, i];
                }
                rank++;
            }
        }

        return rangeMatrix;
    }

    public static void SetSubmatrix<T>(this Matrix<T> matrix, int startRow, int startCol, Matrix<T> submatrix)
    {
        for (int i = 0; i < submatrix.Rows; i++)
        {
            for (int j = 0; j < submatrix.Columns; j++)
            {
                matrix[startRow + i, startCol + j] = submatrix[i, j];
            }
        }
    }

    public static bool IsZeroMatrix<T>(this Matrix<T> matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<T>();

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (!ops.Equals(matrix[i, j], ops.Zero))
                {
                    return false;
                }
            }
        }

        return true;
    }

    public static T FrobeniusNorm<T>(this Matrix<T> matrix)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        T sum = ops.Zero;
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                sum = ops.Add(sum, ops.Multiply(matrix[i, j], matrix[i, j]));
            }
        }

        return ops.Sqrt(sum);
    }

    public static Matrix<T> Negate<T>(this Matrix<T> matrix)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var result = new Matrix<T>(matrix.Rows, matrix.Columns);
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                result[i, j] = ops.Negate(matrix[i, j]);
            }
        }

        return result;
    }

    public static Matrix<T> InverseNewton<T>(this Matrix<T> A, int maxIterations = 100, T? tolerance = default)
    {
        if (A.Rows != A.Columns)
            throw new ArgumentException("Matrix must be square.");

        int n = A.Rows;
        var X = A.Transpose();
        var I = Matrix<T>.CreateIdentity(n);
        var NumOps = MathHelper.GetNumericOperations<T>();
        tolerance ??= NumOps.FromDouble(1e-10);

        for (int k = 0; k < maxIterations; k++)
        {
            var R = I.Subtract(A.Multiply(X));
            if (NumOps.LessThan(R.FrobeniusNorm(), tolerance))
            {
                return X;
            }

            X = X.Add(X.Multiply(R));
        }

        throw new InvalidOperationException("Newton's method did not converge.");
    }

    public static Matrix<T> InverseStrassen<T>(this Matrix<T> A)
    {
        if (A.Rows != A.Columns)
            throw new ArgumentException("Matrix must be square.");

        int n = A.Rows;
        var NumOps = MathHelper.GetNumericOperations<T>();

        if (n == 1)
        {
            return new Matrix<T>(1, 1) { [0, 0] = NumOps.Divide(NumOps.One, A[0, 0]) };
        }

        if (n % 2 != 0)
        {
            throw new ArgumentException("Matrix size must be a power of 2 for Strassen's algorithm.");
        }

        int m = n / 2;

        var A11 = A.Submatrix(0, 0, m, m);
        var A12 = A.Submatrix(0, m, m, m);
        var A21 = A.Submatrix(m, 0, m, m);
        var A22 = A.Submatrix(m, m, m, m);

        var A11_inv = InverseStrassen(A11);
        var S = A22.Subtract(A21.Multiply(A11_inv).Multiply(A12));
        var S_inv = InverseStrassen(S);
        var P = A11_inv.Add(A11_inv.Multiply(A12).Multiply(S_inv).Multiply(A21).Multiply(A11_inv));
        var Q = A11_inv.Multiply(A12).Multiply(S_inv).Negate();
        var R = S_inv.Multiply(A21).Multiply(A11_inv).Negate();

        var result = new Matrix<T>(n, n);
        result.SetSubmatrix(0, 0, P);
        result.SetSubmatrix(0, m, Q);
        result.SetSubmatrix(m, 0, R);
        result.SetSubmatrix(m, m, S_inv);

        return result;
    }

    public static Matrix<T> Inverse<T>(this Matrix<T> matrix, InverseType inverseType = InverseType.GaussianJordan, int maxIterations = 100, T? tolerance = default)
    {
        return inverseType switch
        {
            InverseType.Strassen => InverseStrassen(matrix),
            InverseType.Newton => InverseNewton(matrix, maxIterations, tolerance),
            InverseType.GaussianJordan => InverseGaussianJordanElimination(matrix),
            _ => throw new ArgumentException("Invalid inverse type", nameof(inverseType)),
        };
    }

    public static Matrix<T> Transpose<T>(this Matrix<T> matrix)
    {
        if (matrix == null)
        {
            throw new ArgumentNullException(nameof(matrix), $"{nameof(matrix)} can't be null");
        }

        var rows = matrix.Rows;
        var columns = matrix.Columns;
        if (rows == 0)
        {
            throw new ArgumentException($"{nameof(matrix)} has to contain at least one row of values", nameof(matrix));
        }

        if (columns == 0)
        {
            throw new ArgumentException($"{nameof(matrix)} has to contain at least one column of values", nameof(matrix));
        }

        var newMatrix = Matrix<T>.CreateMatrix<T>(columns, rows);
        for (int i = 0; i < columns; i++)
        {
            for (int j = 0; j < rows; j++)
            {
                newMatrix[i, j] = matrix[j, i];
            }
        }

        return newMatrix;
    }

    public static int GetRank<T>(this Matrix<T> matrix, T threshold)
    {
        var rows = matrix.Rows;
        var columns = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<T>();
        var weightsVector = new Vector<T>(columns);
        var epsilon = ops.FromDouble(1e-10); // Small number instead of Epsilon
        var thresh = ops.GreaterThanOrEquals(threshold, ops.Zero) 
            ? threshold 
            : ops.Multiply(ops.FromDouble(0.5 * Math.Sqrt(rows + columns + 1)), ops.Multiply(weightsVector[0], epsilon));
        int rank = 0;

        for (int i = 0; i < columns; i++)
        {
            if (ops.GreaterThan(weightsVector[i], thresh))
            {
                rank++;
            }
        }

        return rank;
    }

    public static void SwapRows<T>(this Matrix<T> matrix, int row1Index, int row2Index)
    {
        var rows = matrix.Rows;
        for (int i = 0; i < rows; i++)
        {
            (matrix[row2Index, i], matrix[row1Index, i]) = (matrix[row1Index, i], matrix[row2Index, i]);
        }
    }

    public static Matrix<T> InvertDiagonalMatrix<T>(this Matrix<T> matrix)
    {
        var rows = matrix.Rows;
        var invertedMatrix = Matrix<T>.CreateMatrix<T>(rows, rows);
        var ops = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < rows; i++)
        {
            invertedMatrix[i, i] = ops.Divide(ops.One, matrix[i, i]);
        }

        return invertedMatrix;
    }

    public static Matrix<Complex<T>> InvertUnitaryMatrix<T>(this Matrix<Complex<T>> matrix)
    {
        return matrix.Transpose();
    }

    public static bool IsUnitaryMatrix<T>(this Matrix<Complex<T>> matrix, IMatrixDecomposition<Complex<T>> matrixDecomposition)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;

        if (rows != cols)
        {
            return false; // A unitary matrix must be square
        }

        var conjugateTranspose = matrix.ConjugateTranspose();
        var inverse = matrixDecomposition.Invert();

        // Check if conjugate transpose is equal to inverse
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (conjugateTranspose[i, j] != inverse[i, j])
                {
                    return false;
                }
            }
        }

        return true;
    }

    public static T Determinant<T>(this Matrix<T> matrix)
    {
        if (matrix == null)
        {
            throw new ArgumentNullException(nameof(matrix), $"{nameof(matrix)} can't be null");
        }

        if (matrix.Rows == 0)
        {
            throw new ArgumentException($"{nameof(matrix)} has to contain at least one row of values", nameof(matrix));
        }

        var columns = matrix.Columns;
        var rows = matrix.Rows;
        if (rows <= 0)
        {
            throw new ArgumentException($"{nameof(rows)} needs to be an integer larger than 0");
        }

        if (columns <= 0)
        {
            throw new ArgumentException($"{nameof(columns)} needs to be an integer larger than 0");
        }

        if (rows != columns)
        {
            throw new ArgumentException($"You need to have a square matrix to calculate the determinant value so the length of {nameof(rows)} {nameof(columns)} must be equal");
        }

        var ops = MathHelper.GetNumericOperations<T>();

        if (rows == 2)
        {
            return ops.Subtract(
                ops.Multiply(matrix[0, 0], matrix[1, 1]),
                ops.Multiply(matrix[0, 1], matrix[1, 0])
            );
        }
        else
        {
            T determinant = ops.Zero;
            for (int i = 0; i < rows; i++)
            {
                var tempMatrix = new Matrix<T>(rows - 1, rows - 1);
                for (int j = 0; j < rows - 1; j++)
                {
                    for (int k = 0; k < rows - 1; k++)
                    {
                        tempMatrix[j, k] = matrix[j < i ? j : j + 1, k];
                    }
                }

                T subDeterminant = tempMatrix.Determinant();
                T sign = ops.FromDouble(i % 2 == 0 ? 1 : -1);
                T product = ops.Multiply(ops.Multiply(sign, matrix[0, i]), subDeterminant);
                determinant = ops.Add(determinant, product);
            }

            return determinant;
        }
    }

    public static T[] GetRow<T>(this Matrix<T> matrix, int rowIndex)
    {
        return Enumerable.Range(0, matrix.Columns)
                .Select(x => matrix[rowIndex, x])
                .ToArray();
    }

    public static Vector<T> GetSubColumn<T>(this Matrix<T> matrix, int columnIndex, int startRow, int length)
    {
        if (columnIndex < 0 || columnIndex >= matrix.Columns)
            throw new ArgumentOutOfRangeException(nameof(columnIndex));
        if (startRow < 0 || startRow >= matrix.Rows)
            throw new ArgumentOutOfRangeException(nameof(startRow));
        if (length < 0 || startRow + length > matrix.Rows)
            throw new ArgumentOutOfRangeException(nameof(length));

        var result = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            result[i] = matrix[startRow + i, columnIndex];
        }

        return result;
    }

    public static T LogDeterminant<T>(this Matrix<T> matrix)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
            
        if (matrix.Rows != matrix.Columns)
        {
            throw new ArgumentException("Matrix must be square to calculate determinant.");
        }

        // Use LU decomposition to calculate log determinant
        var lu = new LuDecomposition<T>(matrix);
        var U = lu.U;

        T logDet = numOps.Zero;
        for (int i = 0; i < U.Rows; i++)
        {
            logDet = numOps.Add(logDet, numOps.Log(numOps.Abs(U[i, i])));
        }

        return logDet;
    }

    public static Matrix<T> PointwiseMultiply<T>(this Matrix<T> matrix, Matrix<T> other)
    {
        if (matrix.Rows != other.Rows || matrix.Columns != other.Columns)
        {
            throw new ArgumentException("Matrices must have the same dimensions for pointwise multiplication.");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var result = new Matrix<T>(matrix.Rows, matrix.Columns);

        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                result[i, j] = numOps.Multiply(matrix[i, j], other[i, j]);
            }
        }

        return result;
    }

    public static Matrix<T> PointwiseMultiply<T>(this Matrix<T> matrix, Vector<T> vector)
    {
        if (matrix.Rows != vector.Length)
        {
            throw new ArgumentException("The number of rows in the matrix must match the length of the vector.");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        Matrix<T> result = new(matrix.Rows, matrix.Columns);

        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                result[i, j] = numOps.Multiply(matrix[i, j], vector[i]);
            }
        }

        return result;
    }

    public static Matrix<T> AddColumn<T>(this Matrix<T> matrix, Vector<T> column)
    {
        if (matrix.Rows != column.Length)
        {
            throw new ArgumentException("Column length must match matrix row count.");
        }

        Matrix<T> newMatrix = new Matrix<T>(matrix.Rows, matrix.Columns + 1);

        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                newMatrix[i, j] = matrix[i, j];
            }
            newMatrix[i, matrix.Columns] = column[i];
        }

        return newMatrix;
    }

    public static Matrix<T> Submatrix<T>(this Matrix<T> matrix, int startRow, int startCol, int numRows, int numCols)
    {
        if (startRow < 0 || startCol < 0 || startRow + numRows > matrix.Rows || startCol + numCols > matrix.Columns)
        {
            throw new ArgumentOutOfRangeException("Invalid submatrix dimensions");
        }

        Matrix<T> submatrix = new Matrix<T>(numRows, numCols);

        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numCols; j++)
            {
                submatrix[i, j] = matrix[startRow + i, startCol + j];
            }
        }

        return submatrix;
    }

    public static Matrix<T> GetColumns<T>(this Matrix<T> matrix, IEnumerable<int> columnIndices)
    {
        return new Matrix<T>(GetColumnVectors(matrix, columnIndices));
    }

    public static List<Vector<T>> GetColumnVectors<T>(this Matrix<T> matrix, IEnumerable<int> columnIndices)
    {
        var selectedColumns = new List<Vector<T>>();
        foreach (int index in columnIndices)
        {
            if (index < 0 || index >= matrix.Columns)
            {
                throw new ArgumentOutOfRangeException(nameof(columnIndices), $"Column index {index} is out of range.");
            }
            selectedColumns.Add(matrix.GetColumn(index));
        }

        return selectedColumns;
    }

    public static T Max<T>(this Matrix<T> matrix, Func<T, T> selector)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        T max = selector(matrix[0, 0]);

        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                T value = selector(matrix[i, j]);
                if (numOps.GreaterThan(value, max))
                {
                    max = value;
                }
            }
        }

        return max;
    }

    public static Matrix<T> GetRowRange<T>(this Matrix<T> matrix, int startRow, int rowCount)
    {
        Matrix<T> result = new(rowCount, matrix.Columns);
        for (int i = 0; i < rowCount; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                result[i, j] = matrix[startRow + i, j];
            }
        }

        return result;
    }

    public static Vector<T> RowWiseArgmax<T>(this Matrix<T> matrix)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        Vector<T> result = new(matrix.Rows);

        for (int i = 0; i < matrix.Rows; i++)
        {
            T max = matrix[i, 0];
            int maxIndex = 0;
            for (int j = 1; j < matrix.Columns; j++)
            {
                if (numOps.GreaterThan(matrix[i, j], max))
                {
                    max = matrix[i, j];
                    maxIndex = j;
                }
            }
            result[i] = numOps.FromDouble(maxIndex);
        }

        return result;
    }

    public static Matrix<T> KroneckerProduct<T>(this Matrix<T> a, Matrix<T> b)
    {
        int m = a.Rows;
        int n = a.Columns;
        int p = b.Rows;
        int q = b.Columns;

        var numOps = MathHelper.GetNumericOperations<T>();
        Matrix<T> result = new(m * p, n * q);

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                for (int k = 0; k < p; k++)
                {
                    for (int l = 0; l < q; l++)
                    {
                        result[i * p + k, j * q + l] = numOps.Multiply(a[i, j], b[k, l]);
                    }
                }
            }
        }

        return result;
    }

    public static Vector<T> Flatten<T>(this Matrix<T> matrix)
    {
        var vector = new Vector<T>(matrix.Rows * matrix.Columns);
        int index = 0;

        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                vector[index++] = matrix[i, j];
            }
        }

        return vector;
    }

    public static Matrix<T> Reshape<T>(this Matrix<T> matrix, int newRows, int newColumns)
    {
        if (matrix.Rows * matrix.Columns != newRows * newColumns)
        {
            throw new ArgumentException("The total number of elements must remain the same after reshaping.");
        }

        var reshaped = new Matrix<T>(newRows, newColumns);
        int index = 0;
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                int newRow = index / newColumns;
                int newCol = index % newColumns;
                reshaped[newRow, newCol] = matrix[i, j];
                index++;
            }
        }

        return reshaped;
    }
}