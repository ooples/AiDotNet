global using AiDotNet.NumericOperations;

namespace AiDotNet.Helpers;

public static class MatrixHelper
{
    public static T CalculateDeterminantRecursive<T>(Matrix<T> matrix)
    {
        var rows = matrix.Rows;
        var ops = MathHelper.GetNumericOperations<T>();

        if (rows != matrix.Columns)
        {
            throw new ArgumentException("Matrix must be square.");
        }

        if (rows == 1)
        {
            return matrix[0, 0];
        }

        T determinant = ops.Zero;

        for (var i = 0; i < rows; i++)
        {
            var subMatrix = CreateSubMatrix(matrix, 0, i);
            T subDeterminant = CalculateDeterminantRecursive(subMatrix);
            T sign = ops.FromDouble(i % 2 == 0 ? 1 : -1);
            T product = ops.Multiply(ops.Multiply(sign, matrix[0, i]), subDeterminant);
            determinant = ops.Add(determinant, product);
        }

        return determinant;
    }

    private static Matrix<T> CreateSubMatrix<T>(Matrix<T> matrix, int excludeRowIndex, int excludeColumnIndex)
    {
        var rows = matrix.Rows;
        var ops = MathHelper.GetNumericOperations<T>();
        var subMatrix = new Matrix<T>(rows - 1, rows - 1, ops);

        var r = 0;
        for (var i = 0; i < rows; i++)
        {
            if (i == excludeRowIndex)
            {
                continue;
            }

            var c = 0;
            for (var j = 0; j < rows; j++)
            {
                if (j == excludeColumnIndex)
                {
                    continue;
                }

                subMatrix[r, c] = matrix[i, j];
                c++;
            }

            r++;
        }

        return subMatrix;
    }

    public static void ReplaceColumn<T>(T[,] destination, T[,] source, int destColumn, int srcColumn)
    {
        //exceptions
        //Ensure the size of source matrix column is equal to the size of destination matrix column
        //ensure destColumn is in scope of destination, ie destColumn < sizeOf Destinations rows
        //ensure srcColumn is in scope of source, ie srcColumn < sizeOf source rows

        //rows = 0 ; col = 1
        int size = source.GetLength(0);

        for (var i = 0; i < size; i++)
        {
            destination[i, destColumn] = source[i, srcColumn];
        }
    }

    public static Vector<T> GetColumn<T>(this Matrix<T> matrix, int columnIndex)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var column = new Vector<T>(matrix.Rows, ops);
        for (int i = 0; i < matrix.Rows; i++)
        {
            column[i] = matrix[i, columnIndex];
        }

        return column;
    }

    public static T[] GetRow<T>(this Matrix<T> matrix, int rowIndex)
    {
        return Enumerable.Range(0, matrix.Columns)
                .Select(x => matrix[rowIndex, x])
                .ToArray();
    }

    public static Matrix<T> Reshape<T>(this Vector<T> vector, int rows, int columns)
    {
        if (vector == null)
        {
            throw new ArgumentNullException(nameof(vector), $"{nameof(vector)} can't be null");
        }

        if (rows <= 0)
        {
            throw new ArgumentException($"{nameof(rows)} needs to be an integer larger than 0", nameof(rows));
        }

        if (columns <= 0)
        {
            throw new ArgumentException($"{nameof(columns)} needs to be an integer larger than 0", nameof(columns));
        }

        var length = vector.Length;
        if (rows * columns != length)
        {
            throw new ArgumentException($"{nameof(rows)} and {nameof(columns)} multiplied together needs to be equal the array length");
        }

        var matrix = Matrix<T>.CreateMatrix<T>(rows, columns);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                matrix[i, j] = vector[i * columns + j];
            }
        }

        return matrix;
    }

    public static Matrix<T> ToMatrix<T>(this Vector<T> vector)
    {
        int n = vector.Length;
        var matrix = Matrix<T>.CreateMatrix<T>(n, 1);
        for (int i = 0; i < n; ++i)
        {
            matrix[i, 0] = vector[i];
        }
            
        return matrix;
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

    public static Matrix<T> ReduceToHessenbergFormat<T>(Matrix<T> matrix)
    {
        var rows = matrix.Rows;
        var ops = MathHelper.GetNumericOperations<T>();
        var result = Matrix<T>.CreateMatrix<T>(rows, rows);

        for (int k = 0; k < rows - 2; k++)
        {
            var xVector = new Vector<T>(rows - k - 1, ops);
            for (int i = 0; i < rows - k - 1; i++)
            {
                xVector[i] = matrix[k + 1 + i, k];
            }

            var hVector = CreateHouseholderVector(xVector);
            matrix = ApplyHouseholderTransformation(matrix, hVector, k);
        }

        return matrix;
    }

    public static Vector<T> BackwardSubstitution<T>(this Matrix<T> aMatrix, Vector<T> bVector)
    {
        int n = aMatrix.Rows;
        var ops = MathHelper.GetNumericOperations<T>();
        var xVector = new Vector<T>(n, ops);
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

    public static List<T> GetEnumValues<T>(string? ignoreName = null) where T : struct
    {
        var members = typeof(T).GetMembers();
        var result = new List<T>();

        foreach (var member in members)
        {
            //use the member name to get an instance of enumerated type.
            if (Enum.TryParse(member.Name, out T enumType) && (!string.IsNullOrEmpty(ignoreName) && member.Name != ignoreName))
            {
                result.Add(enumType);
            }
        }

        return result;
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
        foreach (var matrixType in GetEnumValues<MatrixType>())
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

    public static Matrix<Complex<T>> ConjugateTranspose<T>(this Matrix<Complex<T>> matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;
        var result = Matrix.CreateComplexMatrix<T>(cols, rows);

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

    public static bool IsPositiveDefiniteMatrix<T>(this Matrix<T> matrix)
    {
        if (!matrix.IsSymmetricMatrix())
        {
            return false; // Positive definite matrices must be symmetric
        }

        var eigenvalues = matrix.Eigenvalues();
        var ops = MathHelper.GetNumericOperations<T>();
        // Check if all eigenvalues are positive
        foreach (var eigenvalue in eigenvalues)
        {
            if (ops.LessThanOrEquals(eigenvalue, ops.Zero))
            {
                return false;
            }
        }

        return true;
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
        var eigenvalues = new Vector<T>(rows, ops);

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

    public static Vector<T> ExtractDiagonal<T>(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        Vector<T> diagonal = new(n);
        for (int i = 0; i < n; i++)
        {
            diagonal[i] = matrix[i, i];
        }

        return diagonal;
    }

    public static Matrix<T> OuterProduct<T>(Vector<T> v1, Vector<T> v2)
    {
        int n = v1.Length;
        var ops = MathHelper.GetNumericOperations<T>();
        Matrix<T> result = new(n, n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                result[i, j] = ops.Multiply(v1[i], v2[j]);
            }
        }

        return result;
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

    public static T Hypotenuse<T>(T x, T y)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        T xabs = ops.Abs(x), yabs = ops.Abs(y), min, max;

        if (ops.LessThan(xabs, yabs))
        {
            min = xabs; max = yabs;
        }
        else
        {
            min = yabs; max = xabs;
        }

        if (ops.Equals(min, ops.Zero))
        {
            return max;
        }

        T u = ops.Divide(min, max);

        return ops.Multiply(max, ops.Sqrt(ops.Add(ops.One, ops.Multiply(u, u))));
    }

    public static T Hypotenuse<T>(params T[] values)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        T sum = ops.Zero;
        foreach (var value in values)
        {
            sum = ops.Add(sum, ops.Multiply(value, value));
        }

        return ops.Sqrt(sum);
    }

    public static Vector<T> ForwardSubstitution<T>(this Matrix<T> aMatrix, Vector<T> bVector)
    {
        int n = aMatrix.Rows;
        var ops = MathHelper.GetNumericOperations<T>();
        var x = new Vector<T>(n, ops);

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

    public static bool IsUpperHessenberg<T>(Matrix<T> matrix, T tolerance)
    {
        var NumOps = MathHelper.GetNumericOperations<T>();
        for (int i = 2; i < matrix.Rows; i++)
        {
            for (int j = 0; j < i - 1; j++)
            {
                if (NumOps.GreaterThan(NumOps.Abs(matrix[i, j]), tolerance))
                {
                    return false;
                }
            }
        }

        return true;
    }

    public static Matrix<T> OrthogonalizeColumns<T>(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;
        Matrix<T> Q = new(m, n);
        var NumOps = MathHelper.GetNumericOperations<T>();

        for (int j = 0; j < n; j++)
        {
            Vector<T> v = matrix.GetColumn(j);
            for (int i = 0; i < j; i++)
            {
                Vector<T> qi = Q.GetColumn(i);
                T r = NumOps.Divide(v.DotProduct(qi), qi.DotProduct(qi));
                v = v.Subtract(qi.Multiply(r));
            }
            T norm = v.Norm();
            if (!NumOps.Equals(norm, NumOps.Zero))
            {
                v = v.Divide(norm);
            }
            Q.SetColumn(j, v);
        }

        return Q;
    }

    public static (T c, T s) ComputeGivensRotation<T>(T a, T b)
    {
        var NumOps = MathHelper.GetNumericOperations<T>();
        if (NumOps.Equals(b, NumOps.Zero))
        {
            return (NumOps.One, NumOps.Zero);
        }

        var t = NumOps.Divide(a, b);
        var u = NumOps.Sqrt(NumOps.Add(NumOps.One, NumOps.Multiply(t, t)));
        var c = NumOps.Divide(NumOps.One, u);
        var s = NumOps.Multiply(c, t);

        return (c, s);
    }

    public static void ApplyGivensRotation<T>(Matrix<T> H, T c, T s, int i, int j, int kStart, int kEnd)
    {
        var NumOps = MathHelper.GetNumericOperations<T>();
        for (int k = kStart; k < kEnd; k++)
        {
            var temp1 = H[i, k];
            var temp2 = H[j, k];
            H[i, k] = NumOps.Add(NumOps.Multiply(c, temp1), NumOps.Multiply(s, temp2));
            H[j, k] = NumOps.Subtract(NumOps.Multiply(NumOps.Negate(s), temp1), NumOps.Multiply(c, temp2));
        }
    }

    public static Vector<T> ToRealVector<T>(this Vector<Complex<T>> vector)
    {
        var count = vector.Length;
        var ops = MathHelper.GetNumericOperations<T>();
        var realVector = new Vector<T>(count, ops);

        for (int i = 0; i < count; i++)
        {
            realVector[i] = vector[i].Real;
        }

        return realVector;
    }

    public static Matrix<T> ApplyHouseholderTransformation<T>(Matrix<T> matrix, Vector<T> vector, int k)
    {
        var rows = matrix.Rows;
        var ops = MathHelper.GetNumericOperations<T>();

        for (int i = k + 1; i < rows; i++)
        {
            T sum = ops.Zero;
            for (int j = k + 1; j < rows; j++)
            {
                sum = ops.Add(sum, ops.Multiply(vector[j - k - 1], matrix[j, i]));
            }
            for (int j = k + 1; j < rows; j++)
            {
                matrix[j, i] = ops.Subtract(matrix[j, i], ops.Multiply(ops.Multiply(ops.FromDouble(2), vector[j - k - 1]), sum));
            }
        }

        for (int i = 0; i < rows; i++)
        {
            T sum = ops.Zero;
            for (int j = k + 1; j < rows; j++)
            {
                sum = ops.Add(sum, ops.Multiply(vector[j - k - 1], matrix[i, j]));
            }
            for (int j = k + 1; j < rows; j++)
            {
                matrix[i, j] = ops.Subtract(matrix[i, j], ops.Multiply(ops.Multiply(ops.FromDouble(2), vector[j - k - 1]), sum));
            }
        }

        return matrix;
    }


    public static Vector<T> CreateHouseholderVector<T>(Vector<T> xVector)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(xVector.Length, ops);
        T norm = ops.Zero;

        for (int i = 0; i < xVector.Length; i++)
        {
            norm = ops.Add(norm, ops.Multiply(xVector[i], xVector[i]));
        }
        norm = ops.Sqrt(norm);

        result[0] = ops.Add(xVector[0], ops.Multiply(ops.SignOrZero(xVector[0]), norm));
        for (int i = 1; i < xVector.Length; i++)
        {
            result[i] = xVector[i];
        }

        T vNorm = ops.Zero;
        for (int i = 0; i < result.Length; i++)
        {
            vNorm = ops.Add(vNorm, ops.Multiply(result[i], result[i]));
        }
        vNorm = ops.Sqrt(vNorm);

        for (int i = 0; i < result.Length; i++)
        {
            result[i] = ops.Divide(result[i], vNorm);
        }

        return result;
    }

    public static (T, Vector<T>) PowerIteration<T>(Matrix<T> aMatrix, int maxIterations, T tolerance)
    {
        var rows = aMatrix.Rows;
        var ops = MathHelper.GetNumericOperations<T>();
        var bVector = new Vector<T>(rows, ops);
        var b2Vector = new Vector<T>(rows, ops);
        T eigenvalue = ops.Zero;

        // Initial guess for the eigenvector
        for (int i = 0; i < rows; i++)
        {
            bVector[i] = ops.One;
        }

        for (int iter = 0; iter < maxIterations; iter++)
        {
            // Multiply A by the vector b
            for (int i = 0; i < rows; i++)
            {
                b2Vector[i] = ops.Zero;
                for (int j = 0; j < rows; j++)
                {
                    b2Vector[i] = ops.Add(b2Vector[i], ops.Multiply(aMatrix[i, j], bVector[j]));
                }
            }

            // Normalize the vector
            T norm = ops.Zero;
            for (int i = 0; i < rows; i++)
            {
                norm = ops.Add(norm, ops.Multiply(b2Vector[i], b2Vector[i]));
            }
            norm = ops.Sqrt(norm);
            for (int i = 0; i < rows; i++)
            {
                b2Vector[i] = ops.Divide(b2Vector[i], norm);
            }

            // Estimate the eigenvalue
            T newEigenvalue = ops.Zero;
            for (int i = 0; i < rows; i++)
            {
                newEigenvalue = ops.Add(newEigenvalue, ops.Multiply(b2Vector[i], b2Vector[i]));
            }

            // Check for convergence
            if (ops.LessThan(ops.Abs(ops.Subtract(newEigenvalue, eigenvalue)), tolerance))
            {
                break;
            }
            eigenvalue = newEigenvalue;
            for (int i = 0; i < rows; i++)
            {
                bVector[i] = b2Vector[i];
            }
        }

        return (eigenvalue, b2Vector);
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

    public static Matrix<T> InvertUsingDecomposition<T>(IMatrixDecomposition<T> decomposition)
    {
        int n = decomposition.A.Rows;
        var inverse = new Matrix<T>(n, n);

        for (int j = 0; j < n; j++)
        {
            var ej = Vector<T>.CreateStandardBasis(n, j);
            var xj = decomposition.Solve(ej);

            for (int i = 0; i < n; i++)
            {
                inverse[i, j] = xj[i];
            }
        }

        return inverse;
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
        var weightsVector = new Vector<T>(columns, ops);
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
        var weightsVector = new Vector<T>(columns, ops);
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
        int rows = matrix.Rows, columns = matrix.Columns, rank = 0;
        var ops = MathHelper.GetNumericOperations<T>();
        var weightsVector = new Vector<T>(columns, ops);
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

    public static int GetRank<T>(this Matrix<T> matrix, T threshold)
    {
        var rows = matrix.Rows;
        var columns = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<T>();
        var weightsVector = new Vector<T>(columns, ops);
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

    public static Vector<T> Extract<T>(this Vector<T> vector, int length)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(length, ops);
        for (int i = 0; i < length; i++)
        {
            result[i] = vector[i];
        }

        return result;
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

    public static void TridiagonalSolve<T>(Vector<T> vector1, Vector<T> vector2, Vector<T> vector3,
    Vector<T> solutionVector, Vector<T> actualVector)
    {
        var size = vector1.Length;
        T bet;
        var ops = MathHelper.GetNumericOperations<T>();
        var gamVector = new Vector<T>(size, ops);

        if (ops.Equals(vector2[0], ops.Zero))
        {
            throw new InvalidOperationException("Not a tridiagonal matrix!");
        }

        bet = vector2[0];
        solutionVector[0] = ops.Divide(actualVector[0], bet);
        for (int i = 1; i < size; i++)
        {
            gamVector[i] = ops.Divide(vector3[i - 1], bet);
            bet = ops.Subtract(vector2[i], ops.Multiply(vector1[i], gamVector[i]));

            if (ops.Equals(bet, ops.Zero))
            {
                throw new InvalidOperationException("Not a tridiagonal matrix!");
            }

            solutionVector[i] = ops.Divide(
                ops.Subtract(actualVector[i], ops.Multiply(vector1[i], solutionVector[i - 1])),
                bet
            );
        }

        for (int i = size - 2; i >= 0; i--)
        {
            solutionVector[i] = ops.Subtract(
                solutionVector[i],
                ops.Multiply(gamVector[i + 1], solutionVector[i + 1])
            );
        }
    }

    public static void BandDiagonalMultiply<T>(int leftSide, int rightSide, Matrix<T> matrix, Vector<T> solutionVector, Vector<T> actualVector)
    {
        var size = matrix.Rows;
        var ops = MathHelper.GetNumericOperations<T>();

        for (int i = 0; i < size; i++)
        {
            var k = i - leftSide;
            var temp = Math.Min(leftSide + rightSide + 1, size - k);
            solutionVector[i] = ops.Zero;
            for (int j = Math.Max(0, -k); j < temp; j++)
            {
                solutionVector[i] = ops.Add(solutionVector[i], ops.Multiply(matrix[i, j], actualVector[j + k]));
            }
        }
    }

    public static Matrix<T> InverseNewton<T>(Matrix<T> A, int maxIterations = 100, T? tolerance = default)
    {
        if (A.Rows != A.Columns)
            throw new ArgumentException("Matrix must be square.");

        var ops = MathHelper.GetNumericOperations<T>();
        int n = A.Rows;
        var X = A.Transpose();
        var I = Matrix<T>.CreateIdentity(n, ops);
        tolerance ??= ops.FromDouble(1e-10);

        for (int k = 0; k < maxIterations; k++)
        {
            var R = I.Subtract(A.Multiply(X));
            if (ops.LessThan(R.FrobeniusNorm(), tolerance))
            {
                return X;
            }

            X = X.Add(X.Multiply(R));
        }

        throw new InvalidOperationException("Newton's method did not converge.");
    }

    public static Matrix<T> InverseStrassen<T>(Matrix<T> A)
    {
        if (A.Rows != A.Columns)
            throw new ArgumentException("Matrix must be square.");

        int n = A.Rows;
        var ops = MathHelper.GetNumericOperations<T>();

        if (n == 1)
        {
            return new Matrix<T>(1, 1, ops) { [0, 0] = ops.Divide(ops.One, A[0, 0]) };
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

        var result = new Matrix<T>(n, n, ops);
        result.SetSubmatrix(0, 0, P);
        result.SetSubmatrix(0, m, Q);
        result.SetSubmatrix(m, 0, R);
        result.SetSubmatrix(m, m, S_inv);

        return result;
    }

    private static Matrix<T> Submatrix<T>(this Matrix<T> matrix, int startRow, int startCol, int rows, int cols)
    {
        var submatrix = new Matrix<T>(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                submatrix[i, j] = matrix[startRow + i, startCol + j];
            }
        }

        return submatrix;
    }

    private static void SetSubmatrix<T>(this Matrix<T> matrix, int startRow, int startCol, Matrix<T> submatrix)
    {
        for (int i = 0; i < submatrix.Rows; i++)
        {
            for (int j = 0; j < submatrix.Columns; j++)
            {
                matrix[startRow + i, startCol + j] = submatrix[i, j];
            }
        }
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
        var result = new Matrix<T>(matrix.Rows, matrix.Columns, ops);
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                result[i, j] = ops.Negate(matrix[i, j]);
            }
        }

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

    public static Matrix<T> CalculateHatMatrix<T>(Matrix<T> features)
    {
        var _numOps = MathHelper.GetNumericOperations<T>();
        var transposeFeatures = features.Transpose();
        var inverseMatrix = transposeFeatures.Multiply(features).Inverse();

        return features.Multiply(inverseMatrix.Multiply(transposeFeatures));
    }
}