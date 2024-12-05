global using Complex = AiDotNet.LinearAlgebra.Complex;
global using AiDotNet.NumericOperations;

namespace AiDotNet.Helpers;

public static class MatrixHelper
{
    public static double CalculateDeterminantRecursive(Matrix<double> matrix)
    {
        var rows = matrix.Rows;

        if (rows != matrix.Columns)
        {
            throw new ArgumentException("Matrix must be square.");
        }

        if (rows == 1)
        {
            return matrix[0, 0];
        }

        double determinant = 0;

        for (var i = 0; i < rows; i++)
        {
            var subMatrix = CreateSubMatrix(matrix, 0, i);
            determinant += Math.Pow(-1, i) * matrix[0, i] * CalculateDeterminantRecursive(subMatrix);
        }

        return determinant;
    }

    public static Matrix<T> Add<T>(this Matrix<T> left, Matrix<T> right) where T : struct
    {
        if (left.Rows != right.Rows || left.Columns != right.Columns)
            throw new ArgumentException("Matrix dimensions are incompatible for addition");

        var ops = MathHelper.GetNumericOperations<T>();
        var result = new Matrix<T>(left.Rows, left.Columns, ops);

        for (int i = 0; i < left.Rows; i++)
        {
            for (int j = 0; j < left.Columns; j++)
            {
                result[i, j] = ops.Add(left[i, j], right[i, j]);
            }
        }

        return result;
    }

    public static Matrix<T> Multiply<T>(this Matrix<T> left, Matrix<T> right)
    {
        if (left.Columns != right.Rows)
            throw new ArgumentException("Matrix dimensions are incompatible for multiplication");

        var ops = MathHelper.GetNumericOperations<T>();
        var result = new Matrix<T>(left.Rows, right.Columns, ops);

        for (int i = 0; i < left.Rows; i++)
        {
            for (int j = 0; j < right.Columns; j++)
            {
                T sum = ops.Zero;
                for (int k = 0; k < left.Columns; k++)
                {
                    sum = ops.Add(sum, ops.Multiply(left[i, k], right[k, j]));
                }
                result[i, j] = sum;
            }
        }

        return result;
    }

    public static Vector<T> Multiply<T>(this Matrix<T> matrix, Vector<T> vector)
    {
        if (matrix.Columns != vector.Length)
            throw new ArgumentException("Matrix columns must match vector length");

        var ops = MathHelper.GetNumericOperations<T>();
        var result = new Vector<T>(matrix.Rows, ops);

        for (int i = 0; i < matrix.Rows; i++)
        {
            T sum = ops.Zero;
            for (int j = 0; j < matrix.Columns; j++)
            {
                sum = ops.Add(sum, ops.Multiply(matrix[i, j], vector[j]));
            }

            result[i] = sum;
        }

        return result;
    }

    public static Matrix<T> Multiply<T>(this Matrix<T> matrix, T scalar)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var result = new Matrix<T>(matrix.Rows, matrix.Columns, ops);
        
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                result[i, j] = ops.Multiply(matrix[i, j], scalar);
            }
        }
        
        return result;
    }

    private static Matrix<double> CreateSubMatrix(Matrix<double> matrix, int excludeRowIndex, int excludeColumnIndex)
    {
        var rows = matrix.Rows;
        var subMatrix = Matrix.CreateDoubleMatrix(rows - 1, rows - 1);

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
    public static void ReplaceColumn(double[,] destination, double[,] source, int destColumn, int srcColumn)
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


    public static Vector<double> HouseholderVector(Vector<double> xVector)
    {
        var ops = MathHelper.GetNumericOperations<double>();
        var result = new Vector<double>(xVector.Length, ops);
        double norm = 0;
        for (int i = 0; i < xVector.Length; i++)
        {
            norm += xVector[i] * xVector[i];
        }
        norm = Math.Sqrt(norm);

        result[0] = xVector[0] + Math.Sign(xVector[0]) * norm;
        for (int i = 1; i < xVector.Length; i++)
        {
            result[i] = xVector[i];
        }

        double vNorm = 0;
        for (int i = 0; i < result.Length; i++)
        {
            vNorm += result[i] * result[i];
        }
        vNorm = Math.Sqrt(vNorm);

        for (int i = 0; i < result.Length; i++)
        {
            result[i] /= vNorm;
        }

        return result;
    }

    public static Matrix<T> Subtract<T>(this Matrix<T> left, Matrix<T> right)
    {
        if (left.Rows != right.Rows || left.Columns != right.Columns)
            throw new ArgumentException("Matrices must have the same dimensions");

        var ops = MathHelper.GetNumericOperations<T>();
        var result = new Matrix<T>(left.Rows, left.Columns, ops);

        for (int i = 0; i < left.Rows; i++)
        {
            for (int j = 0; j < left.Columns; j++)
            {
                result[i, j] = ops.Subtract(left[i, j], right[i, j]);
            }
        }

        return result;
    }

    public static double Determinant(this Matrix<double> matrix)
    {
        if (matrix == null)
        {
            throw new ArgumentNullException(nameof(matrix), $"{nameof(matrix)} can't be null");
        }

        if (matrix.Rows == 0)
        {
            throw new ArgumentException($"{nameof(matrix)} has to contain at least one row of values", nameof(matrix));
        }

        var columns = matrix.GetRow(0).Length;
        var rows = matrix.GetColumn(0).Length;
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

        if (rows == 2)
        {
            return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0];
        }
        else
        {
            double determinant = 0;
            for (int i = 0; i < rows; i++)
            {
                var tempMatrix = Matrix.CreateDoubleMatrix(rows - 1, rows - 1);
                for (int j = 0; j < rows - 1; j++)
                {
                    for (int k = 0; k < rows - 1; k++)
                    {
                        tempMatrix[j, k] = matrix[j < i ? j : j + 1, k < i ? k : k + 1];
                    }
                }

                determinant += Math.Pow(-1, i) * matrix[0, i] * tempMatrix.Determinant();
            }

            return determinant;
        }
    }

    // Helper method to apply Householder reflection to QR decomposition
    public static (Matrix<double> qMatrix, Matrix<double> rMatrix) 
        ApplyHouseholderTransformationToQR(Vector<double> vector, Matrix<double> qMatrix, Matrix<double> rMatrix, int k)
    {
        var rows = vector.Length;
        for (int i = k; i < rows; i++)
        {
            double sum = 0;
            for (int j = k; j < rows; j++)
            {
                sum += vector[j - k] * rMatrix[j, i];
            }
            for (int j = k; j < rows; j++)
            {
                rMatrix[j, i] -= 2 * vector[j - k] * sum;
            }
        }

        for (int i = 0; i < rows; i++)
        {
            double sum = 0;
            for (int j = k; j < rows; j++)
            {
                sum += vector[j - k] * qMatrix[i, j];
            }
            for (int j = k; j < rows; j++)
            {
                qMatrix[i, j] -= 2 * vector[j - k] * sum;
            }
        }

        return (qMatrix, rMatrix);
    }

    public static (Matrix<double>, Matrix<double>) QRDecomposition(Matrix<double> matrix)
    {
        var rMatrix = matrix.Copy();
        var rows = rMatrix.Rows;
        var qMatrix = Matrix<double>.CreateIdentityMatrix<double>(rows);
        var ops = MathHelper.GetNumericOperations<double>();

        for (int k = 0; k < rows - 1; k++)
        {
            var xVector = new Vector<double>(rows - k, ops);
            for (int i = k; i < rows; i++)
            {
                xVector[i - k] = rMatrix[i, k];
            }

            var hVector = HouseholderVector(xVector);
            (qMatrix, rMatrix) = ApplyHouseholderTransformationToQR(hVector, qMatrix, rMatrix, k);
        }

        return (qMatrix, rMatrix);
    }

    public static Matrix<double> ReduceToHessenbergFormat(Matrix<double> matrix)
    {
        var rows = matrix.Rows;
        var result = Matrix.CreateDoubleMatrix(rows, rows);
        var ops = MathHelper.GetNumericOperations<double>();
        for (int k = 0; k < rows - 2; k++)
        {
            var xVector = new Vector<double>(rows - k - 1, ops);
            for (int i = 0; i < rows - k - 1; i++)
            {
                xVector[i] = matrix[k + 1 + i, k];
            }

            var hVector = CreateHouseholderVector(xVector);
            matrix = ApplyHouseholderTransformation(matrix, hVector, k);
        }

        return matrix;
    }

    public static Vector<double> BackwardSubstitution(this Matrix<double> aMatrix, Vector<double> bVector)
    {
        int n = aMatrix.Rows;
        var ops = MathHelper.GetNumericOperations<double>();
        var xVector = new Vector<double>(n, ops);
        for (int i = n - 1; i >= 0; --i)
        {
            xVector[i] = bVector[i];
            for (int j = i + 1; j < n; ++j)
            {
                xVector[i] -= aMatrix[i, j] * xVector[j];
            }
            xVector[i] /= aMatrix[i, i];
        }

        return xVector;
    }

    public static Vector<Complex> BackwardSubstitution(this Matrix<double> aMatrix, Vector<Complex> bVector)
    {
        int n = aMatrix.Rows;
        var ops = MathHelper.GetNumericOperations<Complex>();
        var xVector = new Vector<Complex>(n, ops);
        for (int i = n - 1; i >= 0; --i)
        {
            xVector[i] = bVector[i];
            for (int j = i + 1; j < n; ++j)
            {
                xVector[i] -= new Complex(aMatrix[i, j], 0) * xVector[j];
            }
            xVector[i] /= new Complex(aMatrix[i, i], 0);
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

    public static IEnumerable<MatrixType> GetMatrixTypes<T>(this Matrix<double> matrix, IMatrixDecomposition<T> matrixDecomposition, 
        double tolerance = double.Epsilon, int subDiagonalThreshold = 1, int superDiagonalThreshold = 1, double sparsityThreshold = 0.5, double denseThreshold = 0.5,
        int blockRows = 2, int blockCols = 2)
    {
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
                    if (complexMatrix.IsUnitaryMatrix(matrixDecomposition))
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

    public static bool IsBlockMatrix(this Matrix<double> matrix, int blockRows, int blockCols)
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

    public static Matrix<double> GetBlock(this Matrix<double> matrix, int startRow, int startCol, int blockRows, int blockCols)
    {
        var block = Matrix.CreateDoubleMatrix(blockRows, blockCols);
        for (int i = 0; i < blockRows; i++)
        {
            for (int j = 0; j < blockCols; j++)
            {
                block[i, j] = matrix[startRow + i, startCol + j];
            }
        }

        return block;
    }

    public static bool IsConsistentBlock(this Matrix<double> block)
    {
        var rows = block.Rows;
        var cols = block.Columns;
        
        // Implement your own logic to check consistency within the block
        // For simplicity, let's assume the block is consistent if all elements are the same
        double firstElement = block[0, 0];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (block[i, j] != firstElement)
                {
                    return false;
                }
            }
        }

        return true;
    }

    public static double Norm(this Vector<double> vector)
    {
        double sum = 0.0;
        int n = vector.Length;
        for (int i = 0; i < n; i++)
        {
            sum += Math.Pow(vector[i], 2);
        }

        return Math.Sqrt(sum);
    }

    public static double DotProduct(this Vector<double> aVector, Vector<double> bVector)
    {
        double result = 0.0;
        int n = aVector.Length;
        for (int i = 0; i < n; i++)
        {
            result += aVector[i] * bVector[i];
        }
          
        return result;
    }

    public static Vector<double> DotProduct(this Matrix<double> aMatrix, Vector<double> bVector)
    {
        int m = aMatrix.Rows, n = aMatrix.Columns;
        var ops = MathHelper.GetNumericOperations<double>();
        var result = new Vector<double>(m, ops);
        for (int i = 0; i < m; i++)
        {
            for (int k = 0; k < n; k++)
            {
                result[i] += aMatrix[i, k] * bVector[k];
            }
        }

        return result;
    }

    public static bool IsUpperTriangularMatrix(this Matrix<double> matrix, double tolerance = double.Epsilon)
    {
        for (int i = 1; i < matrix.Rows; i++)
        {
            for (int j = 0; j < i; j++)
            {
                // If any element below the main diagonal is non-zero, the matrix is not upper triangular.
                if (Math.Abs(matrix[i, j]) > tolerance)
                {
                    return false;
                }
            }
        }

        return true;
    }

    public static bool IsSparseMatrix(this Matrix<double> matrix, double sparsityThreshold = 0.5)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;
        int totalElements = rows * cols;
        int zeroCount = 0;

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (matrix[i, j] == 0)
                {
                    zeroCount++;
                }
            }
        }

        return (double)zeroCount / totalElements >= sparsityThreshold;
    }

    public static bool IsDenseMatrix(this Matrix<double> matrix, double denseThreshold = 0.5)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;
        int totalElements = rows * cols;
        int nonZeroCount = 0;

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (matrix[i, j] != 0)
                {
                    nonZeroCount++;
                }
            }
        }

        return (double)nonZeroCount / totalElements >= denseThreshold;
    }

    public static bool IsLowerTriangularMatrix(this Matrix<double> matrix, double tolerance = double.Epsilon)
    {
        var rows = matrix.Rows;
        for (int i = 0; i < rows - 1; i++)
        {
            for (int j = i + 1; j < rows; j++)
            {
                // If any element above the main diagonal is non-zero, the matrix is not lower triangular.
                if (Math.Abs(matrix[i, j]) > tolerance)
                {
                    return false;
                }
            }
        }

        return true;
    }

    public static bool IsSquareMatrix(this Matrix<double> matrix)
    {
        return matrix.Rows == matrix.Columns;
    }

    public static bool IsRectangularMatrix(this Matrix<double> matrix)
    {
        return matrix.Rows != matrix.Columns;
    }

    public static bool IsSymmetricMatrix(this Matrix<double> matrix)
    {
        var rows = matrix.Rows;
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                if (matrix[i, j] != matrix[j, i])
                {
                    return false;
                }
            }
        }

        return true;
    }

    public static bool IsDiagonalMatrix(this Matrix<double> matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;

        if (rows != cols)
        {
            return false; // A diagonal matrix must be square
        }

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (i != j && matrix[i, j] != 0)
                {
                    return false;
                }
            }
        }

        return true;
    }

    public static bool IsIdentityMatrix(this Matrix<double> matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;

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
                    if (matrix[i, j] != 1)
                    {
                        return false; // Diagonal elements must be 1
                    }
                }
                else
                {
                    if (matrix[i, j] != 0)
                    {
                        return false; // Non-diagonal elements must be 0
                    }
                }
            }
        }

        return true;
    }

    public static bool IsSkewSymmetricMatrix(this Matrix<double> matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;

        if (rows != cols)
        {
            return false; // A skew-symmetric matrix must be square
        }

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (matrix[i, j] != -matrix[j, i])
                {
                    return false;
                }
            }
        }

        return true;
    }

    public static bool IsScalarMatrix(this Matrix<double> matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;

        if (rows != cols)
        {
            return false; // A scalar matrix must be square
        }

        double diagonalValue = matrix[0, 0];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (i == j)
                {
                    if (matrix[i, j] != diagonalValue)
                    {
                        return false; // All diagonal elements must be equal
                    }
                }
                else
                {
                    if (matrix[i, j] != 0)
                    {
                        return false; // All off-diagonal elements must be zero
                    }
                }
            }
        }

        return true;
    }

    public static bool IsUpperBidiagonalMatrix(this Matrix<double> matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;

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
                    if (matrix[i, j] != 0)
                    {
                        return false; // Elements below the main diagonal or above the first superdiagonal must be zero
                    }
                }
            }
        }

        return true;
    }

    public static bool IsLowerBidiagonalMatrix(this Matrix<double> matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;

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
                    if (matrix[i, j] != 0)
                    {
                        return false; // Elements above the main diagonal or below the first subdiagonal must be zero
                    }
                }
            }
        }

        return true;
    }

    public static bool IsTridiagonalMatrix(this Matrix<double> matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;

        if (rows != cols)
        {
            return false; // A tridiagonal matrix must be square
        }

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (Math.Abs(i - j) > 1 && matrix[i, j] != 0)
                {
                    return false; // Elements outside the three diagonals must be zero
                }
            }
        }

        return true;
    }

    public static bool IsBandMatrix(this Matrix<double> matrix, int subDiagonalThreshold = 1, int superDiagonalThreshold = 1)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;

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
                    if (matrix[i, j] != 0)
                    {
                        return false; // Elements outside the specified bands must be zero
                    }
                }
            }
        }

        return true;
    }

    public static bool IsHermitianMatrix(this Matrix<Complex> matrix)
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

    public static bool IsSkewHermitianMatrix(this Matrix<Complex> matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;

        if (rows != cols)
        {
            return false; // A skew-Hermitian matrix must be square
        }

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (matrix[i, j] != -matrix[j, i].Conjugate())
                {
                    return false; // Check if element is equal to the negation of its conjugate transpose
                }
            }
        }

        return true;
    }

    public static bool IsOrthogonalMatrix(this Matrix<double> matrix, IMatrixDecomposition<double> matrixDecomposition)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;

        if (rows != cols)
        {
            return false; // An orthogonal matrix must be square
        }

        var transpose = matrix.Transpose();
        var inverse = matrixDecomposition.Invert();

        // Check if transpose is equal to inverse
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (Math.Abs(transpose[i, j] - inverse[i, j]) > 0.0001)
                {
                    return false;
                }
            }
        }

        return true;
    }

    public static Matrix<Complex> ConjugateTranspose(this Matrix<Complex> matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;
        var result = Matrix.CreateComplexMatrix(cols, rows);

        for (int i = 0; i < cols; i++)
        {
            for (int j = 0; j < rows; j++)
            {
                result[i, j] = matrix[j, i].Conjugate();
            }
        }

        return result;
    }

    public static bool IsSingularMatrix(this Matrix<double> matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;

        if (rows != cols)
        {
            return false; // A singular matrix must be square
        }

        // Calculate the determinant
        double determinant = matrix.GetDeterminant();

        // If determinant is zero, the matrix is singular
        return Math.Abs(determinant) < double.Epsilon;
    }

    public static bool IsNonSingularMatrix(this Matrix<double> matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;

        if (rows != cols)
        {
            return false; // A singular matrix must be square
        }

        // Calculate the determinant
        double determinant = matrix.GetDeterminant();

        // If determinant is zero, the matrix is singular
        return Math.Abs(determinant) >= double.Epsilon;
    }

    public static bool IsPositiveDefiniteMatrix(this Matrix<double> matrix)
    {
        if (!matrix.IsSymmetricMatrix())
        {
            return false; // Positive definite matrices must be symmetric
        }

        var eigenvalues = matrix.Eigenvalues().ToArray();
        // Check if all eigenvalues are positive
        foreach (var eigenvalue in eigenvalues)
        {
            if (eigenvalue <= 0)
            {
                return false;
            }
        }

        return true;
    }

    public static bool IsIdempotentMatrix(this Matrix<double> matrix)
    {
        var rows = matrix.Rows;
        var product = matrix.Multiply(matrix);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < rows; j++)
            {
                if (Math.Abs(matrix[i, j] - product[i, j]) > 1e-10)
                {
                    return false;
                }
            }
        }

        return true;
    }

    public static bool IsStochasticMatrix(this Matrix<double> matrix)
    {
        var rows = matrix.Rows;
        var cols = matrix.Columns;

        for (int i = 0; i < rows; i++)
        {
            double rowSum = 0;
            for (int j = 0; j < cols; j++)
            {
                if (matrix[i, j] < 0)
                {
                    return false; // All elements must be non-negative
                }
                rowSum += matrix[i, j];
            }
            if (Math.Abs(rowSum - 1) > 1e-10)
            {
                return false; // Each row must sum to 1
            }
        }

        return true;
    }

    public static bool IsDoublyStochasticMatrix(this Matrix<double> matrix)
    {
        var rows = matrix.Rows;
        var cols = matrix.Columns;

        // Check if matrix is square
        if (!matrix.IsSquareMatrix())
        {
            return false;
        }

        // Check if all elements are non-negative and rows sum to 1
        for (int i = 0; i < rows; i++)
        {
            double rowSum = 0;
            for (int j = 0; j < cols; j++)
            {
                if (matrix[i, j] < 0)
                {
                    return false; // All elements must be non-negative
                }
                rowSum += matrix[i, j];
            }
            if (Math.Abs(rowSum - 1) > 1e-10)
            {
                return false; // Each row must sum to 1
            }
        }

        // Check if columns sum to 1
        for (int j = 0; j < cols; j++)
        {
            double colSum = 0;
            for (int i = 0; i < rows; i++)
            {
                colSum += matrix[i, j];
            }
            if (Math.Abs(colSum - 1) > 1e-10)
            {
                return false; // Each column must sum to 1
            }
        }

        return true;
    }

    public static bool IsAdjacencyMatrix(this Matrix<double> matrix)
    {
        var rows = matrix.Rows;
        var cols = matrix.Columns;

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
                if (matrix[i, j] != 0 && matrix[i, j] != 1)
                {
                    return false;
                }
            }
        }

        return true;
    }

    public static bool IsCirculantMatrix(this Matrix<double> matrix)
    {
        var rows = matrix.Rows;
        var cols = matrix.Columns;

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
                if (matrix[i, j] != matrix[0, (j + i) % cols])
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

        return rows % Math.Sqrt(cols) == 0;
    }

    public static bool IsVandermondeMatrix(this Matrix<double> matrix)
    {
        var rows = matrix.Rows;
        var cols = matrix.Columns;

        // Check if the matrix has at least 2 rows and if each column forms a geometric progression
        for (int j = 1; j < cols; j++)
        {
            for (int i = 1; i < rows; i++)
            {
                // Check if the current element is equal to the previous element multiplied by x_i
                if (Math.Abs(matrix[i, j] - matrix[i - 1, j] * matrix[i, 0]) > 1e-10)
                {
                    return false;
                }
            }
        }

        return true;
    }

    public static bool IsCauchyMatrix(this Matrix<double> matrix)
    {
        var rows = matrix.Rows;
        var cols = matrix.Columns;

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
                double element = matrix[i, j];
                double x = matrix[i, 0];
                double y = matrix[0, j];

                // Avoid division by zero
                if (Math.Abs(x - y) < 1e-10)
                {
                    return false;
                }

                if (Math.Abs(element - 1.0 / (x - y)) > 1e-10)
                {
                    return false;
                }
            }
        }

        return true;
    }

    public static bool IsHilbertMatrix(this Matrix<double> matrix)
    {
        var rows = matrix.Rows;

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
                double expectedValue = 1.0 / (i + j + 1); // Hilbert matrix definition
                if (Math.Abs(matrix[i, j] - expectedValue) > 1e-10)
                {
                    return false;
                }
            }
        }

        return true;
    }

    public static bool IsCompanionMatrix(this Matrix<double> matrix)
    {
        var rows = matrix.Rows;
        var cols = matrix.Columns;

        // Check if the matrix is square
        if (!matrix.IsSquareMatrix())
        {
            return false;
        }

        // Check if the first row contains the coefficients of a polynomial in reverse order
        for (int j = 0; j < cols - 1; j++)
        {
            if (matrix[0, j] != 0 && matrix[0, j] != 1)
            {
                return false;
            }
        }
        if (matrix[0, cols - 1] != 1)
        {
            return false;
        }

        // Check if each subdiagonal contains a 1
        for (int i = 1; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (i == j + 1 && matrix[i, j] != 1)
                {
                    return false;
                }
                if (i != j + 1 && matrix[i, j] != 0)
                {
                    return false;
                }
            }
        }

        return true;
    }

    public static bool IsHankelMatrix(this Matrix<double> matrix)
    {
        var rows = matrix.Rows;
        var cols = matrix.Columns;

        // Check if each element is the same as the one diagonally above and to the right of it
        for (int i = 0; i < rows - 1; i++)
        {
            for (int j = 0; j < cols - 1; j++)
            {
                if (matrix[i, j] != matrix[i + 1, j + 1])
                {
                    return false;
                }
            }
        }

        return true;
    }

    public static bool IsToeplitzMatrix(this Matrix<double> matrix)
    {
        var rows = matrix.Rows;
        var cols = matrix.Columns;

        // Check if each element is the same as the one diagonally above it
        for (int i = 1; i < rows; i++)
        {
            for (int j = 1; j < cols; j++)
            {
                if (matrix[i, j] != matrix[i - 1, j - 1])
                {
                    return false;
                }
            }
        }

        return true;
    }

    public static bool IsLaplacianMatrix(this Matrix<double> matrix)
    {
        var rows = matrix.Rows;

        // Check if the matrix is square
        if (!matrix.IsSquareMatrix())
        {
            return false;
        }

        for (int i = 0; i < rows; i++)
        {
            double rowSum = 0;
            double colSum = 0;

            for (int j = 0; j < rows; j++)
            {
                // Check for symmetry
                if (matrix[i, j] != matrix[j, i])
                {
                    return false;
                }

                // Check if off-diagonal elements are non-positive
                if (i != j && matrix[i, j] > 0)
                {
                    return false;
                }

                // Sum the row and column elements
                rowSum += matrix[i, j];
                colSum += matrix[j, i];
            }

            // Check if the sum of each row and column is zero
            if (Math.Abs(rowSum) > 1e-10 || Math.Abs(colSum) > 1e-10)
            {
                return false;
            }

            // Check if diagonal elements are non-negative and equal to the sum of absolute values of the off-diagonal elements
            double offDiagonalSum = 0;
            for (int j = 0; j < rows; j++)
            {
                if (i != j)
                {
                    offDiagonalSum += Math.Abs(matrix[i, j]);
                }
            }
            if (matrix[i, i] != offDiagonalSum)
            {
                return false;
            }
        }

        return true;
    }

    public static bool IsIncidenceMatrix(this Matrix<double> matrix)
    {
        var rows = matrix.Rows; // number of vertices
        var cols = matrix.Columns; // number of edges

        for (int j = 0; j < cols; j++)
        {
            int countOnes = 0;

            for (int i = 0; i < rows; i++)
            {
                // Check if the element is either 0 or 1
                if (matrix[i, j] != 0 && matrix[i, j] != 1)
                {
                    return false;
                }

                // Count the number of 1's in the current column
                if (matrix[i, j] == 1)
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

    public static bool IsPermutationMatrix(this Matrix<double> matrix)
    {
        var rows = matrix.Rows;
        int cols = matrix.Columns;

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
                if (matrix[i, j] != 0 && matrix[i, j] != 1)
                {
                    return false;
                }

                // Count the number of 1's in the current row
                if (matrix[i, j] == 1)
                {
                    rowCount++;
                }

                // Count the number of 1's in the current column
                if (matrix[j, i] == 1)
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

    public static bool IsInvolutoryMatrix(this Matrix<double> matrix)
    {
        if (!matrix.IsSquareMatrix())
        {
            return false;
        }
        var product = matrix.Multiply(matrix);

        return product.IsIdentityMatrix();
    }

    public static bool IsOrthogonalProjectionMatrix(this Matrix<double> matrix)
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

    public static bool IsPositiveSemiDefiniteMatrix(this Matrix<double> matrix)
    {
        if (!matrix.IsSymmetricMatrix())
        {
            return false; // Positive semi-definite matrices must be symmetric
        }

        var eigenvalues = matrix.Eigenvalues().ToArray();
        // Check if all eigenvalues are non-negative
        foreach (var eigenvalue in eigenvalues)
        {
            if (eigenvalue < 0)
            {
                return false;
            }
        }

        return true;
    }

    public static Vector<double> Eigenvalues(this Matrix<double> matrix)
    {
        // QR algorithm for finding eigenvalues of a symmetric matrix
        var rows = matrix.Rows;
        var a = Matrix.CreateDoubleMatrix(rows, rows);
    
        // Copy the matrix data manually
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < rows; j++)
            {
                a[i, j] = matrix[i, j];
            }
        }

        var ops = MathHelper.GetNumericOperations<double>();
        var eigenvalues = new Vector<double>(rows, ops);

        for (int k = rows - 1; k > 0; k--)
        {
            while (Math.Abs(a[k, k - 1]) > double.Epsilon)
            {
                double mu = a[k, k];
                for (int i = 0; i <= k; i++)
                {
                    for (int j = 0; j <= k; j++)
                    {
                        a[i, j] -= mu * a[k, i] * a[k, j];
                    }
                }

                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < rows; j++)
                    {
                        a[i, j] -= mu * a[i, k] * a[j, k];
                    }
                }
            }

            eigenvalues[k] = a[k, k];
            for (int i = 0; i <= k; i++)
            {
                a[k, i] = a[i, k] = 0.0;
            }
        }
        eigenvalues[0] = a[0, 0];

        return eigenvalues;
    }

    // Method to calculate the determinant of a matrix
    public static double GetDeterminant(this Matrix<double> matrix)
    {
        var rows = matrix.Rows;

        // Base case: for 1x1 matrix, determinant is the single element
        if (rows == 1)
        {
            return matrix[0, 0];
        }

        double det = 0;
        // Recursive case: compute the determinant using cofactor expansion
        for (int j = 0; j < rows; j++)
        {
            // Calculate the cofactor of matrix[0, j]
            var submatrix = Matrix<double>.CreateMatrix<double>(rows - 1, rows - 1);
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
            det += Math.Pow(-1, j) * matrix[0, j] * submatrix.GetDeterminant();
        }

        return det;
    }

    public static bool IsUnitaryMatrix(this Matrix<Complex> matrix, IMatrixDecomposition<Complex> matrixDecomposition)
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

    public static bool IsZeroMatrix(this Matrix<double> matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (matrix[i, j] != 0)
                {
                    return false;
                }
            }
        }

        return true;
    }

    public static Matrix<double> InvertUpperTriangularMatrix(this Matrix<double> matrix)
    {
        int n = matrix.Rows;

        // Create the inverse matrix
        var inverse = Matrix.CreateDoubleMatrix(n, n);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i >= j)
                {
                    inverse[i, j] = (double)1.0 / matrix[i, j];
                }
                else
                {
                    inverse[i, j] = 0;
                }
            }
        }

        return inverse;
    }

    public static Matrix<double> InvertLowerTriangularMatrix(this Matrix<double> matrix)
    {
        int n = matrix.Rows;
        var invL = Matrix.CreateDoubleMatrix(n, n);

        for (int i = 0; i < n; i++)
        {
            invL[i, i] = 1.0 / matrix[i, i];
            for (int j = 0; j < i; j++)
            {
                double sum = 0;
                for (int k = j; k < i; k++)
                {
                    sum += matrix[i, k] * invL[k, j];
                }
                invL[i, j] = -sum / matrix[i, i];
            }
        }

        return invL;
    }

    public static double Hypotenuse(double x, double y)
    {
        double xabs = Math.Abs(x), yabs = Math.Abs(y), min, max;

        if (xabs < yabs)
        {
            min = xabs; max = yabs;
        }
        else
        {
            min = yabs; max = xabs;
        }

        if (min == 0)
        {
            return max;
        }

        double u = min / max;

        return max * Math.Sqrt(1 + u * u);
    }

    public static double Hypotenuse(params double[] values)
    {
        double sum = 0.0;
        foreach (var value in values)
        {
            sum += value * value;
        }

        return Math.Sqrt(sum);
    }

    public static Vector<double> ForwardSubstitution(this Matrix<double> aMatrix, Vector<double> bVector)
    {
        int n = aMatrix.Rows;
        var ops = MathHelper.GetNumericOperations<double>();
        var x = new Vector<double>(n, ops);

        for (int i = 0; i < n; i++)
        {
            x[i] = bVector[i];
            for (int j = 0; j < i; j++)
            {
                x[i] -= aMatrix[i, j] * x[j];
            }
            x[i] /= aMatrix[i, i];
        }

        return x;
    }

    public static Matrix<Complex> ToComplexMatrix(this Matrix<double> matrix)
    {
        var rows = matrix.Rows;
        var complexMatrix = Matrix<Complex>.CreateMatrix<Complex>(rows, rows);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < rows; j++)
            {
                complexMatrix[i, j] = new Complex(matrix[i, j], 0);
            }
        }

        return complexMatrix;
    }

    public static Vector<Complex> ToComplexVector(this Vector<double> vector)
    {
        var count = vector.Length;
        var ops = MathHelper.GetNumericOperations<Complex>();
        var complexVector = new Vector<Complex>(count, ops);
        for (int i = 0; i < count; i++)
        {
            complexVector[i] = new Complex(vector[i], 0);
        }

        return complexVector;
    }

    public static Matrix<double> ToRealMatrix(this Matrix<Complex> matrix)
    {
        var rows = matrix.Rows;
        var realMatrix = Matrix.CreateDoubleMatrix(rows, rows);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < rows; j++)
            {
                realMatrix[i, j] = matrix[i, j].Real;
            }
        }

        return realMatrix;
    }

    public static Vector<double> ToRealVector(this Vector<Complex> vector)
    {
        var count = vector.Length;
        var ops = MathHelper.GetNumericOperations<double>();
        var realVector = new Vector<double>(count, ops);
        for (int i = 0; i < count; i++)
        {
            realVector[i] = vector[i].Real;
        }

        return realVector;
    }

    public static Vector<Complex> ForwardSubstitution(this Matrix<Complex> aMatrix, Vector<Complex> bVector)
    {
        int n = aMatrix.Rows;
        var ops = MathHelper.GetNumericOperations<Complex>();
        var x = new Vector<Complex>(n, ops);

        for (int i = 0; i < n; i++)
        {
            x[i] = bVector[i];
            for (int j = 0; j < i; j++)
            {
                x[i] -= aMatrix[i, j] * x[j];
            }
            x[i] /= aMatrix[i, i];
        }

        return x;
    }

    public static Matrix<double> ApplyHouseholderTransformation(Matrix<double> matrix, Vector<double> vector, int k)
    {
        var rows = matrix.Rows;
        for (int i = k + 1; i < rows; i++)
        {
            double sum = 0;
            for (int j = k + 1; j < rows; j++)
            {
                sum += vector[j - k - 1] * matrix[j, i];
            }
            for (int j = k + 1; j < rows; j++)
            {
                matrix[j, i] -= 2 * vector[j - k - 1] * sum;
            }
        }

        for (int i = 0; i < rows; i++)
        {
            double sum = 0;
            for (int j = k + 1; j < rows; j++)
            {
                sum += vector[j - k - 1] * matrix[i, j];
            }
            for (int j = k + 1; j < rows; j++)
            {
                matrix[i, j] -= 2 * vector[j - k - 1] * sum;
            }
        }

        return matrix;
    }


    public static Vector<double> CreateHouseholderVector(Vector<double> x)
    {
        var ops = MathHelper.GetNumericOperations<double>();
        var v = new Vector<double>(x.Length, ops);
        double norm = 0;
        for (int i = 0; i < x.Length; i++)
        {
            norm += x[i] * x[i];
        }
        norm = Math.Sqrt(norm);

        v[0] = x[0] + Math.Sign(x[0]) * norm;
        for (int i = 1; i < x.Length; i++)
        {
            v[i] = x[i];
        }

        double vNorm = 0;
        for (int i = 0; i < v.Length; i++)
        {
            vNorm += v[i] * v[i];
        }
        vNorm = Math.Sqrt(vNorm);

        for (int i = 0; i < v.Length; i++)
        {
            v[i] /= vNorm;
        }

        return v;
    }

    public static (double, Vector<double>) PowerIteration(Matrix<double> aMatrix, int maxIterations, double tolerance)
    {
        var rows = aMatrix.Rows;
        var ops = MathHelper.GetNumericOperations<double>();
        var bVector = new Vector<double>(rows, ops);
        var b2Vector = new Vector<double>(rows, ops);
        double eigenvalue = 0;

        // Initial guess for the eigenvector
        for (int i = 0; i < rows; i++)
        {
            bVector[i] = 1.0;
        }

        for (int iter = 0; iter < maxIterations; iter++)
        {
            // Multiply A by the vector b
            for (int i = 0; i < rows; i++)
            {
                b2Vector[i] = 0;
                for (int j = 0; j < rows; j++)
                {
                    b2Vector[i] += aMatrix[i, j] * bVector[j];
                }
            }

            // Normalize the vector
            double norm = 0;
            for (int i = 0; i < rows; i++)
            {
                norm += b2Vector[i] * b2Vector[i];
            }
            norm = Math.Sqrt(norm);
            for (int i = 0; i < rows; i++)
            {
                b2Vector[i] /= norm;
            }

            // Estimate the eigenvalue
            double newEigenvalue = 0;
            for (int i = 0; i < rows; i++)
            {
                newEigenvalue += b2Vector[i] * b2Vector[i];
            }

            // Check for convergence
            if (Math.Abs(newEigenvalue - eigenvalue) < tolerance)
            {
                break;
            }
            eigenvalue = newEigenvalue;
            Array.Copy(b2Vector.ToArray(), bVector.ToArray(), rows);
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

    public static Matrix<Complex> InvertUnitaryMatrix(this Matrix<Complex> matrix)
    {
        return matrix.Transpose();
    }

    public static Matrix<double> GaussianEliminationInversion(this Matrix<double> matrix)
    {
        var rows = matrix.Rows;
        var augmentedMatrix = Matrix.CreateDoubleMatrix(rows, 2 * rows);

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
            augmentedMatrix[i, i + rows] = 1;
        }

        // Perform Gaussian elimination with partial pivoting
        for (int i = 0; i < rows; i++)
        {
            // Find pivot row
            int maxRowIndex = i;
            double maxValue = Math.Abs(augmentedMatrix[i, i]);
            for (int k = i + 1; k < rows; k++)
            {
                double absValue = Math.Abs(augmentedMatrix[k, i]);
                if (absValue > maxValue)
                {
                    maxRowIndex = k;
                    maxValue = absValue;
                }
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
            double pivot = augmentedMatrix[i, i];
            for (int j = 0; j < 2 * rows; j++)
            {
                augmentedMatrix[i, j] /= pivot;
            }

            // Make other elements in the column zero
            for (int k = 0; k < rows; k++)
            {
                if (k != i)
                {
                    double factor = augmentedMatrix[k, i];
                    for (int j = 0; j < 2 * rows; j++)
                    {
                        augmentedMatrix[k, j] -= factor * augmentedMatrix[i, j];
                    }
                }
            }
        }

        // Extract the right half of the augmented matrix (the inverse)
        var inverseMatrix = Matrix<double>.CreateMatrix<double>(rows, rows);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < rows; j++)
            {
                inverseMatrix[i, j] = augmentedMatrix[i, j + rows];
            }
        }

        return inverseMatrix;
    }

    public static void GaussJordanElimination(this Matrix<double> matrix, Vector<double> vector, out Matrix<double> inverseMatrix, 
        out Vector<double> coefficients)
    {
        var matrixA = matrix.Copy();
        var matrixB = Matrix<double>.CreateFromVector(vector);
        int rows = matrixA.Rows, columns = matrixB.Columns, rowIndex = 0, columnIndex = 0;
        var indexArray = new int[rows];
        var indexRowArray = new int[rows];
        var indexColumnArray = new int[rows];
        double inverse = 0.0, dummyValue = 0.0;

        for (int i = 0; i < rows; i++)
        {
            var big = 0.0;
            for (int j = 0; j < rows; j++)
            {
                if (indexArray[j] != 1)
                {
                    for (int k = 0; k < rows; k++)
                    {
                        if (indexArray[k] == 0)
                        {
                            var currentValue = Math.Abs(matrixA[j, k]);
                            if (currentValue >= big)
                            {
                                big = currentValue;
                                rowIndex = j;
                                columnIndex = k;
                            }
                        }
                    }
                }
            }
            indexArray[columnIndex] += 1;

            if (rowIndex != columnIndex)
            {
                for (int j = 0; j < rows; j++)
                {
                    matrixA[rowIndex, j] = matrixA[columnIndex, j];
                }
                for (int j = 0; j < columns; j++)
                {
                    matrixB[rowIndex, j] = matrixB[columnIndex, j];
                }
            }
            indexRowArray[i] = rowIndex;
            indexColumnArray[i] = columnIndex;

            if (matrixA[columnIndex, columnIndex] == 0)
            {
                throw new InvalidOperationException("Singular matrix");
            }
            inverse = 1.0 / matrixA[columnIndex, columnIndex];
            matrixA[columnIndex, columnIndex] = 1.0;

            for (int j = 0; j < rows; j++)
            {
                matrixA[columnIndex, j] *= inverse;
            }
            for (int j = 0; j < columns; j++)
            {
                matrixB[columnIndex, j] *= inverse;
            }
            for (int j = 0; j < rows; j++)
            {
                if (j != columnIndex)
                {
                    dummyValue = matrixA[j, columnIndex];
                    matrixA[j, columnIndex] = 0;
                    for (int k = 0; k < rows; k++)
                    {
                        matrixA[k, j] -= matrixA[columnIndex, j] * dummyValue;
                    }
                    for (int k = 0; k < columns; k++)
                    {
                        matrixB[k, j] -= matrixB[columnIndex, j] * dummyValue;
                    }
                }
            }
        }

        for (int i = rows - 1; i >= 0; i--)
        {
            if (indexRowArray[i] != indexColumnArray[i])
            {
                for (int j = 0; j < rows; j++)
                {
                    matrixA[j, indexRowArray[i]] = matrixA[j, indexColumnArray[i]];
                }
            }
        }

        inverseMatrix = matrixA;
        coefficients = new Vector<double>(rows);
        for (int i = 0; i < rows; i++)
        {
            coefficients[i] = matrixB[i, 0];
        }
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

    public static int GetNullity(this Matrix<double> matrix, double threshold = -1)
    {
        var rows = matrix.Rows;
        var columns = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<double>();
        var weightsVector = new Vector<double>(columns, ops);
        var thresh = threshold >= 0 ? threshold : 0.5 * Math.Sqrt(rows + columns + 1) * weightsVector[0] * double.Epsilon;
        int nullity = 0;

        for (int i = 0; i < columns; i++)
        {
            if (weightsVector[i] <= thresh)
            {
                nullity++;
            }
        }

        return nullity;
    }

    public static Matrix<double> Nullspace(this Matrix<double> matrix, double threshold = -1)
    {
        int rows = matrix.Rows, columns = matrix.Columns, nullIndex = 0;
        var ops = MathHelper.GetNumericOperations<double>();
        var weightsVector = new Vector<double>(columns, ops);
        var thresh = threshold >= 0 ? threshold : 0.5 * Math.Sqrt(rows + columns + 1) * weightsVector[0] * double.Epsilon;
        var nullspaceMatrix = Matrix.CreateDoubleMatrix(columns, matrix.GetNullity(thresh));
        var vMatrix = Matrix.CreateDoubleMatrix(columns, columns);

        for (int i = 0; i < columns; i++)
        {
            if (weightsVector[i] <= thresh)
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

    public static Matrix<double> GetRange(this Matrix<double> matrix, double threshold = -1)
    {
        int rows = matrix.Rows, columns = matrix.Columns, rank = 0;
        var ops = MathHelper.GetNumericOperations<double>();
        var weightsVector = new Vector<double>(columns, ops);
        var rangeMatrix = Matrix.CreateDoubleMatrix(rows, matrix.GetRank(threshold));
        var uMatrix = matrix.Copy();

        for (int i = 0; i < columns; i++)
        {
            if (weightsVector[i] > threshold)
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

    public static int GetRank(this Matrix<double> matrix, double threshold = -1)
    {
        var rows = matrix.Rows;
        var columns = matrix.Columns;
        var ops = MathHelper.GetNumericOperations<double>();
        var weightsVector = new Vector<double>(columns, ops);
        var thresh = threshold >= 0 ? threshold : 0.5 * Math.Sqrt(rows + columns + 1) * weightsVector[0] * double.Epsilon;
        int rank = 0;

        for (int i = 0; i < columns; i++)
        {
            if (weightsVector[i] > thresh)
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

    public static void TridiagonalSolve(Vector<double> vector1, Vector<double> vector2, Vector<double> vector3,
        Vector<double> solutionVector, Vector<double> actualVector)
    {
        var size = vector1.Length;
        double bet;
        var ops = MathHelper.GetNumericOperations<double>();
        var gamVector = new Vector<double>(size, ops);

        if (vector2[0] == 0)
        {
            throw new InvalidOperationException("Not a tridiagonal matrix!");
        }

        bet = vector2[0];
        solutionVector[0] = actualVector[0] / bet;
        for (int i = 1; i < size; i++)
        {
            gamVector[i] = vector3[i - 1] / bet;
            bet = vector2[i] - vector1[i] * gamVector[i];

            if (bet == 0)
            {
                throw new InvalidOperationException("Not a tridiagonal matrix!");
            }

            solutionVector[i] = actualVector[i] - vector1[i] * solutionVector[i - 1] / bet;
        }

        for (int i = size - 2; i >= 0; i--)
        {
            solutionVector[i] -= gamVector[i + 1] * solutionVector[i + 1];
        }
    }

    public static void BandDiagonalMultiply(int leftSide, int rightSide, Matrix<double> matrix, Vector<double> solutionVector, Vector<double> actualVector)
    {
        var size = matrix.Rows;

        for (int i = 0; i < size; i++)
        {
            var k = i - leftSide;
            var temp = Math.Min(leftSide + rightSide + 1, size - k);
            solutionVector[i] = 0;
            for (int j = Math.Max(0, -k); j < temp; j++)
            {
                solutionVector[i] += matrix[i, j] * actualVector[j + k];
            }
        }
    }
}