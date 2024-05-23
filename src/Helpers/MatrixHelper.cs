namespace AiDotNet.Helpers;

public static class MatrixHelper
{
    public static double CalculateDeterminantRecursive(double[,] matrix)
    {
        var size = matrix.GetLength(0);

        if (size != matrix.GetLength(1))
        {
            throw new ArgumentException("Matrix must be square.");
        }

        if (size == 1)
        {
            return matrix[0, 0];
        }

        double determinant = 0;

        for (var i = 0; i < size; i++)
        {
            var subMatrix = CreateSubMatrix(matrix, 0, i);
            determinant += Math.Pow(-1, i) * matrix[0, i] * CalculateDeterminantRecursive(subMatrix);
        }

        return determinant;
    }

    private static double[,] CreateSubMatrix(double[,] matrix, int excludeRowIndex, int excludeColumnIndex)
    {
        var size = matrix.GetLength(0);
        var subMatrix = new double[size - 1, size - 1];

        var r = 0;
        for (var i = 0; i < size; i++)
        {
            if (i == excludeRowIndex)
            {
                continue;
            }

            var c = 0;
            for (var j = 0; j < size; j++)
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
        return new Vector<T>(Enumerable.Range(0, matrix.RowCount)
                .Select(x => matrix[x, columnIndex]));
    }

    public static T[] GetRow<T>(this T[,] matrix, int rowIndex)
    {
        return Enumerable.Range(0, matrix.GetLength(1))
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

        var length = vector.Count;
        if (rows * columns != length)
        {
            throw new ArgumentException($"{nameof(rows)} and {nameof(columns)} multiplied together needs to be equal the array length");
        }

        var matrix = new Matrix<T>(rows, columns);
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
        int n = vector.Count;
        var matrix = new Matrix<T>(n, 1);
        for (int i = 0; i < n; ++i)
        {
            matrix[i, 0] = vector[i];
        }
            
        return matrix;
    }

    // Helper method to compute Householder vector
    public static Vector<double> HouseholderVector(Vector<double> xVector)
    {
        var result = new Vector<double>(xVector.Count);
        double norm = 0;
        for (int i = 0; i < xVector.Count; i++)
        {
            norm += xVector[i] * xVector[i];
        }
        norm = Math.Sqrt(norm);

        result[0] = xVector[0] + Math.Sign(xVector[0]) * norm;
        for (int i = 1; i < xVector.Count; i++)
        {
            result[i] = xVector[i];
        }

        double vNorm = 0;
        for (int i = 0; i < result.Count; i++)
        {
            vNorm += result[i] * result[i];
        }
        vNorm = Math.Sqrt(vNorm);

        for (int i = 0; i < result.Count; i++)
        {
            result[i] /= vNorm;
        }

        return result;
    }

    public static Matrix<T> Duplicate<T>(this Matrix<T> matrix)
    {
        if (matrix == null)
        {
            throw new ArgumentNullException(nameof(matrix), $"{nameof(matrix)} can't be null");
        }

        return new LinearAlgebra.Matrix<T>(matrix.Values);
    }

    public static LinearAlgebra.Vector<T> Duplicate<T>(this LinearAlgebra.Vector<T> vector)
    {
        if (vector == null)
        {
            throw new ArgumentNullException(nameof(vector), $"{nameof(vector)} can't be null");
        }

        return new LinearAlgebra.Vector<T>(vector.Values);
    }

    public static double[,] Duplicate(this double[,] matrix)
    {
        if (matrix == null)
        {
            throw new ArgumentNullException(nameof(matrix), $"{nameof(matrix)} can't be null");
        }

        if (matrix.Length == 0)
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

        double[,] result = new double[rows, columns];
        for (int i = 0; i < columns; i++)
        {
            for (int j = 0; j < rows; j++)
            {
                result[i, j] = matrix[i, j];
            }
        }

        return result;
    }

    public static double Determinant(this double[,] matrix)
    {
        if (matrix == null)
        {
            throw new ArgumentNullException(nameof(matrix), $"{nameof(matrix)} can't be null");
        }

        if (matrix.Length == 0)
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
                double[,] tempMatrix = new double[rows - 1, rows - 1];
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
        var rows = vector.Count;
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

    // Helper method to perform QR decomposition using Householder reflections
    public static (Matrix<double>, Matrix<double>) QRDecomposition(Matrix<double> matrix)
    {
        var rMatrix = matrix.Duplicate();
        var rows = rMatrix.RowCount;
        var qMatrix = CreateIdentityMatrix<double>(rows);

        for (int k = 0; k < rows - 1; k++)
        {
            var xVector = new Vector<double>(rows - k);
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
        var rows = matrix.RowCount;
        var result = new Matrix<double>(rows, rows);
        for (int k = 0; k < rows - 2; k++)
        {
            var xVector = new Vector<double>(rows - k - 1);
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
        int n = aMatrix.RowCount;
        var xVector = new Vector<double>(n);
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
        int n = aMatrix.RowCount;
        var xVector = new Vector<Complex>(n);
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

    public static double Norm(this Vector<double> vector)
    {
        double sum = 0.0;
        int n = vector.Count;
        for (int i = 0; i < n; i++)
        {
            sum += Math.Pow(vector[i], 2);
        }

        return Math.Sqrt(sum);
    }

    public static double DotProduct(this Vector<double> aVector, Vector<double> bVector)
    {
        double result = 0.0;
        int n = aVector.Count;
        for (int i = 0; i < n; i++)
        {
            result += aVector[i] * bVector[i];
        }
          
        return result;
    }

    public static Vector<double> DotProduct(this Matrix<double> aMatrix, Vector<double> bVector)
    {
        int m = aMatrix.RowCount, n = aMatrix.ColumnCount;
        var result = new Vector<double>(m);
        for (int i = 0; i < m; i++)
        {
            for (int k = 0; k < n; k++)
            {
                result[i] += aMatrix[i, k] * bVector[k];
            }
        }

        return result;
    }

    public static double[,] Invert(this Matrix<double> matrix)
    {
        // qr inversion
        matrix.Decompose(out double[,] qMatrix, out double[,] rMatrix);
        var rInverted = rMatrix.InvertUpperTriangularMatrix();
        var qTransposed = qMatrix.Transpose();

        return rInverted.DotProduct(qTransposed);
    }

    public static double[,] Invert(this double[] vector)
    {
        int length = vector.Length;
        var result = new double[length, length];
        for (int i = 0; i < length; i++)
        {
            result[i, i] = (double)1.0 / vector[i];
        }

        return result;
    }

    public static bool IsUpperTriangularMatrix(this Matrix<double> matrix, double tolerance = double.Epsilon)
    {
        for (int i = 1; i < matrix.RowCount; i++)
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
        int rows = matrix.RowCount;
        int cols = matrix.ColumnCount;
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
        int rows = matrix.RowCount;
        int cols = matrix.ColumnCount;
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
        var rows = matrix.RowCount;
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
        return matrix.RowCount == matrix.ColumnCount;
    }

    public static bool IsRectangularMatrix(this Matrix<double> matrix)
    {
        return matrix.RowCount != matrix.ColumnCount;
    }

    public static bool IsSymmetricMatrix(this Matrix<double> matrix)
    {
        var rows = matrix.RowCount;
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
        int rows = matrix.RowCount;
        int cols = matrix.ColumnCount;

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
        int rows = matrix.RowCount;
        int cols = matrix.ColumnCount;

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
        int rows = matrix.RowCount;
        int cols = matrix.ColumnCount;

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
        int rows = matrix.RowCount;
        int cols = matrix.ColumnCount;

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
        int rows = matrix.RowCount;
        int cols = matrix.ColumnCount;

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
        int rows = matrix.RowCount;
        int cols = matrix.ColumnCount;

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
        int rows = matrix.RowCount;
        int cols = matrix.ColumnCount;

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
        int rows = matrix.RowCount;
        int cols = matrix.ColumnCount;

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
        int rows = matrix.RowCount;
        int cols = matrix.ColumnCount;

        if (rows != cols)
        {
            return false; // A Hermitian matrix must be square
        }

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (matrix[i, j] != Complex.Conjugate(matrix[j, i]))
                {
                    return false; // Check if element is equal to its conjugate transpose
                }
            }
        }

        return true;
    }

    public static bool IsSkewHermitianMatrix(Matrix<Complex> matrix)
    {
        int rows = matrix.RowCount;
        int cols = matrix.ColumnCount;

        if (rows != cols)
        {
            return false; // A skew-Hermitian matrix must be square
        }

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (matrix[i, j] != -Complex.Conjugate(matrix[j, i]))
                {
                    return false; // Check if element is equal to the negation of its conjugate transpose
                }
            }
        }

        return true;
    }

    public static bool IsOrthogonalMatrix(this Matrix<double> matrix, IMatrixDecomposition<double> matrixDecomposition)
    {
        int rows = matrix.RowCount;
        int cols = matrix.ColumnCount;

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
        int rows = matrix.RowCount;
        int cols = matrix.ColumnCount;
        var result = new Matrix<Complex>(cols, rows);

        for (int i = 0; i < cols; i++)
        {
            for (int j = 0; j < rows; j++)
            {
                result[i, j] = Complex.Conjugate(matrix[j, i]);
            }
        }

        return result;
    }

    public static bool IsSingularMatrix(this Matrix<double> matrix)
    {
        int rows = matrix.RowCount;
        int cols = matrix.ColumnCount;

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
        int rows = matrix.RowCount;
        int cols = matrix.ColumnCount;

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

        var eigenvalues = matrix.Eigenvalues().Values;
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

    public static Vector<double> Eigenvalues(this Matrix<double> matrix)
    {
        // QR algorithm for finding eigenvalues of a symmetric matrix
        var rows = matrix.RowCount;
        var a = new Matrix<double>(rows, rows);
        Array.Copy(matrix.Values, a.Values, matrix.ColumnCount);
        double epsilon = 1e-10; // Precision threshold
        var eigenvalues = new Vector<double>(rows);

        for (int k = rows - 1; k > 0; k--)
        {
            while (Math.Abs(a[k, k - 1]) > epsilon)
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
        var rows = matrix.RowCount;

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
            var submatrix = new Matrix<double>(rows - 1, rows - 1);
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
        int rows = matrix.RowCount;
        int cols = matrix.ColumnCount;

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
        int rows = matrix.RowCount;
        int cols = matrix.ColumnCount;

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
        int n = matrix.RowCount;

        // Create the inverse matrix
        var inverse = new Matrix<double>(n, n);

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
        int n = matrix.RowCount;
        var invL = new Matrix<double>(n, n);

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

    public static Vector<double> ForwardSubstitution(Matrix<double> aMatrix, Vector<double> bVector)
    {
        int n = aMatrix.RowCount;
        var x = new Vector<double>(n);

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
        var rows = matrix.RowCount;
        var complexMatrix = new Matrix<Complex>(rows, rows);
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
        var count = vector.Count;
        var complexVector = new Vector<Complex>(count);
        for (int i = 0; i < count; i++)
        {
            complexVector[i] = new Complex(vector[i], 0);
        }

        return complexVector;
    }

    public static Matrix<double> ToRealMatrix(this Matrix<Complex> matrix)
    {
        var rows = matrix.RowCount;
        var realMatrix = new Matrix<double>(rows, rows);
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
        var count = vector.Count;
        var realVector = new Vector<double>(count);
        for (int i = 0; i < count; i++)
        {
            realVector[i] = vector[i, j].Real;
        }

        return realVector;
    }

    public static Vector<Complex> ForwardSubstitution(Matrix<Complex> aMatrix, Vector<Complex> bVector)
    {
        int n = aMatrix.RowCount;
        var x = new Vector<Complex>(n);

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

    public static double[,] Decompose(this double[,] matrix, out double d)
    {
        // Crout's Method
        if (matrix == null)
        {
            throw new ArgumentNullException(nameof(matrix), $"{nameof(matrix)} can't be null");
        }

        if (matrix.Length == 0)
        {
            throw new ArgumentException($"{nameof(matrix)} has to contain at least one row of values", nameof(matrix));
        }

        var columns = matrix.ColumnCount;
        var rows = matrix.RowCount;
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

        int[] indexArray = new int[rows];
        for (int i = 0; i < rows; i++)
        {
            indexArray[i] = i;
        }

        var vv = new double[rows];
        double max, sum, temp;
        int maxInt = 0;
        d = 1.0;
        var result = matrix.Duplicate();

        for (int i = 0; i < rows; i++)
        {
            max = 0;
            for (int j = 0; j < rows; j++)
            {
                var current = Math.Abs(result[i, j]);
                max = Math.Max(max, current);
            }

            if (max == 0)
            {
                throw new DivideByZeroException($"You will cause a divide by zero exception since your source {nameof(matrix)} contains all zero values");
            }

            vv[i] = (double)1.0 / max;
        }

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < i; j++)
            {
                sum = result[j, i];
                for (int k = 0; k < j; k++)
                {
                    sum -= result[j, k] * result[k, i];
                }

                result[j, i] = sum;
            }

            max = 0;
            for (int j = i; j < rows; j++)
            {
                sum = result[j, i];
                for (int k = 0; k < i; k++)
                {
                    sum -= result[j, k] * result[k, i];
                }

                result[j, i] = sum;
                temp = vv[j] * Math.Abs(sum);
                if (temp >= max)
                {
                    max = temp;
                    maxInt = j;
                }
            }

            if (i != maxInt)
            {
                for (int j = 0; j < rows; j++)
                {
                    temp = result[maxInt, j];
                    result[maxInt, j] = result[i, j];
                    result[i, j] = temp;
                }

                d = d * -1;
                vv[maxInt] = vv[i];
            }

            indexArray[i] = maxInt;
            if (result[i, i] == 0)
            {
                result[i, i] = double.MaxValue;
            }

            if (i != rows - 1)
            {
                temp = (double)1.0 / result[i, i];
                for (int j = 1 + 1; j < rows; j++)
                {
                    result[j, i] *= temp;
                }
            }
        }

        return result;
    }

    public static Matrix<double> ApplyHouseholderTransformation(Matrix<double> matrix, Vector<double> vector, int k)
    {
        var rows = matrix.RowCount;
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

    // Helper method to compute Householder vector
    public static Vector<double> CreateHouseholderVector(Vector<double> x)
    {
        var v = new Vector<double>(x.Count);
        double norm = 0;
        for (int i = 0; i < x.Count; i++)
        {
            norm += x[i] * x[i];
        }
        norm = Math.Sqrt(norm);

        v[0] = x[0] + Math.Sign(x[0]) * norm;
        for (int i = 1; i < x.Count; i++)
        {
            v[i] = x[i];
        }

        double vNorm = 0;
        for (int i = 0; i < v.Count; i++)
        {
            vNorm += v[i] * v[i];
        }
        vNorm = Math.Sqrt(vNorm);

        for (int i = 0; i < v.Count; i++)
        {
            v[i] /= vNorm;
        }

        return v;
    }

    // Power Iteration method to find the dominant eigenvalue and eigenvector
    public static (double, Vector<double>) PowerIteration(Matrix<double> aMatrix, int maxIterations, double tolerance)
    {
        var rows = aMatrix.RowCount;
        var bVector = new Vector<double>(rows);
        var b2Vector = new Vector<double>(rows);
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
            Array.Copy(b2Vector.Values, bVector.Values, rows);
        }

        return (eigenvalue, b2Vector);
    }

    public static void SwapRows<T>(this Matrix<T> matrix, int row1Index, int row2Index)
    {
        var rows = matrix.RowCount;
        for (int i = 0; i < rows; i++)
        {
            var temp = matrix[row1Index, i];
            matrix[row1Index, i] = matrix[row2Index, i];
            matrix[row2Index, i] = temp;
        }
    }

    public static Matrix<double> InvertDiagonalMatrix(this Matrix<double> matrix)
    {
        var rows = matrix.RowCount;
        var invertedMatrix = new Matrix<double>(rows, rows);
        for (int i = 0; i < rows; i++)
        {
            invertedMatrix[i, i] = 1.0 / matrix[i, i];
        }

        return invertedMatrix;
    }

    public static Matrix<Complex> InvertUnitaryMatrix(this Matrix<Complex> matrix)
    {
        return matrix.Transpose();
    }

    public static Matrix<double> GaussianEliminationInversion(this Matrix<double> matrix)
    {
        var rows = matrix.RowCount;
        var augmentedMatrix = new Matrix<double>(rows, 2 * rows);

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
        var inverseMatrix = new Matrix<double>(rows, rows);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < rows; j++)
            {
                inverseMatrix[i, j] = augmentedMatrix[i, j + rows];
            }
        }

        return inverseMatrix;
    }

    public static void GaussJordanElimination(this double[,] matrix, double[,] vector, out double[,] inverseMatrix, out double[,] coefficients)
    {
        var matrixA = matrix.Duplicate();
        var matrixB = vector.Duplicate();
        int rows = matrixA.GetLength(0), columns = matrixB.GetLength(1), rowIndex = 0, columnIndex = 0;
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
        coefficients = matrixB;
    }

    public static T CreateJaggedArray<T>(params int[] lengths)
    {
        return (T)InitializeJaggedArray<T>(typeof(T).GetElementType(), 0, lengths);
    }

    public static object InitializeJaggedArray<T>(Type? type, int index, int[] lengths)
    {
        if (type != null)
        {
            var array = Array.CreateInstance(type, lengths[index]);
            var elementType = type.GetElementType();

            if (elementType != null)
            {
                for (int i = 0; i < lengths[index]; i++)
                {
                    array.SetValue(
                        InitializeJaggedArray<T>(elementType, index + 1, lengths), i);
                }
            }

            return array;
        }

        return new T[lengths[index]];
    }

    public static void DecomposeLQ(this double[,] matrix, out double[,] lMatrix, out double[,] qMatrix)
    {
        // lq decomposition
        if (matrix == null)
        {
            throw new ArgumentNullException(nameof(matrix), $"{nameof(matrix)} can't be null");
        }

        if (matrix.Length == 0)
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

        var aMatrix = matrix.Transpose();
        aMatrix.Decompose(out double[,] q1Matrix, out double[,] rMatrix);
        lMatrix = rMatrix.Transpose();
        qMatrix = q1Matrix.Transpose();
    }

    public static void Decompose(this double[,] matrix, out double[,] qMatrix, out double[,] rMatrix)
    {
        // Householder method qr algo
        if (matrix == null)
        {
            throw new ArgumentNullException(nameof(matrix), $"{nameof(matrix)} can't be null");
        }

        if (matrix.Length == 0)
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

        var qTemp = CreateIdentityMatrix(rows);
        var rTemp = matrix.Duplicate();

        for (int i = 0; i < rows - 1; i++)
        {
            var hMatrix = CreateIdentityMatrix(rows);
            var aVector = new double[rows - i];
            int k = 0;
            for (int j = i; j < rows; j++)
            {
                aVector[k++] = rTemp[j, i];
            }

            double aNorm = aVector.Norm();
            if (aVector[0] < 0.0)
            { 
                aNorm = -aNorm; 
            }

            var vector = new double[aVector.Length];
            for (int j = 0; j < vector.Length; j++)
            {
                vector[j] = aVector[j] / (aVector[0] + aNorm);
                vector[0] = 1.0;
            }

            var hMatrix2 = CreateIdentityMatrix<double>(aVector.Length);
            double vvDot = vector.DotProduct(vector);
            var alpha = vector.Reshape(vector.Length, 1);
            var beta = vector.Reshape(1, vector.Length);
            var aMultB = alpha.DotProduct(beta);

            for (int i2 = 0; i2 < hMatrix2.Length; i2++)
            {
                for (int j2 = 0; j2 < hMatrix2.GetRow(0).Length; j2++)
                {
                    hMatrix2[i2, j2] -= 2.0 / vvDot * aMultB[i2, j2];
                }
            }

            int d = rows - hMatrix2.Length;
            for (int i2 = 0; i2 < hMatrix2.Length; i2++)
            {
                for (int j2 = 0; j2 < hMatrix2.GetRow(0).Length; j2++)
                {
                    hMatrix[i2 + d, j2 + d] = hMatrix2[i2, j2];
                }
            }

            qTemp = qTemp.DotProduct(hMatrix);
            rTemp = hMatrix.DotProduct(rTemp);
        }

        qMatrix = qTemp;
        rMatrix = rTemp;
    }

    public static void Decompose(this double[,] matrix, out double[,] uMatrix, out double[,] vhMatrix, out double[] sVector)
    {
        // svd decomposition using jacobi algo
        var aMatrix = matrix.Duplicate();
        int rows = aMatrix.GetColumn(0).Length, columns = aMatrix.GetRow(0).Length, ct = 1, pass = 0, passMax = Math.Min(5 * columns, 15);
        var qMatrix = CreateIdentityMatrix<double>(columns);
        var tVector = new double[columns];
        var tol = 10 * columns * double.Epsilon;

        // save the column error estimates
        for (int j = 0; j < columns; ++j)
        {
            var cj = aMatrix.GetColumn(j);
            var sj = cj.Norm();
            tVector[j] = double.Epsilon * sj;
        }

        while (ct > 0 && pass <= passMax)
        {
            ct = columns * (columns - 1) / 2;
            for (int j = 0; j < columns - 1; ++j)
            {
                for (int k = j + 1; k < columns; ++k)
                {
                    var cj = aMatrix.GetColumn(j);
                    var ck = aMatrix.GetColumn(k);
                    double p = 2.0 * cj.DotProduct(ck), a = cj.Norm(), sin, cos, b = ck.Norm(), q = a * a - b * b, 
                        v = Hypotenuse(p, q), errorA = tVector[j], errorB = tVector[k];
                    bool sorted = a >= b, orthog = Math.Abs(p) <= tol * (a * b), badA = a < errorA, badB = b < errorB;

                    if (sorted && (orthog || badA || badB))
                    {
                        --ct;
                        continue;
                    }

                    // compute rotation angles
                    if (v == 0 || sorted == false)
                    {
                        cos = 0.0;
                        sin = 1.0;
                    }
                    else
                    {
                        cos = Math.Sqrt((v + q) / (2.0 * v));
                        sin = p / (2.0 * v * cos);
                    }

                    // apply rotation to A (U)
                    for (int i = 0; i < rows; ++i)
                    {
                        double Aik = aMatrix[i, k];
                        double Aij = aMatrix[i, j];
                        aMatrix[i, j] = Aij * cos + Aik * sin;
                        aMatrix[i, k] = -Aij * sin + Aik * cos;
                    }

                    // update singular values
                    tVector[j] = Math.Abs(cos) * errorA + Math.Abs(sin) * errorB;
                    tVector[k] = Math.Abs(sin) * errorA + Math.Abs(cos) * errorB;

                    // apply rotation to Q (V)
                    for (int i = 0; i < columns; ++i)
                    {
                        double Qij = qMatrix[i, j];
                        double Qik = qMatrix[i, k];
                        qMatrix[i, j] = Qij * cos + Qik * sin;
                        qMatrix[i, k] = -Qij * sin + Qik * cos;
                    }
                }
            }

            pass++;
        }

        //  compute singular values
        double prevNorm = -1.0;
        for (int j = 0; j < columns; ++j)
        {
            var column = aMatrix.GetColumn(j);
            double norm = column.Norm();

            // check if singular value is zero
            if (norm == 0.0 || prevNorm == 0.0 || (j > 0 && norm <= tol * prevNorm))
            {
                tVector[j] = 0.0;
                for (int i = 0; i < rows; i++)
                {
                    aMatrix[i, j] = 0.0;
                }
                prevNorm = 0.0;
            }
            else
            {
                tVector[j] = norm;
                for (int i = 0; i < rows; i++)
                {
                    aMatrix[i, j] *= 1.0 / norm;
                }
                prevNorm = norm;
            }
        }

        if (ct > 0)
        {
            Console.WriteLine("Jacobi iterations did not converge");
        }

        uMatrix = aMatrix;
        vhMatrix = qMatrix.Transpose();
        sVector = tVector;

        if (rows < columns)
        {
            uMatrix = uMatrix.Extract(matrix.GetColumn(0).Length, rows);
            vhMatrix = vhMatrix.Extract(rows, vhMatrix.GetRow(0).Length);
            sVector = sVector.Extract(rows);
        }
        else if (rows > columns)
        {
            throw new InvalidOperationException("The Jacobi algorithm for SVD can't decompose when your matrix contains more rows than columns");
        }
    }

    public static Matrix<T> Extract<T>(this Matrix<T> matrix, int rows, int columns)
    {
        var result = new Matrix<T>(rows, columns);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                result[i, j] = matrix[i, j];
            }
        }

        return result;
    }

    public int Nullity(double threshold)
    {
        var rows = _matrix.RowCount;
        var columns = _matrix.ColumnCount;
        var weightsVector = new Vector<double>(columns);
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

    public static Matrix<double> Nullspace(double threshold)
    {
        int rows = _matrix.RowCount, columns = _matrix.ColumnCount, nullIndex = 0;
        var weightsVector = new Vector<double>(columns);
        var thresh = threshold >= 0 ? threshold : 0.5 * Math.Sqrt(rows + columns + 1) * weightsVector[0] * double.Epsilon;
        var nullspaceMatrix = new Matrix<double>(columns, Nullity(thresh));
        var vMatrix = new Matrix<double>(columns, columns);

        for (int i = 0; i < columns; i++)
        {
            if (weightsVector[i] <= thresh)
            {
                for (int j = 0; j < columns; j++)
                {
                    nullspaceMatrix[j][nullIndex] = vMatrix[j][i];
                }
                nullIndex++;
            }
        }

        return nullspaceMatrix;
    }

    public static Matrix<double> Range(double threshold)
    {
        int rows = _matrix.RowCount, columns = _matrix.ColumnCount, rank = 0;
        var weightsVector = new Vector<double>(columns);
        var rangeMatrix = new Matrix<double>(rows, Rank(threshold));
        var uMatrix = _matrix.Duplicate();

        for (int i = 0; i < columns; i++)
        {
            if (weightsVector[i] > threshold)
            {
                for (int j = 0; j < rows; j++)
                {
                    rangeMatrix[j][rank] = uMatrix[j][i];
                }
                rank++;
            }
        }

        return rangeMatrix;
    }

    public static int Rank(double threshold = -1)
    {
        var rows = _matrix.RowCount;
        var columns = _matrix.ColumnCount;
        var weightsVector = new Vector<double>(columns);
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
        var result = new Vector<T>(length);
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

        var rows = matrix.RowCount;
        var columns = matrix.ColumnCount;
        if (rows == 0)
        {
            throw new ArgumentException($"{nameof(matrix)} has to contain at least one row of values", nameof(matrix));
        }

        if (columns == 0)
        {
            throw new ArgumentException($"{nameof(matrix)} has to contain at least one column of values", nameof(matrix));
        }

        var newMatrix = new Matrix<T>(columns, rows);
        for (int i = 0; i < columns; i++)
        {
            for (int j = 0; j < rows; j++)
            {
                newMatrix[i, j] = matrix[j, i];
            }
        }

        return newMatrix;
    }

    public static void TridiagonalSolve(LinearAlgebra.Vector<double> vector1, LinearAlgebra.Vector<double> vector2, LinearAlgebra.Vector<double> vector3, 
        LinearAlgebra.Vector<double> solutionVector, LinearAlgebra.Vector<double> actualVector)
    {
        var size = vector1.Count;
        double bet = 0;
        var gamVector = new LinearAlgebra.Vector<double>(size);

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
        var size = matrix.RowCount;

        for (int i = 0; i < size; i++)
        {
            var k = i - leftSide;
            var temp = Math.Min(leftSide + rightSide + 1, size - k);
            solutionVector[i] = 0;
            for (int j = Math.Max(0, -k); j < temp; j++)
            {
                solutionVector[i] += matrix[i][j] * actualVector[j + k];
            }
        }
    }

    public static Matrix<T> CreateIdentityMatrix<T>(int size)
    {
        if (size <= 1)
        {
            throw new ArgumentException($"{nameof(size)} has to be a minimum of 2", nameof(size));
        }

        var identityMatrix = new Matrix<T>(size, size);
        for (int i = 0; i < size; i++)
        {
            identityMatrix[i, i] = (T)Convert.ChangeType(1, typeof(T));;
        }

        return identityMatrix;
    }

    public static Matrix<double> DotProduct(this Matrix<double> matrixA, Matrix<double> matrixB)
    {
        if (matrixA == null)
        {
            throw new ArgumentNullException(nameof(matrixA), $"{nameof(matrixA)} can't be null");
        }

        if (matrixB == null)
        {
            throw new ArgumentNullException(nameof(matrixB), $"{nameof(matrixB)} can't be null");
        }

        if (matrixA.RowCount == 0)
        {
            throw new ArgumentException($"{nameof(matrixA)} has to contain at least one row of values", nameof(matrixA));
        }

        if (matrixB.RowCount == 0)
        {
            throw new ArgumentException($"{nameof(matrixB)} has to contain at least one row of values", nameof(matrixB));
        }

        var bRows = matrixB.RowCount;
        var bColumns = matrixB.ColumnCount;
        var aColumns = matrixA.ColumnCount;
        if (aColumns != bRows)
        {
            throw new ArgumentException($"The columns in {nameof(matrixA)} has to contain the same amount of rows in {nameof(matrixB)}");
        }

        var aRows = matrixA.RowCount;
        var matrix = new Matrix<double>(aRows, aColumns);
        for (int h = 0; h < aRows; h++)
        {
            for (int i = 0; i < bColumns; i++)
            {
                for (int j = 0; j < aColumns; j++)
                {
                    matrix[h, i] += matrixA[h, j] * matrixB[j, i];
                }
            }
        }

        return matrix;
    }

    public static Matrix<Complex> DotProduct(this Matrix<Complex> matrixA, Matrix<Complex> matrixB)
    {
        if (matrixA == null)
        {
            throw new ArgumentNullException(nameof(matrixA), $"{nameof(matrixA)} can't be null");
        }

        if (matrixB == null)
        {
            throw new ArgumentNullException(nameof(matrixB), $"{nameof(matrixB)} can't be null");
        }

        if (matrixA.RowCount == 0)
        {
            throw new ArgumentException($"{nameof(matrixA)} has to contain at least one row of values", nameof(matrixA));
        }

        if (matrixB.RowCount == 0)
        {
            throw new ArgumentException($"{nameof(matrixB)} has to contain at least one row of values", nameof(matrixB));
        }

        var bRows = matrixB.RowCount;
        var bColumns = matrixB.ColumnCount;
        var aColumns = matrixA.ColumnCount;
        if (aColumns != bRows)
        {
            throw new ArgumentException($"The columns in {nameof(matrixA)} has to contain the same amount of rows in {nameof(matrixB)}");
        }

        var aRows = matrixA.RowCount;
        var matrix = new Matrix<Complex>(aRows, aColumns);
        for (int h = 0; h < aRows; h++)
        {
            for (int i = 0; i < bColumns; i++)
            {
                for (int j = 0; j < aColumns; j++)
                {
                    matrix[h, i] += matrixA[h, j] * matrixB[j, i];
                }
            }
        }

        return matrix;
    }
}