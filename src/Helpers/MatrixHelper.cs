global using Matrix = AiDotNet.LinearAlgebra.Matrix<double>;

namespace AiDotNet.Helpers;

internal static class MatrixHelper
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

    public static T[] GetColumn<T>(this T[,] matrix, int columnIndex)
    {
        return Enumerable.Range(0, matrix.GetLength(0))
                .Select(x => matrix[x, columnIndex])
                .ToArray();
    }

    public static T[] GetRow<T>(this T[,] matrix, int rowIndex)
    {
        return Enumerable.Range(0, matrix.GetLength(1))
                .Select(x => matrix[rowIndex, x])
                .ToArray();
    }

    public static double[,] Reshape(this double[] array, int rows, int columns)
    {
        if (array == null)
        {
            throw new ArgumentNullException(nameof(array), $"{nameof(array)} can't be null");
        }

        if (rows <= 0)
        {
            throw new ArgumentException($"{nameof(rows)} needs to be an integer larger than 0", nameof(rows));
        }

        if (columns <= 0)
        {
            throw new ArgumentException($"{nameof(columns)} needs to be an integer larger than 0", nameof(columns));
        }

        var length = array.Length;
        if (rows * columns != length)
        {
            throw new ArgumentException($"{nameof(rows)} and {nameof(columns)} multiplied together needs to be equal the array length");
        }

        var matrix = new double[rows, columns];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                matrix[i, j] = array[i * columns + j];
            }
        }

        return matrix;
    }

    public static LinearAlgebra.Matrix<T> Duplicate<T>(this LinearAlgebra.Matrix<T> matrix)
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

    public static double Norm(this double[] vector)
    {
        double sum = 0.0;
        int n = vector.Length;
        for (int i = 0; i < n; i++)
        {
            sum += Math.Pow(vector[i], 2);
        }

        return Math.Sqrt(sum);
    }

    public static double DotProduct(this double[] vectorA, double[] vectorB)
    {
        double result = 0.0;
        int n = vectorA.Length;
        for (int i = 0; i < n; i++)
        {
            result += vectorA[i] * vectorB[i];
        }
          
        return result;
    }

    public static double[,] Invert(this double[,] matrix)
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

    public static double[,] InvertSvd(this double[,] matrix)
    {
        // svd inversion
        matrix.Decompose(out double[,] uMatrix, out double[,] vhMatrix, out double[] sVector);
        var sInverted = sVector.Invert();
        var vMatrix = vhMatrix.Transpose();
        var uTransposed = uMatrix.Transpose();

        return vMatrix.DotProduct(sInverted.DotProduct(uTransposed));
    }

    public static bool IsUpperTriangularMatrix(this double[,] matrix)
    {
        for (int i = 1; i < matrix.GetColumn(0).Length; i++)
        {
            for (int j = 0; j < i; j++)
            {
                // If any element below the main diagonal is non-zero, the matrix is not upper triangular.
                if (matrix[i, j] != 0)
                {
                    return false;
                }
            }
        }

        return true;
    }

    public static bool IsLowerTriangularMatrix(this double[,] matrix)
    {
        var rows = matrix.GetColumn(0).Length;
        for (int i = 0; i < rows - 1; i++)
        {
            for (int j = i + 1; j < rows; j++)
            {
                // If any element above the main diagonal is non-zero, the matrix is not lower triangular.
                if (matrix[i, j] != 0)
                {
                    return false;
                }
            }
        }

        return true;
    }

    public static double[,] InvertUpperTriangularMatrix(this double[,] matrix)
    {
        int n = matrix.GetLength(0);

        // Create the inverse matrix
        double[,] inverse = new double[n, n];

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

    public static void ()

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

    public static double[,] Extract(this double[,] matrix, int rows, int columns)
    {
        double[,] result = new double[rows, columns];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                result[i, j] = matrix[i, j];
            }
        }

        return result;
    }

    public static double[] Extract(this double[] vector, int length)
    {
        var result = new double[length];
        for (int i = 0; i < length; i++)
        {
            result[i] = vector[i];
        }

        return result;
    }

    public static T[,] Transpose<T>(this T[,] matrix)
    {
        if (matrix == null)
        {
            throw new ArgumentNullException(nameof(matrix), $"{nameof(matrix)} can't be null");
        }

        if (matrix.Length == 0)
        {
            throw new ArgumentException($"{nameof(matrix)} has to contain at least one row of values", nameof(matrix));
        }

        var rows = matrix.GetColumn(0).Length;
        var columns = matrix.GetRow(0).Length;
        if (rows == 0)
        {
            throw new ArgumentException($"{nameof(matrix)} has to contain at least one row of values", nameof(matrix));
        }

        if (columns == 0)
        {
            throw new ArgumentException($"{nameof(matrix)} has to contain at least one column of values", nameof(matrix));
        }

        var newMatrix = new T[columns, rows];
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

    public static void BandDiagonalMultiply(int leftSide, int rightSide, LinearAlgebra.Matrix<double> matrix, LinearAlgebra.Vector<double> solutionVector, 
        LinearAlgebra.Vector<double> actualVector)
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

    public static LinearAlgebra.Matrix<T> CreateIdentityMatrix<T>(int size)
    {
        if (size <= 1)
        {
            throw new ArgumentException($"{nameof(size)} has to be a minimum of 2", nameof(size));
        }

        var identityMatrix = new LinearAlgebra.Matrix<T>(size, size);
        for (int i = 0; i < size; i++)
        {
            identityMatrix[i, i] = (T)Convert.ChangeType(1, typeof(T));;
        }

        return identityMatrix;
    }

    public static double[,] DotProduct(this double[,] matrixA, double[,] matrixB)
    {
        if (matrixA == null)
        {
            throw new ArgumentNullException(nameof(matrixA), $"{nameof(matrixA)} can't be null");
        }

        if (matrixB == null)
        {
            throw new ArgumentNullException(nameof(matrixB), $"{nameof(matrixB)} can't be null");
        }

        if (matrixA.Length == 0)
        {
            throw new ArgumentException($"{nameof(matrixA)} has to contain at least one row of values", nameof(matrixA));
        }

        if (matrixB.Length == 0)
        {
            throw new ArgumentException($"{nameof(matrixB)} has to contain at least one row of values", nameof(matrixB));
        }

        var bRows = matrixB.GetColumn(0).Length;
        var bColumns = matrixB.GetRow(0).Length;
        var aColumns = matrixA.GetRow(0).Length;
        if (aColumns != bRows)
        {
            throw new ArgumentException($"The columns in {nameof(matrixA)} has to contain the same amount of rows in {nameof(matrixB)}");
        }

        var aRows = matrixA.GetColumn(0).Length;
        var matrix = new double[aRows, aColumns];
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