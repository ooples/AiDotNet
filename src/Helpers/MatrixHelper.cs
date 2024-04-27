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

    public static T[] GetColumn<T>(this T[,] matrix, int columnNumber)
    {
        return Enumerable.Range(0, matrix.GetLength(0))
                .Select(x => matrix[x, columnNumber])
                .ToArray();
    }

    public static T[] GetRow<T>(this T[,] matrix, int rowNumber)
    {
        return Enumerable.Range(0, matrix.GetLength(1))
                .Select(x => matrix[rowNumber, x])
                .ToArray();
    }

    public static double[,] Reshape(this double[] array, int rows, int columns)
    {
        if (array == null)
        {
            throw new ArgumentNullException(nameof(array), $"{nameof(array)} can't be null");
        }

        if (rows < 0)
        {
            throw new ArgumentException($"{nameof(rows)} needs to be an integer larger than 0", nameof(rows));
        }

        if (columns < 0)
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

    public static T[,] Transpose<T>(this T[,] originalMatrix)
    {
        if (originalMatrix == null)
        {
            throw new ArgumentNullException(nameof(originalMatrix), $"{nameof(originalMatrix)} can't be null");
        }

        if (originalMatrix.Length == 0)
        {
            throw new ArgumentException($"{nameof(originalMatrix)} has to contain at least one row of values", nameof(originalMatrix));
        }

        var rows = originalMatrix.GetColumn(0).Length;
        var columns = originalMatrix.GetRow(0).Length;
        if (rows == 0)
        {
            throw new ArgumentException($"{nameof(originalMatrix)} has to contain at least one row of values", nameof(originalMatrix));
        }

        if (columns == 0)
        {
            throw new ArgumentException($"{nameof(originalMatrix)} has to contain at least one column of values", nameof(originalMatrix));
        }

        var newMatrix = new T[columns, rows];
        for (int i = 0; i < columns; i++)
        {
            for (int j = 0; j < rows; j++)
            {
                newMatrix[i, j] = originalMatrix[j, i];
            }
        }

        return newMatrix;
    }

    public static double[,] CreateIdentityMatrix(int size)
    {
        if (size <= 1)
        {
            throw new ArgumentException($"{nameof(size)} has to be a minimum of 2", nameof(size));
        }

        double[,] identityMatrix = new double[size, size];
        for (int i = 0; i < size; i++)
        {
            identityMatrix[i, i] = 1;
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

        var bColumns = matrixB.GetColumn(0).Length;
        var aRows = matrixA.GetRow(0).Length;
        if (aRows != bColumns)
        {
            throw new ArgumentException($"The columns in {nameof(matrixA)} has to contain the same amount of rows in {nameof(matrixB)}");
        }

        var aColumns = matrixA.GetColumn(0).Length;
        var matrix = new double[aColumns, aRows];
        for (int x = 0; x < aColumns; x++)
        {
            for (int i = 0; i < aRows; i++)
            {
                double matrixSum = 0;
                for (int j = 0; j < bColumns; j++)
                {
                    matrixSum += matrixA[x, j] * matrixB[j, i];
                }

                matrix[x, i] = matrixSum;
            }
        }

        return matrix;
    }
}