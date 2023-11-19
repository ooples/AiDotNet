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
}