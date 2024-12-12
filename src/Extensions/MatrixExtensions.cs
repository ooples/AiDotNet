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
}