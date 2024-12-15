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

    public static Vector<T> GetSubColumn<T>(this Matrix<T> matrix, int columnIndex, int startRow, int length)
    {
        if (columnIndex < 0 || columnIndex >= matrix.Columns)
            throw new ArgumentOutOfRangeException(nameof(columnIndex));
        if (startRow < 0 || startRow >= matrix.Rows)
            throw new ArgumentOutOfRangeException(nameof(startRow));
        if (length < 0 || startRow + length > matrix.Rows)
            throw new ArgumentOutOfRangeException(nameof(length));

        var result = new Vector<T>(length, MathHelper.GetNumericOperations<T>());
        for (int i = 0; i < length; i++)
        {
            result[i] = matrix[startRow + i, columnIndex];
        }

        return result;
    }
}