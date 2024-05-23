namespace AiDotNet.LinearAlgebra;

public class Matrix<T> : MatrixBase<T>
{
    public Matrix(IEnumerable<IEnumerable<T>> values) : base(values)
    {
    }

    public Matrix(int rows, int columns) : base(rows, columns)
    {
    }
}