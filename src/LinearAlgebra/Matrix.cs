
namespace AiDotNet.LinearAlgebra;

public class Matrix<T> : MatrixBase<T>
{
    public Matrix(IEnumerable<Vector<T>> values) : base(values)
    {
    }


}
