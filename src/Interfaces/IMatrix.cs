namespace AiDotNet.Interfaces;

public interface IMatrix<T>
{
    public LinearAlgebra.Matrix<T> GetInverse();

    public T GetDeterminant();
}