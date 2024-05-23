namespace AiDotNet.Interfaces;

public interface IMatrix<T>
{
    public Matrix<T> GetInverse();

    public T GetDeterminant();
}