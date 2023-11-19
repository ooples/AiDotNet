namespace AiDotNet.Interfaces;

public interface IMatrix
{
    public T[,] Solve<T>(T[,] matrix, T[] vector);

    public T[,] Transpose<T>(T[,] matrix);

    public T[,] BuildMatrix<T>(T[] values);
}