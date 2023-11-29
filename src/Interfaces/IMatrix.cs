namespace AiDotNet.Interfaces;

public interface IMatrix<T, T2>
{
    public T[] Solve(T[,] matrix1, T[,] matrix2, T[,] matrix3);

    public (T[,] mainMatrix, T[,] subMatrix, T[,] yTerms) BuildMatrix(T[] inputs, T[] outputs, T2 order);
}