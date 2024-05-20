namespace AiDotNet.Interfaces;

public interface ISvdDecomposition<T> : IMatrixDecomposition<T>
{
    public void Decompose(out Matrix<T> matrix, out Vector<int> indexVector);

    public int Rank(T threshold);

    public int Nullity(T threshold);

    public Matrix<T> Range(T threshold);

    public Matrix<T> Nullspace(T threshold);
}