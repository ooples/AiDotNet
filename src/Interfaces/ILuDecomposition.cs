namespace AiDotNet.Interfaces;

public interface ILuDecomposition<T> : IMatrixDecomposition<T>
{
    public void Decompose(out Matrix<T> matrix, out Vector<int> indexVector);

    public void Decompose(out Matrix<T> upperTriangularMatrix, out Matrix<T> lowerTriangularMatrix);
}