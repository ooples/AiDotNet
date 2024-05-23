namespace AiDotNet.Interfaces;

public interface ILuDecomposition<T> : IMatrixDecomposition<T>
{
    public void Decompose(out LinearAlgebra.Matrix<T> matrix, out LinearAlgebra.Vector<int> indexVector);

    public void Decompose(out LinearAlgebra.Matrix<T> upperTriangularMatrix, out LinearAlgebra.Matrix<T> lowerTriangularMatrix);
}