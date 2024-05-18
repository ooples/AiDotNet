namespace AiDotNet.Interfaces;

public interface ISvdDecomposition<T> : IMatrixDecomposition<T>
{
    public void Decompose(out LinearAlgebra.Matrix<T> matrix, out LinearAlgebra.Vector<int> indexVector);

    public int Rank(T threshold);

    public int Nullity(T threshold);

    public LinearAlgebra.Matrix<T> Range(T threshold);

    public LinearAlgebra.Matrix<T> Nullspace(T threshold);
}