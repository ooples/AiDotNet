namespace AiDotNet.Interfaces;

public interface IMatrixDecomposition<T> : IMatrix<T>
{
    public void Solve(LinearAlgebra.Vector<T> vector, out LinearAlgebra.Vector<T> solutionVector);

    public void Solve(LinearAlgebra.Matrix<T> vector, out LinearAlgebra.Matrix<T> solutionMatrix);
}