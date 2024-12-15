namespace AiDotNet.Interfaces;

public interface IMatrixDecomposition<T>
{
    Matrix<T> A { get; }

    Vector<T> Solve(Vector<T> b);
    Matrix<T> Invert();
}