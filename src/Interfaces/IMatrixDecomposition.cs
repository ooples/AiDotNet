namespace AiDotNet.Interfaces;

public interface IMatrixDecomposition<T>
{
    // Decomposes the given matrix using the specific algorithm implemented by the class.
    void Decompose(Matrix<T> aMatrix);

    // Solves the system Ax = b for x, where A is the decomposed matrix and b is the right-hand side vector.
    Vector<T> Solve(Matrix<T> aMatrix, Vector<T> bVector);

    // Returns the inverse of the decomposed matrix and/or matrices.
    Matrix<T> Invert();
}