namespace AiDotNet.LinearAlgebra;

public class NormalDecomposition<T> : IMatrixDecomposition<T>
{
    public Matrix<T> A { get; private set; }
    private Matrix<T> ATA { get; set; }
    private CholeskyDecomposition<T> choleskyDecomposition;

    public NormalDecomposition(Matrix<T> matrix)
    {
        A = matrix;
        ATA = A.Transpose().Multiply(A);
        choleskyDecomposition = new CholeskyDecomposition<T>(ATA);
    }

    public Vector<T> Solve(Vector<T> b)
    {
        var ATb = A.Transpose().Multiply(b);
        return choleskyDecomposition.Solve(ATb);
    }

    public Matrix<T> Invert()
    {
        return choleskyDecomposition.Invert();
    }
}