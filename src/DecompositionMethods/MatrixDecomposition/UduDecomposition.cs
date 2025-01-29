using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Enums.AlgorithmTypes;

namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

public class UduDecomposition<T> : IMatrixDecomposition<T>
{
    private readonly INumericOperations<T> NumOps;

    public Matrix<T> A { get; }
    public Matrix<T> U { get; private set; }
    public Vector<T> D { get; private set; }

    public UduDecomposition(Matrix<T> matrix, UduAlgorithmType algorithm = UduAlgorithmType.Crout)
    {
        if (!matrix.IsSquareMatrix())
            throw new ArgumentException("Matrix must be square for UDU decomposition.");
        A = matrix;
        var n = A.Rows;
        U = new Matrix<T>(n, n);
        D = new Vector<T>(n);
        NumOps = MathHelper.GetNumericOperations<T>();

        Decompose(algorithm);
    }

    public void Decompose(UduAlgorithmType algorithm = UduAlgorithmType.Crout)
    {
        switch (algorithm)
        {
            case UduAlgorithmType.Crout:
                DecomposeCrout();
                break;
            case UduAlgorithmType.Doolittle:
                DecomposeDoolittle();
                break;
            default:
                throw new ArgumentException("Unsupported UDU decomposition algorithm.");
        }
    }

    private void DecomposeCrout()
    {
        int n = A.Rows;
        U = new Matrix<T>(n, n, NumOps);
        D = new Vector<T>(n, NumOps);

        for (int j = 0; j < n; j++)
        {
            T sum = NumOps.Zero;
            for (int k = 0; k < j; k++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(NumOps.Multiply(U[k, j], U[k, j]), D[k]));
            }
            D[j] = NumOps.Subtract(A[j, j], sum);

            U[j, j] = NumOps.One;

            for (int i = j + 1; i < n; i++)
            {
                sum = NumOps.Zero;
                for (int k = 0; k < j; k++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(NumOps.Multiply(U[k, i], U[k, j]), D[k]));
                }
                U[j, i] = NumOps.Divide(NumOps.Subtract(A[j, i], sum), D[j]);
            }
        }
    }

    private void DecomposeDoolittle()
    {
        int n = A.Rows;
        U = new Matrix<T>(n, n, NumOps);
        D = new Vector<T>(n, NumOps);

        for (int i = 0; i < n; i++)
        {
            T sum = NumOps.Zero;
            for (int k = 0; k < i; k++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(NumOps.Multiply(U[k, i], U[k, i]), D[k]));
            }
            D[i] = NumOps.Subtract(A[i, i], sum);

            U[i, i] = NumOps.One;

            for (int j = i + 1; j < n; j++)
            {
                sum = NumOps.Zero;
                for (int k = 0; k < i; k++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(NumOps.Multiply(U[k, i], U[k, j]), D[k]));
                }
                U[i, j] = NumOps.Divide(NumOps.Subtract(A[i, j], sum), D[i]);
            }
        }
    }

    public Vector<T> Solve(Vector<T> b)
    {
        if (b.Length != A.Rows)
            throw new ArgumentException("Vector b must have the same length as the number of rows in matrix A.");

        // Forward substitution
        Vector<T> y = new Vector<T>(b.Length, NumOps);
        for (int i = 0; i < b.Length; i++)
        {
            T sum = NumOps.Zero;
            for (int j = 0; j < i; j++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(U[j, i], y[j]));
            }
            y[i] = NumOps.Subtract(b[i], sum);
        }

        // Diagonal scaling
        for (int i = 0; i < b.Length; i++)
        {
            y[i] = NumOps.Divide(y[i], D[i]);
        }

        // Backward substitution
        Vector<T> x = new Vector<T>(b.Length, NumOps);
        for (int i = b.Length - 1; i >= 0; i--)
        {
            T sum = NumOps.Zero;
            for (int j = i + 1; j < b.Length; j++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(U[i, j], x[j]));
            }
            x[i] = NumOps.Subtract(y[i], sum);
        }

        return x;
    }

    public Matrix<T> Invert()
    {
        int n = A.Rows;
        Matrix<T> inverse = new Matrix<T>(n, n, NumOps);

        for (int i = 0; i < n; i++)
        {
            Vector<T> ei = new Vector<T>(n, NumOps);
            ei[i] = NumOps.One;
            Vector<T> column = Solve(ei);
            for (int j = 0; j < n; j++)
            {
                inverse[j, i] = column[j];
            }
        }

        return inverse;
    }

    public (Matrix<T> U, Vector<T> D) GetFactors()
    {
        return (U, D);
    }
}