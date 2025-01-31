namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

public class CramerDecomposition<T> : IMatrixDecomposition<T>
{
    private readonly INumericOperations<T> NumOps;
    public Matrix<T> A { get; private set; }

    public CramerDecomposition(Matrix<T> matrix)
    {
        NumOps = MathHelper.GetNumericOperations<T>();
        A = matrix;

        if (A.Rows != A.Columns)
        {
            throw new ArgumentException("Cramer's rule requires a square matrix.");
        }
    }

    public Vector<T> Solve(Vector<T> b)
    {
        if (A.Rows != b.Length)
        {
            throw new ArgumentException("The number of rows in A must match the length of b.");
        }

        T detA = Determinant(A);
        if (NumOps.Equals(detA, NumOps.Zero))
        {
            throw new InvalidOperationException("The matrix is singular and cannot be solved using Cramer's rule.");
        }

        Vector<T> x = new(A.Columns);
        for (int i = 0; i < A.Columns; i++)
        {
            Matrix<T> Ai = ReplaceColumn(A, b, i);
            x[i] = NumOps.Divide(Determinant(Ai), detA);
        }

        return x;
    }

    public Matrix<T> Invert()
    {
        T detA = Determinant(A);
        if (NumOps.Equals(detA, NumOps.Zero))
        {
            throw new InvalidOperationException("The matrix is singular and cannot be inverted.");
        }

        Matrix<T> inverse = new(A.Rows, A.Columns);
        for (int i = 0; i < A.Rows; i++)
        {
            for (int j = 0; j < A.Columns; j++)
            {
                T cofactor = Cofactor(A, i, j);
                inverse[j, i] = NumOps.Divide(cofactor, detA);
            }
        }

        return inverse;
    }

    private T Determinant(Matrix<T> matrix)
    {
        if (matrix.Rows == 1)
        {
            return matrix[0, 0];
        }

        T det = NumOps.Zero;
        for (int j = 0; j < matrix.Columns; j++)
        {
            det = NumOps.Add(det, NumOps.Multiply(matrix[0, j], Cofactor(matrix, 0, j)));
        }

        return det;
    }

    private T Cofactor(Matrix<T> matrix, int row, int col)
    {
        Matrix<T> minor = new(matrix.Rows - 1, matrix.Columns - 1);
        int m = 0, n = 0;

        for (int i = 0; i < matrix.Rows; i++)
        {
            if (i == row) continue;
            n = 0;
            for (int j = 0; j < matrix.Columns; j++)
            {
                if (j == col) continue;
                minor[m, n] = matrix[i, j];
                n++;
            }
            m++;
        }

        T sign = (row + col) % 2 == 0 ? NumOps.One : NumOps.Negate(NumOps.One);
        return NumOps.Multiply(sign, Determinant(minor));
    }

    private Matrix<T> ReplaceColumn(Matrix<T> original, Vector<T> column, int colIndex)
    {
        Matrix<T> result = original.Copy();
        for (int i = 0; i < original.Rows; i++)
        {
            result[i, colIndex] = column[i];
        }

        return result;
    }
}