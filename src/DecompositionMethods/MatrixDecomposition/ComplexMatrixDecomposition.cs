namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

public class ComplexMatrixDecomposition<T> : IMatrixDecomposition<Complex<T>>
{
    private readonly IMatrixDecomposition<T> _baseDecomposition;
    private readonly INumericOperations<T> _ops;

    public ComplexMatrixDecomposition(IMatrixDecomposition<T> baseDecomposition)
    {
        _baseDecomposition = baseDecomposition;
        _ops = MathHelper.GetNumericOperations<T>();
    }

    public Matrix<Complex<T>> A
    {
        get
        {
            var baseA = _baseDecomposition.A;
            var complexA = new Matrix<Complex<T>>(baseA.Rows, baseA.Columns);

            for (int i = 0; i < baseA.Rows; i++)
            {
                for (int j = 0; j < baseA.Columns; j++)
                {
                    complexA[i, j] = new Complex<T>(baseA[i, j], _ops.Zero);
                }
            }

            return complexA;
        }
    }

    public Matrix<Complex<T>> Invert()
    {
        var baseInverse = _baseDecomposition.Invert();
        var complexInverse = new Matrix<Complex<T>>(baseInverse.Rows, baseInverse.Columns);

        for (int i = 0; i < baseInverse.Rows; i++)
        {
            for (int j = 0; j < baseInverse.Columns; j++)
            {
                complexInverse[i, j] = new Complex<T>(baseInverse[i, j], _ops.Zero);
            }
        }

        return complexInverse;
    }

    public Vector<Complex<T>> Solve(Vector<Complex<T>> b)
    {
        // Extract real parts of b
        var realB = new Vector<T>(b.Length);
        for (int i = 0; i < b.Length; i++)
        {
            realB[i] = b[i].Real;
        }

        // Solve using base decomposition
        var realSolution = _baseDecomposition.Solve(realB);

        // Convert solution to complex
        var complexSolution = new Vector<Complex<T>>(realSolution.Length);
        for (int i = 0; i < realSolution.Length; i++)
        {
            complexSolution[i] = new Complex<T>(realSolution[i], _ops.Zero);
        }

        return complexSolution;
    }
}