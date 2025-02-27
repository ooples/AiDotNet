using AiDotNet.Enums.AlgorithmTypes;

namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

public class TakagiDecomposition<T> : IMatrixDecomposition<T>
{
    private readonly INumericOperations<T> NumOps;

    public Matrix<T> SigmaMatrix { get; private set; }
    public Matrix<Complex<T>> UnitaryMatrix { get; private set; }
    public Matrix<T> A { get; private set; }

    public TakagiDecomposition(Matrix<T> matrix, TakagiAlgorithmType algorithm = TakagiAlgorithmType.Jacobi)
    {
        NumOps = MathHelper.GetNumericOperations<T>();
        A = matrix;
        (SigmaMatrix, UnitaryMatrix) = Decompose(A, algorithm);
    }

    private (Matrix<T> S, Matrix<Complex<T>> U) Decompose(Matrix<T> matrix, TakagiAlgorithmType algorithm)
    {
        return algorithm switch
        {
            TakagiAlgorithmType.Jacobi => ComputeTakagiJacobi(matrix),
            TakagiAlgorithmType.QR => ComputeTakagiQR(matrix),
            TakagiAlgorithmType.EigenDecomposition => ComputeTakagiEigenDecomposition(matrix),
            TakagiAlgorithmType.PowerIteration => ComputeTakagiPowerIteration(matrix),
            TakagiAlgorithmType.LanczosIteration => ComputeTakagiLanczosIteration(matrix),
            _ => throw new ArgumentException("Unsupported Takagi decomposition algorithm.")
        };
    }

    private (Matrix<T> S, Matrix<Complex<T>> U) ComputeTakagiDefault(Matrix<T> matrix)
    {
        var eigenDecomposition = new EigenDecomposition<T>(matrix);
        var eigenValues = eigenDecomposition.EigenValues;
        var eigenVectors = eigenDecomposition.EigenVectors;

        var rows = matrix.Rows;
        var S = new Matrix<T>(rows, rows);
        var U = new Matrix<Complex<T>>(rows, rows);

        for (int i = 0; i < rows; i++)
        {
            S[i, i] = NumOps.Sqrt(NumOps.Abs(eigenValues[i]));
            for (int j = 0; j < rows; j++)
            {
                U[i, j] = new Complex<T>(eigenVectors[i, j], NumOps.Zero);
            }
        }

        return (S, U);
    }

    private (Matrix<T> S, Matrix<Complex<T>> U) ComputeTakagiJacobi(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        var S = new Matrix<T>(n, n);
        var U = Matrix<Complex<T>>.CreateIdentity(n);
        var A = matrix.Copy();

        const int maxIterations = 100;
        var tolerance = NumOps.FromDouble(1e-10);

        for (int iter = 0; iter < maxIterations; iter++)
        {
            var maxOffDiagonal = NumOps.Zero;
            int p = 0, q = 0;

            // Find the largest off-diagonal element
            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    var absValue = NumOps.Abs(A[i, j]);
                    if (NumOps.GreaterThan(absValue, maxOffDiagonal))
                    {
                        maxOffDiagonal = absValue;
                        p = i;
                        q = j;
                    }
                }
            }

            if (NumOps.LessThan(maxOffDiagonal, tolerance))
            {
                break;
            }

            // Compute the Jacobi rotation
            T app = A[p, p];
            T aqq = A[q, q];
            T apq = A[p, q];
            T theta = NumOps.Divide(NumOps.Subtract(app, aqq), NumOps.Multiply(NumOps.FromDouble(2), apq));
            T t = NumOps.Divide(NumOps.FromDouble(1), NumOps.Add(NumOps.Abs(theta), NumOps.Sqrt(NumOps.Add(NumOps.Square(theta), NumOps.One))));
            if (NumOps.LessThan(theta, NumOps.Zero))
            {
                t = NumOps.Negate(t);
            }
            T c = NumOps.Divide(NumOps.FromDouble(1), NumOps.Sqrt(NumOps.Add(NumOps.Square(t), NumOps.One)));
            T s = NumOps.Multiply(t, c);

            // Update A
            for (int i = 0; i < n; i++)
            {
                if (i != p && i != q)
                {
                    T api = A[p, i];
                    T aqi = A[q, i];
                    A[p, i] = NumOps.Add(NumOps.Multiply(c, api), NumOps.Multiply(s, aqi));
                    A[i, p] = A[p, i];
                    A[q, i] = NumOps.Subtract(NumOps.Multiply(c, aqi), NumOps.Multiply(s, api));
                    A[i, q] = A[q, i];
                }
            }
            A[p, p] = NumOps.Add(NumOps.Multiply(NumOps.Square(c), app), NumOps.Multiply(NumOps.Square(s), aqq));
            A[q, q] = NumOps.Add(NumOps.Multiply(NumOps.Square(s), app), NumOps.Multiply(NumOps.Square(c), aqq));
            A[p, q] = NumOps.Zero;
            A[q, p] = NumOps.Zero;

            // Update U
            for (int i = 0; i < n; i++)
            {
                Complex<T> uip = U[i, p];
                Complex<T> uiq = U[i, q];
                U[i, p] = new Complex<T>(NumOps.Add(NumOps.Multiply(c, uip.Real), NumOps.Multiply(s, uiq.Real)),
                                      NumOps.Add(NumOps.Multiply(c, uip.Imaginary), NumOps.Multiply(s, uiq.Imaginary)));
                U[i, q] = new Complex<T>(NumOps.Subtract(NumOps.Multiply(c, uiq.Real), NumOps.Multiply(s, uip.Real)),
                                      NumOps.Subtract(NumOps.Multiply(c, uiq.Imaginary), NumOps.Multiply(s, uip.Imaginary)));
            }
        }

        // Extract singular values
        for (int i = 0; i < n; i++)
        {
            S[i, i] = NumOps.Sqrt(NumOps.Abs(A[i, i]));
        }

        return (S, U);
    }

    private (Matrix<T> S, Matrix<Complex<T>> U) ComputeTakagiQR(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        var S = new Matrix<T>(n, n);
        var U = Matrix<Complex<T>>.CreateIdentity(n);
        var A = matrix.ToComplexMatrix();

        const int maxIterations = 100;
        var tolerance = NumOps.FromDouble(1e-10);

        for (int iter = 0; iter < maxIterations; iter++)
        {
            // Compute QR decomposition
            var qr = new QrDecomposition<Complex<T>>(A);
            var Q = qr.Q;
            var R = qr.R;

            // Update A
            A = R.Multiply(Q);

            // Update U
            U = U.Multiply(Q);

            // Check for convergence
            bool converged = true;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < i; j++)
                {
                    if (NumOps.GreaterThan(CalculateMagnitude(A[i, j]), tolerance))
                    {
                        converged = false;
                        break;
                    }
                }
                if (!converged) break;
            }

            if (converged) break;
        }

        // Extract singular values
        for (int i = 0; i < n; i++)
        {
            S[i, i] = NumOps.Sqrt(NumOps.Abs(CalculateMagnitude(A[i, i])));
        }

        return (S, U);
    }

    private T CalculateMagnitude(Complex<T> complex)
    {
        return NumOps.Sqrt(NumOps.Add(NumOps.Square(complex.Real), NumOps.Square(complex.Imaginary)));
    }

    private (Matrix<Complex<T>> Q, Matrix<T> R) QRDecomposition(Matrix<T> A)
    {
        int n = A.Rows;
        var Q = Matrix<Complex<T>>.CreateIdentity(n);
        var R = A.Copy();

        for (int j = 0; j < n - 1; j++)
        {
            for (int i = j + 1; i < n; i++)
            {
                T a = R[j, j];
                T b = R[i, j];
                T r = NumOps.Sqrt(NumOps.Add(NumOps.Square(a), NumOps.Square(b)));
                T c = NumOps.Divide(a, r);
                T s = NumOps.Divide(b, r);

                // Update R
                for (int k = j; k < n; k++)
                {
                    T temp = R[j, k];
                    R[j, k] = NumOps.Add(NumOps.Multiply(c, temp), NumOps.Multiply(s, R[i, k]));
                    R[i, k] = NumOps.Subtract(NumOps.Multiply(c, R[i, k]), NumOps.Multiply(s, temp));
                }

                // Update Q
                for (int k = 0; k < n; k++)
                {
                    Complex<T> temp = Q[k, j];
                    Q[k, j] = new Complex<T>(NumOps.Add(NumOps.Multiply(c, temp.Real), NumOps.Multiply(s, Q[k, i].Real)),
                                          NumOps.Add(NumOps.Multiply(c, temp.Imaginary), NumOps.Multiply(s, Q[k, i].Imaginary)));
                    Q[k, i] = new Complex<T>(NumOps.Subtract(NumOps.Multiply(c, Q[k, i].Real), NumOps.Multiply(s, temp.Real)),
                                          NumOps.Subtract(NumOps.Multiply(c, Q[k, i].Imaginary), NumOps.Multiply(s, temp.Imaginary)));
                }
            }
        }

        return (Q, R);
    }

    private (Matrix<T> S, Matrix<Complex<T>> U) ComputeTakagiEigenDecomposition(Matrix<T> matrix)
    {
        var eigenDecomposition = new EigenDecomposition<T>(matrix);
        var eigenValues = eigenDecomposition.EigenValues;
        var eigenVectors = eigenDecomposition.EigenVectors;

        var rows = matrix.Rows;
        var S = new Matrix<T>(rows, rows);
        var U = new Matrix<Complex<T>>(rows, rows);

        for (int i = 0; i < rows; i++)
        {
            S[i, i] = NumOps.Sqrt(NumOps.Abs(eigenValues[i]));
            for (int j = 0; j < rows; j++)
            {
                U[i, j] = new Complex<T>(eigenVectors[i, j], NumOps.Zero);
            }
        }

        return (S, U);
    }

    private (Matrix<T> S, Matrix<Complex<T>> U) ComputeTakagiPowerIteration(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        var S = new Matrix<T>(n, n);
        var U = new Matrix<Complex<T>>(n, n);

        for (int i = 0; i < n; i++)
        {
            var v = Vector<T>.CreateRandom(n);
            var lambda = NumOps.Zero;

            for (int iter = 0; iter < 100; iter++)
            {
                var w = matrix.Multiply(v);
                var newLambda = v.DotProduct(w);
                v = w.Divide(w.Norm());

                if (NumOps.LessThan(NumOps.Abs(NumOps.Subtract(newLambda, lambda)), NumOps.FromDouble(1e-10)))
                {
                    break;
                }
                lambda = newLambda;
            }

            S[i, i] = NumOps.Sqrt(NumOps.Abs(lambda));
            for (int j = 0; j < n; j++)
            {
                U[j, i] = new Complex<T>(v[j], NumOps.Zero);
            }

            // Deflate the matrix
            for (int j = 0; j < n; j++)
            {
                for (int k = 0; k < n; k++)
                {
                    matrix[j, k] = NumOps.Subtract(matrix[j, k], NumOps.Multiply(NumOps.Multiply(v[j], v[k]), lambda));
                }
            }
        }

        return (S, U);
    }

    private (Matrix<T> S, Matrix<Complex<T>> U) ComputeTakagiLanczosIteration(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        var S = new Matrix<T>(n, n);
        var U = new Matrix<Complex<T>>(n, n);

        var v = Vector<T>.CreateRandom(n);
        var V = new List<Vector<T>> { v.Divide(v.Norm()) };
        var alpha = new List<T>();
        var beta = new List<T>();

        for (int j = 0; j < n; j++)
        {
            var w = matrix.Multiply(V[j]);
            var alphaj = V[j].DotProduct(w);
            alpha.Add(alphaj);

            if (j < n - 1)
            {
                w = w.Subtract(V[j].Multiply(alphaj));
                if (j > 0)
                {
                    w = w.Subtract(V[j - 1].Multiply(beta[j - 1]));
                }

                var betaj = w.Norm();
                beta.Add(betaj);

                if (NumOps.LessThan(NumOps.Abs(betaj), NumOps.FromDouble(1e-10)))
                {
                    break;
                }

                V.Add(w.Divide(betaj));
            }
        }

        // Construct tridiagonal matrix
        var T = new Matrix<T>(n, n);
        for (int i = 0; i < alpha.Count; i++)
        {
            T[i, i] = alpha[i];
            if (i < beta.Count)
            {
                T[i, i + 1] = beta[i];
                T[i + 1, i] = beta[i];
            }
        }

        // Compute eigendecomposition of T
        var eigenDecomposition = new EigenDecomposition<T>(T);
        var eigenValues = eigenDecomposition.EigenValues;
        var eigenVectors = eigenDecomposition.EigenVectors;

        for (int i = 0; i < n; i++)
        {
            S[i, i] = NumOps.Sqrt(NumOps.Abs(eigenValues[i]));
            for (int j = 0; j < n; j++)
            {
                U[i, j] = new Complex<T>(NumOps.Zero, NumOps.Zero);
                for (int k = 0; k < V.Count; k++)
                {
                    Complex<T> term = new Complex<T>(NumOps.Multiply(V[k][i], eigenVectors[k, j]), NumOps.Zero);
                    U[i, j] = new Complex<T>(
                        NumOps.Add(U[i, j].Real, term.Real),
                        NumOps.Add(U[i, j].Imaginary, term.Imaginary)
                    );
                }
            }
        }

        return (S, U);
    }

    public Matrix<T> Invert()
    {
        var invSigma = SigmaMatrix.InvertDiagonalMatrix();
        var invU = UnitaryMatrix.InvertUnitaryMatrix();
        var invSigmaComplex = invSigma.ToComplexMatrix();
        var inv = invU.Multiply(invSigmaComplex).Multiply(invU.Transpose());

        return inv.ToRealMatrix();
    }

    public Vector<T> Solve(Vector<T> bVector)
    {
        var bComplex = new Vector<Complex<T>>(bVector.Length);
        for (int i = 0; i < bVector.Length; i++)
        {
            bComplex[i] = new Complex<T>(bVector[i], NumOps.Zero);
        }
        var yVector = UnitaryMatrix.ForwardSubstitution(bComplex);

        var result = SigmaMatrix.ToComplexMatrix().BackwardSubstitution(yVector);
        return result.ToRealVector();
    }
}