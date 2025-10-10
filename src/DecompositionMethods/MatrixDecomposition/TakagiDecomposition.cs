namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

/// <summary>
/// Implements the Takagi factorization for complex symmetric matrices.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Takagi decomposition is a special matrix factorization that works on symmetric matrices.
/// It breaks down a matrix into simpler components that make calculations easier. Think of it like
/// factoring a number into its prime components, but for matrices.
/// </para>
/// </remarks>
public class TakagiDecomposition<T> : IMatrixDecomposition<T>
{
    private readonly INumericOperations<T> _numOps = default!;

    /// <summary>
    /// Gets the diagonal matrix containing the singular values.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Singular values represent the "strength" or "importance" of different dimensions in your data.
    /// Larger singular values indicate more important patterns in the data.
    /// </remarks>
    public Matrix<T> SigmaMatrix { get; private set; }

    /// <summary>
    /// Gets the unitary matrix used in the decomposition.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> A unitary matrix preserves lengths and angles when multiplied with vectors.
    /// It's like rotating or reflecting data without changing its fundamental structure.
    /// </remarks>
    public Matrix<Complex<T>> UnitaryMatrix { get; private set; }

    /// <summary>
    /// Gets the original matrix that was decomposed.
    /// </summary>
    public Matrix<T> A { get; private set; }

    /// <summary>
    /// Initializes a new instance of the TakagiDecomposition class.
    /// </summary>
    /// <param name="matrix">The matrix to decompose</param>
    /// <param name="algorithm">The algorithm to use for decomposition</param>
    /// <remarks>
    /// <b>For Beginners:</b> This constructor takes your input matrix and breaks it down using the specified algorithm.
    /// Different algorithms have different trade-offs in terms of speed and accuracy.
    /// </remarks>
    public TakagiDecomposition(Matrix<T> matrix, TakagiAlgorithmType algorithm = TakagiAlgorithmType.Jacobi)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        A = matrix;
        (SigmaMatrix, UnitaryMatrix) = Decompose(A, algorithm);
    }

    /// <summary>
    /// Decomposes the matrix using the specified algorithm.
    /// </summary>
    /// <param name="matrix">The matrix to decompose</param>
    /// <param name="algorithm">The algorithm to use</param>
    /// <returns>A tuple containing the singular values matrix and unitary matrix</returns>
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

    /// <summary>
    /// Computes the Takagi decomposition using a default approach based on eigendecomposition.
    /// </summary>
    /// <param name="matrix">The matrix to decompose</param>
    /// <returns>A tuple containing the singular values matrix and unitary matrix</returns>
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
            S[i, i] = _numOps.Sqrt(_numOps.Abs(eigenValues[i]));
            for (int j = 0; j < rows; j++)
            {
                U[i, j] = new Complex<T>(eigenVectors[i, j], _numOps.Zero);
            }
        }

        return (S, U);
    }

    /// <summary>
    /// Computes the Takagi decomposition using the Jacobi algorithm.
    /// </summary>
    /// <param name="matrix">The matrix to decompose</param>
    /// <returns>A tuple containing the singular values matrix and unitary matrix</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Jacobi algorithm works by gradually eliminating off-diagonal elements
    /// through a series of rotations. It's like solving a Rubik's cube by focusing on one piece at a time
    /// until the entire puzzle is solved.
    /// </para>
    /// </remarks>
    private (Matrix<T> S, Matrix<Complex<T>> U) ComputeTakagiJacobi(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        var S = new Matrix<T>(n, n);
        var U = Matrix<Complex<T>>.CreateIdentity(n);
        var A = matrix.Clone();

        const int maxIterations = 100;
        var tolerance = _numOps.FromDouble(1e-10);

        for (int iter = 0; iter < maxIterations; iter++)
        {
            var maxOffDiagonal = _numOps.Zero;
            int p = 0, q = 0;

            // Find the largest off-diagonal element
            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    var absValue = _numOps.Abs(A[i, j]);
                    if (_numOps.GreaterThan(absValue, maxOffDiagonal))
                    {
                        maxOffDiagonal = absValue;
                        p = i;
                        q = j;
                    }
                }
            }

            if (_numOps.LessThan(maxOffDiagonal, tolerance))
            {
                break;
            }

            // Compute the Jacobi rotation
            T app = A[p, p];
            T aqq = A[q, q];
            T apq = A[p, q];
            T theta = _numOps.Divide(_numOps.Subtract(app, aqq), _numOps.Multiply(_numOps.FromDouble(2), apq));
            T t = _numOps.Divide(_numOps.FromDouble(1), _numOps.Add(_numOps.Abs(theta), _numOps.Sqrt(_numOps.Add(_numOps.Square(theta), _numOps.One))));
            if (_numOps.LessThan(theta, _numOps.Zero))
            {
                t = _numOps.Negate(t);
            }
            T c = _numOps.Divide(_numOps.FromDouble(1), _numOps.Sqrt(_numOps.Add(_numOps.Square(t), _numOps.One)));
            T s = _numOps.Multiply(t, c);

            // Update A
            for (int i = 0; i < n; i++)
            {
                if (i != p && i != q)
                {
                    T api = A[p, i];
                    T aqi = A[q, i];
                    A[p, i] = _numOps.Add(_numOps.Multiply(c, api), _numOps.Multiply(s, aqi));
                    A[i, p] = A[p, i];
                    A[q, i] = _numOps.Subtract(_numOps.Multiply(c, aqi), _numOps.Multiply(s, api));
                    A[i, q] = A[q, i];
                }
            }
            A[p, p] = _numOps.Add(_numOps.Multiply(_numOps.Square(c), app), _numOps.Multiply(_numOps.Square(s), aqq));
            A[q, q] = _numOps.Add(_numOps.Multiply(_numOps.Square(s), app), _numOps.Multiply(_numOps.Square(c), aqq));
            A[p, q] = _numOps.Zero;
            A[q, p] = _numOps.Zero;

            // Update U
            for (int i = 0; i < n; i++)
            {
                Complex<T> uip = U[i, p];
                Complex<T> uiq = U[i, q];
                U[i, p] = new Complex<T>(_numOps.Add(_numOps.Multiply(c, uip.Real), _numOps.Multiply(s, uiq.Real)),
                                      _numOps.Add(_numOps.Multiply(c, uip.Imaginary), _numOps.Multiply(s, uiq.Imaginary)));
                U[i, q] = new Complex<T>(_numOps.Subtract(_numOps.Multiply(c, uiq.Real), _numOps.Multiply(s, uip.Real)),
                                      _numOps.Subtract(_numOps.Multiply(c, uiq.Imaginary), _numOps.Multiply(s, uip.Imaginary)));
            }
        }

        // Extract singular values
        for (int i = 0; i < n; i++)
        {
            S[i, i] = _numOps.Sqrt(_numOps.Abs(A[i, i]));
        }

        return (S, U);
    }

    /// <summary>
    /// Computes the Takagi decomposition using the QR algorithm.
    /// </summary>
    /// <param name="matrix">The matrix to decompose</param>
    /// <returns>A tuple containing the diagonal matrix S and unitary matrix U</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The QR algorithm is an iterative method that gradually transforms the matrix
    /// into a simpler form. Each iteration brings the matrix closer to a diagonal form,
    /// making it easier to extract the important values.
    /// </para>
    /// </remarks>
    private (Matrix<T> S, Matrix<Complex<T>> U) ComputeTakagiQR(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        var S = new Matrix<T>(n, n);
        var U = Matrix<Complex<T>>.CreateIdentity(n);
        var A = matrix.ToComplexMatrix();

        const int maxIterations = 100;
        var tolerance = _numOps.FromDouble(1e-10);

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
                    if (_numOps.GreaterThan(CalculateMagnitude(A[i, j]), tolerance))
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
            S[i, i] = _numOps.Sqrt(_numOps.Abs(CalculateMagnitude(A[i, i])));
        }

        return (S, U);
    }

    /// <summary>
    /// Calculates the magnitude (absolute value) of a complex number.
    /// </summary>
    /// <param name="complex">The complex number</param>
    /// <returns>The magnitude of the complex number</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The magnitude of a complex number is its distance from zero in the complex plane.
    /// It's calculated using the Pythagorean theorem: sqrt(real² + imaginary²).
    /// </para>
    /// </remarks>
    private T CalculateMagnitude(Complex<T> complex)
    {
        return _numOps.Sqrt(_numOps.Add(_numOps.Square(complex.Real), _numOps.Square(complex.Imaginary)));
    }

    /// <summary>
    /// Performs QR decomposition on a matrix.
    /// </summary>
    /// <param name="A">The matrix to decompose</param>
    /// <returns>A tuple containing the Q and R matrices</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> QR decomposition breaks a matrix into two parts:
    /// Q (an orthogonal matrix) and R (an upper triangular matrix).
    /// This is useful for solving systems of equations and finding eigenvalues.
    /// </para>
    /// </remarks>
    private (Matrix<Complex<T>> Q, Matrix<T> R) QRDecomposition(Matrix<T> A)
    {
        int n = A.Rows;
        var Q = Matrix<Complex<T>>.CreateIdentity(n);
        var R = A.Clone();

        for (int j = 0; j < n - 1; j++)
        {
            for (int i = j + 1; i < n; i++)
            {
                T a = R[j, j];
                T b = R[i, j];
                T r = _numOps.Sqrt(_numOps.Add(_numOps.Square(a), _numOps.Square(b)));
                T c = _numOps.Divide(a, r);
                T s = _numOps.Divide(b, r);

                // Update R
                for (int k = j; k < n; k++)
                {
                    T temp = R[j, k];
                    R[j, k] = _numOps.Add(_numOps.Multiply(c, temp), _numOps.Multiply(s, R[i, k]));
                    R[i, k] = _numOps.Subtract(_numOps.Multiply(c, R[i, k]), _numOps.Multiply(s, temp));
                }

                // Update Q
                for (int k = 0; k < n; k++)
                {
                    Complex<T> temp = Q[k, j];
                    Q[k, j] = new Complex<T>(_numOps.Add(_numOps.Multiply(c, temp.Real), _numOps.Multiply(s, Q[k, i].Real)),
                                          _numOps.Add(_numOps.Multiply(c, temp.Imaginary), _numOps.Multiply(s, Q[k, i].Imaginary)));
                    Q[k, i] = new Complex<T>(_numOps.Subtract(_numOps.Multiply(c, Q[k, i].Real), _numOps.Multiply(s, temp.Real)),
                                          _numOps.Subtract(_numOps.Multiply(c, Q[k, i].Imaginary), _numOps.Multiply(s, temp.Imaginary)));
                }
            }
        }

        return (Q, R);
    }

    /// <summary>
    /// Computes the Takagi decomposition using eigendecomposition.
    /// </summary>
    /// <param name="matrix">The matrix to decompose</param>
    /// <returns>A tuple containing the diagonal matrix S and unitary matrix U</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Eigendecomposition breaks down a matrix using its eigenvalues and eigenvectors.
    /// Eigenvalues represent how much the matrix stretches or shrinks in certain directions,
    /// while eigenvectors represent those directions.
    /// </para>
    /// </remarks>
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
            S[i, i] = _numOps.Sqrt(_numOps.Abs(eigenValues[i]));
            for (int j = 0; j < rows; j++)
            {
                U[i, j] = new Complex<T>(eigenVectors[i, j], _numOps.Zero);
            }
        }

        return (S, U);
    }

    /// <summary>
    /// Computes the Takagi decomposition using the power iteration method.
    /// </summary>
    /// <param name="matrix">The matrix to decompose</param>
    /// <returns>A tuple containing the diagonal matrix S and unitary matrix U</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Power iteration is a simple algorithm that finds the most important direction
    /// (eigenvector) in your data. It works by repeatedly multiplying a vector by the matrix,
    /// which gradually aligns the vector with the dominant eigenvector.
    /// </para>
    /// </remarks>
    private (Matrix<T> S, Matrix<Complex<T>> U) ComputeTakagiPowerIteration(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        var S = new Matrix<T>(n, n);
        var U = new Matrix<Complex<T>>(n, n);

        for (int i = 0; i < n; i++)
        {
            var v = Vector<T>.CreateRandom(n);
            var lambda = _numOps.Zero;

            for (int iter = 0; iter < 100; iter++)
            {
                var w = matrix.Multiply(v);
                var newLambda = v.DotProduct(w);
                v = w.Divide(w.Norm());

                if (_numOps.LessThan(_numOps.Abs(_numOps.Subtract(newLambda, lambda)), _numOps.FromDouble(1e-10)))
                {
                    break;
                }
                lambda = newLambda;
            }

            S[i, i] = _numOps.Sqrt(_numOps.Abs(lambda));
            for (int j = 0; j < n; j++)
            {
                U[j, i] = new Complex<T>(v[j], _numOps.Zero);
            }

            // Deflate the matrix
            for (int j = 0; j < n; j++)
            {
                for (int k = 0; k < n; k++)
                {
                    matrix[j, k] = _numOps.Subtract(matrix[j, k], _numOps.Multiply(_numOps.Multiply(v[j], v[k]), lambda));
                }
            }
        }

        return (S, U);
    }

    /// <summary>
    /// Computes the Takagi decomposition using the Lanczos iteration method.
    /// </summary>
    /// <param name="matrix">The matrix to decompose</param>
    /// <returns>A tuple containing the diagonal matrix S and unitary matrix U</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Lanczos algorithm is a technique that transforms a large matrix into a smaller,
    /// simpler form (called tridiagonal) that's easier to work with. Think of it like summarizing a long
    /// book into key points while preserving the most important information. This makes calculations much
    /// faster while still giving accurate results.
    /// </para>
    /// </remarks>
    private (Matrix<T> S, Matrix<Complex<T>> U) ComputeTakagiLanczosIteration(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        var S = new Matrix<T>(n, n);
        var U = new Matrix<Complex<T>>(n, n);

        // Initialize with random vector
        var v = Vector<T>.CreateRandom(n);
        var _vectors = new List<Vector<T>> { v.Divide(v.Norm()) };
        var _alphaCoefficients = new List<T>();
        var _betaCoefficients = new List<T>();

        for (int j = 0; j < n; j++)
        {
            var w = matrix.Multiply(_vectors[j]);
            var alphaj = _vectors[j].DotProduct(w);
            _alphaCoefficients.Add(alphaj);

            if (j < n - 1)
            {
                w = w.Subtract(_vectors[j].Multiply(alphaj));
                if (j > 0)
                {
                    w = w.Subtract(_vectors[j - 1].Multiply(_betaCoefficients[j - 1]));
                }

                var betaj = w.Norm();
                _betaCoefficients.Add(betaj);

                if (_numOps.LessThan(_numOps.Abs(betaj), _numOps.FromDouble(1e-10)))
                {
                    break;
                }

                _vectors.Add(w.Divide(betaj));
            }
        }

        // Construct tridiagonal matrix
        var T = new Matrix<T>(n, n);
        for (int i = 0; i < _alphaCoefficients.Count; i++)
        {
            T[i, i] = _alphaCoefficients[i];
            if (i < _betaCoefficients.Count)
            {
                T[i, i + 1] = _betaCoefficients[i];
                T[i + 1, i] = _betaCoefficients[i];
            }
        }

        // Compute eigendecomposition of T
        var eigenDecomposition = new EigenDecomposition<T>(T);
        var eigenValues = eigenDecomposition.EigenValues;
        var eigenVectors = eigenDecomposition.EigenVectors;

        for (int i = 0; i < n; i++)
        {
            S[i, i] = _numOps.Sqrt(_numOps.Abs(eigenValues[i]));
            for (int j = 0; j < n; j++)
            {
                U[i, j] = new Complex<T>(_numOps.Zero, _numOps.Zero);
                for (int k = 0; k < _vectors.Count; k++)
                {
                    Complex<T> term = new Complex<T>(_numOps.Multiply(_vectors[k][i], eigenVectors[k, j]), _numOps.Zero);
                    U[i, j] = new Complex<T>(
                        _numOps.Add(U[i, j].Real, term.Real),
                        _numOps.Add(U[i, j].Imaginary, term.Imaginary)
                    );
                }
            }
        }

        return (S, U);
    }

    /// <summary>
    /// Inverts the matrix using the Takagi decomposition.
    /// </summary>
    /// <returns>The inverted matrix</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Matrix inversion is like finding the reciprocal of a number (e.g., 1/x).
    /// When you multiply a matrix by its inverse, you get the identity matrix (similar to how x * (1/x) = 1).
    /// This method uses the decomposition we've already calculated to find the inverse efficiently.
    /// </para>
    /// </remarks>
    public Matrix<T> Invert()
    {
        var invSigma = SigmaMatrix.InvertDiagonalMatrix();
        var invU = UnitaryMatrix.InvertUnitaryMatrix();
        var invSigmaComplex = invSigma.ToComplexMatrix();
        var inv = invU.Multiply(invSigmaComplex).Multiply(invU.Transpose());

        return inv.ToRealMatrix();
    }

    /// <summary>
    /// Solves a linear system of equations Ax = b, where A is the matrix represented by this decomposition.
    /// </summary>
    /// <param name="bVector">The right-hand side vector b in the equation Ax = b</param>
    /// <returns>The solution vector x</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method finds the values of x in the equation Ax = b.
    /// Think of it like solving for x in the equation 3x = 6 (where x = 2),
    /// but with matrices instead of simple numbers. Using the decomposition makes
    /// this process much more efficient than directly inverting the matrix.
    /// </para>
    /// </remarks>
    public Vector<T> Solve(Vector<T> bVector)
    {
        var bComplex = new Vector<Complex<T>>(bVector.Length);
        for (int i = 0; i < bVector.Length; i++)
        {
            bComplex[i] = new Complex<T>(bVector[i], _numOps.Zero);
        }
        var yVector = UnitaryMatrix.ForwardSubstitution(bComplex);

        var result = SigmaMatrix.ToComplexMatrix().BackwardSubstitution(yVector);
        return result.ToRealVector();
    }
}