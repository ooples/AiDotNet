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
public class TakagiDecomposition<T> : MatrixDecompositionBase<T>
{
    /// <summary>
    /// Gets the diagonal matrix containing the singular values.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Singular values represent the "strength" or "importance" of different dimensions in your data.
    /// Larger singular values indicate more important patterns in the data.
    /// </remarks>
    public Matrix<T> SigmaMatrix { get; private set; } = new Matrix<T>(0, 0);

    /// <summary>
    /// Gets the unitary matrix used in the decomposition.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> A unitary matrix preserves lengths and angles when multiplied with vectors.
    /// It's like rotating or reflecting data without changing its fundamental structure.
    /// </remarks>
    public Matrix<Complex<T>> UnitaryMatrix { get; private set; } = new Matrix<Complex<T>>(0, 0);

    private readonly TakagiAlgorithmType _algorithm;

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
        : base(matrix)
    {
        _algorithm = algorithm;

        Decompose();
    }

    /// <summary>
    /// Performs the Takagi decomposition.
    /// </summary>
    protected override void Decompose()
    {
        (SigmaMatrix, UnitaryMatrix) = ComputeDecomposition(A, _algorithm);
    }

    /// <summary>
    /// Computes the Takagi decomposition using the specified algorithm.
    /// </summary>
    /// <param name="matrix">The matrix to decompose</param>
    /// <param name="algorithm">The algorithm to use</param>
    /// <returns>A tuple containing the singular values matrix and unitary matrix</returns>
    private (Matrix<T> S, Matrix<Complex<T>> U) ComputeDecomposition(Matrix<T> matrix, TakagiAlgorithmType algorithm)
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

        // VECTORIZED: Process each column of eigenvectors as a vector operation
        for (int i = 0; i < rows; i++)
        {
            S[i, i] = NumOps.Sqrt(NumOps.Abs(eigenValues[i]));

            Vector<T> eigenVector = eigenVectors.GetColumn(i);
            Vector<Complex<T>> complexVector = new Vector<Complex<T>>(eigenVector.Select(val => new Complex<T>(val, NumOps.Zero)));
            U.SetColumn(i, complexVector);
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

    /// <summary>
    /// Calculates the magnitude (absolute value) of a complex number.
    /// </summary>
    /// <param name="complex">The complex number</param>
    /// <returns>The magnitude of the complex number</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The magnitude of a complex number is its distance from zero in the complex plane.
    /// It's calculated using the Pythagorean theorem: sqrt(real* + imaginary*).
    /// </para>
    /// </remarks>
    private T CalculateMagnitude(Complex<T> complex)
    {
        return NumOps.Sqrt(NumOps.Add(NumOps.Square(complex.Real), NumOps.Square(complex.Imaginary)));
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

        // VECTORIZED: Process each column of eigenvectors as a vector operation
        for (int i = 0; i < rows; i++)
        {
            // For symmetric matrices, singular values = |eigenvalues|
            // (not sqrt(|eigenvalues|) since singular values = sqrt(eigenvalues of A^T A) = sqrt(λ²) = |λ|)
            S[i, i] = NumOps.Abs(eigenValues[i]);

            Vector<T> eigenVector = eigenVectors.GetColumn(i);
            Vector<Complex<T>> complexVector = new Vector<Complex<T>>(eigenVector.Select(val => new Complex<T>(val, NumOps.Zero)));
            U.SetColumn(i, complexVector);
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

            // VECTORIZED: Deflate the matrix using outer product
            Matrix<T> deflation = v.OuterProduct(v).Multiply(lambda);
            matrix = matrix.Subtract(deflation);
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
        // VECTORIZED: Normalize using Engine division
        var normalized = (Vector<T>)Engine.Divide(v, v.Norm());
        var _vectors = new List<Vector<T>> { normalized };
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

                if (NumOps.LessThan(NumOps.Abs(betaj), NumOps.FromDouble(1e-10)))
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
            S[i, i] = NumOps.Sqrt(NumOps.Abs(eigenValues[i]));
            for (int j = 0; j < n; j++)
            {
                U[i, j] = new Complex<T>(NumOps.Zero, NumOps.Zero);
                for (int k = 0; k < _vectors.Count; k++)
                {
                    Complex<T> term = new Complex<T>(NumOps.Multiply(_vectors[k][i], eigenVectors[k, j]), NumOps.Zero);
                    U[i, j] = new Complex<T>(
                        NumOps.Add(U[i, j].Real, term.Real),
                        NumOps.Add(U[i, j].Imaginary, term.Imaginary)
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
    public override Matrix<T> Invert()
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
    /// <para>
    /// For Takagi decomposition A = U * S * U^T, solve x = U * S^(-1) * U^T * b
    /// </para>
    /// </remarks>
    public override Vector<T> Solve(Vector<T> bVector)
    {
        int n = bVector.Length;

        // For Takagi decomposition A = U * S * U^T, solve x = U * S^(-1) * U^T * b
        // Step 1: Compute y = U^H * b (conjugate transpose of U times b)
        var yVector = new Vector<Complex<T>>(n);
        for (int i = 0; i < n; i++)
        {
            Complex<T> sum = new Complex<T>(NumOps.Zero, NumOps.Zero);
            for (int j = 0; j < n; j++)
            {
                // U^H[i,j] = conj(U[j,i])
                var uConjugate = new Complex<T>(UnitaryMatrix[j, i].Real, NumOps.Negate(UnitaryMatrix[j, i].Imaginary));
                var bVal = new Complex<T>(bVector[j], NumOps.Zero);
                var product = new Complex<T>(
                    NumOps.Subtract(NumOps.Multiply(uConjugate.Real, bVal.Real), NumOps.Multiply(uConjugate.Imaginary, bVal.Imaginary)),
                    NumOps.Add(NumOps.Multiply(uConjugate.Real, bVal.Imaginary), NumOps.Multiply(uConjugate.Imaginary, bVal.Real)));
                sum = new Complex<T>(NumOps.Add(sum.Real, product.Real), NumOps.Add(sum.Imaginary, product.Imaginary));
            }
            yVector[i] = sum;
        }

        // Step 2: Compute z = S^(-1) * y (divide by diagonal singular values)
        var zVector = new Vector<Complex<T>>(n);
        for (int i = 0; i < n; i++)
        {
            T sigma = SigmaMatrix[i, i];
            if (NumOps.Equals(sigma, NumOps.Zero))
            {
                // Handle zero singular value - set to zero (pseudoinverse behavior)
                zVector[i] = new Complex<T>(NumOps.Zero, NumOps.Zero);
            }
            else
            {
                zVector[i] = new Complex<T>(
                    NumOps.Divide(yVector[i].Real, sigma),
                    NumOps.Divide(yVector[i].Imaginary, sigma));
            }
        }

        // Step 3: Compute x = U * z
        var xVector = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            Complex<T> sum = new Complex<T>(NumOps.Zero, NumOps.Zero);
            for (int j = 0; j < n; j++)
            {
                var u = UnitaryMatrix[i, j];
                var z = zVector[j];
                var product = new Complex<T>(
                    NumOps.Subtract(NumOps.Multiply(u.Real, z.Real), NumOps.Multiply(u.Imaginary, z.Imaginary)),
                    NumOps.Add(NumOps.Multiply(u.Real, z.Imaginary), NumOps.Multiply(u.Imaginary, z.Real)));
                sum = new Complex<T>(NumOps.Add(sum.Real, product.Real), NumOps.Add(sum.Imaginary, product.Imaginary));
            }
            // Take real part of result
            xVector[i] = sum.Real;
        }

        return xVector;
    }
}
