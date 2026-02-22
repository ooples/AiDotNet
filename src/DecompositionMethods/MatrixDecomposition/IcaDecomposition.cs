namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

/// <summary>
/// Implements Independent Component Analysis (ICA) for blind source separation.
/// </summary>
/// <typeparam name="T">The numeric type used in the matrix.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> ICA is a technique used to separate mixed signals into their independent sources.
/// Imagine you're at a party with multiple people talking, and you have multiple microphones recording
/// the mixed conversations. ICA helps you separate out each individual voice.
/// </para>
/// <para>
/// The key idea is that ICA finds statistically independent components that were combined together.
/// This is different from other decomposition methods because it focuses on statistical independence
/// rather than uncorrelated patterns.
/// </para>
/// <para>
/// Common applications include:
/// - Audio source separation (cocktail party problem)
/// - Brain signal analysis (EEG, fMRI)
/// - Image separation and feature extraction
/// - Financial data analysis
/// - Telecommunications signal processing
/// </para>
/// </remarks>
public class IcaDecomposition<T> : MatrixDecompositionBase<T>
{
    /// <summary>
    /// Gets the unmixing matrix (separation matrix).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The unmixing matrix W transforms the mixed signals back into
    /// independent source signals. If X represents your mixed signals, then S = W * X gives
    /// you the separated independent components.
    /// </para>
    /// <para>
    /// Think of it as a filter that untangles the mixed signals, similar to how noise-canceling
    /// headphones filter out background noise.
    /// </para>
    /// </remarks>
    public Matrix<T> UnmixingMatrix { get; private set; } = new Matrix<T>(0, 0);

    /// <summary>
    /// Gets the mixing matrix (inverse of unmixing matrix).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The mixing matrix A represents how the original independent sources
    /// were combined to create the observed mixed signals. If S represents independent sources,
    /// then X = A * S gives you the mixed observations.
    /// </para>
    /// <para>
    /// This is essentially the "recipe" of how the sources were mixed together.
    /// </para>
    /// </remarks>
    public Matrix<T> MixingMatrix { get; private set; } = new Matrix<T>(0, 0);

    /// <summary>
    /// Gets the independent components (separated sources).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The independent components are the separated source signals that ICA
    /// has extracted from the mixed observations. Each row represents one independent source.
    /// </para>
    /// <para>
    /// For example, in the cocktail party problem:
    /// - Each row might represent one person's voice
    /// - Each column represents a time sample
    /// </para>
    /// </remarks>
    public Matrix<T> IndependentComponents { get; private set; } = new Matrix<T>(0, 0);

    /// <summary>
    /// Gets the mean vector used for centering the data.
    /// </summary>
    public Vector<T> Mean { get; private set; } = new Vector<T>(0);

    /// <summary>
    /// Gets the whitening matrix used in the preprocessing step.
    /// </summary>
    public Matrix<T> WhiteningMatrix { get; private set; } = new Matrix<T>(0, 0);

    private readonly int _numComponents;
    private readonly int _maxIterations;
    private readonly double _tolerance;

    /// <summary>
    /// Initializes a new instance of the ICA decomposition for the specified matrix.
    /// </summary>
    /// <param name="matrix">The matrix to decompose (observations * features).</param>
    /// <param name="components">The number of independent components to extract. If null, uses all components.</param>
    /// <param name="maxIterations">Maximum number of iterations for the FastICA algorithm.</param>
    /// <param name="tolerance">Convergence tolerance.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the ICA decomposition using the FastICA algorithm.
    /// The matrix should be organized with observations (samples) as rows and features (sensors/channels) as columns.
    /// </para>
    /// <para>
    /// Key parameters:
    /// - <paramref name="components"/>: How many independent sources to extract
    /// - <paramref name="maxIterations"/>: How long to search for the optimal separation
    /// - <paramref name="tolerance"/>: How close to convergence is "good enough"
    /// </para>
    /// </remarks>
    public IcaDecomposition(Matrix<T> matrix, int? components = null, int maxIterations = 200, double tolerance = 1e-4)
        : base(matrix)
    {
        _numComponents = components ?? Math.Min(matrix.Rows, matrix.Columns);

        if (_numComponents <= 0 || _numComponents > Math.Min(matrix.Rows, matrix.Columns))
        {
            throw new ArgumentException($"Number of components must be between 1 and {Math.Min(matrix.Rows, matrix.Columns)}.");
        }

        _maxIterations = maxIterations;
        _tolerance = tolerance;
        Decompose();
    }

    /// <summary>
    /// Performs the ICA decomposition.
    /// </summary>
    protected override void Decompose()
    {
        // Perform ICA decomposition
        (UnmixingMatrix, MixingMatrix, IndependentComponents, Mean, WhiteningMatrix) =
            ComputeFastIca(A, _numComponents, _maxIterations, _tolerance);
    }

    /// <summary>
    /// Computes Independent Component Analysis using the FastICA algorithm.
    /// </summary>
    /// <param name="X">The input matrix (observations * features).</param>
    /// <param name="numComponents">Number of independent components to extract.</param>
    /// <param name="maxIterations">Maximum iterations.</param>
    /// <param name="tolerance">Convergence tolerance.</param>
    /// <returns>A tuple containing the unmixing matrix, mixing matrix, independent components, mean, and whitening matrix.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> FastICA is an efficient algorithm for finding independent components.
    /// It works in three main steps:
    /// </para>
    /// <para>
    /// 1. <b>Centering:</b> Subtract the mean from each feature to center the data around zero.
    ///    This is like adjusting your reference point to make calculations easier.
    /// </para>
    /// <para>
    /// 2. <b>Whitening:</b> Transform the data so that features are uncorrelated and have unit variance.
    ///    This is like normalizing the data to a standard scale before separation.
    /// </para>
    /// <para>
    /// 3. <b>Separation:</b> Find the directions of maximum non-Gaussianity, which correspond to
    ///    the independent components. This uses the principle that independent signals are typically
    ///    non-Gaussian (not bell-curve shaped).
    /// </para>
    /// </remarks>
    private (Matrix<T> W, Matrix<T> A, Matrix<T> S, Vector<T> mean, Matrix<T> K)
        ComputeFastIca(Matrix<T> X, int numComponents, int maxIterations, double tolerance)
    {
        // Step 1: Center the data (subtract mean from each column)
        Vector<T> mean = ComputeColumnMean(X);
        Matrix<T> XCentered = CenterData(X, mean);

        // Step 2: Whiten the data using PCA/SVD
        (Matrix<T> XWhitened, Matrix<T> K) = WhitenData(XCentered, numComponents);

        // Step 3: Apply FastICA algorithm to find independent components
        Matrix<T> W = FastIcaAlgorithm(XWhitened, numComponents, maxIterations, tolerance);

        // Compute mixing matrix A (pseudo-inverse of W)
        Matrix<T> mixingMatrix = ComputeMixingMatrix(W, K);

        // Compute independent components S = W * K^T * (X - mean)^T
        Matrix<T> S = W.Multiply(XWhitened.Transpose());

        return (W, mixingMatrix, S, mean, K);
    }

    /// <summary>
    /// Computes the mean of each column in the matrix.
    /// </summary>
    /// <param name="X">Input matrix.</param>
    /// <returns>Vector containing the mean of each column.</returns>
    private Vector<T> ComputeColumnMean(Matrix<T> X)
    {
        int n = X.Columns;
        int m = X.Rows;
        Vector<T> mean = new Vector<T>(n);
        T invM = NumOps.FromDouble(1.0 / m);

        // VECTORIZED: Use GetColumn to extract column and compute sum via dot product
        for (int j = 0; j < n; j++)
        {
            Vector<T> col = X.GetColumn(j);
            // Use dot product with ones vector to compute sum
            Vector<T> ones = new Vector<T>(m);
            for (int i = 0; i < m; i++)
            {
                ones[i] = NumOps.One;
            }
            T sum = col.DotProduct(ones);
            mean[j] = NumOps.Multiply(sum, invM);
        }

        return mean;
    }

    /// <summary>
    /// Centers the data by subtracting the mean from each column.
    /// </summary>
    /// <param name="X">Input matrix.</param>
    /// <param name="mean">Mean vector.</param>
    /// <returns>Centered matrix.</returns>
    private Matrix<T> CenterData(Matrix<T> X, Vector<T> mean)
    {
        int m = X.Rows;
        int n = X.Columns;
        Matrix<T> centered = new Matrix<T>(m, n);

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                centered[i, j] = NumOps.Subtract(X[i, j], mean[j]);
            }
        }

        return centered;
    }

    /// <summary>
    /// Whitens the data using eigenvalue decomposition.
    /// </summary>
    /// <param name="X">Centered input matrix.</param>
    /// <param name="numComponents">Number of components to keep.</param>
    /// <returns>Whitened data matrix and whitening matrix K.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Whitening transforms the data so that features become uncorrelated
    /// and have the same variance. This preprocessing step makes it easier for ICA to find
    /// independent components.
    /// </para>
    /// <para>
    /// Think of it like standardizing test scores from different subjects before comparing them.
    /// It puts everything on the same scale and removes correlations.
    /// </para>
    /// </remarks>
    private (Matrix<T> XWhitened, Matrix<T> K) WhitenData(Matrix<T> X, int numComponents)
    {
        int m = X.Rows;

        // Compute covariance matrix C = (1/m) * X^T * X
        Matrix<T> XT = X.Transpose();
        Matrix<T> C = XT.Multiply(X);
        T scale = NumOps.FromDouble(1.0 / m);

        // VECTORIZED: Scale entire matrix using Engine operations row by row
        for (int i = 0; i < C.Rows; i++)
        {
            Vector<T> row = C.GetRow(i);
            Vector<T> scaledRow = (Vector<T>)Engine.Multiply(row, scale);
            for (int j = 0; j < C.Columns; j++)
            {
                C[i, j] = scaledRow[j];
            }
        }

        // Compute eigenvalue decomposition of covariance matrix
        var eigen = new EigenDecomposition<T>(C);

        // Select top numComponents eigenvectors and eigenvalues
        Matrix<T> E = new Matrix<T>(numComponents, C.Rows);
        Vector<T> D = new Vector<T>(numComponents);

        for (int i = 0; i < numComponents; i++)
        {
            D[i] = eigen.EigenValues[i];
            for (int j = 0; j < C.Rows; j++)
            {
                E[i, j] = eigen.EigenVectors[j, i];
            }
        }

        // Compute whitening matrix K = D^(-1/2) * E
        Matrix<T> K = new Matrix<T>(numComponents, C.Rows);
        for (int i = 0; i < numComponents; i++)
        {
            // Validate eigenvalue is positive before taking square root
            if (NumOps.LessThanOrEquals(D[i], NumOps.Zero))
            {
                throw new InvalidOperationException(
                    $"Eigenvalue at index {i} is non-positive ({D[i]}). " +
                    "ICA whitening requires positive definite covariance matrix. " +
                    "This may indicate ill-conditioned or degenerate data.");
            }

            T invSqrtEigenvalue = NumOps.Divide(NumOps.One, NumOps.Sqrt(D[i]));
            for (int j = 0; j < C.Rows; j++)
            {
                K[i, j] = NumOps.Multiply(E[i, j], invSqrtEigenvalue);
            }
        }

        // Whiten the data: XWhitened = K * X^T
        Matrix<T> XWhitened = K.Multiply(XT).Transpose();

        return (XWhitened, K);
    }

    /// <summary>
    /// Implements the FastICA algorithm to find the unmixing matrix.
    /// </summary>
    /// <param name="X">Whitened data matrix.</param>
    /// <param name="numComponents">Number of components.</param>
    /// <param name="maxIterations">Maximum iterations.</param>
    /// <param name="tolerance">Convergence tolerance.</param>
    /// <returns>The unmixing matrix W.</returns>
    private Matrix<T> FastIcaAlgorithm(Matrix<T> X, int numComponents, int maxIterations, double tolerance)
    {
        int m = X.Rows;
        int n = X.Columns;
        Matrix<T> W = new Matrix<T>(numComponents, n);

        // Initialize W with random values
        var random = RandomHelper.CreateSecureRandom();
        for (int i = 0; i < numComponents; i++)
        {
            for (int j = 0; j < n; j++)
            {
                W[i, j] = NumOps.FromDouble(random.NextDouble() * 2 - 1);
            }
        }

        // Orthogonalize initial W
        W = GramSchmidtOrthogonalization(W);

        T toleranceT = NumOps.FromDouble(tolerance);

        // Iterate to find each independent component
        for (int component = 0; component < numComponents; component++)
        {
            Vector<T> w = W.GetRow(component);

            for (int iteration = 0; iteration < maxIterations; iteration++)
            {
                Vector<T> wOld = w.Clone();

                // Compute w = E{x * g(w^T * x)} - E{g'(w^T * x)} * w
                // where g(u) = tanh(u) is the non-linearity function
                Vector<T> wNew = new Vector<T>(n);
                T invM = NumOps.FromDouble(1.0 / m);

                // VECTORIZED: Precompute g and g' values for all samples
                var gValues = new T[m];
                var gPrimeValues = new T[m];

                for (int i = 0; i < m; i++)
                {
                    Vector<T> x = X.GetRow(i);
                    T wtx = w.DotProduct(x);

                    // g(u) = tanh(u)
                    gValues[i] = Tanh(wtx);
                    // g'(u) = 1 - tanh²(u)
                    gPrimeValues[i] = NumOps.Subtract(NumOps.One, NumOps.Multiply(gValues[i], gValues[i]));
                }

                var gVec = new Vector<T>(gValues);
                var gPrimeVec = new Vector<T>(gPrimeValues);

                // VECTORIZED: Use dot product for sum computations
                for (int j = 0; j < n; j++)
                {
                    // Extract column j from X
                    Vector<T> xCol = X.GetColumn(j);

                    // sum1 = Σ(x[i,j] * g[i]) - use dot product
                    T sum1 = xCol.DotProduct(gVec);

                    // sum2 = Σ(g'[i]) - use dot product with ones vector
                    Vector<T> ones = new Vector<T>(m);
                    for (int k = 0; k < m; k++)
                    {
                        ones[k] = NumOps.One;
                    }
                    T sum2 = gPrimeVec.DotProduct(ones);

                    T avg1 = NumOps.Multiply(sum1, invM);
                    T avg2 = NumOps.Multiply(sum2, invM);
                    wNew[j] = NumOps.Subtract(avg1, NumOps.Multiply(avg2, w[j]));
                }

                w = wNew;

                // Orthogonalize against previous components
                for (int j = 0; j < component; j++)
                {
                    Vector<T> wj = W.GetRow(j);
                    T projection = w.DotProduct(wj);
                    // VECTORIZED: Subtract projection using Engine operations
                    var proj = (Vector<T>)Engine.Multiply(wj, projection);
                    w = (Vector<T>)Engine.Subtract(w, proj);
                }

                // Normalize w
                T norm = w.Norm();
                if (!NumOps.Equals(norm, NumOps.Zero))
                {
                    // VECTORIZED: Normalize using Engine division
                    w = (Vector<T>)Engine.Divide(w, norm);
                }

                // Check for convergence
                T distance = NumOps.Abs(NumOps.Subtract(NumOps.Abs(w.DotProduct(wOld)), NumOps.One));
                if (NumOps.LessThan(distance, toleranceT))
                {
                    break;
                }
            }

            // Store the component
            for (int j = 0; j < n; j++)
            {
                W[component, j] = w[j];
            }
        }

        return W;
    }

    /// <summary>
    /// Computes the hyperbolic tangent (tanh) function using only INumericOperations.
    /// </summary>
    /// <param name="x">Input value.</param>
    /// <returns>tanh(x) = (e^x - e^-x) / (e^x + e^-x).</returns>
    /// <remarks>
    /// This implementation uses only INumericOperations primitives to ensure compatibility
    /// with any numeric type, avoiding Convert.ToDouble which would fail for custom types.
    /// </remarks>
    private T Tanh(T x)
    {
        // tanh(x) = (e^x - e^-x) / (e^x + e^-x)
        T expX = NumOps.Exp(x);
        T negX = NumOps.Negate(x);
        T expNegX = NumOps.Exp(negX);

        T numerator = NumOps.Subtract(expX, expNegX);
        T denominator = NumOps.Add(expX, expNegX);

        return NumOps.Divide(numerator, denominator);
    }

    /// <summary>
    /// Orthogonalizes matrix rows using the Gram-Schmidt process.
    /// </summary>
    /// <param name="matrix">Input matrix.</param>
    /// <returns>Orthogonalized matrix.</returns>
    private Matrix<T> GramSchmidtOrthogonalization(Matrix<T> matrix)
    {
        int rows = matrix.Rows;
        int cols = matrix.Columns;
        Matrix<T> result = new Matrix<T>(rows, cols);

        for (int i = 0; i < rows; i++)
        {
            Vector<T> v = matrix.GetRow(i);

            // Subtract projections onto previous vectors
            for (int j = 0; j < i; j++)
            {
                Vector<T> u = result.GetRow(j);
                T projection = v.DotProduct(u);
                // VECTORIZED: Subtract projection using Engine operations
                var proj = (Vector<T>)Engine.Multiply(u, projection);
                v = (Vector<T>)Engine.Subtract(v, proj);
            }

            // Normalize
            T norm = v.Norm();
            if (!NumOps.Equals(norm, NumOps.Zero))
            {
                // VECTORIZED: Normalize using Engine division
                v = (Vector<T>)Engine.Divide(v, norm);
            }

            for (int k = 0; k < cols; k++)
            {
                result[i, k] = v[k];
            }
        }

        return result;
    }

    /// <summary>
    /// Computes the mixing matrix from the unmixing matrix and whitening matrix.
    /// </summary>
    /// <param name="W">Unmixing matrix.</param>
    /// <param name="K">Whitening matrix.</param>
    /// <returns>Mixing matrix.</returns>
    private Matrix<T> ComputeMixingMatrix(Matrix<T> W, Matrix<T> K)
    {
        // The total unmixing is WK: S = W * K * X_centered^T
        // The mixing matrix is the (pseudo-)inverse of WK: A = pinv(W * K)
        Matrix<T> WK = W.Multiply(K);

        if (WK.Rows == WK.Columns)
        {
            // Square case: use LU decomposition for proper inverse
            var lu = new LuDecomposition<T>(WK);
            return lu.Invert();
        }
        else
        {
            // Non-square case: use pseudo-inverse pinv(WK) = WK^T * (WK * WK^T)^{-1}
            Matrix<T> WKT = WK.Transpose();
            Matrix<T> WKWKt = WK.Multiply(WKT);
            var lu = new LuDecomposition<T>(WKWKt);
            Matrix<T> WKWKtInv = lu.Invert();
            return WKT.Multiply(WKWKtInv);
        }
    }

    /// <summary>
    /// Solves a linear system Ax = b using the ICA decomposition.
    /// </summary>
    /// <param name="b">The right-hand side vector of the equation.</param>
    /// <returns>The solution vector x.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method finds an approximate solution to Ax = b using the ICA factorization.
    /// Since ICA provides a separation A ~= mixing_matrix * unmixing_matrix, we can use this to solve
    /// the linear system, though it's not the primary purpose of ICA.
    /// </para>
    /// <para>
    /// For solving linear systems, consider using decomposition methods specifically designed for that
    /// purpose, such as LU or QR decomposition.
    /// </para>
    /// </remarks>
    public override Vector<T> Solve(Vector<T> b)
    {
        if (b.Length != A.Columns)
        {
            throw new ArgumentException(
                $"Input vector dimension ({b.Length}) must match matrix columns ({A.Columns}).");
        }

        // Use the mixing matrix to solve: A * x = b
        // We approximate: x ~= unmixing_matrix * whitening_matrix^T * (b - mean)

        // Center b
        Vector<T> bCentered = new Vector<T>(b.Length);
        for (int i = 0; i < b.Length; i++)
        {
            bCentered[i] = NumOps.Subtract(b[i], Mean[i]);
        }

        // Apply whitening and unmixing
        Vector<T> whitened = WhiteningMatrix.Multiply(bCentered);
        Vector<T> x = UnmixingMatrix.Transpose().Multiply(whitened);

        return x;
    }

    // Invert() is handled by the base class MatrixDecompositionBase.
    // ICA is not designed for matrix inversion; use LU or SVD decomposition for that purpose.

    /// <summary>
    /// Transforms new data using the learned ICA model.
    /// </summary>
    /// <param name="X">New observations to transform.</param>
    /// <returns>Independent components for the new data.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method applies the learned ICA separation to new data.
    /// It's like using the same "un-mixing recipe" that was learned from the training data
    /// to separate new mixed signals.
    /// </para>
    /// <para>
    /// This is useful when you want to apply the same source separation to new recordings
    /// or measurements.
    /// </para>
    /// </remarks>
    public Matrix<T> Transform(Matrix<T> X)
    {
        if (X.Columns != A.Columns)
        {
            throw new ArgumentException(
                $"Input matrix columns ({X.Columns}) must match original matrix columns ({A.Columns}).");
        }

        // Center the data
        Matrix<T> XCentered = CenterData(X, Mean);

        // Whiten the data
        Matrix<T> XWhitened = XCentered.Multiply(WhiteningMatrix.Transpose());

        // Apply unmixing
        Matrix<T> S = XWhitened.Multiply(UnmixingMatrix.Transpose());

        return S;
    }
}
