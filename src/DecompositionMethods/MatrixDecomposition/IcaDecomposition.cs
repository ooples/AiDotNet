using AiDotNet.Attributes;
using AiDotNet.Enums;

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
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.DimensionalityReduction)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
    [ResearchPaper("Independent Component Analysis: Algorithms and Applications", "https://doi.org/10.1016/S0893-6080(00)00026-5")]
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
        if (X.Rows == 0)
        {
            return new Vector<T>(X.Columns);
        }

        return Engine.ReduceMean(Tensor<T>.FromMatrix(X), new[] { 0 }, keepDims: false).ToVector();
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
        var rowMean = Engine.TensorBroadcastTo(Tensor<T>.FromVector(mean, new[] { 1, n }), new[] { m, n });
        return Engine.TensorSubtract(Tensor<T>.FromMatrix(X), rowMean).ToMatrix();
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

        // Covariance C = X^T X / m
        Matrix<T> XT = X.Transpose();
        Matrix<T> C = Engine.MatrixMultiplyScalar(XT.Multiply(X), NumOps.FromDouble(1.0 / m));

        var eigen = new EigenDecomposition<T>(C);
        int d = C.Rows;

        // Leading numComponents eigenpairs: D (values) and E (eigenvectors as rows).
        var eigenvalues = Tensor<T>.FromVector(eigen.EigenValues);
        var D = Engine.TensorSlice(eigenvalues, new[] { 0 }, new[] { numComponents });
        var E = Engine.TensorTranspose(Engine.TensorSlice(
            Tensor<T>.FromMatrix(eigen.EigenVectors), new[] { 0, 0 }, new[] { d, numComponents }));

        // Whitening needs strictly positive eigenvalues; report the first one that is not.
        var nonPositive = Engine.TensorSubtract(Engine.TensorOnesLike(D), Engine.TensorGreaterThan(D, NumOps.Zero));
        if (Engine.TensorCountNonzero(nonPositive) > 0)
        {
            int i = Engine.TensorArgMax(nonPositive, 0)[0];
            throw new InvalidOperationException(
                $"Eigenvalue at index {i} is non-positive ({D[i]}). " +
                "ICA whitening requires positive definite covariance matrix. " +
                "This may indicate ill-conditioned or degenerate data.");
        }

        // K = diag(1 / sqrt(D)) * E
        var invSqrt = Engine.TensorReciprocal(Engine.TensorSqrt(D));
        Matrix<T> K = Engine.TensorMatMul(Engine.TensorDiagEmbed(invSqrt), E).ToMatrix();

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

        // Random initialization in [-1, 1) from a secure source, then orthonormalize the rows.
        var random = RandomHelper.CreateSecureRandom();
        var uniform = Tensor<T>.FromMatrix(Matrix<T>.CreateRandom(random, numComponents, n));
        var initial = Engine.TensorSubtractScalar(Engine.TensorMultiplyScalar(uniform, NumOps.FromDouble(2)), NumOps.One);
        Matrix<T> W = GramSchmidtOrthogonalization(initial.ToMatrix());

        T toleranceT = NumOps.FromDouble(tolerance);
        T invM = NumOps.FromDouble(1.0 / m);
        T one = NumOps.One;
        var x = Tensor<T>.FromMatrix(X);
        var xT = Engine.TensorTranspose(x);

        for (int component = 0; component < numComponents; component++)
        {
            var w = Tensor<T>.FromVector(W.GetRow(component), new[] { n, 1 });

            for (int iteration = 0; iteration < maxIterations; iteration++)
            {
                var wOld = w;

                // g(u) = tanh(u) and its derivative 1 - tanh(u)^2 for u = X w (one value per sample)
                var g = Engine.TensorTanh(Engine.TensorMatMul(x, w));
                var gPrime = Engine.TensorSubtract(Engine.TensorOnesLike(g), Engine.TensorSquare(g));

                // w+ = mean(x g(w^T x)) - mean(g'(w^T x)) w
                var expectation = Engine.TensorMultiplyScalar(Engine.TensorMatMul(xT, g), invM);
                T meanGPrime = Engine.TensorMean(gPrime);
                w = Engine.TensorSubtract(expectation, Engine.TensorMultiplyScalar(w, meanGPrime));

                // Deflation: remove the projections onto the components found so far.
                if (component > 0)
                {
                    var previous = Engine.TensorSlice(Tensor<T>.FromMatrix(W), new[] { 0, 0 }, new[] { component, n });
                    var projections = Engine.TensorMatMul(previous, w);
                    w = Engine.TensorSubtract(w, Engine.TensorMatMul(Engine.TensorTranspose(previous), projections));
                }

                // Normalize (a zero vector is left as is).
                var norm = Engine.TensorNorm(w);
                if (Engine.TensorCountNonzero(norm) > 0)
                {
                    w = Engine.TensorDivideScalar(w, Engine.TensorSum(norm));
                }

                // Converged when |w . wOld| is within tolerance of 1. TensorVecDot takes rank-1 operands.
                var alignment = Tensor<T>.FromScalar(
                    Engine.TensorVecDot(w.Reshape(new[] { n }), wOld.Reshape(new[] { n })));
                var distance = Engine.TensorAbs(Engine.TensorSubtractScalar(Engine.TensorAbs(alignment), one));
                if (AnyBelow(distance, toleranceT))
                {
                    break;
                }
            }

            W.SetRow(component, w.Reshape(new[] { n }).ToVector());
        }

        return W;
    }

    /// <summary>
    /// Orthogonalizes matrix rows using the Gram-Schmidt process.
    /// </summary>
    /// <param name="matrix">Input matrix.</param>
    /// <returns>Orthogonalized matrix.</returns>
    private Matrix<T> GramSchmidtOrthogonalization(Matrix<T> matrix)
    {
        // Orthonormalizing the rows of M is the thin QR of M^T: the rows of Q^T, with R[i,i] >= 0
        // matching the Gram-Schmidt normalization by a positive norm.
        var (q, _) = ComputePositiveDiagonalQr(matrix.Transpose());
        return q.Transpose();
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
        else if (WK.Rows < WK.Columns)
        {
            // Wide case (more columns than rows): pinv(WK) = WK^T * (WK * WK^T)^{-1}
            Matrix<T> WKT = WK.Transpose();
            Matrix<T> WKWKt = WK.Multiply(WKT);
            var lu = new LuDecomposition<T>(WKWKt);
            Matrix<T> WKWKtInv = lu.Invert();
            return WKT.Multiply(WKWKtInv);
        }
        else
        {
            // Tall case (more rows than columns): pinv(WK) = (WK^T * WK)^{-1} * WK^T
            Matrix<T> WKT = WK.Transpose();
            Matrix<T> WKtWK = WKT.Multiply(WK);
            var lu = new LuDecomposition<T>(WKtWK);
            Matrix<T> WKtWKInv = lu.Invert();
            return WKtWKInv.Multiply(WKT);
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
        Vector<T> bCentered = Engine.Subtract(b, Mean);

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
