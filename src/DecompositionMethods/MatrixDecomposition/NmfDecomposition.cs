using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.DecompositionMethods.MatrixDecomposition;

/// <summary>
/// Implements Non-negative Matrix Factorization (NMF) for matrices with non-negative elements.
/// </summary>
/// <typeparam name="T">The numeric type used in the matrix.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> NMF is a way to break down a matrix containing only non-negative values
/// (zero or positive numbers) into two simpler matrices W and H, where V ~= W * H.
/// Think of it like finding hidden patterns or features in your data.
/// </para>
/// <para>
/// For example, if you have a matrix of movie ratings (all non-negative), NMF can discover:
/// - W: How much each user likes different movie genres (features)
/// - H: How much each movie belongs to different genres (features)
/// </para>
/// <para>
/// Common applications include:
/// - Topic modeling in text documents
/// - Image processing and feature extraction
/// - Collaborative filtering in recommendation systems
/// - Audio source separation
/// - Bioinformatics data analysis
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.DimensionalityReduction)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ResearchPaper("Learning the parts of objects by non-negative matrix factorization", "https://doi.org/10.1038/44565", Year = 1999, Authors = "Daniel D. Lee, H. Sebastian Seung")]
public class NmfDecomposition<T> : MatrixDecompositionBase<T>
{
    /// <summary>
    /// Gets the basis matrix W (features/components).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The W matrix represents the "basis" or "components" in your data.
    /// Each column in W represents a different pattern or feature that NMF discovered.
    /// The rows correspond to the original rows in your data.
    /// </para>
    /// <para>
    /// For example, in text analysis:
    /// - Each row might represent a document
    /// - Each column might represent a topic
    /// - The values show how much each document relates to each topic
    /// </para>
    /// </remarks>
    public Matrix<T> W { get; private set; } = new Matrix<T>(0, 0);

    /// <summary>
    /// Gets the coefficient matrix H (weights/encodings).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The H matrix represents the "coefficients" or "encodings."
    /// Each row corresponds to a feature/component from W, and each column corresponds to
    /// an original column in your data.
    /// </para>
    /// <para>
    /// For example, in text analysis:
    /// - Each row might represent a topic
    /// - Each column might represent a word
    /// - The values show how important each word is to each topic
    /// </para>
    /// </remarks>
    public Matrix<T> H { get; private set; } = new Matrix<T>(0, 0);

    /// <summary>
    /// Gets the number of components (features) used in the factorization.
    /// </summary>
    public int Components { get; private set; }

    private readonly int _maxIterations;
    private readonly double _tolerance;

    /// <summary>
    /// Initializes a new instance of the NMF decomposition for the specified matrix.
    /// </summary>
    /// <param name="matrix">The non-negative matrix to decompose.</param>
    /// <param name="components">The number of components (features) to extract. If not specified, uses min(rows, columns) / 2.</param>
    /// <param name="maxIterations">Maximum number of iterations for the algorithm.</param>
    /// <param name="tolerance">Convergence tolerance. Algorithm stops when change is below this value.</param>
    /// <exception cref="ArgumentException">Thrown if the matrix contains negative values.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the NMF decomposition. The key parameters are:
    /// - <paramref name="components"/>: How many hidden patterns/features to find. Fewer components means more compression but less detail.
    /// - <paramref name="maxIterations"/>: How long to search for the best solution.
    /// - <paramref name="tolerance"/>: How close to the perfect solution is "good enough."
    /// </para>
    /// </remarks>
    public NmfDecomposition(Matrix<T> matrix, int? components = null, int maxIterations = 200, double tolerance = 1e-4)
        : base(matrix)
    {
        // Validate that all elements are non-negative
        ValidateMatrix(matrix, requireNonNegative: true);

        Components = components ?? Math.Max(1, Math.Min(matrix.Rows, matrix.Columns) / 2);

        if (Components <= 0 || Components > Math.Min(matrix.Rows, matrix.Columns))
        {
            throw new ArgumentException($"Number of components must be between 1 and {Math.Min(matrix.Rows, matrix.Columns)}.");
        }

        _maxIterations = maxIterations;
        _tolerance = tolerance;
        Decompose();
    }

    /// <summary>
    /// Performs the NMF decomposition.
    /// </summary>
    protected override void Decompose()
    {
        (this.W, this.H) = ComputeNmf(A, Components, _maxIterations, _tolerance);
    }

    /// <summary>
    /// Computes the Non-negative Matrix Factorization using multiplicative update rules.
    /// </summary>
    /// <param name="V">The input matrix to factorize (V ~= W * H).</param>
    /// <param name="k">Number of components.</param>
    /// <param name="maxIterations">Maximum iterations.</param>
    /// <param name="tolerance">Convergence tolerance.</param>
    /// <returns>A tuple containing the W and H matrices.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method uses an iterative algorithm called "multiplicative update rules."
    /// It starts with random guesses for W and H, then repeatedly improves them until the product W * H
    /// closely approximates the original matrix V.
    /// </para>
    /// <para>
    /// The algorithm works like refining a sketch:
    /// 1. Start with a rough approximation
    /// 2. Compare it to the original
    /// 3. Adjust W and H to reduce the difference
    /// 4. Repeat until the approximation is good enough
    /// </para>
    /// </remarks>
    private (Matrix<T> W, Matrix<T> H) ComputeNmf(Matrix<T> V, int k, int maxIterations, double tolerance)
    {
        int m = V.Rows;
        int n = V.Columns;

        // Validate that T supports floating-point operations (NMF requires small positive epsilon values)
        T testEpsilon = NumOps.FromDouble(1e-10);
        if (NumOps.Equals(testEpsilon, NumOps.Zero))
        {
            throw new ArgumentException(
                "NMF decomposition requires a floating-point INumericOperations<T> implementation. " +
                "Integer-based numeric types cannot represent the small positive values required by the algorithm.");
        }

        // Scale initialization to data magnitude to avoid zero-locking
        T vSum = Engine.TensorSum(Tensor<T>.FromMatrix(V));
        T vMean = NumOps.Divide(vSum, NumOps.FromDouble((double)(m * n)));
        T initScale = NumOps.Sqrt(NumOps.Divide(NumOps.Add(vMean, NumOps.FromDouble(1e-10)), NumOps.FromDouble((double)k)));

        // DETERMINISTIC START (NNDSVD), THEN RANDOM RESTARTS ONLY AS A FALLBACK.
        //
        // Initialization was five restarts from RandomHelper.CreateSecureRandom(), which is
        // unseeded by construction, so the factorization differed on every run and the quality of
        // the result rode on the draw. NNDSVD (Boutsidis & Gallopoulos 2008, "SVD based
        // initialization: A head start for nonnegative matrix factorization") derives the starting
        // point from the data's own singular vectors instead (scikit-learn's nndsvda default does
        // the same). The random restarts are kept as a fallback for inputs whose SVD cannot be formed.
        Matrix<T> bestW = new Matrix<T>(0, 0);
        Matrix<T> bestH = new Matrix<T>(0, 0);
        T bestError = NumOps.MaxValue;

        if (TryInitializeNndsvd(V, m, n, k, out var seedW, out var seedH))
        {
            var (deterministicW, deterministicH) =
                RunNmfTrialFrom(V, seedW, seedH, m, n, k, maxIterations, tolerance);
            bestW = deterministicW;
            bestH = deterministicH;
        }

        if (bestW.Rows == 0)
        {
            const int nRestarts = 5;
            for (int restart = 0; restart < nRestarts; restart++)
            {
                var (trialW, trialH) = RunNmfTrial(V, m, n, k, maxIterations, tolerance, initScale);
                T trialError = ComputeReconstructionError(V, trialW, trialH);

                if (bestW.Rows == 0 || AnyBelow(Tensor<T>.FromScalar(trialError), bestError))
                {
                    bestW = trialW;
                    bestH = trialH;
                    bestError = trialError;
                }
            }
        }

        return (bestW, bestH);
    }

    /// <summary>
    /// Builds a deterministic nonnegative starting point from the data's singular vectors (NNDSVD).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Boutsidis and Gallopoulos, "SVD based initialization: A head start for nonnegative matrix
    /// factorization" (Pattern Recognition, 2008). The leading singular triplet is nonnegative up to
    /// sign, so it seeds the first factor directly. Each later triplet is split into its positive
    /// and negative parts, and whichever side carries more energy seeds that component -- which is
    /// how a nonnegative start is derived from vectors that are not themselves nonnegative.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> instead of guessing the starting factors at random, this reads the
    /// strongest patterns already present in the data and starts from those. The result no longer
    /// changes from run to run, and it usually starts much closer to the answer.
    /// </para>
    /// </remarks>
    /// <returns><c>false</c> when the SVD cannot be formed, leaving the caller its random fallback.</returns>
    private bool TryInitializeNndsvd(
        Matrix<T> V, int m, int n, int k, out Matrix<T> initialW, out Matrix<T> initialH)
    {
        // NNDSVD: each component starts from the dominant non-negative part of its singular pair,
        // scaled by sqrt(sigma). Any SVD failure falls back to random initialization.
        initialW = new Matrix<T>(0, 0);
        initialH = new Matrix<T>(0, 0);

        SvdDecomposition<T> svd;
        try
        {
            svd = new SvdDecomposition<T>(V);
        }
        catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or NotSupportedException)
        {
            return false;
        }

        Matrix<T> u = svd.U;
        Vector<T> singular = svd.S;
        Matrix<T> vt = svd.Vt;
        if (u.Rows != m || vt.Columns != n || singular.Length == 0) return false;

        // Floor keeps every entry strictly positive: a zero entry is a fixed point of the
        // multiplicative update and would never recover.
        T floor = NumOps.FromDouble(1e-8);
        int used = Math.Min(k, singular.Length);

        var uCols = Engine.TensorSlice(Tensor<T>.FromMatrix(u), new[] { 0, 0 }, new[] { m, used });
        var vRows = Engine.TensorSlice(Tensor<T>.FromMatrix(vt), new[] { 0, 0 }, new[] { used, n });
        var sqrtSigma = Engine.TensorSqrt(Engine.TensorAbs(
            Engine.TensorSlice(Tensor<T>.FromVector(singular), new[] { 0 }, new[] { used })));

        // Positive and negative parts, and their norms per component (squares of entries <= 0 add
        // nothing to the positive norm and vice versa).
        var uPos = Engine.ReLU(uCols);
        var uNeg = Engine.ReLU(Engine.TensorNegate(uCols));
        var vPos = Engine.ReLU(vRows);
        var vNeg = Engine.ReLU(Engine.TensorNegate(vRows));
        var positiveMass = Engine.TensorMultiply(Engine.TensorNorm(uPos, 0), Engine.TensorNorm(vPos, 1));
        var negativeMass = Engine.TensorMultiply(Engine.TensorNorm(uNeg, 0), Engine.TensorNorm(vNeg, 1));

        // Use the positive part where its mass is at least the negative part's; component 0 (the
        // Perron vector of a non-negative matrix) uses |u|, |v| directly.
        var usePositive = Engine.TensorSubtract(Engine.TensorOnesLike(positiveMass), Engine.TensorGreaterThan(negativeMass, positiveMass));
        var isFirst = Engine.TensorSetSlice(
            Engine.TensorFullLike(usePositive, NumOps.Zero), Engine.TensorFullLike(new Tensor<T>(new[] { 1 }), NumOps.One), new[] { 0 });

        var useU = Engine.TensorBroadcastTo(usePositive.Reshape(new[] { 1, used }), new[] { m, used });
        var firstU = Engine.TensorBroadcastTo(isFirst.Reshape(new[] { 1, used }), new[] { m, used });
        var partU = Engine.TensorWhere(firstU, Engine.TensorAbs(uCols), Engine.TensorWhere(useU, uPos, uNeg));

        var useV = Engine.TensorBroadcastTo(usePositive.Reshape(new[] { used, 1 }), new[] { used, n });
        var firstV = Engine.TensorBroadcastTo(isFirst.Reshape(new[] { used, 1 }), new[] { used, n });
        var partV = Engine.TensorWhere(firstV, Engine.TensorAbs(vRows), Engine.TensorWhere(useV, vPos, vNeg));

        var w = Engine.TensorMax(Engine.TensorMultiply(partU,
            Engine.TensorBroadcastTo(sqrtSigma.Reshape(new[] { 1, used }), new[] { m, used })), floor);
        var h = Engine.TensorMax(Engine.TensorMultiply(partV,
            Engine.TensorBroadcastTo(sqrtSigma.Reshape(new[] { used, 1 }), new[] { used, n })), floor);

        // Components beyond the available singular values start at the floor.
        initialW = Engine.TensorSetSlice(Engine.TensorFullLike(new Tensor<T>(new[] { m, k }), floor), w, new[] { 0, 0 }).ToMatrix();
        initialH = Engine.TensorSetSlice(Engine.TensorFullLike(new Tensor<T>(new[] { k, n }), floor), h, new[] { 0, 0 }).ToMatrix();
        return true;
    }

    /// <summary>
    /// Runs a single trial of NMF with random initialization.
    /// </summary>
    private (Matrix<T> W, Matrix<T> H) RunNmfTrial(Matrix<T> V, int m, int n, int k, int maxIterations, double tolerance, T initScale)
        => RunNmfTrialFrom(V, InitializeRandomMatrix(m, k, initScale), InitializeRandomMatrix(k, n, initScale),
                           m, n, k, maxIterations, tolerance);

    /// <summary>Runs the multiplicative-update loop from a supplied starting pair.</summary>
    private (Matrix<T> W, Matrix<T> H) RunNmfTrialFrom(Matrix<T> V, Matrix<T> startW, Matrix<T> startH, int m, int n, int k, int maxIterations, double tolerance)
    {
        var v = Tensor<T>.FromMatrix(V);
        var w = Tensor<T>.FromMatrix(startW);
        var h = Tensor<T>.FromMatrix(startH);

        var previousError = Tensor<T>.FromScalar(NumOps.MaxValue);
        T toleranceT = NumOps.FromDouble(tolerance);
        T epsilon = NumOps.FromDouble(1e-10);

        for (int iteration = 0; iteration < maxIterations; iteration++)
        {
            // Lee-Seung multiplicative updates, epsilon-stabilized:
            // H <- H .* (W^T V + eps) ./ (W^T W H + eps)
            var wT = Engine.TensorTranspose(w);
            var numeratorH = Engine.TensorAddScalar(Engine.TensorMatMul(wT, v), epsilon);
            var denominatorH = Engine.TensorAddScalar(Engine.TensorMatMul(Engine.TensorMatMul(wT, w), h), epsilon);
            h = Engine.TensorMultiply(h, Engine.TensorDivide(numeratorH, denominatorH));

            // W <- W .* (V H^T + eps) ./ (W H H^T + eps)
            var hT = Engine.TensorTranspose(h);
            var numeratorW = Engine.TensorAddScalar(Engine.TensorMatMul(v, hT), epsilon);
            var denominatorW = Engine.TensorAddScalar(Engine.TensorMatMul(Engine.TensorMatMul(w, h), hT), epsilon);
            w = Engine.TensorMultiply(w, Engine.TensorDivide(numeratorW, denominatorW));

            // Check convergence every 10 iterations
            if (iteration % 10 == 0)
            {
                var error = Engine.TensorNorm(Engine.TensorSubtract(v, Engine.TensorMatMul(w, h)));
                var errorChange = Engine.TensorAbs(Engine.TensorSubtract(previousError, error.Reshape(previousError.Shape.ToArray())));

                if (AnyBelow(errorChange, toleranceT))
                {
                    break;
                }

                previousError = error.Reshape(previousError.Shape.ToArray());
            }
        }

        return (w.ToMatrix(), h.ToMatrix());
    }

    /// <summary>
    /// Initializes a matrix with random positive values scaled to the data magnitude.
    /// </summary>
    /// <param name="rows">Number of rows.</param>
    /// <param name="cols">Number of columns.</param>
    /// <param name="scale">Scale factor derived from the data (typically sqrt(mean(V)/k)).</param>
    /// <returns>A randomly initialized matrix.</returns>
    private Matrix<T> InitializeRandomMatrix(int rows, int cols, T scale)
    {
        // Entries scale * (0.1 + U[0, 1)) are strictly positive, as multiplicative updates require.
        var random = RandomHelper.CreateSecureRandom();
        var uniform = Tensor<T>.FromMatrix(Matrix<T>.CreateRandom(random, rows, cols));
        return Engine.TensorMultiplyScalar(Engine.TensorAddScalar(uniform, NumOps.FromDouble(0.1)), scale).ToMatrix();
    }

    /// <summary>
    /// Computes the Frobenius norm of the reconstruction error ||V - W * H||.
    /// </summary>
    /// <param name="V">Original matrix.</param>
    /// <param name="W">Basis matrix.</param>
    /// <param name="H">Coefficient matrix.</param>
    /// <returns>The reconstruction error.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The reconstruction error measures how well W * H approximates the original matrix V.
    /// A smaller error means a better approximation. The Frobenius norm is like calculating the "distance"
    /// between two matrices - it's the square root of the sum of all squared differences between corresponding elements.
    /// </para>
    /// </remarks>
    private T ComputeReconstructionError(Matrix<T> V, Matrix<T> basisMatrix, Matrix<T> activationMatrix)
    {
        Matrix<T> WH = basisMatrix.Multiply(activationMatrix);
        // VECTORIZED: Use the inherited FrobeniusNorm method from base class
        Matrix<T> difference = V.Subtract(WH);
        return FrobeniusNorm(difference);
    }

    /// <summary>
    /// Solves a linear system Ax = b using the NMF decomposition.
    /// </summary>
    /// <param name="b">The right-hand side vector of the equation.</param>
    /// <returns>The solution vector x.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method finds an approximate solution to Ax = b using the NMF factorization.
    /// Since NMF provides an approximation A ~= W * H, we solve the system in two steps:
    /// 1. Solve W * y = b for y (using least squares)
    /// 2. Solve H * x = y for x (using least squares)
    /// </para>
    /// <para>
    /// Note that this gives an approximate solution since NMF itself is an approximation.
    /// For exact solutions, consider using other decomposition methods like LU or QR.
    /// </para>
    /// </remarks>
    public override Vector<T> Solve(Vector<T> b)
    {
        // Since A ~= W * H, we solve W * H * x = b
        // First solve W * y = b for y using least squares
        // Then solve H * x = y for x using least squares

        // Solve W * y = b using least squares: y = (W^T * tempW)⁻¹ * W^T * b
        Matrix<T> WT = W.Transpose();
        Matrix<T> WTW = WT.Multiply(W);
        Vector<T> WTb = WT.Multiply(b);

        // Use simple Gaussian elimination for small systems
        Vector<T> y = SolveLinearSystem(WTW, WTb);

        // Solve H * x = y using least squares: x = (H^T * H)⁻¹ * H^T * y
        Matrix<T> HT = H.Transpose();
        Matrix<T> HTH = HT.Multiply(H);
        Vector<T> HTy = HT.Multiply(y);

        Vector<T> x = SolveLinearSystem(HTH, HTy);

        return x;
    }

    /// <summary>
    /// Solves a linear system using Gaussian elimination with partial pivoting.
    /// </summary>
    /// <param name="A">Coefficient matrix.</param>
    /// <param name="b">Right-hand side vector.</param>
    /// <returns>Solution vector.</returns>
    private Vector<T> SolveLinearSystem(Matrix<T> A, Vector<T> b)
    {
        int n = A.Rows;
        if (n == 0)
        {
            return new Vector<T>(0);
        }

        // Gaussian elimination with partial pivoting as one engine solve. A singular normal matrix
        // (rank-deficient factor) falls back to the minimum-norm least-squares solution.
        var a = Tensor<T>.FromMatrix(A);
        var rhs = Tensor<T>.FromVector(b, new[] { n, 1 });
        var (solution, info) = Linalg.SolveEx(a, rhs);
        if (Engine.TensorCountNonzero(info) > 0)
        {
            solution = Engine.TensorMatMul(Linalg.Pinv(a), rhs);
        }

        return solution.Reshape(new[] { n }).ToVector();
    }

    /// <summary>
    /// Reconstructs the original matrix from the factorization.
    /// </summary>
    /// <returns>The reconstructed matrix W * H.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method multiplies W and H back together to reconstruct an approximation
    /// of the original matrix. The result should be close to the original matrix, but won't be exactly
    /// the same due to the dimensionality reduction performed by NMF.
    /// </para>
    /// <para>
    /// You can compare this to the original matrix to see how good the factorization is. The closer
    /// the reconstruction is to the original, the better the factorization captured the essential patterns.
    /// </para>
    /// </remarks>
    public Matrix<T> Reconstruct()
    {
        return W.Multiply(H);
    }
}
