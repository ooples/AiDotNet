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

        // Validate that numeric type supports small positive values (required for NMF algorithm)
        T testEpsilon = NumOps.FromDouble(1e-10);
        if (NumOps.Equals(testEpsilon, NumOps.Zero))
        {
            throw new ArgumentException(
                "NMF decomposition requires a floating-point INumericOperations<T> implementation. " +
                "Integer-based numeric types cannot represent the small positive values required by the algorithm.");
        }

        // Scale initialization to data magnitude to avoid zero-locking
        T vSum = NumOps.Zero;
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                vSum = NumOps.Add(vSum, V[i, j]);
        T vMean = NumOps.Divide(vSum, NumOps.FromDouble((double)(m * n)));
        T initScale = NumOps.Sqrt(NumOps.Divide(NumOps.Add(vMean, NumOps.FromDouble(1e-10)), NumOps.FromDouble((double)k)));

        // Use multiple random restarts to escape degenerate local optima
        // (multiplicative updates can get trapped in zero-locked states)
        Matrix<T> bestW = new Matrix<T>(0, 0);
        Matrix<T> bestH = new Matrix<T>(0, 0);
        T bestError = NumOps.FromDouble(double.MaxValue);
        int nRestarts = 5;

        for (int restart = 0; restart < nRestarts; restart++)
        {
            var (trialW, trialH) = RunNmfTrial(V, m, n, k, maxIterations, tolerance, initScale);
            T trialError = ComputeReconstructionError(V, trialW, trialH);

            if (NumOps.LessThan(trialError, bestError))
            {
                bestW = trialW;
                bestH = trialH;
                bestError = trialError;
            }
        }

        return (bestW, bestH);
    }

    /// <summary>
    /// Runs a single trial of NMF with random initialization.
    /// </summary>
    private (Matrix<T> W, Matrix<T> H) RunNmfTrial(Matrix<T> V, int m, int n, int k, int maxIterations, double tolerance, T initScale)
    {
        Matrix<T> tempW = InitializeRandomMatrix(m, k, initScale);
        Matrix<T> tempH = InitializeRandomMatrix(k, n, initScale);

        T previousError = NumOps.FromDouble(double.MaxValue);
        T toleranceT = NumOps.FromDouble(tolerance);
        T epsilon = NumOps.FromDouble(1e-10);

        for (int iteration = 0; iteration < maxIterations; iteration++)
        {
            // Update H: H = H .* (W^T * V + eps) ./ (W^T * W * H + eps)
            Matrix<T> WT = tempW.Transpose();
            Matrix<T> WTV = WT.Multiply(V);
            Matrix<T> WTW = WT.Multiply(tempW);
            Matrix<T> WTWH = WTW.Multiply(tempH);

            // VECTORIZED: Process each row as a vector using Engine operations
            var epsilonVec = new Vector<T>(n);
            for (int idx = 0; idx < n; idx++) epsilonVec[idx] = epsilon;

            for (int i = 0; i < k; i++)
            {
                Vector<T> numeratorH = WTV.GetRow(i);
                Vector<T> wtwhRow = WTWH.GetRow(i);
                Vector<T> tempHRow = tempH.GetRow(i);

                var stabilizedNumerator = (Vector<T>)Engine.Add(numeratorH, epsilonVec);
                var denominator = (Vector<T>)Engine.Add(wtwhRow, epsilonVec);
                var ratio = (Vector<T>)Engine.Divide(stabilizedNumerator, denominator);
                var updated = (Vector<T>)Engine.Multiply(tempHRow, ratio);
                tempH.SetRow(i, updated);
            }

            // Update W: W = W .* (V * H^T + eps) ./ (W * H * H^T + eps)
            Matrix<T> HT = tempH.Transpose();
            Matrix<T> VHT = V.Multiply(HT);
            Matrix<T> WH = tempW.Multiply(tempH);
            Matrix<T> WHHT = WH.Multiply(HT);

            // VECTORIZED: Process each row as a vector using Engine operations
            var epsilonVecW = new Vector<T>(k);
            for (int idx = 0; idx < k; idx++) epsilonVecW[idx] = epsilon;

            for (int i = 0; i < m; i++)
            {
                Vector<T> numeratorW = VHT.GetRow(i);
                Vector<T> whhtRow = WHHT.GetRow(i);
                Vector<T> tempWRow = tempW.GetRow(i);

                var stabilizedNumerator = (Vector<T>)Engine.Add(numeratorW, epsilonVecW);
                var denominator = (Vector<T>)Engine.Add(whhtRow, epsilonVecW);
                var ratio = (Vector<T>)Engine.Divide(stabilizedNumerator, denominator);
                var updated = (Vector<T>)Engine.Multiply(tempWRow, ratio);
                tempW.SetRow(i, updated);
            }

            // Check for convergence
            if (iteration % 10 == 0)
            {
                T error = ComputeReconstructionError(V, tempW, tempH);
                T errorChange = NumOps.Abs(NumOps.Subtract(previousError, error));

                if (NumOps.LessThan(errorChange, toleranceT))
                {
                    break;
                }

                previousError = error;
            }
        }

        return (tempW, tempH);
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
        var random = RandomHelper.CreateSecureRandom();
        var matrix = new Matrix<T>(rows, cols);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                // Use [0.1, 1.1) * scale to avoid near-zero values that cause
                // zero-locking in multiplicative updates
                T randomValue = NumOps.FromDouble(0.1 + random.NextDouble());
                matrix[i, j] = NumOps.Multiply(scale, randomValue);
            }
        }

        return matrix;
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
        Matrix<T> augmented = new Matrix<T>(n, n + 1);

        // Create augmented matrix [A|b]
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                augmented[i, j] = A[i, j];
            }
            augmented[i, n] = b[i];
        }

        // Forward elimination with partial pivoting
        for (int k = 0; k < n - 1; k++)
        {
            // Find pivot
            int maxRow = k;
            for (int i = k + 1; i < n; i++)
            {
                if (NumOps.GreaterThan(NumOps.Abs(augmented[i, k]), NumOps.Abs(augmented[maxRow, k])))
                {
                    maxRow = i;
                }
            }

            // Swap rows
            if (maxRow != k)
            {
                for (int j = k; j <= n; j++)
                {
                    T temp = augmented[k, j];
                    augmented[k, j] = augmented[maxRow, j];
                    augmented[maxRow, j] = temp;
                }
            }

            // Eliminate column
            for (int i = k + 1; i < n; i++)
            {
                if (!NumOps.Equals(augmented[k, k], NumOps.Zero))
                {
                    T factor = NumOps.Divide(augmented[i, k], augmented[k, k]);
                    for (int j = k; j <= n; j++)
                    {
                        augmented[i, j] = NumOps.Subtract(augmented[i, j], NumOps.Multiply(factor, augmented[k, j]));
                    }
                }
            }
        }

        // Back substitution
        Vector<T> x = new Vector<T>(n);
        for (int i = n - 1; i >= 0; i--)
        {
            // VECTORIZED: Use dot product for sum computation
            T sum = NumOps.Zero;
            if (i < n - 1)
            {
                int remaining = n - i - 1;
                var rowSlice = new T[remaining];
                var xSlice = new T[remaining];
                for (int k = 0; k < remaining; k++)
                {
                    rowSlice[k] = augmented[i, i + 1 + k];
                    xSlice[k] = x[i + 1 + k];
                }
                var rowVec = new Vector<T>(rowSlice);
                var xVec = new Vector<T>(xSlice);
                sum = rowVec.DotProduct(xVec);
            }

            if (!NumOps.Equals(augmented[i, i], NumOps.Zero))
            {
                x[i] = NumOps.Divide(NumOps.Subtract(augmented[i, n], sum), augmented[i, i]);
            }
            else
            {
                x[i] = NumOps.Zero;
            }
        }

        return x;
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
