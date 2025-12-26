using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Non-negative Matrix Factorization for dimensionality reduction.
/// </summary>
/// <remarks>
/// <para>
/// NMF factorizes a non-negative matrix V into two non-negative matrices W and H
/// such that V ≈ W × H. This is useful for data that is naturally non-negative
/// (e.g., images, text term frequencies, audio spectrograms).
/// </para>
/// <para>
/// Unlike PCA, NMF produces parts-based representations where each component
/// represents an additive combination of non-negative features.
/// </para>
/// <para><b>For Beginners:</b> NMF learns "building blocks" from your data:
/// - For images: NMF might learn eyes, noses, mouths as separate components
/// - For text: NMF might learn topics as combinations of words
/// - Unlike PCA, components are always positive (additive, never subtractive)
///
/// Think of it as finding the parts that, when added together, recreate your data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class NMF<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly NMFInit _init;
    private readonly NMFSolver _solver;
    private readonly double _beta;
    private readonly double _alpha;
    private readonly double _l1Ratio;
    private readonly int _maxIterations;
    private readonly double _tolerance;
    private readonly int _randomState;

    // Fitted parameters
    private double[,]? _components; // H matrix (nComponents × nFeatures)
    private double _reconstructionError;
    private int _nIterations;
    private int _nFeaturesIn;

    /// <summary>
    /// Gets the number of components.
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Gets the solver type.
    /// </summary>
    public NMFSolver Solver => _solver;

    /// <summary>
    /// Gets the learned components (H matrix).
    /// </summary>
    public double[,]? Components => _components;

    /// <summary>
    /// Gets the final reconstruction error.
    /// </summary>
    public double ReconstructionError => _reconstructionError;

    /// <summary>
    /// Gets the number of iterations performed.
    /// </summary>
    public int NIterations => _nIterations;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Creates a new instance of <see cref="NMF{T}"/>.
    /// </summary>
    /// <param name="nComponents">Number of components. Defaults to 2.</param>
    /// <param name="init">Initialization method. Defaults to Random.</param>
    /// <param name="solver">Solver algorithm. Defaults to MU (multiplicative update).</param>
    /// <param name="beta">Beta parameter for beta-divergence. Defaults to 2 (Frobenius norm).</param>
    /// <param name="alpha">Regularization parameter. Defaults to 0.</param>
    /// <param name="l1Ratio">L1 ratio in regularization. Defaults to 0.</param>
    /// <param name="maxIterations">Maximum iterations. Defaults to 200.</param>
    /// <param name="tolerance">Convergence tolerance. Defaults to 1e-4.</param>
    /// <param name="randomState">Random seed. Defaults to 0.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public NMF(
        int nComponents = 2,
        NMFInit init = NMFInit.Random,
        NMFSolver solver = NMFSolver.MU,
        double beta = 2.0,
        double alpha = 0.0,
        double l1Ratio = 0.0,
        int maxIterations = 200,
        double tolerance = 1e-4,
        int randomState = 0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nComponents < 1)
        {
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));
        }

        _nComponents = nComponents;
        _init = init;
        _solver = solver;
        _beta = beta;
        _alpha = alpha;
        _l1Ratio = l1Ratio;
        _maxIterations = maxIterations;
        _tolerance = tolerance;
        _randomState = randomState;
    }

    /// <summary>
    /// Fits NMF by learning the components matrix.
    /// </summary>
    /// <param name="data">The training data matrix (must be non-negative).</param>
    protected override void FitCore(Matrix<T> data)
    {
        _nFeaturesIn = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        int k = _nComponents;

        // Convert to double and verify non-negativity
        var V = new double[n, p];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                if (val < 0)
                {
                    throw new ArgumentException(
                        $"NMF requires non-negative input. Found negative value at ({i}, {j}): {val}");
                }
                V[i, j] = val;
            }
        }

        // Initialize W and H
        var (W, H) = InitializeMatrices(V, n, p, k);

        // Run solver
        if (_solver == NMFSolver.MU)
        {
            (W, H, _nIterations) = SolveMultiplicativeUpdate(V, W, H, n, p, k);
        }
        else
        {
            (W, H, _nIterations) = SolveCoordinateDescent(V, W, H, n, p, k);
        }

        // Store components (H matrix)
        _components = H;

        // Compute reconstruction error
        _reconstructionError = ComputeReconstructionError(V, W, H, n, p, k);
    }

    private (double[,] W, double[,] H) InitializeMatrices(double[,] V, int n, int p, int k)
    {
        var W = new double[n, k];
        var H = new double[k, p];
        var random = RandomHelper.CreateSeededRandom(_randomState);

        switch (_init)
        {
            case NMFInit.Random:
                // Random initialization
                double scale = Math.Sqrt(V.Cast<double>().Average() / k);
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < k; j++)
                    {
                        W[i, j] = Math.Abs(random.NextDouble() * scale);
                    }
                }
                for (int i = 0; i < k; i++)
                {
                    for (int j = 0; j < p; j++)
                    {
                        H[i, j] = Math.Abs(random.NextDouble() * scale);
                    }
                }
                break;

            case NMFInit.NNDSVD:
                // Non-negative Double SVD initialization
                InitializeNNDSVD(V, W, H, n, p, k, random);
                break;

            case NMFInit.NNDSVDa:
                // NNDSVD with small values set to average
                InitializeNNDSVD(V, W, H, n, p, k, random);
                double avg = V.Cast<double>().Average();
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < k; j++)
                    {
                        if (W[i, j] < 1e-10) W[i, j] = avg;
                    }
                }
                for (int i = 0; i < k; i++)
                {
                    for (int j = 0; j < p; j++)
                    {
                        if (H[i, j] < 1e-10) H[i, j] = avg;
                    }
                }
                break;
        }

        return (W, H);
    }

    private void InitializeNNDSVD(double[,] V, double[,] W, double[,] H, int n, int p, int k, Random random)
    {
        // Simplified NNDSVD: use SVD and take absolute values
        // First, compute V^T V for eigendecomposition
        var VTV = new double[p, p];
        for (int i = 0; i < p; i++)
        {
            for (int j = i; j < p; j++)
            {
                double sum = 0;
                for (int l = 0; l < n; l++)
                {
                    sum += V[l, i] * V[l, j];
                }
                VTV[i, j] = sum;
                VTV[j, i] = sum;
            }
        }

        var (eigenvalues, eigenvectors) = ComputeEigen(VTV, p, k);

        // Sort by eigenvalue descending
        var indices = Enumerable.Range(0, k)
            .OrderByDescending(i => eigenvalues[i])
            .ToArray();

        // Initialize H with absolute values of right singular vectors
        for (int i = 0; i < k; i++)
        {
            for (int j = 0; j < p; j++)
            {
                H[i, j] = Math.Abs(eigenvectors[indices[i], j]);
            }
        }

        // Initialize W by projecting V onto H
        for (int i = 0; i < n; i++)
        {
            for (int c = 0; c < k; c++)
            {
                double sum = 0;
                for (int j = 0; j < p; j++)
                {
                    sum += V[i, j] * H[c, j];
                }
                W[i, c] = Math.Max(0, sum);
            }
        }
    }

    private (double[] Eigenvalues, double[,] Eigenvectors) ComputeEigen(double[,] matrix, int n, int k)
    {
        var eigenvalues = new double[k];
        var eigenvectors = new double[k, n];
        var A = (double[,])matrix.Clone();
        var random = RandomHelper.CreateSeededRandom(_randomState);

        for (int m = 0; m < k; m++)
        {
            var v = new double[n];
            double norm = 0;
            for (int i = 0; i < n; i++)
            {
                v[i] = random.NextDouble() - 0.5;
                norm += v[i] * v[i];
            }
            norm = Math.Sqrt(norm);
            for (int i = 0; i < n; i++)
            {
                v[i] /= norm;
            }

            for (int iter = 0; iter < 50; iter++)
            {
                var Av = new double[n];
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        Av[i] += A[i, j] * v[j];
                    }
                }

                norm = 0;
                for (int i = 0; i < n; i++)
                {
                    norm += Av[i] * Av[i];
                }
                norm = Math.Sqrt(norm);

                if (norm < 1e-10)
                {
                    // Av is nearly zero - keep current normalized v as eigenvector
                    // Ensure v is still normalized (it should be from previous iteration or init)
                    double vNorm = 0;
                    for (int i = 0; i < n; i++)
                    {
                        vNorm += v[i] * v[i];
                    }
                    vNorm = Math.Sqrt(vNorm);
                    if (vNorm > 1e-10)
                    {
                        for (int i = 0; i < n; i++)
                        {
                            v[i] /= vNorm;
                        }
                    }
                    break;
                }

                for (int i = 0; i < n; i++)
                {
                    v[i] = Av[i] / norm;
                }
            }

            double eigenvalue = 0;
            var Av2 = new double[n];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    Av2[i] += A[i, j] * v[j];
                }
                eigenvalue += v[i] * Av2[i];
            }

            eigenvalues[m] = Math.Max(0, eigenvalue);
            for (int i = 0; i < n; i++)
            {
                eigenvectors[m, i] = v[i];
            }

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    A[i, j] -= eigenvalue * v[i] * v[j];
                }
            }
        }

        return (eigenvalues, eigenvectors);
    }

    private (double[,] W, double[,] H, int Iterations) SolveMultiplicativeUpdate(
        double[,] V, double[,] W, double[,] H, int n, int p, int k)
    {
        double eps = 1e-10;
        double prevError = double.MaxValue;

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            // Compute WH (reconstruction)
            var WH = new double[n, p];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < p; j++)
                {
                    double sum = 0;
                    for (int c = 0; c < k; c++)
                    {
                        sum += W[i, c] * H[c, j];
                    }
                    WH[i, j] = Math.Max(eps, sum);
                }
            }

            // Beta-divergence multiplicative update
            // For beta = 2 (Frobenius): standard update
            // For beta = 1 (KL-divergence): special case
            // For beta = 0 (Itakura-Saito): special case
            // General: uses (WH)^(beta-2) and (WH)^(beta-1)

            // Update H: H *= (W^T * ((WH)^(beta-2) * V)) / (W^T * (WH)^(beta-1) + reg)
            var numeratorH = new double[k, p];
            var denominatorH = new double[k, p];

            for (int i = 0; i < k; i++)
            {
                for (int j = 0; j < p; j++)
                {
                    double numSum = 0;
                    double denSum = 0;
                    for (int l = 0; l < n; l++)
                    {
                        double whPowBetaMinus2 = Math.Pow(WH[l, j], _beta - 2);
                        double whPowBetaMinus1 = Math.Pow(WH[l, j], _beta - 1);
                        numSum += W[l, i] * whPowBetaMinus2 * V[l, j];
                        denSum += W[l, i] * whPowBetaMinus1;
                    }
                    numeratorH[i, j] = numSum;
                    // Add L1/L2 regularization
                    double l1Reg = _alpha * _l1Ratio;
                    double l2Reg = _alpha * (1 - _l1Ratio) * H[i, j];
                    denominatorH[i, j] = denSum + l1Reg + l2Reg + eps;
                }
            }

            // Update H
            for (int i = 0; i < k; i++)
            {
                for (int j = 0; j < p; j++)
                {
                    H[i, j] *= numeratorH[i, j] / denominatorH[i, j];
                    H[i, j] = Math.Max(eps, H[i, j]);
                }
            }

            // Recompute WH after H update
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < p; j++)
                {
                    double sum = 0;
                    for (int c = 0; c < k; c++)
                    {
                        sum += W[i, c] * H[c, j];
                    }
                    WH[i, j] = Math.Max(eps, sum);
                }
            }

            // Update W: W *= (((WH)^(beta-2) * V) * H^T) / ((WH)^(beta-1) * H^T + reg)
            var numeratorW = new double[n, k];
            var denominatorW = new double[n, k];

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < k; j++)
                {
                    double numSum = 0;
                    double denSum = 0;
                    for (int l = 0; l < p; l++)
                    {
                        double whPowBetaMinus2 = Math.Pow(WH[i, l], _beta - 2);
                        double whPowBetaMinus1 = Math.Pow(WH[i, l], _beta - 1);
                        numSum += whPowBetaMinus2 * V[i, l] * H[j, l];
                        denSum += whPowBetaMinus1 * H[j, l];
                    }
                    numeratorW[i, j] = numSum;
                    // Add L1/L2 regularization
                    double l1Reg = _alpha * _l1Ratio;
                    double l2Reg = _alpha * (1 - _l1Ratio) * W[i, j];
                    denominatorW[i, j] = denSum + l1Reg + l2Reg + eps;
                }
            }

            // Update W
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < k; j++)
                {
                    W[i, j] *= numeratorW[i, j] / denominatorW[i, j];
                    W[i, j] = Math.Max(eps, W[i, j]);
                }
            }

            // Check convergence
            double error = ComputeReconstructionError(V, W, H, n, p, k);
            double relChange = Math.Abs(prevError - error) / (prevError + eps);

            if (relChange < _tolerance)
            {
                return (W, H, iter + 1);
            }

            prevError = error;
        }

        return (W, H, _maxIterations);
    }

    private (double[,] W, double[,] H, int Iterations) SolveCoordinateDescent(
        double[,] V, double[,] W, double[,] H, int n, int p, int k)
    {
        // Simplified coordinate descent
        double eps = 1e-10;
        double prevError = double.MaxValue;

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            // Update H by minimizing ||V - WH||^2 + regularization with H >= 0
            for (int c = 0; c < k; c++)
            {
                for (int j = 0; j < p; j++)
                {
                    double numerator = 0;
                    double denominator = eps;

                    for (int i = 0; i < n; i++)
                    {
                        // Residual without current component
                        double residual = V[i, j];
                        for (int l = 0; l < k; l++)
                        {
                            if (l != c)
                            {
                                residual -= W[i, l] * H[l, j];
                            }
                        }
                        numerator += W[i, c] * residual;
                        denominator += W[i, c] * W[i, c];
                    }

                    // L2 regularization: add to denominator
                    denominator += _alpha * (1 - _l1Ratio);

                    // L1 regularization: soft-thresholding on numerator
                    double l1Penalty = _alpha * _l1Ratio;
                    double thresholdedNum = Math.Max(0, numerator - l1Penalty) -
                                            Math.Max(0, -numerator - l1Penalty);

                    H[c, j] = Math.Max(0, thresholdedNum / denominator);
                }
            }

            // Update W by minimizing ||V - WH||^2 + regularization with W >= 0
            for (int i = 0; i < n; i++)
            {
                for (int c = 0; c < k; c++)
                {
                    double numerator = 0;
                    double denominator = eps;

                    for (int j = 0; j < p; j++)
                    {
                        double residual = V[i, j];
                        for (int l = 0; l < k; l++)
                        {
                            if (l != c)
                            {
                                residual -= W[i, l] * H[l, j];
                            }
                        }
                        numerator += H[c, j] * residual;
                        denominator += H[c, j] * H[c, j];
                    }

                    // L2 regularization: add to denominator
                    denominator += _alpha * (1 - _l1Ratio);

                    // L1 regularization: soft-thresholding on numerator
                    double l1Penalty = _alpha * _l1Ratio;
                    double thresholdedNum = Math.Max(0, numerator - l1Penalty) -
                                            Math.Max(0, -numerator - l1Penalty);

                    W[i, c] = Math.Max(0, thresholdedNum / denominator);
                }
            }

            // Check convergence
            double error = ComputeReconstructionError(V, W, H, n, p, k);
            double relChange = Math.Abs(prevError - error) / (prevError + eps);

            if (relChange < _tolerance)
            {
                return (W, H, iter + 1);
            }

            prevError = error;
        }

        return (W, H, _maxIterations);
    }

    private double ComputeReconstructionError(double[,] V, double[,] W, double[,] H, int n, int p, int k)
    {
        double error = 0;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                double WH_ij = 0;
                for (int c = 0; c < k; c++)
                {
                    WH_ij += W[i, c] * H[c, j];
                }
                double diff = V[i, j] - WH_ij;
                error += diff * diff;
            }
        }
        return Math.Sqrt(error);
    }

    /// <summary>
    /// Transforms data by finding the optimal W given learned H.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The coefficient matrix W.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_components is null)
        {
            throw new InvalidOperationException("NMF has not been fitted.");
        }

        int n = data.Rows;
        int p = data.Columns;
        int k = _components.GetLength(0);

        // Validate feature count matches training data
        if (p != _nFeaturesIn)
        {
            throw new ArgumentException(
                $"Input data has {p} features, but model was fitted with {_nFeaturesIn} features.",
                nameof(data));
        }

        // Convert to double, enforcing non-negativity (required for NMF)
        var V = new double[n, p];
        bool hasNegatives = false;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                if (val < 0)
                {
                    hasNegatives = true;
                    val = 0;
                }
                V[i, j] = val;
            }
        }

        if (hasNegatives)
        {
            throw new ArgumentException(
                "NMF requires non-negative input data. Negative values were detected.",
                nameof(data));
        }

        // Initialize W
        var W = new double[n, k];
        double scale = Math.Sqrt(V.Cast<double>().Average() / k);
        var random = RandomHelper.CreateSeededRandom(_randomState);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < k; j++)
            {
                W[i, j] = Math.Abs(random.NextDouble() * scale);
            }
        }

        // Solve for W with fixed H
        double eps = 1e-10;
        for (int iter = 0; iter < 100; iter++)
        {
            // W *= (V H^T) / (W H H^T + eps)
            var VHT = new double[n, k];
            var HHT = new double[k, k];
            var WHHT = new double[n, k];

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < k; j++)
                {
                    double sum = 0;
                    for (int l = 0; l < p; l++)
                    {
                        sum += V[i, l] * _components[j, l];
                    }
                    VHT[i, j] = sum;
                }
            }

            for (int i = 0; i < k; i++)
            {
                for (int j = 0; j < k; j++)
                {
                    double sum = 0;
                    for (int l = 0; l < p; l++)
                    {
                        sum += _components[i, l] * _components[j, l];
                    }
                    HHT[i, j] = sum;
                }
            }

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < k; j++)
                {
                    double sum = 0;
                    for (int l = 0; l < k; l++)
                    {
                        sum += W[i, l] * HHT[l, j];
                    }
                    WHHT[i, j] = sum + eps;
                }
            }

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < k; j++)
                {
                    W[i, j] *= VHT[i, j] / WHHT[i, j];
                    W[i, j] = Math.Max(eps, W[i, j]);
                }
            }
        }

        // Convert to output
        var result = new T[n, k];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < k; j++)
            {
                result[i, j] = NumOps.FromDouble(W[i, j]);
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Reconstructs data from coefficient matrix.
    /// </summary>
    /// <param name="data">The coefficient matrix W.</param>
    /// <returns>Reconstructed data in original space.</returns>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_components is null)
        {
            throw new InvalidOperationException("NMF has not been fitted.");
        }

        int n = data.Rows;
        int inputColumns = data.Columns;
        int p = _nFeaturesIn;
        int k = _components.GetLength(0);

        // Validate input column count matches number of components
        if (inputColumns != k)
        {
            throw new ArgumentException(
                $"Input data has {inputColumns} columns, but model expects coefficient matrix " +
                $"with {k} columns (number of components).",
                nameof(data));
        }

        var result = new T[n, p];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                double sum = 0;
                for (int c = 0; c < k; c++)
                {
                    sum += NumOps.ToDouble(data[i, c]) * _components[c, j];
                }
                result[i, j] = NumOps.FromDouble(sum);
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        int k = _components?.GetLength(0) ?? _nComponents;
        var names = new string[k];
        for (int i = 0; i < k; i++)
        {
            names[i] = $"Component{i + 1}";
        }
        return names;
    }
}

/// <summary>
/// Specifies the initialization method for NMF.
/// </summary>
public enum NMFInit
{
    /// <summary>
    /// Random non-negative initialization.
    /// </summary>
    Random,

    /// <summary>
    /// Non-negative Double SVD initialization (faster convergence).
    /// </summary>
    NNDSVD,

    /// <summary>
    /// NNDSVD with zeros filled with small values (better for sparse data).
    /// </summary>
    NNDSVDa
}

/// <summary>
/// Specifies the solver for NMF optimization.
/// </summary>
public enum NMFSolver
{
    /// <summary>
    /// Multiplicative Update solver (Lee and Seung).
    /// </summary>
    MU,

    /// <summary>
    /// Coordinate Descent solver.
    /// </summary>
    CD
}
