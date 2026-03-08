using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Dictionary Learning for sparse representation.
/// </summary>
/// <remarks>
/// <para>
/// Dictionary Learning finds a dictionary D and sparse codes A such that
/// X ≈ D × A. The dictionary atoms (columns of D) form a basis that allows
/// sparse representation of the data.
/// </para>
/// <para>
/// Unlike PCA which enforces orthogonality, dictionary learning allows
/// overcomplete dictionaries (more atoms than dimensions) and enforces
/// sparsity on the codes.
/// </para>
/// <para><b>For Beginners:</b> Dictionary Learning is like building a "parts library":
/// - Dictionary: Collection of basic building blocks (atoms)
/// - Sparse codes: Which parts to use and how much (mostly zeros)
/// - Goal: Represent each sample using few dictionary atoms
/// - Used for: Image denoising, compression, feature extraction
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class DictionaryLearning<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly double _alpha;
    private readonly int _maxIter;
    private readonly double _tol;
    private readonly DictionaryFitAlgorithm _fitAlgorithm;
    private readonly SparseCodingAlgorithm _transformAlgorithm;
    private readonly int? _randomState;

    // Fitted parameters
    private double[]? _mean;
    private double[,]? _components; // Dictionary atoms (n_atoms x n_features)
    private int _nFeaturesIn;

    /// <summary>
    /// Gets the number of dictionary atoms.
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Gets the sparsity penalty.
    /// </summary>
    public double Alpha => _alpha;

    /// <summary>
    /// Gets the dictionary fitting algorithm.
    /// </summary>
    public DictionaryFitAlgorithm FitAlgorithm => _fitAlgorithm;

    /// <summary>
    /// Gets the sparse coding algorithm for transform.
    /// </summary>
    public SparseCodingAlgorithm TransformAlgorithm => _transformAlgorithm;

    /// <summary>
    /// Gets the mean of each feature.
    /// </summary>
    public double[]? Mean => _mean;

    /// <summary>
    /// Gets the dictionary atoms (each row is an atom).
    /// </summary>
    public double[,]? Components => _components;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Creates a new instance of <see cref="DictionaryLearning{T}"/>.
    /// </summary>
    /// <param name="nComponents">Number of dictionary atoms. Defaults to the number of features.</param>
    /// <param name="alpha">Sparsity regularization parameter. Defaults to 1.0.</param>
    /// <param name="maxIter">Maximum iterations. Defaults to 1000.</param>
    /// <param name="tol">Convergence tolerance. Defaults to 1e-6.</param>
    /// <param name="fitAlgorithm">Algorithm for dictionary learning. Defaults to CD.</param>
    /// <param name="transformAlgorithm">Algorithm for sparse coding. Defaults to LASSO.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public DictionaryLearning(
        int? nComponents = null,
        double alpha = 1.0,
        int maxIter = 1000,
        double tol = 1e-6,
        DictionaryFitAlgorithm fitAlgorithm = DictionaryFitAlgorithm.CD,
        SparseCodingAlgorithm transformAlgorithm = SparseCodingAlgorithm.LASSO,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nComponents.HasValue && nComponents.Value < 1)
        {
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));
        }

        if (alpha < 0)
        {
            throw new ArgumentException("Alpha must be non-negative.", nameof(alpha));
        }

        _nComponents = nComponents ?? 0; // Will be set during fit if 0
        _alpha = alpha;
        _maxIter = maxIter;
        _tol = tol;
        _fitAlgorithm = fitAlgorithm;
        _transformAlgorithm = transformAlgorithm;
        _randomState = randomState;
    }

    /// <summary>
    /// Fits the dictionary using alternating minimization.
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        _nFeaturesIn = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        int k = _nComponents > 0 ? _nComponents : p;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSeededRandom(42);

        // Center the data
        _mean = new double[p];
        var centered = new double[n, p];

        for (int j = 0; j < p; j++)
        {
            double sum = 0;
            for (int i = 0; i < n; i++)
            {
                sum += NumOps.ToDouble(data[i, j]);
            }
            _mean[j] = sum / n;
        }

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                centered[i, j] = NumOps.ToDouble(data[i, j]) - _mean[j];
            }
        }

        // Initialize dictionary with random samples + noise
        _components = new double[k, p];
        var usedIndices = new HashSet<int>();

        for (int c = 0; c < k; c++)
        {
            int idx;
            if (c < n && !usedIndices.Contains(c % n))
            {
                idx = c % n;
            }
            else
            {
                idx = random.Next(n);
            }
            usedIndices.Add(idx);

            double norm = 0;
            for (int j = 0; j < p; j++)
            {
                _components[c, j] = centered[idx, j] + 0.1 * (random.NextDouble() - 0.5);
                norm += _components[c, j] * _components[c, j];
            }

            // Normalize
            norm = Math.Sqrt(norm);
            if (norm > 1e-10)
            {
                for (int j = 0; j < p; j++)
                {
                    _components[c, j] /= norm;
                }
            }
        }

        // Alternating minimization
        var codes = new double[n, k];

        for (int iter = 0; iter < _maxIter; iter++)
        {
            double[] oldComponents = new double[k * p];
            for (int c = 0; c < k; c++)
            {
                for (int j = 0; j < p; j++)
                {
                    oldComponents[c * p + j] = _components[c, j];
                }
            }

            // Step 1: Update codes (sparse coding)
            UpdateCodes(centered, codes, n, p, k);

            // Step 2: Update dictionary
            if (_fitAlgorithm == DictionaryFitAlgorithm.CD)
            {
                UpdateDictionaryCD(centered, codes, n, p, k);
            }
            else
            {
                UpdateDictionaryLeastSquares(centered, codes, n, p, k);
            }

            // Check convergence
            double maxChange = 0;
            for (int c = 0; c < k; c++)
            {
                for (int j = 0; j < p; j++)
                {
                    double change = Math.Abs(_components[c, j] - oldComponents[c * p + j]);
                    maxChange = Math.Max(maxChange, change);
                }
            }

            if (maxChange < _tol)
            {
                break;
            }
        }
    }

    private void UpdateCodes(double[,] data, double[,] codes, int n, int p, int k)
    {
        // Choose sparse coding algorithm
        if (_transformAlgorithm == SparseCodingAlgorithm.OMP)
        {
            // OMP - Orthogonal Matching Pursuit
            for (int i = 0; i < n; i++)
            {
                OMPSparseCoding(data, codes, i, p, k);
            }
        }
        else
        {
            // LASSO-like sparse coding for each sample
            for (int i = 0; i < n; i++)
            {
                SparseCodingForSample(data, codes, i, p, k);
            }
        }
    }

    private void OMPSparseCoding(double[,] data, double[,] codes, int sample, int p, int k)
    {
        // Orthogonal Matching Pursuit algorithm
        // Greedy algorithm that iteratively selects atoms to minimize residual

        // Get the signal to encode
        var x = new double[p];
        for (int j = 0; j < p; j++)
        {
            x[j] = data[sample, j];
        }

        // Initialize residual as the signal
        var residual = (double[])x.Clone();

        // Track selected atoms
        var selectedAtoms = new List<int>();
        var tolerance = 1e-6;

        // Maximum number of atoms to select (based on sparsity level)
        // Use alpha to determine sparsity - higher alpha = fewer atoms
        int maxAtoms = Math.Max(1, (int)Math.Ceiling(k / (1 + _alpha)));
        maxAtoms = Math.Min(maxAtoms, Math.Min(k, p)); // Can't select more than available

        // Initialize all codes to zero
        for (int c = 0; c < k; c++)
        {
            codes[sample, c] = 0;
        }

        // Precompute D^T * x (correlations with signal)
        var Dtx = new double[k];
        for (int c = 0; c < k; c++)
        {
            double sum = 0;
            for (int j = 0; j < p; j++)
            {
                sum += _components![c, j] * x[j];
            }
            Dtx[c] = sum;
        }

        for (int iter = 0; iter < maxAtoms; iter++)
        {
            // Step 1: Find atom most correlated with residual
            int bestAtom = -1;
            double bestCorrelation = 0;

            for (int c = 0; c < k; c++)
            {
                // Skip already selected atoms
                if (selectedAtoms.Contains(c))
                {
                    continue;
                }

                // Compute correlation with residual
                double correlation = 0;
                for (int j = 0; j < p; j++)
                {
                    correlation += _components![c, j] * residual[j];
                }

                if (Math.Abs(correlation) > Math.Abs(bestCorrelation))
                {
                    bestCorrelation = correlation;
                    bestAtom = c;
                }
            }

            if (bestAtom < 0 || Math.Abs(bestCorrelation) < tolerance)
            {
                break;
            }

            // Step 2: Add best atom to selected set
            selectedAtoms.Add(bestAtom);

            // Step 3: Solve least squares problem for selected atoms
            // x = D_selected * coefficients => coefficients = (D_selected^T * D_selected)^-1 * D_selected^T * x
            int numSelected = selectedAtoms.Count;

            // Build D_selected^T * D_selected
            var DtD = new double[numSelected, numSelected];
            for (int i = 0; i < numSelected; i++)
            {
                for (int j = 0; j < numSelected; j++)
                {
                    double sum = 0;
                    for (int f = 0; f < p; f++)
                    {
                        sum += _components![selectedAtoms[i], f] * _components[selectedAtoms[j], f];
                    }
                    DtD[i, j] = sum;
                }
                DtD[i, i] += 1e-10; // Small regularization for stability
            }

            // Build D_selected^T * x
            var DtxSelected = new double[numSelected];
            for (int i = 0; i < numSelected; i++)
            {
                DtxSelected[i] = Dtx[selectedAtoms[i]];
            }

            // Solve via Cholesky or LU decomposition
            var coefficients = SolvePositiveDefinite(DtD, DtxSelected, numSelected);

            // Step 4: Update residual
            for (int j = 0; j < p; j++)
            {
                residual[j] = x[j];
                for (int i = 0; i < numSelected; i++)
                {
                    residual[j] -= coefficients[i] * _components![selectedAtoms[i], j];
                }
            }

            // Update codes for selected atoms
            for (int i = 0; i < numSelected; i++)
            {
                codes[sample, selectedAtoms[i]] = coefficients[i];
            }

            // Check stopping criterion (residual energy)
            double residualNorm = 0;
            for (int j = 0; j < p; j++)
            {
                residualNorm += residual[j] * residual[j];
            }
            residualNorm = Math.Sqrt(residualNorm);

            double signalNorm = 0;
            for (int j = 0; j < p; j++)
            {
                signalNorm += x[j] * x[j];
            }
            signalNorm = Math.Sqrt(signalNorm);

            if (residualNorm < tolerance * signalNorm)
            {
                break;
            }
        }
    }

    private static double[] SolvePositiveDefinite(double[,] A, double[] b, int n)
    {
        // Solve A * x = b where A is positive definite
        // Using Cholesky decomposition: A = L * L^T

        // Cholesky decomposition
        var L = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                double sum = 0;
                for (int k = 0; k < j; k++)
                {
                    sum += L[i, k] * L[j, k];
                }

                if (i == j)
                {
                    double diag = A[i, i] - sum;
                    L[i, j] = diag > 0 ? Math.Sqrt(diag) : 1e-10;
                }
                else
                {
                    L[i, j] = (A[i, j] - sum) / L[j, j];
                }
            }
        }

        // Forward substitution: L * y = b
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            double sum = 0;
            for (int j = 0; j < i; j++)
            {
                sum += L[i, j] * y[j];
            }
            y[i] = (b[i] - sum) / L[i, i];
        }

        // Backward substitution: L^T * x = y
        var x = new double[n];
        for (int i = n - 1; i >= 0; i--)
        {
            double sum = 0;
            for (int j = i + 1; j < n; j++)
            {
                sum += L[j, i] * x[j];
            }
            x[i] = (y[i] - sum) / L[i, i];
        }

        return x;
    }

    private void SparseCodingForSample(double[,] data, double[,] codes, int sample, int p, int k)
    {
        // Coordinate descent for LASSO
        var x = new double[p];
        for (int j = 0; j < p; j++)
        {
            x[j] = data[sample, j];
        }

        // Compute D^T * D
        var DtD = new double[k, k];
        for (int i = 0; i < k; i++)
        {
            for (int j = 0; j < k; j++)
            {
                double sum = 0;
                for (int f = 0; f < p; f++)
                {
                    sum += _components![i, f] * _components[j, f];
                }
                DtD[i, j] = sum;
            }
        }

        // Compute D^T * x
        var Dtx = new double[k];
        for (int i = 0; i < k; i++)
        {
            double sum = 0;
            for (int j = 0; j < p; j++)
            {
                sum += _components![i, j] * x[j];
            }
            Dtx[i] = sum;
        }

        // Initialize codes to zero
        for (int c = 0; c < k; c++)
        {
            codes[sample, c] = 0;
        }

        // Coordinate descent
        for (int iter = 0; iter < 100; iter++)
        {
            double maxChange = 0;

            for (int c = 0; c < k; c++)
            {
                double oldCode = codes[sample, c];

                // Compute residual
                double residual = Dtx[c];
                for (int j = 0; j < k; j++)
                {
                    if (j != c)
                    {
                        residual -= DtD[c, j] * codes[sample, j];
                    }
                }

                // Soft thresholding
                double denominator = DtD[c, c];
                if (denominator < 1e-10)
                {
                    denominator = 1e-10;
                }

                double newCode = SoftThreshold(residual / denominator, _alpha / denominator);
                codes[sample, c] = newCode;

                maxChange = Math.Max(maxChange, Math.Abs(newCode - oldCode));
            }

            if (maxChange < 1e-6)
            {
                break;
            }
        }
    }

    private void UpdateDictionaryCD(double[,] data, double[,] codes, int n, int p, int k)
    {
        // Coordinate descent dictionary update
        for (int c = 0; c < k; c++)
        {
            // Compute residual for all samples without atom c
            var residual = new double[n, p];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < p; j++)
                {
                    residual[i, j] = data[i, j];
                    for (int cc = 0; cc < k; cc++)
                    {
                        if (cc != c)
                        {
                            residual[i, j] -= codes[i, cc] * _components![cc, j];
                        }
                    }
                }
            }

            // Update atom c
            double codesSquared = 0;
            for (int i = 0; i < n; i++)
            {
                codesSquared += codes[i, c] * codes[i, c];
            }

            if (codesSquared < 1e-10)
            {
                continue;
            }

            for (int j = 0; j < p; j++)
            {
                double sum = 0;
                for (int i = 0; i < n; i++)
                {
                    sum += codes[i, c] * residual[i, j];
                }
                _components![c, j] = sum / codesSquared;
            }

            // Normalize atom
            double norm = 0;
            for (int j = 0; j < p; j++)
            {
                norm += _components![c, j] * _components[c, j];
            }
            norm = Math.Sqrt(norm);

            if (norm > 1e-10)
            {
                for (int j = 0; j < p; j++)
                {
                    _components![c, j] /= norm;
                }
            }
        }
    }

    private void UpdateDictionaryLeastSquares(double[,] data, double[,] codes, int n, int p, int k)
    {
        // Least squares dictionary update: D = X^T * A * (A^T * A)^-1

        // Compute A^T * A
        var AtA = new double[k, k];
        for (int i = 0; i < k; i++)
        {
            for (int j = 0; j < k; j++)
            {
                double sum = 0;
                for (int s = 0; s < n; s++)
                {
                    sum += codes[s, i] * codes[s, j];
                }
                AtA[i, j] = sum;
            }
            AtA[i, i] += 1e-6; // Regularization
        }

        var AtAInv = InvertMatrix(AtA, k);

        // Compute X^T * A
        var XtA = new double[p, k];
        for (int i = 0; i < p; i++)
        {
            for (int j = 0; j < k; j++)
            {
                double sum = 0;
                for (int s = 0; s < n; s++)
                {
                    sum += data[s, i] * codes[s, j];
                }
                XtA[i, j] = sum;
            }
        }

        // D = XtA * AtAInv (transposed to get k x p)
        for (int c = 0; c < k; c++)
        {
            for (int j = 0; j < p; j++)
            {
                double sum = 0;
                for (int l = 0; l < k; l++)
                {
                    sum += XtA[j, l] * AtAInv[l, c];
                }
                _components![c, j] = sum;
            }

            // Normalize atom
            double norm = 0;
            for (int j = 0; j < p; j++)
            {
                norm += _components![c, j] * _components[c, j];
            }
            norm = Math.Sqrt(norm);

            if (norm > 1e-10)
            {
                for (int j = 0; j < p; j++)
                {
                    _components![c, j] /= norm;
                }
            }
        }
    }

    private static double SoftThreshold(double x, double lambda)
    {
        if (x > lambda)
        {
            return x - lambda;
        }
        else if (x < -lambda)
        {
            return x + lambda;
        }
        else
        {
            return 0;
        }
    }

    private static double[,] InvertMatrix(double[,] matrix, int n)
    {
        var result = new double[n, n];
        var temp = new double[n, 2 * n];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                temp[i, j] = matrix[i, j];
                temp[i, j + n] = (i == j) ? 1.0 : 0.0;
            }
        }

        for (int i = 0; i < n; i++)
        {
            double maxVal = Math.Abs(temp[i, i]);
            int maxRow = i;
            for (int k = i + 1; k < n; k++)
            {
                if (Math.Abs(temp[k, i]) > maxVal)
                {
                    maxVal = Math.Abs(temp[k, i]);
                    maxRow = k;
                }
            }

            if (maxRow != i)
            {
                for (int j = 0; j < 2 * n; j++)
                {
                    (temp[i, j], temp[maxRow, j]) = (temp[maxRow, j], temp[i, j]);
                }
            }

            double pivot = temp[i, i];
            if (Math.Abs(pivot) < 1e-10)
            {
                pivot = 1e-10;
            }

            for (int j = 0; j < 2 * n; j++)
            {
                temp[i, j] /= pivot;
            }

            for (int k = 0; k < n; k++)
            {
                if (k != i)
                {
                    double factor = temp[k, i];
                    for (int j = 0; j < 2 * n; j++)
                    {
                        temp[k, j] -= factor * temp[i, j];
                    }
                }
            }
        }

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                result[i, j] = temp[i, j + n];
            }
        }

        return result;
    }

    /// <summary>
    /// Transforms data by computing sparse codes.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_mean is null || _components is null)
        {
            throw new InvalidOperationException("DictionaryLearning has not been fitted.");
        }

        int n = data.Rows;
        int p = data.Columns;
        int k = _components.GetLength(0);

        // Center data and compute sparse codes
        var centered = new double[n, p];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                centered[i, j] = NumOps.ToDouble(data[i, j]) - _mean[j];
            }
        }

        var codes = new double[n, k];
        UpdateCodes(centered, codes, n, p, k);

        var result = new T[n, k];
        for (int i = 0; i < n; i++)
        {
            for (int c = 0; c < k; c++)
            {
                result[i, c] = NumOps.FromDouble(codes[i, c]);
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Transforms sparse codes back to original space.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_mean is null || _components is null)
        {
            throw new InvalidOperationException("DictionaryLearning has not been fitted.");
        }

        int n = data.Rows;
        int p = _nFeaturesIn;
        int k = _components.GetLength(0);
        var result = new T[n, p];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                double sum = _mean[j];
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
            names[i] = $"Atom{i + 1}";
        }
        return names;
    }
}

/// <summary>
/// Specifies the algorithm for fitting the dictionary.
/// </summary>
public enum DictionaryFitAlgorithm
{
    /// <summary>
    /// Coordinate descent - updates atoms one at a time.
    /// </summary>
    CD,

    /// <summary>
    /// Least squares - solves for all atoms simultaneously.
    /// </summary>
    LeastSquares
}

/// <summary>
/// Specifies the algorithm for sparse coding (transform).
/// </summary>
public enum SparseCodingAlgorithm
{
    /// <summary>
    /// LASSO (L1 regularization) using coordinate descent.
    /// </summary>
    LASSO,

    /// <summary>
    /// Orthogonal Matching Pursuit - greedy selection.
    /// </summary>
    OMP
}
