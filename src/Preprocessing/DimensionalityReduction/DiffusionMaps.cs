using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Diffusion Maps for nonlinear dimensionality reduction.
/// </summary>
/// <remarks>
/// <para>
/// Diffusion Maps embeds data into a low-dimensional space where Euclidean distances
/// approximate diffusion distances on the data manifold. It simulates a random walk
/// on the data graph and uses the eigenvectors of the diffusion operator.
/// </para>
/// <para>
/// The algorithm:
/// 1. Constructs a kernel matrix (typically Gaussian)
/// 2. Normalizes to create a Markov transition matrix
/// 3. Computes eigenvectors of the diffusion operator
/// 4. Scales eigenvectors by eigenvalue powers for embedding
/// </para>
/// <para><b>For Beginners:</b> Diffusion Maps captures the underlying geometry by:
/// - Simulating how information "diffuses" through the data
/// - Points connected by many short paths are close in diffusion distance
/// - Robust to noise compared to geodesic distances
///
/// Use cases:
/// - Discovering underlying manifold structure
/// - Robust to noise in the data
/// - When you want distances to reflect connectivity, not just proximity
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class DiffusionMaps<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly double _alpha;
    private readonly int _diffusionTime;
    private readonly double _epsilon;
    private readonly int? _randomState;

    // Fitted parameters
    private double[,]? _embedding;
    private double[]? _eigenvalues;
    private int _nSamples;

    /// <summary>
    /// Gets the number of components (dimensions).
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Gets the diffusion time parameter.
    /// </summary>
    public int DiffusionTime => _diffusionTime;

    /// <summary>
    /// Gets the embedding result.
    /// </summary>
    public double[,]? Embedding => _embedding;

    /// <summary>
    /// Gets the eigenvalues.
    /// </summary>
    public double[]? Eigenvalues => _eigenvalues;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="DiffusionMaps{T}"/>.
    /// </summary>
    /// <param name="nComponents">Target dimensionality. Defaults to 2.</param>
    /// <param name="alpha">Normalization factor (0 to 1). Defaults to 0.5.</param>
    /// <param name="diffusionTime">Number of diffusion steps. Defaults to 1.</param>
    /// <param name="epsilon">Kernel bandwidth. If 0, computed automatically. Defaults to 0.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public DiffusionMaps(
        int nComponents = 2,
        double alpha = 0.5,
        int diffusionTime = 1,
        double epsilon = 0,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nComponents < 1)
        {
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));
        }

        if (alpha < 0 || alpha > 1)
        {
            throw new ArgumentException("Alpha must be between 0 and 1.", nameof(alpha));
        }

        if (diffusionTime < 1)
        {
            throw new ArgumentException("Diffusion time must be at least 1.", nameof(diffusionTime));
        }

        _nComponents = nComponents;
        _alpha = alpha;
        _diffusionTime = diffusionTime;
        _epsilon = epsilon;
        _randomState = randomState;
    }

    /// <summary>
    /// Fits Diffusion Maps and computes the embedding.
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        _nSamples = data.Rows;
        int n = data.Rows;
        int p = data.Columns;

        // Validate that we have enough samples for requested components
        // We need at least nComponents + 1 samples because we skip the first eigenvector
        if (_nComponents >= n)
        {
            throw new ArgumentException(
                $"nComponents ({_nComponents}) must be less than number of samples ({n}) for Diffusion Maps. " +
                $"The first eigenvector is skipped, so at least {_nComponents + 1} samples are required.");
        }

        // Convert to double array
        var X = new double[n, p];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                X[i, j] = NumOps.ToDouble(data[i, j]);
            }
        }

        // Step 1: Compute pairwise squared distances
        var distSq = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                double dist = 0;
                for (int k = 0; k < p; k++)
                {
                    double diff = X[i, k] - X[j, k];
                    dist += diff * diff;
                }
                distSq[i, j] = dist;
                distSq[j, i] = dist;
            }
        }

        // Step 2: Compute epsilon (kernel bandwidth) if not provided
        double eps = _epsilon;
        if (eps <= 0)
        {
            // Use median of pairwise distances
            var allDists = new List<double>();
            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    allDists.Add(Math.Sqrt(distSq[i, j]));
                }
            }
            allDists.Sort();
            eps = allDists[allDists.Count / 2];
            eps = eps * eps; // Square it for the kernel
        }

        // Step 3: Compute kernel matrix K
        var K = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                K[i, j] = Math.Exp(-distSq[i, j] / eps);
            }
        }

        // Step 4: Compute row sums (degree)
        var q = new double[n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                q[i] += K[i, j];
            }
        }

        // Step 5: Normalize kernel (alpha-normalization)
        var Kalpha = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (q[i] > 0 && q[j] > 0)
                {
                    Kalpha[i, j] = K[i, j] / Math.Pow(q[i] * q[j], _alpha);
                }
            }
        }

        // Step 6: Compute new row sums
        var d = new double[n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                d[i] += Kalpha[i, j];
            }
        }

        // Step 7: Construct symmetric diffusion operator
        var Ms = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (d[i] > 0 && d[j] > 0)
                {
                    Ms[i, j] = Kalpha[i, j] / Math.Sqrt(d[i] * d[j]);
                }
            }
        }

        // Step 8: Compute eigenvectors
        var (eigenvalues, eigenvectors) = ComputeLargestEigenvectors(Ms, n);

        // Step 9: Create embedding (skip first eigenvector which is constant)
        _eigenvalues = new double[_nComponents];
        _embedding = new double[n, _nComponents];

        for (int k = 0; k < _nComponents; k++)
        {
            int eigIdx = k + 1; // Skip first (constant) eigenvector

            _eigenvalues[k] = eigenvalues[eigIdx];

            // Scale by eigenvalue^t for diffusion time t
            // Eigenvalues of the symmetric normalized Laplacian should be non-negative
            double scale = Math.Pow(eigenvalues[eigIdx], _diffusionTime);

            for (int i = 0; i < n; i++)
            {
                // Standard diffusion map embedding uses eigenvectors scaled by eigenvalue powers
                // Degree normalization was already applied when constructing the symmetric operator Ms
                _embedding[i, k] = eigenvectors[eigIdx, i] * scale;
            }
        }
    }

    private (double[] Eigenvalues, double[,] Eigenvectors) ComputeLargestEigenvectors(double[,] M, int n)
    {
        var eigenvalues = new double[n];
        var eigenvectors = new double[n, n];
        var A = (double[,])M.Clone();

        for (int k = 0; k < Math.Min(n, _nComponents + 2); k++)
        {
            var v = new double[n];
            var random = _randomState.HasValue
                ? RandomHelper.CreateSeededRandom(_randomState.Value + k)
                : RandomHelper.CreateSeededRandom(42 + k);

            for (int i = 0; i < n; i++)
            {
                v[i] = random.NextDouble();
            }

            // Normalize initial vector
            double initNorm = 0;
            for (int i = 0; i < n; i++) initNorm += v[i] * v[i];
            initNorm = Math.Sqrt(initNorm);
            for (int i = 0; i < n; i++) v[i] /= initNorm;

            // Power iteration
            for (int iter = 0; iter < 100; iter++)
            {
                var Av = new double[n];
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        Av[i] += A[i, j] * v[j];
                    }
                }

                double norm = 0;
                for (int i = 0; i < n; i++)
                {
                    norm += Av[i] * Av[i];
                }
                norm = Math.Sqrt(norm);

                if (norm < 1e-10) break;

                for (int i = 0; i < n; i++)
                {
                    v[i] = Av[i] / norm;
                }
            }

            // Compute eigenvalue
            var Av2 = new double[n];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    Av2[i] += A[i, j] * v[j];
                }
            }

            double eigenvalue = 0;
            for (int i = 0; i < n; i++)
            {
                eigenvalue += v[i] * Av2[i];
            }

            eigenvalues[k] = eigenvalue;
            for (int i = 0; i < n; i++)
            {
                eigenvectors[k, i] = v[i];
            }

            // Deflate
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

    /// <summary>
    /// Returns the embedding computed during Fit.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_embedding is null)
        {
            throw new InvalidOperationException("DiffusionMaps has not been fitted.");
        }

        if (data.Rows != _nSamples)
        {
            throw new InvalidOperationException(
                "DiffusionMaps does not support out-of-sample transformation. " +
                "Use FitTransform() on the complete dataset.");
        }

        int n = _embedding.GetLength(0);
        int d = _embedding.GetLength(1);
        var result = new T[n, d];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < d; j++)
            {
                result[i, j] = NumOps.FromDouble(_embedding[i, j]);
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("DiffusionMaps does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        var names = new string[_nComponents];
        for (int i = 0; i < _nComponents; i++)
        {
            names[i] = $"DM{i + 1}";
        }
        return names;
    }
}
