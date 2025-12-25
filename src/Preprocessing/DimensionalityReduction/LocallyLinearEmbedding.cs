using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Locally Linear Embedding for nonlinear dimensionality reduction.
/// </summary>
/// <remarks>
/// <para>
/// LLE preserves local neighborhood structure by representing each point
/// as a weighted linear combination of its neighbors. The embedding is found
/// by preserving these reconstruction weights in lower dimensions.
/// </para>
/// <para>
/// The algorithm:
/// 1. Find k nearest neighbors for each point
/// 2. Compute reconstruction weights that best reconstruct each point from neighbors
/// 3. Find low-dimensional embedding that preserves these weights
/// </para>
/// <para><b>For Beginners:</b> LLE preserves local relationships:
/// - Each point is described by its neighbors
/// - The weights describe "how much" each neighbor contributes
/// - The embedding keeps these relationships intact
/// - Good for unfolding curved manifolds (like the Swiss roll)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class LocallyLinearEmbedding<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly int _nNeighbors;
    private readonly double _reg;
    private readonly LLEMethod _method;

    // Fitted parameters
    private double[,]? _embedding;
    private double[,]? _reconstructionWeights;
    private int _nSamples;
    private int _nFeaturesIn;

    /// <summary>
    /// Gets the number of components.
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Gets the number of neighbors.
    /// </summary>
    public int NNeighbors => _nNeighbors;

    /// <summary>
    /// Gets the regularization parameter.
    /// </summary>
    public double Regularization => _reg;

    /// <summary>
    /// Gets the LLE method.
    /// </summary>
    public LLEMethod Method => _method;

    /// <summary>
    /// Gets the embedding result.
    /// </summary>
    public double[,]? Embedding => _embedding;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="LocallyLinearEmbedding{T}"/>.
    /// </summary>
    /// <param name="nComponents">Target dimensionality. Defaults to 2.</param>
    /// <param name="nNeighbors">Number of neighbors. Defaults to 5.</param>
    /// <param name="reg">Regularization constant. Defaults to 0.001.</param>
    /// <param name="method">LLE algorithm variant. Defaults to Standard.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public LocallyLinearEmbedding(
        int nComponents = 2,
        int nNeighbors = 5,
        double reg = 0.001,
        LLEMethod method = LLEMethod.Standard,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nComponents < 1)
        {
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));
        }

        if (nNeighbors < 1)
        {
            throw new ArgumentException("Number of neighbors must be at least 1.", nameof(nNeighbors));
        }

        _nComponents = nComponents;
        _nNeighbors = nNeighbors;
        _reg = reg;
        _method = method;
    }

    /// <summary>
    /// Fits LLE by computing reconstruction weights and embedding.
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        _nSamples = data.Rows;
        _nFeaturesIn = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        int k = _nNeighbors;

        if (k >= n)
        {
            throw new ArgumentException($"n_neighbors ({k}) must be less than n_samples ({n}).");
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

        // Step 1: Find k nearest neighbors
        var neighbors = FindNeighbors(X, n, p, k);

        // Step 2: Compute reconstruction weights
        _reconstructionWeights = ComputeReconstructionWeights(X, neighbors, n, p, k);

        // Step 3: Compute embedding
        _embedding = _method switch
        {
            LLEMethod.Modified => ComputeModifiedLLEEmbedding(X, neighbors, n, p, k),
            LLEMethod.HLLE => ComputeHLLEEmbedding(X, neighbors, n, p, k),
            LLEMethod.LTSA => ComputeLTSAEmbedding(X, neighbors, n, p, k),
            _ => ComputeStandardLLEEmbedding(n, k)
        };
    }

    private int[][] FindNeighbors(double[,] X, int n, int p, int k)
    {
        var neighbors = new int[n][];

        for (int i = 0; i < n; i++)
        {
            // Compute distances to all other points
            var distances = new (int Index, double Distance)[n - 1];
            int idx = 0;

            for (int j = 0; j < n; j++)
            {
                if (j == i) continue;

                double dist = 0;
                for (int d = 0; d < p; d++)
                {
                    double diff = X[i, d] - X[j, d];
                    dist += diff * diff;
                }

                distances[idx++] = (j, dist);
            }

            // Find k nearest
            neighbors[i] = distances
                .OrderBy(x => x.Distance)
                .Take(k)
                .Select(x => x.Index)
                .ToArray();
        }

        return neighbors;
    }

    private double[,] ComputeReconstructionWeights(double[,] X, int[][] neighbors, int n, int p, int k)
    {
        var W = new double[n, n];

        for (int i = 0; i < n; i++)
        {
            // Get neighbors of point i
            var neighborIdx = neighbors[i];

            // Build local covariance matrix C
            // C[j,l] = (x_i - x_j)' * (x_i - x_l)
            var C = new double[k, k];

            for (int j = 0; j < k; j++)
            {
                for (int l = j; l < k; l++)
                {
                    double sum = 0;
                    for (int d = 0; d < p; d++)
                    {
                        double dij = X[i, d] - X[neighborIdx[j], d];
                        double dil = X[i, d] - X[neighborIdx[l], d];
                        sum += dij * dil;
                    }
                    C[j, l] = sum;
                    C[l, j] = sum;
                }
            }

            // Regularization
            double trace = 0;
            for (int j = 0; j < k; j++)
            {
                trace += C[j, j];
            }
            double regVal = _reg * trace / k;
            if (regVal < 1e-10)
            {
                regVal = 1e-3;
            }

            for (int j = 0; j < k; j++)
            {
                C[j, j] += regVal;
            }

            // Solve C * w = 1 (ones vector)
            var w = SolveLinearSystem(C, k);

            // Normalize weights
            double wSum = w.Sum();
            if (Math.Abs(wSum) > 1e-10)
            {
                for (int j = 0; j < k; j++)
                {
                    w[j] /= wSum;
                }
            }

            // Store weights
            for (int j = 0; j < k; j++)
            {
                W[i, neighborIdx[j]] = w[j];
            }
        }

        return W;
    }

    private double[] SolveLinearSystem(double[,] A, int n)
    {
        // Solve A * x = ones using Gaussian elimination
        var augmented = new double[n, n + 1];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                augmented[i, j] = A[i, j];
            }
            augmented[i, n] = 1.0; // RHS is ones vector
        }

        // Forward elimination
        for (int col = 0; col < n; col++)
        {
            // Find pivot
            int maxRow = col;
            double maxVal = Math.Abs(augmented[col, col]);
            for (int row = col + 1; row < n; row++)
            {
                if (Math.Abs(augmented[row, col]) > maxVal)
                {
                    maxVal = Math.Abs(augmented[row, col]);
                    maxRow = row;
                }
            }

            // Swap rows
            if (maxRow != col)
            {
                for (int j = 0; j <= n; j++)
                {
                    (augmented[col, j], augmented[maxRow, j]) = (augmented[maxRow, j], augmented[col, j]);
                }
            }

            // Eliminate
            double pivot = augmented[col, col];
            if (Math.Abs(pivot) < 1e-10)
            {
                pivot = 1e-10;
            }

            for (int row = col + 1; row < n; row++)
            {
                double factor = augmented[row, col] / pivot;
                for (int j = col; j <= n; j++)
                {
                    augmented[row, j] -= factor * augmented[col, j];
                }
            }
        }

        // Back substitution
        var x = new double[n];
        for (int row = n - 1; row >= 0; row--)
        {
            x[row] = augmented[row, n];
            for (int col = row + 1; col < n; col++)
            {
                x[row] -= augmented[row, col] * x[col];
            }
            double diag = augmented[row, row];
            if (Math.Abs(diag) < 1e-10)
            {
                diag = 1e-10;
            }
            x[row] /= diag;
        }

        return x;
    }

    private double[,] ComputeStandardLLEEmbedding(int n, int k)
    {
        // Compute M = (I - W)' * (I - W)
        var M = new double[n, n];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double sum = 0;

                // (I - W)_ki * (I - W)_kj
                for (int l = 0; l < n; l++)
                {
                    double Ili = (i == l ? 1 : 0) - _reconstructionWeights![l, i];
                    double Ilj = (j == l ? 1 : 0) - _reconstructionWeights[l, j];
                    sum += Ili * Ilj;
                }

                M[i, j] = sum;
            }
        }

        // Find smallest non-zero eigenvectors
        var (eigenvalues, eigenvectors) = ComputeEigen(M, n);

        // Sort by eigenvalue ascending (we want smallest)
        var indices = Enumerable.Range(0, n)
            .OrderBy(i => eigenvalues[i])
            .ToArray();

        // Take components 1 to nComponents (skip the zero eigenvalue)
        int nComp = Math.Min(_nComponents, n - 1);
        var embedding = new double[n, nComp];

        for (int d = 0; d < nComp; d++)
        {
            // Skip the first (zero) eigenvalue
            int idx = indices[d + 1];

            for (int i = 0; i < n; i++)
            {
                embedding[i, d] = eigenvectors[idx, i];
            }
        }

        return embedding;
    }

    private double[,] ComputeModifiedLLEEmbedding(double[,] X, int[][] neighbors, int n, int p, int k)
    {
        // Modified LLE uses multiple weight vectors from local SVD
        // This provides more robust embedding by using d_out weight vectors per point
        // Reference: Zhang & Zha, "Principal Manifolds and Nonlinear Dimensionality Reduction via Tangent Space Alignment"

        int d = _nComponents;
        int numWeightVectors = d + 1; // Number of weight vectors per point

        // Build the regularization matrix for modified LLE
        var M = new double[n, n];

        for (int i = 0; i < n; i++)
        {
            var neighborIdx = neighbors[i];
            int numNeighbors = neighborIdx.Length;

            // Build local covariance matrix for neighborhood
            // First center the neighborhood
            var neighborMean = new double[p];
            for (int j = 0; j < numNeighbors; j++)
            {
                for (int f = 0; f < p; f++)
                {
                    neighborMean[f] += X[neighborIdx[j], f];
                }
            }
            for (int f = 0; f < p; f++)
            {
                neighborMean[f] /= numNeighbors;
            }

            // Create centered neighbor matrix
            var centeredNeighbors = new double[numNeighbors, p];
            for (int j = 0; j < numNeighbors; j++)
            {
                for (int f = 0; f < p; f++)
                {
                    centeredNeighbors[j, f] = X[neighborIdx[j], f] - neighborMean[f];
                }
            }

            // Compute SVD of centered neighbors to get local tangent space
            // We use power iteration to find principal directions
            var (_, localBasis) = ComputeLocalSVD(centeredNeighbors, numNeighbors, p, numWeightVectors);

            // Build local weight matrix: each weight vector is a column
            // Weight matrix W_i = I - V * V^T where V is the local basis
            var localWeightMatrix = new double[numNeighbors, numNeighbors];
            for (int j = 0; j < numNeighbors; j++)
            {
                localWeightMatrix[j, j] = 1.0;
            }

            for (int v = 0; v < numWeightVectors && v < numNeighbors; v++)
            {
                for (int j1 = 0; j1 < numNeighbors; j1++)
                {
                    for (int j2 = 0; j2 < numNeighbors; j2++)
                    {
                        localWeightMatrix[j1, j2] -= localBasis[v, j1] * localBasis[v, j2];
                    }
                }
            }

            // Accumulate into global M matrix
            for (int j1 = 0; j1 < numNeighbors; j1++)
            {
                for (int j2 = 0; j2 < numNeighbors; j2++)
                {
                    M[neighborIdx[j1], neighborIdx[j2]] += localWeightMatrix[j1, j2];
                }
            }
        }

        // Find smallest eigenvectors of M
        var (eigenvalues, eigenvectors) = ComputeEigen(M, n);

        var indices = Enumerable.Range(0, n)
            .OrderBy(idx => eigenvalues[idx])
            .ToArray();

        int nComp = Math.Min(_nComponents, n - 1);
        var embedding = new double[n, nComp];

        for (int c = 0; c < nComp; c++)
        {
            int idx = indices[c + 1]; // Skip first (zero) eigenvalue
            for (int i = 0; i < n; i++)
            {
                embedding[i, c] = eigenvectors[idx, i];
            }
        }

        return embedding;
    }

    private double[,] ComputeHLLEEmbedding(double[,] X, int[][] neighbors, int n, int p, int k)
    {
        // Hessian Locally Linear Embedding
        // Estimates the Hessian of the manifold and uses it for embedding
        // Reference: Donoho & Grimes, "Hessian Eigenmaps"

        int d = _nComponents;
        int dpd1 = d * (d + 1) / 2; // Number of Hessian components

        // Build Hessian estimator matrix
        var H = new double[n, n];

        for (int i = 0; i < n; i++)
        {
            var neighborIdx = neighbors[i];
            int numNeighbors = neighborIdx.Length;

            if (numNeighbors < dpd1 + d + 1)
            {
                // Fall back to standard LLE if not enough neighbors
                return ComputeStandardLLEEmbedding(n, k);
            }

            // Build local covariance and get tangent space
            var neighborMean = new double[p];
            for (int j = 0; j < numNeighbors; j++)
            {
                for (int f = 0; f < p; f++)
                {
                    neighborMean[f] += X[neighborIdx[j], f];
                }
            }
            for (int f = 0; f < p; f++)
            {
                neighborMean[f] /= numNeighbors;
            }

            var centeredNeighbors = new double[numNeighbors, p];
            for (int j = 0; j < numNeighbors; j++)
            {
                for (int f = 0; f < p; f++)
                {
                    centeredNeighbors[j, f] = X[neighborIdx[j], f] - neighborMean[f];
                }
            }

            // Get local tangent coordinates using SVD
            var (singularValues, tangentBasis) = ComputeLocalSVD(centeredNeighbors, numNeighbors, p, d);

            // Project neighbors onto tangent space
            var tangentCoords = new double[numNeighbors, d];
            for (int j = 0; j < numNeighbors; j++)
            {
                for (int c = 0; c < d; c++)
                {
                    double sum = 0;
                    for (int f = 0; f < p; f++)
                    {
                        sum += centeredNeighbors[j, f] * tangentBasis[c, f];
                    }
                    tangentCoords[j, c] = sum;
                }
            }

            // Build Hessian estimation matrix [1, t, t^2, t*t']
            // where t is the tangent coordinate vector
            int numHessianCols = 1 + d + dpd1;
            var Yi = new double[numNeighbors, numHessianCols];

            for (int j = 0; j < numNeighbors; j++)
            {
                Yi[j, 0] = 1.0; // Constant term

                // Linear terms
                for (int c = 0; c < d; c++)
                {
                    Yi[j, 1 + c] = tangentCoords[j, c];
                }

                // Quadratic terms (Hessian)
                int colIdx = 1 + d;
                for (int c1 = 0; c1 < d; c1++)
                {
                    for (int c2 = c1; c2 < d; c2++)
                    {
                        Yi[j, colIdx++] = tangentCoords[j, c1] * tangentCoords[j, c2];
                    }
                }
            }

            // QR decomposition to get null space estimator
            var (Q, _) = ComputeQR(Yi, numNeighbors, numHessianCols);

            // Extract null space vectors (last columns of Q)
            int nullSpaceDim = numNeighbors - numHessianCols;
            if (nullSpaceDim <= 0)
            {
                continue;
            }

            // Build local Hessian estimator: H_i = null * null^T
            for (int j1 = 0; j1 < numNeighbors; j1++)
            {
                for (int j2 = 0; j2 < numNeighbors; j2++)
                {
                    double sum = 0;
                    for (int q = numHessianCols; q < numNeighbors; q++)
                    {
                        sum += Q[j1, q] * Q[j2, q];
                    }
                    H[neighborIdx[j1], neighborIdx[j2]] += sum;
                }
            }
        }

        // Find smallest eigenvectors of H
        var (eigenvalues, eigenvectors) = ComputeEigen(H, n);

        var indices = Enumerable.Range(0, n)
            .OrderBy(idx => eigenvalues[idx])
            .ToArray();

        int nComp = Math.Min(_nComponents, n - 1);
        var embedding = new double[n, nComp];

        for (int c = 0; c < nComp; c++)
        {
            int idx = indices[c + 1];
            for (int i = 0; i < n; i++)
            {
                embedding[i, c] = eigenvectors[idx, i];
            }
        }

        return embedding;
    }

    private double[,] ComputeLTSAEmbedding(double[,] X, int[][] neighbors, int n, int p, int k)
    {
        // Local Tangent Space Alignment
        // Aligns local tangent spaces to produce global embedding
        // Reference: Zhang & Zha, "Principal Manifolds and Nonlinear Dimensionality Reduction"

        int d = _nComponents;

        // Build alignment matrix
        var B = new double[n, n];

        for (int i = 0; i < n; i++)
        {
            var neighborIdx = neighbors[i];
            int numNeighbors = neighborIdx.Length;

            // Center neighbors
            var neighborMean = new double[p];
            for (int j = 0; j < numNeighbors; j++)
            {
                for (int f = 0; f < p; f++)
                {
                    neighborMean[f] += X[neighborIdx[j], f];
                }
            }
            for (int f = 0; f < p; f++)
            {
                neighborMean[f] /= numNeighbors;
            }

            var centeredNeighbors = new double[numNeighbors, p];
            for (int j = 0; j < numNeighbors; j++)
            {
                for (int f = 0; f < p; f++)
                {
                    centeredNeighbors[j, f] = X[neighborIdx[j], f] - neighborMean[f];
                }
            }

            // Compute local tangent space basis
            var (_, tangentBasis) = ComputeLocalSVD(centeredNeighbors, numNeighbors, p, d);

            // Project neighbors onto tangent space
            var Theta = new double[numNeighbors, d];
            for (int j = 0; j < numNeighbors; j++)
            {
                for (int c = 0; c < d; c++)
                {
                    double sum = 0;
                    for (int f = 0; f < p; f++)
                    {
                        sum += centeredNeighbors[j, f] * tangentBasis[c, f];
                    }
                    Theta[j, c] = sum;
                }
            }

            // Build local alignment matrix: G_i = I - Theta * (Theta^T * Theta)^-1 * Theta^T
            // First compute Theta^T * Theta
            var ThetaTTheta = new double[d, d];
            for (int c1 = 0; c1 < d; c1++)
            {
                for (int c2 = 0; c2 < d; c2++)
                {
                    double sum = 0;
                    for (int j = 0; j < numNeighbors; j++)
                    {
                        sum += Theta[j, c1] * Theta[j, c2];
                    }
                    ThetaTTheta[c1, c2] = sum;
                }
                ThetaTTheta[c1, c1] += 1e-6; // Regularization
            }

            var ThetaTThetaInv = InvertSmallMatrix(ThetaTTheta, d);

            // Compute W_i = Theta * (Theta^T * Theta)^-1 * Theta^T
            var Wi = new double[numNeighbors, numNeighbors];
            for (int j1 = 0; j1 < numNeighbors; j1++)
            {
                for (int j2 = 0; j2 < numNeighbors; j2++)
                {
                    double sum = 0;
                    for (int c1 = 0; c1 < d; c1++)
                    {
                        for (int c2 = 0; c2 < d; c2++)
                        {
                            sum += Theta[j1, c1] * ThetaTThetaInv[c1, c2] * Theta[j2, c2];
                        }
                    }
                    Wi[j1, j2] = sum;
                }
            }

            // G_i = I - W_i and accumulate (I - G_i)^T * (I - G_i) = W_i^T * W_i
            for (int j1 = 0; j1 < numNeighbors; j1++)
            {
                for (int j2 = 0; j2 < numNeighbors; j2++)
                {
                    double localTerm = (j1 == j2 ? 1.0 : 0.0) - Wi[j1, j2];

                    // Accumulate into global matrix
                    for (int j3 = 0; j3 < numNeighbors; j3++)
                    {
                        double localTerm2 = (j3 == j2 ? 1.0 : 0.0) - Wi[j3, j2];
                        B[neighborIdx[j1], neighborIdx[j3]] += localTerm * localTerm2;
                    }
                }
            }
        }

        // Find smallest eigenvectors of B
        var (eigenvalues, eigenvectors) = ComputeEigen(B, n);

        var indices = Enumerable.Range(0, n)
            .OrderBy(idx => eigenvalues[idx])
            .ToArray();

        int nComp = Math.Min(_nComponents, n - 1);
        var embedding = new double[n, nComp];

        for (int c = 0; c < nComp; c++)
        {
            int idx = indices[c + 1];
            for (int i = 0; i < n; i++)
            {
                embedding[i, c] = eigenvectors[idx, i];
            }
        }

        return embedding;
    }

    private (double[] SingularValues, double[,] RightVectors) ComputeLocalSVD(double[,] matrix, int m, int n, int k)
    {
        // Compute truncated SVD using power iteration
        // Returns right singular vectors (V) which span the column space

        int numComponents = Math.Min(k, Math.Min(m, n));
        var singularValues = new double[numComponents];
        var rightVectors = new double[numComponents, n];

        // Work on A^T * A for right singular vectors
        var AtA = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double sum = 0;
                for (int r = 0; r < m; r++)
                {
                    sum += matrix[r, i] * matrix[r, j];
                }
                AtA[i, j] = sum;
            }
        }

        var A = (double[,])AtA.Clone();
        var random = RandomHelper.CreateSeededRandom(42);

        for (int c = 0; c < numComponents; c++)
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

            // Power iteration
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

            singularValues[c] = Math.Sqrt(Math.Max(0, eigenvalue));
            for (int i = 0; i < n; i++)
            {
                rightVectors[c, i] = v[i];
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

        return (singularValues, rightVectors);
    }

    private (double[,] Q, double[,] R) ComputeQR(double[,] matrix, int m, int n)
    {
        // Gram-Schmidt QR decomposition
        var Q = new double[m, m];
        var R = new double[m, n];

        // Initialize Q as copy of matrix columns padded with identity
        for (int j = 0; j < n && j < m; j++)
        {
            for (int i = 0; i < m; i++)
            {
                Q[i, j] = matrix[i, j];
            }
        }

        // Fill remaining columns with identity (for null space)
        for (int j = n; j < m; j++)
        {
            Q[j, j] = 1.0;
        }

        // Modified Gram-Schmidt
        for (int j = 0; j < m; j++)
        {
            // Orthogonalize against previous columns
            for (int k = 0; k < j; k++)
            {
                double dot = 0;
                for (int i = 0; i < m; i++)
                {
                    dot += Q[i, k] * Q[i, j];
                }

                if (j < n)
                {
                    R[k, j] = dot;
                }

                for (int i = 0; i < m; i++)
                {
                    Q[i, j] -= dot * Q[i, k];
                }
            }

            // Normalize
            double norm = 0;
            for (int i = 0; i < m; i++)
            {
                norm += Q[i, j] * Q[i, j];
            }
            norm = Math.Sqrt(norm);

            if (j < n)
            {
                R[j, j] = norm;
            }

            if (norm > 1e-10)
            {
                for (int i = 0; i < m; i++)
                {
                    Q[i, j] /= norm;
                }
            }
        }

        return (Q, R);
    }

    private static double[,] InvertSmallMatrix(double[,] matrix, int n)
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

    private (double[] Eigenvalues, double[,] Eigenvectors) ComputeEigen(double[,] matrix, int n)
    {
        var eigenvalues = new double[n];
        var eigenvectors = new double[n, n];
        var A = (double[,])matrix.Clone();

        // Add small regularization for numerical stability
        for (int i = 0; i < n; i++)
        {
            A[i, i] += 1e-10;
        }

        for (int iter = 0; iter < n; iter++)
        {
            var v = new double[n];
            for (int i = 0; i < n; i++)
            {
                v[i] = 1.0 / Math.Sqrt(n);
            }

            for (int powerIter = 0; powerIter < 100; powerIter++)
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

            eigenvalues[iter] = eigenvalue;
            for (int i = 0; i < n; i++)
            {
                eigenvectors[iter, i] = v[i];
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

    /// <summary>
    /// Returns the embedding computed during Fit.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_embedding is null)
        {
            throw new InvalidOperationException("LocallyLinearEmbedding has not been fitted.");
        }

        if (data.Rows != _nSamples)
        {
            throw new InvalidOperationException(
                "LLE does not support out-of-sample transformation. " +
                "Use FitTransform() on the complete dataset.");
        }

        int n = _embedding.GetLength(0);
        int k = _embedding.GetLength(1);
        var result = new T[n, k];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < k; j++)
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
        throw new NotSupportedException("LocallyLinearEmbedding does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        var names = new string[_nComponents];
        for (int i = 0; i < _nComponents; i++)
        {
            names[i] = $"LLE{i + 1}";
        }
        return names;
    }
}

/// <summary>
/// Specifies the LLE algorithm variant.
/// </summary>
public enum LLEMethod
{
    /// <summary>
    /// Standard LLE algorithm.
    /// </summary>
    Standard,

    /// <summary>
    /// Modified LLE with multiple weight vectors.
    /// </summary>
    Modified,

    /// <summary>
    /// Hessian Locally Linear Embedding.
    /// </summary>
    HLLE,

    /// <summary>
    /// Local Tangent Space Alignment.
    /// </summary>
    LTSA
}
