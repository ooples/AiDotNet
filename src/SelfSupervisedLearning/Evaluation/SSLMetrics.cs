using AiDotNet.Helpers;

namespace AiDotNet.SelfSupervisedLearning.Evaluation;

/// <summary>
/// Metrics for monitoring and evaluating self-supervised learning.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> These metrics help track the quality of SSL training
/// and detect potential issues like representation collapse. Monitoring these
/// during training helps ensure the model is learning useful representations.</para>
///
/// <para><b>Key metrics:</b></para>
/// <list type="bullet">
/// <item><b>Representation collapse:</b> All embeddings become identical (very bad)</item>
/// <item><b>Feature std:</b> Standard deviation of features (should be positive)</item>
/// <item><b>Alignment:</b> Similarity between positive pairs (should be high)</item>
/// <item><b>Uniformity:</b> Distribution of features on hypersphere (should be uniform)</item>
/// </list>
/// </remarks>
public class SSLMetrics<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Computes the standard deviation of representations (collapse detection).
    /// </summary>
    /// <param name="representations">Representations [batch_size, dim].</param>
    /// <returns>Standard deviation per dimension, averaged.</returns>
    /// <remarks>
    /// A low standard deviation indicates potential collapse - all representations
    /// are becoming similar. Good representations should have reasonable variance.
    /// </remarks>
    public T ComputeRepresentationStd(Tensor<T> representations)
    {
        var batchSize = representations.Shape[0];
        var dim = representations.Shape[1];

        T totalStd = NumOps.Zero;

        for (int d = 0; d < dim; d++)
        {
            // Compute mean
            T mean = NumOps.Zero;
            for (int b = 0; b < batchSize; b++)
            {
                mean = NumOps.Add(mean, representations[b, d]);
            }
            mean = NumOps.Divide(mean, NumOps.FromDouble(batchSize));

            // Compute variance
            T variance = NumOps.Zero;
            for (int b = 0; b < batchSize; b++)
            {
                var diff = NumOps.Subtract(representations[b, d], mean);
                variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
            }
            variance = NumOps.Divide(variance, NumOps.FromDouble(batchSize));

            totalStd = NumOps.Add(totalStd, NumOps.Sqrt(variance));
        }

        return NumOps.Divide(totalStd, NumOps.FromDouble(dim));
    }

    /// <summary>
    /// Computes alignment loss between positive pairs.
    /// </summary>
    /// <param name="z1">First view representations [batch_size, dim].</param>
    /// <param name="z2">Second view representations [batch_size, dim].</param>
    /// <returns>Average squared distance between positive pairs.</returns>
    /// <remarks>
    /// Lower alignment loss means positive pairs are closer together in
    /// representation space, which is desirable.
    /// </remarks>
    public T ComputeAlignment(Tensor<T> z1, Tensor<T> z2)
    {
        var batchSize = z1.Shape[0];
        var dim = z1.Shape[1];

        // Normalize representations
        var z1Norm = L2Normalize(z1);
        var z2Norm = L2Normalize(z2);

        T totalDist = NumOps.Zero;

        for (int b = 0; b < batchSize; b++)
        {
            T dist = NumOps.Zero;
            for (int d = 0; d < dim; d++)
            {
                var diff = NumOps.Subtract(z1Norm[b, d], z2Norm[b, d]);
                dist = NumOps.Add(dist, NumOps.Multiply(diff, diff));
            }
            totalDist = NumOps.Add(totalDist, dist);
        }

        return NumOps.Divide(totalDist, NumOps.FromDouble(batchSize));
    }

    /// <summary>
    /// Computes uniformity loss (how uniformly distributed embeddings are).
    /// </summary>
    /// <param name="representations">Representations [batch_size, dim].</param>
    /// <param name="t">Temperature parameter (default: 2).</param>
    /// <returns>Uniformity metric (lower is more uniform).</returns>
    /// <remarks>
    /// Measures how uniformly representations are distributed on the hypersphere.
    /// More uniform distributions indicate better representations that capture
    /// diverse information.
    /// </remarks>
    public T ComputeUniformity(Tensor<T> representations, double t = 2.0)
    {
        var batchSize = representations.Shape[0];
        var dim = representations.Shape[1];

        // Normalize representations
        var zNorm = L2Normalize(representations);

        T sum = NumOps.Zero;
        int count = 0;

        for (int i = 0; i < batchSize; i++)
        {
            for (int j = i + 1; j < batchSize; j++)
            {
                // Compute squared distance
                T dist = NumOps.Zero;
                for (int d = 0; d < dim; d++)
                {
                    var diff = NumOps.Subtract(zNorm[i, d], zNorm[j, d]);
                    dist = NumOps.Add(dist, NumOps.Multiply(diff, diff));
                }

                // Add exp(-t * dist)
                sum = NumOps.Add(sum,
                    NumOps.Exp(NumOps.Multiply(NumOps.FromDouble(-t), dist)));
                count++;
            }
        }

        // Return log of average
        var avg = NumOps.Divide(sum, NumOps.FromDouble(Math.Max(count, 1)));
        return NumOps.Log(NumOps.Add(avg, NumOps.FromDouble(1e-8)));
    }

    /// <summary>
    /// Computes the effective rank of the representation matrix.
    /// </summary>
    /// <param name="representations">Representations [batch_size, dim].</param>
    /// <returns>Effective rank (normalized entropy of singular values).</returns>
    /// <remarks>
    /// Effective rank measures the "dimensionality" of the learned representations.
    /// Collapsed representations have low effective rank. Good representations
    /// should use many dimensions effectively.
    /// </remarks>
    public T ComputeEffectiveRank(Tensor<T> representations)
    {
        var dim = representations.Shape[1];

        // Compute covariance matrix
        var cov = ComputeCovarianceMatrix(representations);

        // Compute eigenvalues using Jacobi eigenvalue algorithm for symmetric matrices
        var eigenvalues = ComputeEigenvaluesJacobi(cov, dim);

        // Sum eigenvalues (for normalization)
        T sumEig = NumOps.Zero;
        for (int i = 0; i < dim; i++)
        {
            eigenvalues[i] = NumOps.Abs(eigenvalues[i]);
            eigenvalues[i] = NumOps.Add(eigenvalues[i], NumOps.FromDouble(1e-8));
            sumEig = NumOps.Add(sumEig, eigenvalues[i]);
        }

        // Normalize to get probability distribution and compute entropy
        T entropy = NumOps.Zero;
        for (int i = 0; i < dim; i++)
        {
            var p = NumOps.Divide(eigenvalues[i], sumEig);
            if (NumOps.GreaterThan(p, NumOps.FromDouble(1e-10)))
            {
                entropy = NumOps.Subtract(entropy,
                    NumOps.Multiply(p, NumOps.Log(p)));
            }
        }

        // Effective rank = exp(entropy)
        return NumOps.Exp(entropy);
    }

    /// <summary>
    /// Computes eigenvalues of a symmetric matrix using the Jacobi eigenvalue algorithm.
    /// </summary>
    private T[] ComputeEigenvaluesJacobi(Tensor<T> matrix, int n, int maxIterations = 100)
    {
        // Copy matrix to work array (algorithm modifies in place)
        var A = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                A[i, j] = NumOps.ToDouble(matrix[i, j]);
            }
        }

        const double tolerance = 1e-10;

        for (int iter = 0; iter < maxIterations; iter++)
        {
            // Find largest off-diagonal element
            int p = 0, q = 1;
            double maxOffDiag = Math.Abs(A[0, 1]);

            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    if (Math.Abs(A[i, j]) > maxOffDiag)
                    {
                        maxOffDiag = Math.Abs(A[i, j]);
                        p = i;
                        q = j;
                    }
                }
            }

            // Check convergence
            if (maxOffDiag < tolerance)
            {
                break;
            }

            // Compute rotation angle using Jacobi rotation formula
            double theta;
            if (Math.Abs(A[p, p] - A[q, q]) < 1e-15)
            {
                theta = Math.PI / 4;
            }
            else
            {
                theta = 0.5 * Math.Atan2(2 * A[p, q], A[p, p] - A[q, q]);
            }

            double c = Math.Cos(theta);
            double s = Math.Sin(theta);

            // Apply Givens rotation: A' = G^T * A * G
            // Store old values needed for update
            double app = A[p, p];
            double aqq = A[q, q];
            double apq = A[p, q];

            // Update diagonal elements
            A[p, p] = c * c * app + s * s * aqq - 2 * s * c * apq;
            A[q, q] = s * s * app + c * c * aqq + 2 * s * c * apq;
            A[p, q] = 0; // By construction
            A[q, p] = 0;

            // Update other elements
            for (int k = 0; k < n; k++)
            {
                if (k != p && k != q)
                {
                    double akp = A[k, p];
                    double akq = A[k, q];
                    A[k, p] = c * akp - s * akq;
                    A[p, k] = A[k, p];
                    A[k, q] = s * akp + c * akq;
                    A[q, k] = A[k, q];
                }
            }
        }

        // Eigenvalues are on the diagonal
        var eigenvalues = new T[n];
        for (int i = 0; i < n; i++)
        {
            eigenvalues[i] = NumOps.FromDouble(A[i, i]);
        }

        return eigenvalues;
    }

    /// <summary>
    /// Detects if representations are collapsing.
    /// </summary>
    /// <param name="representations">Representations to check.</param>
    /// <param name="threshold">Threshold for collapse detection (default: 0.01).</param>
    /// <returns>True if collapse is detected.</returns>
    public bool DetectCollapse(Tensor<T> representations, double threshold = 0.01)
    {
        var std = ComputeRepresentationStd(representations);
        return NumOps.ToDouble(std) < threshold;
    }

    /// <summary>
    /// Computes a full set of SSL metrics.
    /// </summary>
    public SSLMetricReport<T> ComputeFullReport(
        Tensor<T> z1, Tensor<T> z2)
    {
        return new SSLMetricReport<T>
        {
            Std1 = ComputeRepresentationStd(z1),
            Std2 = ComputeRepresentationStd(z2),
            Alignment = ComputeAlignment(z1, z2),
            Uniformity1 = ComputeUniformity(z1),
            Uniformity2 = ComputeUniformity(z2),
            EffectiveRank1 = ComputeEffectiveRank(z1),
            EffectiveRank2 = ComputeEffectiveRank(z2),
            CollapseDetected = DetectCollapse(z1) || DetectCollapse(z2)
        };
    }

    /// <summary>
    /// Computes cosine similarity between corresponding pairs.
    /// </summary>
    public T ComputeCosineSimilarity(Tensor<T> z1, Tensor<T> z2)
    {
        var batchSize = z1.Shape[0];
        var dim = z1.Shape[1];

        var z1Norm = L2Normalize(z1);
        var z2Norm = L2Normalize(z2);

        T totalSim = NumOps.Zero;

        for (int b = 0; b < batchSize; b++)
        {
            T dot = NumOps.Zero;
            for (int d = 0; d < dim; d++)
            {
                dot = NumOps.Add(dot, NumOps.Multiply(z1Norm[b, d], z2Norm[b, d]));
            }
            totalSim = NumOps.Add(totalSim, dot);
        }

        return NumOps.Divide(totalSim, NumOps.FromDouble(batchSize));
    }

    private Tensor<T> L2Normalize(Tensor<T> tensor)
    {
        var batchSize = tensor.Shape[0];
        var dim = tensor.Shape[1];
        var result = new T[batchSize * dim];

        for (int i = 0; i < batchSize; i++)
        {
            T sumSquared = NumOps.Zero;
            for (int j = 0; j < dim; j++)
            {
                var val = tensor[i, j];
                sumSquared = NumOps.Add(sumSquared, NumOps.Multiply(val, val));
            }

            var norm = NumOps.Sqrt(NumOps.Add(sumSquared, NumOps.FromDouble(1e-8)));

            for (int j = 0; j < dim; j++)
            {
                result[i * dim + j] = NumOps.Divide(tensor[i, j], norm);
            }
        }

        return new Tensor<T>(result, [batchSize, dim]);
    }

    private Tensor<T> ComputeCovarianceMatrix(Tensor<T> representations)
    {
        var batchSize = representations.Shape[0];
        var dim = representations.Shape[1];

        // Compute mean
        var mean = new T[dim];
        for (int d = 0; d < dim; d++)
        {
            for (int b = 0; b < batchSize; b++)
            {
                mean[d] = NumOps.Add(mean[d], representations[b, d]);
            }
            mean[d] = NumOps.Divide(mean[d], NumOps.FromDouble(batchSize));
        }

        // Compute covariance
        var cov = new T[dim * dim];
        for (int i = 0; i < dim; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                T sum = NumOps.Zero;
                for (int b = 0; b < batchSize; b++)
                {
                    var diff_i = NumOps.Subtract(representations[b, i], mean[i]);
                    var diff_j = NumOps.Subtract(representations[b, j], mean[j]);
                    sum = NumOps.Add(sum, NumOps.Multiply(diff_i, diff_j));
                }

                cov[i * dim + j] = NumOps.Divide(sum, NumOps.FromDouble(batchSize - 1));
                cov[j * dim + i] = cov[i * dim + j]; // Symmetric
            }
        }

        return new Tensor<T>(cov, [dim, dim]);
    }
}
