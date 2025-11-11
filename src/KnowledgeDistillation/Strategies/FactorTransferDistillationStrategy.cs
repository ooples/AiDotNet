using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation.Strategies;

/// <summary>
/// Factor transfer distillation that transfers knowledge through factorized representations.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Production Use:</b> This strategy implements factor transfer from the paper
/// "Paraphrasing Complex Network: Network Compression via Factor Transfer" (Kim et al., 2018).
/// Instead of directly matching high-dimensional representations, it factorizes them into
/// lower-dimensional factors and matches these factors.</para>
///
/// <para><b>Key Concept:</b> High-dimensional neural network representations contain redundancy.
/// Factor transfer extracts the essential information by:
/// 1. Decomposing representations into factors (low-rank approximation)
/// 2. Matching student factors to teacher factors
/// 3. Reducing noise and focusing on important dimensions</para>
///
/// <para><b>Why This Works:</b>
/// - Removes redundancy: Raw features have correlated dimensions
/// - Focuses on variance: Principal components capture most information
/// - More robust: Less sensitive to noise in individual neurons
/// - Compressed transfer: Fewer dimensions to match</para>
///
/// <para><b>Implementation:</b> We provide multiple factorization methods:
/// - LowRankApproximation: Use truncated decomposition (similar to PCA)
/// - NuclearNormMatching: Match based on singular values (nuclear norm)
/// - FactorMatching: Paraphraser network approach with translator</para>
///
/// <para><b>Mathematical Foundation:</b>
/// For feature matrix F ∈ ℝ^(n×d), decompose as F ≈ U*S*V^T where:
/// - U ∈ ℝ^(n×k): Samples in factor space
/// - S ∈ ℝ^(k×k): Singular values (importance weights)
/// - V ∈ ℝ^(d×k): Feature loadings
/// Match student and teacher factors using MSE or Frobenius norm.</para>
///
/// <para><b>Research Basis:</b>
/// - Paraphrasing Complex Network (Kim et al., 2018)
/// - FitNets: Hints for Thin Deep Nets (Romero et al., 2015)
/// - Attention Transfer (Zagoruyko & Komodakis, 2017)</para>
/// </remarks>
public class FactorTransferDistillationStrategy<T> : DistillationStrategyBase<T, Vector<T>>
{
    private readonly double _factorWeight;
    private readonly FactorMode _mode;
    private readonly int _numFactors;
    private readonly bool _normalizeFactors;

    public FactorTransferDistillationStrategy(
        double factorWeight = 0.5,
        FactorMode mode = FactorMode.LowRankApproximation,
        int numFactors = 32,
        bool normalizeFactors = true,
        double temperature = 3.0,
        double alpha = 0.3)
        : base(temperature, alpha)
    {
        if (factorWeight < 0 || factorWeight > 1)
            throw new ArgumentException("Factor weight must be between 0 and 1", nameof(factorWeight));

        if (numFactors <= 0)
            throw new ArgumentException("Number of factors must be positive", nameof(numFactors));

        _factorWeight = factorWeight;
        _mode = mode;
        _numFactors = numFactors;
        _normalizeFactors = normalizeFactors;
    }

    public override T ComputeLoss(Vector<T> studentOutput, Vector<T> teacherOutput, Vector<T>? trueLabels = null)
    {
        ValidateOutputDimensions(studentOutput, teacherOutput, v => v.Length);

        // Standard distillation loss
        var studentSoft = Softmax(studentOutput, Temperature);
        var teacherSoft = Softmax(teacherOutput, Temperature);
        var softLoss = KLDivergence(teacherSoft, studentSoft);
        softLoss = NumOps.Multiply(softLoss, NumOps.FromDouble(Temperature * Temperature));

        T finalLoss;
        if (trueLabels != null)
        {
            ValidateLabelDimensions(studentOutput, trueLabels, v => v.Length);
            var studentProbs = Softmax(studentOutput, 1.0);
            var hardLoss = CrossEntropy(studentProbs, trueLabels);
            finalLoss = NumOps.Add(
                NumOps.Multiply(NumOps.FromDouble(Alpha), hardLoss),
                NumOps.Multiply(NumOps.FromDouble(1.0 - Alpha), softLoss));
        }
        else
        {
            finalLoss = softLoss;
        }

        // Apply factor weight reduction exactly once
        return NumOps.Multiply(finalLoss, NumOps.FromDouble(1.0 - _factorWeight));
    }

    public override Vector<T> ComputeGradient(Vector<T> studentOutput, Vector<T> teacherOutput, Vector<T>? trueLabels = null)
    {
        ValidateOutputDimensions(studentOutput, teacherOutput, v => v.Length);

        int n = studentOutput.Length;
        var gradient = new Vector<T>(n);

        var studentSoft = Softmax(studentOutput, Temperature);
        var teacherSoft = Softmax(teacherOutput, Temperature);

        if (trueLabels != null)
        {
            ValidateLabelDimensions(studentOutput, trueLabels, v => v.Length);
            var studentProbs = Softmax(studentOutput, 1.0);

            for (int i = 0; i < n; i++)
            {
                // Soft gradient (temperature-scaled)
                var softGrad = NumOps.Subtract(studentSoft[i], teacherSoft[i]);
                softGrad = NumOps.Multiply(softGrad, NumOps.FromDouble(Temperature * Temperature));

                // Hard gradient
                var hardGrad = NumOps.Subtract(studentProbs[i], trueLabels[i]);

                // Combined gradient: Alpha * hardGrad + (1 - Alpha) * softGrad
                var combined = NumOps.Add(
                    NumOps.Multiply(NumOps.FromDouble(Alpha), hardGrad),
                    NumOps.Multiply(NumOps.FromDouble(1.0 - Alpha), softGrad));

                // Apply factor weight reduction exactly once
                gradient[i] = NumOps.Multiply(combined, NumOps.FromDouble(1.0 - _factorWeight));
            }
        }
        else
        {
            for (int i = 0; i < n; i++)
            {
                // Soft gradient (temperature-scaled)
                var softGrad = NumOps.Subtract(studentSoft[i], teacherSoft[i]);
                softGrad = NumOps.Multiply(softGrad, NumOps.FromDouble(Temperature * Temperature));

                // Apply factor weight reduction exactly once
                gradient[i] = NumOps.Multiply(softGrad, NumOps.FromDouble(1.0 - _factorWeight));
            }
        }

        return gradient;
    }

    /// <summary>
    /// Computes factor transfer loss by matching factorized representations.
    /// </summary>
    /// <param name="studentFeatures">Student feature matrix [batchSize x featureDim].</param>
    /// <param name="teacherFeatures">Teacher feature matrix [batchSize x featureDim].</param>
    /// <returns>Factor transfer loss.</returns>
    /// <remarks>
    /// <para>Features should be intermediate layer activations collected over a batch.
    /// The method will factorize both representations and match the factors.</para>
    /// </remarks>
    public T ComputeFactorLoss(Vector<T>[] studentFeatures, Vector<T>[] teacherFeatures)
    {
        if (studentFeatures.Length != teacherFeatures.Length)
            throw new ArgumentException("Student and teacher must have same batch size");

        if (studentFeatures.Length == 0)
            return NumOps.Zero;

        T loss = _mode switch
        {
            FactorMode.LowRankApproximation => ComputeLowRankLoss(studentFeatures, teacherFeatures),
            FactorMode.NuclearNormMatching => ComputeNuclearNormLoss(studentFeatures, teacherFeatures),
            FactorMode.FactorMatching => ComputeFactorMatchingLoss(studentFeatures, teacherFeatures),
            _ => throw new NotImplementedException($"Mode {_mode} not implemented")
        };

        return NumOps.Multiply(loss, NumOps.FromDouble(_factorWeight));
    }

    private T ComputeLowRankLoss(Vector<T>[] studentFeatures, Vector<T>[] teacherFeatures)
    {
        // Extract top-k factors from both using simplified SVD approximation
        // For computational efficiency, we use power iteration-based approach

        var studentFactors = ExtractFactors(studentFeatures, _numFactors);
        var teacherFactors = ExtractFactors(teacherFeatures, _numFactors);

        // MSE between factors
        return ComputeMSE(studentFactors, teacherFactors);
    }

    private T ComputeNuclearNormLoss(Vector<T>[] studentFeatures, Vector<T>[] teacherFeatures)
    {
        // Nuclear norm = sum of singular values
        // Match the nuclear norms (captures overall "complexity" of representation)

        var studentSingularValues = ComputeSingularValues(studentFeatures, _numFactors);
        var teacherSingularValues = ComputeSingularValues(teacherFeatures, _numFactors);

        // Match singular value distributions
        T loss = NumOps.Zero;
        int minLength = Math.Min(studentSingularValues.Length, teacherSingularValues.Length);

        for (int i = 0; i < minLength; i++)
        {
            double diff = studentSingularValues[i] - teacherSingularValues[i];
            loss = NumOps.Add(loss, NumOps.FromDouble(diff * diff));
        }

        return NumOps.Divide(loss, NumOps.FromDouble(minLength));
    }

    private T ComputeFactorMatchingLoss(Vector<T>[] studentFeatures, Vector<T>[] teacherFeatures)
    {
        // Paraphraser approach: Learn to match factorized representations
        // Simplified version: Extract factors and match with L2 + cosine similarity

        var studentFactors = ExtractFactors(studentFeatures, _numFactors);
        var teacherFactors = ExtractFactors(teacherFeatures, _numFactors);

        // L2 loss component
        T l2Loss = ComputeMSE(studentFactors, teacherFactors);

        // Cosine similarity component (1 - cos_sim, so 0 when identical)
        T cosineLoss = NumOps.Zero;
        for (int i = 0; i < studentFactors.Length; i++)
        {
            double cosSim = CosineSimilarity(studentFactors[i], teacherFactors[i]);
            cosineLoss = NumOps.Add(cosineLoss, NumOps.FromDouble(1.0 - cosSim));
        }
        cosineLoss = NumOps.Divide(cosineLoss, NumOps.FromDouble(studentFactors.Length));

        // Combined loss (equal weight)
        return NumOps.Add(
            NumOps.Multiply(l2Loss, NumOps.FromDouble(0.5)),
            NumOps.Multiply(cosineLoss, NumOps.FromDouble(0.5)));
    }

    private Vector<T>[] ExtractFactors(Vector<T>[] features, int numFactors)
    {
        // Simplified factor extraction using variance-based approach
        // In practice, this approximates PCA/SVD by finding directions of maximum variance

        int batchSize = features.Length;
        int featureDim = features[0].Length;
        int k = Math.Min(numFactors, Math.Min(batchSize, featureDim));

        // Center the data
        var centered = CenterData(features);

        // Compute covariance matrix (feature space)
        var covMatrix = ComputeCovarianceMatrix(centered);

        // Extract top-k eigenvectors (factors) using power iteration
        var factors = new Vector<T>[k];
        for (int i = 0; i < k; i++)
        {
            factors[i] = ExtractTopEigenvector(covMatrix, featureDim);

            if (_normalizeFactors)
            {
                factors[i] = NormalizeVector(factors[i]);
            }

            // Deflate: remove this component from covariance matrix
            DeflateCovariance(covMatrix, factors[i], featureDim);
        }

        return factors;
    }

    private double[] ComputeSingularValues(Vector<T>[] features, int numValues)
    {
        // Approximate singular values from eigenvalues of covariance matrix
        var centered = CenterData(features);
        var covMatrix = ComputeCovarianceMatrix(centered);
        int featureDim = features[0].Length;
        int k = Math.Min(numValues, featureDim);

        var singularValues = new double[k];

        for (int i = 0; i < k; i++)
        {
            var eigenvector = ExtractTopEigenvector(covMatrix, featureDim);
            double eigenvalue = ComputeEigenvalue(covMatrix, eigenvector, featureDim);

            // Singular value = sqrt(eigenvalue * (n-1)) for covariance matrix
            singularValues[i] = Math.Sqrt(Math.Max(0, eigenvalue * (features.Length - 1)));

            DeflateCovariance(covMatrix, eigenvector, featureDim);
        }

        return singularValues;
    }

    private Vector<T>[] CenterData(Vector<T>[] features)
    {
        int batchSize = features.Length;
        int featureDim = features[0].Length;

        // Compute mean
        var mean = new double[featureDim];
        for (int j = 0; j < featureDim; j++)
        {
            double sum = 0;
            for (int i = 0; i < batchSize; i++)
            {
                sum += Convert.ToDouble(features[i][j]);
            }
            mean[j] = sum / batchSize;
        }

        // Center
        var centered = new Vector<T>[batchSize];
        for (int i = 0; i < batchSize; i++)
        {
            centered[i] = new Vector<T>(featureDim);
            for (int j = 0; j < featureDim; j++)
            {
                double val = Convert.ToDouble(features[i][j]) - mean[j];
                centered[i][j] = NumOps.FromDouble(val);
            }
        }

        return centered;
    }

    private double[,] ComputeCovarianceMatrix(Vector<T>[] centeredFeatures)
    {
        int batchSize = centeredFeatures.Length;
        int featureDim = centeredFeatures[0].Length;
        var cov = new double[featureDim, featureDim];

        // Cov = (1/n) * X^T * X for centered X
        for (int i = 0; i < featureDim; i++)
        {
            for (int j = i; j < featureDim; j++) // Upper triangle only, symmetric
            {
                double sum = 0;
                for (int k = 0; k < batchSize; k++)
                {
                    sum += Convert.ToDouble(centeredFeatures[k][i]) *
                           Convert.ToDouble(centeredFeatures[k][j]);
                }
                cov[i, j] = sum / batchSize;
                cov[j, i] = cov[i, j]; // Symmetric
            }
        }

        return cov;
    }

    private Vector<T> ExtractTopEigenvector(double[,] matrix, int dim)
    {
        // Power iteration to find dominant eigenvector
        var vector = new Vector<T>(dim);

        // Initialize with random values
        var random = new Random();
        for (int i = 0; i < dim; i++)
        {
            vector[i] = NumOps.FromDouble(random.NextDouble() - 0.5);
        }
        vector = NormalizeVector(vector);

        // Power iteration (10 iterations usually sufficient)
        for (int iter = 0; iter < 10; iter++)
        {
            var newVector = new Vector<T>(dim);

            // Matrix-vector multiply
            for (int i = 0; i < dim; i++)
            {
                double sum = 0;
                for (int j = 0; j < dim; j++)
                {
                    sum += matrix[i, j] * Convert.ToDouble(vector[j]);
                }
                newVector[i] = NumOps.FromDouble(sum);
            }

            vector = NormalizeVector(newVector);
        }

        return vector;
    }

    private double ComputeEigenvalue(double[,] matrix, Vector<T> eigenvector, int dim)
    {
        // Eigenvalue = v^T * A * v for normalized eigenvector v
        var Av = new double[dim];

        for (int i = 0; i < dim; i++)
        {
            double sum = 0;
            for (int j = 0; j < dim; j++)
            {
                sum += matrix[i, j] * Convert.ToDouble(eigenvector[j]);
            }
            Av[i] = sum;
        }

        double eigenvalue = 0;
        for (int i = 0; i < dim; i++)
        {
            eigenvalue += Convert.ToDouble(eigenvector[i]) * Av[i];
        }

        return eigenvalue;
    }

    private void DeflateCovariance(double[,] matrix, Vector<T> eigenvector, int dim)
    {
        // Remove component: A := A - λ*v*v^T
        double eigenvalue = ComputeEigenvalue(matrix, eigenvector, dim);

        for (int i = 0; i < dim; i++)
        {
            for (int j = 0; j < dim; j++)
            {
                double vi = Convert.ToDouble(eigenvector[i]);
                double vj = Convert.ToDouble(eigenvector[j]);
                matrix[i, j] -= eigenvalue * vi * vj;
            }
        }
    }

    private Vector<T> NormalizeVector(Vector<T> vector)
    {
        double norm = 0;
        for (int i = 0; i < vector.Length; i++)
        {
            double val = Convert.ToDouble(vector[i]);
            norm += val * val;
        }
        norm = Math.Sqrt(norm);

        var normalized = new Vector<T>(vector.Length);
        for (int i = 0; i < vector.Length; i++)
        {
            normalized[i] = NumOps.FromDouble(Convert.ToDouble(vector[i]) / (norm + Epsilon));
        }

        return normalized;
    }

    private T ComputeMSE(Vector<T>[] vectors1, Vector<T>[] vectors2)
    {
        int n = vectors1.Length;
        T mse = NumOps.Zero;
        int totalElements = 0;

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < vectors1[i].Length; j++)
            {
                double diff = Convert.ToDouble(vectors1[i][j]) - Convert.ToDouble(vectors2[i][j]);
                mse = NumOps.Add(mse, NumOps.FromDouble(diff * diff));
                totalElements++;
            }
        }

        return NumOps.Divide(mse, NumOps.FromDouble(totalElements));
    }

    private double CosineSimilarity(Vector<T> v1, Vector<T> v2)
    {
        T dot = NumOps.Zero, norm1 = NumOps.Zero, norm2 = NumOps.Zero;

        for (int i = 0; i < v1.Length; i++)
        {
            dot = NumOps.Add(dot, NumOps.Multiply(v1[i], v2[i]));
            norm1 = NumOps.Add(norm1, NumOps.Multiply(v1[i], v1[i]));
            norm2 = NumOps.Add(norm2, NumOps.Multiply(v2[i], v2[i]));
        }

        return Convert.ToDouble(dot) / (Math.Sqrt(Convert.ToDouble(norm1)) * Math.Sqrt(Convert.ToDouble(norm2)) + Epsilon);
    }

    private Vector<T> Softmax(Vector<T> logits, double temperature)
    {
        int n = logits.Length;
        var result = new Vector<T>(n);
        var scaled = new T[n];

        for (int i = 0; i < n; i++)
            scaled[i] = NumOps.FromDouble(Convert.ToDouble(logits[i]) / temperature);

        T maxLogit = scaled[0];
        for (int i = 1; i < n; i++)
            if (NumOps.GreaterThan(scaled[i], maxLogit))
                maxLogit = scaled[i];

        T sum = NumOps.Zero;
        var expValues = new T[n];

        for (int i = 0; i < n; i++)
        {
            double val = Convert.ToDouble(NumOps.Subtract(scaled[i], maxLogit));
            expValues[i] = NumOps.FromDouble(Math.Exp(val));
            sum = NumOps.Add(sum, expValues[i]);
        }

        for (int i = 0; i < n; i++)
            result[i] = NumOps.Divide(expValues[i], sum);

        return result;
    }

    private T KLDivergence(Vector<T> p, Vector<T> q)
    {
        T divergence = NumOps.Zero;
        for (int i = 0; i < p.Length; i++)
        {
            double pVal = Convert.ToDouble(p[i]);
            double qVal = Convert.ToDouble(q[i]);
            if (pVal > Epsilon)
                divergence = NumOps.Add(divergence, NumOps.FromDouble(pVal * Math.Log(pVal / (qVal + Epsilon))));
        }
        return divergence;
    }

    private T CrossEntropy(Vector<T> predictions, Vector<T> trueLabels)
    {
        T entropy = NumOps.Zero;
        for (int i = 0; i < predictions.Length; i++)
        {
            double pred = Convert.ToDouble(predictions[i]);
            double label = Convert.ToDouble(trueLabels[i]);
            if (label > Epsilon)
                entropy = NumOps.Add(entropy, NumOps.FromDouble(-label * Math.Log(pred + Epsilon)));
        }
        return entropy;
    }
}

public enum FactorMode
{
    /// <summary>
    /// Extract and match low-rank factors (approximates PCA/SVD).
    /// </summary>
    LowRankApproximation,

    /// <summary>
    /// Match nuclear norms (sum of singular values).
    /// </summary>
    NuclearNormMatching,

    /// <summary>
    /// Paraphraser-style factor matching with L2 + cosine similarity.
    /// </summary>
    FactorMatching
}
