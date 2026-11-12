using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;

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
/// - Attention Transfer (Zagoruyko &amp; Komodakis, 2017)</para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.Compression)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Paraphrasing Complex Network: Network Compression via Factor Transfer",
    "https://arxiv.org/abs/1802.04977",
    Year = 2018,
    Authors = "Jangho Kim, SeongUk Park, Nojun Kwak")]
[ComponentType(ComponentType.DistillationStrategy)]
[PipelineStage(PipelineStage.Training)]
public class FactorTransferDistillationStrategy<T> : DistillationStrategyBase<T>
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

    /// <inheritdoc />
    public override T ComputeLoss(Matrix<T> studentBatchOutput, Matrix<T> teacherBatchOutput, Matrix<T>? trueLabelsBatch = null)
    {
        ValidateOutputDimensions(studentBatchOutput, teacherBatchOutput);
        ValidateLabelDimensions(studentBatchOutput, trueLabelsBatch);

        int batchSize = studentBatchOutput.Rows;

        // Standard distillation loss per sample, with the factor weight reduction applied exactly once
        var sampleLosses = Engine.Multiply(
            DistillationHelper<T>.StandardSampleLosses(studentBatchOutput, teacherBatchOutput, trueLabelsBatch, Temperature, Alpha),
            NumOps.FromDouble(1.0 - _factorWeight));
        T totalLoss = Engine.Sum(sampleLosses);

        return NumOps.Divide(totalLoss, NumOps.FromDouble(batchSize));
    }

    /// <inheritdoc />
    public override Matrix<T> ComputeGradient(Matrix<T> studentBatchOutput, Matrix<T> teacherBatchOutput, Matrix<T>? trueLabelsBatch = null)
    {
        ValidateOutputDimensions(studentBatchOutput, teacherBatchOutput);
        ValidateLabelDimensions(studentBatchOutput, trueLabelsBatch);

        int batchSize = studentBatchOutput.Rows;

        // Combined gradient: Alpha * hardGrad + (1 - Alpha) * softGrad (soft only without labels),
        // with the factor weight reduction applied exactly once
        var gradientBatch = Engine.MatrixMultiplyScalar(
            DistillationHelper<T>.StandardSampleGradients(studentBatchOutput, teacherBatchOutput, trueLabelsBatch, Temperature, Alpha),
            NumOps.FromDouble(1.0 - _factorWeight));

        // Average gradients over the batch
        T oneOverBatch = NumOps.Divide(NumOps.One, NumOps.FromDouble(batchSize));
        gradientBatch = Engine.MatrixMultiplyScalar(gradientBatch, oneOverBatch);

        return gradientBatch;
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

        // Validate feature dimensions
        int studentDim = studentFeatures[0].Length;
        int teacherDim = teacherFeatures[0].Length;

        for (int i = 0; i < studentFeatures.Length; i++)
        {
            if (studentFeatures[i].Length != studentDim)
                throw new ArgumentException($"All student features must have same dimension. Expected {studentDim}, got {studentFeatures[i].Length} at index {i}");
            if (teacherFeatures[i].Length != teacherDim)
                throw new ArgumentException($"All teacher features must have same dimension. Expected {teacherDim}, got {teacherFeatures[i].Length} at index {i}");
            if (studentFeatures[i].Length != teacherFeatures[i].Length)
                throw new ArgumentException($"Student and teacher features must have matching dimensions. Got student={studentFeatures[i].Length}, teacher={teacherFeatures[i].Length} at index {i}");
            if (studentFeatures[i].Length == 0)
                throw new ArgumentException($"Feature vectors cannot be empty at index {i}");
        }

        T loss = _mode switch
        {
            FactorMode.LowRankApproximation => ComputeLowRankLoss(studentFeatures, teacherFeatures),
            FactorMode.NuclearNormMatching => ComputeNuclearNormLoss(studentFeatures, teacherFeatures),
            FactorMode.FactorMatching => ComputeFactorMatchingLoss(studentFeatures, teacherFeatures),
            _ => throw new ArgumentOutOfRangeException(nameof(_mode), _mode, $"Unrecognised FactorMode '{_mode}'. Valid modes: LowRankApproximation, NuclearNormMatching, FactorMatching.")
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
        int minLength = Math.Min(studentSingularValues.Length, teacherSingularValues.Length);
        var diff = Engine.Subtract(
            studentSingularValues.GetSubVector(0, minLength), teacherSingularValues.GetSubVector(0, minLength));
        T loss = Engine.DotProduct(diff, diff);

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

        // Cosine similarity component (1 - cos_sim, so 0 when identical), for factor pairs (i, i)
        int factorCount = studentFactors.Length;
        var cosine = DistillationHelper<T>.CosineSimilarityMatrix(
            DistillationHelper<T>.StackRows(studentFactors), DistillationHelper<T>.StackRows(teacherFactors));
        T pairedCosine = Engine.TensorSum(
            Engine.TensorMultiply(cosine, Tensor<T>.FromMatrix(Matrix<T>.CreateIdentity(factorCount))));
        T cosineLoss = NumOps.Subtract(NumOps.FromDouble(factorCount), pairedCosine);
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
                factors[i] = VectorHelper.Normalize(factors[i]);
            }

            // Deflate: remove this component from covariance matrix
            covMatrix = DeflateCovariance(covMatrix, factors[i], featureDim);
        }

        return factors;
    }

    private Vector<T> ComputeSingularValues(Vector<T>[] features, int numValues)
    {
        // Approximate singular values from eigenvalues of covariance matrix
        var centered = CenterData(features);
        var covMatrix = ComputeCovarianceMatrix(centered);
        int featureDim = features[0].Length;
        int k = Math.Min(numValues, featureDim);

        var eigenvalues = new List<T>(k);

        for (int i = 0; i < k; i++)
        {
            var eigenvector = ExtractTopEigenvector(covMatrix, featureDim);
            eigenvalues.Add(ComputeEigenvalue(covMatrix, eigenvector, featureDim));

            covMatrix = DeflateCovariance(covMatrix, eigenvector, featureDim);
        }

        // Singular value = sqrt(eigenvalue * (n-1)) for covariance matrix (negative estimates clamp to 0)
        var scaled = Engine.Multiply(new Vector<T>(eigenvalues), NumOps.FromDouble(features.Length - 1));
        return Engine.Sqrt(Engine.Max(new Vector<T>(k), scaled));
    }

    private Vector<T>[] CenterData(Vector<T>[] features)
    {
        // Subtract the per-feature mean over the batch from every sample
        var data = DistillationHelper<T>.StackRows(features);
        var mean = Engine.ReduceMean(data, [0], keepDims: true);
        var centeredData = Engine.TensorSubtract(data, Engine.TensorBroadcastTo(mean, data.Shape.ToArray()));
        var centeredMatrix = centeredData.ToMatrix();
        var centered = Enumerable.Range(0, features.Length).Select(i => centeredMatrix.GetRow(i)).ToArray();

        return centered;
    }

    private Tensor<T> ComputeCovarianceMatrix(Vector<T>[] centeredFeatures)
    {
        int batchSize = centeredFeatures.Length;

        // Cov = (1/n) * X^T * X for centered X
        var x = DistillationHelper<T>.StackRows(centeredFeatures);
        var cov = Engine.TensorDivideScalar(
            Engine.TensorMatMul(Engine.TensorTranspose(x), x), NumOps.FromDouble(batchSize));

        return cov;
    }

    private Vector<T> ExtractTopEigenvector(Tensor<T> matrix, int dim)
    {
        // Power iteration to find dominant eigenvector

        // Initialize with random values in [-0.5, 0.5) (unseeded, like the secure random it replaces)
        var vector = Engine.TensorRandomUniformRange<T>(
            [dim], NumOps.FromDouble(-0.5), NumOps.FromDouble(0.5)).ToVector();
        vector = VectorHelper.Normalize(vector);

        // Power iteration (10 iterations usually sufficient)
        for (int iter = 0; iter < 10; iter++)
        {
            // Matrix-vector multiply
            var newVector = MatrixVectorProduct(matrix, vector, dim);

            vector = VectorHelper.Normalize(newVector);
        }

        return vector;
    }

    /// <summary>
    /// A * v for a [dim, dim] matrix and a length-dim vector.
    /// </summary>
    private Vector<T> MatrixVectorProduct(Tensor<T> matrix, Vector<T> vector, int dim)
    {
        return Engine.Reshape(
            Engine.TensorMatMul(matrix, Engine.Reshape(Tensor<T>.FromVector(vector), [dim, 1])), [dim]).ToVector();
    }

    private T ComputeEigenvalue(Tensor<T> matrix, Vector<T> eigenvector, int dim)
    {
        // Eigenvalue = v^T * A * v for normalized eigenvector v
        var Av = MatrixVectorProduct(matrix, eigenvector, dim);

        return Engine.DotProduct(eigenvector, Av);
    }

    private Tensor<T> DeflateCovariance(Tensor<T> matrix, Vector<T> eigenvector, int dim)
    {
        // Remove component: A := A - λ*v*v^T
        T eigenvalue = ComputeEigenvalue(matrix, eigenvector, dim);

        var v = Tensor<T>.FromVector(eigenvector);
        return Engine.TensorSubtract(matrix, Engine.TensorMultiplyScalar(Engine.TensorOuter(v, v), eigenvalue));
    }

    private T ComputeMSE(Vector<T>[] vectors1, Vector<T>[] vectors2)
    {
        // Mean squared difference over every element of every vector pair
        var diff = Engine.TensorSubtract(
            DistillationHelper<T>.StackRows(vectors1), DistillationHelper<T>.StackRows(vectors2));
        T mse = Engine.TensorSum(Engine.TensorMultiply(diff, diff));
        int totalElements = diff.Length;

        return NumOps.Divide(mse, NumOps.FromDouble(totalElements));
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


