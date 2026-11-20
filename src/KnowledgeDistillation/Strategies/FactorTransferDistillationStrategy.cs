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

    /// <summary>
    /// Generator for the power-iteration starting vectors: seeded when a seed is given, otherwise secure.
    /// </summary>
    private readonly Random _random;

    private static readonly INumericOperations<double> WideOps = MathHelper.GetNumericOperations<double>();
    /// <summary>
    /// Creates a factor transfer strategy with a secure, unseeded generator for the power-iteration starts.
    /// </summary>
    public FactorTransferDistillationStrategy(
        double factorWeight,
        FactorMode mode,
        int numFactors,
        bool normalizeFactors,
        double temperature,
        double alpha)
        : this(factorWeight, mode, numFactors, normalizeFactors, temperature, alpha, seed: null)
    {
    }

    /// <summary>
    /// Creates a factor transfer strategy whose power-iteration starts are drawn from a seeded generator.
    /// </summary>
    /// <param name="factorWeight">Weight of the factor loss (between 0 and 1).</param>
    /// <param name="mode">How factors are compared.</param>
    /// <param name="numFactors">Number of factors (eigenvectors) to extract.</param>
    /// <param name="normalizeFactors">Whether extracted factors are normalized.</param>
    /// <param name="temperature">Softmax temperature for the standard distillation loss.</param>
    /// <param name="alpha">Weight of the hard-label loss in the standard distillation loss.</param>
    /// <param name="seed">Seed for the power-iteration starting vectors; null uses a secure, unseeded generator.
    /// With a fixed seed, the same sequence of calls gives identical factors.</param>
    public FactorTransferDistillationStrategy(
        double factorWeight = 0.5,
        FactorMode mode = FactorMode.LowRankApproximation,
        int numFactors = 32,
        bool normalizeFactors = true,
        double temperature = 3.0,
        double alpha = 0.3,
        int? seed = null)
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
        _random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
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

        var studentWide = studentFeatures.Select(row => Tensor<T>.FromVector(row).Cast<double>().ToVector()).ToArray();
        var teacherWide = teacherFeatures.Select(row => Tensor<T>.FromVector(row).Cast<double>().ToVector()).ToArray();
        double loss = _mode switch
        {
            FactorMode.LowRankApproximation => ComputeLowRankLoss(studentWide, teacherWide),
            FactorMode.NuclearNormMatching => ComputeNuclearNormLoss(studentWide, teacherWide),
            FactorMode.FactorMatching => ComputeFactorMatchingLoss(studentWide, teacherWide),
            _ => throw new ArgumentOutOfRangeException(nameof(_mode), _mode, $"Unrecognised FactorMode '{_mode}'. Valid modes: LowRankApproximation, NuclearNormMatching, FactorMatching.")
        };

        return NumOps.FromDouble(loss * _factorWeight);
    }

    private double ComputeLowRankLoss(Vector<double>[] studentFeatures, Vector<double>[] teacherFeatures)
    {
        // Extract top-k factors from both using simplified SVD approximation
        // For computational efficiency, we use power iteration-based approach

        var studentFactors = ExtractFactors(studentFeatures, _numFactors);
        var teacherFactors = ExtractFactors(teacherFeatures, _numFactors);

        // MSE between factors
        return ComputeMSE(studentFactors, teacherFactors);
    }

    private double ComputeNuclearNormLoss(Vector<double>[] studentFeatures, Vector<double>[] teacherFeatures)
    {
        // Nuclear norm = sum of singular values
        // Match the nuclear norms (captures overall "complexity" of representation)

        var studentSingularValues = ComputeSingularValues(studentFeatures, _numFactors);
        var teacherSingularValues = ComputeSingularValues(teacherFeatures, _numFactors);

        // Match singular value distributions
        int minLength = Math.Min(studentSingularValues.Length, teacherSingularValues.Length);
        var diff = Engine.Subtract(
            studentSingularValues.GetSubVector(0, minLength), teacherSingularValues.GetSubVector(0, minLength));
        double loss = Engine.DotProduct(diff, diff);

        return WideOps.Divide(loss, WideOps.FromDouble(minLength));
    }

    private double ComputeFactorMatchingLoss(Vector<double>[] studentFeatures, Vector<double>[] teacherFeatures)
    {
        // Paraphraser approach: Learn to match factorized representations
        // Simplified version: Extract factors and match with L2 + cosine similarity

        var studentFactors = ExtractFactors(studentFeatures, _numFactors);
        var teacherFactors = ExtractFactors(teacherFeatures, _numFactors);

        // L2 loss component
        double l2Loss = ComputeMSE(studentFactors, teacherFactors);

        // Cosine similarity component (1 - cos_sim, so 0 when identical), for factor pairs (i, i)
        int factorCount = studentFactors.Length;
        var a = DistillationHelper<double>.StackRows(studentFactors);
        var b = DistillationHelper<double>.StackRows(teacherFactors);
        var dots = Engine.ReduceSum(Engine.TensorMultiply(a, b), [1], keepDims: false);
        var normA = Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorMultiply(a, a), [1], keepDims: false));
        var normB = Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorMultiply(b, b), [1], keepDims: false));
        var denominator = Engine.TensorMultiply(normA, normB);
        var valid = Engine.TensorGreaterThan(denominator, 1e-10);
        var safe = Engine.TensorWhere(valid, denominator, Tensor<double>.CreateDefault([factorCount], 1.0));
        var cosine = Engine.TensorWhere(valid, Engine.TensorClamp(Engine.TensorDivide(dots, safe), -1.0, 1.0), new Tensor<double>([factorCount]));
        double cosineLoss = 1.0 - Engine.TensorSum(cosine) / factorCount;

        // Combined loss (equal weight)
        return WideOps.Add(
            WideOps.Multiply(l2Loss, WideOps.FromDouble(0.5)),
            WideOps.Multiply(cosineLoss, WideOps.FromDouble(0.5)));
    }

    private Vector<double>[] ExtractFactors(Vector<double>[] features, int numFactors)
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
        var factors = new Vector<double>[k];
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

    private Vector<double> ComputeSingularValues(Vector<double>[] features, int numValues)
    {
        // Approximate singular values from eigenvalues of covariance matrix
        var centered = CenterData(features);
        var covMatrix = ComputeCovarianceMatrix(centered);
        int featureDim = features[0].Length;
        int k = Math.Min(numValues, featureDim);

        var eigenvalues = new List<double>(k);

        for (int i = 0; i < k; i++)
        {
            var eigenvector = ExtractTopEigenvector(covMatrix, featureDim);
            eigenvalues.Add(ComputeEigenvalue(covMatrix, eigenvector, featureDim));

            covMatrix = DeflateCovariance(covMatrix, eigenvector, featureDim);
        }

        // Singular value = sqrt(eigenvalue * (n-1)) for covariance matrix (negative estimates clamp to 0)
        var scaled = Engine.Multiply(new Vector<double>(eigenvalues), WideOps.FromDouble(features.Length - 1));
        return Engine.Sqrt(Engine.Max(new Vector<double>(k), scaled));
    }

    private Vector<double>[] CenterData(Vector<double>[] features)
    {
        // Subtract the per-feature mean over the batch from every sample
        var data = DistillationHelper<double>.StackRows(features);
        var mean = Engine.ReduceMean(data, [0], keepDims: true);
        var centeredData = Engine.TensorSubtract(data, Engine.TensorBroadcastTo(mean, data.Shape.ToArray()));
        var centeredMatrix = centeredData.ToMatrix();
        var centered = Enumerable.Range(0, features.Length).Select(i => centeredMatrix.GetRow(i)).ToArray();

        return centered;
    }

    private Tensor<double> ComputeCovarianceMatrix(Vector<double>[] centeredFeatures)
    {
        int batchSize = centeredFeatures.Length;

        // Cov = (1/n) * X^T * X for centered X
        var x = DistillationHelper<double>.StackRows(centeredFeatures);
        var cov = Engine.TensorDivideScalar(
            Engine.TensorMatMul(Engine.TensorTranspose(x), x), WideOps.FromDouble(batchSize));

        return cov;
    }

    private Vector<double> ExtractTopEigenvector(Tensor<double> matrix, int dim)
    {
        // Power iteration to find dominant eigenvector

        // Initialize with random values in [-0.5, 0.5), one draw seeded from the strategy's generator
        var vector = Engine.TensorRandomUniformRange<double>(
            [dim], WideOps.FromDouble(-0.5), WideOps.FromDouble(0.5), _random.Next()).ToVector();
        vector = VectorHelper.Normalize(vector);

        // Power iteration (10 iterations usually sufficient)
        for (int iter = 0; iter < 10; iter++)
        {
            // Matrix-vector multiply
            var newVector = MatrixVectorProduct(matrix, vector, dim);

            vector = VectorHelper.Normalize(newVector);
        }

        return CanonicalizeSign(vector);
    }

    /// <summary>
    /// Flips an eigenvector so its largest-magnitude component (first one on ties) is non-negative.
    /// </summary>
    /// <remarks>
    /// Eigenvectors are only defined up to sign, and power iteration from a random start returns either
    /// sign. Fixing the sign makes repeated runs, and student/teacher factors of the same subspace, agree.
    /// </remarks>
    private Vector<double> CanonicalizeSign(Vector<double> vector)
    {
        int largest = DistillationHelper<double>.ArgMax(Engine.Abs(vector));
        return WideOps.LessThan(vector[largest], WideOps.Zero)
            ? Engine.Multiply(vector, WideOps.FromDouble(-1.0))
            : vector;
    }

    /// <summary>
    /// A * v for a [dim, dim] matrix and a length-dim vector.
    /// </summary>
    private Vector<double> MatrixVectorProduct(Tensor<double> matrix, Vector<double> vector, int dim)
    {
        return Engine.Reshape(
            Engine.TensorMatMul(matrix, Engine.Reshape(Tensor<double>.FromVector(vector), [dim, 1])), [dim]).ToVector();
    }

    private double ComputeEigenvalue(Tensor<double> matrix, Vector<double> eigenvector, int dim)
    {
        // Eigenvalue = v^T * A * v for normalized eigenvector v
        var Av = MatrixVectorProduct(matrix, eigenvector, dim);

        return Engine.DotProduct(eigenvector, Av);
    }

    private Tensor<double> DeflateCovariance(Tensor<double> matrix, Vector<double> eigenvector, int dim)
    {
        // Remove component: A := A - λ*v*v^T
        double eigenvalue = ComputeEigenvalue(matrix, eigenvector, dim);

        var v = Tensor<double>.FromVector(eigenvector);
        return Engine.TensorSubtract(matrix, Engine.TensorMultiplyScalar(Engine.TensorOuter(v, v), eigenvalue));
    }

    private double ComputeMSE(Vector<double>[] vectors1, Vector<double>[] vectors2)
    {
        // Mean squared difference over every element of every vector pair
        var diff = Engine.TensorSubtract(
            DistillationHelper<double>.StackRows(vectors1), DistillationHelper<double>.StackRows(vectors2));
        double mse = Engine.TensorSum(Engine.TensorMultiply(diff, diff));
        int totalElements = diff.Length;

        return WideOps.Divide(mse, WideOps.FromDouble(totalElements));
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


