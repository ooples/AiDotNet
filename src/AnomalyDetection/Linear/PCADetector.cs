using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.Linear;

/// <summary>
/// Detects anomalies using Principal Component Analysis (PCA) reconstruction error.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> PCA-based anomaly detection works by projecting data onto
/// the main directions of variation (principal components) and measuring how well
/// points can be reconstructed. Anomalies have high reconstruction error.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Fit PCA on training data to find principal components
/// 2. Project each point onto these components
/// 3. Reconstruct the point from the projection
/// 4. Measure reconstruction error (anomaly score)
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Linear relationships in data
/// - When anomalies deviate from the main data structure
/// - High-dimensional data that can be compressed
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Components: min(n_samples, n_features) or specified
/// - Variance threshold: 0.95 (keep 95% of variance)
/// - Contamination: 0.1 (10%)
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Linear)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
    [ResearchPaper("A Geometric Framework for Unsupervised Anomaly Detection", "https://doi.org/10.1007/978-1-4615-0755-0")]
public partial class PCADetector<T> : AnomalyDetectorBase<T>
{
    /// <summary>Eigenvalues below this threshold are treated as zero to avoid division by near-zero values in Mahalanobis distance.</summary>
    private const double EigenvalueFloor = 1e-10;

    private readonly int? _nComponents;
    private readonly double _varianceThreshold;
    private int _fittedComponents;
    [Buffer]
    private Vector<T>? _mean;
    private Vector<T> FittedMean => _mean ?? throw new InvalidOperationException("PCADetector not fitted. Call Fit() first.");
    private Matrix<T>? _components;
    private Matrix<T> FittedComponents => _components ?? throw new InvalidOperationException("PCADetector not fitted. Call Fit() first.");
    [Buffer]
    private Vector<T>? _explainedVariance;
    private Vector<T> FittedExplainedVariance => _explainedVariance ?? throw new InvalidOperationException("PCADetector not fitted. Call Fit() first.");

    /// <summary>
    /// Gets the number of components used.
    /// </summary>
    public int NComponents => _fittedComponents;

    /// <summary>
    /// Gets the variance threshold.
    /// </summary>
    public double VarianceThreshold => _varianceThreshold;

    /// <summary>
    /// Creates a new PCA anomaly detector.
    /// </summary>
    /// <param name="nComponents">
    /// Number of principal components. If null, determined by variance threshold.
    /// </param>
    /// <param name="varianceThreshold">
    /// Proportion of variance to retain. Default is 0.95 (95%).
    /// Used when nComponents is not specified.
    /// </param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public PCADetector(int? nComponents = null, double varianceThreshold = 0.95,
        double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (nComponents.HasValue && nComponents.Value < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(nComponents),
                "NComponents must be at least 1 if specified.");
        }

        if (varianceThreshold <= 0 || varianceThreshold > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(varianceThreshold),
                "VarianceThreshold must be between 0 (exclusive) and 1 (inclusive). Recommended is 0.95.");
        }

        _nComponents = nComponents;
        _varianceThreshold = varianceThreshold;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        var engine = Engine;
        int n = X.Rows;
        int d = X.Columns;

        // Compute mean and center data
        var data = AnomalyTensorOps.ToTensor(X);
        var mean = engine.ReduceMean(data, new[] { 0 }, false);
        _mean = mean.ToVector();
        var centered = AnomalyTensorOps.SubtractRow(data, mean);

        // Compute covariance matrix (sample, n - 1) and its eigendecomposition (largest eigenvalues first)
        var covariance = AnomalyTensorOps.Covariance(centered, NumOps.FromDouble(n - 1));
        var (eigenvalues, eigenvectors) = AnomalyTensorOps.SymmetricEigenDescending(covariance);

        // Determine number of components
        _fittedComponents = _nComponents ?? DetermineComponents(eigenvalues);
        _fittedComponents = Math.Min(_fittedComponents, d);

        // Store components (top eigenvectors)
        _components = engine.TensorSlice(eigenvectors, new[] { 0, 0 }, new[] { _fittedComponents, d }).ToMatrix();
        _explainedVariance = engine.TensorSlice(eigenvalues, new[] { 0 }, new[] { _fittedComponents }).ToVector();

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    /// <inheritdoc/>
    public override Vector<T> ScoreAnomalies(Matrix<T> X)
    {
        EnsureFitted();
        return ScoreAnomaliesInternal(X);
    }

    private Vector<T> ScoreAnomaliesInternal(Matrix<T> X)
    {
        ValidateInput(X);

        var engine = Engine;
        var components = AnomalyTensorOps.ToTensor(FittedComponents);
        var explained = FittedExplainedVariance.AsTensor();

        // Center the points, project onto the components and reconstruct from the projection
        var centered = AnomalyTensorOps.SubtractRow(AnomalyTensorOps.ToTensor(X), FittedMean.AsTensor());
        var projected = engine.TensorMatMul(centered, engine.TensorTranspose(components));

        // Only components whose eigenvalue is above the floor reconstruct the point: deviations along
        // retained zero-variance directions stay in the residual instead of vanishing from both terms
        int k = _fittedComponents;
        var significant = engine.TensorGreaterThan(explained, NumOps.FromDouble(EigenvalueFloor));
        var reconstructed = engine.TensorMatMul(AnomalyTensorOps.MultiplyRow(projected, significant), components);

        // Reconstruction error (residual from unretained components)
        var reconstructionError = engine.ReduceSum(
            engine.TensorSquare(engine.TensorSubtract(centered, reconstructed)), new[] { 1 }, false);

        // Mahalanobis distance in PCA space (distance along retained components, weighted by inverse
        // eigenvalues, over the eigenvalues above the floor). This catches outliers that lie far from the
        // mean along principal component directions, even when reconstruction error is low.
        var inverse = engine.TensorWhere(significant,
            engine.TensorReciprocal(engine.TensorWhere(significant, explained, Tensor<T>.CreateOnes(new[] { k }))),
            Tensor<T>.CreateZeros(new[] { k }));
        var mahalanobis = engine.ReduceSum(
            AnomalyTensorOps.MultiplyRow(engine.TensorSquare(projected), inverse), new[] { 1 }, false);

        // Combined score: Hotelling's T² (Mahalanobis in PC space) + SPE/Q (reconstruction error).
        return engine.TensorAdd(mahalanobis, reconstructionError).ToVector();
    }

    private int DetermineComponents(Tensor<T> eigenvalues)
    {
        // Find number of components to explain variance threshold: the first count whose cumulative
        // positive variance reaches the threshold fraction of the total positive variance
        var engine = Engine;
        int count = eigenvalues.Length;
        var positive = engine.TensorWhere(engine.TensorGreaterThan(eigenvalues, NumOps.Zero), eigenvalues,
            Tensor<T>.CreateZeros(new[] { count }));
        var cumulative = engine.TensorCumSum(positive, 0);
        T totalVariance = engine.TensorSum(engine.TensorSlice(cumulative, new[] { count - 1 }, new[] { 1 }));
        if (!NumOps.GreaterThan(totalVariance, NumOps.Zero))
        {
            return count;
        }

        var reached = engine.ScalarMinusTensor(NumOps.One, engine.TensorLessThan(
            engine.TensorDivideScalar(cumulative, totalVariance), NumOps.FromDouble(_varianceThreshold)));
        if (!NumOps.GreaterThan(engine.TensorSum(reached), NumOps.Zero))
        {
            return count;
        }

        return AnomalyTensorOps.FirstArgMax(reached) + 1;
    }
}
