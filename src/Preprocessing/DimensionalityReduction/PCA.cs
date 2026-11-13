using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Principal Component Analysis for dimensionality reduction.
/// </summary>
/// <remarks>
/// <para>
/// PCA finds the directions of maximum variance in the data and projects the data
/// onto these principal components. This reduces dimensionality while preserving
/// as much variance as possible.
/// </para>
/// <para>
/// PCA is useful for:
/// - Reducing the number of features while retaining most information
/// - Removing multicollinearity between features
/// - Visualizing high-dimensional data
/// - Noise reduction
/// </para>
/// <para><b>For Beginners:</b> PCA transforms your features into new features called
/// "principal components" that are:
/// - Uncorrelated with each other
/// - Ordered by importance (how much variance they explain)
/// - Linear combinations of your original features
///
/// Example: 100 features might be reduced to 10 principal components that capture
/// 95% of the information in your data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.DimensionReducer)]
[PipelineStage(PipelineStage.Preprocessing)]
public class PCA<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int? _nComponents;
    private readonly double? _varianceRatio;
    private readonly bool _whiten;

    // Fitted parameters
    private Vector<T>? _mean;
    private Matrix<T>? _components; // Principal component vectors (eigenvectors), one per row
    private Vector<T>? _explainedVariance;
    private Vector<T>? _explainedVarianceRatio;
    private Vector<T>? _singularValues;
    private Tensor<T>? _whitenScale; // sqrt(explained variance) of the kept components, 1 where ~0
    private int _nFeaturesIn;
    private int _nComponentsOut;

    /// <summary>
    /// Gets the number of components to keep.
    /// </summary>
    public int? NComponents => _nComponents;

    /// <summary>
    /// Gets the target variance ratio to retain.
    /// </summary>
    public double? VarianceRatio => _varianceRatio;

    /// <summary>
    /// Gets whether whitening is applied.
    /// </summary>
    public bool Whiten => _whiten;

    /// <summary>
    /// Gets the mean of each feature.
    /// </summary>
    public Vector<T>? Mean => _mean;

    /// <summary>
    /// Gets the principal components (each row is a component).
    /// </summary>
    public Matrix<T>? Components => _components;

    /// <summary>
    /// Gets the explained variance for each component.
    /// </summary>
    public Vector<T>? ExplainedVariance => _explainedVariance;

    /// <summary>
    /// Gets the explained variance ratio for each component.
    /// </summary>
    public Vector<T>? ExplainedVarianceRatio => _explainedVarianceRatio;

    /// <summary>
    /// Gets the singular values.
    /// </summary>
    public Vector<T>? SingularValues => _singularValues;

    /// <summary>
    /// Gets the number of components after fitting.
    /// </summary>
    public int NComponentsOut => _nComponentsOut;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Creates a new instance of <see cref="PCA{T}"/>.
    /// </summary>
    /// <param name="nComponents">Number of components to keep. If null, keeps all.</param>
    /// <param name="varianceRatio">Keep enough components to explain this variance ratio (0-1). Overrides nComponents if set.</param>
    /// <param name="whiten">If true, scale components to unit variance. Defaults to false.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public PCA(
        int? nComponents = null,
        double? varianceRatio = null,
        bool whiten = false,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nComponents.HasValue && nComponents.Value < 1)
        {
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));
        }

        if (varianceRatio.HasValue && (varianceRatio.Value <= 0 || varianceRatio.Value > 1))
        {
            throw new ArgumentException("Variance ratio must be between 0 and 1.", nameof(varianceRatio));
        }

        _nComponents = nComponents;
        _varianceRatio = varianceRatio;
        _whiten = whiten;
    }

    /// <summary>
    /// Fits PCA by computing principal components.
    /// </summary>
    /// <param name="data">The training data matrix.</param>
    protected override void FitCore(Matrix<T> data)
    {
        var engine = Engine;
        _nFeaturesIn = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Center the data
        var x = Tensor<T>.FromMatrix(data);
        var mean = DimensionalityReductionOps.ColumnMeans(x);
        var centered = DimensionalityReductionOps.SubtractRow(x, mean);

        // Sample covariance (n - 1 denominator)
        var covariance = engine.TensorDivideScalar(
            DimensionalityReductionOps.TransposeMatMul(centered, centered), NumOps.FromDouble(n - 1));

        // Eigen-decomposition, eigenvalues descending; negative rounding noise clamps to zero
        var (eigenvalues, eigenvectors) = DimensionalityReductionOps.SymmetricEigen(covariance, descending: true);
        eigenvalues = engine.TensorClampMin(eigenvalues, NumOps.Zero);

        var explainedVarianceRatio = engine.TensorDivideScalar(eigenvalues, engine.TensorSum(eigenvalues));

        if (_varianceRatio.HasValue)
        {
            // scikit-learn: searchsorted(ratio_cumsum, target, side="right") + 1, so a cumulative ratio exactly
            // equal to the target keeps one more component
            var cumulative = engine.TensorCumSum(explainedVarianceRatio, 0);
            var target = DimensionalityReductionOps.Scalar(NumOps.FromDouble(_varianceRatio.Value));
            int atMost = engine.TensorSearchSorted(cumulative, target, true)[0];
            _nComponentsOut = Math.Min(atMost + 1, p);
        }
        else if (_nComponents.HasValue)
        {
            _nComponentsOut = Math.Min(_nComponents.Value, p);
        }
        else
        {
            _nComponentsOut = p;
        }

        var keptVariance = engine.TensorNarrow(eigenvalues, 0, 0, _nComponentsOut);

        _mean = mean.ToVector();
        _explainedVariance = eigenvalues.ToVector();
        _explainedVarianceRatio = explainedVarianceRatio.ToVector();
        _components = engine.TensorTranspose(
            DimensionalityReductionOps.FirstColumns(eigenvectors, _nComponentsOut)).ToMatrix();
        _singularValues = engine.TensorSqrt(engine.TensorMultiplyScalar(keptVariance, NumOps.FromDouble(n - 1))).ToVector();

        // Whitening scales each component to unit variance: divide by sqrt(explained variance), as in
        // scikit-learn's PCA(whiten=True). Components with (near) zero variance are left unscaled.
        var componentStd = engine.TensorSqrt(keptVariance);
        _whitenScale = engine.TensorWhere(
            engine.TensorGreaterThan(componentStd, DimensionalityReductionOps.Tolerance<T>(1e-10)),
            componentStd,
            engine.TensorOnesLike(componentStd));
    }

    /// <summary>
    /// Transforms data by projecting onto principal components.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The transformed data.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_mean is null || _components is null || _whitenScale is null)
        {
            throw new InvalidOperationException("PCA has not been fitted.");
        }

        var engine = Engine;
        var centered = DimensionalityReductionOps.SubtractRow(Tensor<T>.FromMatrix(data), Tensor<T>.FromVector(_mean));
        var projected = DimensionalityReductionOps.MatMulTranspose(centered, Tensor<T>.FromMatrix(_components));

        if (_whiten)
        {
            projected = engine.TensorDivide(projected, DimensionalityReductionOps.AsRow(_whitenScale));
        }

        return projected.ToMatrix();
    }

    /// <summary>
    /// Transforms data back to original space.
    /// </summary>
    /// <param name="data">The transformed data.</param>
    /// <returns>Data in original feature space.</returns>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_mean is null || _components is null || _whitenScale is null)
        {
            throw new InvalidOperationException("PCA has not been fitted.");
        }

        var engine = Engine;
        var scores = DimensionalityReductionOps.FirstColumns(Tensor<T>.FromMatrix(data), _nComponentsOut);

        if (_whiten)
        {
            scores = DimensionalityReductionOps.ScaleColumns(scores, _whitenScale);
        }

        var reconstructed = engine.TensorMatMul(scores, Tensor<T>.FromMatrix(_components));
        return DimensionalityReductionOps.AddRow(reconstructed, Tensor<T>.FromVector(_mean)).ToMatrix();
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        var names = new string[_nComponentsOut];
        for (int i = 0; i < _nComponentsOut; i++)
        {
            names[i] = $"PC{i + 1}";
        }
        return names;
    }
}
