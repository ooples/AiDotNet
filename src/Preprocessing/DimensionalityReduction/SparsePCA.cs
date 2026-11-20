using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Sparse Principal Component Analysis using L1 regularization.
/// </summary>
/// <remarks>
/// <para>
/// SparsePCA finds sparse principal components by applying L1 regularization
/// (LASSO-like penalty) to the component loadings. This results in components
/// where many loadings are exactly zero, making them more interpretable.
/// </para>
/// <para>
/// Unlike standard PCA where each component is a combination of ALL features,
/// sparse PCA produces components that depend on only a subset of features.
/// </para>
/// <para><b>For Beginners:</b> Sparse PCA creates "simpler" principal components:
/// - Standard PCA: Component = 0.3*Feature1 + 0.2*Feature2 + 0.1*Feature3 + ...
/// - Sparse PCA: Component = 0.5*Feature1 + 0*Feature2 + 0.4*Feature3 + 0*...
/// - Zeros make it easier to interpret what each component represents
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.DimensionReducer)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SparsePCA<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly double _alpha;
    private readonly double _ridge;
    private readonly int _maxIter;
    private readonly double _tol;
    private readonly int? _randomState;

    // Fitted parameters
    private Vector<T>? _mean;
    private Matrix<T>? _components;
    private Vector<T>? _error;
    private int _nFeaturesIn;

    /// <summary>
    /// Gets the number of components.
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Gets the sparsity regularization parameter.
    /// </summary>
    public double Alpha => _alpha;

    /// <summary>
    /// Gets the ridge regularization parameter.
    /// </summary>
    public double Ridge => _ridge;

    /// <summary>
    /// Gets the mean of each feature.
    /// </summary>
    public Vector<T>? Mean => _mean;

    /// <summary>
    /// Gets the sparse components (each row is a component).
    /// </summary>
    public Matrix<T>? Components => _components;

    /// <summary>
    /// Gets the reconstruction error for each iteration.
    /// </summary>
    public Vector<T>? Error => _error;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Creates a new instance of <see cref="SparsePCA{T}"/>.
    /// </summary>
    /// <param name="nComponents">Number of sparse components. Defaults to 2.</param>
    /// <param name="alpha">Sparsity regularization parameter. Higher values produce sparser components. Defaults to 1.0.</param>
    /// <param name="ridge">Ridge regularization for stability. Defaults to 0.01.</param>
    /// <param name="maxIter">Maximum number of iterations. Defaults to 100.</param>
    /// <param name="tol">Convergence tolerance. Defaults to 1e-6.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public SparsePCA(
        int nComponents = 2,
        double alpha = 1.0,
        double ridge = 0.01,
        int maxIter = 100,
        double tol = 1e-6,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nComponents < 1)
        {
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));
        }

        if (alpha < 0)
        {
            throw new ArgumentException("Alpha must be non-negative.", nameof(alpha));
        }

        if (ridge < 0)
        {
            throw new ArgumentException("Ridge must be non-negative.", nameof(ridge));
        }

        _nComponents = nComponents;
        _alpha = alpha;
        _ridge = ridge;
        _maxIter = maxIter;
        _tol = tol;
        _randomState = randomState;
    }

    /// <summary>
    /// Fits Sparse PCA using coordinate descent with L1 regularization.
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        var engine = Engine;
        _nFeaturesIn = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        int k = Math.Min(_nComponents, Math.Min(n, p));

        // Center the data
        var x = Tensor<T>.FromMatrix(data);
        var mean = DimensionalityReductionOps.ColumnMeans(x);
        var centered = DimensionalityReductionOps.SubtractRow(x, mean);
        _mean = mean.ToVector();

        // Initialize components randomly (uniform in [-0.5, 0.5), unit rows)
        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSeededRandom(42);
        var components = DimensionalityReductionOps.NormalizeRows(
            engine.TensorSubtractScalar(DimensionalityReductionOps.Uniform<T>(random, k, p), NumOps.FromDouble(0.5)));

        // Alternating optimization
        var errorHistory = new List<T>();
        T tol = DimensionalityReductionOps.Tolerance<T>(_tol);

        for (int iter = 0; iter < _maxIter; iter++)
        {
            // Update codes (ridge regression) and then the dictionary (L1-penalized coordinate descent)
            var codes = UpdateCodes(centered, components);
            var oldComponents = components;
            components = UpdateDictionary(centered, codes, components);

            // Compute reconstruction error
            errorHistory.Add(ComputeReconstructionError(centered, codes, components));

            // Check convergence
            if (DimensionalityReductionOps.Below(DimensionalityReductionOps.MaxAbsDifference(components, oldComponents), tol))
            {
                break;
            }
        }

        _components = components.ToMatrix();
        _error = new Vector<T>(errorHistory);
    }

    /// <summary>Ridge codes: X C^T (C C^T + ridge I)^-1.</summary>
    private Tensor<T> UpdateCodes(Tensor<T> data, Tensor<T> components)
    {
        var engine = Engine;
        int k = components.Shape[0];

        // Compute C C^T + ridge * I
        var gram = engine.TensorAdd(
            DimensionalityReductionOps.MatMulTranspose(components, components),
            engine.TensorMultiplyScalar(DimensionalityReductionOps.Identity<T>(k), NumOps.FromDouble(_ridge)));

        // codes = (X C^T) gram^-1 (gram is symmetric, so solve gram Z^T = (X C^T)^T)
        var dataProj = DimensionalityReductionOps.MatMulTranspose(data, components);
        return engine.TensorTranspose(DimensionalityReductionOps.Solve(gram, engine.TensorTranspose(dataProj)));
    }

    /// <summary>
    /// One sweep of block coordinate descent over the components: each row is refit to its residual with an L1
    /// soft threshold and renormalized; a component whose codes are all ~0 is left as it was.
    /// </summary>
    private Tensor<T> UpdateDictionary(Tensor<T> data, Tensor<T> codes, Tensor<T> components)
    {
        var engine = Engine;
        int k = components.Shape[0];
        var alpha = DimensionalityReductionOps.Scalar(NumOps.FromDouble(_alpha));
        var floor = DimensionalityReductionOps.Tolerance<T>(1e-10);

        for (int c = 0; c < k; c++)
        {
            var codeColumn = engine.TensorNarrow(codes, 1, c, 1);                // [n, 1]
            var row = engine.TensorNarrow(components, 0, c, 1);                  // [1, p]

            // Residual without this component: X - codes C^T + codes[:, c] C[c, :]
            var residual = engine.TensorAdd(
                engine.TensorSubtract(data, engine.TensorMatMul(codes, components)),
                engine.TensorMatMul(codeColumn, row));

            // Coordinate update with soft thresholding: S(corr / |code|^2, alpha / |code|^2)
            var codesSquared = engine.ReduceSum(engine.TensorSquare(codeColumn), new[] { 0 }, true);   // [1, 1]
            var correlation = DimensionalityReductionOps.TransposeMatMul(codeColumn, residual);       // [1, p]
            var updated = DimensionalityReductionOps.NormalizeRows(DimensionalityReductionOps.SoftThreshold(
                engine.TensorDivide(correlation, codesSquared), engine.TensorDivide(alpha, codesSquared)));

            // Skip components whose codes vanish. TensorWhere does not broadcast its condition, so the [1, 1] flag
            // is expanded to the [1, p] row first.
            var keep = engine.TensorBroadcastTo(engine.TensorLessThan(codesSquared, floor), updated.Shape.ToArray());
            components = DimensionalityReductionOps.WithRow(components, c, engine.TensorWhere(keep, row, updated));
        }

        return components;
    }

    /// <summary>Mean squared reconstruction error of X ~ codes C.</summary>
    private T ComputeReconstructionError(Tensor<T> data, Tensor<T> codes, Tensor<T> components)
    {
        var engine = Engine;
        return engine.TensorMean(engine.TensorSquare(engine.TensorSubtract(data, engine.TensorMatMul(codes, components))));
    }

    /// <summary>
    /// Transforms the data by projecting onto sparse components.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_mean is null || _components is null)
        {
            throw new InvalidOperationException("SparsePCA has not been fitted.");
        }

        var centered = DimensionalityReductionOps.SubtractRow(Tensor<T>.FromMatrix(data), Tensor<T>.FromVector(_mean));
        return DimensionalityReductionOps.MatMulTranspose(centered, Tensor<T>.FromMatrix(_components)).ToMatrix();
    }

    /// <summary>
    /// Transforms data back to original space.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_mean is null || _components is null)
        {
            throw new InvalidOperationException("SparsePCA has not been fitted.");
        }

        int k = _components.Rows;
        var scores = DimensionalityReductionOps.FirstColumns(Tensor<T>.FromMatrix(data), k);
        var reconstructed = Engine.TensorMatMul(scores, Tensor<T>.FromMatrix(_components));
        return DimensionalityReductionOps.AddRow(reconstructed, Tensor<T>.FromVector(_mean)).ToMatrix();
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        int k = _components?.Rows ?? _nComponents;
        var names = new string[k];
        for (int i = 0; i < k; i++)
        {
            names[i] = $"SparsePC{i + 1}";
        }
        return names;
    }
}
