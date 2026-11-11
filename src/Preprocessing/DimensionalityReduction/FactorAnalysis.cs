using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Factor Analysis for dimensionality reduction with noise modeling.
/// </summary>
/// <remarks>
/// <para>
/// Factor Analysis assumes that the observed data X is generated from a set of
/// latent factors F plus feature-specific noise: X = F * W + noise.
/// Unlike PCA, Factor Analysis explicitly models unique variance (noise) for
/// each feature.
/// </para>
/// <para>
/// The model assumes:
/// - X = W * F + ε
/// - Where ε ~ N(0, Ψ) and Ψ is diagonal (unique variances)
/// - F ~ N(0, I) are the latent factors
/// </para>
/// <para><b>For Beginners:</b> Factor Analysis is like PCA but smarter about noise:
/// - PCA assumes all variance is signal
/// - Factor Analysis separates "common variance" (shared across features)
///   from "unique variance" (noise specific to each feature)
/// - Use when your features have different noise levels
/// - Popular in psychology, social sciences, and survey analysis
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.DimensionReducer)]
[PipelineStage(PipelineStage.Preprocessing)]
public class FactorAnalysis<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly int _maxIter;
    private readonly double _tol;
    private readonly FactorRotation _rotation;
    private readonly int? _randomState;

    // Fitted parameters
    private Vector<T>? _mean;
    private Matrix<T>? _components; // Factor loadings (features x factors)
    private Vector<T>? _noiseVariance; // Unique variances (psi)
    private Matrix<T>? _rotatedComponents;
    private int _nFeaturesIn;

    /// <summary>
    /// Gets the number of factors.
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Gets the rotation method.
    /// </summary>
    public FactorRotation Rotation => _rotation;

    /// <summary>
    /// Gets the mean of each feature.
    /// </summary>
    public Vector<T>? Mean => _mean;

    /// <summary>
    /// Gets the factor loadings (each column is a factor).
    /// </summary>
    public Matrix<T>? Components => _rotation == FactorRotation.None ? _components : _rotatedComponents;

    /// <summary>
    /// Gets the unique variance (noise) for each feature.
    /// </summary>
    public Vector<T>? NoiseVariance => _noiseVariance;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="FactorAnalysis{T}"/>.
    /// </summary>
    /// <param name="nComponents">Number of latent factors. Defaults to 2.</param>
    /// <param name="maxIter">Maximum EM iterations. Defaults to 1000.</param>
    /// <param name="tol">Convergence tolerance. Defaults to 1e-4.</param>
    /// <param name="rotation">Factor rotation method. Defaults to None.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public FactorAnalysis(
        int nComponents = 2,
        int maxIter = 1000,
        double tol = 1e-4,
        FactorRotation rotation = FactorRotation.None,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nComponents < 1)
        {
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));
        }

        _nComponents = nComponents;
        _maxIter = maxIter;
        _tol = tol;
        _rotation = rotation;
        _randomState = randomState;
    }

    /// <summary>
    /// Fits Factor Analysis using the EM algorithm.
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        var engine = Engine;
        _nFeaturesIn = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        if (n < 2)
        {
            throw new ArgumentException("Factor Analysis requires at least 2 samples.", nameof(data));
        }

        int k = Math.Min(_nComponents, p);

        // Center the data
        var x = Tensor<T>.FromMatrix(data);
        var mean = DimensionalityReductionOps.ColumnMeans(x);
        var centered = DimensionalityReductionOps.SubtractRow(x, mean);
        _mean = mean.ToVector();

        // Compute sample covariance matrix
        var s = engine.TensorDivideScalar(DimensionalityReductionOps.TransposeMatMul(centered, centered), NumOps.FromDouble(n - 1));
        var sDiagonal = engine.TensorDiagonal(s);

        // Initialize loadings using PCA: W = U_k sqrt(max(lambda, 1e-10))
        var (eigenvalues, eigenvectors) = DimensionalityReductionOps.SymmetricEigen(s, descending: true);
        var scale = engine.TensorSqrt(engine.TensorClampMin(engine.TensorNarrow(eigenvalues, 0, 0, k), NumOps.FromDouble(1e-10)));
        var loadings = DimensionalityReductionOps.ScaleColumns(DimensionalityReductionOps.FirstColumns(eigenvectors, k), scale);

        // Initialize noise variances: psi = max(S_ii - communality, 1e-6)
        T psiFloor = NumOps.FromDouble(1e-6);
        var noiseVariance = engine.TensorClampMin(
            engine.TensorSubtract(sDiagonal, DimensionalityReductionOps.RowSums(engine.TensorSquare(loadings))), psiFloor);

        T tol = NumOps.FromDouble(_tol);
        var identity = DimensionalityReductionOps.Identity<T>(k);
        var count = DimensionalityReductionOps.Scalar(NumOps.FromDouble(n));

        // EM algorithm
        for (int iter = 0; iter < _maxIter; iter++)
        {
            var oldLoadings = loadings;

            // E-step: W^T Psi^-1, M = I + W^T Psi^-1 W, E[F] = Xc (M^-1 W^T Psi^-1)^T
            var wtPsiInv = engine.TensorTranspose(engine.TensorDivide(loadings, DimensionalityReductionOps.AsColumn(noiseVariance)));
            var m = engine.TensorAdd(engine.TensorMatMul(wtPsiInv, loadings), identity);
            var projection = DimensionalityReductionOps.Solve(m, wtPsiInv);                          // M^-1 W^T Psi^-1 [k, p]
            var expectedF = DimensionalityReductionOps.MatMulTranspose(centered, projection);       // [n, k]

            // E[F F^T] = n M^-1 + E[F]^T E[F]
            var expectedFFt = engine.TensorAdd(
                engine.TensorMultiply(DimensionalityReductionOps.Inverse(m), count),
                DimensionalityReductionOps.TransposeMatMul(expectedF, expectedF));

            // E[X F^T] = Xc^T E[F]
            var expectedXFt = DimensionalityReductionOps.TransposeMatMul(centered, expectedF);       // [p, k]

            // M-step: W = E[X F^T] E[F F^T]^-1 (E[F F^T] is symmetric)
            loadings = engine.TensorTranspose(DimensionalityReductionOps.Solve(expectedFFt, engine.TensorTranspose(expectedXFt)));

            // psi = max(S_ii - sum_j W_ij E[X F^T]_ij / n, 1e-6)
            var reconstructed = engine.TensorDivide(
                DimensionalityReductionOps.RowSums(engine.TensorMultiply(loadings, expectedXFt)), count);
            noiseVariance = engine.TensorClampMin(engine.TensorSubtract(sDiagonal, reconstructed), psiFloor);

            // Check convergence
            if (DimensionalityReductionOps.Below(DimensionalityReductionOps.MaxAbsDifference(loadings, oldLoadings), tol))
            {
                break;
            }
        }

        _components = loadings.ToMatrix();
        _noiseVariance = noiseVariance.ToVector();

        // Apply rotation if requested
        if (_rotation != FactorRotation.None)
        {
            _rotatedComponents = ApplyRotation(loadings).ToMatrix();
        }
    }

    private Tensor<T> ApplyRotation(Tensor<T> loadings)
    {
        switch (_rotation)
        {
            case FactorRotation.Varimax:
                return ApplyPairwiseRotation(loadings, gamma: 1.0);

            case FactorRotation.Quartimax:
                return ApplyPairwiseRotation(loadings, gamma: 0.0);

            default:
                return loadings.Clone();
        }
    }

    /// <summary>
    /// Orthomax rotation by planar rotations of every column pair (i, j), 100 sweeps. For each pair, with
    /// u = L_i^2 - L_j^2 and v = 2 L_i L_j summed over rows: angle = atan2(sum 2uv - 2 gamma (sum u)(sum v) / p,
    /// sum (u^2 - v^2) - gamma ((sum u)^2 - (sum v)^2) / p) / 4. gamma = 1 is varimax, gamma = 0 quartimax.
    /// </summary>
    private Tensor<T> ApplyPairwiseRotation(Tensor<T> loadings, double gamma)
    {
        var engine = Engine;
        int p = loadings.Shape[0];
        int k = loadings.Shape[1];
        T two = NumOps.FromDouble(2.0);
        T quarter = NumOps.FromDouble(0.25);
        T gammaOverP = NumOps.FromDouble(gamma / p);
        T twoGammaOverP = NumOps.FromDouble(2.0 * gamma / p);
        var rotated = loadings;

        for (int iter = 0; iter < 100; iter++)
        {
            for (int i = 0; i < k - 1; i++)
            {
                for (int j = i + 1; j < k; j++)
                {
                    var li = engine.TensorNarrow(rotated, 1, i, 1);
                    var lj = engine.TensorNarrow(rotated, 1, j, 1);

                    var u = engine.TensorSubtract(engine.TensorSquare(li), engine.TensorSquare(lj));
                    var v = engine.TensorMultiplyScalar(engine.TensorMultiply(li, lj), two);
                    var a = engine.ReduceSum(u, new[] { 0 }, true);
                    var b = engine.ReduceSum(v, new[] { 0 }, true);
                    var c = engine.ReduceSum(engine.TensorSubtract(engine.TensorSquare(u), engine.TensorSquare(v)), new[] { 0 }, true);
                    var d = engine.ReduceSum(engine.TensorMultiplyScalar(engine.TensorMultiply(u, v), two), new[] { 0 }, true);

                    var num = engine.TensorSubtract(d, engine.TensorMultiplyScalar(engine.TensorMultiply(a, b), twoGammaOverP));
                    var den = engine.TensorSubtract(c, engine.TensorMultiplyScalar(
                        engine.TensorSubtract(engine.TensorSquare(a), engine.TensorSquare(b)), gammaOverP));
                    var angle = engine.TensorMultiplyScalar(engine.TensorAtan2(num, den), quarter);
                    var cos = engine.TensorCos(angle);
                    var sin = engine.TensorSin(angle);

                    // Rotate columns i and j
                    var newI = engine.TensorAdd(engine.TensorMultiply(li, cos), engine.TensorMultiply(lj, sin));
                    var newJ = engine.TensorSubtract(engine.TensorMultiply(lj, cos), engine.TensorMultiply(li, sin));
                    rotated = DimensionalityReductionOps.WithColumn(DimensionalityReductionOps.WithColumn(rotated, i, newI), j, newJ);
                }
            }
        }

        return rotated;
    }

    /// <summary>
    /// Transforms the data by computing factor scores.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_mean is null || _components is null || _noiseVariance is null)
        {
            throw new InvalidOperationException("FactorAnalysis has not been fitted.");
        }

        var engine = Engine;
        var loadingsMatrix = (_rotation == FactorRotation.None ? _components : _rotatedComponents)
            ?? throw new InvalidOperationException("loadings is null where it is required.");
        var loadings = Tensor<T>.FromMatrix(loadingsMatrix);

        // Bartlett scores: (W^T Psi^-1 W)^-1 W^T Psi^-1 (x - mu)
        var wtPsiInv = engine.TensorTranspose(engine.TensorDivide(
            loadings, DimensionalityReductionOps.AsColumn(Tensor<T>.FromVector(_noiseVariance))));
        var wtPsiInvW = engine.TensorMatMul(wtPsiInv, loadings);
        var projection = DimensionalityReductionOps.Solve(wtPsiInvW, wtPsiInv);                     // [k, p]

        var centered = DimensionalityReductionOps.SubtractRow(Tensor<T>.FromMatrix(data), Tensor<T>.FromVector(_mean));
        return DimensionalityReductionOps.MatMulTranspose(centered, projection).ToMatrix();
    }

    /// <summary>
    /// Inverse transformation is not supported for Factor Analysis.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("FactorAnalysis does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        int k = _components?.Columns ?? _nComponents;
        var names = new string[k];
        for (int i = 0; i < k; i++)
        {
            names[i] = $"Factor{i + 1}";
        }
        return names;
    }
}

/// <summary>
/// Specifies the rotation method for Factor Analysis.
/// </summary>
public enum FactorRotation
{
    /// <summary>
    /// No rotation (use raw factor loadings).
    /// </summary>
    None,

    /// <summary>
    /// Varimax rotation - maximizes variance of squared loadings within each factor.
    /// Produces simpler structure with high loadings on few variables per factor.
    /// </summary>
    Varimax,

    /// <summary>
    /// Quartimax rotation - maximizes variance of squared loadings within each variable.
    /// Tends to produce a general factor.
    /// </summary>
    Quartimax
}
