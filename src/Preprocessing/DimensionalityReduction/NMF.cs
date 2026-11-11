using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Non-negative Matrix Factorization for dimensionality reduction.
/// </summary>
/// <remarks>
/// <para>
/// NMF factorizes a non-negative matrix V into two non-negative matrices W and H
/// such that V ≈ W × H. This is useful for data that is naturally non-negative
/// (e.g., images, text term frequencies, audio spectrograms).
/// </para>
/// <para>
/// Unlike PCA, NMF produces parts-based representations where each component
/// represents an additive combination of non-negative features.
/// </para>
/// <para><b>For Beginners:</b> NMF learns "building blocks" from your data:
/// - For images: NMF might learn eyes, noses, mouths as separate components
/// - For text: NMF might learn topics as combinations of words
/// - Unlike PCA, components are always positive (additive, never subtractive)
///
/// Think of it as finding the parts that, when added together, recreate your data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.DimensionReducer)]
[PipelineStage(PipelineStage.Preprocessing)]
public class NMF<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly NMFInit _init;
    private readonly NMFSolver _solver;
    private readonly double _beta;
    private readonly double _alpha;
    private readonly double _l1Ratio;
    private readonly int _maxIterations;
    private readonly double _tolerance;
    private readonly int _randomState;

    // Fitted parameters
    private Matrix<T>? _components; // H matrix (nComponents × nFeatures)
    private T _reconstructionError = MathHelper.GetNumericOperations<T>().Zero;
    private int _nIterations;
    private int _nFeaturesIn;

    /// <summary>
    /// Gets the number of components.
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Gets the solver type.
    /// </summary>
    public NMFSolver Solver => _solver;

    /// <summary>
    /// Gets the learned components (H matrix).
    /// </summary>
    public Matrix<T>? Components => _components;

    /// <summary>
    /// Gets the final reconstruction error.
    /// </summary>
    public T ReconstructionError => _reconstructionError;

    /// <summary>
    /// Gets the number of iterations performed.
    /// </summary>
    public int NIterations => _nIterations;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Creates a new instance of <see cref="NMF{T}"/>.
    /// </summary>
    /// <param name="nComponents">Number of components. Defaults to 2.</param>
    /// <param name="init">Initialization method. Defaults to Random.</param>
    /// <param name="solver">Solver algorithm. Defaults to MU (multiplicative update).</param>
    /// <param name="beta">Beta parameter for beta-divergence. Defaults to 2 (Frobenius norm).</param>
    /// <param name="alpha">Regularization parameter. Defaults to 0.</param>
    /// <param name="l1Ratio">L1 ratio in regularization. Defaults to 0.</param>
    /// <param name="maxIterations">Maximum iterations. Defaults to 200.</param>
    /// <param name="tolerance">Convergence tolerance. Defaults to 1e-4.</param>
    /// <param name="randomState">Random seed. Defaults to 0.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public NMF(
        int nComponents = 2,
        NMFInit init = NMFInit.Random,
        NMFSolver solver = NMFSolver.MU,
        double beta = 2.0,
        double alpha = 0.0,
        double l1Ratio = 0.0,
        int maxIterations = 200,
        double tolerance = 1e-4,
        int randomState = 0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nComponents < 1)
        {
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));
        }

        _nComponents = nComponents;
        _init = init;
        _solver = solver;
        _beta = beta;
        _alpha = alpha;
        _l1Ratio = l1Ratio;
        _maxIterations = maxIterations;
        _tolerance = tolerance;
        _randomState = randomState;
    }

    /// <summary>
    /// Fits NMF by learning the components matrix.
    /// </summary>
    /// <param name="data">The training data matrix (must be non-negative).</param>
    protected override void FitCore(Matrix<T> data)
    {
        _nFeaturesIn = data.Columns;
        int k = _nComponents;

        // Validate non-negativity
        var v = Tensor<T>.FromMatrix(data);
        ThrowIfNegative(v);

        // Initialize W and H
        var (w, h) = InitializeMatrices(v, k);

        // Run solver
        if (_solver == NMFSolver.MU)
        {
            (w, h, _nIterations) = SolveMultiplicativeUpdate(v, w, h);
        }
        else
        {
            (w, h, _nIterations) = SolveCoordinateDescent(v, w, h);
        }

        // Store components (H)
        _components = h.ToMatrix();

        // Compute reconstruction error
        _reconstructionError = ComputeReconstructionError(v, w, h);
    }

    /// <summary>Throws for the first (row-major) negative entry, naming its position and value.</summary>
    private void ThrowIfNegative(Tensor<T> v)
    {
        var engine = Engine;
        if (!DimensionalityReductionOps.Below(engine.TensorMinValue(v), NumOps.Zero))
        {
            return;
        }

        int p = v.Shape[1];
        int flat = DimensionalityReductionOps.FirstArgMax(
            engine.Reshape(engine.TensorLessThan(v, NumOps.Zero), new[] { v.Length }));
        throw new ArgumentException(
            $"NMF requires non-negative input. Found negative value at ({flat / p}, {flat % p}): {v[flat / p, flat % p]}");
    }

    private (Tensor<T> W, Tensor<T> H) InitializeMatrices(Tensor<T> v, int k)
    {
        var engine = Engine;
        int n = v.Shape[0];
        int p = v.Shape[1];
        var random = RandomHelper.CreateSeededRandom(_randomState);

        switch (_init)
        {
            case NMFInit.NNDSVD:
                return InitializeNNDSVD(v, k);

            case NMFInit.NNDSVDa:
            {
                // NNDSVD with (near) zero entries filled by the data average
                var (w, h) = InitializeNNDSVD(v, k);
                T avg = engine.TensorMean(v);
                T floor = NumOps.FromDouble(1e-10);
                w = engine.TensorWhere(engine.TensorLessThan(w, floor), engine.TensorFullLike(w, avg), w);
                h = engine.TensorWhere(engine.TensorLessThan(h, floor), engine.TensorFullLike(h, avg), h);
                return (w, h);
            }

            default:
            {
                // Random initialization scaled by sqrt(mean(V) / k): W draws first, then H
                T scale = NumOps.Sqrt(NumOps.Divide(engine.TensorMean(v), NumOps.FromDouble(k)));
                var w = engine.TensorAbs(engine.TensorMultiplyScalar(DimensionalityReductionOps.Uniform<T>(random, n, k), scale));
                var h = engine.TensorAbs(engine.TensorMultiplyScalar(DimensionalityReductionOps.Uniform<T>(random, k, p), scale));
                return (w, h);
            }
        }
    }

    /// <summary>
    /// NNDSVD-style initialization: H rows are |leading eigenvectors of V^T V| (zero rows beyond rank p) and
    /// W = max(0, V H^T).
    /// </summary>
    private (Tensor<T> W, Tensor<T> H) InitializeNNDSVD(Tensor<T> v, int k)
    {
        var engine = Engine;
        int p = v.Shape[1];
        int available = Math.Min(k, p);

        var (_, eigenvectors) = DimensionalityReductionOps.SymmetricEigen(
            DimensionalityReductionOps.TransposeMatMul(v, v), descending: true);
        var h = engine.TensorAbs(engine.TensorTranspose(DimensionalityReductionOps.FirstColumns(eigenvectors, available)));
        if (available < k)
        {
            h = engine.TensorConcatenate(new[] { h, new Tensor<T>(new[] { k - available, p }) }, 0);
        }

        var w = engine.TensorClampMin(DimensionalityReductionOps.MatMulTranspose(v, h), NumOps.Zero);
        return (w, h);
    }

    private (Tensor<T> W, Tensor<T> H, int Iterations) SolveMultiplicativeUpdate(Tensor<T> v, Tensor<T> w, Tensor<T> h)
    {
        var engine = Engine;
        T eps = NumOps.FromDouble(1e-10);
        T betaMinus2 = NumOps.FromDouble(_beta - 2);
        T betaMinus1 = NumOps.FromDouble(_beta - 1);
        T l1Reg = NumOps.FromDouble(_alpha * _l1Ratio);
        T l2Reg = NumOps.FromDouble(_alpha * (1 - _l1Ratio));
        T prevError = NumOps.MaxValue;

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            // Update H: H *= W^T (WH^(beta-2) * V) / (W^T WH^(beta-1) + l1 + l2 H + eps)
            var wh = engine.TensorClampMin(engine.TensorMatMul(w, h), eps);
            var numeratorH = DimensionalityReductionOps.TransposeMatMul(
                w, engine.TensorMultiply(engine.TensorPower(wh, betaMinus2), v));
            var denominatorH = engine.TensorAddScalar(
                engine.TensorAdd(
                    DimensionalityReductionOps.TransposeMatMul(w, engine.TensorPower(wh, betaMinus1)),
                    engine.TensorMultiplyScalar(h, l2Reg)),
                NumOps.Add(l1Reg, eps));
            h = engine.TensorClampMin(engine.TensorMultiply(h, engine.TensorDivide(numeratorH, denominatorH)), eps);

            // Update W: W *= (WH^(beta-2) * V) H^T / (WH^(beta-1) H^T + l1 + l2 W + eps)
            wh = engine.TensorClampMin(engine.TensorMatMul(w, h), eps);
            var numeratorW = DimensionalityReductionOps.MatMulTranspose(
                engine.TensorMultiply(engine.TensorPower(wh, betaMinus2), v), h);
            var denominatorW = engine.TensorAddScalar(
                engine.TensorAdd(
                    DimensionalityReductionOps.MatMulTranspose(engine.TensorPower(wh, betaMinus1), h),
                    engine.TensorMultiplyScalar(w, l2Reg)),
                NumOps.Add(l1Reg, eps));
            w = engine.TensorClampMin(engine.TensorMultiply(w, engine.TensorDivide(numeratorW, denominatorW)), eps);

            // Check convergence
            T error = ComputeReconstructionError(v, w, h);
            if (RelativeChangeBelow(prevError, error))
            {
                return (w, h, iter + 1);
            }

            prevError = error;
        }

        return (w, h, _maxIterations);
    }

    private (Tensor<T> W, Tensor<T> H, int Iterations) SolveCoordinateDescent(Tensor<T> v, Tensor<T> w, Tensor<T> h)
    {
        // Simplified coordinate descent
        var engine = Engine;
        int k = h.Shape[0];
        T eps = NumOps.FromDouble(1e-10);
        T l2Reg = NumOps.FromDouble(_alpha * (1 - _l1Ratio));
        T l1Penalty = NumOps.FromDouble(_alpha * _l1Ratio);
        T prevError = NumOps.MaxValue;

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            // Update H row by row: H[c] = max(0, S(W[:, c]^T R_c, l1) / (eps + |W[:, c]|^2 + l2))
            for (int c = 0; c < k; c++)
            {
                var wColumn = engine.TensorNarrow(w, 1, c, 1);                              // [n, 1]
                var residual = ResidualWithout(v, w, h, c);                                 // [n, p]
                var numerator = DimensionalityReductionOps.TransposeMatMul(wColumn, residual); // [1, p]
                var denominator = engine.TensorAddScalar(
                    engine.ReduceSum(engine.TensorSquare(wColumn), new[] { 0 }, true), NumOps.Add(eps, l2Reg));
                var updated = engine.TensorClampMin(
                    engine.TensorDivide(DimensionalityReductionOps.SoftThreshold(numerator, l1Penalty), denominator), NumOps.Zero);
                h = DimensionalityReductionOps.WithRow(h, c, updated);
            }

            // Update W column by column: W[:, c] = max(0, S(R_c H[c]^T, l1) / (eps + |H[c]|^2 + l2))
            for (int c = 0; c < k; c++)
            {
                var hRow = engine.TensorNarrow(h, 0, c, 1);                                 // [1, p]
                var residual = ResidualWithout(v, w, h, c);
                var numerator = DimensionalityReductionOps.MatMulTranspose(residual, hRow); // [n, 1]
                var denominator = engine.TensorAddScalar(
                    engine.ReduceSum(engine.TensorSquare(hRow), new[] { 1 }, true), NumOps.Add(eps, l2Reg));
                var updated = engine.TensorClampMin(
                    engine.TensorDivide(DimensionalityReductionOps.SoftThreshold(numerator, l1Penalty), denominator), NumOps.Zero);
                w = DimensionalityReductionOps.WithColumn(w, c, updated);
            }

            // Check convergence
            T error = ComputeReconstructionError(v, w, h);
            if (RelativeChangeBelow(prevError, error))
            {
                return (w, h, iter + 1);
            }

            prevError = error;
        }

        return (w, h, _maxIterations);
    }

    /// <summary>V - W H + W[:, c] H[c]: the residual that component c is fit to.</summary>
    private Tensor<T> ResidualWithout(Tensor<T> v, Tensor<T> w, Tensor<T> h, int c)
    {
        var engine = Engine;
        return engine.TensorAdd(
            engine.TensorSubtract(v, engine.TensorMatMul(w, h)),
            engine.TensorMatMul(engine.TensorNarrow(w, 1, c, 1), engine.TensorNarrow(h, 0, c, 1)));
    }

    /// <summary>|previous - current| / (previous + 1e-10) &lt; tolerance.</summary>
    private bool RelativeChangeBelow(T previous, T current)
    {
        T relChange = NumOps.Divide(
            NumOps.Abs(NumOps.Subtract(previous, current)), NumOps.Add(previous, NumOps.FromDouble(1e-10)));
        return NumOps.LessThan(relChange, NumOps.FromDouble(_tolerance));
    }

    /// <summary>Frobenius norm of V - W H.</summary>
    private T ComputeReconstructionError(Tensor<T> v, Tensor<T> w, Tensor<T> h)
    {
        return DimensionalityReductionOps.Norm(Engine.TensorSubtract(v, Engine.TensorMatMul(w, h)));
    }

    /// <summary>
    /// Transforms data by finding the optimal W given learned H.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The coefficient matrix W.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_components is null)
        {
            throw new InvalidOperationException("NMF has not been fitted.");
        }

        var engine = Engine;
        int n = data.Rows;
        int p = data.Columns;
        int k = _components.Rows;

        if (p != _nFeaturesIn)
        {
            throw new ArgumentException(
                $"Input data has {p} features, but model was fitted with {_nFeaturesIn} features.",
                nameof(data));
        }

        var v = Tensor<T>.FromMatrix(data);
        if (DimensionalityReductionOps.Below(engine.TensorMinValue(v), NumOps.Zero))
        {
            throw new ArgumentException(
                "NMF requires non-negative input data. Negative values were detected.",
                nameof(data));
        }

        // Initialize W randomly (scaled by sqrt(mean(V) / k))
        var h = Tensor<T>.FromMatrix(_components);
        T scale = NumOps.Sqrt(NumOps.Divide(engine.TensorMean(v), NumOps.FromDouble(k)));
        var random = RandomHelper.CreateSeededRandom(_randomState);
        var w = engine.TensorAbs(engine.TensorMultiplyScalar(DimensionalityReductionOps.Uniform<T>(random, n, k), scale));

        // Multiplicative updates for W with H fixed: W *= (V H^T) / (W H H^T + eps)
        T eps = NumOps.FromDouble(1e-10);
        var vht = DimensionalityReductionOps.MatMulTranspose(v, h);
        var hht = DimensionalityReductionOps.MatMulTranspose(h, h);
        for (int iter = 0; iter < 100; iter++)
        {
            var whht = engine.TensorAddScalar(engine.TensorMatMul(w, hht), eps);
            w = engine.TensorClampMin(engine.TensorMultiply(w, engine.TensorDivide(vht, whht)), eps);
        }

        return w.ToMatrix();
    }

    /// <summary>
    /// Reconstructs data from coefficient matrix.
    /// </summary>
    /// <param name="data">The coefficient matrix W.</param>
    /// <returns>Reconstructed data in original space.</returns>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_components is null)
        {
            throw new InvalidOperationException("NMF has not been fitted.");
        }

        int inputColumns = data.Columns;
        int k = _components.Rows;

        // Validate that input has k columns (coefficient matrix W should be n x k)
        if (inputColumns != k)
        {
            throw new ArgumentException(
                $"Input data has {inputColumns} columns, but model expects coefficient matrix " +
                $"with {k} columns (number of components).",
                nameof(data));
        }

        return Engine.TensorMatMul(Tensor<T>.FromMatrix(data), Tensor<T>.FromMatrix(_components)).ToMatrix();
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
            names[i] = $"Component{i + 1}";
        }
        return names;
    }
}

/// <summary>
/// Specifies the initialization method for NMF.
/// </summary>
public enum NMFInit
{
    /// <summary>
    /// Random non-negative initialization.
    /// </summary>
    Random,

    /// <summary>
    /// Non-negative Double SVD initialization (faster convergence).
    /// </summary>
    NNDSVD,

    /// <summary>
    /// NNDSVD with zeros filled with small values (better for sparse data).
    /// </summary>
    NNDSVDa
}

/// <summary>
/// Specifies the solver for NMF optimization.
/// </summary>
public enum NMFSolver
{
    /// <summary>
    /// Multiplicative Update solver (Lee and Seung).
    /// </summary>
    MU,

    /// <summary>
    /// Coordinate Descent solver.
    /// </summary>
    CD
}
