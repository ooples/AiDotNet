using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Dictionary Learning for sparse representation.
/// </summary>
/// <remarks>
/// <para>
/// Dictionary Learning finds a dictionary D and sparse codes A such that
/// X ≈ D × A. The dictionary atoms (columns of D) form a basis that allows
/// sparse representation of the data.
/// </para>
/// <para>
/// Unlike PCA which enforces orthogonality, dictionary learning allows
/// overcomplete dictionaries (more atoms than dimensions) and enforces
/// sparsity on the codes.
/// </para>
/// <para><b>For Beginners:</b> Dictionary Learning is like building a "parts library":
/// - Dictionary: Collection of basic building blocks (atoms)
/// - Sparse codes: Which parts to use and how much (mostly zeros)
/// - Goal: Represent each sample using few dictionary atoms
/// - Used for: Image denoising, compression, feature extraction
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.DimensionReducer)]
[PipelineStage(PipelineStage.Preprocessing)]
public class DictionaryLearning<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly double _alpha;
    private readonly int _maxIter;
    private readonly double _tol;
    private readonly DictionaryFitAlgorithm _fitAlgorithm;
    private readonly SparseCodingAlgorithm _transformAlgorithm;
    private readonly int? _randomState;

    // Fitted parameters
    private Vector<T>? _mean;
    private Matrix<T>? _components; // Dictionary atoms (n_atoms x n_features)
    private int _nFeaturesIn;

    /// <summary>
    /// Gets the number of dictionary atoms.
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Gets the sparsity penalty.
    /// </summary>
    public double Alpha => _alpha;

    /// <summary>
    /// Gets the dictionary fitting algorithm.
    /// </summary>
    public DictionaryFitAlgorithm FitAlgorithm => _fitAlgorithm;

    /// <summary>
    /// Gets the sparse coding algorithm for transform.
    /// </summary>
    public SparseCodingAlgorithm TransformAlgorithm => _transformAlgorithm;

    /// <summary>
    /// Gets the mean of each feature.
    /// </summary>
    public Vector<T>? Mean => _mean;

    /// <summary>
    /// Gets the dictionary atoms (each row is an atom).
    /// </summary>
    public Matrix<T>? Components => _components;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Creates a new instance of <see cref="DictionaryLearning{T}"/>.
    /// </summary>
    /// <param name="nComponents">Number of dictionary atoms. Defaults to the number of features.</param>
    /// <param name="alpha">Sparsity regularization parameter. Defaults to 1.0.</param>
    /// <param name="maxIter">Maximum iterations. Defaults to 1000.</param>
    /// <param name="tol">Convergence tolerance. Defaults to 1e-6.</param>
    /// <param name="fitAlgorithm">Algorithm for dictionary learning. Defaults to CD.</param>
    /// <param name="transformAlgorithm">Algorithm for sparse coding. Defaults to LASSO.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public DictionaryLearning(
        int? nComponents = null,
        double alpha = 1.0,
        int maxIter = 1000,
        double tol = 1e-6,
        DictionaryFitAlgorithm fitAlgorithm = DictionaryFitAlgorithm.CD,
        SparseCodingAlgorithm transformAlgorithm = SparseCodingAlgorithm.LASSO,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nComponents.HasValue && nComponents.Value < 1)
        {
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));
        }

        if (alpha < 0)
        {
            throw new ArgumentException("Alpha must be non-negative.", nameof(alpha));
        }

        _nComponents = nComponents ?? 0; // Will be set during fit if 0
        _alpha = alpha;
        _maxIter = maxIter;
        _tol = tol;
        _fitAlgorithm = fitAlgorithm;
        _transformAlgorithm = transformAlgorithm;
        _randomState = randomState;
    }

    /// <summary>
    /// Fits the dictionary using alternating minimization.
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        var engine = Engine;
        _nFeaturesIn = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        int k = _nComponents > 0 ? _nComponents : p;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSeededRandom(42);

        // Compute mean and center data
        var x = Tensor<T>.FromMatrix(data);
        var mean = DimensionalityReductionOps.ColumnMeans(x);
        var centered = DimensionalityReductionOps.SubtractRow(x, mean);
        _mean = mean.ToVector();

        // Initialize dictionary: atom c starts from sample c (a random sample once c >= n) plus uniform noise
        // in [-0.05, 0.05), drawn atom by atom in the original order, then normalized
        var atoms = new Tensor<T>[k];
        T noiseScale = NumOps.FromDouble(0.1);
        T half = NumOps.FromDouble(0.5);
        for (int c = 0; c < k; c++)
        {
            int idx = c < n ? c : random.Next(n);
            var noise = engine.TensorMultiplyScalar(
                engine.TensorSubtractScalar(DimensionalityReductionOps.Uniform<T>(random, 1, p), half), noiseScale);
            atoms[c] = engine.TensorAdd(engine.TensorNarrow(centered, 0, idx, 1), noise);
        }

        var components = DimensionalityReductionOps.NormalizeRows(engine.TensorConcatenate(atoms, 0));
        T tol = DimensionalityReductionOps.Tolerance<T>(_tol);

        // Alternating optimization
        for (int iter = 0; iter < _maxIter; iter++)
        {
            var oldComponents = components;

            // Sparse coding step: update codes given dictionary
            var codes = UpdateCodes(centered, components);

            // Dictionary update step
            components = _fitAlgorithm == DictionaryFitAlgorithm.CD
                ? UpdateDictionaryCD(centered, codes, components)
                : UpdateDictionaryLeastSquares(centered, codes);

            // Check convergence
            if (DimensionalityReductionOps.Below(DimensionalityReductionOps.MaxAbsDifference(components, oldComponents), tol))
            {
                break;
            }
        }

        _components = components.ToMatrix();
    }

    private Tensor<T> UpdateCodes(Tensor<T> data, Tensor<T> components)
    {
        if (_transformAlgorithm == SparseCodingAlgorithm.OMP)
        {
            // Use Orthogonal Matching Pursuit, one sample at a time (each sample selects its own atoms)
            int n = data.Shape[0];
            var rows = new Tensor<T>[n];
            for (int i = 0; i < n; i++)
            {
                rows[i] = OMPSparseCoding(Engine.TensorNarrow(data, 0, i, 1), components);
            }

            return Engine.TensorConcatenate(rows, 0);
        }

        // Use LASSO coordinate descent for every sample at once
        return SparseCodingLasso(data, components);
    }

    /// <summary>
    /// Orthogonal Matching Pursuit for one sample x ([1, p]): greedily adds the atom most correlated with the
    /// residual and refits the selected atoms by least squares, until the atom budget is used, the best
    /// correlation falls below 1e-6, or the residual norm falls below 1e-6 times the signal norm.
    /// </summary>
    private Tensor<T> OMPSparseCoding(Tensor<T> x, Tensor<T> components)
    {
        var engine = Engine;
        int k = components.Shape[0];
        int p = components.Shape[1];
        T tolerance = DimensionalityReductionOps.Tolerance<T>(1e-6);
        T negativeInfinity = NumOps.FromDouble(double.NegativeInfinity);

        // Maximum number of atoms per sample (controls sparsity via alpha)
        int maxAtoms = Math.Max(1, (int)Math.Ceiling(k / (1 + _alpha)));
        maxAtoms = Math.Min(maxAtoms, Math.Min(k, p));

        var selectedAtoms = new List<int>();
        var selectedMask = new Tensor<T>(new[] { k });
        var code = new Tensor<T>(new[] { 1, k });
        var residual = x;
        var dtx = DimensionalityReductionOps.MatMulTranspose(x, components);                      // [1, k]
        T signalNorm = NumOps.Sqrt(engine.TensorSum(engine.TensorSquare(x)));
        T residualTolerance = NumOps.Multiply(tolerance, signalNorm);

        for (int iter = 0; iter < maxAtoms; iter++)
        {
            // Find the unselected atom with maximum |correlation| with the residual
            var correlation = engine.TensorAbs(engine.Reshape(
                DimensionalityReductionOps.MatMulTranspose(residual, components), new[] { k }));
            var open = engine.TensorWhere(selectedMask, engine.TensorFullLike(correlation, negativeInfinity), correlation);
            int bestAtom = DimensionalityReductionOps.FirstArgMax(open);

            if (DimensionalityReductionOps.Below(engine.TensorMaxValue(open), tolerance))
            {
                break;
            }

            selectedAtoms.Add(bestAtom);
            selectedMask = DimensionalityReductionOps.MarkIndex(selectedMask, bestAtom);

            // Least squares on the selected atoms: (D_S D_S^T + 1e-10 I) c = D_S x
            var selected = DimensionalityReductionOps.IndexTensor(selectedAtoms.ToArray());
            var atoms = DimensionalityReductionOps.SelectRows(components, selected);
            var gram = engine.TensorAdd(
                DimensionalityReductionOps.MatMulTranspose(atoms, atoms),
                engine.TensorMultiplyScalar(DimensionalityReductionOps.Identity<T>(selectedAtoms.Count), DimensionalityReductionOps.Constant<T>(1e-10)));
            var coefficients = DimensionalityReductionOps.Solve(
                gram, engine.Reshape(DimensionalityReductionOps.SelectColumns(dtx, selected), new[] { selectedAtoms.Count }));

            // Update residual and the code
            var coefficientRow = engine.Reshape(coefficients, new[] { 1, selectedAtoms.Count });
            residual = engine.TensorSubtract(x, engine.TensorMatMul(coefficientRow, atoms));
            code = engine.TensorScatter(new Tensor<T>(new[] { 1, k }), engine.Reshape(selected, new[] { 1, selectedAtoms.Count }), coefficientRow, 1);

            // Check residual convergence
            if (DimensionalityReductionOps.Below(DimensionalityReductionOps.Norm(residual), residualTolerance))
            {
                break;
            }
        }

        return code;
    }

    /// <summary>
    /// LASSO codes by cyclic coordinate descent, every sample at once: up to 100 sweeps of
    /// code[:, c] = S((D x)[c] - sum_{j != c} (D D^T)[c, j] code[:, j]) / (D D^T)[c, c], alpha / (D D^T)[c, c]),
    /// each sample stopping on its own once its largest coordinate change in a sweep is below 1e-6.
    /// </summary>
    private Tensor<T> SparseCodingLasso(Tensor<T> data, Tensor<T> components)
    {
        var engine = Engine;
        int n = data.Shape[0];
        int k = components.Shape[0];

        // Precompute D D^T and X D^T
        var dtd = DimensionalityReductionOps.MatMulTranspose(components, components);   // [k, k]
        var dtx = DimensionalityReductionOps.MatMulTranspose(data, components);         // [n, k]
        var alpha = DimensionalityReductionOps.Scalar(NumOps.FromDouble(_alpha));
        T floor = DimensionalityReductionOps.Tolerance<T>(1e-10);
        T stop = DimensionalityReductionOps.Tolerance<T>(1e-6);

        var codes = new Tensor<T>(new[] { n, k });
        var active = DimensionalityReductionOps.Full(new[] { n, 1 }, NumOps.One);

        for (int iter = 0; iter < 100; iter++)
        {
            var maxChange = new Tensor<T>(new[] { n, 1 });

            for (int c = 0; c < k; c++)
            {
                var oldCode = engine.TensorNarrow(codes, 1, c, 1);                       // [n, 1]
                var dtdColumn = engine.TensorNarrow(dtd, 1, c, 1);                       // [k, 1]
                var dtdDiagonal = engine.TensorNarrow(dtdColumn, 0, c, 1);               // [1, 1]

                // residual = (X D^T)[:, c] - codes (D D^T)[:, c] + code[:, c] (D D^T)[c, c]
                var residual = engine.TensorAdd(
                    engine.TensorSubtract(engine.TensorNarrow(dtx, 1, c, 1), engine.TensorMatMul(codes, dtdColumn)),
                    engine.TensorMultiply(oldCode, dtdDiagonal));

                var denominator = engine.TensorClampMin(dtdDiagonal, floor);
                var newCode = DimensionalityReductionOps.SoftThreshold(
                    engine.TensorDivide(residual, denominator), engine.TensorDivide(alpha, denominator));
                newCode = engine.TensorWhere(active, newCode, oldCode);

                maxChange = engine.TensorMax(maxChange, engine.TensorAbs(engine.TensorSubtract(newCode, oldCode)));
                codes = DimensionalityReductionOps.WithColumn(codes, c, newCode);
            }

            // Samples whose sweep changed every code by less than 1e-6 are done
            active = engine.TensorMultiply(active, engine.ScalarMinusTensor(NumOps.One, engine.TensorLessThan(maxChange, stop)));
            if (DimensionalityReductionOps.Below(engine.TensorMaxValue(active), NumOps.One))
            {
                break;
            }
        }

        return codes;
    }

    /// <summary>
    /// Block coordinate descent over the atoms: each atom is refit to its residual by least squares and
    /// renormalized; an atom whose codes are all ~0 is left as it was.
    /// </summary>
    private Tensor<T> UpdateDictionaryCD(Tensor<T> data, Tensor<T> codes, Tensor<T> components)
    {
        var engine = Engine;
        int k = components.Shape[0];
        T floor = DimensionalityReductionOps.Tolerance<T>(1e-10);

        for (int c = 0; c < k; c++)
        {
            var codeColumn = engine.TensorNarrow(codes, 1, c, 1);                // [n, 1]
            var row = engine.TensorNarrow(components, 0, c, 1);                  // [1, p]

            // Compute residual without this atom
            var residual = engine.TensorAdd(
                engine.TensorSubtract(data, engine.TensorMatMul(codes, components)),
                engine.TensorMatMul(codeColumn, row));

            // Least squares update of this atom
            var codesSquared = engine.ReduceSum(engine.TensorSquare(codeColumn), new[] { 0 }, true);   // [1, 1]
            var updated = DimensionalityReductionOps.NormalizeRows(engine.TensorDivide(
                DimensionalityReductionOps.TransposeMatMul(codeColumn, residual), codesSquared));

            // Skip unused atoms. TensorWhere does not broadcast its condition, so the [1, 1] flag is expanded to the
            // [1, p] row first.
            var keep = engine.TensorBroadcastTo(engine.TensorLessThan(codesSquared, floor), updated.Shape.ToArray());
            components = DimensionalityReductionOps.WithRow(components, c, engine.TensorWhere(keep, row, updated));
        }

        return components;
    }

    /// <summary>Least squares dictionary D = (A^T A + 1e-6 I)^-1 A^T X with normalized atoms.</summary>
    private Tensor<T> UpdateDictionaryLeastSquares(Tensor<T> data, Tensor<T> codes)
    {
        var engine = Engine;
        int k = codes.Shape[1];

        var ata = engine.TensorAdd(
            DimensionalityReductionOps.TransposeMatMul(codes, codes),
            engine.TensorMultiplyScalar(DimensionalityReductionOps.Identity<T>(k), DimensionalityReductionOps.Tolerance<T>(1e-6)));
        var atx = DimensionalityReductionOps.TransposeMatMul(codes, data);
        return DimensionalityReductionOps.NormalizeRows(DimensionalityReductionOps.Solve(ata, atx));
    }

    /// <summary>
    /// Transforms data by computing sparse codes.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_mean is null || _components is null)
        {
            throw new InvalidOperationException("DictionaryLearning has not been fitted.");
        }

        var centered = DimensionalityReductionOps.SubtractRow(Tensor<T>.FromMatrix(data), Tensor<T>.FromVector(_mean));
        return UpdateCodes(centered, Tensor<T>.FromMatrix(_components)).ToMatrix();
    }

    /// <summary>
    /// Transforms sparse codes back to original space.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_mean is null || _components is null)
        {
            throw new InvalidOperationException("DictionaryLearning has not been fitted.");
        }

        int k = _components.Rows;
        var codes = DimensionalityReductionOps.FirstColumns(Tensor<T>.FromMatrix(data), k);
        var reconstructed = Engine.TensorMatMul(codes, Tensor<T>.FromMatrix(_components));
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
            names[i] = $"Atom{i + 1}";
        }
        return names;
    }
}

/// <summary>
/// Specifies the algorithm for fitting the dictionary.
/// </summary>
public enum DictionaryFitAlgorithm
{
    /// <summary>
    /// Coordinate descent - updates atoms one at a time.
    /// </summary>
    CD,

    /// <summary>
    /// Least squares - solves for all atoms simultaneously.
    /// </summary>
    LeastSquares
}

/// <summary>
/// Specifies the algorithm for sparse coding (transform).
/// </summary>
public enum SparseCodingAlgorithm
{
    /// <summary>
    /// LASSO (L1 regularization) using coordinate descent.
    /// </summary>
    LASSO,

    /// <summary>
    /// Orthogonal Matching Pursuit - greedy selection.
    /// </summary>
    OMP
}
