using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Bioinformatics;

/// <summary>
/// Significance Analysis of Microarrays (SAM) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// SAM uses a modified t-statistic with a fudge factor to identify significantly
/// differentially expressed genes. It controls the false discovery rate (FDR)
/// using permutation-based estimation.
/// </para>
/// <para><b>For Beginners:</b> SAM was designed for gene expression data where we
/// want to find genes that behave differently between conditions (e.g., healthy vs
/// disease). It uses permutation testing to estimate how many "significant" genes
/// we'd find just by chance, helping control false positives.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SAM<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nPermutations;
    private readonly double _fdrThreshold;
    private readonly double _s0Percentile;
    private readonly int? _randomState;

    private Vector<T>? _dStatistics;
    private Vector<T>? _fdr;
    private Vector<T>? _qValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? DStatistics => _dStatistics;
    public Vector<T>? FDR => _fdr;
    public Vector<T>? QValues => _qValues;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public SAM(
        int nFeaturesToSelect = 100,
        int nPermutations = 100,
        double fdrThreshold = 0.05,
        double s0Percentile = 0.5,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nPermutations < 10)
            throw new ArgumentException("Number of permutations must be at least 10.", nameof(nPermutations));
        if (fdrThreshold <= 0 || fdrThreshold >= 1)
            throw new ArgumentException("FDR threshold must be between 0 and 1.", nameof(fdrThreshold));
        if (s0Percentile < 0 || s0Percentile > 1)
            throw new ArgumentOutOfRangeException(nameof(s0Percentile), "s0Percentile must be in range [0, 1].");

        _nFeaturesToSelect = nFeaturesToSelect;
        _nPermutations = nPermutations;
        _fdrThreshold = fdrThreshold;
        _s0Percentile = s0Percentile;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SAM requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var engine = Engine;
        var X = Tensor<T>.FromMatrix(data);
        var y = Tensor<T>.FromVector(target);

        // Identify two-class labels; the first label is the one that appears first
        if (engine.TensorUnique(y).Length != 2)
            throw new ArgumentException("SAM requires exactly two distinct class labels.");

        var class0 = engine.TensorEquals(y, y.GetFlat(0));
        var class1 = PreprocessingTensorOps.Not(class0);
        int n0 = PreprocessingTensorOps.CountNonZero(class0);
        int n1 = n - n0;

        if (n0 < 2 || n1 < 2)
            throw new ArgumentException("SAM requires at least 2 samples in each class.");

        T count0 = NumOps.FromDouble(n0);
        T count1 = NumOps.FromDouble(n1);
        var class0Row = engine.Reshape(class0, new[] { 1, n });
        var class1Row = engine.Reshape(class1, new[] { 1, n });
        var mean0 = engine.TensorDivideScalar(engine.TensorMatMul(class0Row, X), count0); // [1, p]
        var mean1 = engine.TensorDivideScalar(engine.TensorMatMul(class1Row, X), count1);

        // Standard errors from the pooled within-class variance
        var ss0 = engine.TensorMatMul(class0Row, engine.TensorSquare(engine.TensorSubtract(X, mean0)));
        var ss1 = engine.TensorMatMul(class1Row, engine.TensorSquare(engine.TensorSubtract(X, mean1)));
        var pooledVar = engine.TensorDivideScalar(engine.TensorAdd(ss0, ss1), NumOps.FromDouble(n0 + n1 - 2));
        var standardErrors = engine.Reshape(
            engine.TensorSqrt(engine.TensorMultiplyScalar(pooledVar, NumOps.FromDouble(1.0 / n0 + 1.0 / n1))), new[] { p });

        // s0 (fudge factor) is the percentile of standard errors
        var (sortedSE, _) = engine.TensorSort(standardErrors, 0, false);
        T s0 = sortedSE.GetFlat((int)(_s0Percentile * (p - 1)));
        if (!NumOps.GreaterThan(s0, NumOps.Zero))
            s0 = PreprocessingTensorOps.Tolerance<T>(1e-12);
        var denominator = engine.TensorAddScalar(standardErrors, s0);

        // d-statistics
        var dStatistics = engine.TensorDivide(engine.Reshape(engine.TensorSubtract(mean1, mean0), new[] { p }), denominator);
        _dStatistics = dStatistics.ToVector();

        // Permutation test: the first n0 shuffled rows form class 0
        var columnTotals = engine.ReduceSum(X, new[] { 0 }, true);
        var ones = PreprocessingTensorOps.Filled(new[] { n }, NumOps.One);
        var zeros = new Tensor<T>(new[] { n });
        var indices = Enumerable.Range(0, n).ToArray();
        var inClass0 = new bool[n];
        var permutedD = new Tensor<T>[_nPermutations];

        for (int perm = 0; perm < _nPermutations; perm++)
        {
            // Shuffle indices
            for (int i = n - 1; i > 0; i--)
            {
                int j = random.Next(i + 1);
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }

            Array.Clear(inClass0, 0, n);
            for (int i = 0; i < n0; i++)
                inClass0[indices[i]] = true;

            var sum0 = engine.TensorMatMul(engine.Reshape(engine.TensorWhere(inClass0, ones, zeros), new[] { 1, n }), X);
            var permMean0 = engine.TensorDivideScalar(sum0, count0);
            var permMean1 = engine.TensorDivideScalar(engine.TensorSubtract(columnTotals, sum0), count1);
            permutedD[perm] = engine.TensorDivide(
                engine.Reshape(engine.TensorSubtract(permMean1, permMean0), new[] { p }), denominator);
        }

        // q-values (FDR-adjusted p-values): expected false positives (permuted |d| at or above the observed |d|,
        // per permutation) over the observed d-values at or above it
        // A NaN |d| never compares >= anything (as in master), so NaNs become -1, below every |d| >= 0. The counts
        // are int insertion indices and the ratio is formed from them in double: in T, nPermutations * p (and the
        // counts themselves) overflow Half (max 65504) and stop being exact past 2048.
        var absD = engine.TensorAbs(dStatistics);
        var minusOne = PreprocessingTensorOps.Filled(new[] { p }, NumOps.FromDouble(-1.0));
        var observedIsNumber = engine.TensorEquals(absD, absD);
        var observed = engine.TensorWhere(observedIsNumber, absD, minusOne);
        var (sortedObserved, _) = engine.TensorSort(observed, 0, false);
        var observedBelow = engine.TensorSearchSorted(sortedObserved, observed, false).AsSpan().ToArray();

        // nPermutations is at least 10 (validated in the constructor)
        int permutedCount = _nPermutations * p;
        var permutedAbs = engine.Reshape(engine.TensorAbs(engine.TensorStack(permutedD, 0)), new[] { permutedCount });
        var permuted = engine.TensorWhere(engine.TensorEquals(permutedAbs, permutedAbs), permutedAbs,
            PreprocessingTensorOps.Filled(new[] { permutedCount }, NumOps.FromDouble(-1.0)));
        var (sortedPermuted, _) = engine.TensorSort(permuted, 0, false);
        var permutedBelow = engine.TensorSearchSorted(sortedPermuted, observed, false).AsSpan().ToArray();
        var isNumber = observedIsNumber.Cast<int>().AsSpan().ToArray();

        // FDR = (false positives / nPermutations) / true positives, capped at 1; 1 when there is no true positive
        var fdr = new Tensor<T>(Enumerable.Range(0, p).Select(j =>
        {
            int truePositives = isNumber[j] != 0 ? p - observedBelow[j] : 0;
            double expectedFalsePositives = (double)(permutedCount - permutedBelow[j]) / _nPermutations;
            return NumOps.FromDouble(truePositives > 0 ? Math.Min(expectedFalsePositives / truePositives, 1.0) : 1.0);
        }).ToArray(), new[] { p });

        _fdr = fdr.ToVector();
        _qValues = fdr.ToVector();

        // Select significant features
        var significant = PreprocessingTensorOps.Not(engine.TensorGreaterThan(fdr, NumOps.FromDouble(_fdrThreshold)));
        if (PreprocessingTensorOps.CountNonZero(significant) >= _nFeaturesToSelect)
        {
            var ranked = engine.TensorWhere(significant, absD, PreprocessingTensorOps.Filled(new[] { p }, NumOps.FromDouble(-1.0)));
            _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(ranked, _nFeaturesToSelect);
        }
        else
        {
            // Fall back to top by |d|
            _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(absD, _nFeaturesToSelect);
        }

        IsFitted = true;
    }

    /// <inheritdoc />
    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SAM has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SAM does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SAM has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return Array.Empty<string>();

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
