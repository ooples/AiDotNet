using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Bioinformatics;

/// <summary>
/// Significance Analysis of Microarrays (SAM) for bioinformatics feature selection.
/// </summary>
/// <remarks>
/// <para>
/// SAM uses a modified t-statistic that includes a small constant (fudge factor) in
/// the denominator to avoid false positives from genes with very low variance. It
/// estimates false discovery rate (FDR) using permutation testing.
/// </para>
/// <para><b>For Beginners:</b> When analyzing gene expression data, some genes have
/// very small changes but even smaller variance, making them appear significant by
/// chance. SAM adds a small buffer to prevent these false alarms. It also uses
/// shuffling to estimate how many of our "discoveries" might be false.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SignificanceAnalysis<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nPermutations;
    private readonly double _fdrThreshold;
    private readonly int? _randomState;

    private Vector<T>? _samScores;
    private Vector<T>? _qValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NPermutations => _nPermutations;
    public double FDRThreshold => _fdrThreshold;
    public Vector<T>? SAMScores => _samScores;
    public Vector<T>? QValues => _qValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public SignificanceAnalysis(
        int nFeaturesToSelect = 10,
        int nPermutations = 100,
        double fdrThreshold = 0.05,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nPermutations < 1)
            throw new ArgumentException("Number of permutations must be at least 1.", nameof(nPermutations));
        if (double.IsNaN(fdrThreshold) || fdrThreshold < 0.0 || fdrThreshold > 1.0)
            throw new ArgumentException("FDR threshold must be between 0 and 1.", nameof(fdrThreshold));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nPermutations = nPermutations;
        _fdrThreshold = fdrThreshold;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SignificanceAnalysis requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

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

        // Separate samples by class (binary labels required); class 0 is the smaller label
        if (engine.TensorUnique(y).Length != 2)
            throw new ArgumentException("SignificanceAnalysis requires exactly two distinct target labels.", nameof(target));

        var class0 = engine.TensorEquals(y, engine.TensorMinValue(y));
        var class1 = PreprocessingTensorOps.Not(class0);
        int n0 = PreprocessingTensorOps.CountNonZero(class0);
        int n1 = n - n0;

        if (n0 < 2 || n1 < 2)
        {
            // Not enough samples in each class
            _samScores = new Vector<T>(p);
            _qValues = new Vector<T>(p);
            _selectedIndices = Enumerable.Range(0, Math.Min(_nFeaturesToSelect, p)).ToArray();
            IsFitted = true;
            return;
        }

        T count0 = NumOps.FromDouble(n0);
        T count1 = NumOps.FromDouble(n1);
        var class0Row = engine.Reshape(class0, new[] { 1, n });
        var class1Row = engine.Reshape(class1, new[] { 1, n });
        var mean0 = engine.TensorDivideScalar(engine.TensorMatMul(class0Row, X), count0); // [1, p]
        var mean1 = engine.TensorDivideScalar(engine.TensorMatMul(class1Row, X), count1);

        // Pooled standard deviations; s0 (fudge factor) is their median
        var ss0 = engine.TensorMatMul(class0Row, engine.TensorSquare(engine.TensorSubtract(X, mean0)));
        var ss1 = engine.TensorMatMul(class1Row, engine.TensorSquare(engine.TensorSubtract(X, mean1)));
        var pooledVar = engine.TensorDivideScalar(engine.TensorAdd(ss0, ss1), NumOps.FromDouble(n0 + n1 - 2));
        var pooledStds = engine.Reshape(
            engine.TensorSqrt(engine.TensorMultiplyScalar(pooledVar, NumOps.FromDouble(1.0 / n0 + 1.0 / n1))), new[] { p });

        var (sortedStds, _) = engine.TensorSort(pooledStds, 0, false);
        T s0 = sortedStds.GetFlat(p / 2);
        if (NumOps.IsNaN(s0) || !NumOps.GreaterThan(s0, NumOps.Zero))
            s0 = NumOps.FromDouble(1e-12);
        var denominator = engine.TensorAddScalar(pooledStds, s0);

        // d_i (SAM statistic)
        var scores = engine.TensorDivide(engine.Reshape(engine.TensorSubtract(mean1, mean0), new[] { p }), denominator);
        _samScores = scores.ToVector();
        var absScores = engine.TensorAbs(scores);

        // Permutation testing for FDR estimation: the first n0 shuffled rows form class 0
        var columnTotals = engine.ReduceSum(X, new[] { 0 }, true);
        var ones = PreprocessingTensorOps.Filled(new[] { n }, NumOps.One);
        var zeros = new Tensor<T>(new[] { n });
        var allIndices = Enumerable.Range(0, n).ToArray();
        var inClass0 = new bool[n];
        var exceedCounts = new Tensor<T>(new[] { p });

        for (int perm = 0; perm < _nPermutations; perm++)
        {
            // Shuffle labels
            var shuffled = allIndices.OrderBy(_ => random.Next()).ToArray();
            Array.Clear(inClass0, 0, n);
            for (int i = 0; i < n0; i++)
                inClass0[shuffled[i]] = true;

            var sum0 = engine.TensorMatMul(engine.Reshape(engine.TensorWhere(inClass0, ones, zeros), new[] { 1, n }), X);
            var permMean0 = engine.TensorDivideScalar(sum0, count0);
            var permMean1 = engine.TensorDivideScalar(engine.TensorSubtract(columnTotals, sum0), count1);
            var permScores = engine.TensorAbs(engine.TensorDivide(
                engine.Reshape(engine.TensorSubtract(permMean1, permMean0), new[] { p }), denominator));

            // Permuted scores at or above the observed score
            exceedCounts = engine.TensorAdd(exceedCounts, PreprocessingTensorOps.Not(engine.TensorLessThan(permScores, absScores)));
        }

        // Estimate q-values (FDR)
        var qValues = engine.TensorDivideScalar(exceedCounts, NumOps.FromDouble(_nPermutations));
        _qValues = qValues.ToVector();

        // Select features passing the FDR threshold, by |score|; none passing falls back to the top scores
        var candidates = PreprocessingTensorOps.Not(engine.TensorGreaterThan(qValues, NumOps.FromDouble(_fdrThreshold)));
        int numToSelect = Math.Min(_nFeaturesToSelect, PreprocessingTensorOps.CountNonZero(candidates));
        if (numToSelect == 0)
        {
            _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(absScores, Math.Min(_nFeaturesToSelect, p));
        }
        else
        {
            var ranked = engine.TensorWhere(candidates, absScores, PreprocessingTensorOps.Filled(new[] { p }, NumOps.FromDouble(-1.0)));
            _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(ranked, numToSelect);
        }

        IsFitted = true;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SignificanceAnalysis has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SignificanceAnalysis does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SignificanceAnalysis has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return [];

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
