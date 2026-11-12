using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Regression;

/// <summary>
/// Least Angle Regression (LARS) based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses LARS algorithm to select features by progressively adding features
/// along the direction equiangular to all active features.
/// </para>
/// <para><b>For Beginners:</b> LARS is like a compromise between forward selection
/// and Lasso. It adds features gradually, moving in a direction that's equally
/// correlated with all currently active features. This gives a natural ordering
/// of feature importance.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class LARSSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;

    private Vector<T>? _featureRanks;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? FeatureRanks => _featureRanks;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public LARSSelector(
        int nFeaturesToSelect = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "LARSSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;

        // Standardize X (population std; near-constant columns keep scale 1) and centre y
        var x = FilterStatisticTensorOps.CenterColumns(Tensor<T>.FromMatrix(data));
        var stds = engine.TensorSqrt(engine.TensorDivideScalar(
            SelectorTensorOps.ColumnSumOfSquares(x), NumOps.FromDouble(n)));
        stds = engine.TensorWhere(engine.TensorLessThan(stds, NumOps.FromDouble(1e-10)),
            PreprocessingTensorOps.Filled(new[] { p }, NumOps.One), stds);
        x = engine.TensorDivide(x, engine.Reshape(stds, new[] { 1, p }));
        var xt = engine.TensorTranspose(x);
        var yt = Tensor<T>.FromVector(target);
        var y = engine.Reshape(engine.TensorSubtractScalar(yt, engine.TensorMean(yt)), new[] { n, 1 });

        // Least Angle Regression (Efron, Hastie, Johnstone and Tibshirani, 2004): the fit moves along the
        // direction equiangular to the active features until an inactive feature's correlation with the
        // residual catches up; that feature enters next.
        var fitted = PreprocessingTensorOps.Filled(new[] { n, 1 }, NumOps.Zero);
        var featureRanks = PreprocessingTensorOps.Filled(new[] { p }, NumOps.Zero);
        var activeMask = PreprocessingTensorOps.Filled(new[] { p }, NumOps.Zero);
        var infinity = PreprocessingTensorOps.Filled(new[] { p }, NumOps.FromDouble(double.PositiveInfinity));
        var ranks = engine.TensorAddScalar(engine.TensorNegate(engine.TensorArange<T>(0, p, 1)), NumOps.FromDouble(p));
        T one = NumOps.One;
        T tiny = NumOps.FromDouble(1e-12);
        T positiveInfinity = NumOps.FromDouble(double.PositiveInfinity);
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        var selectionOrder = new List<int>();

        // First entrant: the feature most correlated with y
        var correlations = engine.Reshape(engine.TensorMatMul(xt, y), new[] { p });
        if (p > 0 && numToSelect > 0 && PreprocessingTensorOps.IsAbove(engine.TensorMaxValue(engine.TensorAbs(correlations)), NumOps.Zero))
        {
            int first = PreprocessingTensorOps.FirstArgMax(engine.TensorAbs(correlations));
            selectionOrder.Add(first);
        }

        while (selectionOrder.Count > 0 && selectionOrder.Count < numToSelect)
        {
            int[] active = selectionOrder.ToArray();
            activeMask = engine.TensorIndexCopy(activeMask, 0, new Tensor<int>(active, new[] { active.Length }),
                PreprocessingTensorOps.Filled(new[] { active.Length }, one));

            // Current correlations c = X'(y - mu) and their common active magnitude C
            correlations = engine.Reshape(
                engine.TensorMatMul(xt, engine.TensorSubtract(y, fitted)), new[] { p });
            var activeIndex = new Tensor<int>(active, new[] { active.Length });
            var signs = engine.TensorSign(engine.TensorIndexSelect(correlations, activeIndex, 0));
            var bigC = engine.ReduceMax(engine.TensorAbs(engine.TensorIndexSelect(correlations, activeIndex, 0)), new[] { 0 }, true);

            // Equiangular direction u = X_A w with w = A_A G^-1 1, A_A = (1' G^-1 1)^(-1/2)
            var xa = engine.TensorMultiply(engine.TensorIndexSelect(x, activeIndex, 1),
                engine.Reshape(signs, new[] { 1, active.Length }));
            var gram = engine.TensorMatMul(engine.TensorTranspose(xa), xa);
            var ginvOnes = LinalgDispatch<T>.Solve(gram, PreprocessingTensorOps.Filled(new[] { active.Length, 1 }, one));
            var aa = engine.TensorDivide(PreprocessingTensorOps.Filled(new[] { 1 }, one),
                engine.TensorSqrt(engine.Reshape(engine.ReduceSum(ginvOnes, new[] { 0, 1 }, false), new[] { 1 })));
            var u = engine.TensorMultiply(engine.TensorMatMul(xa, ginvOnes), aa);
            var a = engine.Reshape(engine.TensorMatMul(xt, u), new[] { p });

            // Step to the first inactive feature whose |correlation| catches up: min+ over
            // (C - c_j) / (A_A - a_j) and (C + c_j) / (A_A + a_j)
            var gammaMinus = engine.TensorDivide(engine.TensorSubtract(bigC, correlations), engine.TensorSubtract(aa, a));
            var gammaPlus = engine.TensorDivide(engine.TensorAdd(bigC, correlations), engine.TensorAdd(aa, a));
            gammaMinus = engine.TensorWhere(engine.TensorGreaterThan(gammaMinus, tiny), gammaMinus, infinity);
            gammaPlus = engine.TensorWhere(engine.TensorGreaterThan(gammaPlus, tiny), gammaPlus, infinity);
            var gamma = engine.TensorWhere(activeMask, infinity, engine.TensorMin(gammaMinus, gammaPlus));
            var smallest = engine.TensorMinValue(gamma);
            if (!PreprocessingTensorOps.IsBelow(smallest, positiveInfinity))
                break;

            int next = PreprocessingTensorOps.FirstArgMax(engine.TensorNegate(gamma));
            fitted = engine.TensorAdd(fitted, engine.TensorMultiplyScalar(u, smallest));
            selectionOrder.Add(next);
        }

        // Rank p for the first entrant, p - 1 for the second, ...
        if (selectionOrder.Count > 0)
        {
            var order = selectionOrder.ToArray();
            featureRanks = engine.TensorIndexCopy(featureRanks, 0, new Tensor<int>(order, new[] { order.Length }),
                engine.TensorSlice(ranks, new[] { 0 }, new[] { order.Length }));
        }

        _featureRanks = featureRanks.ToVector();
        _selectedIndices = selectionOrder.OrderBy(index => index).ToArray();
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
            throw new InvalidOperationException("LARSSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("LARSSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LARSSelector has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
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
