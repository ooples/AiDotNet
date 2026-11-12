using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.GameTheory;

/// <summary>
/// Shapley Value-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses Shapley values from cooperative game theory to fairly attribute
/// contribution to each feature based on all possible feature coalitions.
/// </para>
/// <para><b>For Beginners:</b> Shapley values come from game theory and measure
/// how much each "player" (feature) contributes to the team's success (model
/// accuracy). They consider all possible combinations of features, giving a
/// fair and complete picture of each feature's true importance.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class ShapleyFeatureSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nSamples;
    private readonly int? _randomState;

    private Vector<T>? _shapleyValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NSamples => _nSamples;
    public Vector<T>? ShapleyValues => _shapleyValues;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public ShapleyFeatureSelector(
        int nFeaturesToSelect = 10,
        int nSamples = 100,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nSamples = nSamples;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ShapleyFeatureSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;

        var rand = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var model = new SubsetRegression(FilterStatisticTensorOps.AsTensor(data), FilterStatisticTensorOps.AsTensor(target));
        var totals = new Tensor<T>(new[] { p });
        T zero = NumOps.Zero;

        // Monte Carlo approximation of Shapley values
        var keys = new int[p];
        var prefixScores = new T[p + 1];
        prefixScores[0] = zero;
        for (int sample = 0; sample < _nSamples; sample++)
        {
            // Random permutation of features (one random key per feature, ties by index)
            for (int j = 0; j < p; j++)
                keys[j] = rand.Next();
            var permutation = Enumerable.Range(0, p).OrderBy(j => keys[j]).ToArray();

            // R^2 of every prefix of the permutation; feature permutation[i] adds score(i + 1) - score(i)
            for (int i = 0; i < p; i++)
                prefixScores[i + 1] = model.RSquared(permutation.Take(i + 1).ToArray());
            var scores = new Tensor<T>((T[])prefixScores.Clone(), new[] { p + 1 });
            var marginal = Engine.TensorSubtract(
                Engine.TensorSlice(scores, new[] { 1 }, new[] { p }), Engine.TensorSlice(scores, new[] { 0 }, new[] { p }));
            totals = Engine.TensorAdd(totals, FeatureSelectionTensorOps.ScatterToFeatures(marginal, permutation, p, zero));
        }

        // Average over samples
        var shapley = Engine.TensorDivideScalar(totals, NumOps.FromDouble(_nSamples));
        _shapleyValues = shapley.ToVector();
        _selectedIndices = FilterStatisticTensorOps.TopKIndicesSorted(shapley, Math.Min(_nFeaturesToSelect, p));
        IsFitted = true;
    }

    // Linear regression R^2 of the centred target on a feature subset: beta solves (X_S' X_S + 1e-6 I) beta =
    // X_S' (y - mean y) on the raw features, predictions are mean y + X_S beta, and R^2 = max(0, 1 - SSres / SStot)
    // (0 for an empty subset or a constant target).
    private sealed class SubsetRegression
    {
        private readonly Tensor<T> _x;
        private readonly Tensor<T> _centered;
        private readonly Tensor<T> _gram;
        private readonly Tensor<T> _cross;
        private readonly T _totalSquares;

        public SubsetRegression(Tensor<T> x, Tensor<T> y)
        {
            var engine = AiDotNetEngine.Current;
            int n = x.Shape[0];
            _x = x;
            _centered = engine.TensorSubtractScalar(y, engine.TensorMean(y));
            _gram = engine.TensorMatMul(engine.TensorTranspose(x), x);
            _cross = engine.Reshape(
                engine.TensorMatMul(engine.TensorTranspose(x), engine.Reshape(_centered, new[] { n, 1 })), new[] { x.Shape[1] });
            _totalSquares = engine.TensorSumOfSquares(_centered);
        }

        public T RSquared(int[] subset)
        {
            var ops = MathHelper.GetNumericOperations<T>();
            var engine = AiDotNetEngine.Current;
            int k = subset.Length;
            if (k == 0 || !ops.GreaterThan(_totalSquares, ops.FromDouble(1e-10)))
                return ops.Zero;

            var index = new Tensor<int>((int[])subset.Clone(), new[] { k });
            var gram = engine.TensorIndexSelect(engine.TensorIndexSelect(_gram, index, 0), index, 1);
            var system = engine.TensorAdd(gram, engine.TensorMultiplyScalar(engine.TensorEye<T>(k), ops.FromDouble(1e-6)));
            var beta = LinalgDispatch<T>.Solve(system, engine.Reshape(engine.TensorIndexSelect(_cross, index, 0), new[] { k, 1 }));
            var columns = engine.TensorIndexSelect(_x, index, 1);
            var fitted = engine.Reshape(engine.TensorMatMul(columns, beta), new[] { _x.Shape[0] });
            T residual = engine.TensorSumOfSquares(engine.TensorSubtract(_centered, fitted));
            T rSquared = ops.Subtract(ops.One, ops.Divide(residual, _totalSquares));
            return ops.GreaterThan(rSquared, ops.Zero) ? rSquared : ops.Zero;
        }
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
            throw new InvalidOperationException("ShapleyFeatureSelector has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("ShapleyFeatureSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ShapleyFeatureSelector has not been fitted.");

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
