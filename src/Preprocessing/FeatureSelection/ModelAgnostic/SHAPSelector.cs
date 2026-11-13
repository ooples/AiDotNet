using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.ModelAgnostic;

/// <summary>
/// SHAP (SHapley Additive exPlanations) based feature selection.
/// </summary>
/// <remarks>
/// <para>
/// SHAP uses Shapley values from game theory to explain feature importance.
/// Each feature's importance is its average contribution to predictions across
/// all possible feature combinations.
/// </para>
/// <para>
/// The algorithm (Kernel SHAP approximation):
/// 1. Sample feature coalitions (subsets of features)
/// 2. Weight coalitions by SHAP kernel
/// 3. Predict with masked features (baseline substitution)
/// 4. Solve weighted least squares for Shapley values
/// </para>
/// <para><b>For Beginners:</b> SHAP answers: "How much did each feature contribute
/// to this prediction?"
///
/// Imagine a team of features making a prediction. SHAP figures out each
/// player's contribution fairly - not just by removing them, but by considering
/// every possible team combination they could have been in.
///
/// Features with high absolute SHAP values (positive or negative) are important.
/// SHAP provides both global importance and local explanations per sample.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SHAPSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nSamples;
    private readonly int _nCoalitions;
    private readonly int? _randomState;
    private readonly Func<Matrix<T>, Vector<T>>? _predictFunc;

    // Fitted parameters
    private Vector<T>? _shapValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <summary>
    /// Gets the number of features to select.
    /// </summary>
    public int NFeaturesToSelect => _nFeaturesToSelect;

    /// <summary>
    /// Gets the computed SHAP values (mean absolute).
    /// </summary>
    public Vector<T>? ShapValues => _shapValues;

    /// <summary>
    /// Gets the indices of selected features.
    /// </summary>
    public int[]? SelectedIndices => _selectedIndices;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="SHAPSelector{T}"/>.
    /// </summary>
    /// <param name="predictFunc">Function that makes predictions from data.</param>
    /// <param name="nFeaturesToSelect">Number of features to select. Defaults to 10.</param>
    /// <param name="nSamples">Number of background samples for baseline. Defaults to 100.</param>
    /// <param name="nCoalitions">Number of feature coalitions to sample. Defaults to 200.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <param name="columnIndices">The column indices to evaluate, or null for all columns.</param>
    public SHAPSelector(
        Func<Matrix<T>, Vector<T>>? predictFunc = null,
        int nFeaturesToSelect = 10,
        int nSamples = 100,
        int nCoalitions = 200,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
        {
            throw new ArgumentException("Number of features to select must be at least 1.", nameof(nFeaturesToSelect));
        }

        if (nSamples < 1)
        {
            throw new ArgumentException("Number of samples must be at least 1.", nameof(nSamples));
        }

        if (nCoalitions < 1)
        {
            throw new ArgumentException("Number of coalitions must be at least 1.", nameof(nCoalitions));
        }

        _predictFunc = predictFunc;
        _nFeaturesToSelect = nFeaturesToSelect;
        _nSamples = nSamples;
        _nCoalitions = nCoalitions;
        _randomState = randomState;
    }

    /// <summary>
    /// Fits the selector (requires target via specialized Fit method).
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SHAPSelector requires target values for fitting. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <summary>
    /// Fits SHAP selector by computing Shapley values for each feature.
    /// </summary>
    /// <param name="data">The feature matrix.</param>
    /// <param name="target">The target values.</param>
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
        {
            throw new ArgumentException("Target length must match the number of rows in data.");
        }

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        var engine = Engine;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var x = Tensor<T>.FromMatrix(data);

        // Compute baseline values (mean of background samples)
        int nBackground = Math.Min(_nSamples, n);
        var backgroundIndices = Enumerable.Range(0, n).OrderBy(_ => random.Next()).Take(nBackground).ToArray();
        var baseline = engine.ReduceMean(
            engine.TensorIndexSelect(x, new Tensor<int>(backgroundIndices, new[] { nBackground }), 0), new[] { 0 }, false);

        // Sample instances to explain
        int nExplain = Math.Min(_nSamples, n);
        var explainIndices = Enumerable.Range(0, n).OrderBy(_ => random.Next()).Take(nExplain).ToArray();

        // Accumulate |SHAP values|
        var shapAccum = new Tensor<T>(new[] { p });
        foreach (int explainIdx in explainIndices)
        {
            // Coalitions: one uniform draw per feature, a feature joins when its draw exceeds 1/2
            var draws = PopulationSearchOps.Uniform<T>(random, _nCoalitions, p);
            var instanceShap = KernelShap(PopulationSearchOps.Row(x, explainIdx), baseline, draws, Predict);
            shapAccum = engine.TensorAdd(shapAccum, engine.TensorAbs(instanceShap));
        }

        // Average SHAP values
        var shapValues = engine.TensorDivideScalar(shapAccum, NumOps.FromDouble(nExplain));
        _shapValues = shapValues.ToVector();

        // Select top features by SHAP importance
        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(shapValues, nToSelect);

        IsFitted = true;
    }

    /// <summary>
    /// Kernel SHAP (Lundberg and Lee 2017) for one instance [p] against a baseline [p]. Coalition z (row of the
    /// draws [c, p] above 1/2; empty and full ones are skipped) is evaluated on the instance where z_j = 1 and the
    /// baseline elsewhere, and weighted by the Shapley kernel (p - 1) / (C(p, |z|) |z| (p - |z|)). The SHAP
    /// values minimize sum_z w(z) (f(z) - phi_0 - z . phi)^2 subject to phi_0 = f(empty) and
    /// sum phi = f(x) - f(empty), solved by eliminating the last value as in the reference implementation.
    /// </summary>
    internal static Tensor<T> KernelShap(
        Tensor<T> instance, Tensor<T> baseline, Tensor<T> draws, Func<Tensor<T>, Tensor<T>> predict)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int p = instance.Length;
        int c = draws.Shape[0];

        var ends = engine.TensorConcatenate(
            new[] { engine.Reshape(baseline, new[] { 1, p }), engine.Reshape(instance, new[] { 1, p }) }, 0);
        var endPredictions = predict(ends);
        var emptyPrediction = engine.TensorSlice(endPredictions, new[] { 0 }, new[] { 1 });
        var gap = engine.TensorSubtract(engine.TensorSlice(endPredictions, new[] { 1 }, new[] { 1 }), emptyPrediction);
        if (p == 1)
            return gap;

        // Keep the coalitions that are neither empty nor full
        var joined = engine.TensorGreaterThan(draws, ops.FromDouble(0.5));
        // coalition sizes counted in int (a 0/1 sum in T stops counting at 2048 for Half)
        var sizes = engine.ReduceSum(joined.Cast<int>(), new[] { 1 }, false).Cast<T>();
        var proper = PreprocessingTensorOps.MaskPositions(engine.TensorMultiply(
            engine.TensorGreaterThan(sizes, ops.Zero), engine.TensorLessThan(sizes, ops.FromDouble(p))));
        if (proper.Length == 0)
            return engine.TensorDivideScalar(engine.TensorBroadcastTo(gap, new[] { p }), ops.FromDouble(p));

        int m = proper.Length;
        var properIndex = new Tensor<int>(proper, new[] { m });
        var z = engine.TensorIndexSelect(joined, properIndex, 0);
        var size = engine.TensorIndexSelect(sizes, properIndex, 0);
        var shape = new[] { m, p };

        var masked = engine.TensorWhere(
            z,
            engine.TensorBroadcastTo(engine.Reshape(instance, new[] { 1, p }), shape),
            engine.TensorBroadcastTo(engine.Reshape(baseline, new[] { 1, p }), shape));
        var f = engine.TensorSubtract(predict(masked), engine.TensorBroadcastTo(emptyPrediction, new[] { m }));

        // Kernel SHAP weights (p - 1) / (C(p, s) s (p - s)), taken in logs and rescaled so the largest is 1: C(p, s)
        // overflows float once p passes ~130, which zeroed every weight; the weighted least squares solution does
        // not depend on the common scale
        var rest = engine.TensorAddScalar(engine.TensorNegate(size), ops.FromDouble(p));
        var logWeights = engine.TensorNegate(engine.TensorAdd(
            LogBinomialCoefficients(p, size), engine.TensorAdd(engine.TensorLog(size), engine.TensorLog(rest))));
        var weights = engine.TensorExp(engine.TensorSubtract(
            logWeights, engine.TensorBroadcastTo(engine.ReduceMax(logWeights, new[] { 0 }, true), new[] { m })));

        // Eliminate phi_p = gap - sum_{j < p} phi_j: regress f - z_p gap on z_j - z_p (j < p)
        var last = engine.Reshape(engine.TensorSlice(z, new[] { 0, p - 1 }, new[] { m, 1 }), new[] { m });
        var response = engine.TensorSubtract(f, engine.TensorMultiply(last, engine.TensorBroadcastTo(gap, new[] { m })));
        var design = engine.TensorSubtract(
            engine.TensorSlice(z, new[] { 0, 0 }, new[] { m, p - 1 }),
            engine.TensorBroadcastTo(engine.Reshape(last, new[] { m, 1 }), new[] { m, p - 1 }));
        var weighted = PopulationSearchOps.ScaleRows(design, weights);
        var transposed = engine.TensorTranspose(weighted);
        var gram = engine.TensorAdd(
            engine.TensorMatMul(transposed, design), engine.TensorMultiplyScalar(engine.TensorEye<T>(p - 1), PreprocessingTensorOps.Tolerance<T>(1e-10)));
        var rhs = engine.TensorMatMul(transposed, engine.Reshape(response, new[] { m, 1 }));
        var head = engine.Reshape(LinalgDispatch<T>.Solve(gram, rhs), new[] { p - 1 });
        var tail = engine.TensorSubtract(gap, engine.ReduceSum(head, new[] { 0 }, true));
        return engine.TensorConcatenate(new[] { head, tail }, 0);
    }

    /// <summary>
    /// ln C(p, s) = lgamma(p + 1) - lgamma(s + 1) - lgamma(p - s + 1) for every coalition size s [m] (no size is
    /// compared as T, which is not exact past 2048 in Half).
    /// </summary>
    private static Tensor<T> LogBinomialCoefficients(int p, Tensor<T> sizes)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        T one = ops.One;
        var rest = engine.TensorAddScalar(engine.TensorNegate(sizes), ops.FromDouble(p + 1));
        return engine.TensorSubtract(
            engine.TensorLgamma(PreprocessingTensorOps.Filled(new[] { sizes.Length }, ops.FromDouble(p + 1))),
            engine.TensorAdd(engine.TensorLgamma(engine.TensorAddScalar(sizes, one)), engine.TensorLgamma(rest)));
    }


    /// <summary>
    /// Predictions [m] for the rows of <paramref name="rows"/> [m, p]: the user's model, or by default the mean
    /// of the features as a simple proxy.
    /// </summary>
    private Tensor<T> Predict(Tensor<T> rows)
    {
        if (_predictFunc is not null)
            return Tensor<T>.FromVector(_predictFunc(rows.ToMatrix()));

        return Engine.ReduceMean(rows, new[] { 1 }, false);
    }

    /// <summary>
    /// Fits and transforms the data.
    /// </summary>
    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    /// <summary>
    /// Transforms the data by selecting SHAP-important features.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
        {
            throw new InvalidOperationException("SHAPSelector has not been fitted.");
        }

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SHAPSelector does not support inverse transformation.");
    }

    /// <summary>
    /// Gets a boolean mask indicating which features are selected.
    /// </summary>
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
        {
            throw new InvalidOperationException("SHAPSelector has not been fitted.");
        }

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
        {
            mask[idx] = true;
        }

        return mask;
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null)
        {
            return Array.Empty<string>();
        }

        if (inputFeatureNames is null)
        {
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();
        }

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
