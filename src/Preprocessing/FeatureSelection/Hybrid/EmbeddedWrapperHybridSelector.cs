using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Hybrid;

/// <summary>
/// Embedded-Wrapper Hybrid Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Combines embedded (L1 regularization) and wrapper methods: first uses
/// Lasso to identify a candidate set, then refines selection using sequential
/// feature addition/removal.
/// </para>
/// <para><b>For Beginners:</b> This method first uses Lasso regression to
/// automatically shrink unimportant features to zero, giving us a candidate
/// set. Then it fine-tunes this selection by trying to add or remove features
/// to improve model performance. This balances speed with accuracy.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class EmbeddedWrapperHybridSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _lassoAlpha;
    private readonly int _wrapperIterations;

    private Vector<T>? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? FeatureScores => _featureScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public EmbeddedWrapperHybridSelector(
        int nFeaturesToSelect = 10,
        double lassoAlpha = 0.01,
        int wrapperIterations = 20,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _lassoAlpha = lassoAlpha;
        _wrapperIterations = wrapperIterations;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "EmbeddedWrapperHybridSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Normalize features (sample standard deviation)
        var raw = Tensor<T>.FromMatrix(data);
        var means = engine.TensorBroadcastTo(engine.ReduceMean(raw, new[] { 0 }, true), new[] { n, p });
        var centered = engine.TensorSubtract(raw, means);
        var stds = engine.TensorAddScalar(
            engine.TensorSqrt(engine.TensorDivideScalar(
                engine.ReduceSum(engine.TensorSquare(centered), new[] { 0 }, true), NumOps.FromDouble(n - 1))),
            NumOps.FromDouble(1e-10));
        var x = engine.TensorDivide(centered, engine.TensorBroadcastTo(stds, new[] { n, p }));

        var y = Tensor<T>.FromVector(target);
        var yNorm = engine.TensorSubtract(y, engine.TensorBroadcastTo(engine.ReduceMean(y, new[] { 0 }, true), new[] { n }));

        // Phase 1: Embedded - Lasso with coordinate descent
        var beta = LassoCoordinateDescent(x, yNorm, _lassoAlpha, 100);
        var absBeta = engine.TensorAbs(beta);

        // Get candidate set from Lasso: nonzero coefficients by decreasing magnitude, then the rest if too few
        var order = PreprocessingTensorOps.StableOrder(absBeta, descending: true);
        var nonzero = new HashSet<int>(PreprocessingTensorOps.MaskPositions(
            engine.TensorGreaterThan(absBeta, NumOps.FromDouble(1e-10))));
        var candidates = order.Where(j => nonzero.Contains(j)).ToList();
        if (candidates.Count < _nFeaturesToSelect)
            candidates.AddRange(order.Where(j => !nonzero.Contains(j)).Take(_nFeaturesToSelect - candidates.Count));

        // Phase 2: Wrapper - refine using forward/backward selection
        var selected = candidates.Take(_nFeaturesToSelect).ToList();
        var bestScore = EvaluateSubset(x, yNorm, selected);

        for (int iter = 0; iter < _wrapperIterations; iter++)
        {
            bool improved = false;

            // Try to remove features
            for (int i = selected.Count - 1; i >= 0 && selected.Count > 1; i--)
            {
                int feature = selected[i];
                selected.RemoveAt(i);
                var score = EvaluateSubset(x, yNorm, selected);
                if (PopulationSearchOps.Greater(score, bestScore))
                {
                    bestScore = score;
                    improved = true;
                }
                else
                {
                    selected.Insert(i, feature);
                }
            }

            // Try to add features
            foreach (int j in candidates.Where(j => !selected.Contains(j)).ToList())
            {
                if (selected.Count >= _nFeaturesToSelect) break;

                selected.Add(j);
                var score = EvaluateSubset(x, yNorm, selected);
                if (PopulationSearchOps.Greater(score, bestScore))
                {
                    bestScore = score;
                    improved = true;
                }
                else
                {
                    selected.RemoveAt(selected.Count - 1);
                }
            }

            if (!improved) break;
        }

        // Compute final scores: |beta| plus 1 for the selected features
        var featureScores = engine.TensorAdd(absBeta, PopulationSearchOps.IndexMask<T>(selected.ToArray(), p));
        _featureScores = featureScores.ToVector();

        _selectedIndices = selected.OrderBy(i => i).ToArray();
        IsFitted = true;
    }

    /// <summary>
    /// Lasso by cyclic coordinate descent (Friedman, Hastie and Tibshirani 2010) for min (1/2n) |y - X b|^2 +
    /// alpha |b|_1: each sweep updates b_j = S(rho_j, alpha) / ((1/n) sum_i x_ij^2) with
    /// rho_j = (1/n) sum_i x_ij (y_i - sum_{k != j} x_ik b_k) and soft threshold S.
    /// </summary>
    internal static Tensor<T> LassoCoordinateDescent(Tensor<T> x, Tensor<T> y, double alpha, int sweeps)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int n = x.Shape[0];
        int p = x.Shape[1];
        T invN = ops.FromDouble(1.0 / n);
        T zero = ops.Zero;
        T negativeLambda = ops.FromDouble(-alpha);

        var columnScale = engine.TensorMultiplyScalar(engine.ReduceSum(engine.TensorSquare(x), new[] { 0 }, false), invN);
        var safeScale = engine.TensorWhere(
            engine.TensorGreaterThan(columnScale, zero), columnScale, PreprocessingTensorOps.Filled(new[] { p }, ops.One));
        var beta = new Tensor<T>(new[] { p });
        var residual = y;

        for (int sweep = 0; sweep < sweeps; sweep++)
        {
            for (int j = 0; j < p; j++)
            {
                var column = engine.Reshape(engine.TensorSlice(x, new[] { 0, j }, new[] { n, 1 }), new[] { n });
                var old = PopulationSearchOps.At(beta, j);

                // Partial residual with feature j's contribution added back
                var partial = engine.TensorAdd(residual, PopulationSearchOps.Scale(column, old));
                var rho = engine.TensorMultiplyScalar(
                    engine.ReduceSum(engine.TensorMultiply(column, partial), new[] { 0 }, true), invN);
                var shrunk = engine.TensorSubtract(
                    engine.TensorMax(engine.TensorAddScalar(rho, negativeLambda), zero),
                    engine.TensorMax(engine.TensorAddScalar(engine.TensorNegate(rho), negativeLambda), zero));
                var updated = engine.TensorDivide(shrunk, PopulationSearchOps.At(safeScale, j));

                beta = PopulationSearchOps.ReplaceAt(beta, j, updated);
                residual = engine.TensorSubtract(partial, PopulationSearchOps.Scale(column, updated));
            }
        }

        return beta;
    }

    /// <summary>R^2 = 1 - SS_res / sum y^2 of a ridge (1e-6) least-squares fit of centered y on the features.</summary>
    private static Tensor<T> EvaluateSubset(Tensor<T> x, Tensor<T> y, List<int> features)
    {
        if (features.Count == 0)
            return PreprocessingTensorOps.Filled(new[] { 1 }, MathHelper.GetNumericOperations<T>().MinValue);

        var design = SubsetRegressionOps.Columns(x, features);
        var coefficients = SubsetRegressionOps.RidgeCoefficients(design, y, 1e-6);
        return SubsetRegressionOps.RSquared(
            y, SubsetRegressionOps.Predict(design, coefficients), new Tensor<T>(new[] { 1 }), out _);
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
            throw new InvalidOperationException("EmbeddedWrapperHybridSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("EmbeddedWrapperHybridSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("EmbeddedWrapperHybridSelector has not been fitted.");

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
