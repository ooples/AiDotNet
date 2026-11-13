using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Correlation;

/// <summary>
/// Partial Correlation for feature selection controlling for other variables.
/// </summary>
/// <remarks>
/// <para>
/// Partial correlation measures the relationship between two variables while
/// controlling for (removing the effect of) other variables. Helps identify
/// direct relationships vs those mediated by confounders.
/// </para>
/// <para><b>For Beginners:</b> Regular correlation can be misleading if two
/// variables are both caused by a third (confounder). Partial correlation
/// "holds constant" other variables to find the true direct relationship.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class PartialCorrelation<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly bool _controlForAllOthers;

    private Vector<T>? _partialCorrelations;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? PartialCorrelations => _partialCorrelations;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public PartialCorrelation(
        int nFeaturesToSelect = 10,
        bool controlForAllOthers = true,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _controlForAllOthers = controlForAllOthers;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "PartialCorrelation requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        var X = Tensor<T>.FromMatrix(data);
        var y = Tensor<T>.FromVector(target);
        Tensor<T> partialCorrelations;

        if (_controlForAllOthers && p > 1)
        {
            // Correlation matrix of [X | y] (a column whose population std is below 1e-10 correlates 0)
            var fullMatrix = engine.TensorConcatenate(new[] { X, engine.Reshape(y, new[] { n, 1 }) }, 1);
            var corrMatrix = FeatureScoringTensorOps.CorrelationMatrix(fullMatrix, n * 1e-20);
            var targetColumn = engine.Reshape(
                engine.TensorSlice(corrMatrix, new[] { 0, p }, new[] { p, 1 }), new[] { p });

            // Precision matrix via partially pivoted LU. The engine factorizations run only in float and double, so
            // a Half, BFloat16 or decimal matrix is factored in double and the result cast back.
            var precisionPartials = typeof(T) == typeof(float) || typeof(T) == typeof(double)
                ? PrecisionPartialCorrelations(corrMatrix, p)
                : PrecisionPartialCorrelations(corrMatrix.Cast<double>(), p)?.Cast<T>();

            // Fallback to simple correlation if matrix inversion fails
            partialCorrelations = precisionPartials ?? engine.TensorAbs(targetColumn);
        }
        else
        {
            // Simple correlation when not controlling for others
            partialCorrelations = engine.TensorAbs(PreprocessingTensorOps.TargetCorrelations(X, y));
        }

        _partialCorrelations = partialCorrelations.ToVector();

        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(engine.TensorAbs(partialCorrelations), nToSelect);

        IsFitted = true;
    }
    /// <summary>
    /// Partial correlations <c>-P_jy / sqrt(P_jj P_yy)</c> of the first <paramref name="p"/> variables with the last
    /// one, from the precision (inverse) of the [p + 1, p + 1] correlation matrix; null when a pivot of its
    /// partially pivoted LU factorization is below 1e-10 (singular matrix).
    /// </summary>
    private static Tensor<TS>? PrecisionPartialCorrelations<TS>(Tensor<TS> corrMatrix, int p)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<TS>();
        int size = p + 1;
        TS tolerance = PreprocessingTensorOps.Tolerance<TS>(1e-10);
        var (lu, pivots) = AiDotNet.Helpers.LinalgDispatch<TS>.LuFactor(corrMatrix);
        var pivotMagnitudes = engine.TensorAbs(AiDotNet.Helpers.LinalgDispatch<TS>.Diagonal(lu));
        if (PreprocessingTensorOps.CountNonZero(PreprocessingTensorOps.AtLeast(pivotMagnitudes, tolerance)) != size)
            return null;

        var precision = AiDotNet.Helpers.LinalgDispatch<TS>.LuSolve(lu, pivots, engine.TensorEye<TS>(size));
        var diagonal = AiDotNet.Helpers.LinalgDispatch<TS>.Diagonal(precision);
        var pjj = engine.TensorSlice(diagonal, new[] { 0 }, new[] { p });
        TS pyy = engine.TensorMaxValue(engine.TensorSlice(diagonal, new[] { p }, new[] { 1 }));
        var pjy = engine.Reshape(engine.TensorSlice(precision, new[] { 0, p }, new[] { p, 1 }), new[] { p });
        var zeros = new Tensor<TS>(new[] { p });
        var usable = ops.GreaterThan(pyy, tolerance) ? engine.TensorGreaterThan(pjj, tolerance) : zeros;
        var safe = engine.TensorWhere(usable, pjj, PreprocessingTensorOps.Filled(new[] { p }, ops.One));
        return engine.TensorWhere(usable, engine.TensorNegate(engine.TensorDivide(pjy,
            engine.TensorSqrt(engine.TensorMultiplyScalar(safe, pyy)))), zeros);
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
            throw new InvalidOperationException("PartialCorrelation has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("PartialCorrelation does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("PartialCorrelation has not been fitted.");

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
