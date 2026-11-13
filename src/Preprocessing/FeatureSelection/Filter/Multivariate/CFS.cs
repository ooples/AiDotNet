using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Multivariate;

/// <summary>
/// Correlation-based Feature Selection (CFS) for selecting feature subsets.
/// </summary>
/// <remarks>
/// <para>
/// CFS evaluates feature subsets based on a heuristic: good feature sets contain
/// features highly correlated with the class but uncorrelated with each other.
/// Uses greedy forward selection with the CFS merit function.
/// </para>
/// <para><b>For Beginners:</b> CFS looks for features that are good at predicting
/// the target but aren't redundant with each other. If two features give the same
/// information, CFS keeps only one. This prevents selecting features that are
/// copies of each other while maximizing predictive power.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class CFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;

    private Vector<T>? _featureClassCorrelations;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? FeatureClassCorrelations => _featureClassCorrelations;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public CFS(
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
            "CFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;
        var x = FilterStatisticTensorOps.AsTensor(data);
        // |r| with the target and between features, on columns scaled to unit largest deviation: the raw centred sums
        // of squares overflow Half once a deviation passes 256 (r came out NaN); 0 for a constant column
        var classCorrelation = FeatureSelectionTensorOps.AbsoluteCorrelation(x, FilterStatisticTensorOps.AsTensor(target));
        var interCorrelation = Engine.TensorAbs(FeatureSelectionTensorOps.UnitDeviationCorrelationMatrix(x));
        _featureClassCorrelations = classCorrelation.ToVector();

        // greedy forward selection; accumulators are broadcast over the [p] candidates
        T one = NumOps.One;
        T two = NumOps.FromDouble(2);
        var zero = FilterStatisticTensorOps.FilledLike(classCorrelation, NumOps.Zero);
        var count = FilterStatisticTensorOps.FilledLike(classCorrelation, one);
        var selectedClass = zero;
        var selectedPairs = zero;
        var withSelected = zero;
        var (selected, _) = DiscreteInformation<T>.Greedy(p, _nFeaturesToSelect, chosen =>
        {
            if (chosen.Count > 0)
            {
                int last = chosen[chosen.Count - 1];
                selectedClass = Engine.TensorAdd(selectedClass, Broadcast(classCorrelation, last, p));
                selectedPairs = Engine.TensorAdd(selectedPairs, Broadcast(withSelected, last, p));
                withSelected = Engine.TensorAdd(withSelected, PreprocessingTensorOps.ColumnVector(interCorrelation, last));
                count = Engine.TensorAddScalar(count, one);
            }

            var pairSum = Engine.TensorMultiplyScalar(Engine.TensorAdd(selectedPairs, withSelected), two);
            return FilterStatisticTensorOps.GuardedDivide(
                Engine.TensorAdd(selectedClass, classCorrelation),
                Engine.TensorSqrt(Engine.TensorAdd(count, pairSum)), 1e-10, NumOps.Zero);
        });

        _selectedIndices = selected;
        IsFitted = true;
    }

    // entry `index` of a [p] tensor repeated p times
    private Tensor<T> Broadcast(Tensor<T> values, int index, int p) =>
        Engine.TensorBroadcastTo(FeatureSelectionTensorOps.Entries(values, new[] { index }), new[] { p });

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
            throw new InvalidOperationException("CFS has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("CFS does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CFS has not been fitted.");

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
