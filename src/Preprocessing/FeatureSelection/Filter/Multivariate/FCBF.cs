using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Multivariate;

/// <summary>
/// Fast Correlation-Based Filter (FCBF) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// FCBF uses symmetrical uncertainty to identify relevant features and remove
/// redundant ones. It's efficient for high-dimensional data with the concept
/// of predominant features that are not dominated by any other feature.
/// </para>
/// <para><b>For Beginners:</b> FCBF finds features that are strongly related to
/// the target and removes any feature that's "dominated" by another (meaning
/// the other feature is just as good at predicting the target AND is more
/// correlated with this feature). This efficiently eliminates redundancy.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class FCBF<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>
{
    private const int DiscretizationBins = 10;
    private readonly double _threshold;

    private Vector<T>? _symmetricalUncertainties;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public Vector<T>? SymmetricalUncertainties => _symmetricalUncertainties;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public FCBF(
        double threshold = 0.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        _threshold = threshold;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "FCBF requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;
        var info = DiscreteInformation<T>.ForTarget(
            FeatureSelectionTensorOps.ScaledBins(FilterStatisticTensorOps.AsTensor(data), DiscretizationBins), DiscretizationBins, target);
        var suTarget = info.SymmetricUncertainty();
        _symmetricalUncertainties = suTarget.ToVector();

        // Relevant features (SU > threshold) by decreasing SU, ties by index
        var above = Engine.TensorGreaterThan(suTarget, NumOps.FromDouble(_threshold)).Cast<int>().ToArray();
        var relevant = FilterStatisticTensorOps.StableOrder(suTarget, descending: true)
            .Where(j => above[j] == 1)
            .ToList();

        // Remove redundant features: fq is dominated by an earlier fp when SU(fp, fq) >= SU(fq, Y)
        var selectedList = new List<int>();
        var processed = new HashSet<int>();
        foreach (int fp in relevant)
        {
            if (processed.Contains(fp))
                continue;

            selectedList.Add(fp);
            processed.Add(fp);
            var notDominated = Engine.TensorLessThan(info.FeatureSymmetricUncertainty(fp), suTarget).Cast<int>().ToArray();
            foreach (int fq in relevant.Where(f => !processed.Contains(f)).ToList())
            {
                if (notDominated[fq] == 0)
                    processed.Add(fq);
            }
        }

        _selectedIndices = selectedList.OrderBy(x => x).ToArray();

        if (_selectedIndices.Length == 0)
        {
            // Fall back to top by SU
            _selectedIndices = FilterStatisticTensorOps.TopKIndicesSorted(suTarget, Math.Min(10, p));
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
            throw new InvalidOperationException("FCBF has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("FCBF does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FCBF has not been fitted.");

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
