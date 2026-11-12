using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Multivariate;

/// <summary>
/// Conditional Mutual Information Maximization (CMIM) Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features by maximizing mutual information with the target conditioned
/// on already selected features, ensuring each new feature adds unique information.
/// </para>
/// <para><b>For Beginners:</b> CMIM asks "what new information does this feature
/// give me that I don't already have?" It selects features that add something new,
/// conditioned on what you've already picked. This avoids redundant features more
/// rigorously than simple redundancy penalties.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class CMIMSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private Vector<T>? _cmimScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBins => _nBins;
    public Vector<T>? CMIMScores => _cmimScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public CMIMSelector(
        int nFeaturesToSelect = 10,
        int nBins = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBins = nBins;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "CMIMSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        if (n == 0)
            throw new InvalidOperationException("Sequence contains no elements");

        var engine = Engine;

        // Discretize
        var XBins = FeatureSelectionInformationOps.EqualWidthBinsPaddedWidth(Tensor<T>.FromMatrix(data), _nBins);
        var yBins = FeatureSelectionInformationOps.EqualWidthBinsPaddedWidth(engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 }), _nBins);

        // Compute MI(X_j, Y) for all features
        var miWithTarget = FeatureSelectionInformationOps.MutualInformationBits<T>(XBins, _nBins, yBins, _nBins);

        // Greedy forward selection with the CMIM criterion (Fleuret 2004):
        // score_j = min over selected s of I(X_j; Y | X_s), and I(X_j; Y) before anything is selected
        var selected = new List<int>();
        var remaining = FeatureSelectionTensorOps.Filled(new[] { p }, NumOps.One);
        var conditionalMI = miWithTarget;
        Tensor<T>? minConditionalMI = null;
        var cmimScores = new Tensor<T>(new[] { p });

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        while (selected.Count < numToSelect && selected.Count < p)
        {
            // Ties go to the lowest index
            var best = FeatureSelectionTensorOps.TopKWhere(conditionalMI, remaining, 1);
            if (best.Length == 0)
                break;

            int bestFeature = best[0];
            var chosen = FeatureSelectionTensorOps.OneHot<T>(bestFeature, p);
            selected.Add(bestFeature);
            remaining = engine.TensorMultiply(remaining, FeatureSelectionTensorOps.OneMinus(chosen));
            cmimScores = engine.TensorAdd(cmimScores, engine.TensorMultiply(chosen, conditionalMI));

            // I(X_j; Y | X_s) = I(X_j, X_s; Y) - I(X_s; Y), with the joint variable X_s * nBins + X_j
            var pairs = FeatureSelectionInformationOps.Combine(engine.TensorSlice(XBins, new[] { 0, bestFeature }, new[] { n, 1 }), XBins, _nBins);
            var jointMI = FeatureSelectionInformationOps.MutualInformationBits<T>(pairs, _nBins * _nBins, yBins, _nBins);
            var selectedMI = engine.TensorBroadcastTo(engine.TensorSlice(miWithTarget, new[] { bestFeature }, new[] { 1 }), new[] { p });
            var cmi = engine.TensorMax(engine.TensorSubtract(jointMI, selectedMI), NumOps.Zero);
            minConditionalMI = minConditionalMI is null ? cmi : engine.TensorMin(minConditionalMI, cmi);
            conditionalMI = minConditionalMI;
        }

        _cmimScores = cmimScores.ToVector();
        _selectedIndices = selected.OrderBy(x => x).ToArray();
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
            throw new InvalidOperationException("CMIMSelector has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("CMIMSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CMIMSelector has not been fitted.");

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
