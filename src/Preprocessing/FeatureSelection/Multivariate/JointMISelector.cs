using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Multivariate;

/// <summary>
/// Joint Mutual Information Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features by maximizing joint mutual information with the target while
/// minimizing redundancy among selected features.
/// </para>
/// <para><b>For Beginners:</b> Regular mutual information looks at features one
/// at a time. Joint MI considers how features work together. It picks features
/// that provide unique information about the target, avoiding features that just
/// repeat what others already tell you.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class JointMISelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;
    private readonly double _redundancyPenalty;

    private Vector<T>? _jmiScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBins => _nBins;
    public Vector<T>? JMIScores => _jmiScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public JointMISelector(
        int nFeaturesToSelect = 10,
        int nBins = 10,
        double redundancyPenalty = 1.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBins = nBins;
        _redundancyPenalty = redundancyPenalty;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "JointMISelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Discretize features for MI computation
        var XBins = FeatureSelectionInformationOps.EqualWidthBinsPaddedWidth(Tensor<T>.FromMatrix(data), _nBins);
        var yBins = FeatureSelectionInformationOps.EqualWidthBinsPaddedWidth(engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 }), _nBins);

        // Compute MI(X_j, Y) for all features
        var miWithTarget = FeatureSelectionInformationOps.MutualInformationBits<T>(XBins, _nBins, yBins, _nBins);

        // Greedy forward selection with the JMI criterion (Yang & Moody 1999; Brown et al. 2012, eq. 14):
        // score_k = I(X_k;Y) - penalty / |S| * sum over selected j of [I(X_k;X_j) - I(X_k;X_j|Y)]
        var selected = new List<int>();
        var remaining = FeatureSelectionTensorOps.Filled(new[] { p }, NumOps.One);
        var redundancy = new Tensor<T>(new[] { p });
        var scores = miWithTarget;
        var jmiScores = new Tensor<T>(new[] { p });
        T penalty = NumOps.FromDouble(_redundancyPenalty);

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        while (selected.Count < numToSelect && selected.Count < p)
        {
            // Ties go to the lowest index
            var best = FeatureSelectionTensorOps.TopKWhere(scores, remaining, 1);
            if (best.Length == 0)
                break;

            int bestFeature = best[0];
            var chosen = FeatureSelectionTensorOps.OneHot<T>(bestFeature, p);
            selected.Add(bestFeature);
            remaining = engine.TensorMultiply(remaining, FeatureSelectionTensorOps.OneMinus(chosen));
            jmiScores = engine.TensorAdd(jmiScores, engine.TensorMultiply(chosen, scores));

            // I(X_k;X_j) for the newly selected j, and I(X_k;X_j|Y) = I(X_k; (X_j,Y)) - I(X_k;Y)
            var selectedBins = engine.TensorSlice(XBins, new[] { 0, bestFeature }, new[] { n, 1 });
            var miWithSelected = FeatureSelectionInformationOps.MutualInformationBits<T>(XBins, _nBins, selectedBins, _nBins);
            var selectedAndTarget = FeatureSelectionInformationOps.Combine(selectedBins, yBins, _nBins);
            var conditional = engine.TensorMax(engine.TensorSubtract(
                FeatureSelectionInformationOps.MutualInformationBits<T>(XBins, _nBins, selectedAndTarget, _nBins * _nBins),
                miWithTarget), NumOps.Zero);
            redundancy = engine.TensorAdd(redundancy, engine.TensorSubtract(miWithSelected, conditional));

            var meanRedundancy = engine.TensorDivideScalar(redundancy, FeatureSelectionTensorOps.Count<T>(selected.Count));
            scores = engine.TensorSubtract(miWithTarget, engine.TensorMultiplyScalar(meanRedundancy, penalty));
        }

        _jmiScores = jmiScores.ToVector();
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
            throw new InvalidOperationException("JointMISelector has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("JointMISelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("JointMISelector has not been fitted.");

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
