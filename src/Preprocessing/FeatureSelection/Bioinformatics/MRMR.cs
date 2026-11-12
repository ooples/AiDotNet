using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Bioinformatics;

/// <summary>
/// Minimum Redundancy Maximum Relevance (mRMR) feature selection.
/// </summary>
/// <remarks>
/// <para>
/// mRMR selects features that have high relevance to the target (maximum relevance)
/// while minimizing redundancy among selected features. Originally developed for
/// gene expression analysis.
/// </para>
/// <para><b>For Beginners:</b> Imagine choosing team members: you want people with
/// the right skills (relevance to the task) but with different specialties so they
/// don't overlap (minimum redundancy). mRMR finds features that are predictive of
/// the target but provide diverse, non-overlapping information.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class MRMR<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;
    private readonly MRMRMethod _method;

    private Vector<T>? _mrmrScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public enum MRMRMethod
    {
        /// <summary>MID: Mutual Information Difference (I - R)</summary>
        MID,
        /// <summary>MIQ: Mutual Information Quotient (I / R)</summary>
        MIQ
    }

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? MRMRScores => _mrmrScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public MRMR(
        int nFeaturesToSelect = 10,
        int nBins = 10,
        MRMRMethod method = MRMRMethod.MID,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nBins < 2)
            throw new ArgumentException("Number of bins must be at least 2.", nameof(nBins));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBins = nBins;
        _method = method;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "MRMR requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Discretize data and target (one-hot bins)
        var engine = Engine;
        var xBins = PreprocessingTensorOps.RangeScaledBinOneHot(Tensor<T>.FromMatrix(data), _nBins);
        var yBins = PreprocessingTensorOps.RangeScaledBinOneHot(engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 }), _nBins);

        // Mutual information with target
        var relevance = engine.Reshape(
            PreprocessingTensorOps.HistogramMutualInformation(xBins, yBins, _nBins, bits: false, clampAtZero: false),
            new[] { p });

        // Greedy forward selection with mRMR criterion; redundancySum[j] accumulates MI(j, s) over selected s
        var selected = new List<int>();
        var isSelected = new bool[p];
        var redundancySum = new Tensor<T>(new[] { p });
        var excluded = PreprocessingTensorOps.Filled(new[] { p }, NumOps.MinValue);
        T eps = PreprocessingTensorOps.Tolerance<T>(1e-10);
        var selectedScores = new Tensor<T>[p];

        while (selected.Count < Math.Min(_nFeaturesToSelect, p))
        {
            var redundancy = selected.Count > 0
                ? engine.TensorDivideScalar(redundancySum, PreprocessingTensorOps.CountAsScalar<T>(selected.Count))
                : redundancySum;

            var score = _method == MRMRMethod.MID
                ? engine.TensorSubtract(relevance, redundancy)
                : engine.TensorWhere(engine.TensorGreaterThan(redundancy, eps), engine.TensorDivide(relevance, redundancy), relevance);

            var open = engine.TensorWhere(isSelected, excluded, score);
            int bestFeature = PreprocessingTensorOps.FirstArgMax(open);
            selectedScores[bestFeature] = open;
            selected.Add(bestFeature);
            isSelected[bestFeature] = true;

            var bestBins = engine.TensorSlice(xBins, new[] { 0, bestFeature * _nBins }, new[] { n, _nBins });
            redundancySum = engine.TensorAdd(redundancySum, engine.Reshape(
                PreprocessingTensorOps.HistogramMutualInformation(xBins, bestBins, _nBins, bits: false, clampAtZero: false),
                new[] { p }));
        }

        // Each selected feature keeps the mRMR score it was selected with; the rest stay 0
        var scores = new Tensor<T>(new[] { p });
        foreach (int j in selected)
        {
            var position = new Tensor<int>(new[] { j }, new[] { 1 });
            scores = engine.TensorIndexCopy(scores, 0, position, engine.TensorIndexSelect(selectedScores[j], position, 0));
        }

        _mrmrScores = scores.ToVector();
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
            throw new InvalidOperationException("MRMR has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("MRMR does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MRMR has not been fitted.");

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
