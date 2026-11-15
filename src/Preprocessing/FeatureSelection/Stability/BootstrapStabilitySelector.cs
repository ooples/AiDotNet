using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Stability;

/// <summary>
/// Bootstrap Stability-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses bootstrap sampling to assess the stability of feature selection,
/// selecting features that are consistently chosen across multiple bootstrap
/// samples.
/// </para>
/// <para><b>For Beginners:</b> Bootstrap means taking many random samples
/// (with replacement) from your data. This method runs feature selection
/// on each sample and counts how often each feature is selected. Features
/// that are consistently selected are more reliable choices.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class BootstrapStabilitySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBootstraps;
    private readonly double _threshold;

    private Vector<T>? _selectionProbabilities;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? SelectionProbabilities => _selectionProbabilities;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public BootstrapStabilitySelector(
        int nFeaturesToSelect = 10,
        int nBootstraps = 100,
        double threshold = 0.5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBootstraps = nBootstraps;
        _threshold = threshold;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "BootstrapStabilitySelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var rand = RandomHelper.CreateSecureRandom();
        var x = Tensor<T>.FromMatrix(data);
        var y = Tensor<T>.FromVector(target);

        var selectionCounts = new int[p];

        for (int b = 0; b < _nBootstraps; b++)
        {
            // Create bootstrap sample
            var indices = new int[n];
            for (int i = 0; i < n; i++)
                indices[i] = rand.Next(n);

            // Run simple correlation-based selection on bootstrap sample
            var rows = new Tensor<int>(indices, new[] { n });
            var selected = SelectFeatures(Engine.TensorIndexSelect(x, rows, 0), Engine.TensorIndexSelect(y, rows, 0));
            foreach (int j in selected)
                selectionCounts[j]++;
        }

        // Compute selection probabilities
        var probabilities = Engine.TensorDivideScalar(
            new Tensor<int>(selectionCounts, new[] { p }).Cast<T>(), NumOps.FromDouble(_nBootstraps));
        _selectionProbabilities = probabilities.ToVector();

        // Select features above threshold, sorted by probability; if not enough, add more by probability
        var order = PreprocessingTensorOps.StableOrder(probabilities, descending: true);
        int above = PreprocessingTensorOps.CountNonZero(
            PreprocessingTensorOps.AtLeast(probabilities, NumOps.FromDouble(_threshold)));
        var candidates = order.Take(Math.Min(above, _nFeaturesToSelect)).ToList();
        if (candidates.Count < _nFeaturesToSelect)
            candidates.AddRange(order.Skip(above).Take(_nFeaturesToSelect - candidates.Count));

        _selectedIndices = candidates.OrderBy(j => j).ToArray();
        IsFitted = true;
    }

    /// <summary>The features with the largest |Pearson r| on a bootstrap sample (0 for a constant column or target).</summary>
    private int[] SelectFeatures(Tensor<T> x, Tensor<T> y) =>
        PreprocessingTensorOps.TopKIndicesSorted(
            Engine.TensorAbs(PreprocessingTensorOps.TargetCorrelations(x, y)), _nFeaturesToSelect);

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
            throw new InvalidOperationException("BootstrapStabilitySelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("BootstrapStabilitySelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BootstrapStabilitySelector has not been fitted.");

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
