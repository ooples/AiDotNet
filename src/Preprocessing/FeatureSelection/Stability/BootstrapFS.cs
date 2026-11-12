using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Stability;

/// <summary>
/// Bootstrap Feature Selection for robust feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Bootstrap Feature Selection uses bootstrap sampling (sampling with replacement)
/// to estimate the stability of feature selection. Features that are consistently
/// selected across many bootstrap samples are more reliable.
/// </para>
/// <para><b>For Beginners:</b> Bootstrap is like asking the same question many
/// times with slightly different data each time. If a feature is selected most
/// of the time, it's probably genuinely important. If it only gets selected
/// sometimes, it might just be noise.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class BootstrapFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBootstrapSamples;
    private readonly double _confidenceLevel;
    private readonly int? _randomState;

    private Vector<T>? _selectionProbabilities;
    private Vector<T>? _confidenceIntervals;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBootstrapSamples => _nBootstrapSamples;
    public double ConfidenceLevel => _confidenceLevel;
    public Vector<T>? SelectionProbabilities => _selectionProbabilities;
    public Vector<T>? ConfidenceIntervals => _confidenceIntervals;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public BootstrapFS(
        int nFeaturesToSelect = 10,
        int nBootstrapSamples = 100,
        double confidenceLevel = 0.95,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nBootstrapSamples < 1)
            throw new ArgumentException("Number of bootstrap samples must be at least 1.", nameof(nBootstrapSamples));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBootstrapSamples = nBootstrapSamples;
        _confidenceLevel = confidenceLevel;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "BootstrapFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var x = Tensor<T>.FromMatrix(data);
        var y = Tensor<T>.FromVector(target);
        var allScores = new List<Tensor<T>>();
        var selectionCounts = new int[p];

        for (int b = 0; b < _nBootstrapSamples; b++)
        {
            // Create bootstrap sample (with replacement)
            var sampleIndices = new int[n];
            for (int i = 0; i < n; i++)
                sampleIndices[i] = random.Next(n);

            // Compute feature scores on bootstrap sample
            var scores = ComputeScores(x, y, sampleIndices);
            allScores.Add(scores);

            // Track which features would be selected
            foreach (int j in PreprocessingTensorOps.TopKIndicesSorted(scores, _nFeaturesToSelect))
                selectionCounts[j]++;
        }

        // Compute selection probabilities
        var probabilities = Engine.TensorDivideScalar(
            new Tensor<int>(selectionCounts, new[] { p }).Cast<T>(), NumOps.FromDouble(_nBootstrapSamples));
        _selectionProbabilities = probabilities.ToVector();

        // Compute confidence intervals for feature importance (width of the bootstrap percentile interval)
        double alpha = 1 - _confidenceLevel;
        int lowerIdx = (int)(alpha / 2 * _nBootstrapSamples);
        int upperIdx = (int)((1 - alpha / 2) * _nBootstrapSamples);
        var sortedScores = Engine.TensorSort(Engine.TensorStack(allScores.ToArray(), 0), 0, false).Item1;
        var lower = Engine.TensorSliceAxis(sortedScores, 0, Math.Max(0, lowerIdx));
        var upper = Engine.TensorSliceAxis(sortedScores, 0, Math.Min(_nBootstrapSamples - 1, upperIdx));
        _confidenceIntervals = Engine.TensorSubtract(upper, lower).ToVector();

        // Select features by probability (most consistently selected)
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(probabilities, _nFeaturesToSelect);

        IsFitted = true;
    }

    /// <summary>|Pearson r| of every column with the target over the bootstrap rows (0 for a constant column or target).</summary>
    private Tensor<T> ComputeScores(Tensor<T> x, Tensor<T> y, int[] sampleIndices)
    {
        var rows = new Tensor<int>(sampleIndices, new[] { sampleIndices.Length });
        return Engine.TensorAbs(PreprocessingTensorOps.TargetCorrelations(
            Engine.TensorIndexSelect(x, rows, 0), Engine.TensorIndexSelect(y, rows, 0)));
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
            throw new InvalidOperationException("BootstrapFS has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("BootstrapFS does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BootstrapFS has not been fitted.");

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
