using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Hybrid;

/// <summary>
/// Mutual Information and Correlation Hybrid Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Combines mutual information (captures non-linear relationships) with
/// Pearson correlation (captures linear relationships) to identify features
/// that are important under both perspectives.
/// </para>
/// <para><b>For Beginners:</b> Some features have linear relationships
/// (more X means more Y), while others have non-linear patterns (X affects
/// Y in complex ways). This method uses correlation to find linear patterns
/// and mutual information to find any pattern, then combines both scores.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class MICorrelationHybrid<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _miWeight;
    private readonly int _nBins;

    private Vector<T>? _miScores;
    private Vector<T>? _correlationScores;
    private Vector<T>? _hybridScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double MIWeight => _miWeight;
    public Vector<T>? MIScores => _miScores;
    public Vector<T>? CorrelationScores => _correlationScores;
    public Vector<T>? HybridScores => _hybridScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public MICorrelationHybrid(
        int nFeaturesToSelect = 10,
        double miWeight = 0.5,
        int nBins = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (miWeight < 0 || miWeight > 1)
            throw new ArgumentException("MI weight must be between 0 and 1.", nameof(miWeight));

        _nFeaturesToSelect = nFeaturesToSelect;
        _miWeight = miWeight;
        _nBins = nBins;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "MICorrelationHybrid requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        if (n == 0 || p == 0)
            throw new InvalidOperationException("Sequence contains no elements");

        var engine = Engine;
        var X = Tensor<T>.FromMatrix(data);
        var y = engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 });

        // Compute Pearson correlation
        var correlation = FeatureSelectionInformationOps.AbsCorrelationWithTarget(X, y, NumOps.Zero);

        // Compute Mutual Information (nats) over nBins equal-width bins (numpy.histogram bins)
        var xBins = FeatureSelectionInformationOps.EqualWidthBinsExactRange(X, _nBins);
        var yBins = FeatureSelectionInformationOps.EqualWidthBinsExactRange(y, _nBins);
        var mi = FeatureSelectionInformationOps.MutualInformation(
            FeatureSelectionInformationOps.JointCounts<T>(xBins, _nBins, yBins, _nBins), n);

        // Normalize scores
        var normalizedMI = NormalizeByMax(mi);
        var normalizedCorr = NormalizeByMax(correlation);
        var hybrid = engine.TensorAdd(
            engine.TensorMultiplyScalar(normalizedMI, NumOps.FromDouble(_miWeight)),
            engine.TensorMultiplyScalar(normalizedCorr, NumOps.FromDouble(1 - _miWeight)));

        _miScores = mi.ToVector();
        _correlationScores = correlation.ToVector();
        _hybridScores = hybrid.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = FeatureSelectionTensorOps.TopKIndicesSorted(hybrid, numToSelect);

        IsFitted = true;
    }

    /// <summary>Scores divided by their maximum, or all 0 when the maximum is not above 1e-10.</summary>
    private Tensor<T> NormalizeByMax(Tensor<T> scores)
    {
        T max = Engine.TensorMaxValue(scores);
        return NumOps.GreaterThan(max, NumOps.FromDouble(1e-10))
            ? Engine.TensorDivideScalar(scores, max)
            : new Tensor<T>(scores.Shape.ToArray());
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
            throw new InvalidOperationException("MICorrelationHybrid has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("MICorrelationHybrid does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MICorrelationHybrid has not been fitted.");

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
