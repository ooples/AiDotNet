using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Morphological;

/// <summary>
/// Erosion/Dilation based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on morphological properties, measuring how feature values
/// respond to erosion and dilation operations (local minimum/maximum filtering).
/// </para>
/// <para><b>For Beginners:</b> Morphological operations like erosion (shrinking)
/// and dilation (expanding) reveal structural properties of data. Features that
/// change significantly under these operations may capture important local patterns.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class ErosionDilationSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _windowSize;

    private Vector<T>? _morphologicalScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int WindowSize => _windowSize;
    public Vector<T>? MorphologicalScores => _morphologicalScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public ErosionDilationSelector(
        int nFeaturesToSelect = 10,
        int windowSize = 3,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _windowSize = windowSize;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int p = data.Columns;
        int halfWindow = _windowSize / 2;

        // Columns as rows ([p, n]) so the windows slide along the samples
        var engine = Engine;
        var columns = engine.TensorTranspose(Tensor<T>.FromMatrix(data));

        // Morphological gradient (dilation - erosion): features with larger gradients have more local variation
        var dilated = FeatureSelectionTensorOps.SlidingMax(columns, halfWindow);
        var eroded = FeatureSelectionTensorOps.SlidingMin(columns, halfWindow);
        var scores = engine.ReduceMean(engine.TensorSubtract(dilated, eroded), new[] { 1 }, false);
        _morphologicalScores = scores.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, numToSelect);

        IsFitted = true;
    }

    public new Matrix<T> FitTransform(Matrix<T> data)
    {
        Fit(data);
        return Transform(data);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ErosionDilationSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("ErosionDilationSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ErosionDilationSelector has not been fitted.");

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
