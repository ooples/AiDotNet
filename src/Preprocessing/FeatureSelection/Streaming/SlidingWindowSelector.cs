using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Streaming;

/// <summary>
/// Sliding Window Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Maintains a sliding window of recent samples and performs feature selection
/// based on the most recent data, adapting to concept drift.
/// </para>
/// <para><b>For Beginners:</b> Data patterns can change over time (concept drift).
/// This method only looks at the most recent samples in a sliding window, so it
/// adapts to changes. Features that were important last month might not be
/// important now, and this method handles that.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SlidingWindowSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _windowSize;

    private Tensor<T>? _window;
    private Vector<T>? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int WindowSize => _windowSize;
    public Vector<T>? FeatureScores => _featureScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public SlidingWindowSelector(
        int nFeaturesToSelect = 10,
        int windowSize = 1000,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (windowSize < 10)
            throw new ArgumentException("Window size must be at least 10.", nameof(windowSize));

        _nFeaturesToSelect = nFeaturesToSelect;
        _windowSize = windowSize;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SlidingWindowSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        int n = data.Rows;
        int p = data.Columns;
        _nInputFeatures = p;

        // Add new samples to window (features, then the target as the last column)
        var rows = Engine.TensorConcatenate(new[]
        {
            Tensor<T>.FromMatrix(data), Engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 })
        }, 1);
        var window = _window is null
            ? rows
            : n == 0 ? _window : Engine.TensorConcatenate(new[] { _window, rows }, 0);

        // Remove old samples if window is full
        int count = window.Shape[0];
        if (count > _windowSize)
            window = Engine.TensorSlice(window, new[] { count - _windowSize, 0 }, new[] { _windowSize, p + 1 });
        _window = window;

        // Compute feature scores from window
        ComputeScoresFromWindow(p);

        IsFitted = true;
    }

    private void ComputeScoresFromWindow(int p)
    {
        if (_window is null)
            return;

        int windowN = _window.Shape[0];
        if (windowN < 2) return;

        // |Pearson r| of every feature with the target, 0 for a constant feature or target
        var x = Engine.TensorSlice(_window, new[] { 0, 0 }, new[] { windowN, p });
        var y = Engine.Reshape(Engine.TensorSlice(_window, new[] { 0, p }, new[] { windowN, 1 }), new[] { windowN });
        var scores = Engine.TensorAbs(PreprocessingTensorOps.TargetCorrelations(x, y));
        _featureScores = scores.ToVector();
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, Math.Min(_nFeaturesToSelect, p));
    }

    public void PartialFit(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
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
            throw new InvalidOperationException("SlidingWindowSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SlidingWindowSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SlidingWindowSelector has not been fitted.");

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
