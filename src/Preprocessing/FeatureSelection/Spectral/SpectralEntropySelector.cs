using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Spectral;

/// <summary>
/// Spectral Entropy based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their spectral entropy computed from the eigenvalue
/// distribution of their covariance structure.
/// </para>
/// <para><b>For Beginners:</b> This analyzes the "frequency content" of each feature's
/// information. Features with more complex spectral patterns (higher entropy) often
/// contain more useful information for prediction.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SpectralEntropySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _windowSize;

    private Vector<T>? _spectralEntropyScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int WindowSize => _windowSize;
    public Vector<T>? SpectralEntropyScores => _spectralEntropyScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public SpectralEntropySelector(
        int nFeaturesToSelect = 10,
        int windowSize = 32,
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
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        int window = Math.Min(_windowSize, n);
        int numWindows = window == 0 ? 0 : n / window;
        var scores = PreprocessingTensorOps.Filled(new[] { p }, NumOps.Zero);
        if (numWindows > 0 && p > 0)
        {
            // Every window as a column block: [window, numWindows * p].
            var rows = engine.TensorSlice(Tensor<T>.FromMatrix(data), new[] { 0, 0 }, new[] { numWindows * window, p });
            var windows = engine.Reshape(
                engine.TensorPermute(engine.Reshape(rows, new[] { numWindows, window, p }), new[] { 1, 0, 2 }),
                new[] { window, numWindows * p });

            // Spectral entropy (bits) of each window's power spectrum, averaged over the windows.
            var entropy = SelectorTensorOps.ColumnEntropyBits(
                SelectorTensorOps.DftPower(windows), 1e-10, 1e-10);
            scores = engine.ReduceMean(engine.Reshape(entropy, new[] { numWindows, p }), new[] { 0 }, false);
        }

        _spectralEntropyScores = scores.ToVector();
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, Math.Min(_nFeaturesToSelect, p));

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
            throw new InvalidOperationException("SpectralEntropySelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SpectralEntropySelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SpectralEntropySelector has not been fitted.");

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
