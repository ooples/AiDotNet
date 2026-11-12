using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Wavelet;

/// <summary>
/// Wavelet Entropy based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their wavelet entropy, which measures the complexity
/// of features in the time-frequency domain using wavelet decomposition.
/// </para>
/// <para><b>For Beginners:</b> Wavelets break down a signal into different frequency
/// components at different times. The entropy of these components tells us how
/// complex or information-rich a feature is. More complex features often capture
/// more useful patterns.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class WaveletEntropySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _decompositionLevel;

    private Vector<T>? _waveletEntropyScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int DecompositionLevel => _decompositionLevel;
    public Vector<T>? WaveletEntropyScores => _waveletEntropyScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public WaveletEntropySelector(
        int nFeaturesToSelect = 10,
        int decompositionLevel = 3,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _decompositionLevel = decompositionLevel;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        var scores = PreprocessingTensorOps.Filled(new[] { p }, NumOps.Zero);
        if (p > 0)
        {
            // Haar decomposition: energy of each level's detail coefficients, then of the final approximation
            var current = Tensor<T>.FromMatrix(data);
            var energies = new List<Tensor<T>>();
            for (int level = 0; level < _decompositionLevel && current.Shape[0] >= 2; level++)
            {
                var (approximation, detail) = FeatureSelectionTensorOps.HaarStep(current);
                energies.Add(engine.Reshape(FeatureSelectionTensorOps.ColumnSumOfSquares(detail), new[] { 1, p }));
                current = approximation;
            }

            energies.Add(engine.Reshape(FeatureSelectionTensorOps.ColumnSumOfSquares(current), new[] { 1, p }));

            // Entropy (bits) of the energy distribution; near-zero total energy scores 0
            var energy = engine.Concat(energies, 0);
            var totalEnergy = engine.ReduceSum(energy, new[] { 0 }, false);
            scores = FeatureSelectionTensorOps.ZeroUnless(
                PreprocessingTensorOps.AtLeast(totalEnergy, NumOps.FromDouble(1e-10)),
                FeatureSelectionTensorOps.ColumnEntropyBits(energy, double.NegativeInfinity, 1e-10));
        }

        _waveletEntropyScores = scores.ToVector();
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
            throw new InvalidOperationException("WaveletEntropySelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("WaveletEntropySelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("WaveletEntropySelector has not been fitted.");

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
