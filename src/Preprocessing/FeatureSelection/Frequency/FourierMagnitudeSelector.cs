using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Frequency;

/// <summary>
/// Fourier Magnitude based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their spectral energy in the frequency domain,
/// identifying features with strong periodic components.
/// </para>
/// <para><b>For Beginners:</b> The Fourier transform reveals periodic patterns
/// in data. Features with high spectral energy contain strong repeating patterns
/// that may be useful for prediction.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class FourierMagnitudeSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;

    private Vector<T>? _spectralEnergies;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? SpectralEnergies => _spectralEnergies;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public FourierMagnitudeSelector(
        int nFeaturesToSelect = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        var engine = Engine;

        // master averaged each column with LINQ, which throws on an empty sequence
        if (n == 0 && p > 0)
            throw new InvalidOperationException("Sequence contains no elements");

        // Remove the mean (DC component) of every feature
        var centered = FilterStatisticTensorOps.CenterColumns(FilterStatisticTensorOps.AsTensor(data));

        // DFT energy over every positive frequency below Nyquist (k = 1 .. ceil(n/2) - 1, the non-DC bins of
        // numpy.fft.rfft without an even length's Nyquist bin), divided by n
        int frequencies = FilterStatisticTensorOps.PositiveFrequencyCount(n);
        var energies = frequencies > 0
            ? engine.TensorDivideScalar(
                engine.ReduceSum(FilterStatisticTensorOps.ColumnDftPower(centered, frequencies), new[] { 0 }, false),
                NumOps.FromDouble(n))
            : FilterStatisticTensorOps.Filled(new[] { p }, NumOps.Zero);
        _spectralEnergies = energies.ToVector();

        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(energies, Math.Min(_nFeaturesToSelect, p));

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
            throw new InvalidOperationException("FourierMagnitudeSelector has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("FourierMagnitudeSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FourierMagnitudeSelector has not been fitted.");

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
