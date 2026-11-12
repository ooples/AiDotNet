using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Frequency;

/// <summary>
/// Power Spectral Density based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their power spectral density, identifying features
/// with concentrated power in specific frequency bands.
/// </para>
/// <para><b>For Beginners:</b> Power spectral density shows how signal power
/// is distributed across frequencies. Features with high spectral density
/// ratios have power concentrated in meaningful frequency ranges.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class PowerSpectralDensitySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _lowFreqCutoff;
    private readonly double _highFreqCutoff;

    private Vector<T>? _psdScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double LowFreqCutoff => _lowFreqCutoff;
    public double HighFreqCutoff => _highFreqCutoff;
    public Vector<T>? PSDScores => _psdScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public PowerSpectralDensitySelector(
        int nFeaturesToSelect = 10,
        double lowFreqCutoff = 0.0,
        double highFreqCutoff = 0.5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _lowFreqCutoff = lowFreqCutoff;
        _highFreqCutoff = highFreqCutoff;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        var engine = Engine;

        // Periodogram over every positive frequency below Nyquist (k = 1 .. ceil(n/2) - 1, the non-DC bins of
        // numpy.fft.rfft without an even length's Nyquist bin)
        int frequencies = FilterStatisticTensorOps.PositiveFrequencyCount(n);
        Tensor<T> scores;
        if (frequencies == 0)
        {
            scores = FilterStatisticTensorOps.Filled(new[] { p }, NumOps.Zero);
        }
        else
        {
            // Apply Hann window w[i] = 0.5 (1 - cos(2 pi i / (n - 1)))
            var window = engine.TensorMultiplyScalar(
                FilterStatisticTensorOps.OneMinus(engine.TensorCos(engine.TensorMultiplyScalar(
                    engine.TensorArange<T>(0, n, 1), NumOps.FromDouble(2 * Math.PI / (n - 1))))),
                NumOps.FromDouble(0.5));
            var windowed = engine.TensorMultiply(
                FilterStatisticTensorOps.AsTensor(data), FilterStatisticTensorOps.ColumnBroadcast(window, p));

            // Power |X_k|^2 / n of every frequency ([frequencies, p])
            var power = engine.TensorDivideScalar(
                FilterStatisticTensorOps.ColumnDftPower(windowed, frequencies), NumOps.FromDouble(n));
            var totalPower = engine.ReduceSum(power, new[] { 0 }, false);

            // Power inside the [lowFreqCutoff n, highFreqCutoff n] frequency band
            int lowIdx = (int)(_lowFreqCutoff * n);
            int highIdx = (int)(_highFreqCutoff * n);
            int first = Math.Max(1, lowIdx);
            int last = Math.Min(frequencies, highIdx);
            var bandPower = last >= first
                ? engine.ReduceSum(
                    engine.TensorIndexSelect(
                        power, new Tensor<int>(Enumerable.Range(first - 1, last - first + 1).ToArray(), new[] { last - first + 1 }), 0),
                    new[] { 0 }, false)
                : FilterStatisticTensorOps.Filled(new[] { p }, NumOps.Zero);

            // Score is ratio of band power to total power
            scores = FilterStatisticTensorOps.GuardedDivide(bandPower, totalPower, 1e-10, NumOps.Zero);
        }

        _psdScores = scores.ToVector();

        _selectedIndices = FilterStatisticTensorOps.TopKIndicesSorted(scores, Math.Min(_nFeaturesToSelect, p));

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
            throw new InvalidOperationException("PowerSpectralDensitySelector has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("PowerSpectralDensitySelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("PowerSpectralDensitySelector has not been fitted.");

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
