using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Signal-to-Noise Ratio (SNR) for binary classification feature selection.
/// </summary>
/// <remarks>
/// <para>
/// SNR measures the ratio of the difference in class means to the sum of their
/// standard deviations. Higher SNR indicates features where classes are well
/// separated relative to their internal variation.
/// </para>
/// <para><b>For Beginners:</b> Think of the "signal" as the difference between
/// class averages (what distinguishes them) and "noise" as the variation within
/// each class (what makes it hard to tell them apart). A high SNR means the
/// distinguishing signal is much stronger than the confusing noise.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SignalToNoiseRatio<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _minSNR;

    private double[]? _snrValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? SNRValues => _snrValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public SignalToNoiseRatio(
        int nFeaturesToSelect = 10,
        double minSNR = 0.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _minSNR = minSNR;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SignalToNoiseRatio requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Separate samples by class (binary)
        var class0 = new List<int>();
        var class1 = new List<int>();

        for (int i = 0; i < n; i++)
        {
            if (NumOps.ToDouble(target[i]) < 0.5)
                class0.Add(i);
            else
                class1.Add(i);
        }

        if (class0.Count == 0 || class1.Count == 0)
            throw new ArgumentException("Both classes must have at least one sample.");

        int n0 = class0.Count;
        int n1 = class1.Count;

        _snrValues = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Compute means for each class
            double mean0 = 0, mean1 = 0;
            foreach (int i in class0)
                mean0 += NumOps.ToDouble(data[i, j]);
            mean0 /= n0;

            foreach (int i in class1)
                mean1 += NumOps.ToDouble(data[i, j]);
            mean1 /= n1;

            // Compute standard deviations
            double var0 = 0, var1 = 0;
            foreach (int i in class0)
            {
                double diff = NumOps.ToDouble(data[i, j]) - mean0;
                var0 += diff * diff;
            }
            double std0 = Math.Sqrt(var0 / n0);

            foreach (int i in class1)
            {
                double diff = NumOps.ToDouble(data[i, j]) - mean1;
                var1 += diff * diff;
            }
            double std1 = Math.Sqrt(var1 / n1);

            // Signal-to-Noise Ratio
            double noise = std0 + std1;
            _snrValues[j] = noise > 1e-10 ? Math.Abs(mean0 - mean1) / noise : 0;
        }

        // Select features above threshold or top by SNR
        var candidates = new List<int>();
        for (int j = 0; j < p; j++)
            if (_snrValues[j] >= _minSNR)
                candidates.Add(j);

        if (candidates.Count >= _nFeaturesToSelect)
        {
            _selectedIndices = candidates
                .OrderByDescending(j => _snrValues[j])
                .Take(_nFeaturesToSelect)
                .OrderBy(x => x)
                .ToArray();
        }
        else
        {
            _selectedIndices = _snrValues
                .Select((snr, idx) => (SNR: snr, Index: idx))
                .OrderByDescending(x => x.SNR)
                .Take(_nFeaturesToSelect)
                .Select(x => x.Index)
                .OrderBy(x => x)
                .ToArray();
        }

        IsFitted = true;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SignalToNoiseRatio has not been fitted.");

        int numRows = data.Rows;
        int numCols = _selectedIndices.Length;
        var result = new T[numRows, numCols];

        for (int i = 0; i < numRows; i++)
            for (int j = 0; j < numCols; j++)
                result[i, j] = data[i, _selectedIndices[j]];

        return new Matrix<T>(result);
    }

    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SignalToNoiseRatio does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SignalToNoiseRatio has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

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
