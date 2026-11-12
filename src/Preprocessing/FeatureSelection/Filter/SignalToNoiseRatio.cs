using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

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
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SignalToNoiseRatio<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _minSNR;

    private Vector<T>? _snrValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? SNRValues => _snrValues;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
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

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SignalToNoiseRatio requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        var engine = Engine;

        // Separate samples by class (binary)
        var class0 = FeatureSelectionTensorOps.BelowMask(target, 0.5, out int n0);
        var class1 = FeatureSelectionTensorOps.OneMinus(class0);
        int n1 = n - n0;

        if (n0 == 0 || n1 == 0)
            throw new ArgumentException("Both classes must have at least one sample.");

        // Class means and (population) standard deviations of every feature
        var x = FeatureSelectionTensorOps.AsTensor(data);
        var (mean0, ss0) = FeatureSelectionTensorOps.MaskedColumnMoments(x, class0, n0);
        var (mean1, ss1) = FeatureSelectionTensorOps.MaskedColumnMoments(x, class1, n1);
        var std0 = engine.TensorSqrt(engine.TensorDivideScalar(ss0, NumOps.FromDouble(n0)));
        var std1 = engine.TensorSqrt(engine.TensorDivideScalar(ss1, NumOps.FromDouble(n1)));

        // Signal-to-Noise Ratio |mean0 - mean1| / (std0 + std1); classes that do not overlap at all have
        // infinite SNR, a constant feature 0
        var snr = FeatureSelectionTensorOps.SeparationAwareRatio(
            engine.TensorAbs(engine.TensorSubtract(mean0, mean1)), engine.TensorAdd(std0, std1), 1e-10);
        _snrValues = snr.ToVector();

        // Select features above threshold or top by SNR
        _selectedIndices = FeatureSelectionTensorOps.CandidatesOrTopK(snr, NumOps.FromDouble(_minSNR), _nFeaturesToSelect);

        IsFitted = true;
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
            throw new InvalidOperationException("SignalToNoiseRatio has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SignalToNoiseRatio does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SignalToNoiseRatio has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
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
